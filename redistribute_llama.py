from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch_redistribute.redistribute import RedistributeContext
from torch.distributed.tensor.placement_types import Replicate
from torch_redistribute.utils import llama_redistribute_plan, llama_tp_plan, printr
from torchtune import training, utils
from torchtune.models.llama2 import llama2


def main():
    backend = "nccl"
    device_name = "cuda"
    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = utils.get_device(device="cuda")
    device_mesh = init_device_mesh(device_name, (world_size,))

    model_kwargs = {
        "vocab_size": 32000,
        "num_layers": 22,
        "num_heads": 32,
        "num_kv_heads": 4,
        "embed_dim": 2048,
        "max_seq_len": 2048,
        "intermediate_dim": 5632,
        "attn_dropout": 0.0,
        "norm_eps": 1e-5,
    }

    # First, let's setup the model which will be sharded with FSDP.
    printr("Initializing FSDP model")
    with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
        model = llama2(**model_kwargs)

    training.shard_model(
        model=model,
        shard_conditions=[
            partial(training.get_shard_conditions, custom_sharded_layers=None)
        ],
        cpu_offload=False,
    )

    with training.set_default_dtype(torch.bfloat16), device:
        for m in model.modules():
            # RoPE is not covered in state dict
            if hasattr(m, "rope_init"):
                m.rope_init()

    model.to_empty(device=device)

    if rank == 0:
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(
            message="Memory usage after FSDP model init", stats=memory_stats
        )

    # Now let's setup the model which will be sharded with TP
    printr("Initializing TP model")
    with torch.device("meta"):
        tp_model = llama2(**model_kwargs)
        tp_model = training.prepare_mha_for_tp(tp_model, device_mesh)
        parallelize_module(
            tp_model, device_mesh=device_mesh, parallelize_plan=llama_tp_plan
        )

    # Memory usage *shouldn't* have increased, but the correct hooks are now
    # installed for TP
    if rank == 0:
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(
            message="Memory usage after TP model init", stats=memory_stats
        )

    # Let's setup the context manager which will handle redistribution
    redistributed_model_ctx = RedistributeContext(
        model, tp_model, device_mesh, llama_redistribute_plan
    )
    torch.distributed.barrier()
    # dummy sequences with shape [b, seq_len]
    dummy_seqs = torch.randint(0, 32000, (1, 265)).to(device, dtype=torch.long)

    # let's perform some dummy forwards + backwards passes with FSDP

    # out = model(dummy_seqs.clone())
    # out.mean().backward()
    # model.zero_grad(set_to_none=True)
    # torch.cuda.empty_cache()

    # now let's enter the context manager and perform a forwards pass with TP
    with redistributed_model_ctx as tp_model, torch.no_grad():
        # First, let's check that we haven't duplicated weights in memory
        if rank == 0:
            memory_stats = training.get_memory_stats(device=device)
            training.log_memory_stats(
                message="Memory usage after FSDP -> TP redistribution",
                stats=memory_stats,
            )
        tp_model(dummy_seqs.clone())

    if rank == 0:
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(
            message="Memory usage after TP -> FSDP redistribution", stats=memory_stats
        )

    # another forwards + backwards pass with FSDP
    out = model(dummy_seqs.clone())
    out.mean().backward()
    model.zero_grad(set_to_none=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
