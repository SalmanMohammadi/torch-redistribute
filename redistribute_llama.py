from torchtune.models.llama2 import llama2
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torchtune import training, utils
from functools import partial
from torch_redistribute.utils import distribute_llama, printr
from torch_redistribute.redistribute import RedistributeContext

def main():
    backend = "nccl"
    device_name = "cuda"
    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = utils.get_device(device="cuda")
    device_mesh = init_device_mesh(device_name, (world_size,), mesh_dim_names=["data"])

    model_kwargs = {
        "vocab_size": 32000,
        "num_layers": 22,
        "num_heads": 32,
        "num_kv_heads": 4,
        "embed_dim": 2048,
        "max_seq_len": 2048,
        "intermediate_dim": 5632,
        "attn_dropout": 0.0,
        "norm_eps": 1e-5
    }

    # First, let's setup the model which will be sharded with FSDP.
    printr("Initializing FSDP model")
    with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
        model = llama2(**model_kwargs)

    training.shard_model(
            model=model,
            shard_conditions=[partial(training.get_shard_conditions, custom_sharded_layers=None)],
            cpu_offload=False
        )

    with training.set_default_dtype(torch.bfloat16), device:
        for m in model.modules():
            # RoPE is not covered in state dict
            if hasattr(m, "rope_init"):
                m.rope_init()

    model.to_empty(device=device)

    if rank == 0:   
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(memory_stats)

    # Now let's setup the model which will be sharded with TP
    printr("Initializing TP model")
    with torch.device("meta"), torch.no_grad(), training.set_default_dtype(torch.bfloat16):
        tp_model = llama2(**model_kwargs)
        tp_model = distribute_llama(tp_model, device_mesh)

    # Memory usage *shouldn't* have increased, but the correct hooks are now
    # installed for TP
    if rank == 0:   
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(memory_stats)

    # Let's setup the context manager which will handle redistribution
    redistributed_model_ctx = RedistributeContext(model, tp_model, device_mesh)
    torch.distributed.barrier()
    # dummy sequences with shape [b, seq_len]
    dummy_seqs = torch.randint(0, 32000, (1, 265)).to(device, dtype=torch.long)

    # let's perform some dummy forwards + backwards passes with FSDP
    # out = model(dummy_seqs.clone())
    # out.mean().backward()

    # now let's enter the context manager and perform a forwards pass with TP
    with redistributed_model_ctx as tp_model, torch.no_grad():
        # First, let's check that we haven't duplicated weights in memory
        printr("Memory usage after FSDP -> TP redistribution")
        if rank == 0:      
            memory_stats = training.get_memory_stats(device=device)
            training.log_memory_stats(memory_stats)
        # tp_model(dummy_seqs.clone())

    printr("Memory usage after TP -> FSDP redistribution")
    if rank == 0:      
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(memory_stats)

    # another forwards + backwards pass with FSDP
    # out = model(dummy_seqs.clone())
    # out.mean().backward()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
