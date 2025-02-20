from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch_redistribute.redistribute import RedistributeContext
from torch_redistribute.utils import llama_tp_plan, llama_tt_tp_plan, printr
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


    with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
        model = llama2(**model_kwargs)
        model = training.prepare_mha_for_tp(model, device_mesh)
        parallelize_module(
            model, device_mesh=device_mesh, parallelize_plan=llama_tp_plan
        )
    model.to_empty(device=device)

    if rank == 0:
        memory_stats = training.get_memory_stats(device=device)
        training.log_memory_stats(
            message="Memory usage after TP model init", stats=memory_stats
        )

    dummy_seqs = torch.randint(0, 32000, (1, 265)).to(device, dtype=torch.long)
    out = model(dummy_seqs)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
