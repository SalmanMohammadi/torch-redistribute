# torchrun --nnodes=1 --nproc-per-node=2 test_redistribute_ctx_manager.py
import os
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    ParallelStyle,
    RowwiseParallel,
)

from torch_redistribute.model import DummyModel, FeedForward
from torch_redistribute.redistribute import RedistributeContext
from torch_redistribute.utils import (
    dummy_redistribute_plan,
    print_tensor_distribution,
    printr,
)


def print_tensor_storage(tensor: torch.Tensor, name: str):
    """Print minimal unique identifying info about a tensor's storage."""
    printr(f"{name} storage id: {tensor.untyped_storage()._cdata}")


def main():
    if torch.cuda.is_available():
        backend = "nccl"
        device_name = "cuda"
    else:
        backend = "gloo"
        device_name = "cpu"
    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device_mesh = init_device_mesh(device_name, (world_size,), mesh_dim_names=["data"])

    tp_model = DummyModel(dim=6, depth=2)
    model = DummyModel(dim=6, depth=2)
    fully_shard(model, mesh=device_mesh)
    out = model(torch.randn(8, 6))
    out.mean().backward()
    rank = dist.get_rank()

    redistributed_model_ctx = RedistributeContext(
        model, tp_model, device_mesh, dummy_redistribute_plan
    )

    print_tensor_distribution(model.ff_1.layers[0].w1.weight, "model.ff_1.layers[0].w1.weight", rank)

    with redistributed_model_ctx as generation_model, torch.no_grad():
        printr("=" * 50)
        printr("Inside context manager - first redistribution")
        printr(generation_model.ff_1.layers[0].w1.weight)
        generation_model(torch.randn(8, 6))
        print_tensor_distribution(
            generation_model.ff_1.layers[0].w1.weight, "generation_model.ff_1.layers[0].w1.weight", rank
        )
        torch.distributed.barrier()

    printr("=" * 50)
    printr("After first redistribution")
    print_tensor_storage(model.w2.weight, "model1.w2.weight (after first)")
    print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)

    out = model(torch.randn(8, 6))
    out.mean().backward()

    with redistributed_model_ctx as generation_model, torch.no_grad():
        printr("=" * 50)
        printr("Inside context manager - second redistribution")
        print_tensor_storage(
            generation_model.w2.weight, "generation_model.w2.weight (during second)"
        )
        print_tensor_distribution(
            generation_model.w2.weight, "generation_model.w2.weight", rank
        )
        generation_model(torch.randn(8, 6))
        torch.distributed.barrier()

    printr("=" * 50)
    printr("After second redistribution")
    print_tensor_storage(model.w2.weight, "model1.w2.weight (after second)")
    print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)

    out = model(torch.randn(8, 6))
    out.mean().backward()
    dist.destroy_process_group()


if __name__ == "__main__":
    import os

    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = (
        "DETAIL"  # set to DETAIL for runtime logging.
    )
    main()
