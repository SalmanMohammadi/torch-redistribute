from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from torch_redistribute.model import FeedForward
from torch_redistribute.redistribute import RedistributeContext
from torch_redistribute.utils import print_tensor_distribution, printr


def print_tensor_storage(tensor: torch.Tensor, name: str):
    """Print minimal unique identifying info about a tensor's storage."""
    printr(f"{name} storage id: {tensor.untyped_storage()._cdata}")


def main():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=["data"])

    model_cls = partial(FeedForward, dim=512, hidden_dim=512, bias=False)
    model = model_cls()
    fully_shard(model, mesh=device_mesh)
    rank = dist.get_rank()

    redist_ctx = RedistributeContext(model, model_cls, device_mesh)

    print_tensor_storage(model.w2.weight, "model1.w2.weight (before first)")
    print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)

    with redist_ctx:
        printr("=" * 50)
        printr("Inside context manager - first redistribution")
        print_tensor_storage(model.w2.weight, "model1.w2.weight (during first)")
        print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)
        torch.distributed.barrier()

    printr("=" * 50)
    printr("After first redistribution")
    print_tensor_storage(model.w2.weight, "model1.w2.weight (after first)")
    print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)

    with redist_ctx:
        printr("=" * 50)
        printr("Inside context manager - second redistribution")
        print_tensor_storage(model.w2.weight, "model1.w2.weight (during second)")
        print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)
        torch.distributed.barrier()

    printr("=" * 50)
    printr("After second redistribution")
    print_tensor_storage(model.w2.weight, "model1.w2.weight (after second)")
    print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
