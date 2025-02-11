import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch_redistribute.model import FeedForward
from torch_redistribute.utils import print_tensor_distribution


def main():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))
    rank = dist.get_rank()

    # Create model and apply tensor parallel plan
    model = FeedForward(dim=6, hidden_dim=8, bias=False)
    tp_plan = {
        "w1": ColwiseParallel(),
        "w2": RowwiseParallel(),
        "w3": ColwiseParallel()
    }
    parallelize_module(model, device_mesh, tp_plan)

    # Print distribution of weights
    print_tensor_distribution(model.w1.weight, "w1.weight", rank)
    print_tensor_distribution(model.w2.weight, "w2.weight", rank)
    print_tensor_distribution(model.w3.weight, "w3.weight", rank)

    # Test forward pass
    x = torch.randn(8, 6)
    out = model(x)
    if rank == 0:
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {out.shape}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
