# torchrun --nnodes=1 --nproc-per-node=2 redistribute.py
import sys

import torch
import torch.distributed as dist
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import init_device_mesh, Replicate, Shard, distribute_tensor, distribute_module
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch_redistribute.style import (
    RedistributeColWiseParallel,
    RedistributeRowWiseParallel,
    ReplicateParallel
)
from torch.distributed.tensor.debug import CommDebugMode, visualize_sharding

from torch_redistribute.utils import print_tensor_distribution
from torch_redistribute.redistribute import FeedForward


redistribute_parallelize_plan = {
    "w1": RedistributeColWiseParallel(),
    "w2": RedistributeRowWiseParallel(),
    "w3": RedistributeColWiseParallel(),
}


def redistribute(layer: nn.Module, device_mesh):
    parallelize_module(layer, device_mesh, redistribute_parallelize_plan)


def printr0(str):
    if dist.get_rank() == 0:
        print(str)

def print_model(model):
    rank = dist.get_rank()
    print_tensor_distribution(model.w1.weight, "model.w1", rank)
    print_tensor_distribution(model.w2.weight, "model.w2", rank)
    print_tensor_distribution(model.w3.weight, "model.w3", rank)


def distribute_replicate(model: nn.Module, device_mesh):
    parallelize_module(model, device_mesh, ReplicateParallel())

def main():
    dist.init_process_group(backend="gloo")

    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))

    model = FeedForward(dim=6, hidden_dim=8)
    fully_shard(model, mesh=device_mesh)
    if dist.get_rank() == 0:
        import pdb
        pdb.set_trace()
    # distribute_replicate(model, device_mesh)
    # print_model(model)
    # model(torch.randn(4, 6))

    # torch.cpu.synchronize()
    # if dist.get_rank() == 0:
    #     printr0("=" * 80)
    #     printr0("Redistributing")
    # with torch.no_grad():
    #     redistribute(model, device_mesh)
    #     print_model(model)
    #     model(torch.randn(4, 6))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
