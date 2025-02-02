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

def print_tensor_distribution(param, name: str, rank: int):
    """Custom function to print tensor distribution information"""
    if rank == 0:
        print(f"\n{name}:")
        print("-" * 50)
        print(f"Global shape: {param.shape}")
        print(f"Local shape: {param.to_local().shape}")
        print(f"Placement: {param.placements}")
        
        # For 2D tensors, use visualize_sharding
        if len(param.shape) == 2:
            print("\nSharding Visualization:")
            visualize_sharding(param, name)
        # For 1D tensors, print custom visualization
        else:
            local_size = param.to_local().size(0)
            total_size = param.size(0)
            if "[Replicate()]" in str(param.placements):
                print("\nReplicated across all ranks")
                print(f"Each rank has full tensor of size {total_size}")
            else:
                start_idx = rank * local_size
                end_idx = start_idx + local_size
                print(f"\nSharded along dimension 0")
                print(f"Rank {rank} handles indices {start_idx} to {end_idx-1}")


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (Optional[nn.Module]): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        dim,
        hidden_dim,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``, where ``in_dim`` is the
            input dimension of both ``gate_proj`` and ``up_proj``.

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``, where ``out_dim`` is the \
            output dimension of ``down_proj``.
        """
        h = self.activation(self.w1(x))
        h = h * self.w3(x)
        h = self.w2(h)
        return h


redistribute_parallelize_plan = {
    "w1": RedistributeColWiseParallel(),
    "w2": RedistributeRowWiseParallel(),
    "w3": RedistributeColWiseParallel(),
}

parallelize_plan = {
    "w1": ColwiseParallel(),
    "w2": RowwiseParallel(),
    "w3": ColwiseParallel(),
}


def distribute_ff(layer: nn.Module, device_mesh):
    parallelize_module(layer, device_mesh, parallelize_plan)


def redistribute_ff(layer: nn.Module, device_mesh):
    parallelize_module(layer, device_mesh, redistribute_parallelize_plan)


def redistribute_ff_dtensor(layer: nn.Module, device_mesh):
    layer.w1.weight = nn.Parameter(
        layer.w1.weight.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
    )
    layer.w2.weight = nn.Parameter(
        layer.w2.weight.redistribute(device_mesh=device_mesh, placements=[Shard(1)])
    )
    layer.w3.weight = nn.Parameter(
        layer.w3.weight.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
    )


def printr0(str):
    if dist.get_rank() == 0:
        print(str)


def print_model(model):
    rank = dist.get_rank()
    print_tensor_distribution(model.w1.weight, "model.w1", rank)
    print_tensor_distribution(model.w2.weight, "model.w2", rank)
    print_tensor_distribution(model.w3.weight, "model.w3", rank)
    # print(
    #     f"\nRank {dist.get_rank()}:\t model.w1 local shape: {model.w1.weight.to_local().shape}"
    # )
    # print(
    #     f"\nRank {dist.get_rank()}:\t model.w2 local shape: {model.w2.weight.to_local().shape}"
    # )
    # print(
    #     f"\nRank {dist.get_rank()}:\t model.w3 local shape: {model.w3.weight.to_local().shape}"
    # )


def distribute_replicate(model: nn.Module, device_mesh):
    parallelize_module(model, device_mesh, ReplicateParallel())

def main():
    dist.init_process_group(backend="gloo")

    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))

    model = FeedForward(dim=6, hidden_dim=8)

    # printr0("Using FSDP")
    # distribute_module(model, device_mesh)
    distribute_replicate(model, device_mesh)
    # dummy_input = torch.randn(6, 4)
    print_model(model)
    model(torch.randn(4, 6))
    print("=" * 80)
    printr0("Redistributing")
    with torch.no_grad():
        redistribute_ff(model, device_mesh)
        print_model(model)
        model(torch.randn(4, 6))
    # expected after redistribution
    # Rank 0:  model.w1 local shape: torch.Size([4, 6])
    # Rank 1:  model.w1 local shape: torch.Size([4, 6])

    # Rank 0:  model.w2 local shape: torch.Size([8, 3])
    # Rank 1:  model.w2 local shape: torch.Size([8, 3])

    # Rank 0:  model.w3 local shape: torch.Size([4, 6])
    # Rank 1:  model.w3 local shape: torch.Size([4, 6])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
