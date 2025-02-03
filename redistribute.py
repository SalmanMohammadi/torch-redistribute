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

# credit to @msaroufim for this function
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

def get_fsdp_modules(model):
    return [module for module in model.modules() if hasattr(module, "_get_fsdp_state")]

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
