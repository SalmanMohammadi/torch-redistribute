# torchrun --nnodes=1 --nproc-per-node=2 redistribute.py
import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor import Shard
from torch.distributed.tensor import init_device_mesh
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
import sys
from .style import RedistributeColWiseParallel, RedistributeRowWiseParallel

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
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
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
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h


def distribute_ff(layer: nn.Module, device_mesh):
    parallelize_module(layer.w1, device_mesh, ColwiseParallel())
    parallelize_module(layer.w2, device_mesh, RowwiseParallel())
    parallelize_module(layer.w3, device_mesh, ColwiseParallel())

def redistribute_ff(layer: nn.Module, device_mesh):
    parallelize_module(layer.w1, device_mesh, RedistributeColWiseParallel())
    parallelize_module(layer.w2, device_mesh, RedistributeRowWiseParallel())
    parallelize_module(layer.w3, device_mesh, RedistributeColWiseParallel())


def redistribute_ff_dtensor(layer: nn.Module, device_mesh):
    layer.w1.weight = nn.Parameter(layer.w1.weight.redistribute(device_mesh=device_mesh, placements=[Shard(0)]))
    layer.w2.weight = nn.Parameter(layer.w2.weight.redistribute(device_mesh=device_mesh, placements=[Shard(1)]))
    layer.w3.weight = nn.Parameter(layer.w3.weight.redistribute(device_mesh=device_mesh, placements=[Shard(0)]))
    


def printr0(str):
    if dist.get_rank() == 0:
        print(str)  

def print_model(model):
    print(f"\nRank {dist.get_rank()}:\t model.w1 local shape: {model.w1.weight.to_local().shape}")
    print(f"\nRank {dist.get_rank()}:\t model.w2 local shape: {model.w2.weight.to_local().shape}")
    print(f"\nRank {dist.get_rank()}:\t model.w3 local shape: {model.w3.weight.to_local().shape}")

def main(arg):
    dist.init_process_group(backend="gloo")

    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size, ))
    
    model = FeedForward(dim=6, hidden_dim=8)
    # if arg and arg[0] == "fsdp":
    #     printr0("Using FSDP")
    #     model = fully_shard(model, mesh=device_mesh)
    # else:
    #     printr0("Using TP")
    # # fully_shard(model, mesh=device_mesh)
    #     distribute_ff(model, device_mesh)

    printr0("Using FSDP")
    fully_shard(model, mesh=device_mesh)
    print_model(model)
    torch.cpu.synchronize()

    printr0("Redistributing")
    redistribute_ff(model, device_mesh)
    print_model(model)
    # expected after redistribution
    # Rank 0:  model.w1 local shape: torch.Size([4, 6])
    # Rank 1:  model.w1 local shape: torch.Size([4, 6])

    # Rank 0:  model.w2 local shape: torch.Size([8, 3])
    # Rank 1:  model.w2 local shape: torch.Size([8, 3])

    # Rank 0:  model.w3 local shape: torch.Size([4, 6])
    # Rank 1:  model.w3 local shape: torch.Size([4, 6])

    dist.destroy_process_group()

if __name__ == "__main__":
    main(sys.argv[1:])