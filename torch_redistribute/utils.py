import torch.nn as nn
from torch.distributed.tensor.debug import CommDebugMode, visualize_sharding
from torch.distributed.tensor.parallel import parallelize_module

from torch_redistribute.style import (
    RedistributeColWiseParallel,
    RedistributeRowWiseParallel,
)
from torchtune import training
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate

# credit to @msaroufim for this funcition
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


redistribute_parallelize_plan = {
    "w1": RedistributeColWiseParallel(),
    "w2": RedistributeRowWiseParallel(),
    "w3": RedistributeColWiseParallel(),
}


def redistribute(layer: nn.Module, device_mesh):
    parallelize_module(layer, device_mesh, redistribute_parallelize_plan)

def distribute(layer: nn.Module, device_mesh):
    parallelize_module(layer, device_mesh, {
        "w1": ColwiseParallel(),
        "w2": RowwiseParallel(),
        "w3": ColwiseParallel(),
    })

def distribute_llama(model: nn.Module, device_mesh):
    model = training.prepare_mha_for_tp(model, device_mesh)
    llama_tp_plan = {
        "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
        "output": ColwiseParallel(output_layouts=Replicate()),
        "layers.*.attn.q_proj": ColwiseParallel(),
        "layers.*.attn.k_proj": ColwiseParallel(),
        "layers.*.attn.v_proj": ColwiseParallel(),
        "layers.*.attn.output_proj": RowwiseParallel(),
        "layers.*.mlp.w1": ColwiseParallel(),
        "layers.*.mlp.w2": RowwiseParallel(),
        "layers.*.mlp.w3": ColwiseParallel(),
    }
    parallelize_module(model, device_mesh, llama_tp_plan)
    return model

def redistribute_llama(model: nn.Module, device_mesh):
    llama_redistribute_plan = {
        "tok_embeddings": RedistributeRowWiseParallel(input_layouts=Replicate()),
        "output": RedistributeColWiseParallel(output_layouts=Replicate()),
        "layers.*.attn.q_proj": RedistributeColWiseParallel(),
        "layers.*.attn.k_proj": RedistributeColWiseParallel(),
        "layers.*.attn.v_proj": RedistributeColWiseParallel(),
        "layers.*.attn.output_proj": RedistributeRowWiseParallel(),
        "layers.*.mlp.w1": RedistributeColWiseParallel(),
        "layers.*.mlp.w2": RedistributeRowWiseParallel(),
        "layers.*.mlp.w3": RedistributeColWiseParallel(),
    }
    parallelize_module(model, device_mesh, llama_redistribute_plan)
    return model

def printr(*args):
    import torch.distributed as dist

    if dist.get_rank() == 0:
        print(*args)
