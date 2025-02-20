from torch.distributed.tensor.debug import CommDebugMode, visualize_sharding
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from torch_redistribute.style import (
    RedistributeColWiseParallel,
    RedistributeRowWiseParallel,
)


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


dummy_redistribute_plan = {
    "w1": RedistributeColWiseParallel(),
    "w2": RedistributeRowWiseParallel(),
    "w3": RedistributeColWiseParallel(),
}


dummy_tp_plan = {
    "w1": ColwiseParallel(),
    "w2": RowwiseParallel(),
    "w3": ColwiseParallel(),
}


llama_tt_tp_plan = {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    ),
    "output": ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=True
    ),
    "norm": SequenceParallel(),
    "layers.*.attn": PrepareModuleInput(
        input_layouts=(Shard(1), None),
        desired_input_layouts=(Replicate(), None),
    ),
    "layers.*.attn.sa_norm": SequenceParallel(),
    "layers.*.attn.q_proj": ColwiseParallel(),
    "layers.*.attn.k_proj": ColwiseParallel(),
    "layers.*.attn.v_proj": ColwiseParallel(),
    "layers.*.attn.output_proj": RowwiseParallel(output_layouts=Shard(1)),
    "layers.*.mlp_norm": SequenceParallel(),
    "layers.*.mlp": PrepareModuleInput(
        input_layouts=(Shard(1), None),
        desired_input_layouts=(Replicate(), None),
    ),
    "layers.*.mlp.w1": ColwiseParallel(),
    "layers.*.mlp.w2": RowwiseParallel(),
    "layers.*.mlp.w3": ColwiseParallel(),
}

llama_tp_plan = {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(),
    ),
    "output": ColwiseParallel(output_layouts=Replicate()),
    "layers.*.attn.q_proj": ColwiseParallel(),
    "layers.*.attn.k_proj": ColwiseParallel(),
    "layers.*.attn.v_proj": ColwiseParallel(),
    "layers.*.attn.output_proj": RowwiseParallel(),
    "layers.*.mlp.w1": ColwiseParallel(),
    "layers.*.mlp.w2": RowwiseParallel(),
    "layers.*.mlp.w3": ColwiseParallel(),
}


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


def printr(*args):
    import torch.distributed as dist

    if dist.get_rank() == 0:
        print(*args)
