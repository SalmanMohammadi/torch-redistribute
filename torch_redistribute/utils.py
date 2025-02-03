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
