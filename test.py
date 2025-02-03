# torchrun --nnodes=1 --nproc-per-node=2 test.py

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import contextmanager
from torch.distributed.fsdp import fully_shard
import torch.distributed as dist
from torch_redistribute.redistribute import FeedForward
from torch.distributed.tensor import init_device_mesh
def _unwrap_fsdp_module(module):
    """Recursively unwraps FSDP-wrapped modules in-place."""
    for name, child in module.named_children():
        if isinstance(child, FSDP):
            # Replace FSDP layer with its inner module
            inner_module = child.module
            setattr(module, name, inner_module)
            # Recursively process the inner module's children
            _unwrap_fsdp_module(inner_module)
        else:
            _unwrap_fsdp_module(child)
    return module

# @contextmanager
def fsdp_to_tp_context(model):
    """Context manager to transition model from FSDP to TP+FSDP."""
    # Step 1: Unwrap FSDP layers
    original_model = model
    unwrapped_model = _unwrap_fsdp_module(model)
    
    # Step 2: Apply Tensor Parallel transformations
    # reparallelize_model_for_tp(unwrapped_model)  # User-defined function
    
    # Step 3: Re-apply FSDP sharding
    fully_shard(unwrapped_model)  # Assuming this is your FSDP initialization
    
    try:
        yield unwrapped_model
    finally:
        # Note: Reverting is non-trivial; this context makes permanent changes
        pass

def main():
    dist.init_process_group(backend="gloo")

    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))

    model = FeedForward(dim=6, hidden_dim=8)
    fully_shard(model, mesh=device_mesh)
    if dist.get_rank() == 0:
        import pdb
        pdb.set_trace()
    
if __name__ == "__main__":
    main()