import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed._composable.fsdp.fully_shard import  fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch_redistribute.utils import redistribute
from torch_redistribute.model import FeedForward

def get_fsdp_modules(model):
    return [module for module in model.modules() if _get_module_fsdp_state(module)]

def test_fsdp_tp_switch(model, device_mesh):
    fsdp_modules = get_fsdp_modules(model)
    original_pre_forwards = {} # Store original _pre_forward methods

    # 1. Selectively Disable FSDP's Parameter Management in _pre_forward
    for module in fsdp_modules:
        state = _get_module_fsdp_state(module)
        if not state:
            continue

        original_pre_forwards[module] = state._pre_forward # Save original

        # Modified _pre_forward that bypasses parameter sharding/init
        def _bypass_param_mgmt_pre_forward(fsdp_state, module, args, kwargs): # Matches FSDPState._pre_forward signature
            # Keep other parts of pre_forward if needed, but bypass parameter logic
            return args, kwargs # Just return original args, kwargs - NO FSDP param handling

        state._pre_forward = functools.partial(_bypass_param_mgmt_pre_forward, state) # Partial to bind state as first arg

    print(f"weighta before redis", model.w2.weight.placements) # Placement before redistribute
    # 2. Apply tensor parallelism (redistribute)
    redistribute(model, device_mesh)
    print(f"weighta after redis", model.w2.weight.placements) # Placement after redistribute

    dummy_input = torch.randn(4, 6)
    model(dummy_input) # Test forward pass in TP mode

    # 3. Restore original _pre_forward hooks
    for module in fsdp_modules:
        if module in original_pre_forwards:
            state = _get_module_fsdp_state(module)
            state._pre_forward = original_pre_forwards[module] # Restore original

    # 4. Reshard with FSDP (after restoring hooks)
    for module in fsdp_modules:
        state = _get_module_fsdp_state(module)
        if state:
            state._reset_lazy_init()
        if hasattr(module, "reshard"):
            module.reshard()
    print(f"weighta after reshard", model.w2.weight.placements) # Placement after reshard


def main():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))

    model = FeedForward(dim=6, hidden_dim=8)
    print(f"weighta before fsdp", model.w2.weight) # Placement before FSDP
    fully_shard(model, mesh=device_mesh)
    print(f"weighta after fsdp", model.w2.weight.placements) # Placement after FSDP

    test_fsdp_tp_switch(model, device_mesh) # Call the test function
    print(f"weighta after test_fsdp_tp_switch", model.w2.weight.placements) # Placement after test function

    dist.destroy_process_group()


if __name__ == "__main__":
    main()