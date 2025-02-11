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
    original_attrs = {}
    state_hooks = []

    # 1. Disable FSDP forward processing
    fsdp_modules = get_fsdp_modules(model)
    param_groups = []
    
    # 1. Destroy FSDP parameter groups
    for module in fsdp_modules:
        state = _get_module_fsdp_state(module)
        if state and state._fsdp_param_group:
            param_groups.append(state._fsdp_param_group)
            state._fsdp_param_group = None

    # 2. Apply tensor parallelism
    print(f"weight before TP", model.w2.weight.placements)
    # 3. Apply tensor parallelism
    print(f"weighta before redis", model.w2.weight.placements) # Placement before redistribute
    # 2. Apply tensor parallelism (redistribute)
    redistribute(model, device_mesh)
    print(f"weighta after redis", model.w2.weight.placements) # Placement after redistribute

    dummy_input = torch.randn(8``, 6)
    model(dummy_input) # Test forward pass in TP mode

    # 3. Restore original _pre_forward hooks
    # 4. Restore original behavior
    for module in fsdp_modules:
        if module in original_attrs:
            module._call_impl = original_attrs[module]['_call_impl']
            
    for state, pre_hook, post_hook in state_hooks:
        state._pre_forward = pre_hook
        state._post_forward = post_hook

    # 5. Reset FSDP states
    for module in fsdp_modules:
        if hasattr(module, "reshard"):
            module.reshard()
        state = _get_module_fsdp_state(module)
        state._reset_lazy_init()
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