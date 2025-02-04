# torchrun --nnodes=1 --nproc-per-node=2 fsdp_unwrap.py
import functools
from contextlib import contextmanager

import torch

import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import init_device_mesh
from torch_redistribute.utils import redistribute
from torch_redistribute.model import FeedForward


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


def get_fsdp_modules(model):
    return [
        module
        for module in model.modules()
        if _get_module_fsdp_state(module) is not None
    ]
def unwrap_fsdp(model, device_mesh):
    fsdp_modules = get_fsdp_modules(model)
    original_attrs = {}
    state_hooks = []
    original_fsdp_states = {} # Store original FSDP states

    for module in fsdp_modules:
        state = _get_module_fsdp_state(module)
        if not state:
            continue

        # Save original state, hooks, and attributes
        original_fsdp_states[module] = state # Save the entire state
        state_hooks.append((state, state._pre_forward, state._post_forward))
        original_attrs[module] = {
            "_call_impl": module._call_impl,
            "_fsdp_use_tp": False, 
        }

        # Remove FSDP state temporarily
        del module._fsdp_state # Remove the _fsdp_state attribute

        # Replace hooks and _call_impl (although state is removed, for completeness)
        state._pre_forward = lambda _, m, args, kwargs: (args, kwargs)
        state._post_forward = lambda _, m, args, output: output

        def _patched_call(module, *args, **kwargs):
            if original_attrs[module]["_fsdp_use_tp"]:
                return module.forward(*args, **kwargs)
            return original_attrs[module]["_call_impl"](*args, **kwargs)

        module._call_impl = functools.partial(_patched_call, module)
        original_attrs[module]["_fsdp_use_tp"] = True

    dummy_input = torch.randn(4, 6)
    redistribute(model, device_mesh)
    print(f"weighta fter redis", model.w2.weight)
    print(f"w2 placements {model.w2.weight.placements}")
    model(dummy_input) # Run forward pass with FSDP state removed


    # --- Restoration Phase ---
    for module in fsdp_modules:
        if module in original_attrs:
            module._call_impl = original_attrs[module]['_call_impl']

        # Restore FSDP state
        if module in original_fsdp_states:
            module._fsdp_state = original_fsdp_states[module] # Restore the state

    for state, pre_hook, post_hook in state_hooks:
        state._pre_forward = pre_hook
        state._post_forward = post_hook

    # Reshard (after restoring state)
    for module in fsdp_modules:
        state = _get_module_fsdp_state(module) # Get state again AFTER restoring
        if state: 
            state._reset_lazy_init() 
        if hasattr(module, "reshard"):
            module.reshard()
    print(f"weighta fter reshard", model.w2.weight)
def main():
    dist.init_process_group(backend="gloo")

    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))

    model = FeedForward(dim=6, hidden_dim=8)
    print(f"weighta before fsdp", model.w2.weight)
    fully_shard(model, mesh=device_mesh)
    print(f"weighta after fsdp", model.w2.weight)
    unwrap_fsdp(model, device_mesh)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
