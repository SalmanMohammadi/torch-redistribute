# torchrun --nnodes=1 --nproc-per-node=2 test.py
import functools
from contextlib import contextmanager

import torch

import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp.fully_shard import _get_module_fsdp_state
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
    state_hooks = []
    original_attrs = {}
    for module in fsdp_modules:
        state = _get_module_fsdp_state(module)
        if not state:
            continue

        # Save original hooks and replace with dummies
        state_hooks.append((state, state._pre_forward, state._post_forward))

        # Replacement functions that bypass FSDP logic
        state._pre_forward = lambda _, m, args, kwargs: (args, kwargs)
        state._post_forward = lambda _, m, args, output: output

        # 2. Override __call__ to bypass nn.Module hooks
        original_attrs[module] = {
            "_call_impl": module._call_impl,
            "_fsdp_use_tp": False,  # Flag for tracking
        }

        def _patched_call(module, *args, **kwargs):
            if original_attrs[module]["_fsdp_use_tp"]:
                return module.forward(*args, **kwargs)
            return original_attrs[module]["_call_impl"](*args, **kwargs)

        module._call_impl = functools.partial(_patched_call, module)
        original_attrs[module]["_fsdp_use_tp"] = True
    dummy_input = torch.randn(4, 6)
    redistribute(model, device_mesh)
    model(dummy_input)
    if dist.get_rank() == 0:
        import pdb

        pdb.set_trace()

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
    print(f"weighta fter reshard", model.w2.weight)
    
    if dist.get_rank() == 0:
        import pdb

        pdb.set_trace()

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
