from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources

from torch_redistribute.utils import printr, redistribute, distribute
import torch.distributed.tensor._dispatch as op_dispatch
import torch.distributed.tensor._random as random
import torch.nn as nn
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor._collective_utils import check_tensor_meta, mesh_broadcast
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._random import (
    is_rng_supported_mesh,
    OffsetBasedRNGTracker,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._api import distribute_tensor
import inspect
import warnings
from collections.abc import Sequence
from typing import Any, Callable, cast, Optional
from typing_extensions import deprecated


class RedistributeContext:
    def __init__(
        self,
        model: nn.Module,
        tp_model: nn.Module,
        device_mesh: torch.distributed.device_mesh.DeviceMesh,
    ):
        self.model = model # this module should contain pre-post forward/backward hooks for FSDP
        self.tp_model = tp_model
        self.device_mesh = device_mesh
        self.fsdp_placements = {}

        for m_name, module in model.named_modules():
            if not m_name:  # Skip the root module
                continue
            module_spec = {
                name: param.placements for name, param in module.named_parameters()
            }
            self.fsdp_placements[m_name] = module_spec
        
    def _restore_fsdp_state(self):
        for (m_name, module), (_, generate_module) in zip(
            self.model.named_modules(), self.tp_model.named_modules()
        ):
            if not m_name:
                continue
            for (name, param), (_, generate_param) in zip(
                module.named_parameters(), generate_module.named_parameters()
            ):
                target_placement = self.fsdp_placements[m_name][name]
                if param.placements != target_placement:
                    redistributed_param = param.redistribute(
                        device_mesh=self.device_mesh,
                        placements=target_placement,
                    )
                    module.register_parameter(
                        name,
                        nn.Parameter(redistributed_param),
                    )
                    # free_storage(generate_param)
                    # free_storage(param)
    
    def _register_parameters(self, module: nn.Module, tp_module: nn.Module):
        for (module_name, module_a), (_, module_b) in zip(
            module.named_modules(), tp_module.named_modules()
        ):
            if module_name:
                printr(module_name)
                if "." in module_name:
                    self._register_parameters(module_a, module_b)
                else:
                    for param_name, param in module_a.named_parameters():
                        module_b.register_parameter(param_name, param)

    def _to_tensor_parallel(self):
        # now we want to redistribute the underlying tensors, but we don't want to 
        # override the FSDP forward
        redistribute(self.model, self.device_mesh)
        # self._register_parameters(self.model, self.tp_model)
        for (module_name, module_a), (_, module_b) in zip(
            reversed(list(self.model.named_modules())), reversed(list(self.tp_model.named_modules()))
        ):
            if module_name:
                printr("module name", module_name)
                for param_name, param in reversed(list(module_a.named_parameters())):
                    printr("param name", param_name)
                    if "." in param_name:
                        continue
                    module_b.register_parameter(param_name, param)

    def __enter__(self):
        # torch.distributed.barrier()
        self._to_tensor_parallel()
        return self.tp_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # torch.distributed.barrier()
        self._restore_fsdp_state()




def redistribute_module_weights_only(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
) -> nn.Module:
    """
    This function expose three functions to control the parameters/inputs/outputs of the module:

    1. To perform sharding on the module before runtime execution by specifying the
    ``partition_fn`` (i.e. allow user to convert Module parameters to :class:`DTensor`
    parameters according to the `partition_fn` specified).
    2. To control the inputs or outputs of the module during runtime execution by
    specifying the ``input_fn`` and ``output_fn``. (i.e. convert the input to
    :class:`DTensor`, convert the output back to ``torch.Tensor``)

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the ``device_mesh``). If ``partition_fn`` is not specified,
            by default we replicate all module parameters of ``module`` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. ``input_fn`` will be installed as a module
            ``forward_pre_hook`` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. ``output_fn`` will be
            installed as a module ``forward_hook`` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all ``DTensor`` s.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_module``
        return nn.Module with PyTorch/XLA SPMD annotated parameters. See
        `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.

    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    already_distributed = getattr(module, "_distribute_module_applied", False)
    if already_distributed:
        raise RuntimeError(
            "distribute_module should only be called once on a module, "
            "but it has already been called on this module!"
        )

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        # This function loop over the immediate module parameters and
        # buffers, replicate all non DTensor params/buffers to DTensor
        # parameters/buffers, if they have not been partitioned in the
        # partition_fn, we can't easily use `module._apply` here
        # because we don't know what happened inside partition_fn as
        # user could do anything, i.e. install hooks, and we want to
        # preserve those.
        full_replicate = [Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and not isinstance(param, DTensor):
                m.register_parameter(
                    key,
                    nn.Parameter(distribute_tensor(param.data, mesh, full_replicate)),
                )
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(buffer, DTensor):
                m._buffers[key] = distribute_tensor(buffer, mesh, full_replicate)

    if partition_fn is None:
        # if partition_fn not specified, we by default replicate
        # all module params/buffers
        for name, submod in module.named_modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        # apply partition_fun to submodules
        for name, submod in module.named_modules():
            partition_fn(name, submod, device_mesh)
            replicate_module_params_buffers(submod, device_mesh)

    return module
