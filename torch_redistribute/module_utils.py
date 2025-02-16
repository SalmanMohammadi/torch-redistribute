import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from typing import Callable, Optional
from torch.distributed.tensor._api import distribute_tensor

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

    Returns:
        A module that contains parameters/buffers that are all ``DTensor`` s.
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