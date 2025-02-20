from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel
)

from torch_redistribute.module_utils import redistribute_module_weights_only


class ReplicateParallel(ParallelStyle):
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(inputs[0], device_mesh)
        return input_tensor

    def _apply(self, module, device_mesh):
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=self._prepare_input_fn,
        )

class RedistributeColWiseParallel(ColwiseParallel):

    def _partition_linear_fn(self, name, module, device_mesh):
        for name, param in module.named_parameters():
            if not isinstance(param, DTensor):
                raise ValueError(
                    f"Weight {name} in module {module.__class__.__name__} is not a DTensor!"
                )
            elif param.placements != (Shard(0),):
                dist_param = nn.Parameter(
                    param.redistribute(
                        device_mesh=device_mesh,
                        placements=[Shard(0)],
                    )
                )
                module.register_parameter(name, nn.Parameter(dist_param))
                free_storage(param.to_local())

    def _partition_embedding_fn(self, name, module, device_mesh):
        for name, param in module.named_parameters():
            if not isinstance(param, DTensor):
                raise ValueError(
                    f"Weight {name} in module {module.__class__.__name__} is not a DTensor!"
                )
            elif param.placements != (Shard(1),):
                dist_param = param.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(1)],
                )
                module.register_parameter(name, nn.Parameter(dist_param))
                free_storage(param.to_local())

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return redistribute_module_weights_only(
            module,
            device_mesh,
            partition_fn,
        )


class RedistributeRowWiseParallel(RowwiseParallel):

    def _partition_linear_fn(self, name, module, device_mesh):
        if not isinstance(module.weight, DTensor):
            raise ValueError(
                f"Weight {name} in module {module.__class__.__name__} is not a DTensor!"
            )
        elif module.weight.placements != (Shard(1),):
            weight = module.weight
            redistributed_weight = weight.redistribute(
                device_mesh=device_mesh,
                placements=[Shard(1)],
            )
            module.register_parameter("weight", nn.Parameter(redistributed_weight))
            free_storage(weight.to_local())

            if getattr(module, "bias", None) is not None:
                bias = module.bias
                redistributed_bias = bias.redistribute(
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
                module.register_parameter("bias", nn.Parameter(redistributed_bias))
                free_storage(bias.to_local())

    def _partition_embedding_fn(self, name, module, device_mesh):
        for name, param in module.named_parameters():
            if not isinstance(param, DTensor):
                raise ValueError(
                    f"Weight {name} in module {module.__class__.__name__} is not a DTensor!"
                )
            elif param.placements != (Shard(0),):
                dist_param = param.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(0)],
                )
                module.register_parameter(name, nn.Parameter(dist_param))
                free_storage(param.to_local())

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return redistribute_module_weights_only(
            module,
            device_mesh,
            partition_fn,
        )
