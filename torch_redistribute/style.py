import torch
import torch.nn as nn
from torch.distributed.tensor import (
    distribute_module,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)


class ReplicateParallel(ParallelStyle):
    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
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
        if not isinstance(module.weight, DTensor):
            super()._partition_linear_fn(name, module, device_mesh)
        else:
            module.weight = nn.Parameter(
                module.weight.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(0)],
                )
            )
            if getattr(module, "bias", None) is not None:
                module.bias = nn.Parameter(
                    module.bias.redistribute(
                        device_mesh=device_mesh,
                        placements=[Replicate()],
                    )
                )

    def _partition_embedding_fn(self, name, module, device_mesh):
        for param in module.parameters():
            if not isinstance(param, DTensor):
                super()._partition_embedding_fn(name, module, device_mesh)
            else:
                param = nn.Parameter(
                    param.redistribute(
                        device_mesh=device_mesh,
                        placements=[Shard(0)],
                    )
                )


class RedistributeRowWiseParallel(RowwiseParallel):

    def _partition_linear_fn(self, name, module, device_mesh):
        if not isinstance(module.weight, DTensor):
            super()._partition_linear_fn(name, module, device_mesh)
        else:
            module.weight = nn.Parameter(
                module.weight.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(1)],
                )
            )
            if getattr(module, "bias", None) is not None:
                module.bias = nn.Parameter(
                    module.bias.redistribute(
                        device_mesh=device_mesh,
                        placements=[Replicate()],
                    )
                )

    def _partition_embedding_fn(self, name, module, device_mesh):
        for param in module.parameters():
            if not isinstance(param, DTensor):
                super()._partition_embedding_fn(name, module, device_mesh)
            else:
                param = nn.Parameter(
                    param.redistribute(
                        device_mesh=device_mesh,
                        placements=[Shard(1)],
                    )
                )
