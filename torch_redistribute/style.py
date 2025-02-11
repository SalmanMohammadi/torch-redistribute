import torch
import torch.nn as nn
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
        for name, param in module.named_parameters():
            if isinstance(param, DTensor):
                if param.placements == [Shard(0)]:
                    continue
                dist_param = nn.Parameter(
                    param.redistribute(
                        device_mesh=device_mesh,
                        placements=[Shard(0)],
                    )
                )
                del param
            else:
                dist_param = distribute_tensor(
                    param,
                    device_mesh,
                    [Shard(0)],
                    src_data_rank=self.src_data_rank,
                )
            module.register_parameter(name, nn.Parameter(dist_param))

    def _partition_embedding_fn(self, name, module, device_mesh):
        for param in module.parameters():
            if not isinstance(param, DTensor):
                super()._partition_embedding_fn(name, module, device_mesh)
            else:
                old_weight = param
                redistributed_weight = param.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(0)],
                )
                param = nn.Parameter(redistributed_weight)
                del old_weight
                del redistributed_weight


class RedistributeRowWiseParallel(RowwiseParallel):

    def _partition_linear_fn(self, name, module, device_mesh):
        if not isinstance(module.weight, DTensor):
            super()._partition_linear_fn(name, module, device_mesh)
        elif module.weight.placements != [Shard(1)]:
            weight = module.weight
            redistributed_weight = weight.redistribute(
                device_mesh=device_mesh,
                placements=[Shard(1)],
            )
            del weight
            module.register_parameter("weight", nn.Parameter(redistributed_weight))
            if getattr(module, "bias", None) is not None:
                bias = module.bias
                redistributed_bias = bias.redistribute(
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
                del bias
                module.register_parameter("bias", nn.Parameter(redistributed_bias))

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
