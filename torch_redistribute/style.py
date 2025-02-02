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
    parallelize_module,
    ParallelStyle,
    RowwiseParallel,
)


class ReplicateParallel(ParallelStyle):
    def __init__(self):
        super().__init__()

    # def _partition_linear_fn(self, name, module, device_mesh):
    #     module.register_parameter(
    #         "weight",
    #         nn.Parameter(distribute_tensor(module.weight, device_mesh, [Replicate()])),
    #     )
    #     if getattr(module, "bias", None) is not None:
    #         # The Linear module has bias
    #         module.register_parameter(
    #             "bias",
    #             nn.Parameter(
    #                 distribute_tensor(module.bias, device_mesh, [Replicate()])
    #             ),
    #         )

    def _prepare_input_fn(mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(inputs[0], device_mesh)
        return input_tensor

    def _prepare_output_fn(mod, outputs, device_mesh):
        return outputs
    def _apply(self, module, device_mesh):
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=self._prepare_input_fn,
            output_fn=self._prepare_output_fn,
        )


class RedistributeColWiseParallel(ColwiseParallel):

    # @staticmethod
    # def _prepare_input_fn(
    #     input_layouts, desired_input_layouts, mod, inputs, device_mesh
    # ):
    #     import torch.distributed as dist

    #     torch.distributed.barrier()
    #     if dist.get_rank() == 0:
    #         import pdb

    #         pdb.set_trace()
    #     x = 0

    #     torch.distributed.barrier()
    #     return super(
    #         RedistributeColWiseParallel, RedistributeColWiseParallel
    #     )._prepare_input_fn(
    #         input_layouts, desired_input_layouts, mod, inputs, device_mesh
    #     )

    # @staticmethod
    # def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mes):
    #     import torch.distributed as dist

    #     torch.distributed.barrier()
    #     if dist.get_rank() == 0:
    #         import pdb

    #         pdb.set_trace()
    #     x = 0

    #     torch.distributed.barrier()
    #     return super()._prepare_output_fn(
    #         output_layouts, use_local_output, mod, outputs, device_mes
    #     )

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
