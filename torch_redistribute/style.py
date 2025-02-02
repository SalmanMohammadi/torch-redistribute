import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, Replicate, Shard, DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)


class RedistributeColWiseParallel(ColwiseParallel):

    def _partition_linear_fn(self, name, module, device_mesh):
        return super()._partition_linear_fn(name, module, device_mesh)

    def _partition_embedding_fn(self, name, module, device_mesh):
        return super()._partition_embedding_fn(name, module, device_mesh)



class RedistributeRowWiseParallel(RowwiseParallel):

    def _partition_linear_fn(self, name, module, device_mesh):
        if not isinstance(module.weight, DTensor):
            super()._partition_linear_fn(name, module, device_mesh)
        else:
            module.weight = nn.Parameter(module.weight.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(1)],
                ))
            if getattr(module, "bias", None) is not None:
                module.bias = nn.Parameter(module.bias.redistribute(
                        device_mesh=device_mesh,
                        placements=[Replicate()],
                    ))
            
    def _partition_embedding_fn(self, name, module, device_mesh):
        for param in module.parameters():
            if not isinstance(param, DTensor):
                super()._partition_embedding_fn(name, module, device_mesh)
            else:   
                param = nn.Parameter(param.redistribute(
                    device_mesh=device_mesh,
                    placements=[Shard(1)],
                ))
            
