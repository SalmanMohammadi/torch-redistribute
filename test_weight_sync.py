import functools

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.distributed.tensor import Replicate, DTensor
from torch_redistribute.model import FeedForward
from torch_redistribute.utils import print_tensor_distribution, printr, redistribute


def print_tensor_storage(tensor: torch.Tensor, name: str):
    """Print minimal unique identifying info about a tensor's storage."""
    printr(f"{name} storage id: {tensor.untyped_storage()._cdata}")


def main():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=["data"])

    # model 1 - policy model being trained
    model = FeedForward(dim=512, hidden_dim=512, bias=False)
    fully_shard(
        model, mesh=device_mesh
    )  # This is fine since only Linear layers have parameters
    printr(model)
    rank = dist.get_rank()
    baseline_dtensor_specs = {}
    for m_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_params = {}
            for name, param in module.named_parameters():
                module_params[name] = param.placements

            baseline_dtensor_specs[m_name] = module_params
    print_tensor_storage(model.w2.weight, "model1.w2.weight")
    print_tensor_distribution(model.w2.weight, "model1.w2.weight", rank)
    torch.distributed.barrier()
    # model 2 - model being used for inference
    with torch.device("meta"), torch.no_grad():
        model2 = FeedForward(dim=6, hidden_dim=8)

    printr("Now redistributing base model")

    printr("=" * 50)

    redistribute(model, device_mesh)
    print_tensor_storage(model.w2.weight, "model1.w2.weight")
    printr("=" * 50)

    printr("Now registering parameters")
    for module_a, module_b in zip(model.modules(), model2.modules()):
        torch.distributed.barrier()
        if isinstance(module_a, nn.Linear):
            for param_name, param in module_a.named_parameters():
                torch.distributed.barrier()
                module_b.register_parameter(param_name, param)

    # printr(model2, model2._parameters)
    torch.distributed.barrier()

    print_tensor_storage(model.w2.weight, "model.w2.weight")
    print_tensor_storage(model2.w2.weight, "model2.w2.weight")
    printr("=" * 50)

    # printr("Redistributing to shard on dim 1", "=" * 50)
    # redistribute(model, device_mesh)
    print_tensor_distribution(model2.w2.weight, "model2.w2.weight", rank)
    print_tensor_distribution(model.w2.weight, "w2.weight", rank)
    printr("=" * 50)
    printr("Now redistributing base model to FSDP")
    torch.distributed.barrier()
    for m_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for name, param in module.named_parameters():
                if param.placements != baseline_dtensor_specs[m_name][name]:
                    printr(
                        "Registering parameter",
                        name,
                        param.placements,
                        baseline_dtensor_specs[m_name][name],
                    )
                    module.register_parameter(
                        name,
                        nn.Parameter(
                            param.redistribute(
                                device_mesh=device_mesh,
                                placements=baseline_dtensor_specs[m_name][name],
                            ),
                        ),
                    )
                    del param

    print_tensor_distribution(model.w2.weight, "w2.weight", rank)
    print_tensor_storage(model.w2.weight, "model.w2.weight")
    print_tensor_storage(model2.w2.weight, "model2.w2.weight")
    printr("=" * 50)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
