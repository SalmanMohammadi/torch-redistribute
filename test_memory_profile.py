import functools

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.tensor import Shard
from torch.profiler import profile, ProfilerActivity, record_function
from torch_redistribute.model import FeedForward
from torch_redistribute.utils import print_tensor_distribution, printr, redistribute


def print_tensor_storage(tensor: torch.Tensor, name: str):
    """Print minimal unique identifying info about a tensor's storage."""
    printr(f"{name} storage id: {tensor.untyped_storage()._cdata}")


@torch.no_grad()
def main():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cpu", (world_size,))

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_modules=True,  # Track module info
    ) as prof:
        # model 1 - policy model being trained
        with record_function("create_model1"):
            model = FeedForward(dim=6, hidden_dim=8, bias=True)
            fully_shard(model, mesh=device_mesh)

        rank = dist.get_rank()
        print_tensor_storage(model.w2.weight, "model1.w2.weight")

        # model 2 - model being used for inference
        with record_function("create_model2"):
            with torch.device("meta"), torch.no_grad():
                model2 = FeedForward(dim=6, hidden_dim=8)

        print_tensor_storage(model2.w2.weight, "model2.w2.weight")
        printr("=" * 50)

        # Register parameters with cleaned names
        with record_function("register_parameters"):
            for module_a, module_b in zip(model.modules(), model2.modules()):
                if isinstance(module_a, nn.Linear):
                    for param_name, param in module_a.named_parameters():
                        printr(param_name, param)
                        torch.distributed.barrier()
                        with record_function("redistribute_and_register"):
                            module_b.register_parameter(
                                param_name,
                                nn.Parameter(
                                    param.redistribute(
                                        device_mesh=device_mesh,
                                        placements=[Shard(1)],
                                    )
                                ),
                            )

    if rank == 0:
        from pathlib import Path

        Path("./logs").mkdir(parents=True, exist_ok=True)
        # Export traces for different profilers
        prof.export_chrome_trace(
            "./logs/redistribution_trace.json"
        )  # For Chrome tracing
        prof.export_stacks(
            "./logs/profiler_stacks.txt", "self_cpu_time_total"
        )  # Flame graph

        # Export memory timeline
        with open("./logs/memory_timeline.json", "w") as f:
            memory_events = []
            for event in prof.events():
                if event.cpu_memory_usage != 0:  # Only include memory events
                    event_data = {
                        "name": event.name,
                        "ph": "X",  # Complete event
                        "ts": event.time_range.start,
                        "dur": event.time_range.end - event.time_range.start,
                        "pid": 0,
                        "tid": 0,
                        "args": {
                            "cpu_memory_usage": event.cpu_memory_usage,
                            "cpu_memory_usage_mb": event.cpu_memory_usage
                            / (1024 * 1024),
                        },
                    }
                    # Add shapes if available
                    if hasattr(event, "shapes"):
                        event_data["args"]["shapes"] = str(event.shapes)
                    memory_events.append(event_data)

            import json

            json.dump({"traceEvents": memory_events}, f, indent=2)

    printr("=" * 50)


if __name__ == "__main__":
    main()
