from contextlib import contextmanager
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage
from torch.distributed.tensor import DTensor

from torch_redistribute.utils import printr, redistribute


class RedistributeContext:
    def __init__(
        self,
        model: nn.Module,
        model_cls: Callable,
        device_mesh: torch.distributed.device_mesh.DeviceMesh,
    ):
        self.model = model
        with torch.device("meta"), torch.no_grad():
            self.generate_model = model_cls()
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
            self.model.named_modules(), self.generate_model.named_modules()
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
                    free_storage(generate_param)
                    free_storage(param)

    def _to_tensor_parallel(self):
        redistribute(self.model, self.device_mesh)
        for (module_name, module_a), (_, module_b) in zip(
            self.model.named_modules(), self.generate_model.named_modules()
        ):
            if module_name:
                for param_name, param in module_a.named_parameters():
                    printr(module_a, param)
                    module_b.register_parameter(param_name, param)

    def __enter__(self):
        # torch.distributed.barrier()
        self._to_tensor_parallel()
        return self.generate_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # torch.distributed.barrier()
        self._restore_fsdp_state()
