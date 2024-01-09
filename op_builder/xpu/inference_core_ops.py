# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from .builder import SYCLAutoOpBuilder


class InferenceCoreBuilder(SYCLAutoOpBuilder):
    BUILD_VAR = "DS_BUILD_INFERENCE_CORE_OPS"
    NAME = "inference_core_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.kernels{self.NAME}'

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 6:
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def get_prefix(self):
        ds_path = self.deepspeed_src_path("deepspeed")
        return "deepspeed" if os.path.isdir(ds_path) else ".."

    def sources(self):
        sources = [
            "inference/v2/kernels/core_ops/core_ops.cpp",
            "inference/v2/kernels/core_ops/bias_activations/bias_activation.cpp",
            "inference/v2/kernels/core_ops/bias_activations/bias_activation.cu",
            "inference/v2/kernels/core_ops/cuda_layer_norm/layer_norm.cpp",
            "inference/v2/kernels/core_ops/cuda_layer_norm/layer_norm.cu",
            "inference/v2/kernels/core_ops/cuda_rms_norm/rms_norm.cpp",
            "inference/v2/kernels/core_ops/cuda_rms_norm/rms_norm.cu",
            "inference/v2/kernels/core_ops/gated_activations/gated_activation_kernels.cpp",
            "inference/v2/kernels/core_ops/gated_activations/gated_activation_kernels.cu",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        return []

    def include_paths(self):
        sources = [
            'inference/v2/kernels/core_ops/bias_activations',
            'inference/v2/kernels/core_ops/blas_kernels',
            'inference/v2/kernels/core_ops/cuda_layer_norm',
            'inference/v2/kernels/core_ops/cuda_rms_norm',
            'inference/v2/kernels/core_ops/gated_activations',
            'inference/v2/kernels/includes',
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]

        return sources
