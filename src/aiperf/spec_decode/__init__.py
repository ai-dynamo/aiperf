# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.spec_decode.protocols import SpecDecodeAdapterProtocol
from aiperf.spec_decode.vllm_adapter import VLLMSpecDecodeAdapter

__all__ = [
    "SpecDecodeAdapterProtocol",
    "VLLMSpecDecodeAdapter",
]
