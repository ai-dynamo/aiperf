# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.dataset.loader.exgentic import ExgenticDatasetLoader
from aiperf.dataset.loader.exgentic_filters import V2_UNSUPPORTED_FILTER_PAIRS


class ExgenticV2DatasetLoader(ExgenticDatasetLoader):
    """Replay Exgentic v2 traces with the shared Exgentic converter."""

    hf_revision = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814"
    unsupported_filter_pairs = V2_UNSUPPORTED_FILTER_PAIRS
    supports_benchmark_filter = True
