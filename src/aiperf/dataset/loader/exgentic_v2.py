# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.dataset.loader.exgentic import ExgenticDatasetLoader


class ExgenticV2DatasetLoader(ExgenticDatasetLoader):
    """Replay Exgentic v2 traces with the shared Exgentic converter."""

    hf_revision = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814"
