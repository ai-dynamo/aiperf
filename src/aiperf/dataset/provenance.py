# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.dataset.config import PublicDataset
from aiperf.plugin import plugins

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig


def public_dataset_provenance(cfg: BenchmarkConfig) -> dict[str, object] | None:
    """Return stable source metadata for the configured public dataset.

    Returns None unless the run's resolved dataset is a public dataset.
    """
    dataset = cfg.get_default_dataset()
    if not isinstance(dataset, PublicDataset):
        return None

    loader = str(dataset.dataset)
    loader_metadata = plugins.get_public_dataset_loader_metadata(loader)
    hf_dataset_name = dataset.hf_weka_dataset or loader_metadata.hf_dataset_name
    hf_subset = (
        dataset.hf_subset
        if dataset.hf_subset is not None
        else loader_metadata.hf_subset
    )

    provenance: dict[str, object] = {
        "source_type": "public_dataset",
        "loader": loader,
    }
    if hf_dataset_name is not None:
        provenance.update(
            {
                "hf_dataset_name": hf_dataset_name,
                "hf_split": loader_metadata.hf_split,
            }
        )
    if hf_subset is not None:
        provenance["hf_subset"] = hf_subset
    if dataset.entries_explicit:
        provenance["num_dataset_entries"] = dataset.entries
    return provenance
