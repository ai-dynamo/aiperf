# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.dataset.config import FileDataset, PublicDataset
from aiperf.config.dataset.remote_ref import RemoteRef
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


def remote_dataset_provenance(cfg: BenchmarkConfig) -> dict[str, object] | None:
    """Return the exact remote source a run replayed, or None for non-remote datasets.

    Written into exported artifacts, so it carries only redacted references
    (``RemoteRef.display`` and the stored ``pinned_ref``), never a presigned value.
    """
    dataset = cfg.get_default_dataset()
    if not isinstance(dataset, FileDataset) or not isinstance(dataset.path, RemoteRef):
        return None
    ref = dataset.path
    provenance: dict[str, object] = {
        "source_type": "remote_file",
        "ref": ref.display,
        "resolved_version": ref.resolved_version,
    }
    from aiperf.dataset.remote.service import cached_source

    try:
        source = cached_source(ref)
    except RuntimeError:
        # Exporters can run where the cache is absent; keep the identity we have.
        return provenance
    provenance.update(
        {
            "pinned_ref": source.probe.pinned_ref,
            "format": str(dataset.format)
            if "format" in dataset.model_fields_set
            else None,
            "files": [f.model_dump(mode="json") for f in source.files],
            "manifest_sha256": source.probe.manifest_sha256,
        }
    )
    if source.probe.data_files is not None and source.probe.manifest is None:
        provenance["layout"] = "hf_parquet"
    return provenance
