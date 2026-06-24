# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.plugin import plugins

if TYPE_CHECKING:
    from aiperf.common.config import UserConfig


def public_dataset_provenance(user_config: UserConfig) -> dict[str, object] | None:
    """Return stable source metadata for the configured public dataset."""
    input_config = user_config.input
    if not isinstance(input_config.public_dataset, str):
        return None

    loader = str(input_config.public_dataset)
    loader_metadata = plugins.get_public_dataset_loader_metadata(loader)
    hf_dataset_name = input_config.hf_weka_dataset or loader_metadata.hf_dataset_name
    hf_subset = (
        input_config.hf_dataset_subset
        if input_config.hf_dataset_subset is not None
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
    if "num_dataset_entries" in input_config.conversation.model_fields_set:
        provenance["num_dataset_entries"] = (
            input_config.conversation.num_dataset_entries
        )
    return provenance
