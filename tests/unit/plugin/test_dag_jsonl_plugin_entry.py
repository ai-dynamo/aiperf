# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, PluginType


class TestDagJsonlRegistered:
    def test_enum_member_present(self):
        assert "dag_jsonl" in {m.value for m in CustomDatasetType}

    def test_loader_class_resolves(self):
        cls = plugins.get_class(
            PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType("dag_jsonl")
        )
        assert cls is not None
        assert cls.__name__ == "DagJsonlLoader"
