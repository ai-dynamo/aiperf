# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, EndpointType, PluginType


class TestRawEndpointRegistered:
    def test_raw_endpoint_class_resolves(self):
        cls = plugins.get_class(PluginType.ENDPOINT, EndpointType.RAW)
        assert cls is not None
        assert cls.__name__ == "RawEndpoint"


class TestLoaderEntriesRegistered:
    @pytest.mark.parametrize(
        "loader_name,expected_cls",
        [
            ("raw_payload", "RawPayloadDatasetLoader"),
            ("inputs_json", "InputsJsonPayloadLoader"),
        ],
    )
    def test_loader_resolves(self, loader_name: str, expected_cls: str):
        cls = plugins.get_class(
            PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType(loader_name)
        )
        assert cls is not None
        assert cls.__name__ == expected_cls
