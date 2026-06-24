# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiperf.plugin.enums import PluginType
from aiperf.records.records_manager_processing import load_results_processors


class _ValidResultsProcessor:
    def __init__(self, **_: object) -> None:
        pass

    async def process_result(self, record_data: object) -> None:
        pass

    async def summarize(self) -> list[object]:
        return []

    async def finalize(self) -> None:
        pass


class _SideChannelProcessor:
    def __init__(self, **_: object) -> None:
        pass

    async def process_record(self, record: object) -> None:
        pass


def test_load_results_processors_skips_plugins_without_results_protocol() -> None:
    host = MagicMock()
    host.service_id = "records-manager"
    host.run = MagicMock()
    host.pub_client = MagicMock()
    entries = [SimpleNamespace(name="side"), SimpleNamespace(name="valid")]

    def get_class(category: PluginType, name: str) -> type:
        assert category == PluginType.RESULTS_PROCESSOR
        return _SideChannelProcessor if name == "side" else _ValidResultsProcessor

    with (
        patch(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            return_value=iter(entries),
        ),
        patch(
            "aiperf.records.records_manager_processing.plugins.get_class",
            side_effect=get_class,
        ),
    ):
        processors = load_results_processors(host)

    assert [processor.__class__ for processor in processors] == [_ValidResultsProcessor]
    host.attach_child_lifecycle.assert_called_once_with(processors[0])
