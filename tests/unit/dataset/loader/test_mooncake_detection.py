# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, PluginType


@pytest.mark.parametrize(
    "field,value",
    [
        param("assistant_responses", "live", id="live"),
        param("assistant_responses", "recorded", id="recorded"),
        param("assistant_responses", "LIVE", id="uppercase"),
        param("assistant_responses", "lve", id="invalid-value"),
        param("assistant_responses", None, id="null-value"),
        param("assistant_responses", 1, id="non-string-value"),
        param("message_mode", "delta", id="obsolete-name"),
    ],
)  # fmt: skip
def test_declared_response_mode_autodetects_as_mooncake(
    tmp_path: Path,
    create_cfg_and_composer: Callable[[], tuple[CLIConfig, CustomDatasetComposer]],
    field: str,
    value: Any,
) -> None:
    record = {
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Hello"}],
        field: value,
    }
    path = tmp_path / "trace.jsonl"
    path.write_bytes(orjson.dumps(record) + b"\n")
    _, composer = create_cfg_and_composer()

    detected, _ = DatasetResolver._detect_type(str(path))

    assert detected == CustomDatasetType.MOONCAKE_TRACE
    assert composer._infer_dataset_type(str(path)) == CustomDatasetType.MOONCAKE_TRACE


@pytest.mark.parametrize(
    "fields,error",
    [
        param({"assistant_responses": "lve"}, "assistant_responses", id="typo"),
        param({"assistant_responses": None}, "assistant_responses", id="null"),
        param({"assistant_responses": 1}, "assistant_responses", id="non-string"),
        param(
            {"message_mode": "delta"},
            "'message_mode' has been renamed to 'assistant_responses'",
            id="obsolete-name",
        ),
        param(
            {"assistant_responses": "live", "message_mode": "delta"},
            "'message_mode' has been renamed to 'assistant_responses'",
            id="both-names",
        ),
    ],
)  # fmt: skip
def test_invalid_response_mode_rejected_after_autodetection(
    tmp_path: Path,
    create_cfg_and_composer: Callable[[], tuple[CLIConfig, CustomDatasetComposer]],
    mock_prompt_generator: Mock,
    fields: dict[str, Any],
    error: str,
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(
        orjson.dumps(
            {
                "session_id": "s1",
                "messages": [{"role": "user", "content": "Hello"}],
                **fields,
            }
        )
        + b"\n"
    )
    _, composer = create_cfg_and_composer()
    detected = composer._infer_dataset_type(str(path))
    loader_class = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, detected)
    loader = loader_class(filename=path, prompt_generator=mock_prompt_generator)

    with pytest.raises(ValidationError, match=error):
        loader.load_dataset()
