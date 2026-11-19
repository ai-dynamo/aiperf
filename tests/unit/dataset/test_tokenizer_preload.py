# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the forkserver tokenizer preload module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from aiperf.dataset import _tokenizer_preload


def test_preload_loads_tokenizer_in_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIPERF_LOADER_PRELOAD_TOKENIZER", "gpt2")
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(var, "unset")
        monkeypatch.delenv(var, raising=False)

    seen: dict[str, str | None] = {}

    def _record_env(*_args: object, **_kwargs: object) -> MagicMock:
        seen["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        seen["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return MagicMock()

    try:
        with patch(
            "aiperf.common.tokenizer.Tokenizer.from_pretrained",
            side_effect=_record_env,
        ):
            _tokenizer_preload._preload()
    finally:
        _tokenizer_preload.clear_preloaded()

    assert seen == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
