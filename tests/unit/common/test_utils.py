# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.utils import load_json_str


class TestLoadJsonStrErrors:
    """Tests that load_json_str raises the original error for both str and bytes."""

    @pytest.mark.parametrize(
        "json_str",
        [
            pytest.param("{not valid json}", id="invalid-str"),
            pytest.param("", id="empty-str"),
            pytest.param(b"", id="empty-bytes"),
            pytest.param(b"not json", id="invalid-bytes"),
            pytest.param('{"key": ', id="truncated"),
            pytest.param('{"key": 1,}', id="trailing-comma"),
        ],
    )  # fmt: skip
    def test_invalid_input_raises_decode_error(self, json_str: str | bytes) -> None:
        with pytest.raises(orjson.JSONDecodeError):
            load_json_str(json_str)

    def test_validation_func_error_propagates(self) -> None:
        def fail(_: object) -> None:
            raise ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            load_json_str('{"key": 1}', func=fail)
