# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.exceptions import AIPerfMultiError
from aiperf.common.utils import (
    call_all_functions,
    call_all_functions_self,
    load_json_str,
)


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


class TestCallAllFunctions:
    """Test call_all_functions and call_all_functions_self error handling."""

    @pytest.mark.asyncio
    async def test_call_all_functions_logs_and_raises_on_error(self) -> None:
        def bad_func() -> None:
            raise RuntimeError("boom")

        with pytest.raises(AIPerfMultiError):
            await call_all_functions([bad_func])

    @pytest.mark.asyncio
    async def test_call_all_functions_self_logs_and_raises_on_error(self) -> None:
        class Dummy:
            pass

        def bad_method(self_) -> None:
            raise RuntimeError("boom")

        with pytest.raises(AIPerfMultiError):
            await call_all_functions_self(Dummy(), [bad_method])
