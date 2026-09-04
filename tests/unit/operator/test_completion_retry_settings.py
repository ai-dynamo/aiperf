# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retry settings must be coerced before they are compared or used as a delay.

The gate reads two settings off OperatorEnvironment.RESULTS. Tests routinely
stub that object with a partial mock, whose auto-created attributes are mocks
rather than numbers -- and `mock <= 0` raises TypeError from inside a kopf
completion handler, on the results-fetch failure path, so the CR retries
forever instead of surfacing the real error. A None delay is worse than
useless: kopf reads it as "retry immediately", turning a paced retry into a
hot loop against the apiserver.
"""

from unittest.mock import MagicMock

import pytest

from aiperf.operator.handlers._completion_retry import _coerce_settings_float


class TestCoerceSettingsFloat:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (5, 5.0),
            (5.5, 5.5),
            ("7", 7.0),
            (0, 0.0),
        ],
    )  # fmt: skip
    def test_numeric_values_pass_through(self, raw, expected):
        assert _coerce_settings_float(raw) == expected

    def test_mock_is_coerced_and_never_raises(self):
        """A partially-stubbed settings object must not raise into kopf.

        MagicMock defines __float__, so coercion yields a number and the
        comparison downstream is safe -- which is the whole point. It is the
        raw `mock <= 0` that raises TypeError.
        """
        result = _coerce_settings_float(MagicMock(), default=5.0)
        assert isinstance(result, float)

    def test_raw_comparison_is_what_breaks(self):
        """Pin the actual failure this helper exists to prevent."""
        with pytest.raises(TypeError):
            _ = MagicMock() <= 0

    def test_none_falls_back_to_the_default(self):
        assert _coerce_settings_float(None, default=5.0) == 5.0

    def test_garbage_string_falls_back(self):
        assert _coerce_settings_float("not-a-number", default=2.5) == 2.5

    def test_default_is_zero_when_unspecified(self):
        """A missing budget disables the retry rather than enabling it."""
        assert _coerce_settings_float(object()) == 0.0
