# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error-type classification for the OTel GenAI semantic convention mapping.

Regression coverage for ``_classify_error_type``: a client-side cancellation
(``--request-cancellation-rate``) surfaces as HTTP 499 (Client Closed Request)
with a ``RequestCancellationError`` type and must classify as ``cancelled``, not
collapse into the generic ``http_4xx`` bucket alongside real 4xx server errors.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.models import ErrorDetails
from aiperf.post_processors.strategies.genai_semconv import _classify_error_type


@pytest.mark.parametrize(
    "error,expected",
    [
        param(
            ErrorDetails(
                message="Request cancelled after 5s",
                code=499,
                type="RequestCancellationError",
            ),
            "cancelled",
            id="code_499_with_cancellation_type",
        ),
        param(
            ErrorDetails(message="Request cancelled", code=499),
            "cancelled",
            id="code_499_alone",
        ),
        param(
            ErrorDetails(
                message="Request cancelled", type="RequestCancellationError"
            ),
            "cancelled",
            id="cancellation_type_without_code",
        ),
        param(
            ErrorDetails(message="Bad Request", code=400),
            "http_4xx",
            id="real_400_stays_http_4xx",
        ),
        param(
            ErrorDetails(message="Not Found", code=404, type="HTTPStatusError"),
            "http_4xx",
            id="real_404_stays_http_4xx",
        ),
        param(
            ErrorDetails(message="Server Error", code=503),
            "http_5xx",
            id="real_503_stays_http_5xx",
        ),
        param(
            ErrorDetails(message="timed out", type="TimeoutError"),
            "timeout",
            id="timeout_type",
        ),
        param(None, None, id="no_error_returns_none"),
    ],
)  # fmt: skip
def test_classify_error_type(error: ErrorDetails | None, expected: str | None) -> None:
    assert _classify_error_type(error) == expected
