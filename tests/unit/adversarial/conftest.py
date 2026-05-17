# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-export the records parser fixtures so adversarial probe files can
depend on ``setup_inference_parser`` without copy-pasting the mocked
``CommunicationMixin`` setup."""

from tests.unit.records.conftest import (  # noqa: F401
    inference_result_parser,
    mock_tokenizer,
    sample_turn,
    setup_inference_parser,
)
