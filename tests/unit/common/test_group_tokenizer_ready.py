# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import msgspec

from aiperf.common.pod_lifecycle_structs import GroupTokenizerReady


def test_group_tokenizer_ready_round_trips() -> None:
    msg = GroupTokenizerReady(
        service_id="wgm-0",
        bundles={"gpt2": "/tmp/aiperf_tokenizers/run-1/gpt2"},
    )
    raw = msgspec.json.encode(msg)
    decoded = msgspec.json.decode(raw, type=GroupTokenizerReady)
    assert decoded == msg


def test_group_tokenizer_ready_failure_carries_error() -> None:
    msg = GroupTokenizerReady(
        service_id="wgm-0",
        bundles={},
        success=False,
        error_message="503 after retries",
    )
    assert msg.success is False
    assert msg.error_message == "503 after retries"
