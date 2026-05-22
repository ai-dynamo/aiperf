# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models import Turn


class TestTurnRawPayload:
    def test_default_none(self):
        t = Turn()
        assert t.raw_payload is None

    def test_set_and_round_trip(self):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "Qwen/Qwen3-0.6B",
            "max_tokens": 32,
        }
        t = Turn(role="user", raw_payload=payload)
        assert t.raw_payload == payload
        restored = Turn.model_validate(t.model_dump())
        assert restored.raw_payload == payload

    def test_raw_payload_does_not_disturb_other_fields(self):
        t = Turn(role="user", raw_payload={"messages": []})
        assert t.texts == []
        assert t.images == []
        assert t.raw_messages is None
