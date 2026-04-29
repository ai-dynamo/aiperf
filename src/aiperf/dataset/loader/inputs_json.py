# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs JSON payload loader for verbatim API replay.

Loads AIPerf InputsFile format (``{"data": [{"session_id": "...", "payloads": [...]}]}``)
as raw payloads. Preserves multi-turn session structure. Each payload is sent
directly to the transport with zero endpoint formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseRawPayloadLoader
from aiperf.dataset.loader.models import InputsJsonSession


class InputsJsonPayloadLoader(BaseRawPayloadLoader):
    """Dataset loader for AIPerf inputs.json files with raw payloads.

    Reads a JSON file with structure::

        {"data": [{"session_id": "abc", "payloads": [{...}, {...}]}]}

    Each session maps to a multi-turn Conversation. Each payload in the
    ``payloads`` list becomes a Turn with ``raw_payload`` set, so the
    transport sends it verbatim without endpoint formatting.
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True for InputsFile format: top-level ``data`` list with ``payloads`` items."""
        if isinstance(data, dict):
            data_list = data.get("data")
            if isinstance(data_list, list) and len(data_list) > 0:
                first = data_list[0]
                if isinstance(first, dict) and isinstance(first.get("payloads"), list):
                    return True

        if filename is not None:
            path = Path(filename)
            if path.is_file() and path.suffix == ".json":
                try:
                    content = orjson.loads(path.read_bytes())
                    return cls.can_load(data=content)
                except Exception:
                    return False

        return False

    def load_dataset(self) -> dict[str, list[InputsJsonSession]]:
        """Load the JSON file and parse each entry into InputsJsonSession.

        Returns:
            Dictionary of session_id -> [InputsJsonSession].
        """
        path = Path(self.filename)
        content = orjson.loads(path.read_bytes())
        data_list = content["data"]

        result: dict[str, list[InputsJsonSession]] = {}
        for entry in data_list:
            session = InputsJsonSession(
                session_id=entry["session_id"],
                payloads=entry["payloads"],
            )
            result[session.session_id] = [session]

        self.info(
            f"Loaded {len(result)} sessions "
            f"({sum(len(s[0].payloads) for s in result.values())} total turns)"
        )
        return result

    def convert_to_conversations(
        self, data: dict[str, list[InputsJsonSession]]
    ) -> list[Conversation]:
        """Convert InputsJsonSession entries to Conversations with raw_payload turns.

        Args:
            data: Dictionary of session_id -> [InputsJsonSession].

        Returns:
            List of Conversations with multi-turn raw payloads.
        """
        conversations: list[Conversation] = []
        for session_id, sessions in data.items():
            for session in sessions:
                turns = [Turn(role="user", raw_payload=p) for p in session.payloads]
                conversations.append(
                    Conversation(
                        session_id=session_id,
                        turns=turns,
                        context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                    )
                )
        return conversations
