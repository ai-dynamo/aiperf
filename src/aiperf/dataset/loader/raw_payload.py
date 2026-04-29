# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw payload JSONL loader for verbatim API replay.

Each JSONL line is a complete API request body sent directly to the transport
with zero formatting. Produces raw_payload on every turn for payload mmap bypass.

Supports two input modes:
- **Single file**: each line = one single-turn conversation.
- **Directory**: each ``.jsonl`` file = one multi-turn conversation, lines = turns.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseRawPayloadLoader
from aiperf.dataset.loader.models import RawPayload


class RawPayloadDatasetLoader(BaseRawPayloadLoader):
    """Dataset loader for raw payload JSONL files or directories.

    **Single file mode**: each line in the JSONL file is a complete API request
    payload (a JSON object containing at minimum a ``messages`` key). Each line
    becomes a single-turn conversation.

    **Directory mode**: each ``.jsonl`` file in the directory is one multi-turn
    conversation. Lines within a file are ordered turns. The filename (stem) is
    used as the session ID.

    Every Turn carries ``raw_payload`` -- the transport sends it verbatim
    without any endpoint formatting.
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True when data is a chat API payload or filename is a directory of JSONL files.

        Rejects agentic trajectory records (``conversation_id`` present) and
        InputsFile structures (``data`` key holding a list).
        """
        if data is not None:
            if not isinstance(data.get("messages"), list):
                return False
            if "conversation_id" in data:
                return False
            return not isinstance(data.get("data"), list)

        if filename is not None:
            path = Path(filename)
            if path.is_dir():
                return _dir_has_raw_payload_jsonl(path)

        return False

    def load_dataset(self) -> dict[str, list[RawPayload]]:
        """Load from a single JSONL file or a directory of JSONL files.

        - Single file: each line -> one session (single-turn).
        - Directory: each .jsonl file -> one session (multi-turn, lines = turns).

        Returns:
            Dictionary of session_id -> list[RawPayload].
        """
        path = Path(self.filename)
        if path.is_dir():
            return self._load_directory(path)
        return self._load_single_file(path)

    def _load_single_file(self, path: Path) -> dict[str, list[RawPayload]]:
        data: dict[str, list[RawPayload]] = defaultdict(list)
        with open(path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                payload = orjson.loads(line)
                session_id = self.session_id_generator.next()
                data[session_id].append(RawPayload(payload=payload))

        self.info(f"Loaded {len(data)} raw payload conversations from file")
        return dict(data)

    def _load_directory(self, directory: Path) -> dict[str, list[RawPayload]]:
        data: dict[str, list[RawPayload]] = {}
        total_turns = 0

        for jsonl_file in sorted(directory.glob("*.jsonl")):
            session_id = self.session_id_generator.next()
            payloads: list[RawPayload] = []
            with open(jsonl_file, "rb") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    payloads.append(RawPayload(payload=orjson.loads(line)))

            if payloads:
                data[session_id] = payloads
                total_turns += len(payloads)

        self.info(
            f"Loaded {len(data)} conversations ({total_turns} total turns) "
            f"from directory"
        )
        return data

    def convert_to_conversations(
        self, data: dict[str, list[RawPayload]]
    ) -> list[Conversation]:
        """Convert RawPayload entries to Conversations with raw_payload turns.

        Args:
            data: Dictionary of session_id -> [RawPayload].

        Returns:
            List of Conversations.
        """
        conversations: list[Conversation] = []
        for session_id, payloads in data.items():
            turns = [Turn(role="user", raw_payload=rp.payload) for rp in payloads]
            conversations.append(
                Conversation(
                    session_id=session_id,
                    turns=turns,
                    context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                )
            )
        return conversations


def _dir_has_raw_payload_jsonl(directory: Path) -> bool:
    """Check if a directory contains at least one JSONL file with a raw payload line."""
    for jsonl_file in directory.glob("*.jsonl"):
        try:
            with open(jsonl_file, "rb") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    record = orjson.loads(line)
                    return isinstance(record, dict) and isinstance(
                        record.get("messages"), list
                    )
        except Exception:
            continue
    return False
