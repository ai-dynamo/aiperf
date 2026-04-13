# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privacy-preserving trace anonymization.

Converts raw OpenAI-compatible chat logs into Mooncake traces with
block-hashed prefix patterns. Strips all text content, preserving
only token counts and hash ID sequences for prefix-cache-aware benchmarking.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import IO, TYPE_CHECKING

import orjson
from pydantic import Field, field_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher

if TYPE_CHECKING:
    from aiperf.common.tokenizer import Tokenizer

_logger = logging.getLogger(__name__)


class RawConversationRecord(AIPerfBaseModel):
    """A single raw conversation record from input JSONL."""

    messages: list[dict] = Field(
        description="OpenAI-compatible message array with 'role' and 'content' keys."
    )
    output: str = Field(
        description="Assistant response text, used only for output_length counting."
    )
    timestamp: int | float | None = Field(
        default=None,
        description="Request timestamp in milliseconds since trace start.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier for grouping multi-turn conversations.",
    )

    @field_validator("messages")
    @classmethod
    def messages_must_be_valid(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must contain at least one message")
        for i, msg in enumerate(v):
            if "role" not in msg:
                raise ValueError(f"message {i} missing required 'role' key")
            if "content" not in msg:
                raise ValueError(f"message {i} missing required 'content' key")
        return v


class AnonymizeResult(AIPerfBaseModel):
    """Result summary from trace anonymization."""

    total_processed: int = Field(
        description="Number of records successfully processed."
    )
    total_skipped: int = Field(description="Number of malformed records skipped.")
    sessions_detected: int = Field(description="Number of unique sessions found.")
    unique_hash_ids: int = Field(description="Number of unique hash IDs generated.")
    no_timestamps_warning: bool = Field(description="Whether input had no timestamps.")
    output_file: Path = Field(description="Path to the output file.")


def _process_record(
    record: RawConversationRecord,
    messages: list[dict],
    session_id: str | None,
    tokenizer: Tokenizer,
    hasher: RollingHasher,
    block_size: int,
    f: IO[str],
) -> int:
    """Process a single record and write to output file.

    Args:
        record: The raw conversation record.
        messages: Messages to tokenize (may be accumulated for multi-turn).
        session_id: Session ID to include in output, or None.
        tokenizer: Tokenizer instance.
        hasher: RollingHasher instance.
        block_size: Tokens per block.
        f: Output file handle.

    Returns:
        1 (count of records processed).
    """
    templated = tokenizer.apply_chat_template(messages)
    input_ids = tokenizer.encode(templated)
    output_ids = tokenizer.encode(record.output)

    blocks = [
        input_ids[i : i + block_size] for i in range(0, len(input_ids), block_size)
    ]
    hash_ids = hasher.hash_token_blocks(blocks) if blocks else []

    output_record: dict = {
        "input_length": len(input_ids),
        "output_length": len(output_ids),
        "hash_ids": hash_ids,
    }
    if record.timestamp is not None:
        output_record["timestamp"] = record.timestamp
    if session_id is not None:
        output_record["session_id"] = session_id

    f.write(orjson.dumps(output_record).decode() + "\n")
    return 1


def anonymize_trace(
    input_file: Path,
    output_file: Path,
    tokenizer: Tokenizer,
    block_size: int = 512,
) -> AnonymizeResult:
    """Convert raw chat logs into a privacy-preserving Mooncake trace.

    Reads OpenAI-compatible conversation records, tokenizes using the
    target model's tokenizer and chat template, hashes token blocks via
    RollingHasher, and writes Mooncake trace JSONL.

    Args:
        input_file: Path to input JSONL with raw conversation logs.
        output_file: Path to write output Mooncake trace JSONL.
        tokenizer: Tokenizer instance for the target model.
        block_size: Tokens per block for hashing.

    Returns:
        AnonymizeResult with processing summary.
    """
    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")

    hasher = RollingHasher(block_size=block_size)
    records: list[RawConversationRecord] = []
    total_skipped = 0

    with open(input_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
                record = RawConversationRecord.model_validate(data)
                records.append(record)
            except Exception as e:
                _logger.warning("Skipping line %d: %s", line_num, e)
                total_skipped += 1

    has_timestamps = any(r.timestamp is not None for r in records)

    # Per-session accumulated message history for multi-turn conversations
    session_history: dict[str, list[dict]] = defaultdict(list)
    session_ids_seen: set[str] = set()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_processed = 0

    # Process records in original order to preserve global request sequence.
    # Multi-turn sessions maintain per-session accumulated context.
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            if record.session_id is None:
                # Independent request: no message accumulation
                total_processed += _process_record(
                    record,
                    list(record.messages),
                    None,
                    tokenizer,
                    hasher,
                    block_size,
                    f,
                )
            else:
                # Multi-turn session: accumulate messages across turns
                sid = record.session_id
                session_ids_seen.add(sid)
                session_history[sid].extend(record.messages)
                total_processed += _process_record(
                    record,
                    session_history[sid],
                    sid,
                    tokenizer,
                    hasher,
                    block_size,
                    f,
                )
                session_history[sid].append(
                    {"role": "assistant", "content": record.output}
                )

    stats = hasher.get_stats()
    sessions_detected = len(session_ids_seen)

    return AnonymizeResult(
        total_processed=total_processed,
        total_skipped=total_skipped,
        sessions_detected=sessions_detected,
        unique_hash_ids=stats["total_hashes"],
        no_timestamps_warning=not has_timestamps,
        output_file=output_file,
    )
