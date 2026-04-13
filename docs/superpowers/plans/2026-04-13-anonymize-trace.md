# `aiperf anonymize-trace` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `aiperf anonymize-trace` CLI command that converts raw OpenAI-compatible chat logs into privacy-preserving Mooncake traces with block-hashed prefix patterns.

**Architecture:** Thin CLI command in `cli_commands/` delegates to core logic in `dataset/synthesis/anonymize.py`. Core logic uses the existing `RollingHasher` for block hashing and AIPerf's `Tokenizer` wrapper for tokenization. The `Tokenizer` class needs an `apply_chat_template` method added since it doesn't currently expose this HuggingFace functionality.

**Tech Stack:** Cyclopts (CLI), Pydantic (input validation), orjson (JSON I/O), Rich (summary output), existing RollingHasher, existing Tokenizer wrapper.

**Spec:** `docs/superpowers/specs/2026-04-13-anonymize-trace-design.md`

---

### Task 1: Add `apply_chat_template` to `Tokenizer`

**Files:**
- Modify: `src/aiperf/common/tokenizer.py`
- Test: `tests/unit/dataset/synthesis/test_anonymize.py` (new file)

The AIPerf `Tokenizer` wrapper doesn't expose `apply_chat_template`. We need it to convert message arrays into templated strings before tokenizing.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/synthesis/test_anonymize.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trace anonymization."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.tokenizer import Tokenizer


class TestTokenizerApplyChatTemplate:
    """Tests for Tokenizer.apply_chat_template."""

    def test_apply_chat_template_delegates_to_underlying(self) -> None:
        """Test that apply_chat_template calls the underlying tokenizer."""
        tokenizer = Tokenizer()
        mock_hf = MagicMock()
        mock_hf.apply_chat_template.return_value = "<|user|>Hello<|end|>"
        tokenizer._tokenizer = mock_hf

        messages = [{"role": "user", "content": "Hello"}]
        result = tokenizer.apply_chat_template(messages)

        assert result == "<|user|>Hello<|end|>"
        mock_hf.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )

    def test_apply_chat_template_not_initialized_raises(self) -> None:
        """Test that calling apply_chat_template before init raises."""
        tokenizer = Tokenizer()

        with pytest.raises(Exception, match="not initialized"):
            tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run pytest tests/unit/dataset/synthesis/test_anonymize.py::TestTokenizerApplyChatTemplate -v`
Expected: FAIL — `Tokenizer` has no `apply_chat_template` method.

- [ ] **Step 3: Implement `apply_chat_template` on `Tokenizer`**

Add to `src/aiperf/common/tokenizer.py`, inside the `Tokenizer` class, after the `decode` method:

```python
    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply the model's chat template to a list of messages.

        Converts an OpenAI-compatible message array into a formatted string
        using the model's chat template (e.g., ChatML, Llama format).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tokenize: Whether to return token IDs instead of string.
            add_generation_prompt: Whether to append the generation prompt.

        Returns:
            Formatted string with the chat template applied.
        """
        self._require_init()
        return self._tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run pytest tests/unit/dataset/synthesis/test_anonymize.py::TestTokenizerApplyChatTemplate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add src/aiperf/common/tokenizer.py tests/unit/dataset/synthesis/test_anonymize.py
git commit -s -m "feat: add apply_chat_template to Tokenizer wrapper"
```

---

### Task 2: Input validation model

**Files:**
- Create: `src/aiperf/dataset/synthesis/anonymize.py`
- Test: `tests/unit/dataset/synthesis/test_anonymize.py` (append)

Define the Pydantic model for validating input JSONL records.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dataset/synthesis/test_anonymize.py`:

```python
import orjson

from aiperf.dataset.synthesis.anonymize import RawConversationRecord


class TestRawConversationRecord:
    """Tests for input record validation."""

    def test_valid_single_turn(self) -> None:
        """Test parsing a valid single-turn record."""
        data = {
            "timestamp": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "output": "Hi there",
        }
        record = RawConversationRecord.model_validate(data)
        assert record.timestamp == 100
        assert len(record.messages) == 1
        assert record.output == "Hi there"
        assert record.session_id is None

    def test_valid_multi_turn(self) -> None:
        """Test parsing a valid multi-turn record with session_id."""
        data = {
            "timestamp": 200,
            "session_id": "sess_1",
            "messages": [{"role": "user", "content": "Explain ML"}],
            "output": "Machine learning is...",
        }
        record = RawConversationRecord.model_validate(data)
        assert record.session_id == "sess_1"

    def test_missing_messages_raises(self) -> None:
        """Test that missing messages field raises validation error."""
        data = {"timestamp": 0, "output": "response"}
        with pytest.raises(Exception):
            RawConversationRecord.model_validate(data)

    def test_missing_output_raises(self) -> None:
        """Test that missing output field raises validation error."""
        data = {"timestamp": 0, "messages": [{"role": "user", "content": "Hi"}]}
        with pytest.raises(Exception):
            RawConversationRecord.model_validate(data)

    def test_empty_messages_raises(self) -> None:
        """Test that empty messages array raises validation error."""
        data = {"timestamp": 0, "messages": [], "output": "response"}
        with pytest.raises(Exception):
            RawConversationRecord.model_validate(data)

    def test_no_timestamp_is_valid(self) -> None:
        """Test that timestamp is optional."""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "output": "Hi",
        }
        record = RawConversationRecord.model_validate(data)
        assert record.timestamp is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run pytest tests/unit/dataset/synthesis/test_anonymize.py::TestRawConversationRecord -v`
Expected: FAIL — `anonymize` module doesn't exist.

- [ ] **Step 3: Create the input model**

Create `src/aiperf/dataset/synthesis/anonymize.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privacy-preserving trace anonymization.

Converts raw OpenAI-compatible chat logs into Mooncake traces with
block-hashed prefix patterns. Strips all text content, preserving
only token counts and hash ID sequences for prefix-cache-aware benchmarking.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from aiperf.common.models import AIPerfBaseModel


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
    def messages_must_be_non_empty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run pytest tests/unit/dataset/synthesis/test_anonymize.py::TestRawConversationRecord -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add src/aiperf/dataset/synthesis/anonymize.py tests/unit/dataset/synthesis/test_anonymize.py
git commit -s -m "feat: add RawConversationRecord input model for anonymize-trace"
```

---

### Task 3: Core anonymization logic

**Files:**
- Modify: `src/aiperf/dataset/synthesis/anonymize.py`
- Test: `tests/unit/dataset/synthesis/test_anonymize.py` (append)

Implement the `anonymize_trace` function that reads input, tokenizes, hashes, and writes output.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/dataset/synthesis/test_anonymize.py`:

```python
import tempfile
from pathlib import Path

from aiperf.dataset.synthesis.anonymize import anonymize_trace


class TestAnonymizeTrace:
    """Tests for the core anonymize_trace function."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer that simulates realistic behavior."""
        tokenizer = MagicMock()
        # apply_chat_template returns a formatted string
        tokenizer.apply_chat_template.return_value = "<|user|>Hello<|end|>"
        # encode returns token IDs (10 tokens for any input)
        tokenizer.encode.return_value = list(range(10))
        return tokenizer

    def test_single_turn_produces_valid_output(self, mock_tokenizer: MagicMock) -> None:
        """Test that single-turn input produces valid Mooncake trace output."""
        input_data = [
            {"timestamp": 0, "messages": [{"role": "user", "content": "Hello"}], "output": "Hi there"},
            {"timestamp": 100, "messages": [{"role": "user", "content": "Bye"}], "output": "Goodbye"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output.jsonl")

        try:
            result = anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            assert output_path.exists()
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

            record_0 = orjson.loads(lines[0])
            assert "timestamp" in record_0
            assert "input_length" in record_0
            assert "output_length" in record_0
            assert "hash_ids" in record_0
            assert isinstance(record_0["hash_ids"], list)
            assert record_0["timestamp"] == 0

            # No text content should be in the output
            record_str = lines[0]
            assert "Hello" not in record_str
            assert "Hi there" not in record_str

            assert result.total_processed == 2
            assert result.total_skipped == 0
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_multi_turn_prefix_sharing(self, mock_tokenizer: MagicMock) -> None:
        """Test that multi-turn sessions produce shared prefix hash_ids."""
        # Make encode return different lengths for accumulated vs single messages
        call_count = 0
        def mock_encode(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First session turn: 8 tokens (2 blocks of 4)
                return list(range(8))
            else:
                # Second session turn (accumulated): 16 tokens (4 blocks of 4)
                return list(range(16))
        mock_tokenizer.encode.side_effect = mock_encode

        # Template returns different strings for different message counts
        def mock_template(messages, tokenize=False, add_generation_prompt=True):
            return f"template_{len(messages)}"
        mock_tokenizer.apply_chat_template.side_effect = mock_template

        input_data = [
            {"timestamp": 0, "session_id": "s1", "messages": [{"role": "user", "content": "Hello"}], "output": "Hi"},
            {"timestamp": 100, "session_id": "s1", "messages": [{"role": "user", "content": "More"}], "output": "Sure"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_mt.jsonl")

        try:
            anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

            record_0 = orjson.loads(lines[0])
            record_1 = orjson.loads(lines[1])

            # Both should have session_id
            assert record_0["session_id"] == "s1"
            assert record_1["session_id"] == "s1"

            # Turn 2 should have more hash_ids than turn 1
            assert len(record_1["hash_ids"]) > len(record_0["hash_ids"])

            # Turn 2's first hash_ids should match turn 1's (shared prefix)
            for i in range(len(record_0["hash_ids"])):
                assert record_0["hash_ids"][i] == record_1["hash_ids"][i]
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_missing_timestamp_warning(self, mock_tokenizer: MagicMock) -> None:
        """Test that missing timestamps produce a warning in the result."""
        input_data = [
            {"messages": [{"role": "user", "content": "Hello"}], "output": "Hi"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_no_ts.jsonl")

        try:
            result = anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            assert result.no_timestamps_warning

            lines = output_path.read_text().strip().split("\n")
            record = orjson.loads(lines[0])
            assert "timestamp" not in record
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_prefix_overlap_across_requests(self, mock_tokenizer: MagicMock) -> None:
        """Test that independent requests with shared prefixes produce shared hash_ids."""
        # Both requests produce the same 8 tokens (same system prompt + similar question)
        mock_tokenizer.encode.return_value = list(range(8))
        mock_tokenizer.apply_chat_template.return_value = "same_template"

        input_data = [
            {"timestamp": 0, "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Q1"}], "output": "A1"},
            {"timestamp": 100, "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Q2"}], "output": "A2"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in input_data:
                f.write(orjson.dumps(record).decode() + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_name("output_prefix.jsonl")

        try:
            anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            lines = output_path.read_text().strip().split("\n")
            record_0 = orjson.loads(lines[0])
            record_1 = orjson.loads(lines[1])

            # Same tokenization = same hash_ids (identical prefix)
            assert record_0["hash_ids"] == record_1["hash_ids"]
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_malformed_line_skipped(self, mock_tokenizer: MagicMock) -> None:
        """Test that malformed lines are skipped with count."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"messages": [{"role": "user", "content": "Good"}], "output": "Hi"}\n')
            f.write('not valid json\n')
            f.write('{"messages": [], "output": "empty"}\n')
            f.write('{"messages": [{"role": "user", "content": "Also good"}], "output": "Bye"}\n')
            input_path = Path(f.name)

        output_path = input_path.with_name("output_skip.jsonl")

        try:
            result = anonymize_trace(
                input_file=input_path,
                output_file=output_path,
                tokenizer=mock_tokenizer,
                block_size=4,
            )

            assert result.total_processed == 2
            assert result.total_skipped == 2

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run pytest tests/unit/dataset/synthesis/test_anonymize.py::TestAnonymizeTrace -v`
Expected: FAIL — `anonymize_trace` function doesn't exist.

- [ ] **Step 3: Implement the core anonymization function**

Update `src/aiperf/dataset/synthesis/anonymize.py` — add imports and the function after the `RawConversationRecord` class:

```python
# Add these imports at the top (after existing imports):
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from pydantic import Field, field_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher

if TYPE_CHECKING:
    from aiperf.common.tokenizer import Tokenizer

_logger = logging.getLogger(__name__)


class AnonymizeResult(AIPerfBaseModel):
    """Result summary from trace anonymization."""

    total_processed: int = Field(description="Number of records successfully processed.")
    total_skipped: int = Field(description="Number of malformed records skipped.")
    sessions_detected: int = Field(description="Number of unique sessions found.")
    unique_hash_ids: int = Field(description="Number of unique hash IDs generated.")
    no_timestamps_warning: bool = Field(
        description="Whether input had no timestamps."
    )
    output_file: Path = Field(description="Path to the output file.")


def anonymize_trace(
    input_file: Path,
    output_file: Path,
    tokenizer: "Tokenizer",
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
    hasher = RollingHasher(block_size=block_size)
    records: list[RawConversationRecord] = []
    total_skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
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

    # Group by session_id
    sessions: dict[str | None, list[RawConversationRecord]] = defaultdict(list)
    for record in records:
        sessions[record.session_id].append(record)

    # Sort turns within each session by timestamp (or preserve input order)
    for session_records in sessions.values():
        session_records.sort(key=lambda r: r.timestamp if r.timestamp is not None else 0)

    # Check for timestamps
    has_timestamps = any(r.timestamp is not None for r in records)

    # Process sessions and write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_processed = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for session_id, session_records in sessions.items():
            accumulated_messages: list[dict] = []

            for record in session_records:
                accumulated_messages.extend(record.messages)

                # Apply chat template and tokenize
                templated = tokenizer.apply_chat_template(accumulated_messages)
                input_ids = tokenizer.encode(templated)
                output_ids = tokenizer.encode(record.output)

                # Split into blocks and hash
                blocks = [
                    input_ids[i : i + block_size]
                    for i in range(0, len(input_ids), block_size)
                ]
                hash_ids = hasher.hash_token_blocks(blocks) if blocks else []

                # Build output record
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
                total_processed += 1

                # Add assistant response to history for next turn
                accumulated_messages.append(
                    {"role": "assistant", "content": record.output}
                )

    stats = hasher.get_stats()
    sessions_detected = sum(1 for sid in sessions if sid is not None)

    return AnonymizeResult(
        total_processed=total_processed,
        total_skipped=total_skipped,
        sessions_detected=sessions_detected,
        unique_hash_ids=stats["total_hashes"],
        no_timestamps_warning=not has_timestamps,
        output_file=output_file,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run pytest tests/unit/dataset/synthesis/test_anonymize.py::TestAnonymizeTrace -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add src/aiperf/dataset/synthesis/anonymize.py tests/unit/dataset/synthesis/test_anonymize.py
git commit -s -m "feat: implement core anonymize_trace function"
```

---

### Task 4: CLI command

**Files:**
- Create: `src/aiperf/cli_commands/anonymize_trace.py`
- Modify: `src/aiperf/cli.py`

Wire up the CLI command following the existing `analyze-trace` pattern.

- [ ] **Step 1: Create the CLI command**

Create `src/aiperf/cli_commands/anonymize_trace.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for anonymizing conversation traces."""

from __future__ import annotations

from pathlib import Path

from cyclopts import App

app = App(name="anonymize-trace")


@app.default
def anonymize_trace(
    input_file: Path,
    model: str,
    output_file: Path | None = None,
    block_size: int = 512,
) -> None:
    """Anonymize raw chat logs into privacy-preserving Mooncake traces.

    Converts OpenAI-compatible conversation logs into traces with block-hashed
    prefix patterns. Strips all text content, preserving only token counts and
    hash ID sequences for prefix-cache-aware benchmarking.

    The --model argument specifies the TARGET model you intend to benchmark
    against, not the model that generated the original logs. The target model's
    tokenizer and chat template are used to produce accurate token counts and
    prefix patterns.

    Args:
        input_file: Path to input JSONL with raw conversation logs.
        model: HuggingFace model name for tokenizer and chat template (target model).
        output_file: Path to output Mooncake trace JSONL. Defaults to <input>_anonymized.jsonl.
        block_size: Tokens per block for hashing (default: 512).
    """
    from rich.console import Console

    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.synthesis.anonymize import anonymize_trace as _anonymize

    console = Console(width=120)

    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise SystemExit(1)

    if output_file is None:
        output_file = input_file.with_name(f"{input_file.stem}_anonymized.jsonl")

    console.print(f"Loading tokenizer: {model}")
    tokenizer = Tokenizer.from_pretrained(model)

    console.print(f"Processing: {input_file}")
    result = _anonymize(
        input_file=input_file,
        output_file=output_file,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    console.print()
    console.print("[bold]Anonymization Summary[/bold]")
    console.print(f"  Requests processed: {result.total_processed:,}")
    if result.total_skipped > 0:
        console.print(f"  Requests skipped:   {result.total_skipped:,}")
    if result.sessions_detected > 0:
        console.print(f"  Sessions detected:  {result.sessions_detected:,}")
    console.print(f"  Unique hash IDs:    {result.unique_hash_ids:,}")
    console.print(f"  Output file:        {result.output_file}")

    if result.no_timestamps_warning:
        console.print()
        console.print(
            "[yellow]Warning: No timestamps found in input. "
            "The output trace will not support --fixed-schedule replay. "
            "Consider adding timestamps or using --request-rate during replay.[/yellow]"
        )
```

- [ ] **Step 2: Register the command in `cli.py`**

In `src/aiperf/cli.py`, add a new line after the `analyze-trace` registration (line 32):

```python
app.command("aiperf.cli_commands.anonymize_trace:app", name="anonymize-trace")
```

- [ ] **Step 3: Verify the command is discoverable**

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run aiperf --help`
Expected: `anonymize-trace` appears in the command list.

Run: `cd /Users/bhamm/AIPerf_test/aiperf && uv run aiperf anonymize-trace --help`
Expected: Shows help with `--input-file`, `--model`, `--output-file`, `--block-size` arguments and the description about target model.

- [ ] **Step 4: Commit**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add src/aiperf/cli_commands/anonymize_trace.py src/aiperf/cli.py
git commit -s -m "feat: add anonymize-trace CLI command"
```

---

### Task 5: Tutorial documentation

**Files:**
- Create: `docs/tutorials/anonymize-trace.md`
- Modify: `README.md`

- [ ] **Step 1: Write the tutorial**

Create `docs/tutorials/anonymize-trace.md`:

```markdown
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Anonymize Trace
---
# Privacy-Preserving Trace Anonymization

Share realistic LLM workload traces without exposing sensitive prompt content.

## Overview

Production LLM traces are valuable for benchmarking because they capture real-world
patterns: input/output length distributions, request timing, and prefix sharing from
repeated system prompts or multi-turn conversations. However, sharing these traces
directly would expose user data, proprietary prompts, and PII.

`aiperf anonymize-trace` solves this by converting raw chat logs into Mooncake traces
where all text is replaced with block hash IDs. The hash sequences preserve prefix
overlap patterns (enabling KV cache-aware benchmarking) while making it impossible
to recover the original text.

## Preparing Your Input

Create a JSONL file where each line is a conversation record with OpenAI-compatible messages:

### Single-Turn Example

```jsonl
{"timestamp": 0, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "output": "The capital of France is Paris."}
{"timestamp": 100, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain machine learning in simple terms."}], "output": "Machine learning is a type of AI that allows computers to learn from data."}
```

Note how both requests share the same system prompt. The anonymized trace will reflect this shared prefix through matching hash IDs.

### Multi-Turn Example

Use `session_id` to group turns within a conversation:

```jsonl
{"timestamp": 0, "session_id": "user_42", "messages": [{"role": "user", "content": "What is Python?"}], "output": "Python is a programming language."}
{"timestamp": 5000, "session_id": "user_42", "messages": [{"role": "user", "content": "Show me a hello world example"}], "output": "print('Hello, World!')"}
```

Each turn only needs its own new messages. The anonymizer automatically accumulates the full conversation history (including prior assistant responses) when computing hash IDs for later turns.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | array | OpenAI-compatible messages with `role` and `content` |
| `output` | string | Assistant response text (used only for token counting) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | number | Milliseconds since trace start (for `--fixed-schedule` replay) |
| `session_id` | string | Groups turns into multi-turn conversations |

## Choosing Your Target Model

The `--model` argument specifies the model you intend to **benchmark against**, not the model that generated the original logs.

This matters because:

- **Chat template**: Different models use different chat formats (ChatML, Llama, Mistral, etc.). The template tokens are part of what gets cached, so prefix patterns depend on which template is applied.
- **Tokenization**: Token counts and block boundaries vary by tokenizer. A trace anonymized for Llama will have different `input_length` values than one for Mistral.

### Example: Migrating from a Proprietary API

If you have production logs from Claude or GPT-4 and want to evaluate switching to a self-hosted model:

```bash
# Anonymize for benchmarking against Llama 3.1 70B
aiperf anonymize-trace \
  --input-file production_logs.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --output-file llama_trace.jsonl

# Anonymize for benchmarking against Mistral
aiperf anonymize-trace \
  --input-file production_logs.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output-file mistral_trace.jsonl
```

## Running the Command

```bash
aiperf anonymize-trace \
  --input-file raw_logs.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --output-file anonymized_trace.jsonl \
  --block-size 512
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-file` | Yes | | Path to input JSONL |
| `--model` | Yes | | HuggingFace model name (target model) |
| `--output-file` | No | `<input>_anonymized.jsonl` | Output path |
| `--block-size` | No | 512 | Tokens per hash block |

The default `--block-size` of 512 matches common KV cache page sizes. Smaller values increase hash granularity but produce larger `hash_ids` arrays.

## Verifying the Output

Inspect the anonymized trace with `aiperf analyze-trace`:

```bash
aiperf analyze-trace --input-file anonymized_trace.jsonl --block-size 512
```

This shows ISL/OSL distributions, prefix reuse ratios, and theoretical cache hit rates,
letting you verify the trace captures meaningful prefix sharing patterns.

## Replaying the Trace

Use the anonymized trace as a benchmark workload:

```bash
# With timestamps (fixed schedule replay)
aiperf profile \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file anonymized_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule

# Without timestamps (use request rate instead)
aiperf profile \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file anonymized_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --request-rate 10 \
  --concurrency 50
```

## What Gets Shared vs. What Stays Protected

| Shared | Protected |
|--------|-----------|
| Request timestamps | Actual prompt text |
| Input/output token counts | Token IDs |
| Block hash ID sequences | Assistant responses |
| Prefix cache hit patterns | User information |
| Session grouping | Proprietary system prompts |
```

- [ ] **Step 2: Add tutorial to README.md tutorial index**

In `README.md`, under the "Workloads and Data" section (after the "Prefix Synthesis" line around line 153), add:

```markdown
- [Anonymize Trace](docs/tutorials/anonymize-trace.md) - Privacy-preserving trace sharing
```

- [ ] **Step 3: Commit**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add docs/tutorials/anonymize-trace.md README.md
git commit -s -m "docs: add anonymize-trace tutorial and README entry"
```

---

### Task 6: Update synthesis API docs

**Files:**
- Modify: `docs/api/synthesis.md`

- [ ] **Step 1: Add anonymize-trace to synthesis API docs**

Append the following section at the end of `docs/api/synthesis.md`:

```markdown

---

### Functions

#### `anonymize_trace`

Convert raw chat logs into a privacy-preserving Mooncake trace.

```python
from aiperf.dataset.synthesis.anonymize import anonymize_trace

result = anonymize_trace(
    input_file=Path("raw_logs.jsonl"),
    output_file=Path("anonymized.jsonl"),
    tokenizer=tokenizer,
    block_size=512,
)
```

**Parameters:**
- `input_file` (Path): Path to input JSONL with raw conversation logs
- `output_file` (Path): Path to write output Mooncake trace JSONL
- `tokenizer` (Tokenizer): Tokenizer instance for the target model
- `block_size` (int): Tokens per block for hashing (default: 512)

**Returns:** `AnonymizeResult` with fields:
- `total_processed` (int): Records successfully processed
- `total_skipped` (int): Malformed records skipped
- `sessions_detected` (int): Unique multi-turn sessions found
- `unique_hash_ids` (int): Unique hash IDs generated
- `no_timestamps_warning` (bool): Whether input had no timestamps
- `output_file` (Path): Path to the output file

---

#### `RawConversationRecord`

Pydantic model for validating input JSONL records.

**Fields:**
- `messages` (list[dict]): OpenAI-compatible message array (required, non-empty)
- `output` (str): Assistant response text (required)
- `timestamp` (int | float | None): Request timestamp in milliseconds (optional)
- `session_id` (str | None): Session identifier for multi-turn grouping (optional)
```

- [ ] **Step 2: Commit**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add docs/api/synthesis.md
git commit -s -m "docs: add anonymize-trace to synthesis API reference"
```

---

### Task 7: Run pre-commit and full test suite

**Files:** All modified files

- [ ] **Step 1: Run ruff format and lint**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
ruff format . && ruff check --fix .
```

Fix any issues found.

- [ ] **Step 2: Run pre-commit on all files**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
pre-commit run --all-files
```

Fix any issues (copyright headers, codespell, CLI docs regeneration, etc.). This may auto-regenerate `docs/cli-options.md` — stage those changes.

- [ ] **Step 3: Run unit tests**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
uv run pytest tests/unit/ -n auto
```

Expected: All pass, including the new `test_anonymize.py` tests.

- [ ] **Step 4: Commit any fixes**

```bash
cd /Users/bhamm/AIPerf_test/aiperf
git add -A
git commit -s -m "chore: fix lint, formatting, and pre-commit issues"
```
