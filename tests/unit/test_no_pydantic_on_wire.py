# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforces the post-P3 invariant: no Pydantic on the ZMQ wire.

These grep-style assertions lock in the terminal state of the msgspec
migration so future changes can't silently reintroduce a Pydantic
serialization path on inter-service messages.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MESSAGES_DIR = REPO_ROOT / "src" / "aiperf" / "common" / "messages"
MESSAGE_CODECS = REPO_ROOT / "src" / "aiperf" / "common" / "message_codecs.py"


def _iter_py_files(root: Path):
    return sorted(root.rglob("*.py"))


def test_no_pydantic_imports_in_common_messages():
    """No file under src/aiperf/common/messages/ may import from pydantic."""
    offenders: list[str] = []
    pattern = re.compile(r"^\s*(from pydantic\b|import pydantic\b)", re.MULTILINE)
    for path in _iter_py_files(MESSAGES_DIR):
        text = path.read_text()
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "Pydantic imports found in message modules (post-P3 this path is "
        f"msgspec-only): {offenders}"
    )


def test_message_codecs_has_no_pydantic_import():
    """message_codecs.py may not import pydantic or BaseModel."""
    text = MESSAGE_CODECS.read_text()
    assert "from pydantic" not in text, "message_codecs.py still imports pydantic"
    assert "BaseModel" not in text, "message_codecs.py still references BaseModel"


def test_no_model_dump_or_model_validate_in_messages():
    """model_dump / model_validate must not appear in message modules.

    Pydantic leftovers would indicate a codec fallback or shim creeping back
    in. ``model_dump`` and ``model_dump_json`` shim methods on Message itself
    are allowed — they are msgspec-backed. Grep only flags CALLS: tokens
    followed by an open-paren.
    """
    call_pattern = re.compile(r"\.(?:model_dump|model_validate)(?:_json)?\s*\(")
    offenders: list[tuple[str, int]] = []
    allowed_files = {MESSAGES_DIR / "base_messages.py"}  # defines the shims
    for path in _iter_py_files(MESSAGES_DIR):
        if path in allowed_files:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if call_pattern.search(line):
                offenders.append((str(path.relative_to(REPO_ROOT)), lineno))
    assert not offenders, (
        f"Pydantic model_dump/model_validate calls found in message modules: {offenders}"
    )


def test_json_message_codec_class_is_deleted():
    """The transitional JsonMessageCodec class is gone after P3."""
    text = MESSAGE_CODECS.read_text()
    assert "class JsonMessageCodec" not in text
    assert "JSON_MESSAGE_CODEC" not in text
    assert "class PydanticMsgpackCodec" not in text
