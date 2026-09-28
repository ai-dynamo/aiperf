# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The basic log handler must survive a console that cannot encode the message.

Rich renders tables with box-drawing characters. On a cp1252 console, the
Windows default, writing one raises inside ``StreamHandler.emit`` and logging
discards the entire record, so the operator loses the line rather than the
character.
"""

import io
import logging

import pytest

from aiperf.common.logging import (
    _basic_formatter,
    _create_basic_handler,
    _stream_encoding,
)

_HEAVY_RULE = "\u2501" * 8
_TABLE_LIKE = f"\n  concurrency  \n {_HEAVY_RULE} \n  1            \n"


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="p",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def _cp1252_stream() -> tuple[io.BytesIO, io.TextIOWrapper]:
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")


def test_a_strict_console_drops_the_whole_record_without_the_formatter() -> None:
    """Characterises the bug: a plain formatter loses the line, not just the rule."""
    raw, stream = _cp1252_stream()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    with pytest.raises(UnicodeEncodeError):
        stream.write(handler.format(_record(_TABLE_LIKE)))

    stream.flush()
    assert raw.getvalue() == b""


def test_the_line_survives_on_a_cp1252_console() -> None:
    raw, stream = _cp1252_stream()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_basic_formatter("cp1252"))

    handler.emit(_record(_TABLE_LIKE))
    stream.flush()

    written = raw.getvalue().decode("cp1252")
    assert "concurrency" in written
    assert "\u2501" not in written, "the rule cannot be encoded, so it is substituted"
    assert "?" in written


def test_utf8_output_is_left_alone() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", newline="")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_basic_formatter("utf-8"))

    handler.emit(_record(_TABLE_LIKE))
    stream.flush()

    assert _HEAVY_RULE in raw.getvalue().decode("utf-8")


@pytest.mark.parametrize(
    ("encoding", "substitutes"),
    [
        ("utf-8", False),
        ("UTF8", False),
        ("cp1252", True),
        ("ascii", True),
        ("not-a-codec", True),
    ],
)
def test_formatter_choice_follows_the_encoding(
    encoding: str, substitutes: bool
) -> None:
    formatted = _basic_formatter(encoding).format(_record(_TABLE_LIKE))
    assert ("\u2501" in formatted) is not substitutes


@pytest.mark.parametrize(
    ("attr", "expected"),
    [("cp1252", "cp1252"), (None, "utf-8")],
    ids=["reports-encoding", "reports-none"],
)
def test_stream_encoding_defaults_to_utf8(attr: str | None, expected: str) -> None:
    class _Stream:
        encoding = attr

    assert _stream_encoding(_Stream()) == expected


def test_the_basic_handler_wires_the_substituting_formatter_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the wiring, not just the helper.

    ``_basic_formatter`` being correct is no use if ``_create_basic_handler``
    does not reach for it, and that is the seam a refactor would break.
    """
    raw, stream = _cp1252_stream()
    monkeypatch.setattr("aiperf.common.logging.sys.stdout", stream)

    handler = _create_basic_handler(logging.INFO)
    handler.emit(_record(_TABLE_LIKE))
    stream.flush()

    written = raw.getvalue().decode("cp1252")
    assert "concurrency" in written
    assert "\u2501" not in written
