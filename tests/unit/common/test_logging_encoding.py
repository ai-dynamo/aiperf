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


@pytest.mark.parametrize(
    ("encoding", "substitutes"),
    [
        ("utf-8", False),
        ("UTF8", False),
        ("cp1252", True),
        ("not-a-codec", True),
        (None, True),
    ],
)
def test_formatter_choice_follows_the_encoding(
    encoding: str | None, substitutes: bool
) -> None:
    formatted = _basic_formatter(encoding).format(_record(_TABLE_LIKE))
    assert ("\u2501" in formatted) is not substitutes


@pytest.mark.parametrize(
    ("attr", "expected"),
    [("cp1252", "cp1252"), (None, None), ("", None)],
    ids=["reports-encoding", "reports-none", "reports-empty"],
)
def test_stream_encoding_reports_none_when_the_stream_is_silent(
    attr: str | None, expected: str | None
) -> None:
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


class _SilentAboutEncoding:
    """A wrapper that forwards writes but reports no encoding.

    Stands in for the stream layers that hide the terminal's encoding. The sink
    underneath is still strict cp1252, so guessing UTF-8 here would drop the
    record.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        return self._wrapped.write(text)

    def flush(self) -> None:
        self._wrapped.flush()


def test_a_stream_that_reports_no_encoding_is_not_assumed_to_be_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, strict_cp1252 = _cp1252_stream()
    monkeypatch.setattr(
        "aiperf.common.logging.sys.stdout", _SilentAboutEncoding(strict_cp1252)
    )

    handler = _create_basic_handler(logging.INFO)
    handler.emit(_record(_TABLE_LIKE))
    strict_cp1252.flush()

    written = raw.getvalue().decode("cp1252")
    assert "concurrency" in written
    assert "━" not in written
