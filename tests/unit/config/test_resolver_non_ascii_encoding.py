# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-resolver text reads must be UTF-8 regardless of the process locale.

Regression for the bug where ``_count_records_and_sessions`` /
``_check_timing_data`` / ``_collect_dag_session_and_fork_ids`` used a bare
``open(file_path)``. Under a non-UTF-8 default locale (``LANG=C`` /
Windows cp1252) that decodes with ascii, so a dataset file with any
non-ASCII byte either crashes with ``UnicodeDecodeError`` (count path,
which only catches ``OSError``) or silently mojibakes to a wrong result
(timing path, whose ``except (OSError, ValueError)`` swallows the decode
error and returns ``False``).

Rather than mutate the process-global locale, these tests patch ``open`` so
that any text read WITHOUT an explicit ``encoding`` falls back to ascii --
exactly the behavior under ``LANG=C``. The fix passes ``encoding="utf-8"``
explicitly, so the resolver reads succeed here; an unfixed bare ``open``
would fail.
"""

from __future__ import annotations

import builtins

import pytest

from aiperf.config.dataset.resolver import (
    DatasetResolver,
    _collect_dag_session_and_fork_ids,
)
from aiperf.plugin.enums import CustomDatasetType

# A curly quote, an em dash and an accented vowel -- all non-ASCII, so an
# ascii decode of any line containing them raises UnicodeDecodeError.
_NON_ASCII_TEXT = "café — “hello”"


@pytest.fixture()
def ascii_default_locale(monkeypatch):
    """Force text ``open`` with no explicit encoding to use ascii.

    Simulates a ``LANG=C`` process without touching the real locale, so the
    test is deterministic and side-effect free under xdist.
    """
    real_open = builtins.open

    def fake_open(file, mode="r", *args, encoding=None, **kwargs):
        if encoding is None and "b" not in mode:
            encoding = "ascii"
        return real_open(file, mode, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_count_records_reads_non_ascii_under_c_locale(tmp_path, ascii_default_locale):
    path = tmp_path / "single_turn.jsonl"
    path.write_text(
        f'{{"text": "{_NON_ASCII_TEXT}", "session_id": "s1"}}\n', encoding="utf-8"
    )

    records, sessions = DatasetResolver._count_records_and_sessions(
        str(path), CustomDatasetType.SINGLE_TURN
    )
    assert (records, sessions) == (1, 1)


def test_check_timing_data_reads_non_ascii_under_c_locale(
    tmp_path, ascii_default_locale
):
    # Non-ASCII text alongside a real timestamp: the decode must succeed so the
    # timing field is actually seen (pre-fix this silently returned False).
    path = tmp_path / "timed.jsonl"
    path.write_text(
        f'{{"text": "{_NON_ASCII_TEXT}", "timestamp": 123}}\n', encoding="utf-8"
    )

    assert (
        DatasetResolver._check_timing_data(
            str(path), None, CustomDatasetType.SINGLE_TURN
        )
        is True
    )


def test_collect_dag_ids_reads_non_ascii_under_c_locale(tmp_path, ascii_default_locale):
    path = tmp_path / "dag.jsonl"
    path.write_text(
        f'{{"session_id": "root", "turns": [{{"text": "{_NON_ASCII_TEXT}", '
        f'"spawns": ["child"]}}]}}\n',
        encoding="utf-8",
    )

    all_ids, referenced = _collect_dag_session_and_fork_ids(str(path))
    assert all_ids == {"root"}
    assert referenced == {"child"}
