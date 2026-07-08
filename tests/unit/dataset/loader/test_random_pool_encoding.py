# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RandomPool file reads must be UTF-8 regardless of the process locale.

Regression for a branch wholesale-port that dropped ``encoding="utf-8"`` from
``RandomPoolDatasetLoader._load_dataset_from_file``'s ``open(file_path)``
(origin/main has it). Under a non-UTF-8 default locale (``LANG=C`` /
Windows cp1252 / ``PYTHONUTF8=0``) a bare ``open`` decodes with ascii, so any
pool entry with a non-ASCII byte crashes the loader with ``UnicodeDecodeError``.

Rather than mutate the process-global locale, ``ascii_default_locale`` patches
``open`` so any text read WITHOUT an explicit ``encoding`` falls back to ascii
-- exactly the behavior under ``LANG=C``. The fix passes ``encoding="utf-8"``
explicitly, so these reads succeed; an unfixed bare ``open`` would fail.
"""

from __future__ import annotations

import builtins

import pytest

from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader

# Accented vowels + CJK: an ascii decode of any line containing them raises
# UnicodeDecodeError.
_NON_ASCII_TEXT = "café résumé 你好"


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


def test_random_pool_reads_non_ascii_under_c_locale(
    tmp_path, ascii_default_locale, default_user_run
):
    """Loading a pool file with non-ASCII text must not crash under LANG=C."""
    path = tmp_path / "pool.jsonl"
    path.write_text(f'{{"text": "{_NON_ASCII_TEXT}"}}\n', encoding="utf-8")

    loader = RandomPoolDatasetLoader(filename=str(path), run=default_user_run)
    dataset = loader.load_dataset()

    pool = dataset[path.name]
    assert len(pool) == 1
    assert pool[0].text == _NON_ASCII_TEXT


def test_random_pool_non_ascii_round_trips(tmp_path, default_user_run):
    """Non-ASCII pool text is preserved exactly (byte-for-byte) on load."""
    path = tmp_path / "pool.jsonl"
    path.write_text(
        f'{{"text": "{_NON_ASCII_TEXT}"}}\n{{"text": "emoji 🚀 ok"}}\n',
        encoding="utf-8",
    )

    loader = RandomPoolDatasetLoader(filename=str(path), run=default_user_run)
    pool = loader.load_dataset()[path.name]

    assert [entry.text for entry in pool] == [_NON_ASCII_TEXT, "emoji 🚀 ok"]
