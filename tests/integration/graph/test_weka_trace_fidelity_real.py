# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REAL empirical proof: a live trie ``--export-level raw`` run matches the recorded trace.

This is the Task-7 acceptance gate's real-run half. It drives an ACTUAL
``aiperf profile --export-level raw`` of the weka segment-trie IR path (a
subprocess run against the in-repo mock server) over the
subagent fixture, then runs the offline fidelity reports
(:mod:`tools.weka_trace_fidelity`) on the produced ``profile_export_raw.jsonl`` and
asserts BOTH:

* :func:`content_vs_real_trace` (criterion 2) -- each dispatched record's
  reconstructed prompt equals what the recorded trace prescribes, and
* :func:`causality_timing_vs_real_trace` (criterion 3) -- the dispatch causal order
  and relative inter-request timing match the recorded trace.

It lives in the INTEGRATION lane (not component) on purpose: the report functions
rebuild the trie graph in-process via ``build_trie_graph``, so the in-process
tokenizer must be the SAME real ``gpt2`` the subprocess uses. The
component-integration package patches ``Tokenizer.from_pretrained`` to a
FakeTokenizer (which the subprocess would NOT see), so the reconstruction would
diverge there; the integration package uses the real tokenizer end-to-end.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tools.weka_trace_fidelity import (
    causality_timing_vs_real_trace,
    content_vs_real_trace,
    load_raw_export,
)

_REPO = Path(__file__).resolve().parents[3]
_FIX = _REPO / "tests" / "unit" / "graph" / "fixtures" / "weka_subagent.json"
_MODEL = "claude-opus-4-5-20251101"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


@pytest.mark.integration
def test_real_trie_raw_export_matches_real_trace(tmp_path: Path) -> None:
    """A live trie raw-export run PASSES content + causality/timing vs the real trace."""
    venv = _REPO / ".venv" / "bin"
    mock_bin = venv / "aiperf-mock-server"
    aiperf_bin = venv / "aiperf"
    if not mock_bin.exists() or not aiperf_bin.exists():
        pytest.skip("aiperf / aiperf-mock-server not installed in .venv")

    port = _free_port()
    artifact_dir = tmp_path / "artifacts"
    env = {
        **os.environ,
        "NO_PROXY": "127.0.0.1,localhost",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }

    mock = subprocess.Popen(
        [str(mock_bin), "--port", str(port), "--ttft", "0", "--itl", "0"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_port(port), "mock server did not come up"
        cmd = [
            str(aiperf_bin),
            "profile",
            "--input-file",
            str(_FIX),
            "--url",
            f"http://127.0.0.1:{port}",
            "--endpoint-type",
            "chat",
            "--model",
            _MODEL,
            "--tokenizer",
            "gpt2",
            "--num-conversations",
            "1",
            "--concurrency",
            "4",
            "--benchmark-duration",
            "30",
            "--export-level",
            "raw",
            "--artifact-dir",
            str(artifact_dir),
            "--random-seed",
            "42",
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock.kill()

    assert proc.returncode == 0, (
        f"aiperf profile (trie path) exited {proc.returncode}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )

    raw = next(artifact_dir.rglob("profile_export_raw.jsonl"), None)
    assert raw is not None, "no profile_export_raw.jsonl produced"
    records = load_raw_export(raw)
    profiling = [r for r in records if r.phase == "profiling"]
    assert profiling, "no profiling records dispatched on the trie path"

    # The subprocess ran with ``--tokenizer gpt2``; the tool defaults to the
    # bare live-run builtin tokenizer, so the run's knob is passed explicitly.
    content = content_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    causality = causality_timing_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    assert content.passed, content.render()
    assert causality.passed, causality.render()
