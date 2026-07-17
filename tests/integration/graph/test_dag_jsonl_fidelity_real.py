# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REAL empirical proof: legacy vs graph dag_jsonl raw exports are wire-parity-equal.

This is the Task-7 acceptance gate's real-run half (the EXTERNAL proof). The
in-process wire-parity golden gate
(``tests/component_integration/graph/test_dag_jsonl_byte_parity.py``) proves the
graph adapter emits payload-identical wire bodies (canonical order-insensitive
comparison) to the legacy path at the transport seam; the hermetic half of this gate
(``tests/component_integration/graph/test_dag_jsonl_fidelity.py``) drives
:func:`tools.dag_jsonl_fidelity.prove_parity` on hand-crafted synthetic export
fixtures. This file runs REAL ``aiperf profile --export-level raw`` subprocesses
and diffs the produced raw exports.

It starts ONE shared ``aiperf-mock-server`` process (its per-prompt response
generation is deterministic WITHIN a process -- ``aiperf_mock_server/tokens.py``
seeds off ``hash(...)`` -- so BOTH runs must hit the SAME server for fork-child
prompts, which embed the parent's live reply, to agree), then runs legacy
(``--custom-dataset-type dag_jsonl``) and graph (``--graph-format dag_jsonl``)
against it with identical ``--model``/``--url``/``--random-seed 42`` and
STREAMING ON (the first streaming parity exercise -- SSE responses flow through
``build_assistant_turn`` on both planes), a single worker, single pass.
``prove_parity`` on the two ``profile_export_raw.jsonl`` files must PASS every
criterion.

It lives in the INTEGRATION lane (not component) on purpose: it spawns real
subprocesses against a live mock server, so the component-integration package's
in-process patches (e.g. the FakeTokenizer) do NOT cross the subprocess
boundary. Mirrors ``tests/integration/graph/test_weka_trace_fidelity_real.py``
(server-startup helpers, ``.venv/bin`` skip guard, NO_PROXY/HF_HUB_OFFLINE env,
300s subprocess timeout).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
from pytest import param

from tools.dag_jsonl_fidelity import prove_parity

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO / "tests" / "fixtures" / "dag" / "graph_parity"
_MODEL = "test-model"


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


def _run_profile(
    aiperf_bin: Path,
    *,
    fixture: Path,
    url: str,
    format_flag: list[str],
    artifact_dir: Path,
    env: dict[str, str],
    extra: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run one single-pass, streaming ``aiperf profile`` pass with warmup off.

    Warmup is OFF because no ``--warmup-*`` trigger flag is passed: a warmup
    phase is only built when ``--warmup-request-count`` / ``--warmup-num-sessions``
    / ``--warmup-duration`` is explicitly set (see
    ``src/aiperf/config/flags/_converter_warmup.py``), and those fields are all
    ``gt=0`` so there is no "zero to disable" value -- omitting the trigger IS
    the disable. ``prove_parity`` additionally filters to the PROFILING phase,
    so even a stray warmup row on either plane cannot inflate the comparison.
    """
    cmd = [
        str(aiperf_bin),
        "profile",
        "--input-file",
        str(fixture),
        "--url",
        url,
        "--endpoint-type",
        "chat",
        "--model",
        _MODEL,
        "--tokenizer",
        "gpt2",
        *format_flag,
        "--num-conversations",
        "1",
        "--workers-max",
        "1",
        "--record-processor-service-count",
        "1",
        "--streaming",
        "--export-level",
        "raw",
        "--artifact-dir",
        str(artifact_dir),
        "--random-seed",
        "42",
        "--ui",
        "simple",
        *extra,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_name",
    [
        param("fork_minimal.dag.jsonl", id="fork-minimal"),
        param("mixed_full.dag.jsonl", id="mixed-full"),
    ],
)  # fmt: skip
def test_dag_jsonl_graph_raw_export_matches_legacy(
    tmp_path: Path, fixture_name: str
) -> None:
    """Legacy vs graph dag_jsonl raw exports are byte-parity-equal (streaming ON).

    ONE shared mock server answers both runs so the deterministic per-prompt
    replies (and thus every FORK/spawn child prompt that embeds a parent reply)
    are identical across the two planes.
    """
    venv = _REPO / ".venv" / "bin"
    mock_bin = venv / "aiperf-mock-server"
    aiperf_bin = venv / "aiperf"
    if not mock_bin.exists() or not aiperf_bin.exists():
        pytest.skip("aiperf / aiperf-mock-server not installed in .venv")

    fixture = _FIXTURE_DIR / fixture_name
    assert fixture.is_file(), f"missing fixture {fixture}"

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "NO_PROXY": "127.0.0.1,localhost",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    legacy_dir = tmp_path / "legacy"
    graph_dir = tmp_path / "graph"

    mock = subprocess.Popen(
        [str(mock_bin), "--port", str(port), "--ttft", "0", "--itl", "0"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_port(port), "mock server did not come up"
        legacy_proc = _run_profile(
            aiperf_bin,
            fixture=fixture,
            url=url,
            format_flag=["--custom-dataset-type", "dag_jsonl"],
            artifact_dir=legacy_dir,
            env=env,
            # Legacy fork fanout dispatches children as separate sessions, so it
            # needs >=2 session slots or a parent and its child serialize and the
            # fork seeding stalls. The graph run below omits it because its replay
            # lanes self-schedule fork/spawn children off the parent's completion.
            extra=["--concurrency", "2"],
        )
        assert legacy_proc.returncode == 0, (
            f"legacy profile exited {legacy_proc.returncode}\n"
            f"STDERR tail:\n{legacy_proc.stderr[-3000:]}"
        )
        graph_proc = _run_profile(
            aiperf_bin,
            fixture=fixture,
            url=url,
            format_flag=["--graph-format", "dag_jsonl"],
            artifact_dir=graph_dir,
            env=env,
            extra=[],
        )
        assert graph_proc.returncode == 0, (
            f"graph profile exited {graph_proc.returncode}\n"
            f"STDERR tail:\n{graph_proc.stderr[-3000:]}"
        )
    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock.kill()

    legacy_raw = next(legacy_dir.rglob("profile_export_raw.jsonl"), None)
    graph_raw = next(graph_dir.rglob("profile_export_raw.jsonl"), None)
    assert legacy_raw is not None, "legacy run produced no profile_export_raw.jsonl"
    assert graph_raw is not None, "graph run produced no profile_export_raw.jsonl"

    report = prove_parity(legacy_raw, graph_raw)
    assert report.passed, report.render()
    # Non-vacuous: at least one request was actually compared on both planes.
    assert report.messages.checked >= 1, report.render()
