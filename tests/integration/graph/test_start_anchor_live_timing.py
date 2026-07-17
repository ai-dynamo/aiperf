# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIVE proof that start-anchored edges dispatch at parent DISPATCH, not completion.

Drives ACTUAL ``aiperf profile --export-level raw`` runs of a purpose-built weka
trace (``weka_start_anchor.json``) against the in-repo mock server at two server
speeds and compares the EXACT firing timestamps of four nodes, keyed by the
recorded ``out`` value that survives to the wire as ``payload.max_tokens``:

* ``trace_start_anchor:0`` -- the long parent P (``out=200``); ``ttft 500ms +
  200 x itl`` per token.
* ``agent_spawn:0`` -- a mid-flight spawn child C (``out=30``) start-anchored to
  P at +2.5s.
* ``agent_chain:0`` -- a mid-flight overlap child Q (``out=45``) start-anchored
  to P at +5.0s.
* ``trace_start_anchor:1`` -- the END-anchored tail R (``out=60``), fired
  ``1.0s`` after P COMPLETES.

The proof: the two start-anchored children dispatch at a CONSTANT dispatch-to-
dispatch offset (2.5s / 5.0s from P's dispatch) at BOTH server speeds -- even in
the slow run, where P is still in flight (~12.5s) when both children fire -- while
the end-anchored tail tracks P's actual COMPLETION (compresses to ~2.5s when the
server is fast, expands to ~13.5s when slow). Start-anchored offsets are invariant
to server speed; end-anchored offsets are not. That contrast IS the feature.

Two operational levers make the four nodes measurable:

* The t* snapshot window is off by default (``--trajectory-start-min/max-ratio``
  unset => t*=0), so every node dispatches LIVE in profiling. An engaged window
  could snapshot the pre-t* nodes (P, C, Q) into cache-priming warmup HISTORY
  and leave only the tail dispatched.
* ``--extra-inputs ignore_eos:true`` makes the mock emit EXACTLY ``max_tokens``
  output tokens (the mock otherwise samples a variable, prompt-hash-seeded count),
  so P's duration is the intended ``ttft + out x itl`` and the end-anchored tail's
  completion offset is deterministic.

Lives in the INTEGRATION lane (not component): the fidelity report functions
rebuild the trie graph in-process with the real ``gpt2`` tokenizer the subprocess
uses, so the reconstruction must not diverge on a patched FakeTokenizer.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import orjson
import pytest
from pytest import param

from tools.weka_trace_fidelity import (
    causality_timing_vs_real_trace,
    content_vs_real_trace,
)

_REPO = Path(__file__).resolve().parents[3]
_FIX = _REPO / "tests" / "unit" / "graph" / "fixtures" / "weka_start_anchor.json"
_MODEL = "claude-opus-4-5-20251101"

# Record -> node identity key: each node's recorded ``out`` reaches the wire as
# ``payload.max_tokens`` (``dispatch_overrides.max_tokens == out``).
_PARENT_OUT = 200  # trace_start_anchor:0 -- the long parent P
_SPAWN_OUT = 30  # agent_spawn:0 -- spawn child, start-anchored to P at +2.5s
_CHAIN_OUT = 45  # agent_chain:0 -- overlap child, start-anchored to P at +5.0s
_TAIL_OUT = 60  # trace_start_anchor:1 -- END-anchored tail, 1.0s after P ends


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


def _run_profile(tmp_path: Path, *, ttft: int, itl: int, ignore_eos: bool) -> Path:
    """Drive one live ``aiperf profile`` of the start-anchor fixture; return the raw export.

    ``ttft`` / ``itl`` set the mock latency model (ms). ``ignore_eos`` forces the
    mock to emit exactly ``max_tokens`` tokens (needed for a deterministic parent
    duration; irrelevant at zero latency).
    """
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
        # t*=0 => full native replay: EVERY node dispatches live in profiling.
    }

    mock = subprocess.Popen(
        [str(mock_bin), "--port", str(port), "--ttft", str(ttft), "--itl", str(itl)],
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
            "60",
            "--export-level",
            "raw",
            "--artifact-dir",
            str(artifact_dir),
            "--random-seed",
            "42",
        ]
        if ignore_eos:
            cmd += ["--extra-inputs", "ignore_eos:true"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock.kill()

    assert proc.returncode == 0, (
        f"aiperf profile (start-anchor) exited {proc.returncode}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )
    raw = next(artifact_dir.rglob("profile_export_raw.jsonl"), None)
    assert raw is not None, "no profile_export_raw.jsonl produced"
    return raw


def _profiling_starts(raw: Path) -> dict[int, int]:
    """Map ``payload.max_tokens`` -> earliest profiling ``request_start_ns`` (ns).

    ``max_tokens == out`` is the record->node identity key. Full-replay with
    ``--num-conversations 1`` dispatches each node once, so each key maps to one
    record; ``min`` is defensive against any recycle re-fire.
    """
    starts: dict[int, int] = {}
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        meta = obj.get("metadata", {}) or {}
        if meta.get("benchmark_phase") != "profiling":
            continue
        start = meta.get("request_start_ns")
        payload = obj.get("payload", {}) or {}
        # The native LlmNode.max_tokens cap is endpoint-mapped to the wire
        # token field (max_completion_tokens for chat).
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
        if start is None or max_tokens is None:
            continue
        prev = starts.get(max_tokens)
        if prev is None or start < prev:
            starts[max_tokens] = start
    return starts


def _assert_no_streaming_metrics(raw: Path, label: str) -> tuple[str, str]:
    """Regression: an all-``"n"`` graph run surfaces NO fabricated streaming metrics.

    This fixture records only non-streaming (``"n"``) nodes and the run is launched
    with the GLOBAL ``--streaming`` flag OFF. The run-level gate is per-record,
    so graph workloads do not drop STREAMING_ONLY metrics wholesale (a mixed
    graph CAN surface TTFT for its streaming nodes) -- but here ZERO records
    stream, so the per-record predicate must exclude every record and no TTFT may
    be fabricated. Asserts ``time_to_first_token`` is ABSENT and
    ``streamed_request_count`` is absent or exactly 0. Returns
    ``(ttft_state, streamed_count_state)`` strings for the report.
    """
    summary = raw.parent / "profile_export_aiperf.json"
    assert summary.exists(), f"[{label}] no profile_export_aiperf.json beside {raw}"
    data = orjson.loads(summary.read_bytes())

    # Positive control: a valid export always carries request_count, so an empty
    # or malformed export cannot let the absence assertions below pass vacuously.
    assert "request_count" in data, (
        f"[{label}] request_count missing from export; empty/malformed summary "
        f"would make the streaming-absence checks vacuous"
    )

    assert "time_to_first_token" not in data, (
        f"[{label}] time_to_first_token present in an all-'n' (zero-stream) run: "
        f"{data.get('time_to_first_token')}"
    )
    src = data.get("streamed_request_count")
    if src is None:
        streamed_state = "absent"
    else:
        assert src.get("avg") in (0, 0.0), (
            f"[{label}] streamed_request_count={src.get('avg')} in a zero-stream run"
        )
        streamed_state = f"present avg={src.get('avg')}"
    return "absent", streamed_state


@pytest.mark.integration
@pytest.mark.parametrize(
    ("ttft", "itl", "label"),
    [
        param(500, 5, "fast", id="fast"),
        param(500, 60, "slow", id="slow"),
    ],
)  # fmt: skip
def test_start_anchor_live_timing(
    tmp_path: Path, ttft: int, itl: int, label: str
) -> None:
    """Start-anchored children fire at CONSTANT parent-dispatch offsets; the tail tracks completion."""
    raw = _run_profile(tmp_path, ttft=ttft, itl=itl, ignore_eos=True)
    starts = _profiling_starts(raw)
    for out in (_PARENT_OUT, _SPAWN_OUT, _CHAIN_OUT, _TAIL_OUT):
        assert out in starts, (
            f"[{label}] no profiling record for out={out}; got {sorted(starts)}"
        )

    # Regression: relaxed gate must NOT fabricate streaming metrics for an
    # all-"n" (zero-stream) graph run launched without global --streaming.
    ttft_state, streamed_state = _assert_no_streaming_metrics(raw, label)
    print(
        f"[{label}] no-stream metrics: time_to_first_token={ttft_state} "
        f"streamed_request_count={streamed_state}"
    )

    p0 = starts[_PARENT_OUT]
    d_spawn = (starts[_SPAWN_OUT] - p0) / 1e9
    d_chain = (starts[_CHAIN_OUT] - p0) / 1e9
    d_tail = (starts[_TAIL_OUT] - p0) / 1e9
    parent_finish = _parent_finish_offset(raw, p0)
    print(
        f"[{label}] ttft={ttft} itl={itl}: parent_finish={parent_finish:.3f}s "
        f"d_spawn={d_spawn:.3f}s d_chain={d_chain:.3f}s d_tail={d_tail:.3f}s"
    )

    # Start-anchored edges gate off the parent's DISPATCH, so their dispatch-to-
    # dispatch offset is INVARIANT to server speed (identical fast and slow).
    assert abs(d_spawn - 2.5) <= 0.25, f"[{label}] spawn offset {d_spawn:.3f}s != 2.5s"
    assert abs(d_chain - 5.0) <= 0.25, f"[{label}] chain offset {d_chain:.3f}s != 5.0s"

    if label == "slow":
        # Parent (~12.5s) is verifiably IN FLIGHT when both children dispatch.
        assert d_spawn < 12.0 and d_chain < 12.0, (
            f"[{label}] children did not dispatch mid-parent-flight: "
            f"d_spawn={d_spawn:.3f} d_chain={d_chain:.3f}"
        )
        # END-anchored tail = parent_finish (~12.5s) + recorded 1.0s gap.
        assert abs(d_tail - 13.5) <= 1.0, (
            f"[slow] tail at {d_tail:.3f}s, expected ~13.5s"
        )
    else:
        # END-anchored tail COMPRESSES with a fast server: parent_finish(~1.5)+1.0.
        assert abs(d_tail - 2.5) <= 1.0, f"[fast] tail at {d_tail:.3f}s, expected ~2.5s"
        # Content fidelity is latency-independent -- assert it on the fast run.
        # The subprocess ran with ``--tokenizer gpt2``; pass the run's knob.
        content = content_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
        assert content.passed, content.render()


def _parent_finish_offset(raw: Path, p0: int) -> float:
    """Parent P's completion offset (s) from its own dispatch -- for print/context."""
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        meta = obj.get("metadata", {}) or {}
        if meta.get("benchmark_phase") != "profiling":
            continue
        payload = obj.get("payload", {}) or {}
        cap = payload.get("max_tokens") or payload.get("max_completion_tokens")
        if cap == _PARENT_OUT:
            end = meta.get("request_end_ns")
            if end is not None:
                return (end - p0) / 1e9
    return float("nan")


@pytest.mark.integration
def test_start_anchor_content_causality_fidelity(tmp_path: Path) -> None:
    """Reconstruction content + causality both PASS at zero server latency.

    :func:`causality_timing_vs_real_trace` reconstructs each END-anchored edge
    from its predecessor's DISPATCH + the warped edge delay -- valid only when the
    predecessor returns ~instantly (its docstring drives ``--ttft 0 --itl 0``). On
    a latency-bearing run the tail's observed dispatch is parent_FINISH + delay,
    off by the parent's processing time -- which is exactly what
    :func:`test_start_anchor_live_timing` measures DIRECTLY. So the causality
    reconstruction is asserted at zero latency (its design point), over the
    IDENTICAL graph geometry. Content fidelity is latency-independent and is also
    asserted on the fast run above.
    """
    raw = _run_profile(tmp_path, ttft=0, itl=0, ignore_eos=False)
    # The subprocess ran with ``--tokenizer gpt2``; pass the run's knob.
    content = content_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    causality = causality_timing_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    assert content.passed, content.render()
    assert causality.passed, causality.render()
