# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIVE proof that a post-TTFT overlapped child gates on the parent's OBSERVED
first token, while a pre-TTFT sibling stays dispatch-anchored.

Drives ACTUAL ``aiperf profile --export-level raw`` runs of a purpose-built weka
trace (``weka_first_token_anchor.json``) whose STREAMING parent (``type: "s"``,
``out=200``) overlaps two async children, keyed by the recorded ``out`` value
that survives to the wire as ``payload.max_tokens``:

* ``trace_first_token_anchor:0`` -- the long STREAMING parent P (``out=200``);
  ``ttft + 200 x itl``.
* ``agent_pre:0``  -- the PRE-TTFT child (``out=45``), recorded at ``t=1.0``
  (before P's recorded first token at ``ttft=2.0s``): a pure DISPATCH anchor,
  D=1.0s.
* ``agent_post:0`` -- the POST-TTFT child (``out=30``), recorded at ``t=4.0``
  (at/after P's recorded first token): a first-token-refined anchor,
  D'=D-ttft=2.0s.
* ``trace_first_token_anchor:1`` -- the END-anchored tail R (``out=60``), fired
  after P COMPLETES.

The proof: two runs identical EXCEPT ``--ttft`` (1000ms vs 5000ms) move the
post-TTFT child by EXACTLY the controlled 4.0s TTFT delta -- because it gates at
the parent's OBSERVED first token (``~ttft``) + D' -- while the pre-TTFT control
holds at ~1.0s because it gates at the parent's DISPATCH + D, invariant to server
speed. That the delta transfers to the post-TTFT child and NOT to the pre-TTFT
child IS the feature. Both children fire while the parent is verifiably in flight
(``ttft + 200 x 60ms`` = 13s low / 17s high).

Two operational levers make the nodes measurable (identical to
``test_start_anchor_live_timing``):

* The t* snapshot window is off by default (``--trajectory-start-min/max-ratio``
  unset => t*=0), so every node dispatches LIVE in profiling (an engaged window
  could snapshot the pre-t* nodes into warmup HISTORY).
* ``--extra-inputs ignore_eos:true`` makes the mock emit EXACTLY ``max_tokens``
  output tokens so P's duration is the intended ``ttft + out x itl``.

The parent is STREAMING so the worker emits a ``FirstToken`` at the mock's first
SSE chunk (``~ttft``): the graph first-token observer stamps P's observed first
token and releases the post-TTFT child at that wall + D'. A pre-TTFT child never
subscribes to the first-token latch; it rides the dispatch anchor.

Lives in the INTEGRATION lane: the structural precondition rebuilds the trie
graph in-process, and the fidelity of the live run depends on the real ``gpt2``
tokenizer the subprocess uses.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)

_REPO = Path(__file__).resolve().parents[3]
_FIX = _REPO / "tests" / "unit" / "graph" / "fixtures" / "weka_first_token_anchor.json"
_MODEL = "claude-opus-4-5-20251101"

# Record -> node identity key: each node's recorded ``out`` reaches the wire as
# ``payload.max_tokens`` (``dispatch_overrides.max_tokens == out``).
_PARENT_OUT = 200  # trace_first_token_anchor:0 -- the long STREAMING parent P
_PRE_OUT = 45  # agent_pre:0  -- PRE-TTFT child, pure dispatch anchor (D=1.0s)
_POST_OUT = 30  # agent_post:0 -- POST-TTFT child, first-token anchor (D'=2.0s)
_TAIL_OUT = 60  # trace_first_token_anchor:1 -- END-anchored tail, after P completes

# D' (``delay_after_predecessor_first_token_us``) for the post-TTFT child, in
# seconds: gate = observed_first_token + D'. Recorded start 4.0s - ttft 2.0s.
_D_PRIME_S = 2.0
# D (``delay_after_predecessor_start_us``) for the pre-TTFT child, in seconds:
# gate = parent_dispatch + D, invariant to server speed.
_D_PRE_S = 1.0
_TOL_S = 0.25


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
    tmp_path: Path, *, ttft: int, itl: int, streaming: bool = True
) -> Path:
    """Drive one live ``aiperf profile`` of the first-token fixture; return the raw export.

    ``ttft`` / ``itl`` set the mock latency model (ms); ``ignore_eos`` is always
    forced so the streaming parent emits exactly ``max_tokens`` output tokens and
    its duration is the intended ``ttft + out x itl``.

    ``streaming`` toggles the GLOBAL ``--streaming`` run flag. With the per-node
    wire override the recorded ``"s"`` parent streams from its own
    ``dispatch_overrides.stream`` even when the global flag is OFF (recorded mode
    wins for graph credits), so the ``streaming=False`` run proves the parent
    still emits a mid-flight ``FirstToken`` without the run itself being
    streaming, while the recorded ``"n"`` children stay non-streaming in BOTH
    modes.
    """
    venv = _REPO / ".venv" / "bin"
    mock_bin = venv / "aiperf-mock-server"
    aiperf_bin = venv / "aiperf"
    if not mock_bin.exists() or not aiperf_bin.exists():
        pytest.skip("aiperf / aiperf-mock-server not installed in .venv")

    port = _free_port()
    tag = "stream" if streaming else "nostream"
    artifact_dir = tmp_path / f"artifacts_ttft{ttft}_{tag}"
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
            "--extra-inputs",
            "ignore_eos:true",
        ]
        if streaming:
            # The GLOBAL ``--streaming`` flag is now only a FALLBACK: the recorded
            # per-node ``dispatch_overrides.stream`` (``apply_run_level_payload_options``)
            # stamps the wire-body ``stream`` per credit and the transport reads
            # SSE vs JSON per request. The ``streaming=False`` run below proves the
            # recorded ``"s"`` parent streams (and emits a mid-flight ``FirstToken``)
            # WITHOUT this flag; here it exercises the mixed global-on case where
            # the recorded ``"n"`` children must STILL stay non-streaming.
            cmd.append("--streaming")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock.kill()

    assert proc.returncode == 0, (
        f"aiperf profile (first-token, ttft={ttft}) exited {proc.returncode}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )
    raw = next(artifact_dir.rglob("profile_export_raw.jsonl"), None)
    assert raw is not None, "no profile_export_raw.jsonl produced"
    return raw


def _profiling_starts(raw: Path) -> dict[int, int]:
    """Map ``payload.max_tokens`` -> earliest profiling ``request_start_ns`` (ns).

    ``max_tokens == out`` is the record->node identity key. Full-replay with
    ``--num-conversations 1`` dispatches each node once; ``min`` is defensive
    against any recycle re-fire.
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


def _parent_finish_offset(raw: Path, p0: int) -> float:
    """Parent P's completion offset (s) from its own dispatch -- flight-window proof."""
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


def _offsets(raw: Path) -> tuple[float, float, float]:
    """Extract ``(d_pre, d_post, parent_finish)`` (all seconds from P's dispatch)."""
    starts = _profiling_starts(raw)
    for out in (_PARENT_OUT, _PRE_OUT, _POST_OUT, _TAIL_OUT):
        assert out in starts, f"no profiling record for out={out}; got {sorted(starts)}"
    p0 = starts[_PARENT_OUT]
    d_pre = (starts[_PRE_OUT] - p0) / 1e9
    d_post = (starts[_POST_OUT] - p0) / 1e9
    parent_finish = _parent_finish_offset(raw, p0)
    return d_pre, d_post, parent_finish


def _response_counts(raw: Path) -> dict[int, int]:
    """Map ``payload.max_tokens`` -> wire ``responses`` count of its earliest record.

    A STREAMING request yields MANY ``responses`` entries (one ``SSEMessage`` per
    SSE chunk + terminators); a non-streaming request yields exactly ONE JSON
    body. Keyed by the earliest-start profiling record per node so it lines up
    with :func:`_profiling_starts`.
    """
    best: dict[int, tuple[int, int]] = {}
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
        n_resp = len(obj.get("responses") or [])
        prev = best.get(max_tokens)
        if prev is None or start < prev[0]:
            best[max_tokens] = (start, n_resp)
    return {mt: n for mt, (_start, n) in best.items()}


def _assert_mixed_mode_wire(raw: Path, label: str) -> tuple[int, int, int, int]:
    """Recorded per-node stream mode reached the wire; return the response counts.

    The recorded STREAMING parent P (``out=200``) must have MULTIPLE ``responses``
    (SSE chunks); each recorded ``"n"`` child (``out=45/30/60``) must have EXACTLY
    ONE (a single JSON body) -- even in a global ``--streaming`` run, the per-node
    override keeps them non-streaming. Returns
    ``(parent, pre, post, tail)`` counts for the report.
    """
    counts = _response_counts(raw)
    for out in (_PARENT_OUT, _PRE_OUT, _POST_OUT, _TAIL_OUT):
        assert out in counts, (
            f"[{label}] no response count for out={out}; got {sorted(counts)}"
        )
    assert counts[_PARENT_OUT] > 1, (
        f"[{label}] recorded streaming parent (out={_PARENT_OUT}) had "
        f"{counts[_PARENT_OUT]} wire responses, expected MULTIPLE SSE chunks"
    )
    for out in (_PRE_OUT, _POST_OUT, _TAIL_OUT):
        assert counts[out] == 1, (
            f"[{label}] recorded 'n' child (out={out}) had {counts[out]} wire "
            "responses, expected EXACTLY ONE non-streaming JSON body"
        )
    return counts[_PARENT_OUT], counts[_PRE_OUT], counts[_POST_OUT], counts[_TAIL_OUT]


def _summary_metrics(raw: Path) -> dict:
    """Parse the JSON summary export (``profile_export_aiperf.json``) beside the raw JSONL.

    The metrics JSON exporter (``exporters/metrics_json_exporter.py``) writes one
    top-level key per metric ``tag``; each value is a ``JsonMetricResult`` dict
    whose aggregate-counter value lives under ``avg`` and whose TTFT ``avg`` is in
    the display unit (ms). Both exports write to the SAME artifacts dir, so the
    summary sits beside the raw JSONL.
    """
    summary = raw.parent / "profile_export_aiperf.json"
    assert summary.exists(), f"no profile_export_aiperf.json beside {raw}"
    return orjson.loads(summary.read_bytes())


def _parent_dispatch_count(raw: Path) -> int:
    """Count profiling records whose ``max_tokens == _PARENT_OUT`` (one per parent dispatch).

    The recorded ``"s"`` parent (``out=200``) is the ONLY streaming node, so its
    dispatch count is the ground-truth streamed-request denominator; with
    ``--num-conversations 1`` this is the number of trace instances.
    """
    n = 0
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
            n += 1
    return n


def _assert_streaming_metrics(
    raw: Path, label: str, ttft_s: float
) -> tuple[int, int, float]:
    """Prove the per-record streaming metrics from the JSON summary export.

    * ``streamed_request_count`` == the parent-dispatch count (only the recorded
      ``"s"`` parent streams; the three recorded ``"n"`` children do not);
    * ``request_count`` == 4x that (parent + 3 children per trace instance);
    * TTFT avg within ``_TOL_S`` of the CONTROLLED ``--ttft`` -- the numerical
      proof that the non-streamed children (whose FULL latencies run ~2.3-13s) are
      EXCLUDED from the TTFT distribution. A broken per-record guard would fold a
      child's single completion timestamp in as its "first token" and drag the
      TTFT mean far above the parent's ~``ttft``.

    Returns ``(streamed_request_count, request_count, ttft_avg_s)`` for the report.
    """
    summary = _summary_metrics(raw)
    n_parent = _parent_dispatch_count(raw)
    assert n_parent >= 1, f"[{label}] no parent dispatch in raw export"

    src = summary.get("streamed_request_count")
    assert src is not None, f"[{label}] streamed_request_count absent from summary"
    assert src.get("avg") == n_parent, (
        f"[{label}] streamed_request_count={src.get('avg')} != "
        f"{n_parent} parent dispatch(es)"
    )

    rc = summary.get("request_count")
    assert rc is not None, f"[{label}] request_count absent from summary"
    assert rc.get("avg") == 4 * n_parent, (
        f"[{label}] request_count={rc.get('avg')} != 4x parent dispatches "
        f"({4 * n_parent})"
    )

    ttft = summary.get("time_to_first_token")
    assert ttft is not None, (
        f"[{label}] time_to_first_token absent -- the streaming parent must "
        "produce a TTFT under the relaxed per-record gate"
    )
    ttft_avg_ms = ttft.get("avg")
    assert ttft_avg_ms is not None, f"[{label}] time_to_first_token.avg is null"
    ttft_avg_s = ttft_avg_ms / 1000.0
    assert abs(ttft_avg_s - ttft_s) <= _TOL_S, (
        f"[{label}] TTFT avg {ttft_avg_s:.3f}s not within {_TOL_S}s of controlled "
        f"--ttft {ttft_s:.3f}s -- non-streamed children (~2.3-13s full latencies) "
        "must be EXCLUDED; pollution here means the per-record guard is broken"
    )
    return int(src["avg"]), int(rc["avg"]), ttft_avg_s


# --- structural precondition (cheap; before any live run) ------------------


_BLOCK_SIZE = 64


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + _BLOCK_SIZE))
    return out


def _stub_partial_tail_tokens(n_tokens: int, seed: str) -> list[int]:
    base = sum(ord(c) for c in seed) * 1000
    return list(range(base, base + n_tokens))


def _stub_decode_tokens_to_text(tokens: list[int]) -> str:
    return "|".join(str(t) for t in tokens)


_STUB_CALLBACKS = ReconCallbacks(
    decode_block_tokens=_stub_decode_block_tokens,
    sample_partial_tail_tokens=_stub_partial_tail_tokens,
    decode_tokens_to_text=_stub_decode_tokens_to_text,
)


@pytest.mark.integration
def test_first_token_fixture_structural_preconditions() -> None:
    """The trie build lowers the fixture to the exact post-TTFT edge geometry.

    Guards the live proof: if the fixture ever stops producing a pre-TTFT pure
    dispatch anchor + a post-TTFT first-token anchor + an end-anchored tail, the
    live timings below would be measuring the wrong thing.
    """
    parsed, _pool = build_trie_graph(
        WekaTrace.model_validate(orjson.loads(_FIX.read_bytes())),
        callbacks=_STUB_CALLBACKS,
    )
    incoming: dict[str, list] = defaultdict(list)
    for edge in parsed.graph.edges:
        incoming[edge.target].append(edge)

    parent = "trace_first_token_anchor:0"

    # parent -> agent_pre:0: pre-TTFT (recorded start 1.0 < ttft 2.0) => pure
    # dispatch anchor, D=1.0s, no first-token refinement.
    (pre,) = incoming["agent_pre:0"]
    assert pre.source == parent
    assert pre.delay_after_predecessor_start_us == pytest.approx(1.0e6)
    assert pre.delay_after_predecessor_first_token_us is None

    # parent -> agent_post:0: post-TTFT (recorded start 4.0 >= ttft 2.0) =>
    # first-token anchor, D=4.0s with D'=D-ttft=2.0s.
    (post,) = incoming["agent_post:0"]
    assert post.source == parent
    assert post.delay_after_predecessor_start_us == pytest.approx(4.0e6)
    assert post.delay_after_predecessor_first_token_us == pytest.approx(2.0e6)

    # parent -> tail: END-anchored tail, from the parent ALONE (async children
    # excluded from the AND-join), completion + recorded 1.5s gap.
    (tail,) = incoming["trace_first_token_anchor:1"]
    assert tail.source == parent
    assert tail.delay_after_predecessor_us == pytest.approx(1.5e6)
    assert tail.delay_after_predecessor_start_us is None
    assert tail.delay_after_predecessor_first_token_us is None

    # The parent must be STREAMING so the worker emits a FirstToken (the store
    # builder derives the wire ``stream`` flag from the native field).
    assert parsed.graph.nodes[parent].streaming is True


# --- live proof ------------------------------------------------------------


@pytest.mark.integration
def test_first_token_live_timing(tmp_path: Path) -> None:
    """The post-TTFT child tracks the OBSERVED first token; the pre-TTFT holds.

    Two runs identical except ``--ttft`` (1000ms vs 5000ms): the controlled 4.0s
    TTFT delta transfers EXACTLY to the post-TTFT child (observed_first_token +
    D'), while the pre-TTFT dispatch-anchored control stays at ~1.0s.
    """
    raw_low = _run_profile(tmp_path, ttft=1000, itl=60)
    d_pre_low, d_post_low, finish_low = _offsets(raw_low)
    n_parent_low, n_pre_low, n_post_low, n_tail_low = _assert_mixed_mode_wire(
        raw_low, "low"
    )
    src_low, rc_low, ttft_avg_low = _assert_streaming_metrics(raw_low, "low", 1.0)
    print(
        f"[low ] ttft=1000 itl=60: parent_finish={finish_low:.3f}s "
        f"d_pre={d_pre_low:.3f}s d_post={d_post_low:.3f}s "
        f"resp(parent={n_parent_low} pre={n_pre_low} post={n_post_low} "
        f"tail={n_tail_low}) "
        f"streamed_request_count={src_low} request_count={rc_low} "
        f"ttft_avg={ttft_avg_low:.3f}s"
    )

    raw_high = _run_profile(tmp_path, ttft=5000, itl=60)
    d_pre_high, d_post_high, finish_high = _offsets(raw_high)
    n_parent_high, n_pre_high, n_post_high, n_tail_high = _assert_mixed_mode_wire(
        raw_high, "high"
    )
    src_high, rc_high, ttft_avg_high = _assert_streaming_metrics(raw_high, "high", 5.0)
    print(
        f"[high] ttft=5000 itl=60: parent_finish={finish_high:.3f}s "
        f"d_pre={d_pre_high:.3f}s d_post={d_post_high:.3f}s "
        f"resp(parent={n_parent_high} pre={n_pre_high} post={n_post_high} "
        f"tail={n_tail_high}) "
        f"streamed_request_count={src_high} request_count={rc_high} "
        f"ttft_avg={ttft_avg_high:.3f}s"
    )

    for label, ttft_s, d_pre, d_post, finish in (
        ("low", 1.0, d_pre_low, d_post_low, finish_low),
        ("high", 5.0, d_pre_high, d_post_high, finish_high),
    ):
        # Post-TTFT child = OBSERVED first token (~ttft) + D' (2.0s).
        assert abs(d_post - (ttft_s + _D_PRIME_S)) <= _TOL_S, (
            f"[{label}] post-TTFT child at {d_post:.3f}s, "
            f"expected {ttft_s + _D_PRIME_S:.3f}s (first token + D')"
        )
        # Pre-TTFT child = dispatch anchor, INVARIANT to server speed.
        assert abs(d_pre - _D_PRE_S) <= _TOL_S, (
            f"[{label}] pre-TTFT child at {d_pre:.3f}s, expected {_D_PRE_S:.3f}s "
            "(dispatch anchor must NOT move with --ttft)"
        )
        # Both children fired while the parent was verifiably IN FLIGHT.
        assert d_post < finish and d_pre < finish, (
            f"[{label}] children did not fire mid-parent-flight: "
            f"d_pre={d_pre:.3f} d_post={d_post:.3f} parent_finish={finish:.3f}"
        )

    # The controlled +4.0s TTFT delta transfers EXACTLY to the post-TTFT child.
    assert abs((d_post_high - d_post_low) - 4.0) <= _TOL_S, (
        f"post-TTFT delta {d_post_high - d_post_low:.3f}s != 4.0s "
        f"(low={d_post_low:.3f} high={d_post_high:.3f})"
    )


@pytest.mark.integration
def test_first_token_live_timing_no_global_streaming(tmp_path: Path) -> None:
    """The recorded ``"s"`` parent streams WITHOUT the global ``--streaming`` flag.

    Same fixture and levers as the low run of :func:`test_first_token_live_timing`
    (``ttft=1000`` / ``itl=60``) but with NO global ``--streaming``. The recorded
    per-node ``dispatch_overrides.stream=True`` wins for graph credits, so the
    parent still streams and emits a mid-flight ``FirstToken``: the post-TTFT
    child anchors at observed_first_token + D' (~3.0s) and the pre-TTFT child
    holds at its dispatch anchor (~1.0s), IDENTICAL to the global-on low run. The
    recorded ``"n"`` children stay non-streaming here too (single JSON body).
    """
    raw = _run_profile(tmp_path, ttft=1000, itl=60, streaming=False)
    d_pre, d_post, finish = _offsets(raw)
    n_parent, n_pre, n_post, n_tail = _assert_mixed_mode_wire(raw, "no-stream")
    # SAME expectations as the flagged ttft=1000 low run: per-record semantics, not
    # flag semantics -- streamed_request_count == parent dispatches, request_count
    # == 4x, TTFT avg ~= the controlled 1.0s (children EXCLUDED). Asserting against
    # the identical targets IS the "same as global-on low run" proof.
    src, rc, ttft_avg = _assert_streaming_metrics(raw, "no-stream", 1.0)
    print(
        f"[no-stream] ttft=1000 itl=60 (NO --streaming): "
        f"parent_finish={finish:.3f}s d_pre={d_pre:.3f}s d_post={d_post:.3f}s "
        f"resp(parent={n_parent} pre={n_pre} post={n_post} tail={n_tail}) "
        f"streamed_request_count={src} request_count={rc} "
        f"ttft_avg={ttft_avg:.3f}s"
    )

    # Post-TTFT child = OBSERVED first token (~ttft 1.0s) + D' (2.0s) = 3.0s: the
    # parent MUST have streamed a mid-flight FirstToken despite no global flag.
    assert abs(d_post - (1.0 + _D_PRIME_S)) <= _TOL_S, (
        f"[no-stream] post-TTFT child at {d_post:.3f}s, expected "
        f"{1.0 + _D_PRIME_S:.3f}s (first token + D') -- recorded parent must "
        "stream without --streaming"
    )
    # Pre-TTFT child = pure dispatch anchor (~1.0s), unaffected by streaming mode.
    assert abs(d_pre - _D_PRE_S) <= _TOL_S, (
        f"[no-stream] pre-TTFT child at {d_pre:.3f}s, expected {_D_PRE_S:.3f}s"
    )
    # Both children fired while the parent was verifiably IN FLIGHT.
    assert d_post < finish and d_pre < finish, (
        f"[no-stream] children did not fire mid-parent-flight: "
        f"d_pre={d_pre:.3f} d_post={d_post:.3f} parent_finish={finish:.3f}"
    )
