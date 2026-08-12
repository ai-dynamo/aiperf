from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import textwrap
from pathlib import Path

import pytest

from aiperf.accuracy.graders._codegen_worker_client import (
    CodegenGradingWorker,
    CodegenWorkerError,
)

pytestmark = pytest.mark.asyncio


def _write_worker(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_worker.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


# Receives a request, signals receipt via a file, then blocks until a "go"
# file appears before responding.  Used to guarantee fut is unresolved when
# we cancel, working around a Python 3.11 asyncio.wait_for bug where an
# already-resolved future silently absorbs the CancelledError.
_GATED_OK = """
    import sys, orjson, pathlib, time
    gate_dir = pathlib.Path(sys.argv[1])
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        (gate_dir / f"recv_{req['id']}").touch()
        while not (gate_dir / "go").exists():
            time.sleep(0.005)
        resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": 1.0}}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""


def _write_gated_worker(tmp_path: Path) -> tuple[list[str], Path]:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    script = tmp_path / "gated_worker.py"
    script.write_text(textwrap.dedent(_GATED_OK))
    return [sys.executable, str(script), str(gate_dir)], gate_dir


# Echoes pass@1=1.0 for every request, correlating id.
_ECHO_OK = """
    import sys, orjson
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": 1.0}}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""

# Buffers the first 4 requests and responds in REVERSE id order to exercise
# the demux table (correct demux requires id matching, not position matching).
_REVERSE_BATCH_OF_4 = """
    import sys, orjson
    buf = []
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        buf.append(req)
        if len(buf) == 4:
            for r in reversed(buf):
                resp = {"id": r["id"], "ok": True, "metrics": {"pass@1": r["id"] * 0.1}}
                sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
                sys.stdout.buffer.flush()
            buf = []
"""


class TestHappyPath:
    async def test_grade_returns_metrics(self, tmp_path) -> None:
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            metrics = await worker.grade_codegen(
                [{"input_output": "{}"}], [["x"]], timeout=30
            )
            assert metrics == {"pass@1": 1.0}
        finally:
            await worker.aclose()

    async def test_second_grade_reuses_same_worker(self, tmp_path) -> None:
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            pid1 = worker._proc.pid  # type: ignore[union-attr]
            await worker.grade_codegen([{"input_output": "{}"}], [["y"]], timeout=30)
            assert worker._proc.pid == pid1  # type: ignore[union-attr]
        finally:
            await worker.aclose()


class TestConcurrency:
    async def test_concurrent_grades_all_complete(self, tmp_path) -> None:
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            results = await asyncio.gather(
                *[
                    w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                    for _ in range(5)
                ]
            )
            assert all(r == {"pass@1": 1.0} for r in results)
        finally:
            await w.aclose()

    async def test_concurrent_grades_demux_by_id_not_position(self, tmp_path) -> None:
        # 4 concurrent grades; mock responds in reverse order.
        # If demux were position-based, callers would get wrong metrics.
        w = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _REVERSE_BATCH_OF_4)
        )
        try:
            results = await asyncio.gather(
                *[
                    w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                    for _ in range(4)
                ]
            )
            # IDs 1-4 → pass@1 values 0.1, 0.2, 0.3, 0.4 (one per caller)
            values = sorted(r["pass@1"] for r in results)
            assert values == pytest.approx([0.1, 0.2, 0.3, 0.4])
        finally:
            await w.aclose()

    async def test_fault_cancels_all_pending_futures(self, tmp_path) -> None:
        # Worker dies immediately after the first line — all concurrent callers
        # should raise CodegenWorkerError, not hang.
        w = CodegenGradingWorker(
            worker_cmd=_write_worker(
                tmp_path,
                """
                import sys
                sys.stdin.buffer.readline()  # consume one line then exit
                """,
            )
        )
        try:
            with pytest.raises(CodegenWorkerError):
                await asyncio.gather(
                    *[
                        w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=10)
                        for _ in range(3)
                    ],
                    return_exceptions=False,
                )
        finally:
            await w.aclose()

    async def test_stale_id_after_timeout_does_not_crash(self, tmp_path) -> None:
        # Reader receives a response for an id that the caller already timed out on.
        # The stale future was already removed from _pending; the reader must skip it.
        # Use _ECHO_OK with a very short timeout so the grade times out, then send
        # a second grade to prove the worker (if restarted) still works.
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            with pytest.raises(CodegenWorkerError):
                await w.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=0.000001
                )
            # If stale id handling is broken, the second grade would hang or crash.
            result = await w.grade_codegen(
                [{"input_output": "{}"}], [["x"]], timeout=10
            )
            assert result == {"pass@1": 1.0}
        finally:
            await w.aclose()

    async def test_aclose_with_pending_futures_does_not_hang(self, tmp_path) -> None:
        hang_worker = """
            import sys, time
            for line in sys.stdin.buffer:
                time.sleep(3600)
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, hang_worker))
        grade_task = asyncio.create_task(
            w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=60)
        )
        await asyncio.sleep(0.05)  # let grade_task start and block
        await w.aclose()  # must not hang even with grade_task pending
        grade_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, CodegenWorkerError):
            await grade_task

    async def test_concurrent_grades_return_correct_results(self, tmp_path) -> None:
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            results = await asyncio.gather(
                *[
                    w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                    for _ in range(4)
                ]
            )
            assert all(r == {"pass@1": 1.0} for r in results)
        finally:
            await w.aclose()


# The very first grade (client request id==1) hangs forever to trigger a
# client-side timeout+kill. The client's request id is monotonic and survives
# the restart, so the respawned worker sees id==2 and responds immediately.
_HANG_THEN_OK = """
    import sys, orjson, time
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        if req["id"] == 1:
            time.sleep(3600)
        resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": 1.0}}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""


# Responds OK to the first request (id==1), hangs on every later one.
_OK_THEN_HANG = """
    import sys, orjson, time
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        if req["id"] != 1:
            time.sleep(3600)
        resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": 1.0}}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""


class TestTimeoutRestart:
    async def test_timeout_raises_and_next_grade_respawns(self, tmp_path) -> None:
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _HANG_THEN_OK))
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=0.2
                )
            assert worker._proc is None  # killed
            # A fresh worker is spawned; this one's "first" request is fast.
            metrics = await worker.grade_codegen(
                [{"input_output": "{}"}], [["y"]], timeout=30
            )
            assert metrics == {"pass@1": 1.0}
        finally:
            await worker.aclose()

    async def test_timeout_on_proven_worker_does_not_count_as_start_failure(
        self, tmp_path
    ) -> None:
        # First request succeeds (worker becomes proven); the second hangs and
        # times out. A proven worker's timeout is a runtime fault, not a startup
        # failure, so _start_failures must stay 0.
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _OK_THEN_HANG))
        try:
            await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            assert worker._worker_proven is True
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["y"]], timeout=0.2
                )
            assert worker._start_failures == 0
        finally:
            await worker.aclose()

    async def test_timeout_on_unproven_worker_does_not_count_as_start_failure(
        self, tmp_path
    ) -> None:
        # Regression: a slow FIRST grade (worker never proven) is a per-grade
        # timeout, not a worker-startup failure. Counting it would let a few slow
        # problems at the start of a run trip the cap and disable all grading.
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _HANG_THEN_OK))
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=0.2
                )
            assert worker._start_failures == 0
        finally:
            await worker.aclose()


class TestCancellation:
    async def test_cancellation_propagates_without_killing_worker(
        self, tmp_path
    ) -> None:
        # Cancellation removes the request from _pending and re-raises; it does
        # not fault the worker because the protocol is not desynced (the late
        # response is dropped as a stale id by _dispatch_response). Concurrent
        # grades continue unaffected.
        #
        # Uses a gated worker (not _ECHO_OK) so fut is guaranteed unresolved
        # when grade.cancel() fires.  On Python 3.11, asyncio.wait_for silently
        # absorbs CancelledError if the inner future already has a result, so
        # we must ensure the worker has not yet responded at cancel time.
        worker_cmd, gate_dir = _write_gated_worker(tmp_path)
        worker = CodegenGradingWorker(worker_cmd=worker_cmd)
        try:
            grade = asyncio.create_task(
                worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            )
            for _ in range(200):  # wait until the worker received the request
                if (gate_dir / "recv_1").exists():
                    break
                await asyncio.sleep(0.01)
            # Worker holds at the gate — fut is unresolved — cancel is safe.
            grade.cancel()
            with pytest.raises(asyncio.CancelledError):
                await grade
            assert worker._proc is not None  # worker stays alive after cancel
            # Release the gate so the stale response drains and the next grade works.
            (gate_dir / "go").touch()
            result = await worker.grade_codegen(
                [{"input_output": "{}"}], [["x"]], timeout=10
            )
            assert result == {"pass@1": 1.0}
        finally:
            await worker.aclose()


# Exits immediately without responding (simulates import crash on startup).
_DIE_ON_START = """
    import sys
    sys.exit(1)
"""


class TestCrashAndCap:
    async def test_worker_that_dies_on_start_hits_cap(self, tmp_path) -> None:
        worker = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _DIE_ON_START), max_start_failures=3
        )
        try:
            for _ in range(3):
                with pytest.raises(CodegenWorkerError):
                    await worker.grade_codegen(
                        [{"input_output": "{}"}], [["x"]], timeout=5
                    )
            # Cap reached: further grades fast-fail without spawning.
            with pytest.raises(CodegenWorkerError, match="unavailable after"):
                await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
        finally:
            await worker.aclose()


# Reads a request then emits a non-JSON line, simulating a worker whose stdout
# has desynced. The raw bytes must not propagate as a decode error.
_EMIT_GARBAGE = """
    import sys
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        sys.stdout.buffer.write(b"not json\\n")
        sys.stdout.buffer.flush()
"""


# Claims success but omits the metrics dict, simulating the grading path that
# returned ok:true without results. This is the class closest to the real bug:
# it must fault instead of returning junk (or KeyError-ing) to the grader.
_EMIT_OK_NO_METRICS = """
    import sys, orjson
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        resp = {"id": req["id"], "ok": True}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""


# Emits a single line larger than the client's StreamReader limit so
# readline() raises ValueError (asyncio.LimitOverrunError). The overrun must
# route through the fault path instead of escaping it and desyncing the client.
_EMIT_OVERSIZED_LINE = """
    import sys
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        sys.stdout.buffer.write(b"a" * (17 * 1024 * 1024) + b"\\n")
        sys.stdout.buffer.flush()
"""


# Always echoes a fixed WRONG id regardless of the request id, simulating a
# worker whose responses no longer correlate to requests (a desync). With the
# demux reader, the stale id is silently dropped and the caller times out.
_ECHO_WRONG_ID = """
    import sys, orjson
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        orjson.loads(line)
        resp = {"id": 999, "ok": True, "metrics": {"pass@1": 1.0}}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""


class TestOversizedResponse:
    async def test_oversized_line_faults_and_kills_worker(self, tmp_path) -> None:
        worker = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _EMIT_OVERSIZED_LINE)
        )
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=30
                )
            assert worker._proc is None
        finally:
            await worker.aclose()


class TestResponseIdMismatch:
    async def test_wrong_id_faults_and_kills_worker(self, tmp_path) -> None:
        # With the demux reader, a wrong id is treated as a stale response and
        # silently dropped. The caller's future is never resolved, so it times out
        # and calls _handle_fault, which kills the worker.
        worker = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _ECHO_WRONG_ID)
        )
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=0.5
                )
            assert worker._proc is None
        finally:
            await worker.aclose()


class TestMalformedResponse:
    async def test_non_json_response_faults_and_kills_worker(self, tmp_path) -> None:
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _EMIT_GARBAGE))
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
            assert worker._proc is None
        finally:
            await worker.aclose()

    async def test_ok_without_metrics_faults_and_kills_worker(self, tmp_path) -> None:
        worker = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _EMIT_OK_NO_METRICS)
        )
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
            assert worker._proc is None
        finally:
            await worker.aclose()


# Writes a distinctive marker to stderr, flushes, then exits without responding.
_STDERR_THEN_DIE = """
    import sys
    sys.stderr.write("WORKER_DIAG_MARKER: boom\\n")
    sys.stderr.flush()
    sys.exit(1)
"""


class TestStderrDiagnostics:
    async def test_worker_stderr_tail_logged_on_fault(
        self, tmp_path, monkeypatch
    ) -> None:
        """On a fault the worker's captured stderr tail is logged so operators can
        see why it died, instead of a bare 'exited before responding'."""
        import aiperf.accuracy.graders._codegen_worker_client as wc

        logged: list[str] = []
        monkeypatch.setattr(
            wc._log, "debug", lambda m: logged.append(m() if callable(m) else m)
        )
        worker = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _STDERR_THEN_DIE)
        )
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=30
                )
            assert any("WORKER_DIAG_MARKER" in msg for msg in logged), logged
        finally:
            await worker.aclose()


class TestProcessGroup:
    @pytest.mark.skipif(
        not hasattr(os, "getpgid"), reason="process groups unavailable on this platform"
    )
    async def test_worker_is_spawned_as_process_group_leader(self, tmp_path) -> None:
        """start_new_session=True makes the worker its own process-group leader, so
        _kill can killpg the whole group (worker + lighteval's forked sandbox
        grandchildren) instead of leaking the children when the worker dies."""
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            pid = worker._proc.pid  # type: ignore[union-attr]
            # A session leader's process-group id equals its own pid; without
            # start_new_session the worker would share the test runner's group.
            assert os.getpgid(pid) == pid
        finally:
            await worker.aclose()


class TestDeathPipe:
    @pytest.mark.skipif(
        not hasattr(os, "killpg"),
        reason="death pipe is POSIX-only; Windows has no killpg to reap with",
    )
    async def test_death_pipe_held_for_worker_life_then_closed(self, tmp_path) -> None:
        # The client holds the death-pipe write end open for the worker's whole
        # life (so the parent's exit — even os._exit — signals the worker to
        # reap itself), and closes it on teardown.
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            assert worker._death_w is not None
        finally:
            await worker.aclose()


class TestCoverageGaps:
    """Targeted tests for previously uncovered branches."""

    async def test_grade_codegen_proc_is_none_raises(self) -> None:
        # Covers the proc-is-None guard added after _ensure_worker (line 107-109).
        w = CodegenGradingWorker.__new__(CodegenGradingWorker)
        w._cmd = ["python", "-c", ""]
        w._max_start_failures = 3
        w._proc = None
        w._spawn_lock = asyncio.Lock()
        w._pending = {}
        w._reader_task = None
        w._next_id = 0
        w._start_failures = 0
        w._worker_proven = False
        from collections import deque

        w._stderr_tail = deque(maxlen=64)
        w._stderr_task = None
        w._death_w = None
        # Manually set _proc to a sentinel with returncode=None so _ensure_worker returns,
        # then null out stdin so the None-stdin branch is hit.

        class _FakeStdin:
            pass

        class _FakeProc:
            returncode = None
            stdin = None
            stdout = None
            stderr = None
            pid = 99999

        w._proc = _FakeProc()  # type: ignore[assignment]
        with pytest.raises(CodegenWorkerError, match="not running"):
            await w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)

    async def test_grade_codegen_write_error_raises_worker_error(
        self, tmp_path: Path
    ) -> None:
        # Covers OSError from write/drain (lines 113-117).
        # Use _ECHO_OK but close stdin immediately after spawn to force BrokenPipeError.
        broken_worker = """
            import sys
            sys.stdin.close()
            import time; time.sleep(10)
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, broken_worker))
        try:
            with pytest.raises(CodegenWorkerError):
                await w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
        finally:
            await w.aclose()

    async def test_ensure_worker_respawns_after_worker_exits(
        self, tmp_path: Path
    ) -> None:
        # Worker serves one request then exits. The second grade_codegen() call
        # must detect the dead worker via returncode, respawn, and succeed.
        die_after_one = """
            import sys, orjson
            line = sys.stdin.buffer.readline()
            req = orjson.loads(line)
            resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": 1.0}}
            sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
            sys.stdout.buffer.flush()
            sys.exit(0)
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, die_after_one))
        try:
            result1 = await w.grade_codegen(
                [{"input_output": "{}"}], [["x"]], timeout=10
            )
            assert result1 == {"pass@1": 1.0}
            first_proc = w._proc
            # Let the worker finish exiting so _ensure_worker sees returncode != None.
            if first_proc is not None:
                await first_proc.wait()
            result2 = await w.grade_codegen(
                [{"input_output": "{}"}], [["x"]], timeout=10
            )
            assert result2 == {"pass@1": 1.0}
            assert w._proc is not first_proc, (
                "expected a new worker process after respawn"
            )
        finally:
            await w.aclose()

    async def test_aclose_with_pending_futures_sets_exception(
        self, tmp_path: Path
    ) -> None:
        # Covers lines 316-317: aclose() sets CodegenWorkerError on pending futures.
        hang_worker = """
            import sys, time
            for line in sys.stdin.buffer:
                time.sleep(3600)
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, hang_worker))
        grade_task = asyncio.create_task(
            w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=60)
        )
        await asyncio.sleep(0.1)
        await w.aclose()
        with pytest.raises((CodegenWorkerError, asyncio.CancelledError)):
            await grade_task

    async def test_dispatch_response_null_id_faults(self, tmp_path: Path) -> None:
        # Covers the req_id is None guard in _dispatch_response (line 209-212).
        null_id_worker = """
            import sys, orjson
            for line in sys.stdin.buffer:
                line = line.strip()
                if not line:
                    continue
                resp = {"id": None, "ok": False, "error": "bad parse"}
                sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
                sys.stdout.buffer.flush()
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, null_id_worker))
        try:
            with pytest.raises(CodegenWorkerError):
                await w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
        finally:
            await w.aclose()

    async def test_run_reader_json_decode_error_faults(self, tmp_path: Path) -> None:
        # Covers the JSONDecodeError path in _run_reader (line 254).
        garbage_worker = """
            import sys
            for line in sys.stdin.buffer:
                sys.stdout.buffer.write(b"not json at all\\n")
                sys.stdout.buffer.flush()
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, garbage_worker))
        try:
            with pytest.raises(CodegenWorkerError):
                await w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
        finally:
            await w.aclose()

    async def test_aclose_sets_exception_on_unfulfilled_pending_futures(
        self, tmp_path: Path
    ) -> None:
        # Covers lines 316-317: aclose() calls set_exception on pending futures
        # that are still waiting. The test injects a future manually to guarantee
        # _pending is non-empty when aclose() runs (avoids looptime timing issues).
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        w._pending[99] = fut
        await w.aclose()
        assert fut.done()
        with pytest.raises(CodegenWorkerError, match="closed"):
            fut.result()

    async def test_dispatch_response_ok_false_sets_exception_and_marks_proven(
        self, tmp_path: Path
    ) -> None:
        # Covers lines 221-225: ok=False dispatch path.
        error_worker = """
            import sys, orjson
            for line in sys.stdin.buffer:
                req = orjson.loads(line.strip())
                resp = {"id": req["id"], "ok": False, "error": "deliberate error"}
                sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
                sys.stdout.buffer.flush()
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, error_worker))
        try:
            with pytest.raises(CodegenWorkerError, match="deliberate error"):
                await w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=10)
            assert w._worker_proven  # ok=False counts as proven (worker responded)
        finally:
            await w.aclose()

    async def test_dispatch_response_unhashable_id_faults(self, tmp_path: Path) -> None:
        # Covers lines 215-217: TypeError on _pending.pop(unhashable_id).
        list_id_worker = """
            import sys, orjson
            for line in sys.stdin.buffer:
                resp = {"id": [1, 2], "ok": True, "metrics": {"pass@1": 1.0}}
                sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
                sys.stdout.buffer.flush()
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, list_id_worker))
        try:
            with pytest.raises(CodegenWorkerError):
                await w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
        finally:
            await w.aclose()

    async def test_drain_stderr_none_reader_is_noop(self) -> None:
        # Covers line 194: _drain_stderr returns immediately when reader is None.
        w = CodegenGradingWorker.__new__(CodegenGradingWorker)
        from collections import deque

        w._stderr_tail = deque(maxlen=64)
        # Should complete without error
        await w._drain_stderr(None)
