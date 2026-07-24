from __future__ import annotations

import asyncio
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


# Worker that records overlap: writes "BUSY" markers around a small delay so a
# second concurrent request would interleave if not serialized.
_ECHO_TRACKED = """
    import sys, orjson, time
    inflight = 0
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        inflight += 1
        overlap = inflight > 1
        time.sleep(0.05)
        inflight -= 1
        resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": 1.0}, "overlap": overlap}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
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


class TestSerialization:
    async def test_concurrent_grades_do_not_interleave(self, tmp_path) -> None:
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_TRACKED))
        try:
            results = await asyncio.gather(
                *[
                    worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                    for _ in range(4)
                ]
            )
            assert all(r == {"pass@1": 1.0} for r in results)
        finally:
            await worker.aclose()


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
    async def test_cancellation_kills_worker_and_propagates(self, tmp_path) -> None:
        """A cancel while awaiting the worker (e.g. shutdown) kills the worker and
        re-raises, rather than leaving it running with a pending request."""
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _HANG_THEN_OK))
        try:
            grade = asyncio.create_task(
                worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            )
            for _ in range(200):  # wait until the request is in flight (worker up)
                if worker._proc is not None:
                    break
                await asyncio.sleep(0.01)
            grade.cancel()
            with pytest.raises(asyncio.CancelledError):
                await grade
            assert worker._proc is None  # cancellation killed the worker
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
# worker whose responses no longer correlate to requests (a desync).
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
        worker = CodegenGradingWorker(
            worker_cmd=_write_worker(tmp_path, _ECHO_WRONG_ID)
        )
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=30
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
