from __future__ import annotations

import asyncio
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
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
            assert worker._worker_proven is True
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


class TestMalformedResponse:
    async def test_non_json_response_faults_and_kills_worker(self, tmp_path) -> None:
        worker = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _EMIT_GARBAGE))
        try:
            with pytest.raises(CodegenWorkerError):
                await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=5)
            assert worker._proc is None
        finally:
            await worker.aclose()
