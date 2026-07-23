from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from aiperf.accuracy.graders._codegen_worker_client import (
    CodegenGradingWorker,
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
