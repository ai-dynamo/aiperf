from __future__ import annotations

import io
import multiprocessing as mp
import subprocess
import sys
import textwrap
from typing import Any

import orjson
import pytest

from aiperf.accuracy.graders import _codegen_worker as worker

_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


def _fake_codegen_ok(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
    return {"pass@1": 1.0}, {}


def _fake_codegen_boom(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
    raise RuntimeError("sandbox exploded")


def _fake_codegen_list_pass(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
    return {"pass@1": [1.0]}, {}


class TestHandleRequest:
    def test_ok_request_returns_metrics_with_id(self) -> None:
        req = {
            "id": 7,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }
        resp = worker.handle_request(req, _fake_codegen_ok)
        assert resp == {"id": 7, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_list_shaped_pass_at_1_is_preserved(self) -> None:
        # lighteval returns pass@1 as a list on some pins; it must survive
        # coercion rather than be dropped as non-numeric (silent 0.000 bug).
        req = {
            "id": 9,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }
        resp = worker.handle_request(req, _fake_codegen_list_pass)
        assert resp["ok"] is True
        assert resp["metrics"]["pass@1"] == [1.0]

    def test_codegen_exception_becomes_error_response(self) -> None:
        req = {
            "id": 3,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }
        resp = worker.handle_request(req, _fake_codegen_boom)
        assert resp["id"] == 3
        assert resp["ok"] is False
        assert "sandbox exploded" in resp["error"]

    def test_malformed_request_missing_fields_is_error(self) -> None:
        resp = worker.handle_request({"id": 5}, _fake_codegen_ok)
        assert resp["id"] == 5
        assert resp["ok"] is False
        assert resp["error"]

    def test_non_object_request_is_error_not_crash(self) -> None:
        # A valid-but-non-object JSON frame must not raise (which would kill the
        # worker loop); it returns the promised error response with id=None.
        resp = worker.handle_request([1, 2, 3], _fake_codegen_ok)
        assert resp["id"] is None
        assert resp["ok"] is False
        assert resp["error"]

    def test_non_finite_metric_values_are_dropped(self) -> None:
        # NaN/Inf must not cross the JSONL boundary (repo NaN/Inf discipline).
        def _nan_inf(*_a: Any, **_k: Any) -> tuple[dict[str, Any], Any]:
            return {"pass@1": float("nan"), "extra": float("inf"), "ok": 1.0}, {}

        resp = worker.handle_request(
            {"id": 9, "evaluation_sample": [{}], "generated_code": [["x"]]}, _nan_inf
        )
        assert resp["ok"] is True
        assert "pass@1" not in resp["metrics"]
        assert "extra" not in resp["metrics"]
        assert resp["metrics"]["ok"] == 1.0


class TestRunWorkerLoop:
    def _run(self, requests: list[bytes], codegen_fn) -> list[dict]:
        stdin = io.BytesIO(b"".join(r + b"\n" for r in requests))
        out = io.BytesIO()
        worker.run_worker_loop(stdin, out, codegen_fn)
        out.seek(0)
        return [orjson.loads(line) for line in out.read().splitlines() if line]

    def test_processes_each_request_in_order(self) -> None:
        reqs = [
            orjson.dumps(
                {"id": 1, "evaluation_sample": [{}], "generated_code": [["a"]]}
            ),
            orjson.dumps(
                {"id": 2, "evaluation_sample": [{}], "generated_code": [["b"]]}
            ),
        ]
        resps = self._run(reqs, _fake_codegen_ok)
        assert [r["id"] for r in resps] == [1, 2]
        assert all(r["ok"] for r in resps)

    def test_eof_stops_the_loop(self) -> None:
        resps = self._run([], _fake_codegen_ok)
        assert resps == []

    def test_garbled_line_yields_error_response(self) -> None:
        resps = self._run([b"{not json"], _fake_codegen_ok)
        assert len(resps) == 1
        assert resps[0]["ok"] is False
        assert resps[0]["id"] is None


class TestForceFork:
    @pytest.mark.skipif(not _FORK_AVAILABLE, reason="fork unavailable")
    def test_sets_fork_start_method(self) -> None:
        # Run in a subprocess: _force_fork mutates the process-global start
        # method, so doing it in the pytest process could leak `fork` into later
        # spawn-based tests (the finally can't restore a previously-unset method).
        probe = textwrap.dedent(
            """
            import multiprocessing as mp
            from aiperf.accuracy.graders import _codegen_worker as w
            mp.set_start_method("spawn", force=True)
            w._force_fork()
            print(mp.get_start_method())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
        )
        assert result.stdout.strip() == "fork", result.stderr


class TestStdoutGuardSubprocess:
    def test_only_protocol_frames_reach_stdout(self, tmp_path) -> None:
        # A tiny program that installs the guard, prints junk to stdout, then
        # writes one protocol frame. Only the frame may appear on real stdout.
        script = tmp_path / "guard_probe.py"
        script.write_text(
            textwrap.dedent(
                """
                import sys
                from aiperf.accuracy.graders import _codegen_worker as w
                proto = w._install_stdout_guard()
                print("LIBRARY NOISE TO STDOUT")
                sys.stdout.flush()
                proto.write(b'{"frame": 1}\\n')
                proto.flush()
                """
            )
        )
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, timeout=30
        )
        assert result.stdout == b'{"frame": 1}\n'
        assert b"LIBRARY NOISE" in result.stderr
