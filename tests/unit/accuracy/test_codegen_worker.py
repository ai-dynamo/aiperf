from __future__ import annotations

import io
import multiprocessing as mp
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.accuracy.graders import _codegen_worker as worker

_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


def _fake_compute_metrics(
    results: dict, k_list: list[int] | None = None
) -> dict[str, Any]:
    # Mirrors compute_metrics_from_results: returns {"pass@1": <float>} using
    # the single-problem results dict {0: [[True, True, ...]]} passed by handle_batch.
    result_list = results.get(0, [[-2]])  # [-2] = compile error
    if result_list and all(x > 0 for x in result_list[0]):
        return {"pass@1": 1.0}
    return {"pass@1": 0.0}


def _fake_codegen_batch_ok(
    samples: list, generations: list, **_kwargs: Any
) -> tuple[dict[str, Any], dict[int, list]]:
    # Returns aggregate metrics (ignored by handle_batch) and per-problem results.
    n = len(samples)
    raw_results = {i: [[True]] for i in range(n)}  # all pass
    return {"pass@1": 1.0}, raw_results


def _fake_codegen_batch_boom(
    samples: list, generations: list, **_kwargs: Any
) -> tuple[dict[str, Any], dict[int, list]]:
    raise RuntimeError("pool exploded")


class TestHandleBatch:
    def _req(self, req_id: int) -> dict[str, Any]:
        return {
            "id": req_id,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }

    def test_single_request_returns_one_ok_response(self) -> None:
        resps = worker.handle_batch(
            [self._req(1)], _fake_codegen_batch_ok, _fake_compute_metrics
        )
        assert len(resps) == 1
        assert resps[0] == {"id": 1, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_batch_of_n_calls_codegen_fn_once(self) -> None:
        call_count = 0

        def counting_codegen(samples, generations, **kwargs):
            nonlocal call_count
            call_count += 1
            n = len(samples)
            return {"pass@1": 1.0}, {i: [[True]] for i in range(n)}

        reqs = [self._req(i) for i in range(1, 5)]
        resps = worker.handle_batch(reqs, counting_codegen, _fake_compute_metrics)
        assert call_count == 1
        assert len(resps) == 4
        assert all(r["ok"] for r in resps)
        assert [r["id"] for r in resps] == [1, 2, 3, 4]

    def test_response_order_matches_request_order(self) -> None:
        reqs = [self._req(i) for i in [7, 3, 99]]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert [r["id"] for r in resps] == [7, 3, 99]

    def test_batch_exception_returns_error_for_all(self) -> None:
        reqs = [self._req(i) for i in range(1, 4)]
        resps = worker.handle_batch(
            reqs, _fake_codegen_batch_boom, _fake_compute_metrics
        )
        assert len(resps) == 3
        assert all(not r["ok"] for r in resps)
        assert all("pool exploded" in r["error"] for r in resps)

    def test_malformed_request_in_batch_does_not_affect_others(self) -> None:
        reqs = [
            self._req(1),
            {"id": 2},  # missing evaluation_sample + generated_code
            self._req(3),
        ]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert len(resps) == 3
        assert resps[0] == {"id": 1, "ok": True, "metrics": {"pass@1": 1.0}}
        assert resps[1]["id"] == 2
        assert not resps[1]["ok"]
        assert resps[2] == {"id": 3, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_non_object_request_in_batch_is_error(self) -> None:
        reqs = [[1, 2, 3], self._req(5)]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert len(resps) == 2
        assert resps[0]["id"] is None
        assert not resps[0]["ok"]
        assert resps[1] == {"id": 5, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_parse_error_sentinel_produces_error_response(self) -> None:
        # run_worker_loop encodes JSON decode errors as {"_parse_error": "..."}.
        reqs = [{"_parse_error": "unexpected token"}, self._req(2)]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert len(resps) == 2
        assert resps[0]["id"] is None
        assert not resps[0]["ok"]
        assert "unexpected token" in resps[0]["error"]
        assert resps[1]["ok"]

    def test_list_shaped_pass_at_1_is_preserved(self) -> None:
        # lighteval returns pass@1 as a list on some pins; _coerce_metrics must
        # preserve it rather than drop it (silent 0.000 bug if dropped).
        def list_metrics(_results: dict, **_kw: Any) -> dict[str, Any]:
            return {"pass@1": [1.0]}

        resps = worker.handle_batch(
            [self._req(1)], _fake_codegen_batch_ok, list_metrics
        )
        assert resps[0]["ok"] is True
        assert resps[0]["metrics"]["pass@1"] == [1.0]

    def test_compute_metrics_exception_produces_error_response(self) -> None:
        def boom_compute(_results: dict, **_kw: Any) -> dict[str, Any]:
            raise RuntimeError("metrics exploded")

        resps = worker.handle_batch(
            [self._req(1)], _fake_codegen_batch_ok, boom_compute
        )
        assert not resps[0]["ok"]
        assert "metrics exploded" in resps[0]["error"]

    def test_truncate_error_long_string(self) -> None:
        long_err = "x" * 5000
        truncated = worker._truncate_error(long_err)
        assert len(truncated) < 5000
        assert truncated.endswith("...[truncated]")

    def test_is_number_non_numeric_returns_false(self) -> None:
        assert not worker._is_number("notanumber")
        assert not worker._is_number(None)
        assert not worker._is_number([1, 2])

    def test_non_finite_metric_values_are_dropped(self) -> None:
        # NaN/Inf must not cross the JSONL boundary (repo NaN/Inf discipline).
        def _nan_inf(
            samples: list, generations: list, **_kwargs: Any
        ) -> tuple[dict[str, Any], dict[int, list]]:
            n = len(samples)
            return {"pass@1": 1.0}, {i: [[True]] for i in range(n)}

        def _nan_compute(results: dict, **_kwargs: Any) -> dict[str, Any]:
            return {"pass@1": float("nan"), "extra": float("inf"), "ok": 1.0}

        req = {"id": 9, "evaluation_sample": [{}], "generated_code": [["x"]]}
        resps = worker.handle_batch([req], _nan_inf, _nan_compute)
        assert resps[0]["ok"] is True
        assert "pass@1" not in resps[0]["metrics"]
        assert "extra" not in resps[0]["metrics"]
        assert resps[0]["metrics"]["ok"] == 1.0


class TestRunWorkerLoopBatch:
    def _run(
        self,
        payloads: list[dict[str, Any]],
        codegen_fn: Callable[
            ..., tuple[dict[str, Any], dict[int, list]]
        ] = _fake_codegen_batch_ok,
        compute_metrics_fn: Callable[..., dict[str, Any]] = _fake_compute_metrics,
    ) -> list[dict[str, Any]]:
        # Write all payloads to a BytesIO pipe so they are already queued when
        # run_worker_loop reads; this exercises the non-blocking drain path.
        data = b"".join(orjson.dumps(p) + b"\n" for p in payloads)
        stdin = io.BytesIO(data)
        out = io.BytesIO()
        worker.run_worker_loop(stdin, out, codegen_fn, compute_metrics_fn)
        out.seek(0)
        return [orjson.loads(line) for line in out if line.strip()]

    def _req(self, req_id: int) -> dict[str, Any]:
        return {
            "id": req_id,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }

    def test_pre_queued_requests_are_batched_in_one_call(self) -> None:
        call_count = 0

        def counting_codegen(samples, generations, **kwargs):
            nonlocal call_count
            call_count += 1
            n = len(samples)
            return {"pass@1": 1.0}, {i: [[True]] for i in range(n)}

        reqs = [self._req(i) for i in range(1, 4)]
        resps = self._run(reqs, counting_codegen)
        assert call_count == 1
        assert len(resps) == 3

    def test_responses_carry_correct_ids(self) -> None:
        reqs = [self._req(i) for i in [10, 20, 30]]
        resps = self._run(reqs)
        assert {r["id"] for r in resps} == {10, 20, 30}

    def test_drain_in_memory_splits_on_newlines(self) -> None:
        data = b'{"id":1}\n{"id":2}\n'
        result = worker._drain_in_memory(io.BytesIO(data))
        assert result == [b'{"id":1}', b'{"id":2}']

    def test_drain_buffered_no_fcntl_and_no_fd_uses_in_memory_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(worker, "_HAS_FCNTL", False)
        data = io.BytesIO(b'{"id":1}\n{"id":2}\n')
        result = worker._drain_buffered(data)
        assert result == [b'{"id":1}', b'{"id":2}']

    def test_drain_buffered_no_fcntl_with_real_fd_skips_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(worker, "_HAS_FCNTL", False)
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b'{"id":1}\n')
            with os.fdopen(r_fd, "rb") as reader:
                r_fd = -1  # fdopen takes ownership
                result = worker._drain_buffered(reader)
            assert result == []  # skipped, not blocked
        finally:
            if r_fd >= 0:
                os.close(r_fd)
            os.close(w_fd)

    def test_drain_buffered_complete_line_then_empty_pipe_stops(self) -> None:
        # BufferedReader.peek() returns b"" (not BlockingIOError) when the kernel
        # pipe buffer is empty with write end still open. Verifies that the drain
        # reads one complete line and stops cleanly on the next b"" peek.
        req = self._req(77)
        data = orjson.dumps(req) + b"\n"
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, data)  # one complete line, write end still open
            with os.fdopen(r_fd, "rb") as reader:
                r_fd = -1
                result = worker._drain_buffered(reader)
            assert result == [data.strip()]
        finally:
            if r_fd >= 0:
                os.close(r_fd)
            os.close(w_fd)

    def test_drain_buffered_partial_line_breaks_without_readline(self) -> None:
        # Covers lines 202-207: peek returns bytes without newline → break.
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"partial-no-newline")  # no \n, write end still open
            with os.fdopen(r_fd, "rb") as reader:
                r_fd = -1
                result = worker._drain_buffered(reader)
            assert result == []  # nothing drained; no complete line
        finally:
            if r_fd >= 0:
                os.close(r_fd)
            os.close(w_fd)

    def test_run_worker_loop_skips_blank_lines(self) -> None:
        # Covers line 237: blank lines between requests are skipped (continue).
        reqs = [self._req(1), self._req(2)]
        data = b"\n" + orjson.dumps(reqs[0]) + b"\n\n" + orjson.dumps(reqs[1]) + b"\n"
        stdin = io.BytesIO(data)
        out = io.BytesIO()
        worker.run_worker_loop(
            stdin, out, _fake_codegen_batch_ok, _fake_compute_metrics
        )
        out.seek(0)
        resps = [orjson.loads(ln) for ln in out if ln.strip()]
        assert {r["id"] for r in resps} == {1, 2}

    def test_partial_jsonl_line_is_deferred_to_next_cycle(self) -> None:
        # Exercises the O_NONBLOCK peek(0) + partial-line guard in _drain_buffered.
        # A partial write (no trailing newline) must not be submitted as a request;
        # only after the newline arrives should the line be processed.
        import threading
        import time

        req = self._req(42)
        line = orjson.dumps(req) + b"\n"
        r_fd, w_fd = os.pipe()
        out = io.BytesIO()

        with os.fdopen(r_fd, "rb") as reader:
            t = threading.Thread(
                target=worker.run_worker_loop,
                args=(reader, out, _fake_codegen_batch_ok, _fake_compute_metrics),
                daemon=True,
            )
            t.start()

            # Write the request in two parts: body first, newline second
            os.write(w_fd, line[:-1])  # partial — no newline yet
            time.sleep(0.05)  # give the loop a chance to drain

            # Worker must still be blocked (no complete line yet)
            assert t.is_alive()
            out.seek(0)
            assert out.read() == b""  # nothing written yet

            # Complete the line and close stdin to trigger EOF after processing
            os.write(w_fd, b"\n")
            os.close(w_fd)
            t.join(timeout=5)

        assert not t.is_alive()
        out.seek(0)
        resps = [orjson.loads(ln) for ln in out if ln.strip()]
        assert len(resps) == 1
        assert resps[0]["id"] == 42
        assert resps[0]["ok"] is True


class TestRunWorkerLoop:
    def _run(self, requests: list[bytes], codegen_fn) -> list[dict]:
        stdin = io.BytesIO(b"".join(r + b"\n" for r in requests))
        out = io.BytesIO()
        worker.run_worker_loop(stdin, out, codegen_fn, _fake_compute_metrics)
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
        resps = self._run(reqs, _fake_codegen_batch_ok)
        assert [r["id"] for r in resps] == [1, 2]
        assert all(r["ok"] for r in resps)

    def test_eof_stops_the_loop(self) -> None:
        resps = self._run([], _fake_codegen_batch_ok)
        assert resps == []

    def test_garbled_line_yields_error_response(self) -> None:
        resps = self._run([b"{not json"], _fake_codegen_batch_ok)
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


class TestProtocolFdIsolation:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
    def test_protocol_fd_closed_in_forked_children(self, tmp_path: Path) -> None:
        # lighteval forks sandbox children that run arbitrary generated code.
        # They must NOT inherit the protocol fd, or that code could write to the
        # client's JSONL response channel and spoof/desync grading.
        script = tmp_path / "fork_probe.py"
        script.write_text(
            textwrap.dedent(
                """
                import os
                from aiperf.accuracy.graders import _codegen_worker as w
                proto = w._install_stdout_guard()
                pfd = proto.fileno()
                pid = os.fork()
                if pid == 0:
                    try:
                        os.write(pfd, b"SPOOFED\\n")
                    except OSError:
                        pass
                    os._exit(0)
                os.waitpid(pid, 0)
                proto.write(b"LEGIT\\n")
                proto.flush()
                """
            )
        )
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, timeout=30
        )
        assert result.stdout == b"LEGIT\n", result.stdout
