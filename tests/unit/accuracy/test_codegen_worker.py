from __future__ import annotations

import io
from typing import Any

import orjson

from aiperf.accuracy.graders import _codegen_worker as worker


def _fake_codegen_ok(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
    return {"pass@1": 1.0}, {}


def _fake_codegen_boom(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
    raise RuntimeError("sandbox exploded")


class TestHandleRequest:
    def test_ok_request_returns_metrics_with_id(self) -> None:
        req = {
            "id": 7,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }
        resp = worker.handle_request(req, _fake_codegen_ok)
        assert resp == {"id": 7, "ok": True, "metrics": {"pass@1": 1.0}}

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
