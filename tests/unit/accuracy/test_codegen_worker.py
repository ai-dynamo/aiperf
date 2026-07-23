from __future__ import annotations

from typing import Any

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
