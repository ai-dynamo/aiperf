# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from aiperf.common.models import TextResponseData
from aiperf.endpoints.response_mixin import JMESPathResponseMixin


class _StubMixinHost(JMESPathResponseMixin):
    """Minimal subclass that supplies the attributes the mixin reads.

    The mixin reads ``self.model_endpoint.endpoint.extra`` for the
    ``response_field`` config and uses ``self.info`` / ``self.error`` /
    ``self.warning`` for logging. Stub them all.
    """

    def __init__(self, response_field: str | None):
        extra = [("response_field", response_field)] if response_field else []
        self.model_endpoint = MagicMock()
        self.model_endpoint.endpoint.extra = extra
        self.logged_info: list[str] = []
        self.logged_error: list[str] = []
        self.logged_warning: list[str] = []
        self._init_response_parser()

    def info(self, msg: str) -> None:
        self.logged_info.append(msg)

    def error(self, msg: str) -> None:
        self.logged_error.append(msg)

    def warning(self, msg: str) -> None:
        self.logged_warning.append(msg)

    def auto_detect_and_extract(self, json_obj):
        if isinstance(json_obj, dict) and "auto" in json_obj:
            return TextResponseData(text=json_obj["auto"])
        return None

    def make_text_response_data(self, text: str):
        return TextResponseData(text=text)

    def convert_to_response_data(self, value):
        return TextResponseData(text=str(value))


class _StubResponse:
    """Concrete stub implementing the InferenceServerResponse protocol."""

    def __init__(
        self,
        perf_ns: int,
        text: str | None = None,
        json_obj=None,
    ):
        self.perf_ns = perf_ns
        self._text = text
        self._json = json_obj

    def get_raw(self):
        return self._text

    def get_text(self):
        return self._text

    def get_json(self):
        return self._json


def _build_response(payload: bytes | str | None, json_obj=None):
    text = payload.decode() if isinstance(payload, bytes) else payload
    return _StubResponse(perf_ns=42, text=text, json_obj=json_obj)


class TestJMESPathResponseMixinCompile:
    def test_no_response_field_compiles_to_none(self):
        host = _StubMixinHost(response_field=None)
        assert host._compiled_jmespath is None
        assert all("Compiled JMESPath" not in m for m in host.logged_info)

    def test_valid_response_field_compiles(self):
        host = _StubMixinHost(response_field="result.text")
        assert host._compiled_jmespath is not None
        assert any("Compiled JMESPath query" in m for m in host.logged_info)

    def test_malformed_response_field_logs_and_falls_back(self):
        host = _StubMixinHost(response_field="!!! not valid jmespath !!!")
        assert host._compiled_jmespath is None
        assert host.logged_error, "Expected an error log on compile failure"
        assert any("auto-detect" in m.lower() for m in host.logged_error), (
            f"Expected auto-detect mention; got logs={host.logged_error!r}"
        )


class TestJMESPathResponseMixinParse:
    def test_falls_back_to_auto_detect_on_search_failure(self):
        host = _StubMixinHost(response_field="result.text")
        r = _build_response(payload=b'{"auto":"hello"}', json_obj={"auto": "hello"})
        parsed = host.parse_response(r)
        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "hello"

    def test_jmespath_match_wins(self):
        host = _StubMixinHost(response_field="result.text")
        r = _build_response(
            payload=b'{"result":{"text":"jp"}}',
            json_obj={"result": {"text": "jp"}},
        )
        parsed = host.parse_response(r)
        assert parsed is not None
        assert parsed.data.text == "jp"
