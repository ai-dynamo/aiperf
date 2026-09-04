# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for endpoint credential consumption and rehydration."""

import pytest

from aiperf.common.endpoint_credentials import (
    AIPERF_INJECTED_API_KEY,
    OPENAI_API_KEY,
    EndpointCredentialInjection,
    apply_endpoint_credentials,
    consume_endpoint_credentials,
)
from aiperf.common.redact import REDACTED_VALUE
from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun

_AMBIENT_KEY = "ambient-shell-value"
_INJECTED_KEY = "injected-transport-value"

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://my-internal-vllm:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 10,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
        }
    ],
}


def _benchmark_run(tmp_path, api_key: str | None) -> BenchmarkRun:
    """Build a minimal BenchmarkRun whose endpoint carries ``api_key``."""
    cfg = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
    cfg.endpoint.api_key = api_key
    return BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="run_0001",
    )


@pytest.fixture(autouse=True)
def _clean_credential_env(monkeypatch):
    monkeypatch.delenv(AIPERF_INJECTED_API_KEY, raising=False)
    monkeypatch.delenv(OPENAI_API_KEY, raising=False)


class TestConsumeEndpointCredentials:
    def test_consume_pops_openai_api_key_without_promoting_it(self, monkeypatch):
        """The alias is popped for hygiene but is not the private transport."""
        import os

        monkeypatch.setenv(OPENAI_API_KEY, _AMBIENT_KEY)

        credentials = consume_endpoint_credentials()

        assert OPENAI_API_KEY not in os.environ
        assert credentials.api_key == _AMBIENT_KEY
        assert credentials.api_key_from_alias is True

    def test_consume_private_transport_is_not_an_alias(self, monkeypatch):
        monkeypatch.setenv(AIPERF_INJECTED_API_KEY, _INJECTED_KEY)
        monkeypatch.setenv(OPENAI_API_KEY, _AMBIENT_KEY)

        credentials = consume_endpoint_credentials()

        assert credentials.api_key == _INJECTED_KEY
        assert credentials.api_key_from_alias is False


class TestApplyEndpointCredentials:
    def test_alias_never_fills_an_unset_api_key(self, tmp_path):
        """An ambient OPENAI_API_KEY must not attach to an unconfigured endpoint.

        The parent resolves ``endpoint.api_key`` and forwards it through
        ``AIPERF_INJECTED_API_KEY``; an unset key means the user configured
        none, so the shell variable has no authored value to rehydrate.
        """
        run = _benchmark_run(tmp_path, None)
        credentials = EndpointCredentialInjection(
            api_key=_AMBIENT_KEY,
            api_key_from_alias=True,
            headers=None,
            urls=None,
        )

        apply_endpoint_credentials(run, credentials)

        assert run.cfg.endpoint.api_key is None

    def test_alias_fills_a_redacted_api_key(self, tmp_path):
        """A redacted placeholder proves the user authored a key."""
        run = _benchmark_run(tmp_path, REDACTED_VALUE)
        credentials = EndpointCredentialInjection(
            api_key=_AMBIENT_KEY,
            api_key_from_alias=True,
            headers=None,
            urls=None,
        )

        apply_endpoint_credentials(run, credentials)

        assert run.cfg.endpoint.api_key == _AMBIENT_KEY

    def test_private_transport_fills_an_unset_api_key(self, tmp_path):
        """The private transport is authoritative: the parent set it deliberately."""
        run = _benchmark_run(tmp_path, None)
        credentials = EndpointCredentialInjection(
            api_key=_INJECTED_KEY,
            api_key_from_alias=False,
            headers=None,
            urls=None,
        )

        apply_endpoint_credentials(run, credentials)

        assert run.cfg.endpoint.api_key == _INJECTED_KEY

    def test_require_resolved_ignores_an_unset_api_key(self, tmp_path):
        """An endpoint with no configured key is fully resolved by definition."""
        run = _benchmark_run(tmp_path, None)
        credentials = EndpointCredentialInjection(
            api_key=_AMBIENT_KEY,
            api_key_from_alias=True,
            headers=None,
            urls=None,
        )

        apply_endpoint_credentials(run, credentials, require_resolved=True)

        assert run.cfg.endpoint.api_key is None


def _ws_benchmark_run(tmp_path) -> BenchmarkRun:
    """Build a BenchmarkRun on a credential-free ws:// Responses endpoint.

    The endpoint validates at construction (no credentials over ws://); the
    injection step is what tries to attach a secret afterward.
    """
    kwargs = {
        **_MINIMAL_CONFIG_KWARGS,
        "endpoint": {
            "urls": ["ws://my-internal-vllm:8000/v1/responses"],
            "type": "responses",
            "use_server_token_count": True,
        },
    }
    cfg = BenchmarkConfig(**kwargs)
    return BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="run_0001",
    )


def test_injected_key_onto_ws_endpoint_is_rejected(tmp_path):
    """Rehydrating a secret onto a ws:// endpoint would leak it in cleartext.

    The overlay mutates the endpoint in place, and BaseConfig has no
    validate_assignment, so the ws:// TLS gate must be re-asserted explicitly.
    """
    run = _ws_benchmark_run(tmp_path)
    credentials = EndpointCredentialInjection(
        api_key=_INJECTED_KEY,
        api_key_from_alias=False,
        headers=None,
        urls=None,
    )

    with pytest.raises(ValueError, match="unencrypted 'ws://'"):
        apply_endpoint_credentials(run, credentials)


def test_ambient_shell_key_does_not_reach_an_unconfigured_endpoint(
    tmp_path, monkeypatch
):
    """End-to-end consume+apply: the exposure the fix closes."""
    monkeypatch.setenv(OPENAI_API_KEY, _AMBIENT_KEY)
    run = _benchmark_run(tmp_path, None)

    apply_endpoint_credentials(run, consume_endpoint_credentials())

    assert run.cfg.endpoint.api_key is None
