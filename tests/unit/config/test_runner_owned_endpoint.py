# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint IDs stay portable until the selected native runner preflight."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.config import (
    AIPerfConfig,
    BenchmarkRun,
    EndpointConfig,
    build_benchmark_plan,
    load_config_from_string,
)
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.orchestrator.rust_wire import build_authored_run_request


def _authored_run(config: AIPerfConfig, tmp_path: Path) -> dict:
    """Project one config through the strict protocol-v2 authored builder."""
    return build_authored_run_request(
        _run_from_config(config, tmp_path),
        operation="execute",
    )


def _run_from_config(config: AIPerfConfig, tmp_path: Path) -> BenchmarkRun:
    plan = build_benchmark_plan(config)
    return BenchmarkRun(
        benchmark_id="runner-owned-endpoint",
        cfg=plan.configs[0],
        variation=plan.variations[0],
        artifact_dir=tmp_path,
        random_seed=7,
    )


def test_messages_yaml_reaches_native_wire_and_artifacts_redact_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-secret")
    monkeypatch.setenv("ANTHROPIC_HEADER_KEY", "sk-ant-header-secret")
    config = load_config_from_string(
        textwrap.dedent(
            f"""\
            benchmark:
              models: [mock-model]
              endpoint:
                urls: [https://api.anthropic.com]
                type: " messages "
                path: /v1/messages
                streaming: true
                api_key: ${{ANTHROPIC_API_KEY}}
                headers:
                  x-api-key: ${{ANTHROPIC_HEADER_KEY}}
                  anthropic-version: "2023-06-01"
              datasets:
                - name: main
                  type: synthetic
                  entries: 1
                  prompts:
                    isl: 8
                    osl: 2
              phases:
                - name: profiling
                  type: concurrency
                  requests: 1
                  concurrency: 1
              tokenizer:
                name: builtin
              gpu_telemetry:
                enabled: false
              server_metrics:
                enabled: false
              artifacts:
                dir: {tmp_path}
            """
        )
    )

    assert config.benchmark.endpoint.type == "messages"
    exported = config.model_dump_json()
    assert "sk-ant-api-secret" not in exported
    assert "sk-ant-header-secret" not in exported
    assert exported.count("<redacted>") >= 2

    endpoint = _authored_run(config, tmp_path)["run"]["cfg"]["endpoint"]
    assert endpoint["type"] == "messages"
    assert endpoint["path"] == "/v1/messages"
    assert endpoint["streaming"] is True
    # Credentials intentionally cross only the in-memory/stdin runner ABI.
    assert endpoint["api_key"] == "sk-ant-api-secret"
    assert endpoint["headers"] == {
        "x-api-key": "sk-ant-header-secret",
        "anthropic-version": "2023-06-01",
    }


def test_custom_cli_endpoint_id_passes_through_without_python_registration(
    tmp_path: Path,
) -> None:
    config = resolve_config(
        CLIConfig(
            model_names=["mock-model"],
            endpoint_type=" acme_chat ",
            concurrency=1,
            request_count=1,
            tokenizer_name="builtin",
            no_gpu_telemetry=True,
            no_server_metrics=True,
            artifact_directory=tmp_path,
        )
    )

    assert config.benchmark.endpoint.type == "acme_chat"
    request = _authored_run(config, tmp_path)
    assert request["run"]["cfg"]["endpoint"]["type"] == "acme_chat"


def test_custom_endpoint_id_with_template_is_preserved_structurally() -> None:
    endpoint = EndpointConfig(
        urls=["http://localhost:8000"],
        type="acme_chat",
        template={"body": '{"prompt": "{{ prompt }}"}'},
    )

    assert endpoint.type == "acme_chat"
    assert endpoint.template is not None
    assert endpoint.template.body == '{"prompt": "{{ prompt }}"}'


def test_dag_jsonl_format_and_rows_pass_to_runner_without_linearization(
    tmp_path: Path,
) -> None:
    authored_rows = [
        {
            "session_id": "root",
            "turns": [
                {
                    "timestamp": 17,
                    "messages": [{"role": "user", "content": "root turn"}],
                    "forks": ["child"],
                    "opaque_extension": {"preserve": [1, 2, 3]},
                }
            ],
        },
        {
            "session_id": "child",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "child turn"}],
                    "spawns": [{"children": ["grandchild"]}],
                }
            ],
        },
    ]
    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
                "datasets": [
                    {
                        "name": "graph",
                        "type": "file",
                        "format": "dag_jsonl",
                        "records": authored_rows,
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 2,
                        "concurrency": 1,
                    }
                ],
                "tokenizer": {"name": "builtin"},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
                "artifacts": {"dir": str(tmp_path)},
            }
        }
    )

    dataset = _authored_run(config, tmp_path)["run"]["cfg"]["datasets"][0]

    assert dataset["format"] == "dag_jsonl"
    assert dataset["records"] == authored_rows


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_endpoint_id_rejects_empty_normalized_strings(value: str) -> None:
    with pytest.raises(ValidationError, match="at least 1|non-empty string"):
        EndpointConfig(urls=["http://localhost:8000"], type=value)
    with pytest.raises(ValidationError, match="at least 1|non-empty string"):
        CLIConfig(endpoint_type=value)


def test_config_and_cli_construction_do_not_resolve_a_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-execution config paths must not discover a runner")

    monkeypatch.setattr(
        "aiperf.orchestrator.native_execution.resolve_native_binary", fail_if_called
    )

    endpoint = EndpointConfig(
        urls=["http://localhost:8000"],
        type="messages",
    )
    cli = CLIConfig(endpoint_type="future_compiled_endpoint")
    schema = AIPerfConfig.model_json_schema()

    assert endpoint.type == "messages"
    assert cli.endpoint_type == "future_compiled_endpoint"
    assert schema["$defs"]["EndpointConfig"]["properties"]["type"]["type"] == ("string")
