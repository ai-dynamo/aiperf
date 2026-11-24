# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import textwrap

import pytest
from cyclopts import App

from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config import BenchmarkRun
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.flags._converter_endpoint import build_endpoint
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.loader import build_benchmark_plan, load_config_from_string
from aiperf.plugin.enums import EndpointType


def _parse_cli_args(argv: list[str]) -> CLIConfig:
    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:
        captured["config"] = cli_config

    try:
        app(
            [
                "--url",
                "http://localhost:8000",
                "--model",
                "test-model",
                *argv,
            ],
            exit_on_error=False,
        )
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["config"]


def test_allow_empty_content_cli_defaults_false_and_unset() -> None:
    config = _parse_cli_args([])

    assert config.allow_empty_content is False
    assert "allow_empty_content" not in config.model_fields_set


def test_allow_empty_content_cli_maps_to_endpoint() -> None:
    config = _parse_cli_args(["--allow-empty-content"])

    assert config.allow_empty_content is True
    assert build_endpoint(config)["allow_empty_content"] is True


def test_allow_empty_content_unset_is_not_emitted_by_converter() -> None:
    assert "allow_empty_content" not in build_endpoint(CLIConfig())


def test_allow_empty_content_yaml_reaches_runtime_endpoint() -> None:
    config = load_config_from_string(
        textwrap.dedent(
            """\
            benchmark:
              models: [test-model]
              endpoint:
                urls: [http://localhost:8000/v1/chat/completions]
                type: chat
                streaming: true
                allow_empty_content: true
              datasets:
                - name: default
                  type: synthetic
              phases:
                - name: profiling
                  type: concurrency
                  concurrency: 1
                  requests: 1
            """
        ),
        substitute_env=True,
    )
    plan = build_benchmark_plan(config)
    cfg = plan.configs[0]
    run = BenchmarkRun(
        benchmark_id="test-run",
        sweep_id=plan.sweep_id,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )

    assert run.cfg.endpoint.allow_empty_content is True
    assert ModelEndpointInfo.from_run(run).endpoint.allow_empty_content is True


@pytest.mark.parametrize(
    ("streaming", "endpoint_type", "warns"),
    [
        (True, EndpointType.CHAT, False),
        (False, EndpointType.CHAT, True),
        (True, EndpointType.COMPLETIONS, True),
    ],
)  # fmt: skip
def test_allow_empty_content_warns_when_inactive(
    caplog: pytest.LogCaptureFixture,
    streaming: bool,
    endpoint_type: EndpointType,
    warns: bool,
) -> None:
    with caplog.at_level(logging.WARNING, logger="aiperf.config.endpoint"):
        EndpointConfig(
            urls=["http://localhost:8000"],
            type=endpoint_type,
            streaming=streaming,
            allow_empty_content=True,
        )

    assert ("--allow-empty-content has no effect" in caplog.text) is warns
