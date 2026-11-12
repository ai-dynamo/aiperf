# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-URL headers: parsing, CLI and YAML routing, and endpoint validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import orjson
import pytest
from cyclopts import App
from pydantic import ValidationError
from pytest import param

from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config import BenchmarkRun
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.flags._converter_endpoint import build_endpoint
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan, load_config_from_string
from aiperf.config.loader.parsing import MAX_URL_HEADER_INDEX, parse_url_headers
from tests.unit.conftest import make_cli_with_fields_set

DP_URL = "http://localhost:8000"


def _parse_cli_args(argv: list[str]) -> CLIConfig:
    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:
        captured["config"] = cli_config

    try:
        app(["--model", "test-model", *argv], exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["config"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        param(None, None, id="none"),
        param("0:X-data-parallel-rank:0", [{"X-data-parallel-rank": "0"}], id="cli-form"),
        param("2:X-Route:a", [{}, {}, {"X-Route": "a"}], id="sparse-index-pads"),
        param(" 1 : X-Route : a:b:c ", [{}, {"X-Route": "a:b:c"}], id="value-keeps-colons"),
        param('{"1": {"X-Route": "a"}}', [{}, {"X-Route": "a"}], id="json-index-mapping"),
        param('[{}, {"X-Route": "a"}]', [{}, {"X-Route": "a"}], id="json-list"),
        param([{"X-Route": 7}, {}], [{"X-Route": "7"}, {}], id="yaml-list-stringified"),
        param({"1": {"X-Route": "b"}}, [{}, {"X-Route": "b"}], id="yaml-index-mapping"),
        param(
            ["0:X-Route:a", "0:X-Pool:p", "1:X-Route:b", "0:X-Route:c"],
            [{"X-Route": "c", "X-Pool": "p"}, {"X-Route": "b"}],
            id="repeated-flags-merge-last-wins",
        ),
        param(
            ["0:X-Route:a", "0:x-route:b", "0:X-ROUTE:c"],
            [{"X-ROUTE": "c"}],
            id="repeated-names-replace-case-insensitively",
        ),
        param(
            {"0": {"X-Route": "a", "x-route": "b"}},
            [{"x-route": "b"}],
            id="mapping-names-replace-case-insensitively",
        ),
    ],
)  # fmt: skip
def test_parse_url_headers_accepted_forms(raw: object, expected: object) -> None:
    assert parse_url_headers(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        param("X-Route:a", id="missing-index"),
        param("0:X-Route", id="missing-value"),
        param("0::a", id="empty-name"),
        param("-1:X-Route:a", id="negative-index"),
        param("first:X-Route:a", id="non-integer-index"),
        param({"x": {"X-Route": "a"}}, id="non-integer-mapping-key"),
        param({"0": "X-Route:a"}, id="headers-not-a-mapping"),
        param("[1, 2]", id="json-list-of-non-mappings"),
        param("{not json", id="bad-json"),
        param('"0:X:a"', id="quoted-index"),
        param(3, id="unsupported-type"),
        param("1000000000:X-Route:a", id="index-above-cap"),
    ],
)  # fmt: skip
def test_parse_url_headers_invalid_raises_value_error(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_url_headers(raw)


def test_url_headers_cli_flag_maps_to_endpoint() -> None:
    config = _parse_cli_args(
        [
            "--url", DP_URL,
            "--url", DP_URL,
            "--url-header", "0:X-data-parallel-rank:0",
            "--url-header", "1:X-data-parallel-rank:1",
        ]
    )  # fmt: skip

    endpoint = build_endpoint(config)

    assert endpoint["urls"] == [DP_URL, DP_URL]
    assert endpoint["url_headers"] == [
        {"X-data-parallel-rank": "0"},
        {"X-data-parallel-rank": "1"},
    ]
    assert EndpointConfig(**endpoint).url_headers == endpoint["url_headers"]


def test_url_headers_unset_is_not_emitted_by_converter() -> None:
    config = _parse_cli_args(["--url", DP_URL])

    assert "url_headers" not in config.model_fields_set
    assert "url_headers" not in build_endpoint(config)


def test_url_headers_index_out_of_range_raises() -> None:
    with pytest.raises(ValidationError, match="URL index 2 is out of range"):
        EndpointConfig(urls=[DP_URL, DP_URL], url_headers=["2:X-Route:a"])


@pytest.mark.parametrize(
    "url_headers",
    [
        param(["1000000000:X-Route:a"], id="cli-form"),
        param({"1000000000": {"X-Route": "a"}}, id="index-mapping"),
    ],
)  # fmt: skip
def test_url_headers_huge_index_is_rejected_before_padding(url_headers: object) -> None:
    with pytest.raises(ValidationError, match="URL index 1000000000 is out of range"):
        EndpointConfig(urls=[DP_URL, DP_URL], url_headers=url_headers)


def test_parse_url_headers_checks_url_count_before_padding() -> None:
    with pytest.raises(ValueError, match="URL index 5 is out of range"):
        parse_url_headers("5:X-Route:a", url_count=2)
    assert parse_url_headers("1:X-Route:a", url_count=2) == [{}, {"X-Route": "a"}]


def test_parse_url_headers_known_url_count_is_the_only_bound() -> None:
    # with the url count known, the fixed cap does not apply on top of it
    url_count = MAX_URL_HEADER_INDEX + 2
    aligned = [{} for _ in range(url_count - 1)] + [{"X-Route": "last"}]
    parsed = parse_url_headers(aligned, url_count=url_count)
    assert parsed is not None
    assert len(parsed) == url_count
    assert parsed[-1] == {"X-Route": "last"}
    last = url_count - 1
    assert parse_url_headers(f"{last}:X-Route:a", url_count=url_count)[-1] == {
        "X-Route": "a"
    }
    # without a url count the fixed cap still guards padding
    with pytest.raises(ValueError, match="maximum"):
        parse_url_headers(f"{MAX_URL_HEADER_INDEX + 1}:X-Route:a")


@pytest.mark.parametrize(
    "name",
    [
        param("Authorization", id="authorization"),
        param("x-api-key", id="api-key"),
        param("X-Tenant-Token", id="token-substring"),
    ],
)  # fmt: skip
def test_url_headers_credential_names_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="credential-bearing"):
        EndpointConfig(urls=[DP_URL], url_headers=[{name: "secret"}])


def test_url_headers_survive_json_round_trip() -> None:
    """Serialized runs reach worker processes as JSON."""
    original = EndpointConfig(urls=[DP_URL, DP_URL], url_headers=["1:X-Route:b"])

    payload = orjson.loads(orjson.dumps(original.model_dump(mode="json")))
    restored = EndpointConfig.model_validate(payload)

    assert restored.url_headers == [{}, {"X-Route": "b"}]


def test_url_headers_yaml_reaches_runtime_endpoint() -> None:
    config = load_config_from_string(
        textwrap.dedent(
            """\
            benchmark:
              models: [test-model]
              endpoint:
                urls: [http://localhost:8000, http://localhost:8000]
                type: chat
                headers:
                  X-Team: perf
                url_headers:
                  - {X-data-parallel-rank: "0"}
                  - {X-data-parallel-rank: "1"}
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
    assert len(plan.configs) == 1
    cfg = plan.configs[0]
    run = BenchmarkRun(
        benchmark_id="test-run",
        sweep_id=plan.sweep_id,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )

    info = ModelEndpointInfo.from_run(run).endpoint
    assert info.get_headers(0) == {"X-Team": "perf", "X-data-parallel-rank": "0"}
    assert info.get_headers(1) == {"X-Team": "perf", "X-data-parallel-rank": "1"}


def test_url_headers_cli_overrides_config_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """\
            schemaVersion: "2.0"
            benchmark:
              model: test-model
              endpoint:
                urls: [http://localhost:8000, http://localhost:8000]
                url_headers:
                  - {X-data-parallel-rank: "7"}
              dataset:
                type: synthetic
              phases:
                type: concurrency
                concurrency: 1
                requests: 1
            """
        )
    )

    resolved = resolve_config(
        make_cli_with_fields_set(
            url_headers=["0:X-data-parallel-rank:0", "1:X-data-parallel-rank:1"]
        ),
        config_file,
    )

    assert resolved.benchmark.endpoint.url_headers == [
        {"X-data-parallel-rank": "0"},
        {"X-data-parallel-rank": "1"},
    ]
    assert resolved.benchmark.endpoint.urls == [DP_URL, DP_URL]
