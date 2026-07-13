# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `aiperf config init` CLI command."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from aiperf.cli_commands.config import expand, init, validate
from aiperf.orchestrator.rust_wire import build_authored_run_request

_VALID_BENCHMARK = textwrap.dedent("""\
benchmark:
  models: [llama]
  endpoint:
    urls: ["http://x:8000/v1/chat/completions"]
  datasets:
    - name: main
      type: synthetic
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
""")


class _RecordingRunnerInstallation:
    """Exact-image stand-in that records authored validation without I/O."""

    def __init__(
        self,
        *,
        catalog: dict | None = None,
        accept_preflight: bool = True,
    ):
        self.capabilities = catalog or {
            "schema_version": "1.0",
            "endpoint": {"chat": {}},
            "transport": {"http": {}},
            "synthetic": {"synthetic": {}},
        }
        self.accept_preflight = accept_preflight
        self.projected_runs = []
        self.validated_runs = []
        self.validation_error: Exception | None = None

    def project_authored_request(self, run, *, operation: str):
        self.projected_runs.append(run)
        return build_authored_run_request(
            run,
            operation=operation,
        )

    def preflight_request(self, request):
        if not self.accept_preflight:
            raise RuntimeError("catalog does not advertise requested ids")

    def supports_pair(self, backend_id: str, workload_id: str) -> bool:
        del backend_id, workload_id
        return False

    def validate_authored_run(self, run):
        self.validated_runs.append(run)
        if self.validation_error is not None:
            raise self.validation_error
        return {"success": True}

    def validate_authored_request(self, request, *, benchmark_id: str):
        assert request["operation"] == "validate"
        assert request["run"]["benchmark_id"] == benchmark_id
        self.validated_runs.append(self.projected_runs[-1])
        if self.validation_error is not None:
            raise self.validation_error
        return {"success": True}


def _select_runner(
    monkeypatch: pytest.MonkeyPatch,
    installation: _RecordingRunnerInstallation,
) -> list[None]:
    resolutions: list[None] = []

    def resolve():
        resolutions.append(None)
        return installation

    monkeypatch.setattr(
        "aiperf.orchestrator.runner_installation.RunnerInstallation.resolve",
        resolve,
    )
    return resolutions


def _write_grid_sweep(tmp_path: Path) -> Path:
    """Write a minimal grid-sweep YAML to ``tmp_path/sweep.yaml``."""
    yaml_str = (
        "sweep:\n"
        "  type: grid\n"
        "  parameters:\n"
        "    phases.profiling.concurrency: [1, 2, 4]\n" + _VALID_BENCHMARK
    )
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml_str, encoding="utf-8")
    return path


def _write_adaptive_sweep(tmp_path: Path) -> Path:
    yaml_str = (
        "sweep:\n"
        "  type: adaptive_search\n"
        "  planner: bayesian\n"
        "  max_iterations: 5\n"
        "  n_initial_points: 2\n"
        "  search_space:\n"
        "    - {path: phases.profiling.concurrency, kind: int, lo: 1, hi: 16}\n"
        "  objectives:\n"
        "    - {metric: output_token_throughput, direction: MAXIMIZE}\n"
        + _VALID_BENCHMARK
    )
    path = tmp_path / "adaptive.yaml"
    path.write_text(yaml_str, encoding="utf-8")
    return path


class TestConfigInit:
    """End-to-end behavior of `aiperf config init`."""

    def test_list_prints_categories(self, capsys: pytest.CaptureFixture[str]) -> None:
        init(list_templates=True)
        out = capsys.readouterr().out
        assert "Getting Started" in out
        assert "minimal" in out

    def test_list_with_category_filter(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        init(list_templates=True, category="Load Testing")
        out = capsys.readouterr().out
        assert "goodput_slo" in out
        assert "minimal" not in out  # Getting Started, filtered out

    def test_search_filters_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        init(search="goodput")
        out = capsys.readouterr().out
        assert "goodput_slo" in out

    def test_search_no_match_prints_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        init(search="zzz_no_such_template")
        out = capsys.readouterr().out
        assert "No templates match" in out
        assert "aiperf config init --list" in out

    def test_template_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        init(template="minimal")
        out = capsys.readouterr().out
        assert "title: Minimal Configuration" in out
        assert "model:" in out
        assert "SPDX-" not in out  # SPDX header is stripped

    def test_template_with_overrides_writes_to_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.yaml"
        init(
            template="minimal",
            model="my-llama",
            url="http://svc:9000",
            output=output,
        )
        content = output.read_text(encoding="utf-8")
        assert "my-llama" in content
        assert "http://svc:9000" in content
        assert "SPDX-" not in content

    def test_unknown_template_exits_nonzero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            init(template="zzz_no_such_template")
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Unknown template" in err

    def test_no_args_prints_hint_and_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            init()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--template" in err
        assert "--list" in err


class TestConfigExpand:
    """End-to-end behavior of `aiperf config expand`."""

    def test_text_format_lists_variations(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        expand(path)
        out = capsys.readouterr().out
        assert "GridSweep" in out
        assert "3 variations" in out
        for value in (1, 2, 4):
            assert f"concurrency_{value}" in out

    def test_full_includes_benchmark_body(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        expand(path, full=True)
        out = capsys.readouterr().out
        # YAML body of each variation must be inlined.
        assert "models:" in out
        assert out.count("phases:") >= 3

    def test_index_filters_to_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        expand(path, index=1)
        out = capsys.readouterr().out
        assert "[1]" in out
        assert "concurrency_2" in out
        # Other variations must not appear.
        assert "concurrency_1 " not in out  # trailing space avoids substring match
        assert "concurrency_4" not in out

    def test_index_out_of_range_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        with pytest.raises(SystemExit) as exc:
            expand(path, index=99)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "out of range" in err

    def test_json_format_is_parseable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        expand(path, fmt="json")
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["sweep_type"] == "GridSweep"
        assert len(payload["variations"]) == 3
        assert {v["dir_name"] for v in payload["variations"]} == {
            "concurrency_1",
            "concurrency_2",
            "concurrency_4",
        }
        # Without --full, body is omitted.
        assert "benchmark" not in payload["variations"][0]

    def test_json_full_includes_body(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        expand(path, full=True, fmt="json")
        payload = json.loads(capsys.readouterr().out)
        bodies = [v["benchmark"] for v in payload["variations"]]
        concurrencies = [b["phases"][0]["concurrency"] for b in bodies]
        assert concurrencies == [1, 2, 4]

    def test_yaml_format_is_parseable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_grid_sweep(tmp_path)
        expand(path, fmt="yaml")
        payload = yaml.safe_load(capsys.readouterr().out)
        assert payload["sweep_type"] == "GridSweep"
        assert len(payload["variations"]) == 3

    def test_missing_file_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(SystemExit) as exc:
            expand(missing)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    def test_adaptive_search_rejects_with_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_adaptive_sweep(tmp_path)
        with pytest.raises(SystemExit) as exc:
            expand(path)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "adaptive_search" in err
        assert "dynamically" in err


class TestConfigValidate:
    """End-to-end behavior of `aiperf config validate`."""

    def test_clean_config_reports_valid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "benchmark.yaml"
        path.write_text(_VALID_BENCHMARK, encoding="utf-8")
        validate(path)
        captured = capsys.readouterr()
        assert "Configuration valid" in captured.out
        assert "warning" not in captured.out
        assert captured.err == ""

    def test_warnings_print_to_stderr_but_exit_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `slos.time_to_first_token` without streaming triggers a warning in
        # validate_config_file without raising.
        yaml_str = textwrap.dedent("""\
            benchmark:
              models: [llama]
              endpoint:
                urls: ["http://x:8000/v1/chat/completions"]
                streaming: false
              datasets:
                - name: main
                  type: synthetic
              phases:
                - name: profiling
                  type: concurrency
                  requests: 10
                  concurrency: 1
              slos:
                time_to_first_token: 100
            """)
        path = tmp_path / "benchmark.yaml"
        path.write_text(yaml_str, encoding="utf-8")
        validate(path)
        captured = capsys.readouterr()
        assert "Warning:" in captured.err
        assert "time_to_first_token" in captured.err
        assert "Configuration valid with 1 warning" in captured.out

    def test_missing_file_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(SystemExit) as exc:
            validate(missing)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Error" in err

    def test_invalid_config_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Missing required `phases` block under benchmark.
        path = tmp_path / "broken.yaml"
        path.write_text(
            "benchmark:\n  models: [llama]\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            validate(path)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Error" in err

    def test_slos_streaming_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # SLO on time_to_first_token without streaming -> warning.
        yaml_str = textwrap.dedent("""\
            benchmark:
              models: [llama]
              endpoint:
                urls: ["http://x:8000/v1/chat/completions"]
              datasets:
                - name: main
                  type: synthetic
              phases:
                - name: profiling
                  type: concurrency
                  requests: 10
                  concurrency: 1
              slos:
                time_to_first_token: 250
            """)
        path = tmp_path / "ttft_no_stream.yaml"
        path.write_text(yaml_str, encoding="utf-8")
        validate(path)
        captured = capsys.readouterr()
        assert "time_to_first_token" in captured.err
        assert "streaming" in captured.err
        assert "Configuration valid with 1 warning" in captured.out

    def test_unknown_slo_metric_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Typo'd SLO metric name fails fast against the metric registry.
        yaml_str = textwrap.dedent("""\
            benchmark:
              models: [llama]
              endpoint:
                urls: ["http://x:8000/v1/chat/completions"]
                streaming: true
              datasets:
                - name: main
                  type: synthetic
              phases:
                - name: profiling
                  type: concurrency
                  requests: 10
                  concurrency: 1
              slos:
                tyme_to_furst_token: 250
            """)
        path = tmp_path / "typo_slo.yaml"
        path.write_text(yaml_str, encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            validate(path)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Unknown SLO metric" in err
        assert "tyme_to_furst_token" in err

    def test_native_validation_receives_every_expanded_variation_without_resolution(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        missing_dataset = tmp_path / "missing.jsonl"
        artifact_dir = tmp_path / "never-created"
        config_file = tmp_path / "sweep.yaml"
        config_file.write_text(
            textwrap.dedent(f"""\
                variables:
                  concurrency: 1
                sweep:
                  type: grid
                  parameters:
                    variables.concurrency: [1, 3, 5]
                benchmark:
                  models: [remote/model]
                  tokenizer:
                    name: remote/tokenizer
                  endpoint:
                    urls: ["http://x:8000/v1/chat/completions"]
                    streaming: true
                  datasets:
                    - name: main
                      type: file
                      path: {missing_dataset}
                      format: single_turn
                  phases:
                    - name: profiling
                      type: concurrency
                      requests: 10
                      concurrency: "{{{{ concurrency }}}}"
                  artifacts:
                    dir: {artifact_dir}
                  gpu_telemetry:
                    enabled: false
                  server_metrics:
                    enabled: false
                """),
            encoding="utf-8",
        )
        installation = _RecordingRunnerInstallation()
        resolutions = _select_runner(monkeypatch, installation)
        monkeypatch.setattr(
            "aiperf.config.resolution.resolvers.build_default_resolver_chain",
            lambda: pytest.fail("config validate entered the runtime resolver chain"),
        )

        validate(config_file)

        assert len(resolutions) == 1
        assert [
            run.cfg.phases[0].concurrency for run in installation.validated_runs
        ] == [1, 3, 5]
        assert [run.variation.values for run in installation.validated_runs] == [
            {"variables.concurrency": 1},
            {"variables.concurrency": 3},
            {"variables.concurrency": 5},
        ]
        assert len(installation.projected_runs) == 3
        assert not missing_dataset.exists()
        assert not artifact_dir.exists()
        assert "Configuration valid" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "installation",
        [
            _RecordingRunnerInstallation(catalog={"endpoint": {"chat": {}}}),
            _RecordingRunnerInstallation(accept_preflight=False),
        ],
        ids=["unusable-catalog", "catalog-miss"],
    )
    def test_native_validation_preserves_python_only_compatibility_without_pair(
        self,
        installation: _RecordingRunnerInstallation,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_file = tmp_path / "benchmark.yaml"
        config_file.write_text(_VALID_BENCHMARK, encoding="utf-8")
        _select_runner(monkeypatch, installation)

        validate(config_file)

        assert installation.validated_runs == []
        if "schema_version" not in installation.capabilities:
            assert installation.projected_runs == []
        else:
            assert len(installation.projected_runs) == 1
        assert "Configuration valid" in capsys.readouterr().out

    def test_native_validation_rejection_is_a_config_error(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_file = tmp_path / "benchmark.yaml"
        config_file.write_text(_VALID_BENCHMARK, encoding="utf-8")
        installation = _RecordingRunnerInstallation()
        installation.validation_error = RuntimeError("authored phase rejected")
        _select_runner(monkeypatch, installation)

        with pytest.raises(SystemExit) as exc:
            validate(config_file)

        assert exc.value.code == 1
        error = capsys.readouterr().err
        assert "Native runner validation failed" in error
        assert "authored phase rejected" in error
        assert "base" in error
