# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: artifacts.user_files render and write during a real aiperf profile run.

Drives the in-process AIPerf CLI runner with a YAML config that contains an
``artifacts.user_files`` block, then asserts the rendered files materialize
under the run directory with structured types preserved (int as int) for JSON
and verbatim text for the text format. A second test asserts that an undefined
variable in user_files content aborts the run with a clear error.
"""

from pathlib import Path

import orjson
import pytest
import yaml

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


def _write_config(
    config_path: Path,
    artifact_dir: Path,
    *,
    extra_user_files_yaml: str = "",
) -> None:
    """Write a minimum-runnable AIPerf config YAML with an artifacts.user_files block.

    The config drives the same FakeTransport-backed run as other component-integration
    tests (no real HTTP) and pins ``artifacts.dir`` so the run-time materializer
    writes into a known location.
    """
    config_path.write_text(
        f"""
variables:
  isl: 1024
  osl: 512
models:
  - {defaults.model}
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
artifacts:
  dir: {artifact_dir}
  user_files:
    - path: input_config.json
      format: json
      content:
        isl: "{{{{ isl }}}}"
        osl: "{{{{ osl }}}}"
        endpoint: "{{{{ endpoint_url }}}}"
        model: "{{{{ model }}}}"
    - path: notes.txt
      content: |
        run for {{{{ model }}}}
        isl={{{{ isl }}}} osl={{{{ osl }}}}
{extra_user_files_yaml}
datasets:
  default:
    type: synthetic
    entries: 5
    prompts:
      isl: 8
      osl: 4
phases:
  - name: default
    type: concurrency
    requests: 3
    concurrency: 1
runtime:
  ui: {defaults.ui}
"""
    )


@pytest.mark.component_integration
def test_user_files_e2e_renders_json_and_text(tmp_path: Path, cli: AIPerfCLI) -> None:
    """A config with two user_files entries produces correctly-rendered files.

    Asserts:
    - JSON file exists with structured types preserved (int 1024, not "1024")
    - Text file exists with the templated model name interpolated verbatim
    - Both land directly under ``artifacts.dir`` (the run dir for local CLI runs)
    """
    config_path = tmp_path / "config.yaml"
    artifact_dir = tmp_path / "aiperf_output"
    _write_config(config_path, artifact_dir)

    cli.run_sync(
        f"aiperf profile --config {config_path}",
        timeout=defaults.timeout,
    )

    # Local-CLI layout: run dir IS artifact_dir (no <ns>/<name>/<epoch> nesting).
    json_path = artifact_dir / "input_config.json"
    notes_path = artifact_dir / "notes.txt"
    assert json_path.exists(), (
        f"missing {json_path}; tree: {list(artifact_dir.iterdir())}"
    )
    assert notes_path.exists(), f"missing {notes_path}"

    data = orjson.loads(json_path.read_bytes())
    # Structured types must survive: "{{ isl }}" with isl=1024 renders to int 1024,
    # NOT "1024". This is what _coerce_scalar in user_files.py guarantees for json/yaml.
    assert data == {
        "isl": 1024,
        "osl": 512,
        "endpoint": "http://localhost:8000",
        "model": defaults.model,
    }
    assert isinstance(data["isl"], int)
    assert isinstance(data["osl"], int)

    notes = notes_path.read_text()
    assert defaults.model in notes
    assert "isl=1024 osl=512" in notes


@pytest.mark.component_integration
def test_user_files_missing_variable_aborts_run(tmp_path: Path, cli: AIPerfCLI) -> None:
    """An undefined jinja2 variable in user_files content aborts the run with a clear error.

    StrictUndefined in user_files.py raises UserFileError naming the offending path
    and the missing variable; exit_on_error catches it and exits non-zero.
    """
    config_path = tmp_path / "config.yaml"
    artifact_dir = tmp_path / "aiperf_output"
    extra = """    - path: bad.json
      format: json
      content:
        x: "{{ does_not_exist }}"
"""
    _write_config(config_path, artifact_dir, extra_user_files_yaml=extra)

    result = cli.run_sync(
        f"aiperf profile --config {config_path}",
        timeout=defaults.timeout,
        assert_success=False,
    )

    assert result.exit_code != 0, (
        f"expected non-zero exit; got {result.exit_code}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "does_not_exist" in combined, (
        f"expected 'does_not_exist' in output; got:\n{combined}"
    )
    # The offending path should be named so the user can find it in their config.
    assert "bad.json" in combined, (
        f"expected 'bad.json' in error output; got:\n{combined}"
    )

    # Fail-fast: render error must abort BEFORE benchmark output exists.
    forbidden = {"profile_export.json", "records", "checkpoints"}
    if artifact_dir.exists():
        children_recursive = {p.name for p in artifact_dir.rglob("*")}
        leaked = children_recursive & forbidden
        assert not leaked, (
            f"benchmark output materialized despite render failure: {leaked}"
        )


@pytest.mark.component_integration
def test_user_files_e2e_renders_yaml(tmp_path: Path, cli: AIPerfCLI) -> None:
    """`format: yaml` renders structured content via PyYAML safe_dump.

    Asserts the file round-trips through ``yaml.safe_load`` to the originally
    declared structure with rendered scalar coercion (int stays int, not "1024").
    """
    config_path = tmp_path / "config.yaml"
    artifact_dir = tmp_path / "aiperf_output"
    config_path.write_text(
        f"""
variables:
  isl: 1024
  osl: 512
models:
  - {defaults.model}
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
artifacts:
  dir: {artifact_dir}
  user_files:
    - path: deployment.yaml
      format: yaml
      content:
        gpu_count: 8
        isl: "{{{{ isl }}}}"
        osl: "{{{{ osl }}}}"
        endpoint: "{{{{ endpoint_url }}}}"
        nested:
          model: "{{{{ model }}}}"
datasets:
  default:
    type: synthetic
    entries: 5
    prompts:
      isl: 8
      osl: 4
phases:
  - name: default
    type: concurrency
    requests: 3
    concurrency: 1
runtime:
  ui: {defaults.ui}
"""
    )

    cli.run_sync(
        f"aiperf profile --config {config_path}",
        timeout=defaults.timeout,
    )

    yaml_path = artifact_dir / "deployment.yaml"
    assert yaml_path.exists(), f"missing {yaml_path}"

    data = yaml.safe_load(yaml_path.read_text())
    assert data == {
        "gpu_count": 8,
        "isl": 1024,
        "osl": 512,
        "endpoint": "http://localhost:8000",
        "nested": {"model": defaults.model},
    }
    assert isinstance(data["isl"], int)
    assert isinstance(data["nested"]["model"], str)
