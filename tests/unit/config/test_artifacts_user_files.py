# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ArtifactsConfig.user_files: wired in, surviving load, not rendered at load time."""

from aiperf.config.loader import load_config_from_string

# Mirror tests/unit/config/test_variables_persist.py: BenchmarkConfig requires
# datasets+phases (min_length=1), so prepend test-specific YAML to this base.
_BASE_YAML = """
benchmark:
  models:
    - test/model
  endpoint:
    type: chat
    urls: ["http://localhost:8000"]
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
"""


def test_user_files_default_empty():
    config = load_config_from_string(_BASE_YAML)
    assert config.benchmark.artifacts.user_files == []


def test_user_files_round_trips_through_config_load():
    yaml_str = """
variables:
  isl: 1024
benchmark:
  artifacts:
    user_files:
      - path: input_config.json
        format: json
        content:
          isl: "{{ isl }}"
          note: "fixed string"
  models:
    - test/model
  endpoint:
    type: chat
    urls: ["http://localhost:8000"]
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
"""
    config = load_config_from_string(yaml_str)
    assert len(config.benchmark.artifacts.user_files) == 1
    entry = config.benchmark.artifacts.user_files[0]
    assert entry.path == "input_config.json"
    assert entry.format == "json"
    # SKIP_TEMPLATE_PATH_PREFIXES includes ``benchmark.artifacts.user_files``,
    # so content survives the loader verbatim and is rendered at run-time
    # (with epoch/job_name/endpoint_url/... in scope) by materialize_user_files.
    assert entry.content == {"isl": "{{ isl }}", "note": "fixed string"}
