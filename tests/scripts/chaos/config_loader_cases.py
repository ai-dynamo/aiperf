# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial cases for the schema-v2 YAML config loader and its subpackages.

Each case writes a hand-crafted YAML under ``ctx.fixtures`` and drives the
``aiperf config validate`` CLI (cheap, ~0.5s per case) or
``aiperf profile --config`` (where downstream wiring matters). The expectation
table covers:

* loader internals -- jinja rendering, env-var expansion, sweep dotted-path
  validation, parsing/normalizers
* subpackage Pydantic models -- sweep, dataset, comm, plot, resolution

Adversarial inputs that surface bugs are classified ``BUG_*`` by the harness
and triaged separately. This file does not fix surfaced bugs.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import orjson

from tests.scripts.chaos.harness import Case, Context, run_cmd


def _write(ctx: Context, name: str, body: str) -> Path:
    cfg = ctx.fixtures / f"{name}.yaml"
    cfg.write_text(dedent(body).lstrip("\n"))
    return cfg


def _validate(cfg: Path) -> list[str]:
    return ["uv", "run", "aiperf", "config", "validate", str(cfg)]


def _profile(cfg: Path) -> list[str]:
    return ["uv", "run", "aiperf", "profile", "--config", str(cfg)]


def _fail_with_log(log: Path, message: str) -> tuple[int, str]:
    with log.open("a") as out:
        out.write(f"\n{message}\n")
    return 1, log.read_text(errors="replace")


def _replace_once(text: str, old: str, new: str, log: Path, label: str) -> str:
    count = text.count(old)
    if count != 1:
        message = (
            f"{label} setup failed: expected exactly one occurrence "
            f"of {old!r}, found {count}"
        )
        _fail_with_log(log, f"AssertionError: {message}")
        raise AssertionError(f"AssertionError: {message}")
    return text.replace(old, new, 1)


def _read_profile_input_config(
    artifacts: Path, log: Path, label: str = "CLI override assertion"
) -> dict[str, object] | None:
    export = artifacts / "profile_export_aiperf.json"
    if not export.exists():
        _fail_with_log(
            log,
            f"{label} failed: expected profile export at {export}",
        )
        return None
    try:
        profile = orjson.loads(export.read_bytes())
        input_config = profile["input_config"]
    except (orjson.JSONDecodeError, KeyError, TypeError) as exc:
        _fail_with_log(
            log,
            f"{label} failed: could not read input_config from {export}: {exc!r}",
        )
        return None
    if not isinstance(input_config, dict):
        _fail_with_log(
            log,
            f"{label} failed: input_config in {export} was "
            f"{type(input_config).__name__}, not dict",
        )
        return None
    return input_config


def _minimal_body(ctx: Context, **overrides: str) -> str:
    url = overrides.get("url", ctx.url)
    return f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 16
              osl: 8
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
          artifacts:
            dir: {ctx.artifacts / "v2-loader-stub"}
          runtime:
            ui: none
          gpuTelemetry:
            enabled: false
          tokenizer:
            name: builtin
    """


# ---------- Jinja rendering chaos ----------


def case_jinja_unbalanced_braces(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "mock-{{{{ unclosed"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_jinja_unclosed_block(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "mock-{{% if true %}}only-open"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_jinja_strict_undefined_var(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "{{{{ definitely_undefined_variable }}}}"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_jinja_python_introspection(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "{{{{ ''.__class__.__mro__[1].__subclasses__() }}}}"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_jinja_in_skipped_template_field(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: template
            path: /v1/x
            template:
              body: "{{{{ definitely_undefined_at_request_time_only }}}}"
              responseField: choices.0.message.content
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Env-var expansion chaos ----------


def case_envvar_missing_required(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "${{DEFINITELY_UNSET_MODEL_NAME_XYZZY}}"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_envvar_unterminated_brace(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "mock-${{UNTERMINATED_VAR_NAME"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_envvar_invalid_identifier(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "mock-${{1INVALID-IDENTIFIER}}"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_envvar_type_coerced_into_int_field(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: "${{DEFINITELY_UNSET_CONCURRENCY:not-a-number}}"
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_envvar_empty_default_into_required_field(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: "${{DEFINITELY_UNSET_MODEL_XYZ:}}"
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Sweep dotted-path chaos ----------


def case_sweep_path_empty(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "": [1, 2, 3]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_path_leading_dot(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            ".phases.profiling.rate": [1.0, 2.0]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_path_double_dot(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "phases..profiling.rate": [1.0, 2.0]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_path_envelope_prefix(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "benchmark.phases.profiling.rate": [1.0, 2.0]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_path_non_sweepable_first(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "random_seed": [1, 2, 3]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_path_unknown_field(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "phases.profiling.NOT_A_REAL_FIELD": [1, 2, 3]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Parsing / normalizer chaos ----------


def case_yaml_duplicate_top_level_keys(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: first-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
        benchmark:
          model: second-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_duplicate_nested_endpoint_keys(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: http://127.0.0.1:1
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_duplicate_nested_prompt_keys(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              isl: 16
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_duplicate_phase_list_item_keys(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            - name: profiling
              type: concurrency
              concurrency: 1
              concurrency: 2
              requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_non_string_phase_list_item_key(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            - name: profiling
              type: concurrency
              concurrency: 1
              requests: 1
              123: bad
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_non_string_nested_benchmark_key(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
          123: non-string-key
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_flat_shape_mixed_with_benchmark(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = (
        _minimal_body(ctx).rstrip()
        + """
        model: flat-shape-model
        endpoint:
          url: http://127.0.0.1:1
          type: chat
        """
    )
    return run_cmd(_profile(_write(ctx, name, body)), log, ctx, 60)


def case_yaml_conflicting_singular_plural_aliases(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: single-model
          models:
            - plural-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          datasets:
            - name: other
              type: synthetic
              entries: 1
              prompts:
                isl: 8
                osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_python_object_directive(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: !!python/object/apply:os.system ["echo CHAOS_RCE_MARKER"]
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_anchor_cycle(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cfg = ctx.fixtures / f"{name}.yaml"
    cfg.write_text('schemaVersion: "2.0"\nbenchmark: &b\n  parent: *b\n')
    return run_cmd(_validate(cfg), log, ctx, 30)


def case_yaml_bom_prefixed(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cfg = ctx.fixtures / f"{name}.yaml"
    cfg.write_bytes("﻿".encode() + _minimal_body(ctx).encode())
    return run_cmd(_validate(cfg), log, ctx, 30)


def case_yaml_tabs_in_indent(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cfg = ctx.fixtures / f"{name}.yaml"
    cfg.write_text(
        'schemaVersion: "2.0"\n'
        "benchmark:\n"
        "\tmodel: mock-model\n"
        f"\tendpoint:\n\t\turl: {ctx.url}\n\t\ttype: chat\n"
    )
    return run_cmd(_validate(cfg), log, ctx, 30)


def case_yaml_unknown_top_level_key(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _minimal_body(ctx).rstrip() + "\nthis_is_not_a_real_top_level_key: 42\n"
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_yaml_schema_version_unsupported(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schema_version: "99.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Plot envelope chaos ----------


def case_plot_path_traversal(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _minimal_body(ctx).rstrip() + "\nplot: ../../../etc/passwd\n"
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_plot_nonexistent_path(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = (
        _minimal_body(ctx).rstrip()
        + "\nplot: /tmp/aiperf_chaos_definitely_missing_plot_config.yaml\n"
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_plot_inline_unknown_field(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = (
        _minimal_body(ctx).rstrip()
        + "\nplot:\n  not_a_real_plot_field: true\n  another_unknown: 42\n"
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Sweep config chaos ----------


def case_sweep_grid_empty_values(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "phases.profiling.rate": []
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_grid_nan_value(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "phases.profiling.rate": [1.0, .nan, 3.0]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_unknown_type(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: not-a-real-sweep-type
          parameters:
            "phases.profiling.rate": [1.0, 2.0]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Dataset subpackage chaos ----------


def case_dataset_empty_inline_records(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: file
            records: []
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_dataset_unknown_type(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: definitely-not-a-real-dataset-type
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_dataset_file_missing_path(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: file
            path: /tmp/aiperf_chaos_definitely_missing_dataset_file.jsonl
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_profile(_write(ctx, name, body)), log, ctx, 60)


# ---------- Resolution / runtime override chaos ----------


def case_runtime_unknown_ui(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "ui: none",
        "ui: not-a-real-ui-mode",
        log,
        "Runtime UI chaos",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_artifacts_dir_root(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        f"dir: {ctx.artifacts / 'v2-loader-stub'}",
        "dir: /",
        log,
        "Artifacts dir chaos",
    )
    return run_cmd(_profile(_write(ctx, name, body)), log, ctx, 30)


def case_comm_invalid_zmq_scheme(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _minimal_body(ctx).rstrip() + dedent(
        """
        comm:
          backend: zmq
          eventBus:
            sub: not-a-real-scheme://127.0.0.1:9999
        """
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_profile_cli_url_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _minimal_body(ctx, url="http://127.0.0.1:1").rstrip()
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--url",
        ctx.url,
        "--artifact-dir",
        str(artifacts),
    ]
    return run_cmd(cmd, log, ctx, 60)


def case_profile_cli_url_overrides_envvar_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _minimal_body(
        ctx,
        url="${BAD_URL_FOR_CLI_OVERRIDE:http://127.0.0.1:1}",
    ).rstrip()
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--url",
        ctx.url,
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        endpoint = input_config["endpoint"]
        urls = endpoint["urls"]
        observed = urls[0]
    except (KeyError, IndexError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read endpoint.urls[0] from "
            f"input_config: {exc!r}",
        )
    if observed != ctx.url:
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.urls[0] "
            f"{ctx.url!r}, observed {observed!r}",
        )
    return rc, text


def case_profile_cli_tokenizer_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "name: builtin",
        "name: yaml-tokenizer",
        log,
        "CLI tokenizer override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--tokenizer",
        "builtin",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        tokenizer = input_config["tokenizer"]
        observed = tokenizer["name"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read tokenizer.name from "
            f"input_config: {exc!r}",
        )
    if observed != "builtin":
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected tokenizer.name "
            f"'builtin', observed {observed!r}",
        )
    return rc, text


def case_profile_cli_artifact_dir_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    yaml_artifacts = ctx.artifacts / f"{name}-yaml-is-file"
    yaml_artifacts.write_text("regular file, not an artifact directory")
    cli_artifacts = ctx.artifacts / f"{name}-cli"
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        f"dir: {ctx.artifacts / 'v2-loader-stub'}",
        f"dir: {yaml_artifacts}",
        log,
        "CLI artifact-dir override",
    )
    cmd = [*_profile(_write(ctx, name, body)), "--artifact-dir", str(cli_artifacts)]
    return run_cmd(cmd, log, ctx, 60)


def case_profile_cli_ui_and_loadgen_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _minimal_body(ctx).rstrip()
    for old, new in (
        ("ui: none", "ui: dashboard"),
        ("concurrency: 1", "concurrency: 3"),
        ("requests: 1", "requests: 3"),
    ):
        body = _replace_once(body, old, new, log, "CLI UI/loadgen override")
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--ui",
        "simple",
        "--request-count",
        "1",
        "--concurrency",
        "1",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    export = artifacts / "profile_export_aiperf.json"
    if not export.exists():
        return _fail_with_log(
            log,
            f"CLI override assertion failed: expected profile export at {export}",
        )
    try:
        profile = orjson.loads(export.read_bytes())
        input_config = profile["input_config"]
        phase = input_config["phases"][0]
        observed = {
            "runtime.ui": input_config["runtime"]["ui"],
            "phases[0].concurrency": phase["concurrency"],
            "phases[0].requests": phase["requests"],
        }
    except (orjson.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read expected values from "
            f"{export}: {exc!r}",
        )
    expected = {
        "runtime.ui": "simple",
        "phases[0].concurrency": 1,
        "phases[0].requests": 1,
    }
    if observed != expected:
        with log.open("a") as out:
            out.write(
                "\nCLI override assertion failed: "
                f"expected {expected!r}, observed {observed!r}\n"
            )
        return 1, log.read_text(errors="replace")
    return rc, text


def case_yaml_empty_document(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cfg = ctx.fixtures / f"{name}.yaml"
    cfg.write_text("")
    return run_cmd(_validate(cfg), log, ctx, 30)


def case_yaml_list_document(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cfg = ctx.fixtures / f"{name}.yaml"
    cfg.write_text("- schemaVersion: '2.0'\n- benchmark: {}\n")
    return run_cmd(_validate(cfg), log, ctx, 30)


def case_yaml_boolean_top_level_key(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _minimal_body(ctx).rstrip() + "\ntrue: boolean-top-level-key\n"
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_jinja_variable_self_reference(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = "variables:\n  loop: '{{ loop }}'\n" + _minimal_body(ctx).rstrip()
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_jinja_variables_feed_benchmark_fields(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        variables:
          profile_concurrency: 2
          profile_requests: "{{{{ profile_concurrency }}}}"
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 2
            prompts:
              isl: 16
              osl: 8
          phases:
            type: concurrency
            concurrency: "{{{{ profile_concurrency }}}}"
            requests: "{{{{ profile_requests }}}}"
          artifacts:
            dir: {ctx.artifacts / "v2-loader-stub"}
          runtime:
            ui: none
          gpuTelemetry:
            enabled: false
          tokenizer:
            name: builtin
    """
    artifacts = ctx.artifacts / name
    cmd = [*_profile(_write(ctx, name, body)), "--artifact-dir", str(artifacts)]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(
        artifacts, log, label="Jinja variable assertion"
    )
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        phase = input_config["phases"][0]
        observed = {
            "phases[0].concurrency": phase["concurrency"],
            "phases[0].requests": phase["requests"],
        }
    except (KeyError, IndexError, TypeError) as exc:
        return _fail_with_log(
            log,
            "Jinja variable assertion failed: could not read phase values from "
            f"input_config: {exc!r}",
        )
    expected = {"phases[0].concurrency": 2, "phases[0].requests": 2}
    if observed != expected:
        return _fail_with_log(
            log,
            "Jinja variable assertion failed: expected rendered phase values "
            f"{expected!r}, observed {observed!r}",
        )
    return rc, text


def case_profile_cli_endpoint_type_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat",
        "type: completions",
        log,
        "CLI endpoint-type override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--endpoint-type",
        "chat",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["type"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read endpoint.type from "
            f"input_config: {exc!r}",
        )
    if observed != "chat":
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.type "
            f"'chat', observed {observed!r}",
        )
    return rc, text


def case_profile_cli_streaming_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            streaming: false\n          dataset:",
        log,
        "CLI streaming override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--streaming",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["streaming"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read endpoint.streaming from "
            f"input_config: {exc!r}",
        )
    if observed is not True:
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.streaming "
            f"True, observed {observed!r}",
        )
    return rc, text


def case_profile_cli_custom_endpoint_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            path: /definitely-not-real\n          dataset:",
        log,
        "CLI custom-endpoint override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--custom-endpoint",
        "/v1/chat/completions",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["path"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read endpoint.path from "
            f"input_config: {exc!r}",
        )
    if observed != "/v1/chat/completions":
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.path "
            f"'/v1/chat/completions', observed {observed!r}",
        )
    return rc, text


def case_profile_cli_request_timeout_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            timeout: 45.0\n          dataset:",
        log,
        "CLI request-timeout override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--request-timeout-seconds",
        "7.5",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["timeout"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read endpoint.timeout from "
            f"input_config: {exc!r}",
        )
    if observed != 7.5:
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.timeout "
            f"7.5, observed {observed!r}",
        )
    return rc, text


def case_profile_cli_no_gpu_telemetry_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "enabled: false",
        "enabled: true",
        log,
        "CLI no-gpu-telemetry override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--no-gpu-telemetry",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["gpu_telemetry"]["enabled"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read "
            f"gpu_telemetry.enabled from input_config: {exc!r}",
        )
    if observed is not False:
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected gpu_telemetry.enabled "
            f"False, observed {observed!r}",
        )
    return rc, text


def case_profile_cli_connection_reuse_strategy_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            connection_reuse: never\n          dataset:",
        log,
        "CLI connection-reuse-strategy override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--connection-reuse-strategy",
        "sticky-user-sessions",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["connection_reuse"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read "
            f"endpoint.connection_reuse from input_config: {exc!r}",
        )
    if observed != "sticky-user-sessions":
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.connection_reuse "
            f"'sticky-user-sessions', observed {observed!r}",
        )
    return rc, text


def case_profile_cli_use_server_token_count_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            use_server_token_count: false\n          dataset:",
        log,
        "CLI use-server-token-count override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--use-server-token-count",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["use_server_token_count"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read "
            f"endpoint.use_server_token_count from input_config: {exc!r}",
        )
    if observed is not True:
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.use_server_token_count "
            f"True, observed {observed!r}",
        )
    return rc, text


def case_profile_cli_wait_for_model_timeout_zero_overrides_yaml(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            wait_for_model_timeout: 10.0\n          dataset:",
        log,
        "CLI wait-for-model-timeout override",
    )
    artifacts = ctx.artifacts / name
    cmd = [
        *_profile(_write(ctx, name, body)),
        "--wait-for-model-timeout",
        "0",
        "--artifact-dir",
        str(artifacts),
    ]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(artifacts, log)
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["endpoint"]["wait_for_model_timeout"]
    except (KeyError, TypeError) as exc:
        return _fail_with_log(
            log,
            "CLI override assertion failed: could not read "
            f"endpoint.wait_for_model_timeout from input_config: {exc!r}",
        )
    if observed != 0:
        return _fail_with_log(
            log,
            "CLI override assertion failed: expected endpoint.wait_for_model_timeout "
            f"0, observed {observed!r}",
        )
    return rc, text


def case_yaml_null_benchmark(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = """
        schemaVersion: "2.0"
        benchmark: null
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_dataset_prompts_scalar(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "prompts:\n              isl: 16\n              osl: 8",
        "prompts: not-a-mapping",
        log,
        "Dataset prompts scalar",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_envvar_url_like_default_in_model_export(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "model: mock-model",
        "model: ${AIPERF_CHAOS_UNSET_MODEL_URL:http://models.example.test:8080/mock-model}",
        log,
        "Env-var default assertion",
    )
    artifacts = ctx.artifacts / name
    ctx.env.pop("AIPERF_CHAOS_UNSET_MODEL_URL", None)
    cmd = [*_profile(_write(ctx, name, body)), "--artifact-dir", str(artifacts)]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(
        artifacts, log, label="Env-var default assertion"
    )
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        observed = input_config["models"]["items"][0]["name"]
    except (KeyError, IndexError, TypeError) as exc:
        return _fail_with_log(
            log,
            "Env-var default assertion failed: could not read "
            f"models.items[0].name from input_config: {exc!r}",
        )
    expected = "http://models.example.test:8080/mock-model"
    if observed != expected:
        return _fail_with_log(
            log,
            "Env-var default assertion failed: expected models.items[0].name "
            f"{expected!r}, observed {observed!r}",
        )
    return rc, text


def case_envvar_numeric_phase_defaults_export(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _minimal_body(ctx).rstrip()
    for old, new in (
        ("concurrency: 1", "concurrency: ${AIPERF_CHAOS_UNSET_CONCURRENCY:2}"),
        ("requests: 1", "requests: ${AIPERF_CHAOS_UNSET_REQUESTS:2}"),
        ("entries: 1", "entries: 2"),
    ):
        body = _replace_once(body, old, new, log, "Env-var numeric default")
    artifacts = ctx.artifacts / name
    ctx.env.pop("AIPERF_CHAOS_UNSET_CONCURRENCY", None)
    ctx.env.pop("AIPERF_CHAOS_UNSET_REQUESTS", None)
    cmd = [*_profile(_write(ctx, name, body)), "--artifact-dir", str(artifacts)]
    rc, text = run_cmd(cmd, log, ctx, 60)
    if rc != 0:
        return rc, text

    input_config = _read_profile_input_config(
        artifacts, log, label="Env-var numeric default assertion"
    )
    if input_config is None:
        return 1, log.read_text(errors="replace")
    try:
        phase = input_config["phases"][0]
        observed = {
            "phases[0].concurrency": phase["concurrency"],
            "phases[0].requests": phase["requests"],
        }
    except (KeyError, IndexError, TypeError) as exc:
        return _fail_with_log(
            log,
            "Env-var numeric default assertion failed: could not read phase values "
            f"from input_config: {exc!r}",
        )
    expected = {"phases[0].concurrency": 2, "phases[0].requests": 2}
    if observed != expected:
        return _fail_with_log(
            log,
            "Env-var numeric default assertion failed: expected rendered int phase "
            f"values {expected!r}, observed {observed!r}",
        )
    return rc, text


def case_plot_inline_nested_wrong_type(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = (
        _minimal_body(ctx).rstrip()
        + """
        plot:
          visualization:
            single_run_defaults: []
            multi_run_defaults: []
            single_run_plots: {}
            multi_run_plots: {}
          settings:
            server_metrics_downsampling:
              window_size_seconds: not-a-number
        """
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_sweep_duplicate_parameter_paths(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        sweep:
          type: grid
          parameters:
            "phases.profiling.concurrency": [1]
            "phases.profiling.concurrency": [2]
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_variables_scalar(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        'schemaVersion: "2.0"',
        'variables: not-a-mapping\n        schemaVersion: "2.0"',
        log,
        "Variables scalar",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_variables_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        'schemaVersion: "2.0"',
        'variables:\n          - not\n          - a\n          - mapping\n        schemaVersion: "2.0"',
        log,
        "Variables list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_multi_run_num_runs_zero(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        'schemaVersion: "2.0"',
        'schemaVersion: "2.0"\n        multiRun:\n          numRuns: 0',
        log,
        "Multi-run numRuns zero",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_random_seed_negative(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        'schemaVersion: "2.0"',
        'schemaVersion: "2.0"\n        randomSeed: -1',
        log,
        "Random seed negative",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_phase_duplicate_names(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "phases:\n            type: concurrency\n            concurrency: 1\n            requests: 1",
        "phases:\n            - name: profiling\n              type: concurrency\n              concurrency: 1\n              requests: 1\n            - name: profiling\n              type: concurrency\n              concurrency: 1\n              requests: 1",
        log,
        "Duplicate phase names",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_phase_first_seamless_true(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            - name: profiling
              type: concurrency
              concurrency: 1
              requests: 1
              seamless: true
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_phase_prefill_non_streaming_endpoint(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
            streaming: false
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 2
            prefillConcurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_urls_empty_list(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = """
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            urls: []
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_url_empty_string(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = """
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: ""
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_url_unsupported_scheme(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        f"url: {ctx.url}",
        "url: ftp://127.0.0.1:8000/v1/chat/completions",
        log,
        "Endpoint URL unsupported scheme",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_tokenizer_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "tokenizer:\n            name: builtin",
        "tokenizer: []",
        log,
        "Tokenizer list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_runtime_workers_zero(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "runtime:\n            ui: none",
        "runtime:\n            ui: none\n            workers: 0",
        log,
        "Runtime workers zero",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_logging_scalar(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "gpuTelemetry:\n            enabled: false",
        "gpuTelemetry:\n            enabled: false\n          logging: trace",
        log,
        "Logging scalar",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_dataset_entries_zero(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "entries: 1",
        "entries: 0",
        log,
        "Dataset entries zero",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_artifacts_unknown_summary_format(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = f"""
        schemaVersion: "2.0"
        benchmark:
          model: mock-model
          endpoint:
            url: {ctx.url}
            type: chat
          dataset:
            type: synthetic
            entries: 1
            prompts:
              isl: 8
              osl: 4
          phases:
            type: concurrency
            concurrency: 1
            requests: 1
          artifacts:
            dir: {ctx.artifacts / "unknown-summary-format"}
            summary: [json, definitely-not-a-format]
    """
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_metrics_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "tokenizer:\n            name: builtin",
        "tokenizer:\n            name: builtin\n          metrics:\n            - not-a-mapping",
        log,
        "Metrics list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_slos_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "tokenizer:\n            name: builtin",
        "tokenizer:\n            name: builtin\n          slos:\n            - request_latency\n            - 500",
        log,
        "SLOs list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_accuracy_unknown_benchmark(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "tokenizer:\n            name: builtin",
        "tokenizer:\n            name: builtin\n          accuracy:\n            benchmark: definitely-not-a-benchmark",
        log,
        "Accuracy unknown benchmark",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_server_metrics_invalid_format(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "tokenizer:\n            name: builtin",
        "tokenizer:\n            name: builtin\n          serverMetrics:\n            enabled: true\n            formats: [json, definitely-not-a-format]",
        log,
        "Server metrics invalid format",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_headers_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            headers:\n              - Authorization: Bearer token\n          dataset:",
        log,
        "Endpoint headers list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_extra_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            extra:\n              - temperature: 0.7\n          dataset:",
        log,
        "Endpoint extra list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_transport_unknown(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            transport: definitely-not-a-transport\n          dataset:",
        log,
        "Endpoint transport unknown",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_endpoint_url_strategy_unknown(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "type: chat\n          dataset:",
        "type: chat\n            urlStrategy: definitely-not-a-url-strategy\n          dataset:",
        log,
        "Endpoint URL strategy unknown",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_gpu_telemetry_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        "gpuTelemetry:\n            enabled: false",
        "gpuTelemetry:\n            - http://127.0.0.1:9400/metrics",
        log,
        "GPU telemetry list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


def case_artifacts_dir_list(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    body = _replace_once(
        _minimal_body(ctx).rstrip(),
        f"dir: {ctx.artifacts / 'v2-loader-stub'}",
        "dir:\n              - ./not-a-path-scalar",
        log,
        "Artifacts dir list",
    )
    return run_cmd(_validate(_write(ctx, name, body)), log, ctx, 30)


# ---------- Case registry ----------


def build_config_loader_cases() -> list[Case]:
    return [
        # Jinja
        Case(
            "v2-jinja-unbalanced-braces",
            "GRACEFUL_FAILURE_REQUIRED",
            case_jinja_unbalanced_braces,
            "jinja template with unbalanced {{ braces must fail cleanly",
        ),
        Case(
            "v2-jinja-unclosed-block",
            "GRACEFUL_FAILURE_REQUIRED",
            case_jinja_unclosed_block,
            "jinja template with unclosed {% if %} block must fail cleanly",
        ),
        Case(
            "v2-jinja-strict-undefined-var",
            "GRACEFUL_FAILURE_REQUIRED",
            case_jinja_strict_undefined_var,
            "StrictUndefined must reject reference to undefined jinja var",
        ),
        Case(
            "v2-jinja-python-introspection",
            "FLAG_FOR_REVIEW",
            case_jinja_python_introspection,
            "jinja accessing __class__/__mro__ should be sandboxed or rejected",
        ),
        Case(
            "v2-jinja-in-skipped-template-field",
            "PASS_REQUIRED",
            case_jinja_in_skipped_template_field,
            "jinja inside endpoint.template.body is intentionally skipped at load time",
        ),
        Case(
            "v2-jinja-variable-self-reference",
            "GRACEFUL_FAILURE_REQUIRED",
            case_jinja_variable_self_reference,
            "variables entry that references itself must fail cleanly as a cycle",
        ),
        Case(
            "v2-jinja-variables-feed-benchmark-fields",
            "PASS_REQUIRED",
            case_jinja_variables_feed_benchmark_fields,
            "profile export must show variables rendered into benchmark phase fields",
        ),
        Case(
            "v2-variables-scalar",
            "GRACEFUL_FAILURE_REQUIRED",
            case_variables_scalar,
            "variables provided as a scalar must fail config validation cleanly before Jinja rendering",
        ),
        Case(
            "v2-variables-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_variables_list,
            "variables provided as a list must fail config validation cleanly before Jinja rendering",
        ),
        # Env vars
        Case(
            "v2-envvar-missing-required",
            "GRACEFUL_FAILURE_REQUIRED",
            case_envvar_missing_required,
            "unset ${VAR} without default must produce a clear loader error",
        ),
        Case(
            "v2-envvar-unterminated-brace",
            "GRACEFUL_FAILURE_REQUIRED",
            case_envvar_unterminated_brace,
            "${UNTERMINATED must not silently pass through as literal",
        ),
        Case(
            "v2-envvar-invalid-identifier",
            "FLAG_FOR_REVIEW",
            case_envvar_invalid_identifier,
            "${1INVALID-IDENTIFIER} does not match ENV_VAR_PATTERN; literal pass-through is acceptable but worth review",
        ),
        Case(
            "v2-envvar-type-coerced-into-int",
            "GRACEFUL_FAILURE_REQUIRED",
            case_envvar_type_coerced_into_int_field,
            "${VAR:not-a-number} expanded into int field must fail Pydantic validation cleanly",
        ),
        Case(
            "v2-envvar-empty-default-into-required",
            "GRACEFUL_FAILURE_REQUIRED",
            case_envvar_empty_default_into_required_field,
            "empty default ${VAR:} into a required model name should fail validation, not silently allow empty string",
        ),
        Case(
            "v2-envvar-url-like-default-in-model-export",
            "PASS_REQUIRED",
            case_envvar_url_like_default_in_model_export,
            "profile export must preserve env-var default values containing :// and colon separators",
        ),
        Case(
            "v2-envvar-numeric-phase-defaults-export",
            "PASS_REQUIRED",
            case_envvar_numeric_phase_defaults_export,
            "profile export must show numeric env-var defaults coerced into int phase fields",
        ),
        # Sweep dotted-path
        Case(
            "v2-sweep-path-empty",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_path_empty,
            "sweep parameter with empty path must be rejected",
        ),
        Case(
            "v2-sweep-path-leading-dot",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_path_leading_dot,
            "sweep parameter path starting with '.' must be rejected",
        ),
        Case(
            "v2-sweep-path-double-dot",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_path_double_dot,
            "sweep parameter path with consecutive dots must be rejected",
        ),
        Case(
            "v2-sweep-path-envelope-prefix",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_path_envelope_prefix,
            "sweep parameter path with redundant benchmark. prefix must be rejected",
        ),
        Case(
            "v2-sweep-path-non-sweepable-first",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_path_non_sweepable_first,
            "sweep parameter path targeting random_seed must be rejected",
        ),
        Case(
            "v2-sweep-path-unknown-field",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_path_unknown_field,
            "sweep parameter path that does not resolve to a real field must fail cleanly",
        ),
        # Parsing / normalizers
        Case(
            "v2-yaml-empty-document",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_empty_document,
            "empty YAML document must fail loader validation cleanly",
        ),
        Case(
            "v2-yaml-list-document",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_list_document,
            "top-level YAML list must fail loader validation cleanly, not crash mapping assumptions",
        ),
        Case(
            "v2-yaml-boolean-top-level-key",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_boolean_top_level_key,
            "boolean top-level YAML key must fail string-key validation at the root",
        ),
        Case(
            "v2-yaml-duplicate-top-level-keys",
            "FLAG_FOR_REVIEW",
            case_yaml_duplicate_top_level_keys,
            "duplicate top-level benchmark: keys -- second-wins is permissive; loader should warn",
        ),
        Case(
            "v2-yaml-duplicate-nested-endpoint-keys",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_duplicate_nested_endpoint_keys,
            "duplicate nested benchmark.endpoint.url keys must fail cleanly, not silently last-win",
        ),
        Case(
            "v2-yaml-duplicate-nested-prompt-keys",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_duplicate_nested_prompt_keys,
            "duplicate nested benchmark.dataset.prompts.isl keys must fail cleanly, proving duplicate detection recurses beyond endpoint",
        ),
        Case(
            "v2-yaml-duplicate-phase-list-item-keys",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_duplicate_phase_list_item_keys,
            "duplicate keys inside benchmark.phases list items must fail cleanly, proving recursive duplicate detection walks lists",
        ),
        Case(
            "v2-yaml-non-string-phase-list-item-key",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_non_string_phase_list_item_key,
            "non-string YAML key inside a benchmark.phases list item must fail cleanly before path-based expansion",
        ),
        Case(
            "v2-yaml-non-string-nested-benchmark-key",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_non_string_nested_benchmark_key,
            "non-string YAML key under benchmark must fail cleanly before path-based expansion",
        ),
        Case(
            "v2-yaml-flat-shape-mixed-with-benchmark",
            "PASS_REQUIRED",
            case_yaml_flat_shape_mixed_with_benchmark,
            "profile must keep nested benchmark endpoint over invalid flat endpoint during flat-shape auto-migration",
        ),
        Case(
            "v2-yaml-conflicting-singular-plural-aliases",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_conflicting_singular_plural_aliases,
            "benchmark containing both singular and plural model/dataset aliases must fail cleanly",
        ),
        Case(
            "v2-yaml-python-object-directive",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_python_object_directive,
            "!!python/object directive must not execute (no code execution from YAML)",
        ),
        Case(
            "v2-yaml-anchor-cycle",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_anchor_cycle,
            "self-referential YAML anchor must fail cleanly, not infinite-recurse",
        ),
        Case(
            "v2-yaml-bom-prefixed",
            "PASS_REQUIRED",
            case_yaml_bom_prefixed,
            "BOM-prefixed YAML should still parse (or fail cleanly with a clear error)",
        ),
        Case(
            "v2-yaml-tabs-in-indent",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_tabs_in_indent,
            "tabs used as indentation must fail YAML parse cleanly",
        ),
        Case(
            "v2-yaml-unknown-top-level-key",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_unknown_top_level_key,
            "unknown top-level key under extra=forbid envelope must be rejected",
        ),
        Case(
            "v2-yaml-schema-version-unsupported",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_schema_version_unsupported,
            "schema_version that is not 2.0 must be rejected with a clear message",
        ),
        Case(
            "v2-yaml-null-benchmark",
            "GRACEFUL_FAILURE_REQUIRED",
            case_yaml_null_benchmark,
            "benchmark: null must fail config validation cleanly instead of bypassing required benchmark fields",
        ),
        Case(
            "v2-phase-duplicate-names",
            "GRACEFUL_FAILURE_REQUIRED",
            case_phase_duplicate_names,
            "benchmark.phases list containing duplicate phase names must fail config validation cleanly",
        ),
        Case(
            "v2-phase-first-seamless-true",
            "GRACEFUL_FAILURE_REQUIRED",
            case_phase_first_seamless_true,
            "first benchmark phase with seamless: true must fail config validation cleanly",
        ),
        Case(
            "v2-phase-prefill-non-streaming-endpoint",
            "GRACEFUL_FAILURE_REQUIRED",
            case_phase_prefill_non_streaming_endpoint,
            "prefillConcurrency requires endpoint.streaming=true and must fail cleanly when streaming is false",
        ),
        # Plot envelope
        Case(
            "v2-plot-path-traversal",
            "GRACEFUL_FAILURE_REQUIRED",
            case_plot_path_traversal,
            "plot: bare-string with .. traversal must not load /etc/passwd",
        ),
        Case(
            "v2-plot-nonexistent-path",
            "GRACEFUL_FAILURE_REQUIRED",
            case_plot_nonexistent_path,
            "plot: bare-string pointing at missing file must fail with clear error",
        ),
        Case(
            "v2-plot-inline-unknown-field",
            "GRACEFUL_FAILURE_REQUIRED",
            case_plot_inline_unknown_field,
            "plot envelope with unknown fields must be rejected (extra=forbid)",
        ),
        Case(
            "v2-plot-inline-nested-wrong-type",
            "GRACEFUL_FAILURE_REQUIRED",
            case_plot_inline_nested_wrong_type,
            "plot envelope nested server_metrics_downsampling.window_size_seconds wrong type must fail cleanly",
        ),
        # Sweep config
        Case(
            "v2-sweep-grid-empty-values",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_grid_empty_values,
            "grid sweep with empty value list must be rejected",
        ),
        Case(
            "v2-sweep-grid-nan-value",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_grid_nan_value,
            "grid sweep containing NaN must be rejected (FiniteFloat contract)",
        ),
        Case(
            "v2-sweep-unknown-type",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_unknown_type,
            "sweep type that is not grid/qmc/adaptive must be rejected",
        ),
        Case(
            "v2-sweep-duplicate-parameter-paths",
            "GRACEFUL_FAILURE_REQUIRED",
            case_sweep_duplicate_parameter_paths,
            "duplicate YAML keys under sweep.parameters must fail cleanly before last-write-wins hides a duplicate sweep path",
        ),
        # Dataset subpackage
        Case(
            "v2-dataset-empty-inline-records",
            "GRACEFUL_FAILURE_REQUIRED",
            case_dataset_empty_inline_records,
            "file dataset with empty inline records list must be rejected",
        ),
        Case(
            "v2-dataset-unknown-type",
            "GRACEFUL_FAILURE_REQUIRED",
            case_dataset_unknown_type,
            "dataset type that is not registered must be rejected with a clear error",
        ),
        Case(
            "v2-dataset-file-missing-path",
            "GRACEFUL_FAILURE_REQUIRED",
            case_dataset_file_missing_path,
            "file dataset pointing at non-existent path must fail profile cleanly",
        ),
        Case(
            "v2-dataset-prompts-scalar",
            "GRACEFUL_FAILURE_REQUIRED",
            case_dataset_prompts_scalar,
            "synthetic dataset.prompts provided as a scalar must fail config validation cleanly",
        ),
        Case(
            "v2-dataset-entries-zero",
            "GRACEFUL_FAILURE_REQUIRED",
            case_dataset_entries_zero,
            "synthetic dataset.entries: 0 must fail config validation cleanly against the ge=1 schema bound",
        ),
        # Resolution / runtime
        Case(
            "v2-runtime-unknown-ui",
            "GRACEFUL_FAILURE_REQUIRED",
            case_runtime_unknown_ui,
            "runtime.ui set to unknown value must be rejected by the enum validator",
        ),
        Case(
            "v2-artifacts-dir-root",
            "GRACEFUL_FAILURE_REQUIRED",
            case_artifacts_dir_root,
            "artifacts.dir at filesystem root must fail cleanly (permission denied or guard)",
        ),
        Case(
            "v2-comm-invalid-zmq-scheme",
            "GRACEFUL_FAILURE_REQUIRED",
            case_comm_invalid_zmq_scheme,
            "comm.eventBus.sub with non-ZMQ scheme must be rejected",
        ),
        Case(
            "v2-endpoint-urls-empty-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_urls_empty_list,
            "endpoint.urls empty list must fail config validation cleanly against the min_length schema bound",
        ),
        Case(
            "v2-endpoint-url-empty-string",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_url_empty_string,
            "endpoint.url empty string must fail config validation cleanly before request execution",
        ),
        Case(
            "v2-endpoint-url-unsupported-scheme",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_url_unsupported_scheme,
            "endpoint.url with unsupported ftp:// scheme must fail config validation cleanly before request execution",
        ),
        Case(
            "v2-tokenizer-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_tokenizer_list,
            "tokenizer provided as a list must fail config validation cleanly against the object schema",
        ),
        Case(
            "v2-runtime-workers-zero",
            "GRACEFUL_FAILURE_REQUIRED",
            case_runtime_workers_zero,
            "runtime.workers: 0 must fail config validation cleanly against the ge=1 schema bound",
        ),
        Case(
            "v2-logging-scalar",
            "GRACEFUL_FAILURE_REQUIRED",
            case_logging_scalar,
            "logging provided as a scalar must fail config validation cleanly against the object schema",
        ),
        Case(
            "v2-artifacts-unknown-summary-format",
            "GRACEFUL_FAILURE_REQUIRED",
            case_artifacts_unknown_summary_format,
            "artifacts.summary containing an unknown export format must fail enum validation cleanly",
        ),
        Case(
            "v2-metrics-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_metrics_list,
            "metrics provided as a list must fail config validation cleanly against the object schema",
        ),
        Case(
            "v2-slos-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_slos_list,
            "slos provided as a list must fail config validation cleanly against the metric-threshold mapping schema",
        ),
        Case(
            "v2-accuracy-unknown-benchmark",
            "GRACEFUL_FAILURE_REQUIRED",
            case_accuracy_unknown_benchmark,
            "accuracy.benchmark with an unknown plugin name must fail enum validation cleanly",
        ),
        Case(
            "v2-server-metrics-invalid-format",
            "GRACEFUL_FAILURE_REQUIRED",
            case_server_metrics_invalid_format,
            "serverMetrics.formats containing an unknown export format must fail enum validation cleanly",
        ),
        Case(
            "v2-endpoint-headers-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_headers_list,
            "endpoint.headers provided as a list must fail config validation cleanly against the string mapping schema",
        ),
        Case(
            "v2-endpoint-extra-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_extra_list,
            "endpoint.extra provided as a list must fail config validation cleanly against the request-body mapping schema",
        ),
        Case(
            "v2-endpoint-transport-unknown",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_transport_unknown,
            "endpoint.transport set to an unknown plugin name must fail enum validation cleanly",
        ),
        Case(
            "v2-endpoint-url-strategy-unknown",
            "GRACEFUL_FAILURE_REQUIRED",
            case_endpoint_url_strategy_unknown,
            "endpoint.urlStrategy set to an unknown strategy must fail enum validation cleanly",
        ),
        Case(
            "v2-gpu-telemetry-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_gpu_telemetry_list,
            "gpuTelemetry provided as a list must fail config validation cleanly against the object-or-string shorthand schema",
        ),
        Case(
            "v2-artifacts-dir-list",
            "GRACEFUL_FAILURE_REQUIRED",
            case_artifacts_dir_list,
            "artifacts.dir provided as a list must fail config validation cleanly against the path scalar schema",
        ),
        Case(
            "v2-multi-run-num-runs-zero",
            "GRACEFUL_FAILURE_REQUIRED",
            case_multi_run_num_runs_zero,
            "multiRun.numRuns: 0 must fail config validation cleanly against the ge=1 schema bound",
        ),
        Case(
            "v2-random-seed-negative",
            "GRACEFUL_FAILURE_REQUIRED",
            case_random_seed_negative,
            "randomSeed: -1 must fail config validation cleanly against the non-negative seed bound",
        ),
        Case(
            "v2-profile-cli-url-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_url_overrides_yaml,
            "profile YAML+CLI merge must let --url override an invalid YAML endpoint.url",
        ),
        Case(
            "v2-profile-cli-url-overrides-envvar-yaml",
            "PASS_REQUIRED",
            case_profile_cli_url_overrides_envvar_yaml,
            "profile YAML+CLI merge must let --url override endpoint.url after YAML env-var substitution",
        ),
        Case(
            "v2-profile-cli-tokenizer-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_tokenizer_overrides_yaml,
            "profile YAML+CLI merge must export the CLI --tokenizer value, not the YAML tokenizer",
        ),
        Case(
            "v2-profile-cli-artifact-dir-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_artifact_dir_overrides_yaml,
            "profile YAML+CLI merge must let --artifact-dir override YAML artifacts.dir",
        ),
        Case(
            "v2-profile-cli-ui-and-loadgen-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_ui_and_loadgen_overrides_yaml,
            "profile YAML+CLI merge must accept --ui and loadgen overrides on top of YAML",
        ),
        Case(
            "v2-profile-cli-endpoint-type-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_endpoint_type_overrides_yaml,
            "profile YAML+CLI merge must export CLI --endpoint-type over YAML endpoint.type",
        ),
        Case(
            "v2-profile-cli-streaming-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_streaming_overrides_yaml,
            "profile YAML+CLI merge must export CLI --streaming over YAML endpoint.streaming: false",
        ),
        Case(
            "v2-profile-cli-custom-endpoint-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_custom_endpoint_overrides_yaml,
            "profile YAML+CLI merge must export CLI --custom-endpoint over YAML endpoint.path",
        ),
        Case(
            "v2-profile-cli-request-timeout-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_request_timeout_overrides_yaml,
            "profile YAML+CLI merge must export CLI --request-timeout-seconds over YAML endpoint.timeout",
        ),
        Case(
            "v2-profile-cli-no-gpu-telemetry-overrides-yaml",
            "FLAG_FOR_REVIEW",
            case_profile_cli_no_gpu_telemetry_overrides_yaml,
            "profile YAML+CLI merge currently does not export --no-gpu-telemetry over YAML gpuTelemetry.enabled: true",
        ),
        Case(
            "v2-profile-cli-connection-reuse-strategy-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_connection_reuse_strategy_overrides_yaml,
            "profile YAML+CLI merge must export CLI --connection-reuse-strategy over YAML endpoint.connection_reuse",
        ),
        Case(
            "v2-profile-cli-use-server-token-count-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_use_server_token_count_overrides_yaml,
            "profile YAML+CLI merge must export CLI --use-server-token-count over YAML endpoint.use_server_token_count: false",
        ),
        Case(
            "v2-profile-cli-wait-for-model-timeout-zero-overrides-yaml",
            "PASS_REQUIRED",
            case_profile_cli_wait_for_model_timeout_zero_overrides_yaml,
            "profile YAML+CLI merge must export CLI --wait-for-model-timeout 0 over positive YAML endpoint.wait_for_model_timeout",
        ),
    ]
