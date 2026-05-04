# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube sweep` subcommand: submit an AIPerfSweep CR to the cluster.

The sweep command reads a YAML config file that contains both the base
benchmark config (the same shape as `aiperf kube profile -f ...`) and one or
both of the optional top-level keys ``sweep:`` and ``multi_run:``. Those keys
are hoisted out of the benchmark and placed under ``AIPerfSweep.spec``; the
rest of the YAML becomes ``AIPerfSweep.spec.template.spec.benchmark``.

For v1, the command only submits the CR and returns. Live tailing/attach to
sweep progress is future work.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.kube import KubeOptions
from aiperf.config.v1 import ServiceConfig, UserConfig

if TYPE_CHECKING:
    from pathlib import Path

app = App(name="sweep")


_DETACH_PARAM = Parameter(
    name=["-d", "--detach"],
    help="Exit after submitting (don't tail). v1 always behaves as detach=True.",
)
_DRY_RUN_PARAM = Parameter(
    name="--dry-run",
    negative=(),
    help="Print the AIPerfSweep CR as JSON without submitting it.",
)
_TRIALS_PARAM = Parameter(
    name="--trials",
    help="Multi-run trials per sweep cell; overrides multi_run.trials in the YAML.",
)
_COOLDOWN_PARAM = Parameter(
    name="--cooldown",
    help="Cooldown seconds between multi-run trials (overrides YAML).",
)
_CONV_METRIC_PARAM = Parameter(
    name="--convergence-metric",
    help="Stop multi-run early when this metric converges (e.g. ttft_p99).",
)
_CONV_MIN_PARAM = Parameter(
    name="--min-runs",
    help="Minimum runs before convergence is checked (default 3).",
)
_CONV_MAX_PARAM = Parameter(
    name="--max-runs",
    help="Hard cap on runs even if not converged (default 10).",
)
_CONV_THRESH_PARAM = Parameter(
    name="--convergence-threshold",
    help="Relative convergence threshold (default 0.05 = 5%).",
)


@app.default
async def sweep(
    *,
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
    kube_options: KubeOptions,
    multi_run_trials: Annotated[int | None, _TRIALS_PARAM] = None,
    cooldown_seconds: Annotated[float, _COOLDOWN_PARAM] = 0.0,
    convergence_metric: Annotated[str | None, _CONV_METRIC_PARAM] = None,
    convergence_min_runs: Annotated[int, _CONV_MIN_PARAM] = 3,
    convergence_max_runs: Annotated[int, _CONV_MAX_PARAM] = 10,
    convergence_threshold: Annotated[float, _CONV_THRESH_PARAM] = 0.05,
    detach: Annotated[bool, _DETACH_PARAM] = False,  # noqa: ARG001 - reserved for future tailing
    dry_run: Annotated[bool, _DRY_RUN_PARAM] = False,
) -> None:
    """Submit an AIPerfSweep CR for parameter or multi-run benchmarks.

    The config file (``--config <file>``) must contain a base AIPerfConfig plus
    an optional top-level ``sweep:`` and/or ``multi_run:`` section. Those are
    hoisted out of the benchmark config and placed at the AIPerfSweep.spec
    level; the rest becomes ``spec.template.spec.benchmark``.

    Examples:
        # Parameter sweep declared in YAML
        aiperf kube sweep -f sweep.yaml --image aiperf:latest

        # Multi-run repeats with cooldown, no parameter axis
        aiperf kube sweep -f bench.yaml --image aiperf:latest \\
            --trials 5 --cooldown 30
    """
    from aiperf import cli_utils
    from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

    if service_config is None:
        service_config = ServiceConfig()
    # service_config is currently only consumed when we eventually load+validate
    # the YAML through the v1->v2 path; v1 sweep submits the raw YAML directly,
    # so service_config is reserved here for parity with profile/generate.
    _ = service_config

    config_file = getattr(user_config, "config_file", None)
    with cli_utils.exit_on_error(title="Error Running Kubernetes Sweep"):
        cr_dict = _build_sweep_cr_dict(
            config_file=config_file,
            kube_options=kube_options,
            user_config=user_config,
            multi_run_trials=multi_run_trials,
            cooldown_seconds=cooldown_seconds,
            convergence_metric=convergence_metric,
            convergence_min_runs=convergence_min_runs,
            convergence_max_runs=convergence_max_runs,
            convergence_threshold=convergence_threshold,
        )
        if dry_run:
            import orjson

            from aiperf.kubernetes import console as kube_console

            kube_console.console.print(
                orjson.dumps(cr_dict, option=orjson.OPT_INDENT_2).decode(),
                highlight=False,
            )
            return
        await _submit_sweep(
            cr_dict=cr_dict,
            kube_options=kube_options,
            namespace=kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE,
        )


def _extract_envelope_keys(
    raw: dict[str, Any], *, config_file: Path
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Pull (bench_dict, sweep_cfg, multirun_cfg, envelope_variables) from a
    user YAML file. Accepts three shapes: bare AIPerfConfig, AIPerfJob CR,
    AIPerfSweep CR (the last is rejected).
    """
    is_aiperf_cr = (
        isinstance(raw, dict)
        and isinstance(raw.get("apiVersion"), str)
        and raw["apiVersion"].startswith("aiperf.nvidia.com")
    )
    if is_aiperf_cr and raw.get("kind") == "AIPerfSweep":
        raise ValueError(
            f"'{config_file}' is already an AIPerfSweep CR. Use "
            f"`kubectl apply -f {config_file}` to submit it, or pass a plain "
            f"AIPerfConfig YAML / AIPerfJob CR to have `aiperf kube sweep` "
            f"build the sweep CR."
        )
    if is_aiperf_cr and raw.get("kind") == "AIPerfJob":
        cr_spec = dict(raw.get("spec") or {})
        benchmark_raw = cr_spec.get("benchmark") or {}
        sweep_cfg = cr_spec.pop("sweep", None) or benchmark_raw.pop("sweep", None)
        multirun = (
            cr_spec.pop("multiRun", None)
            or cr_spec.pop("multi_run", None)
            or benchmark_raw.pop("multi_run", None)
            or benchmark_raw.pop("multiRun", None)
        )
        return benchmark_raw, sweep_cfg, multirun, cr_spec.pop("variables", None)

    sweep_cfg = raw.pop("sweep", None)
    multirun = raw.pop("multi_run", None) or raw.pop("multiRun", None)
    envelope_variables = raw.pop("variables", None)
    if isinstance(raw.get("benchmark"), dict) and not (
        "models" in raw or "model" in raw or "endpoint" in raw
    ):
        return raw["benchmark"], sweep_cfg, multirun, envelope_variables
    return raw, sweep_cfg, multirun, envelope_variables


def _build_multirun_cfg(
    multirun_cfg_from_yaml: Any,
    *,
    multi_run_trials: int | None,
    cooldown_seconds: float,
) -> dict[str, Any] | None:
    """Merge YAML-declared multi_run with CLI overrides. CLI wins per docs."""
    multirun_cfg: dict[str, Any] | None = None
    if multirun_cfg_from_yaml is not None:
        multirun_cfg = dict(multirun_cfg_from_yaml)
    if multi_run_trials is not None:
        multirun_cfg = multirun_cfg or {}
        multirun_cfg["trials"] = multi_run_trials
    if cooldown_seconds:
        multirun_cfg = multirun_cfg or {}
        multirun_cfg["cooldownSeconds"] = cooldown_seconds
    return multirun_cfg


def _build_sweep_cr_dict(
    *,
    config_file: Path | None,
    kube_options: KubeOptions,
    user_config: UserConfig | None = None,
    multi_run_trials: int | None,
    cooldown_seconds: float,
    convergence_metric: str | None,
    convergence_min_runs: int,
    convergence_max_runs: int,
    convergence_threshold: float,
) -> dict[str, Any]:
    """Build an AIPerfSweep CR dict from a YAML config file with sweep config.

    The config file must contain (at minimum) a base AIPerfConfig. Optional
    top-level ``sweep:`` and ``multi_run:`` keys are extracted and placed under
    ``AIPerfSweep.spec``; the remainder becomes ``spec.template.spec.benchmark``.

    When ``user_config`` is provided and carries explicitly-set CLI flags
    (e.g. ``--search-recipe``, ``--ttft-sla-ms``, ``--streaming``), those
    overrides are deep-merged onto the YAML before AIPerfConfig validation,
    and recipe-expanded ``sweep`` / ``multi_run`` blocks bubble up to the CR
    spec the same way YAML-declared ones do.

    Raises:
        ValueError: ``config_file`` is None — `aiperf kube sweep` requires a
            YAML config (no flag-only invocation supported).
    """
    if config_file is None:
        raise ValueError(
            "`aiperf kube sweep` requires --config <file> with a base AIPerfConfig "
            "and optional top-level `sweep:`/`multi_run:` keys."
        )

    import yaml

    from aiperf.config import AIPerfConfig, dump_config
    from aiperf.config.loader import expand_config_dict

    raw = yaml.safe_load(config_file.read_text()) or {}
    bench_dict, sweep_cfg, multirun_cfg_from_yaml, envelope_variables = (
        _extract_envelope_keys(raw, config_file=config_file)
    )

    # Render Jinja2 / ${ENV_VAR} in the benchmark portion before submission so
    # unresolved `{{ ... }}` literals never trip AIPerfSweepSpec.model_validate
    # below or reach the operator. Mirrors `aiperf kube profile -f`'s pipeline.
    if envelope_variables:
        wrapped = {"variables": envelope_variables, **bench_dict}
        wrapped = expand_config_dict(wrapped)
        wrapped.pop("variables", None)
        bench_dict = wrapped
    else:
        bench_dict = expand_config_dict(bench_dict)

    # Deep-merge explicitly-set CLI flags (e.g. `--search-recipe`,
    # `--ttft-sla-ms`, `--streaming`) onto the YAML before validation. Recipe
    # expansion produces `sweep` / `multi_run` blocks which we hoist to the
    # AIPerfSweep spec instead of embedding inside the benchmark.
    bench_dict, sweep_cfg, multirun_cfg_from_yaml = _apply_cli_overrides(
        bench_dict=bench_dict,
        user_config=user_config,
        sweep_cfg=sweep_cfg,
        multirun_cfg_from_yaml=multirun_cfg_from_yaml,
    )

    # Validate via AIPerfConfig so v1->v2 shorthand promotions
    # (`model:`/`dataset:`/`phases:`) expand to the long-form the operator
    # expects -- matching the path `kube profile` takes for CR-shaped input.
    config = AIPerfConfig.model_validate({"benchmark": bench_dict})
    bench_dict = yaml.safe_load(dump_config(config)).get("benchmark", {})

    deployment = kube_options.to_deployment_config()
    deployment_dict = deployment.model_dump(
        mode="json", by_alias=True, exclude_defaults=True
    )
    template_spec: dict[str, Any] = {
        **deployment_dict,
        "image": kube_options.image,
        "benchmark": bench_dict,
    }

    spec: dict[str, Any] = {"template": {"spec": template_spec}}
    if sweep_cfg is not None:
        spec["sweep"] = sweep_cfg

    multirun_cfg = _build_multirun_cfg(
        multirun_cfg_from_yaml,
        multi_run_trials=multi_run_trials,
        cooldown_seconds=cooldown_seconds,
    )
    if multirun_cfg:
        spec["multiRun"] = multirun_cfg

    if convergence_metric is not None:
        spec["convergence"] = {
            "metric": convergence_metric,
            "minRuns": convergence_min_runs,
            "maxRuns": convergence_max_runs,
            "threshold": convergence_threshold,
        }

    name = kube_options.name or _name_from_config_file(config_file)
    cr_dict: dict[str, Any] = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfSweep",
        "metadata": {"name": name},
        "spec": spec,
    }
    # Validate before submission so users see the curated AIPerfSweepSpec
    # error messages, not a raw apiserver CRD validation 422.
    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

    # Round-trip through Pydantic with by_alias=True so snake_case keys from
    # the v1 converter (e.g. `adaptive_search`) get serialised as the
    # CRD-canonical camelCase (`adaptiveSearch`). Without this, the apiserver
    # rejects with `strict decoding error: unknown field
    # "spec.multiRun.adaptive_search"` even though our local validator (which
    # has populate_by_name=True) accepts it.
    spec = AIPerfSweepSpec.model_validate(spec).model_dump(
        mode="json", by_alias=True, exclude_defaults=True, exclude_none=True
    )
    cr_dict["spec"] = spec
    return cr_dict


# K8s `MultiRunConfig` (sweep_models.py) is `extra="forbid"` and only accepts
# this set of keys. Grid-recipe-only fields (`post_process`, `sla_filters`)
# get stripped before bubbling the recipe-driven multi_run block up to the
# AIPerfSweep CR -- those have no controller-side consumer yet.
_K8S_MULTIRUN_KEYS: frozenset[str] = frozenset(
    {
        "trials",
        "cooldown_seconds",
        "cooldownSeconds",
        "auto_set_seed",
        "autoSetSeed",
        "disable_warmup_after_first",
        "disableWarmupAfterFirst",
        "mode",
        "adaptive_search",
        "adaptiveSearch",
    }
)


def _apply_cli_overrides(
    *,
    bench_dict: dict[str, Any],
    user_config: UserConfig | None,
    sweep_cfg: Any,
    multirun_cfg_from_yaml: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """Merge user_config CLI overrides onto bench_dict; bubble recipe-driven
    `sweep` / `multi_run` blocks up to the AIPerfSweep spec.

    Recipe expansion (``--search-recipe X --ttft-sla-ms 200``) produces a
    sweep block (grid recipes) and/or a multi_run.adaptive_search block (BO
    recipes). Both belong on AIPerfSweep.spec, NOT on the embedded benchmark.
    YAML-declared sweep / multi_run keys win over recipe-driven ones; recipes
    only fill in when the YAML didn't already supply them.

    Returns (merged bench_dict, resolved sweep_cfg, resolved multirun_cfg).
    """
    from aiperf.cli_commands.kube._kube_common import _build_v1_overrides, _deep_merge

    if user_config is None:
        return bench_dict, sweep_cfg, multirun_cfg_from_yaml
    overrides = _build_v1_overrides(user_config)
    if not overrides:
        return bench_dict, sweep_cfg, multirun_cfg_from_yaml

    recipe_sweep = overrides.pop("sweep", None)
    recipe_multirun = overrides.pop("multi_run", None)
    if overrides:
        bench_dict = _deep_merge(bench_dict, overrides)
    if recipe_sweep is not None and sweep_cfg is None:
        sweep_cfg = recipe_sweep
    if recipe_multirun is not None and multirun_cfg_from_yaml is None:
        filtered = {k: v for k, v in recipe_multirun.items() if k in _K8S_MULTIRUN_KEYS}
        if filtered:
            multirun_cfg_from_yaml = filtered
    return bench_dict, sweep_cfg, multirun_cfg_from_yaml


def _name_from_config_file(config_file: Path) -> str:
    """Derive a DNS-1123 label name from ``config_file.stem``.

    Args:
        config_file: Path to the user's sweep YAML.

    Returns:
        ``"<stem>-sweep"``: lowercased, every disallowed char collapsed into a
        single ``-``, leading/trailing ``-`` stripped (both before AND after
        the 30-char truncation, so a cut that lands inside a hyphen run does
        not produce ``...---sweep``), with ``"aiperf"`` substituted when the
        stem sanitizes to empty (e.g. ``___.yaml``).

    The returned string always matches DNS-1123 label rules
    (``[a-z0-9]([-a-z0-9]*[a-z0-9])?``, ≤63 chars). The 30-char stem cap
    leaves head-room for ``-sweep`` (6) and any downstream
    ``-vNNNN-tNN`` suffixes (10) that ``derive_child_name`` appends.
    """
    stem = config_file.stem.lower()
    sanitized = re.sub(r"[^a-z0-9-]", "-", stem)
    # Collapse runs of '-' into a single '-' so '__a__b__' doesn't become
    # '--a--b--' which then survives the strip with embedded '--'.
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    safe_stem = sanitized[:30].rstrip("-")
    # Fall back when sanitization eats everything (all-underscore stems,
    # leading-only-special stems, empty stems). DNS-1123 labels must start
    # with [a-z0-9].
    if not safe_stem:
        safe_stem = "aiperf"
    return f"{safe_stem}-sweep"


async def _submit_sweep(
    *,
    cr_dict: dict[str, Any],
    kube_options: KubeOptions,
    namespace: str,
) -> None:
    """Apply the AIPerfSweep CR to the cluster via CustomObjectsApi."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client

    cr_dict["metadata"]["namespace"] = namespace
    async with k8s_client(
        kubeconfig=getattr(kube_options, "kubeconfig", None),
        context=getattr(kube_options, "kube_context", None),
    ) as api:
        custom = k8s.CustomObjectsApi(api)
        try:
            await custom.create_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfsweeps",
                body=cr_dict,
            )
        except k8s.ApiException as e:
            if getattr(e, "status", None) == 409:
                raise RuntimeError(
                    f"AIPerfSweep {namespace}/{cr_dict['metadata']['name']} "
                    "already exists. Pass --name to choose a different name, "
                    "or delete the existing CR first."
                ) from e
            raise
        kube_console.console.print(
            f"AIPerfSweep {namespace}/{cr_dict['metadata']['name']} created."
        )
        # Persist for `aiperf kube last-benchmark` parity with profile.
        kube_console.save_last_benchmark(
            cr_dict["metadata"]["name"],
            namespace,
            name=getattr(kube_options, "name", None),
        )
