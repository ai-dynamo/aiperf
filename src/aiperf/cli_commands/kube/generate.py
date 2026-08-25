# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube generate command: output Kubernetes YAML manifests to stdout."""

from __future__ import annotations

import sys
from typing import Annotated, Any

from cyclopts import App, Parameter

from aiperf.config import AIPerfConfig
from aiperf.config.flags.kube_cli_config import KubeCLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION

app = App(name="generate")

AIPERF_KIND = "AIPerfJob"
AIPERF_SWEEP_KIND = "AIPerfSweep"


def _choose_kind(envelope: AIPerfConfig) -> str:
    """Pick the CR kind matching the envelope's required execution path."""
    from aiperf.kubernetes.sweep_routing import requires_sweep_controller

    return AIPERF_SWEEP_KIND if requires_sweep_controller(envelope) else AIPERF_KIND


def _reject_orchestrated_direct_workload(config: AIPerfConfig) -> None:
    """Reject workloads that raw JobSet manifests cannot execute faithfully."""
    from aiperf.kubernetes.sweep_routing import requires_sweep_controller

    if not requires_sweep_controller(config):
        return

    from aiperf import cli_utils

    cli_utils.raise_startup_error_and_exit(
        "`--no-operator` can execute only one benchmark run; it cannot own "
        "parameter-sweep or multi-run orchestration.\n"
        "Use `aiperf kube generate --operator ...` to emit an AIPerfSweep CR, "
        "or submit it directly with `aiperf kube sweep ...`.",
        title="Kubernetes Operator Required",
    )


def _ensure_sweep_block(spec: dict[str, Any], config: AIPerfConfig) -> None:
    """Add a one-cell carrier when only ``multiRun`` requires orchestration."""
    if config.sweep is not None or spec.get("sweep") is not None:
        return
    from aiperf.kubernetes.sweep_routing import one_cell_sweep

    spec["sweep"] = one_cell_sweep(convergence=config.multi_run.convergence is not None)


def _build_sweep_spec(
    config: AIPerfConfig, kube_options: KubeOptions
) -> dict[str, Any]:
    """Build a flat AIPerfSweep CR spec from an envelope-shaped AIPerfConfig.

    Mirrors :meth:`KubeOptions.to_crd_spec` (used for AIPerfJob) while
    validating the sweep-specific envelope. Output is the flat envelope shape
    AIPerfSweepSpec expects — no template wrapping.
    """
    import math

    envelope = config.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    kube_options.apply_total_workers_override(envelope)
    if config._raw_envelope is not None:
        from aiperf.kubernetes.spec_converter import restore_jinja_templates

        envelope = restore_jinja_templates(envelope, config._raw_envelope)
    benchmark = envelope.pop("benchmark")

    dc = kube_options.to_deployment_config()
    from aiperf.common.endpoint_credentials import (
        validate_kubernetes_credential_transport,
    )

    validate_kubernetes_credential_transport(
        config.benchmark.endpoint, dc.pod_template.env
    )
    dc_dict = dc.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    if (
        "total_workers" in kube_options.model_fields_set
        and kube_options.total_workers > 0
    ):
        concurrency = max(
            (
                getattr(phase, "concurrency", 1) or 1
                for phase in config.benchmark.phases
            ),
            default=1,
        )
        dc_dict["connectionsPerWorker"] = max(
            1, math.ceil(concurrency / kube_options.total_workers)
        )

    sweep_dict = envelope.get("sweep")
    if (
        isinstance(sweep_dict, dict)
        and "type" not in sweep_dict
        and config.sweep is not None
    ):
        # Direct/programmatic configs can omit a defaulted discriminator from
        # their authored field set. The CRD union still requires the tag.
        sweep_dict["type"] = type(config.sweep).model_fields["type"].default

    spec = {"benchmark": benchmark, **envelope, **dc_dict}
    _ensure_sweep_block(spec, config)
    return spec


def _resolve_spec_and_name(
    cli_config: KubeCLIConfig,
    kube_options: KubeOptions,
):
    """Return (spec, config, name) from either an AIPerfJob CR file or CLI flags.

    A parameter sweep or multi-run workload is built for AIPerfSweep; an
    ordinary single run is built for AIPerfJob.
    """
    from aiperf.cli_commands.kube.profile import (
        _build_cr_spec_and_config,
        _resolve_config,
        _try_load_aiperfjob_cr,
        generate_benchmark_name,
    )

    config_file = cli_config.config_file
    cr_raw = _try_load_aiperfjob_cr(config_file) if config_file is not None else None
    if cr_raw is not None:
        # CR format: use spec as primary benchmark config; CLI K8s flags overlay
        spec, config = _build_cr_spec_and_config(
            cr_raw, kube_options, cli_config=cli_config
        )
        _ensure_sweep_block(spec, config)
        cr_name = cr_raw.get("metadata", {}).get("name")
        name = kube_options.name or cr_name or generate_benchmark_name(config)
    else:
        config = _resolve_config(cli_config, config_file)
        from aiperf.kubernetes.sweep_routing import requires_sweep_controller

        if requires_sweep_controller(config):
            spec = _build_sweep_spec(config, kube_options)
        else:
            spec = kube_options.to_crd_spec(config)
        name = kube_options.name or generate_benchmark_name(config)
    return spec, config, name


def _dump_raw_manifests(
    *,
    config,
    kube_options: KubeOptions,
    name: str,
    namespace: str,
    yaml,
    deployment_spec: dict[str, Any] | None = None,
):
    """Apply k8s runtime config and write raw manifests (Namespace, RBAC, ConfigMap, JobSet)."""
    from aiperf.cli_commands.kube._kube_common import resolve_total_workers
    from aiperf.config import AIPerfConfig
    from aiperf.kubernetes.environment import K8sEnvironment
    from aiperf.kubernetes.resources import KubernetesDeployment, NamespaceSpec
    from aiperf.kubernetes.spec_converter import (
        apply_k8s_runtime_config,
        apply_worker_config,
    )

    config_dict = config.model_dump(
        mode="python",
        exclude_unset=True,
        exclude_none=True,
        context={"include_secrets": True},
    )
    benchmark_dict = config_dict.get("benchmark", {})
    apply_k8s_runtime_config(benchmark_dict, name, namespace)
    config_dict["benchmark"] = benchmark_dict
    config = AIPerfConfig.model_validate(config_dict)

    if deployment_spec is None:
        deploy_config = kube_options.to_deployment_config()
    else:
        from aiperf.kubernetes.spec_converter import extract_deployment_config

        deploy_config = extract_deployment_config(deployment_spec)
    # Longer TTL without operator — pods must stay alive for manual
    # results retrieval via `aiperf kube results`.
    ttl_authored_in_spec = deployment_spec is not None and any(
        key in deployment_spec
        for key in ("ttlSecondsAfterFinished", "ttl_seconds_after_finished")
    )
    if "ttl_seconds" not in kube_options.model_fields_set and not ttl_authored_in_spec:
        deploy_config.ttl_seconds_after_finished = (
            K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS
        )
    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.benchmark.phases),
        default=1,
    )
    total_workers = resolve_total_workers(
        kube_options,
        concurrency=concurrency,
        connections_per_worker=deploy_config.connections_per_worker,
        configured_workers=config.benchmark.runtime.workers,
    )
    num_pods = apply_worker_config(config, total_workers)

    deployment = KubernetesDeployment(
        job_id=name,
        namespace=namespace,
        worker_replicas=num_pods,
        config=config,
        deployment=deploy_config,
    )
    from aiperf.common.endpoint_credentials import (
        validate_kubernetes_credential_transport,
    )

    validate_kubernetes_credential_transport(
        config.benchmark.endpoint, deploy_config.pod_template.env
    )

    manifests = deployment.get_all_manifests()
    if not any(manifest.get("kind") == "Namespace" for manifest in manifests):
        manifests.insert(0, NamespaceSpec(name=namespace).to_k8s_manifest())

    for i, manifest in enumerate(manifests):
        if i > 0:
            sys.stdout.write("---\n")
        yaml.dump(manifest, sys.stdout)
    return config


def _print_memory_estimate(config, kube_options: KubeOptions, spec) -> None:
    from aiperf.cli_commands.kube._kube_common import resolve_total_workers
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.benchmark.phases),
        default=1,
    )
    total_workers = resolve_total_workers(
        kube_options,
        concurrency=concurrency,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
        configured_workers=config.benchmark.runtime.workers,
    )
    mem_est = estimate_memory(
        config,
        total_workers=total_workers,
        workers_per_pod=config.benchmark.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    # Banner is informational; route through stderr_console so the YAML on
    # stdout stays a clean kubectl-pipeable stream.
    kube_console.stderr_console.print(f"\n{format_estimate(mem_est)}", highlight=False)


@app.default
async def generate(
    *,
    cli_config: KubeCLIConfig,
    kube_options: KubeOptions | None = None,
    operator: Annotated[
        bool,
        Parameter(
            name="--operator",
            negative=(),
            help="Output an AIPerfJob or AIPerfSweep CR (requires the operator).",
        ),
    ] = False,
    no_operator: Annotated[
        bool,
        Parameter(
            name="--no-operator",
            negative=(),
            help="Output raw K8s manifests (Namespace, RBAC, ConfigMap, JobSet).",
        ),
    ] = False,
) -> None:
    """Generate Kubernetes YAML manifests for an AIPerf benchmark.

    Specify --operator to output the matching workload CR (AIPerfJob for one
    run, AIPerfSweep for parameter sweeps and multi-run workloads), or
    --no-operator to output raw manifests (Namespace, RBAC, ConfigMap, JobSet)
    that work without the operator.

    Examples:
        # Generate the matching operator workload CR
        aiperf kube generate --operator --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

        # Generate raw manifests (no operator needed)
        aiperf kube generate --no-operator --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

        # Pipe directly to kubectl
        aiperf kube generate --no-operator ... | kubectl apply -f -
    """
    from aiperf import cli_utils

    if not operator and not no_operator:
        cli_utils.raise_startup_error_and_exit(
            "Specify --operator (AIPerfJob CR) or --no-operator (raw manifests)",
            title="Error Generating Kubernetes Manifests",
        )
    if operator and no_operator:
        cli_utils.raise_startup_error_and_exit(
            "Cannot use both --operator and --no-operator",
            title="Error Generating Kubernetes Manifests",
        )
    import ruamel.yaml

    kube_options = kube_options or KubeOptions()
    with cli_utils.exit_on_error(title="Error Generating Kubernetes Manifests"):
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        spec, config, name = _resolve_spec_and_name(cli_config, kube_options)
        namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE

        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False

        if no_operator:
            _reject_orchestrated_direct_workload(config)
            config = _dump_raw_manifests(
                config=config,
                kube_options=kube_options,
                name=name,
                namespace=namespace,
                yaml=yaml,
                deployment_spec=spec,
            )
        else:
            cr = {
                "apiVersion": AIPERF_API_VERSION,
                "kind": _choose_kind(config),
                "metadata": {"name": name, "namespace": namespace},
                "spec": spec,
            }
            yaml.dump(cr, sys.stdout)

        _print_memory_estimate(config, kube_options, spec)
