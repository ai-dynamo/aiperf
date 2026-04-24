# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube generate command: output Kubernetes YAML manifests to stdout."""

from __future__ import annotations

import sys
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.config.cli_model import CLIModel
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION

app = App(name="generate")

AIPERF_KIND = "AIPerfJob"


def _resolve_spec_and_name(cli_model: CLIModel, kube_options: KubeOptions):
    """Return (spec, config, name) from either an AIPerfJob CR file or CLI flags."""
    from aiperf.cli_commands.kube.profile import (
        _build_cr_spec_and_config,
        _resolve_config,
        _try_load_aiperfjob_cr,
        generate_benchmark_name,
    )

    config_file = getattr(cli_model, "config_file", None)
    cr_raw = _try_load_aiperfjob_cr(config_file) if config_file is not None else None
    if cr_raw is not None:
        # CR format: use spec as primary benchmark config; CLI K8s flags overlay
        spec, config = _build_cr_spec_and_config(cr_raw, kube_options)
        cr_name = cr_raw.get("metadata", {}).get("name")
        name = kube_options.name or cr_name or generate_benchmark_name(config)
    else:
        config = _resolve_config(cli_model, config_file)
        spec = kube_options.to_crd_spec(config)
        name = kube_options.name or generate_benchmark_name(config)
    return spec, config, name


def _dump_raw_manifests(
    *, config, kube_options: KubeOptions, name: str, namespace: str, yaml
):
    """Apply k8s runtime config and write raw manifests (Namespace, RBAC, ConfigMap, JobSet)."""
    import math

    from aiperf.config import AIPerfConfig
    from aiperf.kubernetes.environment import K8sEnvironment
    from aiperf.kubernetes.resources import KubernetesDeployment
    from aiperf.operator.spec_converter import (
        apply_k8s_runtime_config,
        apply_worker_config,
    )

    config_dict = config.model_dump(mode="json", exclude_none=True)
    apply_k8s_runtime_config(config_dict, name, namespace)
    config = AIPerfConfig.model_validate(config_dict)

    deploy_config = kube_options.to_deployment_config()
    # Longer TTL without operator — pods must stay alive for manual
    # results retrieval via `aiperf kube results`.
    if "ttl_seconds" not in kube_options.model_fields_set:
        deploy_config.ttl_seconds_after_finished = (
            K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS
        )
    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.phases.values()),
        default=1,
    )
    total_workers = max(
        1, math.ceil(concurrency / deploy_config.connections_per_worker)
    )
    num_pods = apply_worker_config(config, total_workers)

    deployment = KubernetesDeployment(
        job_id=name,
        namespace=namespace,
        worker_replicas=num_pods,
        config=config,
        deployment=deploy_config,
    )

    for i, manifest in enumerate(deployment.get_all_manifests()):
        if i > 0:
            sys.stdout.write("---\n")
        yaml.dump(manifest, sys.stdout)
    return config


def _print_memory_estimate(config, kube_options: KubeOptions, spec) -> None:
    from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

    mem_est = estimate_memory(
        config,
        total_workers=kube_options.workers,
        workers_per_pod=config.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    print(f"\n{format_estimate(mem_est)}", file=sys.stderr)


@app.default
async def generate(
    *,
    cli_model: CLIModel,
    kube_options: KubeOptions,
    operator: Annotated[
        bool,
        Parameter(
            name="--operator",
            negative=(),
            help="Output an AIPerfJob CR (requires operator on target cluster).",
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

    Specify --operator to output an AIPerfJob CR (requires the operator to be
    installed on the target cluster), or --no-operator to output raw manifests
    (Namespace, RBAC, ConfigMap, JobSet) that work without the operator.

    Examples:
        # Generate AIPerfJob CR (operator mode)
        aiperf kube generate --operator --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

        # Generate raw manifests (no operator needed)
        aiperf kube generate --no-operator --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

        # Pipe directly to kubectl
        aiperf kube generate --no-operator ... | kubectl apply -f -
    """
    if not operator and not no_operator:
        raise SystemExit(
            "Specify --operator (AIPerfJob CR) or --no-operator (raw manifests)"
        )
    if operator and no_operator:
        raise SystemExit("Cannot use both --operator and --no-operator")
    import ruamel.yaml

    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Generating Kubernetes Manifests"):
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        spec, config, name = _resolve_spec_and_name(cli_model, kube_options)
        namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE

        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False

        if no_operator:
            config = _dump_raw_manifests(
                config=config,
                kube_options=kube_options,
                name=name,
                namespace=namespace,
                yaml=yaml,
            )
        else:
            cr = {
                "apiVersion": AIPERF_API_VERSION,
                "kind": AIPERF_KIND,
                "metadata": {"name": name, "namespace": namespace},
                "spec": spec,
            }
            yaml.dump(cr, sys.stdout)

        _print_memory_estimate(config, kube_options, spec)
