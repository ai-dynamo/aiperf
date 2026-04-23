# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube profile command: create an AIPerfJob CR to run a benchmark."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.cli_model import CLIModel
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION

if TYPE_CHECKING:
    from kubernetes_asyncio.client import (
        CoreV1Api,
        CustomObjectsApi,
        RbacAuthorizationV1Api,
    )

    from aiperf.config import AIPerfConfig

app = App(name="profile")

AIPERF_KIND = "AIPerfJob"


def _try_load_aiperfjob_cr(path: Any) -> dict | None:
    """Parse path as YAML and return the raw dict if it is an AIPerfJob CR.

    Returns None if the file cannot be parsed or is not an AIPerfJob CR.
    The caller owns the single file read; no further reads are needed.
    """
    import yaml

    try:
        raw = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - any YAML/IO failure means "not an AIPerfJob CR"; caller handles other paths
        return None
    if (
        isinstance(raw, dict)
        and raw.get("apiVersion", "").startswith("aiperf.nvidia.com")
        and raw.get("kind") == AIPERF_KIND
    ):
        return raw
    return None


def _build_cr_spec_and_config(raw: dict, kube_options: Any) -> tuple[dict, Any]:
    """Build (overlaid_spec, AIPerfConfig) from a parsed AIPerfJob CR dict.

    Extracts benchmark config from the CR spec, then overlays CLI K8s
    deployment options (image, podTemplate, workers, etc.) on top.
    The returned spec is ready to submit to the operator.
    """
    import copy
    import math

    from aiperf.operator.spec_converter import extract_benchmark_config

    spec = copy.deepcopy(dict(raw.get("spec", {})))
    config = extract_benchmark_config(spec)

    dc = kube_options.to_deployment_config()
    dc_dict = dc.model_dump(mode="json", by_alias=True, exclude_defaults=True)

    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.phases.values()),
        default=1,
    )
    dc_dict["connectionsPerWorker"] = max(
        1, math.ceil(concurrency / kube_options.workers)
    )

    spec.update(dc_dict)
    return spec, config


def generate_benchmark_name(config: AIPerfConfig) -> str:
    """Generate a short benchmark name from config.

    Used by both profile and generate commands.

    Args:
        config: AIPerfConfig instance.

    Returns:
        A short hyphenated name like "qwen3-openai-throughput".
    """
    import re

    model_name = config.get_model_names()[0].split("/")[-1].lower()
    endpoint_type = str(config.endpoint.type)
    first_phase = next(iter(config.phases.values()))
    phase_type = str(first_phase.type)
    raw = "-".join([model_name, endpoint_type, phase_type])
    # Sanitize to valid DNS label: replace invalid chars, strip leading/trailing hyphens
    return re.sub(r"[^a-z0-9-]", "-", raw).strip("-")[:40]


def _build_cr(
    name: str,
    namespace: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete AIPerfJob CR dict."""
    return {
        "apiVersion": AIPERF_API_VERSION,
        "kind": AIPERF_KIND,
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": spec,
    }


def _resolve_config(cli_model: Any, config_file: Any) -> Any:
    """Return AIPerfConfig from a plain YAML file or CLI flags."""
    if config_file is not None:
        from aiperf.config.loader import load_config

        return load_config(config_file)
    from aiperf.config.cli_converter import build_aiperf_config

    return build_aiperf_config(cli_model)


@app.default
async def profile(
    *,
    cli_model: CLIModel,
    kube_options: KubeOptions,
    detach: Annotated[
        bool,
        Parameter(
            name=["-d", "--detach"],
            help="Exit immediately after deploying (don't wait for completion). Automatically enabled in non-interactive environments (pipes, CI/CD).",
        ),
    ] = False,
    no_wait: Annotated[
        bool,
        Parameter(
            name="--no-wait",
            negative=(),
            help="Don't wait for pods to be ready before attaching (advanced).",
        ),
    ] = False,
    attach_port: Annotated[
        int,
        Parameter(
            name="--attach-port",
            help="Local port for API port-forward (default: 0 = ephemeral).",
        ),
    ] = 0,
    skip_endpoint_check: Annotated[
        bool,
        Parameter(
            name="--skip-endpoint-check",
            negative=(),
            help="Skip endpoint health validation before deploying.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            negative=(),
            help="Print the AIPerfJob CR as JSON without submitting it.",
        ),
    ] = False,
    no_operator: Annotated[
        bool,
        Parameter(
            name="--no-operator",
            negative=(),
            help="Force direct deployment without the operator. Automatically enabled if the AIPerfJob CRD is not installed on the cluster.",
        ),
    ] = False,
) -> None:
    """Run a benchmark in Kubernetes.

    Auto-detects whether the AIPerf operator is installed. If the AIPerfJob
    CRD exists, creates a CR and lets the operator handle deployment. Otherwise,
    falls back to direct manifest creation (JobSet, ConfigMap, RBAC).
    Use --no-operator to force direct mode.

    Examples:
        # Auto-detect (operator if available, direct otherwise)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --workers-max 10

        # Force direct mode (no operator)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --no-operator

        # CI/CD: deploy and exit immediately
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --detach
    """

    from aiperf import cli_utils
    from aiperf.kubernetes import console as kube_console

    with cli_utils.exit_on_error(title="Error Running Kubernetes Benchmark"):
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        config_file = getattr(cli_model, "config_file", None)
        cr_raw = (
            _try_load_aiperfjob_cr(config_file) if config_file is not None else None
        )
        if cr_raw is not None:
            # CR format: use spec as primary benchmark config; CLI K8s flags overlay
            spec, config = _build_cr_spec_and_config(cr_raw, kube_options)
            cr_name = cr_raw.get("metadata", {}).get("name")
            name = kube_options.name or cr_name or generate_benchmark_name(config)
        else:
            config = _resolve_config(cli_model, config_file)
            spec = kube_options.to_crd_spec(config)
            name = kube_options.name or generate_benchmark_name(config)

        namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE

        # Print memory estimate
        from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

        mem_est = estimate_memory(
            config,
            total_workers=kube_options.workers,
            workers_per_pod=config.runtime.workers_per_pod,
            connections_per_worker=spec.get("connectionsPerWorker", 100),
        )
        kube_console.console.print(format_estimate(mem_est), highlight=False)

        use_operator = not no_operator
        if use_operator and not dry_run:
            use_operator = await _operator_available(kube_options)

        if use_operator:
            await _deploy_via_operator(
                spec,
                kube_options,
                config,
                name,
                namespace,
                dry_run=dry_run,
                detach=detach,
                no_wait=no_wait,
                attach_port=attach_port,
            )
        else:
            await _deploy_direct(
                config,
                kube_options,
                name,
                namespace,
                dry_run=dry_run,
                detach=detach,
                no_wait=no_wait,
                attach_port=attach_port,
            )


async def _wait_or_detach(
    name: str,
    namespace: str,
    kube_options: KubeOptions,
    *,
    detach: bool,
    no_wait: bool,
    attach_port: int,
    hint: str = "",
) -> None:
    """Either attach to the benchmark or detach with a status message."""
    import sys

    from aiperf.kubernetes import console as kube_console

    is_interactive = sys.stdout.isatty()
    should_detach = detach or not is_interactive

    if not is_interactive and not detach:
        kube_console.print_warning(
            "Non-interactive environment detected, using detach mode"
        )

    if should_detach:
        kube_console.print_detach_info(name, namespace, name=kube_options.name)
        if hint:
            kube_console.print_info(hint)
        return

    try:
        from aiperf.kubernetes import attach as kube_attach

        await kube_attach.auto_attach_workflow(
            name,
            namespace,
            attach_port,
            wait_for_ready=not no_wait,
            kubeconfig=kube_options.kubeconfig,
            kube_context=kube_options.kube_context,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        kube_console.print_interrupt_info(name, namespace)
        if hint:
            kube_console.print_info(hint)


async def _operator_available(kube_options: KubeOptions) -> bool:
    """Check if the AIPerfJob CRD is installed on the cluster.

    Returns True if the operator CRD exists (operator mode), False otherwise
    (direct mode). Logs which mode is selected.
    """
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import AIPERF_JOB_GROUP, AIPERF_JOB_PLURAL

    crd_name = f"{AIPERF_JOB_PLURAL}.{AIPERF_JOB_GROUP}"
    try:
        async with k8s_client(
            kubeconfig=kube_options.kubeconfig,
            context=kube_options.kube_context,
        ) as api:
            await k8s_client_mod.ApiextensionsV1Api(
                api
            ).read_custom_resource_definition(
                crd_name,
            )
        kube_console.print_info("AIPerfJob CRD detected, using operator mode")
        return True
    except ApiException as e:
        if e.status != 404:
            kube_console.print_info(
                f"AIPerfJob CRD not found, deploying directly (no operator) [ApiException: {e}]"
            )
            return False
        kube_console.print_info(
            "AIPerfJob CRD not found, deploying directly (no operator)"
        )
        return False
    except Exception as e:  # noqa: BLE001 - any unrecognized error falls back to direct-deploy mode with a user-facing message
        kube_console.print_info(
            f"AIPerfJob CRD not found, deploying directly (no operator) [{type(e).__name__}: {e}]"
        )
        return False


async def _replace_existing_cr_if_complete(
    custom: CustomObjectsApi,
    *,
    name: str,
    namespace: str,
    kube_context: str | None,
) -> None:
    """Delete a prior AIPerfJob CR with matching name if it finished.

    Raises ``SystemExit`` if the CR is still Running or Pending; returns
    silently when there is no prior CR (404).
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.cr_refs import (
        AIPERF_JOB_GROUP,
        AIPERF_JOB_PLURAL,
        AIPERF_JOB_VERSION,
    )

    try:
        existing = await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=name,
        )
    except ApiException as e:
        if e.status == 404:
            return
        raise

    phase = (existing.get("status") or {}).get("phase", "")
    if phase in ("Running", "Pending"):
        ctx_flag = f" --context {kube_context}" if kube_context else ""
        raise SystemExit(
            f"AIPerfJob {name} is already {phase}. "
            f"Delete it first: kubectl{ctx_flag} delete aiperfjob {name} -n {namespace}"
        )
    kube_console.print_info(f"Replacing completed AIPerfJob {name}")
    await custom.delete_namespaced_custom_object(
        group=AIPERF_JOB_GROUP,
        version=AIPERF_JOB_VERSION,
        plural=AIPERF_JOB_PLURAL,
        namespace=namespace,
        name=name,
    )
    await asyncio.sleep(2)


async def _deploy_via_operator(
    spec: dict[str, Any],
    kube_options: KubeOptions,
    config: Any,
    name: str,
    namespace: str,
    *,
    dry_run: bool,
    detach: bool,
    no_wait: bool,
    attach_port: int,
) -> None:
    """Deploy by creating an AIPerfJob CR (requires operator)."""

    from aiperf.kubernetes import console as kube_console

    cr = _build_cr(name, namespace, spec)

    if dry_run:
        import orjson

        output = orjson.dumps(cr, option=orjson.OPT_INDENT_2).decode()
        kube_console.console.print(output, highlight=False)
        return

    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import (
        AIPERF_JOB_GROUP,
        AIPERF_JOB_PLURAL,
        AIPERF_JOB_VERSION,
    )

    async with k8s_client(
        kubeconfig=kube_options.kubeconfig,
        context=kube_options.kube_context,
    ) as api:
        core = k8s_client_mod.CoreV1Api(api)
        custom = k8s_client_mod.CustomObjectsApi(api)

        try:
            await core.create_namespace(
                body=k8s_client_mod.V1Namespace(
                    metadata=k8s_client_mod.V1ObjectMeta(name=namespace),
                )
            )
        except ApiException as e:
            if e.status != 409:
                raise

        await _replace_existing_cr_if_complete(
            custom,
            name=name,
            namespace=namespace,
            kube_context=kube_options.kube_context,
        )

        try:
            await custom.create_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                body=cr,
            )
        except ApiException as e:
            import orjson

            detail = ""
            if e.body:
                try:
                    body = orjson.loads(e.body)
                    detail = body.get("message", "")
                except (orjson.JSONDecodeError, TypeError):
                    detail = e.body[:200] if e.body else ""
            raise SystemExit(
                f"Failed to create AIPerfJob {namespace}/{name}: {detail}"
            ) from e

    kube_console.print_cr_submission_summary(
        name=name,
        namespace=namespace,
        image=kube_options.image,
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
        model_names=config.get_model_names(),
        connections_per_worker=spec.get("connectionsPerWorker"),
    )

    kube_console.save_last_benchmark(name, namespace, name=kube_options.name)

    if detach:
        ctx_flag = (
            f" --kube-context {kube_options.kube_context}"
            if kube_options.kube_context
            else ""
        )
        kube_console.print_info(
            f"Detached. Watch with: aiperf kube watch {name}{ctx_flag}"
        )
        return

    from aiperf.kubernetes.attach import watch_job

    await watch_job(
        namespace=namespace,
        job_id=name,
        timeout=600,
        kubeconfig=kube_options.kubeconfig,
        kube_context=kube_options.kube_context,
    )


async def _apply_manifest(
    manifest: dict[str, Any],
    *,
    core: CoreV1Api,
    rbac: RbacAuthorizationV1Api,
    custom: CustomObjectsApi,
    default_namespace: str,
) -> str | None:
    """Create one K8s resource from a manifest.

    Returns the ``"Kind/name"`` label on success (so the caller can log it),
    or None if the kind is not recognised. 409 AlreadyExists is not handled
    here; the caller catches and distinguishes it.
    """
    kind = manifest["kind"]
    res_name = manifest["metadata"]["name"]
    manifest_ns = manifest["metadata"].get("namespace") or default_namespace
    label = f"{kind}/{res_name}"

    from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION

    if kind == "Namespace":
        await core.create_namespace(body=manifest)
    elif kind == "ConfigMap":
        await core.create_namespaced_config_map(namespace=manifest_ns, body=manifest)
    elif kind == "Role":
        await rbac.create_namespaced_role(namespace=manifest_ns, body=manifest)
    elif kind == "RoleBinding":
        await rbac.create_namespaced_role_binding(namespace=manifest_ns, body=manifest)
    elif kind == "JobSet":
        await custom.create_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=manifest_ns,
            body=manifest,
        )
    else:
        return None
    return label


def _print_manifests_yaml(manifests: list[dict[str, Any]]) -> None:
    """Emit all manifests as a multi-document YAML stream to stdout."""
    import sys

    import ruamel.yaml

    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False
    for i, manifest in enumerate(manifests):
        if i > 0:
            sys.stdout.write("---\n")
        yaml.dump(manifest, sys.stdout)


async def _deploy_direct(
    config: Any,
    kube_options: KubeOptions,
    name: str,
    namespace: str,
    *,
    dry_run: bool,
    detach: bool,
    no_wait: bool,
    attach_port: int,
) -> None:
    """Deploy directly without the operator (creates all K8s resources)."""
    import math

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.resources import KubernetesDeployment
    from aiperf.operator.spec_converter import (
        apply_k8s_runtime_config,
        apply_worker_config,
    )

    # Apply K8s runtime config (ZMQ dual-bind, API service, etc.)
    config_dict = config.model_dump(mode="json", exclude_none=True)
    apply_k8s_runtime_config(config_dict, name, namespace)
    from aiperf.config import AIPerfConfig

    config = AIPerfConfig.model_validate(config_dict)

    deploy_config = kube_options.to_deployment_config()

    # In direct mode (no operator), the controller pod keeps the API alive
    # for results retrieval. Use a longer TTL so pods stay available until
    # the user runs `aiperf kube results`. The user can send --shutdown to
    # release resources after downloading.
    # Only bump TTL when the user didn't explicitly set --ttl-seconds.
    # KubeOptions defaults to 300; if it's still that default, apply the
    # longer direct-mode TTL so pods stay alive for results retrieval.
    from aiperf.kubernetes.environment import K8sEnvironment

    if "ttl_seconds" not in kube_options.model_fields_set:
        deploy_config.ttl_seconds_after_finished = (
            K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS
        )

    # Calculate workers
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
        namespace=kube_options.namespace,  # None → auto_namespace creates the namespace
        worker_replicas=num_pods,
        config=config,
        deployment=deploy_config,
        model_names=config.get_model_names(),
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
    )
    effective_ns = deployment.effective_namespace

    manifests = deployment.get_all_manifests()

    if dry_run:
        _print_manifests_yaml(manifests)
        return

    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client

    async with k8s_client(
        kubeconfig=kube_options.kubeconfig,
        context=kube_options.kube_context,
    ) as api:
        core = k8s_client_mod.CoreV1Api(api)
        rbac = k8s_client_mod.RbacAuthorizationV1Api(api)
        custom = k8s_client_mod.CustomObjectsApi(api)

        for manifest in manifests:
            kind = manifest["kind"]
            res_name = manifest["metadata"]["name"]
            try:
                label = await _apply_manifest(
                    manifest,
                    core=core,
                    rbac=rbac,
                    custom=custom,
                    default_namespace=effective_ns,
                )
            except ApiException as exc:
                if exc.status == 409:
                    kube_console.print_info(f"{kind}/{res_name} already exists")
                    continue
                raise
            if label is None:
                kube_console.print_warning(f"Unknown resource kind: {kind}, skipping")
                continue
            kube_console.print_success(f"Created {label}")

    kube_console.print_cr_submission_summary(
        name=name,
        namespace=effective_ns,
        image=kube_options.image,
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
        model_names=config.get_model_names(),
        connections_per_worker=deploy_config.connections_per_worker,
    )

    kube_console.save_last_benchmark(name, effective_ns, name=kube_options.name)

    await _wait_or_detach(
        name,
        effective_ns,
        kube_options,
        detach=detach,
        no_wait=no_wait,
        attach_port=attach_port,
        hint="Retrieve results: aiperf kube results --shutdown",
    )
