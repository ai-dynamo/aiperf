# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube results command: retrieve benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

if TYPE_CHECKING:
    from aiperf.kubernetes.cli_helpers import ResolvedSweep

app = App(name="results")


@app.default
async def results(
    job_id: Annotated[str | None, Parameter(help="The AIPerf job ID or AIPerfSweep name to get results from (default: last deployed job).")] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    output: Annotated[Path | None, Parameter(help="Output directory for results (default: ./artifacts/{name}).")] = None,
    from_pods: Annotated[bool, Parameter(name="--from-pods", help="Retrieve results from benchmark pods instead of the operator. Tries the controller API first, falls back to kubectl cp.")] = False,
    all_artifacts: Annotated[bool, Parameter(name=["--all", "-a"], negative="--summary-only", help="Download all artifacts. Use --summary-only to download only summary results.")] = True,
    shutdown: Annotated[bool, Parameter(name="--shutdown", help="Shut down the API service after downloading results. Only takes effect with --from-pods.")] = False,
    port: Annotated[int, Parameter(name="--port", help="Local port for API port-forward (default: 0 = ephemeral).")] = 0,
    operator_namespace: Annotated[str | None, Parameter(name="--operator-namespace", help="Namespace where the operator is deployed. Auto-detected (cluster-wide pod search) when omitted.")] = None,
    run: Annotated[str | None, Parameter(name="--run", help="Pin to a specific historical run (epoch from `aiperf kube results list-runs`). Default: latest.")] = None,
    variation: Annotated[int | None, Parameter(name=["-v", "--variation"], help="When job_id is an AIPerfSweep name, target child variation index (0..199). Resolves to <sweep>-v<idx:02d>[-t<trial>] and downloads that single child instead of the whole sweep.")] = None,
    trial: Annotated[int | None, Parameter(name=["-t", "--trial"], help="Trial index (0..9) within a sweep variation. Requires -v.")] = None,
) -> None:  # fmt: skip
    """Retrieve results from an AIPerf benchmark.

    Defaults to retrieving from the operator's PVC storage (works even after
    benchmark pods are deleted). Use --from-pods to retrieve directly from the
    benchmark pods: tries the controller API first, falls back to kubectl cp.
    Use --summary-only to download only summary results. Use --shutdown with
    --from-pods to shut down the API service after downloading, allowing the
    controller pod to exit cleanly. If no job_id is given, uses the last
    deployed benchmark. Use --run <epoch> to pin to a historical run (see
    ``aiperf kube results list-runs``).

    For an AIPerfSweep, pass the sweep name to fetch all child results, or
    add ``-v <idx>`` (and optional ``-t <trial>``) to download just one child
    variation as a single-job result.

    Examples:
        aiperf kube results                    # last deployed job (operator)
        aiperf kube results abc123             # specific job
        aiperf kube results --output ./out     # custom directory
        aiperf kube results --summary-only     # summary only
        aiperf kube results --from-pods        # from benchmark pods
        aiperf kube results --from-pods --shutdown
        aiperf kube results --run 1714150923   # pin to historical run
        aiperf kube results my-sweep -v 7      # single sweep variation
        aiperf kube results my-sweep -v 5 -t 0 # specific trial
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_name

    manage_options = manage_options or KubeManageOptions()
    if job_id is not None:
        child = resolve_child_name(job_id, variation=variation, trial=trial)
        if child is not None:
            job_id = child
    with cli_utils.exit_on_error(title="Error Retrieving Results"):
        success = await _run_results(
            job_id=job_id,
            manage_options=manage_options,
            output=output,
            from_pods=from_pods,
            all_artifacts=all_artifacts,
            shutdown=shutdown,
            port=port,
            operator_namespace=operator_namespace,
            run=run,
        )
        if not success:
            raise SystemExit(1)


# Alias retained for external callers / tests that import by name.
results_cmd = results


def _validate_run_arg(run: str | None, *, from_pods: bool) -> None:
    """Reject malformed ``--run`` values before any k8s/HTTP traffic."""
    from aiperf.operator.results_layout import EPOCH_RE

    if run is None:
        return
    if not EPOCH_RE.match(run):
        raise ValueError(
            f"Invalid --run value '{run}'. Expected decimal epoch-seconds from "
            "`aiperf kube results list-runs`."
        )
    if from_pods:
        raise ValueError(
            "--run is only supported for operator-backed downloads; "
            "drop --from-pods (benchmark pods only hold the latest run)."
        )


def _default_output_dir(
    *, output: Path | None, namespace: str, job_name: str, run: str | None
) -> Path:
    """Return ``output`` if provided, else the default artifact path.

    When ``run`` is set, the default embeds the namespace + job + epoch so
    historical downloads don't overwrite the latest-run directory.
    """
    if output is not None:
        return output
    if run is not None:
        return Path(f"./artifacts/{namespace}__{job_name}__{run}")
    return Path(f"./artifacts/{job_name}")


def _default_sweep_output_dir(
    namespace: str, sweep_name: str, *, run: str | None = None
) -> Path:
    """Default per-sweep output directory.

    Sweeps span many child jobs, so the default path embeds both the namespace
    and the sweep name to avoid colliding with single-job artifact dirs. When a
    historical run is pinned, include the epoch to avoid overwriting prior runs.
    """
    suffix = f"__{run}" if run is not None else ""
    return Path(f"./artifacts/sweep__{namespace}__{sweep_name}{suffix}")


async def _resolve_op_ns(
    api: object, *, explicit: str | None, quiet: bool = False
) -> str:
    """Resolve operator namespace, announcing auto-detected non-default values.

    Used by both ``aiperf kube results`` and ``aiperf kube results list-runs`` to
    keep the explicit-flag-vs-auto-detect-vs-default policy in one place. The
    ``quiet`` flag suppresses the announcement when JSON output is enabled.
    """
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import resolve_operator_namespace
    from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

    op_ns = await resolve_operator_namespace(api, explicit=explicit)
    if explicit is None and op_ns != DEFAULT_OPERATOR_NAMESPACE and not quiet:
        kube_console.print_info(f"Auto-detected operator namespace: {op_ns}")
    return op_ns


async def _run_results(
    *,
    job_id: str | None,
    manage_options: KubeManageOptions,
    output: Path | None,
    from_pods: bool,
    all_artifacts: bool,
    shutdown: bool,
    port: int,
    operator_namespace: str | None,
    run: str | None = None,
) -> bool:
    """Resolve a job or sweep and return whether every requested file arrived."""
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import results as kube_results
    from aiperf.kubernetes.cli_helpers import ResolvedSweep
    from aiperf.kubernetes.client import find_jobset

    _validate_run_arg(run, from_pods=from_pods)

    resolved = await cli_helpers.resolve_target(
        job_id,
        manage_options.namespace,
        kubeconfig=manage_options.kubeconfig,
        kube_context=manage_options.kube_context,
    )
    if not resolved:
        return False

    if isinstance(resolved, ResolvedSweep):
        return await _run_sweep_results(
            resolved=resolved,
            output=output,
            from_pods=from_pods,
            run=run,
            manage_options=manage_options,
            operator_namespace=operator_namespace,
            port=port,
        )

    job_id = resolved.job_id
    ns = resolved.namespace
    api = resolved.api

    try:
        op_ns = await _resolve_op_ns(api, explicit=operator_namespace)

        jobset_info = await find_jobset(api, job_id, ns)

        output_dir = _default_output_dir(
            output=output, namespace=ns, job_name=resolved.job_info.name, run=run
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        kube_creds = {
            "kubeconfig": manage_options.kubeconfig,
            "kube_context": manage_options.kube_context,
        }

        if from_pods:
            retrieval_success, used_api = await _retrieve_from_pods(
                job_id=job_id,
                ns=ns,
                output_dir=output_dir,
                jobset_info=jobset_info,
                api=api,
                port=port,
                all_artifacts=all_artifacts,
                kube_creds=kube_creds,
            )
        else:
            retrieval_success = await _retrieve_from_operator(
                job_id=job_id,
                ns=ns,
                output_dir=output_dir,
                api=api,
                port=port,
                op_ns=op_ns,
                run=run,
                kube_creds=kube_creds,
            )
            used_api = False

        if shutdown and used_api and retrieval_success:
            retrieval_success = await kube_results.shutdown_api_service(
                job_id, ns, api, port, **kube_creds
            )
        return retrieval_success
    finally:
        await resolved.aclose()


async def _run_sweep_results(
    *,
    resolved: ResolvedSweep,
    output: Path | None,
    from_pods: bool,
    run: str | None,
    manage_options: KubeManageOptions,
    operator_namespace: str | None,
    port: int,
) -> bool:
    """Sweep counterpart to the job branch of :func:`_run_results`.

    Rejects flags that don't apply to sweeps, resolves the operator
    namespace, fans out per-child via
    :func:`retrieve_sweep_results_from_operator`, prints a summary, and
    closes the resolver's API client.

    Returns:
        True only when every requested sweep artifact was downloaded.
    """
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes import results as kube_results

    try:
        if from_pods:
            kube_console.print_error(
                "--from-pods is not supported for AIPerfSweep CRs. "
                "Sweep child results live on the operator PVC; omit --from-pods."
            )
            return False
        op_ns = await _resolve_op_ns(resolved.api, explicit=operator_namespace)
        output_dir = output or _default_sweep_output_dir(
            resolved.namespace, resolved.name, run=run
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        ok = await kube_results.retrieve_sweep_results_from_operator(
            resolved.name,
            resolved.namespace,
            output_dir,
            resolved.api,
            local_port=port,
            operator_namespace=op_ns,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            run=run,
        )
        if ok:
            kube_console.print_results_summary(str(output_dir))
        else:
            kube_console.print_error(
                f"Sweep {resolved.namespace}/{resolved.name}: one or more aggregate artifacts "
                "or children failed to download (see errors above)."
            )
        return ok
    finally:
        await resolved.aclose()


async def _retrieve_from_operator(
    *,
    job_id: str,
    ns: str,
    output_dir: Path,
    api: object,
    port: int,
    op_ns: str,
    run: str | None,
    kube_creds: dict,
) -> bool:
    """Pull results via the operator's PVC-backed sidecar; print outcome.

    Split from ``_run_results`` so the parent stays under the 80-line ergonomics
    ceiling. Mirrors the ``_retrieve_from_pods`` shape: returns the success flag
    and prints a summary or actionable error to the kube console.
    """
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes import results as kube_results

    retrieval_success = await kube_results.retrieve_results_from_operator(
        job_id,
        ns,
        output_dir,
        api,
        local_port=port,
        operator_namespace=op_ns,
        run=run,
        **kube_creds,
    )
    if retrieval_success:
        kube_console.print_results_summary(str(output_dir))
    else:
        kube_console.print_error(
            f"Could not retrieve results from operator for job: {job_id}"
        )
        kube_console.print_info(
            "The operator may not have fetched results yet. "
            "Try --from-pods to retrieve directly from the benchmark pods."
        )
    return retrieval_success


async def _retrieve_from_pods(
    *,
    job_id: str,
    ns: str,
    output_dir: Path,
    jobset_info: object,
    api: object,
    port: int,
    all_artifacts: bool,
    kube_creds: dict,
) -> tuple[bool, bool]:
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes import results as kube_results

    if all_artifacts:
        retrieval_success = await kube_results.retrieve_all_artifacts(
            job_id,
            ns,
            output_dir,
            jobset_info,
            api,
            port,
            **kube_creds,
        )
        used_api = True
    else:
        # --summary-only: try API first, fall back to kubectl cp
        retrieval_success = await kube_results.retrieve_results_from_api(
            job_id,
            ns,
            output_dir,
            jobset_info,
            api,
            local_port=port,
            **kube_creds,
        )
        used_api = True
        if not retrieval_success:
            kube_console.print_warning(
                "Could not retrieve results from API. Trying kubectl cp..."
            )
            used_api = False
            if jobset_info:
                retrieval_success = await kube_results.retrieve_results_from_pod(
                    job_id,
                    ns,
                    output_dir,
                    jobset_info,
                    api,
                    **kube_creds,
                )

    if retrieval_success:
        kube_console.print_results_summary(str(output_dir))
    else:
        kube_console.print_error(
            f"Could not retrieve results from pods for job: {job_id}"
        )
        kube_console.print_info(
            "Pods may have been deleted. Try without --from-pods to retrieve from operator storage."
        )
    return retrieval_success, used_api


@app.command(name="list-runs")
async def list_runs(
    job_id: Annotated[str | None, Parameter(help="AIPerf job ID or AIPerfSweep name to list runs for (default: last deployed job).")] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    output: Annotated[Literal["text", "json"], Parameter(name=["-o", "--output"], help="Output format: 'text' for table, 'json' for machine-parseable.")] = "text",
    preview: Annotated[bool, Parameter(name="--preview", help="Show which runs would be reaped under current retention settings (read-only; no deletion).")] = False,
    operator_namespace: Annotated[str | None, Parameter(name="--operator-namespace", help="Namespace where the operator is deployed. Auto-detected (cluster-wide pod search) when omitted.")] = None,
    variation: Annotated[int | None, Parameter(name=["-v", "--variation"], help="When job_id is an AIPerfSweep name, target child variation index (0..199). Resolves to <sweep>-v<idx:02d>[-t<trial>].")] = None,
    trial: Annotated[int | None, Parameter(name=["-t", "--trial"], help="Trial index (0..9) within a sweep variation. Requires -v.")] = None,
) -> None:  # fmt: skip
    """List all historical runs of a benchmark job.

    Queries the operator's ``/api/v1/results/<ns>/<job_id>/runs`` endpoint and
    prints either a table (default) or the raw JSON payload. With ``--preview``,
    also fetches ``/api/v1/config/retention`` and annotates each row with
    whether the current policy would reap it (latest is always protected).

    Examples:
        aiperf kube results list-runs                 # last deployed job
        aiperf kube results list-runs foo             # specific job
        aiperf kube results list-runs foo --output json
        aiperf kube results list-runs foo --preview   # mark reap candidates
        aiperf kube results list-runs my-sweep -v 7   # specific sweep child
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_name

    manage_options = manage_options or KubeManageOptions()
    if job_id is not None:
        child = resolve_child_name(job_id, variation=variation, trial=trial)
        if child is not None:
            job_id = child
    with cli_utils.exit_on_error(title="Error Listing Runs"):
        success = await _run_list_runs(
            job_id=job_id,
            manage_options=manage_options,
            output=output,
            preview=preview,
            operator_namespace=operator_namespace,
        )
        if not success:
            raise SystemExit(1)


def _render_list_runs_payload(
    payload: dict, *, output: Literal["text", "json"], preview: bool
) -> None:
    """Render the list-runs payload as JSON or a text table."""
    import orjson

    from aiperf.cli_commands.kube import _runs_render

    if output == "json":
        from aiperf.kubernetes import console as kube_console

        kube_console.console.print(
            orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode(),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
    else:
        _runs_render.print_runs_table(payload, preview=preview)


async def _run_list_runs(
    *,
    job_id: str | None,
    manage_options: KubeManageOptions,
    output: Literal["text", "json"],
    preview: bool,
    operator_namespace: str | None,
) -> bool:
    """Fetch and render historical runs, returning False if no job resolves."""
    import logging

    from aiperf.cli_commands.kube import _runs_render
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes.client import find_operator_pod
    from aiperf.kubernetes.port_forward import port_forward_with_status
    from aiperf.kubernetes.results_operator import RESULTS_SERVER_PORT

    kube_logger = logging.getLogger("aiperf.kube")
    original_level = kube_logger.level
    if output == "json":
        kube_logger.setLevel(logging.WARNING)

    resolved = None
    try:
        resolved = await cli_helpers.resolve_job(
            job_id,
            manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            quiet=output == "json",
        )
        if not resolved:
            return False

        job_id = resolved.job_id
        namespace = resolved.namespace
        api = resolved.api

        op_ns = await _resolve_op_ns(
            api, explicit=operator_namespace, quiet=output == "json"
        )

        pod_info = await find_operator_pod(api, namespace=op_ns)
        if not pod_info:
            raise RuntimeError(
                f"Operator pod not found in namespace '{op_ns}'. "
                "Is the aiperf-operator deployed?"
            )
        pod_name, _phase = pod_info

        async with port_forward_with_status(
            op_ns,
            pod_name,
            0,
            remote_port=RESULTS_SERVER_PORT,
            verify_api=False,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        ) as port:
            payload, retention = await _fetch_runs_and_retention(
                base_url=f"http://localhost:{port}",
                namespace=namespace,
                job_id=job_id,
                preview=preview,
            )
    finally:
        kube_logger.setLevel(original_level)
        if resolved is not None:
            await resolved.aclose()

    if preview and retention is not None:
        _runs_render.annotate_preview(payload, retention)
    _render_list_runs_payload(payload, output=output, preview=preview)
    return True


async def _fetch_runs_and_retention(
    *,
    base_url: str,
    namespace: str,
    job_id: str,
    preview: bool,
) -> tuple[dict, dict | None]:
    """GET ``/runs`` (+ optionally ``/config/retention``) and return their payloads.

    Split from ``_run_list_runs`` to keep each function below the 80-line
    ergonomics ceiling — the port-forward/logging ceremony belongs in the
    outer function, the HTTP shape belongs here.
    """
    import aiohttp
    import orjson

    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=30)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        runs_url = f"{base_url}/api/v1/results/{namespace}/{job_id}/runs"
        async with session.get(runs_url) as resp:
            if resp.status == 404:
                raise RuntimeError(
                    f"No runs found for {namespace}/{job_id}. "
                    "The job may not have completed yet, or the operator "
                    "has not captured any runs."
                )
            resp.raise_for_status()
            payload = await resp.json(loads=orjson.loads)

        retention: dict | None = None
        if preview:
            async with session.get(f"{base_url}/api/v1/config/retention") as r2:
                r2.raise_for_status()
                retention = await r2.json(loads=orjson.loads)

    return payload, retention
