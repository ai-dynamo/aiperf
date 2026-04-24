# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube results command: retrieve benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="results")


@app.default
async def results(
    job_id: Annotated[str | None, Parameter(help="The AIPerf job ID to get results from (default: last deployed job).")] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    output: Annotated[Path | None, Parameter(help="Output directory for results (default: ./artifacts/{name}).")] = None,
    from_pods: Annotated[bool, Parameter(name="--from-pods", help="Retrieve results from benchmark pods instead of the operator. Tries the controller API first, falls back to kubectl cp.")] = False,
    all_artifacts: Annotated[bool, Parameter(name=["--all", "-a"], negative="--summary-only", help="Download all artifacts. Use --summary-only to download only summary results.")] = True,
    shutdown: Annotated[bool, Parameter(name="--shutdown", help="Shut down the API service after downloading results. Only takes effect with --from-pods.")] = False,
    port: Annotated[int, Parameter(name="--port", help="Local port for API port-forward (default: 0 = ephemeral).")] = 0,
    operator_namespace: Annotated[str, Parameter(name="--operator-namespace", help="Namespace where the operator is deployed.")] = "aiperf-system",
) -> None:  # fmt: skip
    """Retrieve results from an AIPerf benchmark.

    Defaults to retrieving from the operator's PVC storage (works even after
    benchmark pods are deleted). Use --from-pods to retrieve directly from the
    benchmark pods: tries the controller API first, falls back to kubectl cp.
    Use --summary-only to download only summary results. Use --shutdown with
    --from-pods to shut down the API service after downloading, allowing the
    controller pod to exit cleanly. If no job_id is given, uses the last
    deployed benchmark.

    Examples:
        aiperf kube results                    # last deployed job (operator)
        aiperf kube results abc123             # specific job
        aiperf kube results --output ./out     # custom directory
        aiperf kube results --summary-only     # summary only
        aiperf kube results --from-pods        # from benchmark pods
        aiperf kube results --from-pods --shutdown
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()
    with cli_utils.exit_on_error(title="Error Retrieving Results"):
        await _run_results(
            job_id=job_id,
            manage_options=manage_options,
            output=output,
            from_pods=from_pods,
            all_artifacts=all_artifacts,
            shutdown=shutdown,
            port=port,
            operator_namespace=operator_namespace,
        )


# Alias retained for external callers / tests that import by name.
results_cmd = results


async def _run_results(
    *,
    job_id: str | None,
    manage_options: KubeManageOptions,
    output: Path | None,
    from_pods: bool,
    all_artifacts: bool,
    shutdown: bool,
    port: int,
    operator_namespace: str,
) -> None:
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes import results as kube_results
    from aiperf.kubernetes.client import find_jobset

    resolved = await cli_helpers.resolve_job(
        job_id,
        manage_options.namespace,
        kubeconfig=manage_options.kubeconfig,
        kube_context=manage_options.kube_context,
    )
    if not resolved:
        return

    job_id = resolved.job_id
    ns = resolved.namespace
    api = resolved.api

    jobset_info = await find_jobset(api, job_id, ns)

    output_dir = output or Path(f"./artifacts/{resolved.job_info.name}")
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
        retrieval_success = await kube_results.retrieve_results_from_operator(
            job_id,
            ns,
            output_dir,
            api,
            local_port=port,
            operator_namespace=operator_namespace,
            **kube_creds,
        )
        used_api = False
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

    if shutdown and used_api and retrieval_success:
        await kube_results.shutdown_api_service(job_id, ns, api, port, **kube_creds)


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
    job_id: Annotated[str | None, Parameter(help="AIPerf job ID to list runs for (default: last deployed job).")] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    output: Annotated[Literal["text", "json"], Parameter(name=["-o", "--output"], help="Output format: 'text' for table, 'json' for machine-parseable.")] = "text",
    operator_namespace: Annotated[str, Parameter(name="--operator-namespace", help="Namespace where the operator is deployed.")] = "aiperf-system",
) -> None:  # fmt: skip
    """List all historical runs of a benchmark job.

    Queries the operator's ``/api/v1/results/<ns>/<job_id>/runs`` endpoint and
    prints either a table (default) or the raw JSON payload.

    Examples:
        aiperf kube results list-runs                 # last deployed job
        aiperf kube results list-runs foo             # specific job
        aiperf kube results list-runs foo --output json
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()
    with cli_utils.exit_on_error(title="Error Listing Runs"):
        await _run_list_runs(
            job_id=job_id,
            manage_options=manage_options,
            output=output,
            operator_namespace=operator_namespace,
        )


async def _run_list_runs(
    *,
    job_id: str | None,
    manage_options: KubeManageOptions,
    output: Literal["text", "json"],
    operator_namespace: str,
) -> None:
    import logging

    import aiohttp
    import orjson

    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import find_operator_pod
    from aiperf.kubernetes.port_forward import port_forward_with_status
    from aiperf.kubernetes.results_operator import RESULTS_SERVER_PORT
    from aiperf.transports.aiohttp_client import create_tcp_connector

    kube_logger = logging.getLogger("aiperf.kube")
    original_level = kube_logger.level
    if output == "json":
        kube_logger.setLevel(logging.WARNING)

    try:
        resolved = await cli_helpers.resolve_job(
            job_id,
            manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
        if not resolved:
            return

        job_id = resolved.job_id
        namespace = resolved.namespace
        api = resolved.api

        pod_info = await find_operator_pod(api, namespace=operator_namespace)
        if not pod_info:
            raise RuntimeError(
                f"Operator pod not found in namespace '{operator_namespace}'. "
                "Is the aiperf-operator deployed?"
            )
        pod_name, _phase = pod_info

        async with port_forward_with_status(
            operator_namespace,
            pod_name,
            0,
            remote_port=RESULTS_SERVER_PORT,
            verify_api=False,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        ) as port:
            url = f"http://localhost:{port}/api/v1/results/{namespace}/{job_id}/runs"
            timeout = aiohttp.ClientTimeout(total=30)
            connector = create_tcp_connector()
            async with (
                aiohttp.ClientSession(timeout=timeout, connector=connector) as session,
                session.get(url) as resp,
            ):
                if resp.status == 404:
                    raise RuntimeError(
                        f"No runs found for {namespace}/{job_id}. "
                        "The job may not have completed yet, or the operator "
                        "has not captured any runs."
                    )
                resp.raise_for_status()
                payload = await resp.json(loads=orjson.loads)
    finally:
        kube_logger.setLevel(original_level)

    if output == "json":
        kube_console.console.print(
            orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode(),
            highlight=False,
        )
    else:
        _print_runs_table(payload)


def _print_runs_table(payload: dict) -> None:
    """Render a ``RunHistoryListResponse`` payload as a rich table."""
    from datetime import datetime, timezone

    from rich.table import Table

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.console import _human_size

    runs = payload.get("runs", [])
    namespace = payload.get("namespace", "")
    job_id = payload.get("job_id", "")

    if not runs:
        kube_console.print_info(f"No runs found for {namespace}/{job_id}")
        return

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("EPOCH", style="cyan")
    table.add_column("TIMESTAMP", style="dim")
    table.add_column("FILES", justify="right")
    table.add_column("SIZE", justify="right")
    table.add_column("LATEST", justify="center")

    for run in runs:
        ts = datetime.fromtimestamp(
            run.get("mtime_epoch", 0), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        latest = "[green]✓[/green]" if run.get("is_latest") else ""
        table.add_row(
            str(run.get("epoch", "")),
            ts,
            str(run.get("file_count", 0)),
            _human_size(int(run.get("total_size_bytes", 0))),
            latest,
        )

    kube_console.console.print(table)
