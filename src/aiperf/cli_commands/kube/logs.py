# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube logs command: retrieve logs from benchmark pods."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="logs")


def _collect_log_targets(
    pods: list[Any], container: str | None
) -> list[tuple[Any, str]]:
    """Build the list of (pod, container) pairs to fetch logs for."""
    targets: list[tuple[Any, str]] = []
    for pod in pods:
        containers = (pod.spec.containers if pod.spec else []) or []
        container_names = [c.name for c in containers]
        target_containers = [container] if container else container_names
        for cont in target_containers:
            if cont in container_names:
                targets.append((pod, cont))
    return targets


async def _stream_pod_log(
    core: Any,
    *,
    pod_name: str,
    namespace: str,
    container: str,
    tail: int | None,
) -> None:
    """Follow a single pod/container's log to stdout until the stream ends."""
    from aiperf.kubernetes import console as kube_console

    tail_kwargs = {"tail_lines": tail} if tail is not None else {}
    raw = await core.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        container=container,
        follow=True,
        _preload_content=False,
        **tail_kwargs,
    )
    try:
        async for line in raw.content:
            kube_console.console.print(
                line.decode("utf-8", errors="replace").rstrip("\n"),
                highlight=False,
                markup=False,
                soft_wrap=True,
            )
    finally:
        await raw.release()


async def _print_pod_log(
    core: Any,
    *,
    pod_name: str,
    namespace: str,
    container: str,
    tail: int | None,
) -> None:
    """Print one pod/container's buffered logs to stdout."""
    from aiperf.kubernetes import console as kube_console

    log_kwargs: dict[str, Any] = {}
    if tail is not None:
        log_kwargs["tail_lines"] = tail
    log_text = await core.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        container=container,
        **log_kwargs,
    )
    kube_console.console.print(
        log_text.rstrip("\n") if log_text else "",
        highlight=False,
        markup=False,
        soft_wrap=True,
    )


async def _report_missing_pods(api: Any, job_id: str, namespace: str) -> bool:
    """Explain an empty pod list; return whether the benchmark itself exists.

    Pods are garbage-collected once a run finishes and its CR is deleted, so an
    empty list is routine for a benchmark that is still addressable and must not
    be reported as a missing target.
    """
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import console as kube_console

    if await cli_helpers.target_exists(api, job_id, namespace):
        kube_console.print_warning(
            f"No pods found for {job_id} in namespace {namespace}. "
            "Its pods may already have been garbage-collected."
        )
        return True
    kube_console.print_error(f"No AIPerf job found with ID: {job_id}")
    kube_console.print_info(f"Searched namespace: {namespace}")
    kube_console.print_action("Run 'aiperf kube list' to see available jobs")
    return False


def _report_saved_logs(saved: Any, output: Path) -> None:
    """Print an outcome line that matches what actually reached disk.

    An unconditional success line hides both a wholly empty dump and a partial
    one, and ``kubectl logs`` failures are otherwise invisible because their
    stderr never surfaces anywhere else.
    """
    from aiperf.kubernetes import console as kube_console

    for failure in saved.failures:
        kube_console.print_warning(f"Could not save logs -- {failure}")
    if saved.wrote_anything:
        kube_console.print_success(
            f"Saved logs for {len(saved.files_written)} of {saved.pods_matched} "
            f"pod(s) to {output}/logs/"
        )
    else:
        kube_console.print_warning(
            f"No logs written to {output}/logs/: none of the "
            f"{saved.pods_matched} matching pod(s) returned any output"
        )


async def _save_logs_to_directory(
    job_id: str,
    namespace: str,
    output: Path,
    manage_options: KubeManageOptions,
) -> bool:
    """Save all pod logs for a job to an output directory.

    Returns whether the benchmark itself could be addressed, so the caller can
    distinguish "nothing left to collect" from "no such benchmark".
    """
    from aiperf.kubernetes import client
    from aiperf.kubernetes import logs as kube_logs

    async with client.k8s_client(
        kubeconfig=manage_options.kubeconfig,
        context=manage_options.kube_context,
    ) as api:
        saved = await kube_logs.save_pod_logs(
            job_id,
            namespace,
            output,
            api,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
        if not saved.pods_matched:
            return await _report_missing_pods(api, job_id, namespace)
    _report_saved_logs(saved, output)
    return True


async def _emit_target_log(
    core: Any,
    pod: Any,
    cont: str,
    *,
    namespace: str,
    follow: bool,
    tail: int | None,
) -> None:
    """Print header then stream or dump a single target's log; errors logged."""
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console

    pod_name = pod.metadata.name
    kube_console.print_header(f"{pod_name}/{cont}")
    try:
        if follow:
            await _stream_pod_log(
                core,
                pod_name=pod_name,
                namespace=namespace,
                container=cont,
                tail=tail,
            )
        else:
            await _print_pod_log(
                core,
                pod_name=pod_name,
                namespace=namespace,
                container=cont,
                tail=tail,
            )
    except ApiException as e:
        kube_console.print_error(f"Error getting logs: {e}")


async def _print_pod_logs(
    job_id: str,
    namespace: str,
    *,
    container: str | None,
    follow: bool,
    tail: int | None,
    manage_options: KubeManageOptions,
) -> bool:
    """Fetch pods for the job and print (or follow) logs to stdout.

    Returns whether the benchmark itself could be addressed, so the caller can
    distinguish "nothing left to collect" from "no such benchmark".
    """
    from kubernetes_asyncio import client as k8s_client_mod

    from aiperf.kubernetes import client
    from aiperf.kubernetes import console as kube_console

    async with client.k8s_client(
        kubeconfig=manage_options.kubeconfig,
        context=manage_options.kube_context,
    ) as api:
        core = k8s_client_mod.CoreV1Api(api)
        pods = await client.get_pods(api, namespace, client.job_selector(job_id))

        if not pods:
            return await _report_missing_pods(api, job_id, namespace)

        targets = _collect_log_targets(pods, container)
        if not targets:
            kube_console.print_warning("No matching containers found")
            return True

        if follow and len(targets) > 1:
            kube_console.print_warning(
                f"Follow mode streams one container at a time. "
                f"Showing {targets[0][0].metadata.name}/{targets[0][1]} "
                f"({len(targets)} targets total). "
                f"Use --container to select a specific container."
            )

        for pod, cont in targets:
            await _emit_target_log(
                core,
                pod,
                cont,
                namespace=namespace,
                follow=follow,
                tail=tail,
            )
            if follow:
                break  # Only follow one target
    return True


@app.default
async def logs(
    job_id: Annotated[
        str | None,
        Parameter(
            help="The AIPerf job ID or AIPerfSweep name to get logs from (default: last deployed job)."
        ),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    container: Annotated[
        str | None, Parameter(help="Specific container name to get logs from.")
    ] = None,
    follow: Annotated[
        bool, Parameter(name=["-f", "--follow"], help="Follow log output in real-time.")
    ] = False,
    tail: Annotated[
        int | None, Parameter(help="Number of lines from the end to show.")
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            name=["-o", "--output"],
            help="Directory to save log files (one per pod). Prints to stdout if not set.",
        ),
    ] = None,
    variation: Annotated[
        int | None,
        Parameter(
            name=["-v", "--variation"],
            help="When job_id is an AIPerfSweep name, target child variation index (0..199). Resolves to <sweep>-v<idx:02d>[-t<trial>].",
        ),
    ] = None,
    trial: Annotated[
        int | None,
        Parameter(
            name=["-t", "--trial"],
            help="Trial index (0..9) within a sweep variation. Requires -v.",
        ),
    ] = None,
    ignore_not_found: Annotated[
        bool,
        Parameter(
            name="--ignore-not-found",
            help="Exit 0 instead of 1 when the benchmark does not exist (mirrors kubectl).",
        ),
    ] = False,
) -> None:
    """Get logs from AIPerf benchmark pods.

    Shows logs from all pods and containers associated with the job.
    If no job_id is specified, uses the last deployed benchmark.

    Use --output to save logs to a directory instead of printing to stdout.
    Each pod's logs are saved as <output>/logs/{pod-name}.log.

    Exits 1 when the target cannot be addressed at all (no such AIPerfJob,
    AIPerfSweep or JobSet, or no job_id and no last-benchmark record), so the
    command works as a CI existence check. A benchmark that exists but whose
    pods are gone still exits 0. Pass ``--ignore-not-found`` to keep exit 0 for
    a missing benchmark too.

    Examples:
        # Get logs from last deployed job
        aiperf kube logs

        # Get logs from a specific job
        aiperf kube logs abc123

        # Get logs from a specific container
        aiperf kube logs --container control-plane

        # Follow logs in real-time
        aiperf kube logs -f

        # Get last 100 lines
        aiperf kube logs --tail 100

        # Save logs to a directory
        aiperf kube logs --output ./my-logs

        # Target a specific sweep variation
        aiperf kube logs my-sweep -v 7
        aiperf kube logs my-sweep -v 5 -t 0

        # Tolerate an already-deleted benchmark in a cleanup script
        aiperf kube logs abc123 --ignore-not-found
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_name

    manage_options = manage_options or KubeManageOptions()
    if job_id is not None:
        child = resolve_child_name(job_id, variation=variation, trial=trial)
        if child is not None:
            job_id = child

    with cli_utils.exit_on_error(title="Error Getting Logs"):
        from aiperf.kubernetes import cli_helpers

        resolved = cli_helpers.resolve_job_id_and_namespace(
            job_id, manage_options.namespace
        )
        if not resolved:
            cli_helpers.exit_target_not_found(ignore_not_found=ignore_not_found)
            return
        job_id, namespace = resolved

        if output:
            found = await _save_logs_to_directory(
                job_id, namespace, output, manage_options
            )
        else:
            found = await _print_pod_logs(
                job_id,
                namespace,
                container=container,
                follow=follow,
                tail=tail,
                manage_options=manage_options,
            )
        if not found:
            cli_helpers.exit_target_not_found(ignore_not_found=ignore_not_found)
