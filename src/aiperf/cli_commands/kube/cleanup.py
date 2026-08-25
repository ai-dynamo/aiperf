# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube cleanup command: bulk-remove finished benchmarks from a namespace."""

from __future__ import annotations

from typing import Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="cleanup")

_TERMINAL_PHASES = frozenset({"Completed", "Failed", "Cancelled"})
_SWEEP_TERMINAL_PHASES = frozenset(
    {"Succeeded", "Failed", "Cancelled", "PartiallyFailed"}
)


def _phase(cr: dict[str, Any]) -> str:
    return (cr.get("status") or {}).get("phase") or "Pending"


def _name(cr: dict[str, Any]) -> str:
    return (cr.get("metadata") or {}).get("name") or "<unnamed>"


def _is_terminal(plural: str, cr: dict[str, Any]) -> bool:
    """Use the workload kind's terminal phase vocabulary."""
    from aiperf.kubernetes.cr_refs import AIPERF_SWEEP_PLURAL

    phases = (
        _SWEEP_TERMINAL_PHASES if plural == AIPERF_SWEEP_PLURAL else _TERMINAL_PHASES
    )
    return _phase(cr) in phases


@app.default
async def cleanup(
    *,
    manage_options: KubeManageOptions | None = None,
    all_benchmarks: Annotated[
        bool,
        Parameter(
            name=["-a", "--all"],
            help="Also remove benchmarks that are still running (they are cancelled first).",
        ),
    ] = False,
    force: Annotated[
        bool,
        Parameter(name=["-f", "--force"], help="Skip the confirmation prompt."),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="List what would be removed and exit without deleting anything.",
        ),
    ] = False,
) -> None:
    """Remove finished AIPerf benchmarks from a namespace.

    By default only terminal benchmarks (Completed, Failed, Cancelled) are
    removed, so a cleanup cannot take out a run in progress. Their JobSets and
    pods are garbage-collected via ownerReferences; results already harvested
    onto the operator's PVC are untouched.

    Examples:
        # See what would go
        aiperf kube cleanup --dry-run

        # Remove finished benchmarks in the default namespace
        aiperf kube cleanup

        # Remove everything, cancelling anything still running
        aiperf kube cleanup --all --force
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Cleaning Up Benchmarks"):
        from kubernetes_asyncio import client as k8s_client_mod

        from aiperf.cli_commands.kube._kube_delete import (
            confirm_action,
            list_aiperf_crs,
        )
        from aiperf.kubernetes import console as kube_console
        from aiperf.kubernetes.client import k8s_client
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE
        from aiperf.kubernetes.cr_refs import AIPERF_JOB_GROUP, AIPERF_JOB_VERSION

        namespace = manage_options.namespace or DEFAULT_BENCHMARK_NAMESPACE

        async with k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            custom = k8s_client_mod.CustomObjectsApi(api)
            everything = await list_aiperf_crs(custom, namespace=namespace)
            if not everything:
                kube_console.print_info(
                    f"No AIPerf benchmarks found in namespace {namespace}"
                )
                return

            targets = [
                (plural, cr)
                for plural, cr in everything
                if all_benchmarks or _is_terminal(plural, cr)
            ]
            skipped = len(everything) - len(targets)
            if not targets:
                kube_console.print_info(
                    f"Nothing to clean up in {namespace}: all {skipped} benchmark(s) "
                    f"are still running (use --all to include them)."
                )
                return

            for plural, cr in targets:
                kube_console.print_info(f"  {plural[:-1]} {_name(cr)} ({_phase(cr)})")
            if dry_run:
                kube_console.print_info(
                    f"--dry-run: {len(targets)} benchmark(s) would be removed."
                )
                return
            if not force and not confirm_action(
                f"Remove {len(targets)} benchmark(s) from namespace {namespace}?"
            ):
                kube_console.print_info("Aborted.")
                return

            removed = 0
            for plural, cr in targets:
                name = _name(cr)
                if all_benchmarks and not _is_terminal(plural, cr):
                    # Ask the operator to wind the run down first so it stamps
                    # a Cancelled phase and tears the JobSet down in order,
                    # rather than yanking the CR out from under it.
                    await custom.patch_namespaced_custom_object(
                        group=AIPERF_JOB_GROUP,
                        version=AIPERF_JOB_VERSION,
                        plural=plural,
                        namespace=namespace,
                        name=name,
                        body={"spec": {"cancel": True}},
                    )
                await custom.delete_namespaced_custom_object(
                    group=AIPERF_JOB_GROUP,
                    version=AIPERF_JOB_VERSION,
                    plural=plural,
                    namespace=namespace,
                    name=name,
                )
                kube_console.clear_last_benchmark_if_matches(name, namespace)
                removed += 1

            kube_console.print_success(
                f"Removed {removed} benchmark(s) from namespace {namespace}"
            )
            if skipped:
                kube_console.print_info(
                    f"Left {skipped} running benchmark(s) alone (use --all to include them)."
                )
