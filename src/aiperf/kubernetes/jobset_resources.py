# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for JobSet container resource/port allocation.

These helpers are factored out of `AIPerfJobSetSpec` so the main spec module
stays under the file-size budget. All functions are pure: no `self`, no
module-level mutable state.
"""

from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib


def split_weighted_total(total: int, weights: list[int]) -> list[int]:
    """Split an integer total across weighted buckets.

    Uses a largest-remainder allocation so the sum is preserved exactly.
    """
    if not weights:
        return []
    if total <= 0:
        return [0] * len(weights)

    total_weight = sum(weights)
    raw_shares = [total * weight / total_weight for weight in weights]
    shares = [int(share) for share in raw_shares]
    remainder = total - sum(shares)

    ranked = sorted(
        range(len(weights)),
        key=lambda idx: raw_shares[idx] - shares[idx],
        reverse=True,
    )
    for idx in ranked[:remainder]:
        shares[idx] += 1

    return shares


def format_mcpu(mcpu: int) -> str:
    """Format millicores as a Kubernetes quantity."""
    if mcpu % 1000 == 0:
        return str(mcpu // 1000)
    return f"{mcpu}m"


def format_mib(mib: int) -> str:
    """Format MiB as a Kubernetes memory quantity."""
    return f"{mib}Mi"


def _compute_cpu_shares(
    total_mcpu: int,
    worker_count: int,
    record_processor_count: int,
    record_processor_cpu_request: str | None,
) -> list[int]:
    """Compute per-container CPU shares for a worker pod.

    When a fixed per-record-processor CPU request is configured, that value is
    pinned and the remaining budget is split across manager + workers.
    """
    cpu_weights = [100] + ([131] * worker_count) + ([389] * record_processor_count)
    if record_processor_cpu_request is None or record_processor_count == 0:
        return split_weighted_total(total_mcpu, cpu_weights)

    record_processor_mcpu = int(round(parse_cpu(record_processor_cpu_request) * 1000))
    fixed_total = record_processor_mcpu * record_processor_count
    remaining_mcpu = max(0, total_mcpu - fixed_total)
    non_record_weights = [100] + ([131] * worker_count)
    return (
        split_weighted_total(remaining_mcpu, non_record_weights)
        + [record_processor_mcpu] * record_processor_count
    )


def split_worker_pod_resources(
    worker_pod_resources: dict[str, dict[str, str]] | None,
    worker_count: int,
    record_processor_count: int,
    record_processor_cpu_request: str | None,
    *,
    burstable: bool,
) -> list[dict[str, dict[str, str]] | None]:
    """Split the worker-pod budget across manager/worker/record-processor containers.

    See :meth:`AIPerfJobSetSpec._split_worker_pod_resources` for the rationale:
    the external API remains pod-oriented (`WORKER_POD` is the total budget)
    and internally we divide across containers with measurement-derived weights.
    """
    total_containers = 1 + worker_count + record_processor_count
    if worker_pod_resources is None:
        return [None] * total_containers

    total_mcpu = int(round(parse_cpu(worker_pod_resources["requests"]["cpu"]) * 1000))
    total_mib = parse_memory_mib(worker_pod_resources["requests"]["memory"])

    # These weights reflect the measured relative cost noted in the K8s
    # environment comments: workers are lighter than record processors,
    # while the worker-pod-manager remains a small but non-zero share.
    memory_weights = [128] + ([80] * worker_count) + ([256] * record_processor_count)

    cpu_shares = _compute_cpu_shares(
        total_mcpu,
        worker_count,
        record_processor_count,
        record_processor_cpu_request,
    )
    memory_shares = split_weighted_total(total_mib, memory_weights)

    resources: list[dict[str, dict[str, str]]] = []
    for mcpu, mib in zip(cpu_shares, memory_shares, strict=True):
        entry: dict[str, dict[str, str]] = {
            "requests": {
                "cpu": format_mcpu(mcpu),
                "memory": format_mib(mib),
            },
        }
        if not burstable:
            entry["limits"] = {
                "cpu": format_mcpu(mcpu),
                "memory": format_mib(mib),
            }
        resources.append(entry)
    return resources


def allocate_worker_health_ports(
    worker_count: int,
    record_processor_count: int,
) -> tuple[int, list[int], list[int]]:
    """Allocate unique health ports for every container in a worker pod.

    Containers in a pod share a network namespace, so each service container
    needs its own port even though probes are scoped per container.
    """
    ports = K8sEnvironment.PORTS
    manager_port = ports.WORKER_HEALTH
    worker_ports = list(range(manager_port + 1, manager_port + 1 + worker_count))
    record_processor_start = max(
        ports.RECORD_PROCESSOR_HEALTH,
        manager_port + 1 + worker_count,
    )
    record_processor_ports = list(
        range(
            record_processor_start,
            record_processor_start + record_processor_count,
        )
    )

    allocated = [manager_port, *worker_ports, *record_processor_ports]
    if allocated and max(allocated) > 65535:
        raise ValueError(
            f"Not enough port space to allocate unique worker-container health ports: "
            f"manager_port={manager_port}, worker_count={len(worker_ports)}, "
            f"record_processor_count={len(record_processor_ports)}, "
            f"max allocated port {max(allocated)} exceeds 65535. "
            f"Reduce --workers or lower base health port."
        )
    return manager_port, worker_ports, record_processor_ports
