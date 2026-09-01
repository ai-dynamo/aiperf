# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes enumerations."""

from __future__ import annotations

from aiperf.common.enums import CaseInsensitiveStrEnum


class RestartPolicy(CaseInsensitiveStrEnum):
    """Kubernetes pod restart policies.

    See https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy
    """

    ALWAYS = "Always"
    """The kubelet always restarts the container after it exits."""

    ON_FAILURE = "OnFailure"
    """The kubelet restarts the container only when it exits with a non-zero status."""

    NEVER = "Never"
    """The kubelet does not restart the container."""


class ImagePullPolicy(CaseInsensitiveStrEnum):
    """Kubernetes image pull policies.

    See https://kubernetes.io/docs/concepts/containers/images/#image-pull-policy
    """

    ALWAYS = "Always"
    """Every time the kubelet launches a container, it queries the registry to resolve the name to a digest.
    Uses cached image if digest matches, otherwise pulls the image."""

    NEVER = "Never"
    """The kubelet does not try fetching the image. Startup fails if the image is not already present locally."""

    IF_NOT_PRESENT = "IfNotPresent"
    """The image is pulled only if it is not already present locally."""


class JobSetStatus(CaseInsensitiveStrEnum):
    """JobSet status from status.conditions (Completed/Failed) or derived (Running/Unknown).

    See https://jobset.sigs.k8s.io/docs/reference/jobset.v1alpha2
    """

    RUNNING = "Running"
    """JobSet is in progress; no Completed or Failed condition True."""

    COMPLETED = "Completed"
    """JobSet condition Completed is True."""

    FAILED = "Failed"
    """JobSet condition Failed is True."""

    UNKNOWN = "Unknown"
    """Status could not be determined."""

    def to_pod_phase(self) -> PodPhase:
        """Equivalent PodPhase for display/styling."""
        return _JOBSET_TO_POD_PHASE[self]

    @classmethod
    def from_str(cls, s: str) -> JobSetStatus | None:
        """Parse a status string, returning None for unrecognized values."""
        try:
            return cls(s)
        except ValueError:
            return None


class PodPhase(CaseInsensitiveStrEnum):
    """Kubernetes pod phases.

    See https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase
    """

    PENDING = "Pending"
    """Pod accepted by cluster but containers not yet ready. Includes scheduling and image pull time."""

    RUNNING = "Running"
    """Pod bound to a node with all containers created. At least one is running or starting."""

    SUCCEEDED = "Succeeded"
    """All containers terminated successfully and will not be restarted."""

    FAILED = "Failed"
    """All containers terminated, at least one with non-zero exit or killed by the system."""

    UNKNOWN = "Unknown"
    """Pod state could not be obtained, typically due to node communication error."""

    @property
    def is_retrievable(self) -> bool:
        """Whether files can be retrieved from a pod in this phase."""
        return self in (PodPhase.RUNNING, PodPhase.SUCCEEDED)


# Mapping from JobSetStatus to PodPhase for display/styling
_JOBSET_TO_POD_PHASE: dict[JobSetStatus, PodPhase] = {
    JobSetStatus.RUNNING: PodPhase.RUNNING,
    JobSetStatus.COMPLETED: PodPhase.SUCCEEDED,
    JobSetStatus.FAILED: PodPhase.FAILED,
    JobSetStatus.UNKNOWN: PodPhase.UNKNOWN,
}
