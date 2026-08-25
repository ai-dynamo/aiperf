# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure label selector builders for AIPerf-owned Kubernetes resources."""

from __future__ import annotations

from aiperf.kubernetes.constants import AIPerfLabels, JobSetLabels


def job_selector(job_id: str) -> str:
    """Build the label selector for all AIPerf resources belonging to a job.

    Combines the repo-wide ``AIPerfLabels.SELECTOR`` (``app.kubernetes.io/part-of=aiperf``)
    with the per-job ``AIPerfLabels.JOB_ID`` into a single comma-separated selector
    string consumable by any ``list_*`` / ``delete_*`` k8s API.

    Args:
        job_id: AIPerf job ID (the value stored on ``metadata.labels[aiperf.nvidia.com/job-id]``).

    Returns:
        A selector string like ``"app.kubernetes.io/part-of=aiperf,aiperf.nvidia.com/job-id=<job_id>"``.

    Raises:
        Never raises — pure string construction.
    """
    return f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID}={job_id}"


def controller_selector(job_id: str) -> str:
    """Label selector for the controller pod of a job."""
    return (
        f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID}={job_id},"
        f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
    )
