# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote dataset references in ``aiperf kube`` submissions.

The CLI host pins each remote reference before it renders the workload CR, so
every pod replays the same version. A host that cannot read an ``hf://`` or
``s3://`` source submits the reference unpinned, and the dataset-manager pod
pins it with the pod's own credentials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig

HOST_UNREADABLE_KINDS = frozenset(
    {RemoteErrorKind.AUTH_MISSING, RemoteErrorKind.AUTH_DENIED, RemoteErrorKind.GATED}
)
"""Access-stage failures that mean this machine lacks credentials the pod may have."""

_POD_CREDENTIAL_SCHEMES = frozenset({"hf", "s3"})
"""Schemes whose pod may hold credentials this machine lacks (an HF token
Secret, IRSA, Pod Identity). An ``http(s)://`` pod fetches the same URL with the
same query string, so it cannot succeed where this machine failed."""


def pin_remote_datasets_for_kubernetes(benchmark: BenchmarkConfig) -> None:
    """Run the remote access stage on the CLI host before the CR is rendered.

    Mutates ``benchmark`` in place: remote references gain ``resolved_version``
    and manifest defaults, and deferred fixed-schedule promotion is settled. On
    an auth-class failure for an ``hf://`` or ``s3://`` reference, the reference
    stays unpinned and a warning is printed. Calls ``asyncio.run``, so async
    callers run it in a worker thread.

    Raises:
        RemoteSourceError: The source is unusable for a reason other than this
            machine's credentials, or an ``http(s)://`` source refused this
            machine.
        ConfigurationError: This machine cannot read the source, and the CLI
            deferred fixed-schedule promotion until the first record is known.
    """
    from aiperf.dataset.remote.preflight import preflight_remote_access

    try:
        preflight_remote_access([benchmark])
    except RemoteSourceError as e:
        if not _pod_may_read(benchmark, e):
            raise
        _require_settled_fixed_schedule(benchmark, e)
        _warn_unpinned(e)


def _pod_may_read(benchmark: BenchmarkConfig, error: RemoteSourceError) -> bool:
    """Whether the dataset-manager pod's credentials could succeed where this machine failed."""
    if error.kind not in HOST_UNREADABLE_KINDS:
        return False
    # preflight_remote_access probes only the default dataset.
    return benchmark.get_default_dataset().path.scheme in _POD_CREDENTIAL_SCHEMES


def _require_settled_fixed_schedule(
    benchmark: BenchmarkConfig, error: RemoteSourceError
) -> None:
    """Refuse a run whose phase type still depends on the unread first record.

    The dataset-manager pod could settle it, but only in its own copy of the
    run; every other pod would keep the phase the CLI built optimistically.
    """
    from aiperf.config.dataset import FileDataset

    dataset = benchmark.get_default_dataset()
    if (
        not isinstance(dataset, FileDataset)
        or dataset.remote_fixed_schedule_check is None
    ):
        return
    raise ConfigurationError(
        f"This machine cannot read {error.ref} ({error.kind}), so AIPerf cannot tell "
        "whether the trace has timestamps and should replay on its recorded "
        "schedule. Pass --fixed-schedule to replay on its timestamps or "
        "--no-fixed-schedule to keep the selected timing mode, or run from a "
        "machine that can read the source."
    )


def _warn_unpinned(error: RemoteSourceError) -> None:
    from rich.markup import escape

    from aiperf.kubernetes import console as kube_console

    kube_console.print_warning(
        escape(
            f"This machine cannot read {error.ref} ({error.kind}), so it is "
            "submitted unpinned; the dataset-manager pod will pin it with the "
            "pod's own credentials. Pin it explicitly (@<commit> or ?versionId=) "
            "to fix the version before the pod starts."
        )
    )
