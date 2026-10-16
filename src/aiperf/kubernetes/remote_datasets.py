# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote dataset references in ``aiperf kube`` submissions.

The CLI host pins each remote reference before it renders the workload CR, so
every pod replays the same version. A host that lacks credentials for an
``hf://`` or ``s3://`` source, or cannot connect to a source of any scheme,
submits the reference unpinned, and the dataset-manager pod pins it with the
pod's own credentials and network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlsplit

from aiperf.common.endpoint_credentials import (
    AIPERF_INJECTED_DATASET_URLS,
    secret_env_names,
)
from aiperf.common.redact import REDACTED_VALUE
from aiperf.config.dataset.remote_ref import carries_credential, remote_dataset_refs
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.kubernetes.preflight import CheckResult, CheckStatus

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.dataset.remote_ref import RemoteRef
    from aiperf.config.deployment import PodTemplateConfig

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
    an auth-class failure for an ``hf://`` or ``s3://`` reference, or a
    connection failure (DNS, connect, timeout; not an HTTP 429/5xx) for any
    scheme, the reference stays unpinned and a warning is printed. A reference whose credential is
    already ``<redacted>`` (a resource written by ``aiperf kube generate``) is
    not read here at all: only the pod holds its full URL. Calls
    ``asyncio.run``, so async callers run it in a worker thread.

    Raises:
        RemoteSourceError: The source is unusable for a reason other than this
            machine's credentials, or an ``http(s)://`` source refused this
            machine.
        ConfigurationError: This machine cannot read the source, and the CLI
            deferred fixed-schedule promotion until the first record is known.
    """
    from aiperf.dataset.remote.preflight import preflight_remote_access

    ref = _default_remote_ref(benchmark)
    if ref is None:
        return
    if REDACTED_VALUE in ref.ref:
        _defer_to_pod(
            benchmark,
            f"This machine cannot read {ref.display} (its credential is redacted)",
            warn=ref.resolved_version is None,
        )
        return
    try:
        preflight_remote_access([benchmark])
    except RemoteSourceError as e:
        if not _pod_may_read(ref, e):
            raise
        _defer_to_pod(benchmark, _host_failure_reason(e))


def _default_remote_ref(benchmark: BenchmarkConfig) -> RemoteRef | None:
    """The default dataset's remote reference; the access stage reads only that one."""
    from aiperf.config.dataset.remote_ref import RemoteRef

    path = getattr(benchmark.get_default_dataset(), "path", None)
    return path if isinstance(path, RemoteRef) else None


def _pod_may_read(ref: RemoteRef, error: RemoteSourceError) -> bool:
    """Whether the dataset-manager pod could succeed where this machine failed.

    A source that only the cluster's network reaches (a private endpoint, an
    in-cluster object store) fails here before any HTTP answer, for every scheme.
    """
    if error.connection_failed:
        return True
    if error.kind not in HOST_UNREADABLE_KINDS:
        return False
    return ref.scheme in _POD_CREDENTIAL_SCHEMES


def _host_failure_reason(error: RemoteSourceError) -> str:
    if error.connection_failed:
        return f"This machine could not connect to {error.ref} ({error.kind})"
    return f"This machine cannot read {error.ref} ({error.kind})"


def _defer_to_pod(
    benchmark: BenchmarkConfig, reason: str, *, warn: bool = True
) -> None:
    """Leave the reference for the dataset-manager pod, or refuse when that is unsafe."""
    _require_settled_fixed_schedule(benchmark, reason)
    if warn:
        _warn_unpinned(reason)


def _require_settled_fixed_schedule(benchmark: BenchmarkConfig, reason: str) -> None:
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
        f"{reason}, so AIPerf cannot tell whether the trace has timestamps and "
        "should replay on its recorded schedule. Pass --fixed-schedule to replay "
        "on its timestamps or --no-fixed-schedule to keep the selected timing "
        "mode, or run from a machine that can read the source."
    )


def _warn_unpinned(reason: str) -> None:
    from rich.markup import escape

    from aiperf.kubernetes import console as kube_console

    kube_console.print_warning(
        escape(
            f"{reason}, so it is submitted unpinned; the dataset-manager pod will "
            "pin it with the pod's own credentials. Pin it explicitly (@<commit> "
            "or ?versionId=) to fix the version before the pod starts."
        )
    )


CHECK_NAME = "Remote Datasets"
_EXPIRY_WARNING = timedelta(hours=1)
_SIGNED_DATE_PREFIXES = ("x-amz-", "x-goog-")


def presigned_expiry(url: str) -> datetime | None:
    """When a presigned URL stops working, read from its own query parameters.

    Reads ``X-Amz-Date`` + ``X-Amz-Expires`` (S3 SigV4), ``X-Goog-Date`` +
    ``X-Goog-Expires`` (GCS V4) and the Azure SAS ``se`` field. Returns None
    when the URL states no expiry or states it in a form AIPerf cannot parse.
    """
    params = {
        name.lower(): value
        for name, value in parse_qsl(urlsplit(url).query, keep_blank_values=True)
    }
    for prefix in _SIGNED_DATE_PREFIXES:
        signed, lifetime = params.get(f"{prefix}date"), params.get(f"{prefix}expires")
        if signed and lifetime:
            return _signed_expiry(signed, lifetime)
    if sas_expiry := params.get("se"):
        return _sas_expiry(sas_expiry)
    return None


def _signed_expiry(signed: str, lifetime: str) -> datetime | None:
    try:
        start = datetime.strptime(signed, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return start + timedelta(seconds=int(lifetime))
    except (ValueError, OverflowError):
        return None


def _sas_expiry(value: str) -> datetime | None:
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=UTC)
    return expiry.astimezone(UTC)


def _has_hf_credential(pod_template: PodTemplateConfig) -> bool:
    """A Secret-backed ``HF_TOKEN``, or ``HF_TOKEN_PATH`` naming a mounted token file."""
    if "HF_TOKEN" in secret_env_names(pod_template.env):
        return True
    return any(
        isinstance(item, dict) and item.get("name") == "HF_TOKEN_PATH"
        for item in pod_template.env
    )


def _has_aws_identity(pod_template: PodTemplateConfig) -> bool:
    if pod_template.service_account_name not in (None, "default"):
        return True
    return any(
        isinstance(item, dict) and str(item.get("name", "")).startswith("AWS_")
        for item in pod_template.env
    )


def _result(status: CheckStatus, message: str, *hints: str) -> CheckResult:
    return CheckResult(
        name=CHECK_NAME, status=status, message=message, hints=list(hints)
    )


def _evaluate_hf(
    ref: RemoteRef, needs_auth: bool | None, pod_template: PodTemplateConfig
) -> CheckResult:
    if needs_auth is None:
        return _result(
            CheckStatus.WARN,
            f"Could not tell whether {ref.display} needs a Hugging Face token; "
            "the dataset-manager pod will find out when it starts",
        )
    if needs_auth and not _has_hf_credential(pod_template):
        return _result(
            CheckStatus.FAIL,
            f"{ref.display} is private or gated, and the pods have no "
            "Secret-backed HF_TOKEN or HF_TOKEN_PATH",
            "Add --env-from-secrets HF_TOKEN=<secret>/<key>, or a "
            "spec.podTemplate.env entry HF_TOKEN with valueFrom.secretKeyRef",
        )
    state = "needs a token, which the pods have" if needs_auth else "is public"
    return _result(CheckStatus.PASS, f"{ref.display} {state}")


def _evaluate_s3(
    ref: RemoteRef, needs_auth: bool | None, pod_template: PodTemplateConfig
) -> CheckResult:
    if needs_auth is False:
        return _result(CheckStatus.PASS, f"{ref.display} is publicly readable")
    if _has_aws_identity(pod_template):
        return _result(
            CheckStatus.PASS,
            f"{ref.display} will be read with the pods' AWS env or service account",
        )
    return _result(
        CheckStatus.WARN,
        f"{ref.display} is not publicly readable, and the pod spec shows no AWS "
        "credentials (no AWS_* env and the default service account). IRSA, EKS "
        "Pod Identity and node roles are not visible from the spec, so this is "
        "only a warning",
        "Use --service-account with a ServiceAccount that can read the bucket, or "
        "--env-from-secrets AWS_ACCESS_KEY_ID=<secret>/<key> and "
        "AWS_SECRET_ACCESS_KEY=<secret>/<key>",
    )


def _evaluate_url(ref: RemoteRef, now: datetime) -> CheckResult:
    if not carries_credential(ref.ref):
        return _result(CheckStatus.PASS, f"{ref.display} needs no credentials")
    expiry = presigned_expiry(ref.ref)
    if expiry is None:
        return _result(
            CheckStatus.PASS, f"{ref.display} is presigned and states no expiry"
        )
    stamp = expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
    if expiry <= now:
        return _result(
            CheckStatus.FAIL,
            f"The presigned URL {ref.display} expired at {stamp}",
            "Generate a new presigned URL and update the Secret behind "
            f"{AIPERF_INJECTED_DATASET_URLS}",
        )
    if expiry - now < _EXPIRY_WARNING:
        minutes = int((expiry - now).total_seconds() // 60)
        return _result(
            CheckStatus.WARN,
            f"The presigned URL {ref.display} expires at {stamp}, in {minutes} min; "
            "queueing (for example in Kueue) can delay the download past that",
            "Generate a URL that stays valid for the expected queue time plus the run",
        )
    return _result(
        CheckStatus.PASS, f"The presigned URL {ref.display} is valid until {stamp}"
    )


def evaluate_remote_dataset(
    ref: RemoteRef,
    needs_auth: bool | None,
    pod_template: PodTemplateConfig,
    *,
    now: datetime,
) -> CheckResult:
    """The pod-credential verdict for one remote dataset reference.

    ``needs_auth`` comes from :func:`aiperf.dataset.remote.anonymous.anonymous_needs_auth`;
    None means it could not be told.
    """
    if ref.scheme == "hf":
        return _evaluate_hf(ref, needs_auth, pod_template)
    if ref.scheme == "s3":
        return _evaluate_s3(ref, needs_auth, pod_template)
    return _evaluate_url(ref, now)


def _worst(results: list[CheckResult]) -> CheckStatus:
    for status in (CheckStatus.FAIL, CheckStatus.WARN):
        if any(result.status == status for result in results):
            return status
    return CheckStatus.PASS


async def check_remote_datasets(
    datasets: Iterable[object],
    pod_template: PodTemplateConfig,
    *,
    now: datetime | None = None,
) -> CheckResult:
    """One CheckResult covering every remote dataset reference against the pod spec.

    Shared by ``aiperf kube`` (before submission) and the operator's workload
    preflight (which also covers resources applied with kubectl). Needs-auth
    comes from an anonymous probe, so neither caller needs the pod's credentials.
    """
    refs = remote_dataset_refs(datasets)
    if not refs:
        return _result(CheckStatus.SKIP, "No remote dataset references")
    from aiperf.dataset.remote import anonymous

    at = now or datetime.now(UTC)
    # Concurrent, so several slow sources still fit the operator's preflight budget.
    needs_auth = await asyncio.gather(
        *(anonymous.anonymous_needs_auth(ref) for ref in refs)
    )
    results = [
        evaluate_remote_dataset(ref, needs, pod_template, now=at)
        for ref, needs in zip(refs, needs_auth, strict=True)
    ]
    return CheckResult(
        name=CHECK_NAME,
        status=_worst(results),
        message="; ".join(result.message for result in results),
        hints=[hint for result in results for hint in result.hints],
    )


def report_remote_dataset_checks(
    benchmark: BenchmarkConfig, workload_spec: dict[str, Any]
) -> None:
    """Run :func:`check_remote_datasets` for ``aiperf kube``: fail on FAIL, print WARN.

    ``workload_spec`` is the rendered AIPerfJob or AIPerfSweep spec; its pod
    template is what the pods will run with. Calls ``asyncio.run``, so async
    callers run it in a worker thread.

    Raises:
        ConfigurationError: A check failed.
    """
    if not remote_dataset_refs(benchmark.datasets):
        return
    from aiperf.kubernetes.spec_converter import extract_deployment_config

    pod_template = extract_deployment_config(workload_spec).pod_template
    result = asyncio.run(check_remote_datasets(benchmark.datasets, pod_template))
    if result.status == CheckStatus.FAIL:
        raise ConfigurationError(" ".join([f"{result.message}.", *result.hints]))
    if result.status == CheckStatus.WARN:
        from rich.markup import escape

        from aiperf.kubernetes import console as kube_console

        # One WARNING record: machine-readable commands keep only WARNING and
        # above (on stderr), so hints logged at INFO would be dropped there.
        kube_console.print_warning(
            escape(" ".join([f"{result.message}.", *result.hints]))
        )
