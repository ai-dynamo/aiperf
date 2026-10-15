# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset-manager startup on Kubernetes: rehydrate, preflight, report failures.

The dataset-manager container exits before bootstrap when a dataset cannot be
used. Kubernetes keeps a container's termination message after the operator
deletes the JobSet, so one redacted line written there is what the operator
copies into the AIPerfJob status.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from aiperf.common.endpoint_credentials import (
    apply_injected_dataset_urls,
    consume_injected_dataset_urls,
    redact_credential_text,
)
from aiperf.common.redact import is_sensitive_query_parameter_name, redact_text
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.remote.errors import RemoteStage

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

TERMINATION_LOG_PATH = Path("/dev/termination-log")
"""Kubernetes' default ``terminationMessagePath``."""

MAX_TERMINATION_MESSAGE_BYTES = 4096
"""The kubelet truncates a termination message to 4 KiB."""

POD_PREFIX = f"[dataset-manager {RemoteStage.POD}]"
"""Prefix that names where the failure happened in ``status.error``."""


def _dataset_url_secrets(url: str) -> list[str]:
    """The full URL plus each credential inside it.

    A server can echo a signature on its own (S3's ``<SignatureProvided>``), where
    neither the whole-URL match nor the query-string patterns would find it.
    """
    parts = urlsplit(url)
    values = [url]
    if parts.password:
        values.extend((parts.password, unquote(parts.password)))
    for pair in parts.query.split("&"):
        name, _, value = pair.partition("=")
        if value and is_sensitive_query_parameter_name(name):
            values.extend((value, unquote(value)))
    return values


def termination_message(error: BaseException, secrets: Iterable[str]) -> str:
    """One redacted line describing ``error``, prefixed and capped at 4 KiB.

    ``secrets`` are plaintext credentials this process holds (injected URLs,
    endpoint credentials); they are scrubbed even where no pattern would catch
    them. The cut never splits a UTF-8 character.
    """
    text = (
        str(error)
        if isinstance(error, ConfigurationError)
        else f"{type(error).__name__}: {error}"
    )
    # Token-wise URL redaction first: applied to the whole text, the URL redactor
    # swallows every word after a raw signature value.
    scrubbed = redact_credential_text(redact_text(text), secrets)
    line = " ".join(f"{POD_PREFIX} {scrubbed}".split())
    capped = line.encode("utf-8")[:MAX_TERMINATION_MESSAGE_BYTES]
    return capped.decode("utf-8", errors="ignore")


def write_termination_message(
    error: BaseException, secrets: Iterable[str], *, path: Path | None = None
) -> None:
    """Best-effort write of :func:`termination_message` to the termination log.

    ``path`` defaults to :data:`TERMINATION_LOG_PATH`, read at call time so tests
    can redirect it. A missing or read-only log (outside Kubernetes) is ignored:
    the error still reaches stderr and the exit code.
    """
    target = TERMINATION_LOG_PATH if path is None else path
    with contextlib.suppress(OSError):
        target.write_text(termination_message(error, secrets), encoding="utf-8")


def prepare_dataset_manager_run(
    run: BenchmarkRun, *, endpoint_secrets: Sequence[str]
) -> None:
    """Make every dataset in ``run`` ready before the dataset-manager bootstraps.

    Swaps presigned dataset URLs back in from ``AIPERF_INJECTED_DATASET_URLS``,
    then runs :func:`aiperf.cli_runner._preflight.preflight_datasets_for_run`.
    On any failure the redacted reason goes to the termination log and the error
    is re-raised, so ``aiperf service`` exits 1.
    """
    from aiperf.cli_runner._preflight import preflight_datasets_for_run

    secrets = list(endpoint_secrets)
    try:
        urls = consume_injected_dataset_urls()
        for url in urls or []:
            secrets.extend(_dataset_url_secrets(url))
        apply_injected_dataset_urls(run, urls)
        preflight_datasets_for_run(run)
    except Exception as e:
        write_termination_message(e, secrets)
        raise
