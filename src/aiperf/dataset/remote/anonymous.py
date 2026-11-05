# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whether a remote dataset needs credentials, decided without using any.

The operator has none of the pod's credentials, and the CLI host may hold
credentials the pod lacks, so both ask the source anonymously.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiperf.config.dataset.remote_ref import RemoteRef, carries_credential
from aiperf.dataset.remote import service
from aiperf.dataset.remote.manifest import ROOT_MANIFEST_NAME

PROBE_TIMEOUT_S = 10.0
"""Upper bound on one anonymous metadata call; the operator preflight has 30 s for everything."""


async def anonymous_needs_auth(ref: RemoteRef) -> bool | None:
    """Whether reading ``ref`` needs credentials; None when that cannot be told.

    HTTP(S) needs auth exactly when the URL carries a credential. Hugging Face
    needs it for a gated or private repo. S3 needs it when an unsigned
    ``HeadObject`` fails for any reason. Offline mode answers None for both.
    """
    if ref.scheme in ("http", "https"):
        return carries_credential(ref.ref)
    if service.is_remote_offline():
        return None
    probe = _hf_needs_auth if ref.scheme == "hf" else _s3_needs_auth
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_S):
            return await asyncio.to_thread(probe, ref)
    except TimeoutError:
        return None if ref.scheme == "hf" else True


def _status_code(error: BaseException) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


def _hf_needs_auth(ref: RemoteRef) -> bool | None:
    """Read the repo's metadata with no token: gated or private means needs auth."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    parsed = ref.parsed
    try:
        info = HfApi().dataset_info(
            parsed.location,
            revision=ref.resolved_version or parsed.revision,
            timeout=PROBE_TIMEOUT_S,
            token=False,
        )
    except RevisionNotFoundError:
        return None
    except RepositoryNotFoundError:
        # Includes GatedRepoError; HF answers a private repo with this too.
        return True
    except HfHubHTTPError as e:
        return True if _status_code(e) in (401, 403) else None
    except Exception:
        # A network failure says nothing about the repo; the pod finds out for real.
        return None
    return bool(info.gated) or bool(info.private)


def _anonymous_s3_client() -> Any:
    import botocore.session
    from botocore import UNSIGNED
    from botocore.config import Config

    config = Config(
        signature_version=UNSIGNED,
        connect_timeout=PROBE_TIMEOUT_S,
        read_timeout=PROBE_TIMEOUT_S,
        retries={"mode": "standard", "max_attempts": 1},
    )
    return botocore.session.get_session().create_client("s3", config=config)


def _s3_probe_key(ref: RemoteRef) -> str:
    """The object an anonymous HEAD asks about: the key, or a prefix's manifest."""
    parsed = ref.parsed
    key = parsed.path or ""
    return f"{key}{ROOT_MANIFEST_NAME}" if parsed.is_container else key


def _s3_needs_auth(ref: RemoteRef) -> bool:
    """An unsigned HeadObject; any failure means the pod needs credentials."""
    params = {"Bucket": ref.parsed.location, "Key": _s3_probe_key(ref)}
    if ref.parsed.revision:
        params["VersionId"] = ref.parsed.revision
    try:
        _anonymous_s3_client().head_object(**params)
    except Exception:
        # A refusal, a network failure, or no botocore: assume credentials are needed.
        return True
    return False
