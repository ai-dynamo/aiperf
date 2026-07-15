# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet release/version helpers.

Utilities for resolving the JobSet CRD manifest URL, querying the latest
release from GitHub, and formatting install hints for CLI output.
"""

# Known-good fallback version for JobSet CRD installation
JOBSET_FALLBACK_VERSION = "v0.5.2"
JOBSET_GITHUB_REPO = "kubernetes-sigs/jobset"


def get_jobset_manifest_url(version: str | None = None) -> str:
    """Build the JobSet manifest URL for a given version.

    Args:
        version: JobSet release tag (e.g. "v0.5.2"). If None, uses the fallback.

    Returns:
        URL to the JobSet manifests.yaml for kubectl apply.
    """
    v = version or JOBSET_FALLBACK_VERSION
    return (
        f"https://github.com/{JOBSET_GITHUB_REPO}/releases/download/{v}/manifests.yaml"
    )


async def get_latest_jobset_version() -> str | None:
    """Query GitHub API for the latest JobSet release tag.

    Returns:
        Latest release tag (e.g. "v0.7.1"), or None if the lookup fails.
    """
    import aiohttp
    import orjson

    from aiperf.transports.aiohttp_client import create_tcp_connector

    url = f"https://api.github.com/repos/{JOBSET_GITHUB_REPO}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    try:
        connector = create_tcp_connector()
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5), connector=connector
            ) as session,
            session.get(url, headers=headers) as resp,
        ):
            data = orjson.loads(await resp.read())
            tag = data.get("tag_name")
            return tag if isinstance(tag, str) else None
    except (aiohttp.ClientError, orjson.JSONDecodeError, TimeoutError):
        return None


def get_jobset_install_hint(version: str | None = None) -> str:
    """Get a user-facing hint for installing JobSet CRD.

    Args:
        version: Specific version tag, or None for fallback.

    Returns:
        Formatted install command string.
    """
    url = get_jobset_manifest_url(version)
    return f"Install JobSet: kubectl apply --server-side -f {url}"
