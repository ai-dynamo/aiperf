# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Region to DNS-suffix mapping for deriving AWS service hostnames.

The suffix comes from botocore's partition table when botocore is installed,
which it is wherever a SageMaker run can actually sign requests. This runs
during ``EndpointConfig`` validation, which also has to work without the
optional ``aiperf[aws]`` extra, so the import is optional and a commercial/China
fallback remains. Getting the suffix wrong only affects the *derived* base URL
-- an explicit ``--url`` always wins, which is what VPC/PrivateLink and
custom-domain deployments use.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any

_CHINA_SUFFIX = "amazonaws.com.cn"
_COMMERCIAL_SUFFIX = "amazonaws.com"
# Explicit ranges rather than re.IGNORECASE: under IGNORECASE, [a-z] also
# matches non-ASCII letters that case-fold into it, such as the Kelvin sign.
_REGION_ID = re.compile(r"[A-Za-z]+(-[A-Za-z]+)+-[0-9]+")


@cache
def _partitions() -> tuple[dict[str, Any], ...]:
    """botocore's partition table, or empty when botocore is not installed."""
    try:
        from botocore.loaders import create_loader

        return tuple(create_loader().load_data("endpoints")["partitions"])
    except Exception:  # noqa: BLE001 - missing extra or unreadable data: use the fallback
        return ()


def dns_suffix(region: str) -> str:
    """Return the DNS suffix for ``region``'s AWS partition.

    Matched the way botocore resolves endpoints: a partition's listed regions
    first, then its region pattern, so a region AWS adds to a known partition
    resolves before botocore lists it. The EU sovereign cloud (``eusc-*``,
    ``amazonaws.eu``) and the ISO partitions do not use ``amazonaws.com``.

    Without botocore, or for a region no partition claims, only China gets a
    different suffix; everything else falls back to the commercial one rather
    than raising, since a wrong guess is recoverable (pass ``--url``) and shows
    up immediately as a DNS failure.

    Args:
        region: AWS region id, e.g. ``us-west-2``. Case-insensitive.

    Returns:
        The DNS suffix, e.g. ``amazonaws.com`` or ``amazonaws.com.cn``.
    """
    normalized = region.lower()
    for partition in _partitions():
        if normalized in partition.get("regions", {}) or re.match(
            partition["regionRegex"], normalized
        ):
            return partition["dnsSuffix"]
    return _CHINA_SUFFIX if normalized.startswith("cn-") else _COMMERCIAL_SUFFIX


def is_region_id(region: str) -> bool:
    """Whether ``region`` has the shape of an AWS region id, e.g. ``us-west-2``.

    Checked before a region is spliced into a hostname: it lands between two DNS
    labels, so a ``.``, ``/``, ``#``, ``@`` or ``:`` could move a SigV4-signed
    request to a host outside AWS. Only the shape is checked, not membership in a
    list, for the same reason ``dns_suffix`` does not raise on unknown regions.

    Args:
        region: Candidate region id. Case-insensitive.

    Returns:
        True if ``region`` is letters-and-hyphens ending in ``-<number>``.
    """
    return _REGION_ID.fullmatch(region) is not None
