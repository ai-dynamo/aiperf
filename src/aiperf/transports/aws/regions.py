# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Region to DNS-suffix mapping for deriving AWS service hostnames.

Pure string handling with no botocore import on purpose: this runs during
``EndpointConfig`` validation, which has to work whether or not the optional
``aiperf[aws]`` extra is installed. Getting the suffix wrong only affects the
*derived* base URL -- an explicit ``--url`` always wins, which is what
VPC/PrivateLink and custom-domain deployments use.
"""

from __future__ import annotations

import re

_CHINA_SUFFIX = "amazonaws.com.cn"
_COMMERCIAL_SUFFIX = "amazonaws.com"
# Explicit ranges rather than re.IGNORECASE: under IGNORECASE, [a-z] also
# matches non-ASCII letters that case-fold into it, such as the Kelvin sign.
_REGION_ID = re.compile(r"[A-Za-z]+(-[A-Za-z]+)+-[0-9]+")


def dns_suffix(region: str) -> str:
    """Return the DNS suffix for ``region``'s AWS partition.

    Only the China partition uses a different suffix. GovCloud regions
    (``us-gov-*``) read like a separate partition but resolve under the
    commercial suffix, so they are deliberately not special-cased.

    Unknown or future regions fall back to the commercial suffix rather than
    raising: AWS adds regions regularly, and a wrong guess here is both
    recoverable (pass ``--url``) and immediately visible as a DNS failure.

    Args:
        region: AWS region id, e.g. ``us-west-2``. Case-insensitive.

    Returns:
        The DNS suffix, e.g. ``amazonaws.com`` or ``amazonaws.com.cn``.
    """
    return _CHINA_SUFFIX if region.lower().startswith("cn-") else _COMMERCIAL_SUFFIX


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
