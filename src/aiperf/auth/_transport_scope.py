# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The transport -> SigV4 credential scope lookup, in one place.

Two callers need this and must never disagree: ``EndpointConfig`` decides at
config time whether ``--aws-service`` is *required*, and ``SigV4RequestSigner``
resolves the actual signing name at runtime. If those answered from separate
copies, a drifted lookup would let validation accept a setup the signer then
resolves to ``None`` -- handing ``service=None`` to ``SigV4Auth`` and turning a
config error into a 403 partway into a run.

Deliberately a leaf module: nothing is imported at module scope, so
``aiperf.config`` can use it without pulling in ``aiperf.auth.sigv4_signer``
(and without an import cycle), and it stays importable when the optional
``aiperf[aws]`` extra is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.plugin.enums import TransportType


def transport_botocore_service_id(transport: TransportType | str | None) -> str | None:
    """Return the botocore service id the given transport speaks, if any.

    Read off the transport class so that adding an AWS transport needs no change
    here. Returns None for transports that are not tied to one AWS API -- the
    built-in HTTP transport included, since it may front any service.
    """
    if transport is None:
        return None

    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    try:
        transport_cls = plugins.get_class(PluginType.TRANSPORT, str(transport))
    except Exception:
        return None
    return getattr(transport_cls, "botocore_service_id", None)
