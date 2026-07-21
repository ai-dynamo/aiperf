# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aiperf.auth.base_signer import SignedRequest
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.optional_dependencies import aws_dependency_message

if TYPE_CHECKING:
    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo


class SigV4RequestSigner(AIPerfLifecycleMixin):
    """AWS SigV4 request signer using botocore.

    Signs HTTP requests with AWS Signature Version 4 for authenticating
    against SageMaker, Bedrock, API Gateway, and other SigV4-protected endpoints.
    Uses botocore's credential chain for automatic credential discovery and refresh.

    Credential handling has two layers, addressing different failure modes:

    1. Per-``sign()`` expiry-based refresh: ``get_frozen_credentials()`` is
       called on every request and triggers botocore's built-in
       refresh-if-needed check on the currently-resolved credential object
       (e.g. renewing an STS/SSO/IRSA token before it expires). This alone
       protects a long-running benchmark from *expiry*.
    2. Periodic full re-resolution (``AIPERF_AWS_CREDENTIAL_RERESOLVE_INTERVAL``,
       default 15 minutes, 0 disables it): a background task re-runs
       botocore's entire credential provider chain from scratch, independent
       of request traffic, rather than refreshing the already-resolved
       object. Expiry-based refresh alone never notices if the *source*
       changed - e.g. a static key rotated in ``~/.aws/credentials``, or a
       ``credential_process`` re-issuing a new credential - because it keeps
       reusing the same provider instance it resolved at init time. Without
       periodic re-resolution, a multi-hour benchmark started right before a
       credential rotation would keep signing with the old (possibly
       revoked/leaked) credential for its entire duration. Running this as a
       background task (rather than a check inline in ``sign()``) means it
       fires on a fixed cadence even during a quiet/bursty traffic lull, and
       - since it runs via ``asyncio.to_thread`` - a slow credential source
       (SSO, ``credential_process``) never blocks the event loop or an
       in-flight ``sign()`` call.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.region: str | None = model_endpoint.endpoint.aws_region
        self.service: str | None = model_endpoint.endpoint.aws_service
        self.profile: str | None = model_endpoint.endpoint.aws_profile
        self._credentials = None

    @on_init
    async def _init_credentials(self) -> None:
        """Initialize botocore session and resolve credentials."""
        try:
            import botocore.session  # noqa: F401
        except ImportError as e:
            raise ImportError(
                aws_dependency_message("SigV4 request signing is enabled")
            ) from e

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        self._SigV4Auth = SigV4Auth
        self._AWSRequest = AWSRequest
        self._Credentials = Credentials

        self._reresolve_credentials()

    def _reresolve_credentials(self) -> None:
        """Run botocore's credential provider chain from scratch.

        Called synchronously at init, and periodically (off the event loop,
        via asyncio.to_thread, see _periodic_reresolve_credentials below) - see
        the class docstring for why this is distinct from the per-call
        get_frozen_credentials() refresh.
        """
        import botocore.session

        session = botocore.session.Session()
        if self.profile:
            session.set_config_variable("profile", self.profile)

        # get_credentials() returns botocore's own credential object (a
        # RefreshableCredentials instance for STS/SSO/IRSA-sourced
        # credentials). We keep that object - not a frozen snapshot - so
        # that sign() can call get_frozen_credentials() on every request,
        # which triggers botocore's built-in expiry-based refresh check.
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError(
                "No AWS credentials found. Configure via environment variables, "
                "~/.aws/credentials, or IAM role."
            )
        # A single reference assignment is atomic under the GIL, so sign()
        # (running on the event loop) can safely read self._credentials
        # while this runs concurrently in a background thread.
        self._credentials = credentials

    @background_task(
        interval=lambda self: Environment.AWS.CREDENTIAL_RERESOLVE_INTERVAL or math.inf,
        immediate=False,
    )
    def _periodic_reresolve_credentials(self) -> None:
        """Background task: re-run the credential provider chain on a fixed cadence.

        A 0 (disabled) interval becomes math.inf: asyncio.sleep(math.inf)
        simply waits forever rather than firing, so "disabled" needs no
        separate code path - the same lambda that reports the configured
        interval handles it. Runs via asyncio.to_thread (this is a plain,
        non-async method - see the class docstring) and any exception here
        is caught, logged, and treated as non-fatal by the shared
        @background_task loop, so a failed re-resolution just leaves
        self._credentials untouched and the next attempt tries again.
        """
        self._reresolve_credentials()

    async def sign(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> SignedRequest:
        """Sign an HTTP request with AWS SigV4.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full request URL
            headers: Current request headers (will be included in signature)
            body: Request body bytes (hashed for signature)

        Returns:
            SignedRequest with original + auth headers merged
        """
        # get_frozen_credentials() re-resolves (and refreshes if expired)
        # on every call - see the class docstring for the two-layer
        # refresh/re-resolve strategy.
        frozen = self._credentials.get_frozen_credentials()
        credentials = self._Credentials(
            frozen.access_key, frozen.secret_key, frozen.token
        )

        request = self._AWSRequest(method=method, url=url, data=body, headers=headers)
        self._SigV4Auth(credentials, self.service, self.region).add_auth(request)
        return SignedRequest(headers=dict(request.headers))
