# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SageMaker Runtime routing options.

Its own module, like ``control_hooks.py``, so ``EndpointConfig`` stays a flat
list of endpoint-level settings and this group can grow without pushing that
model back over the field-count guardrail.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig


class SageMakerConfig(BaseConfig):
    """SageMaker Runtime routing options.

    Grouped rather than flattened onto ``EndpointConfig`` because they only
    apply to one transport, and because ``EndpointConfig`` is already at the
    field count where the repo's ergonomics guardrail asks for sub-models.
    Follows the same shape as ``reset_kv_cache`` / ``server_profiler``.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint_name: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of the SageMaker endpoint to invoke. Setting this is "
            "normally all that is needed: it selects the SageMaker transport, enables "
            "SigV4 signing, and derives the runtime base URL from aws_region.",
        ),
    ]

    target_model: Annotated[
        str | None,
        Field(
            default=None,
            description="SageMaker TargetModel value, for multi-model endpoints. Not "
            "sent on streaming requests, which the API does not accept it on. Defaults "
            "to the request's model name when not set explicitly.",
        ),
    ]

    inference_component_name: Annotated[
        str | None,
        Field(
            default=None,
            description="SageMaker InferenceComponentName to target, for endpoints "
            "hosting multiple inference components.",
        ),
    ]

    target_variant: Annotated[
        str | None,
        Field(
            default=None,
            description="SageMaker production variant to pin every request to, "
            "bypassing the endpoint's traffic split. Useful for benchmarking one "
            "variant of an A/B deployment in isolation.",
        ),
    ]
