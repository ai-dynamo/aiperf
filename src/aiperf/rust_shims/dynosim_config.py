# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed authoring surface for the ``dynosim`` runner transports.

The ``dynosim`` transports run one in-process Dynamo mocker co-simulation of a
trace or synthetic workload with no sockets, HTTP, or frontend.  The clock rides
on the transport ID: ``dynosim_offline`` fast-forwards a deterministic virtual
clock for byte-exact replay, and ``dynosim_online`` drives the same engine under
a wall clock for live-throughput measurement.  Both are selected through
``aiperf profile`` as ``transport.type: dynosim_offline`` / ``dynosim_online``;
there is no separate ``aiperf dynosim`` command and no ``replay_mode`` field.

These models mirror the runner's strict decoder
(``rust/runner/src/offline_execution.rs`` ``DynosimTransportSpec`` and its
``deny_unknown_fields`` sub-specs) field-for-field so the projected
``transport.config`` object round-trips into the exact Rust schema.  The engine and
router objects are deliberately left as free-form JSON: they are ``MockEngineArgs``
/ ``KvRouterConfig`` (``RawValue`` on the wire) owned and validated by Dynamo, and
copying their ~60-field surface into Python would only invite drift.

The trace/dataset and concurrency/rate axes are NOT authored here — they reuse the
shared ``benchmark.dataset`` (trace file + format) and ``benchmark.phases`` surface,
identical to the online path.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig

__all__ = [
    "DynamoBuildFeature",
    "DynamoKvEventVisibility",
    "DynosimAicConfig",
    "DynosimArtifactConfig",
    "DynosimRouterMode",
    "DynosimSlaConfig",
    "DynosimTopology",
    "DynosimTransportConfig",
]


class DynosimTopology(str, Enum):
    """Authored offline deployment topology (mirrors ``DynosimTopologySpec``)."""

    SINGLE = "single"
    AGGREGATED = "aggregated"
    DISAGGREGATED = "disaggregated"


class DynosimRouterMode(str, Enum):
    """Router policy for routed topologies (mirrors ``DynosimRouterSpec``)."""

    ROUND_ROBIN = "round_robin"
    KV = "kv"


class DynamoKvEventVisibility(str, Enum):
    """KV-event publication point (mirrors ``DynamoKvEventVisibilitySpec``)."""

    PASS_START = "pass_start"
    PASS_END = "pass_end"


class DynamoBuildFeature(str, Enum):
    """Optional runner build capability an authored run may require.

    Spellings match the Cargo feature names the runner advertises
    (``DynamoBuildFeature``, kebab-case).
    """

    DYNAMO_ROUTER_RUNTIME = "dynamo-router-runtime"
    DYNAMO_ZMQ_EVENTS = "dynamo-zmq-events"
    DYNAMO_KVBM_OFFLOAD = "dynamo-kvbm-offload"
    DYNAMO_AIC_FORWARD_PASS = "dynamo-aic-forward-pass"
    DYNAMO_PROFILE = "dynamo-profile"
    DYNAMO_FULL = "dynamo-full"
    DYNAMO_PARITY = "dynamo-parity"


class DynosimAicConfig(BaseConfig):
    """Structured AIConfigurator overrides (mirrors ``DynosimAicSpec``).

    All fields are optional; the runner requires ``backend``, ``system``, and
    ``model_path`` together once any AIC field is set.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Annotated[
        str | None, Field(default=None, description="AIC serving backend identity.")
    ]
    system: Annotated[
        str | None, Field(default=None, description="AIC GPU system identity.")
    ]
    backend_version: Annotated[
        str | None,
        Field(default=None, description="Performance-database backend version."),
    ]
    model_path: Annotated[
        str | None, Field(default=None, description="Hugging Face model path for AIC.")
    ]
    tp_size: Annotated[
        int | None, Field(default=None, gt=0, description="Tensor-parallel degree.")
    ]
    moe_tp_size: Annotated[
        int | None, Field(default=None, gt=0, description="MoE tensor-parallel degree.")
    ]
    moe_ep_size: Annotated[
        int | None, Field(default=None, gt=0, description="MoE expert-parallel degree.")
    ]
    attention_dp_size: Annotated[
        int | None,
        Field(default=None, gt=0, description="Attention data-parallel degree."),
    ]
    nextn: Annotated[
        int | None,
        Field(default=None, gt=0, description="Speculative (MTP) draft-token count."),
    ]
    gemm_dtype: Annotated[
        str | None, Field(default=None, description="GEMM quantization override.")
    ]
    moe_dtype: Annotated[
        str | None, Field(default=None, description="MoE quantization override.")
    ]
    fmha_dtype: Annotated[
        str | None, Field(default=None, description="Attention/FMHA quantization override.")
    ]
    kv_cache_dtype: Annotated[
        str | None, Field(default=None, description="KV-cache quantization override.")
    ]
    comm_dtype: Annotated[
        str | None, Field(default=None, description="Collective/comm quantization override.")
    ]
    nextn_accept_rates: Annotated[
        str | None,
        Field(
            default=None,
            description="Comma-separated conditional draft acceptance rates.",
        ),
    ]


class DynosimSlaConfig(BaseConfig):
    """Canonical goodput thresholds owned by Dynamo's collector.

    Mirrors ``DynosimSlaSpec``; each bound is optional and non-negative.
    """

    model_config = ConfigDict(extra="forbid")

    ttft_ms: Annotated[
        float | None,
        Field(default=None, ge=0, description="Maximum time to first token (ms)."),
    ]
    itl_ms: Annotated[
        float | None,
        Field(default=None, ge=0, description="Maximum mean inter-token latency (ms)."),
    ]
    e2e_ms: Annotated[
        float | None,
        Field(default=None, ge=0, description="Maximum end-to-end latency (ms)."),
    ]


class DynosimArtifactConfig(BaseConfig):
    """Backend-owned outputs written after a successful run (mirrors ``DynosimArtifactSpec``).

    Paths are relative to the run artifact target.  ``kv_event_visibility`` requires
    ``worker_artifacts_json`` (enforced by the runner).
    """

    model_config = ConfigDict(extra="forbid")

    report_json: Annotated[
        Path | None,
        Field(default=None, description="Canonical aggregate Dynamo JSON report path."),
    ]
    per_request_jsonl: Annotated[
        Path | None,
        Field(default=None, description="Canonical per-request Dynamo JSONL path."),
    ]
    worker_artifacts_json: Annotated[
        Path | None,
        Field(
            default=None,
            description="Timed worker/request/KV artifact JSON for trace workloads.",
        ),
    ]
    kv_event_visibility: Annotated[
        DynamoKvEventVisibility | None,
        Field(default=None, description="Pass-start/pass-end KV visibility override."),
    ]


class DynosimTransportConfig(BaseConfig):
    """Typed ``transport.config`` for ``transport.type: dynosim_offline`` / ``dynosim_online``.

    Mirrors the runner's ``DynosimTransportSpec`` (``deny_unknown_fields``).  The
    ``engine``/``prefill_engine``/``decode_engine``/``router`` objects are opaque
    ``MockEngineArgs``/``KvRouterConfig`` JSON, preserved verbatim for Dynamo to
    validate.  The clock axis is not a field here — it rides on the transport ID.
    """

    model_config = ConfigDict(extra="forbid")

    engine_profile: Annotated[
        Path | None,
        Field(
            default=None,
            description="JSON engine profile consumed by Dynamo's canonical parser.",
        ),
    ]
    engine: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Inline aggregate/single MockEngineArgs object (passed through verbatim).",
        ),
    ]
    prefill_engine: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Inline disaggregated prefill MockEngineArgs object."),
    ]
    decode_engine: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Inline disaggregated decode MockEngineArgs object."),
    ]
    router: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Inline KvRouterConfig object (passed through verbatim)."),
    ]
    router_policy_config: Annotated[
        Path | None,
        Field(
            default=None,
            description="Startup router policy-family YAML path overriding the inline router field.",
        ),
    ]
    router_model_name: Annotated[
        str | None,
        Field(default=None, description="Model selector for a multi-model router policy document."),
    ]
    aic: Annotated[
        DynosimAicConfig | None,
        Field(default=None, description="Optional structured AIConfigurator overrides."),
    ]
    capture_per_request: Annotated[
        bool,
        Field(
            default=False,
            description="Capture backend per-request records even without a JSONL artifact.",
        ),
    ]
    sla: Annotated[
        DynosimSlaConfig,
        Field(default_factory=DynosimSlaConfig, description="Canonical goodput thresholds."),
    ]
    topology: Annotated[
        DynosimTopology,
        Field(default=DynosimTopology.SINGLE, description="Deployment topology."),
    ]
    workers: Annotated[
        int, Field(default=1, gt=0, description="Aggregate worker count.")
    ]
    prefill_workers: Annotated[
        int, Field(default=1, gt=0, description="Disaggregated prefill worker count.")
    ]
    decode_workers: Annotated[
        int, Field(default=1, gt=0, description="Disaggregated decode worker count.")
    ]
    router_mode: Annotated[
        DynosimRouterMode,
        Field(default=DynosimRouterMode.ROUND_ROBIN, description="Router policy for routed topologies."),
    ]
    required_features: Annotated[
        set[DynamoBuildFeature],
        Field(
            default_factory=set,
            description="Optional build capabilities that must exist in the exact runner image.",
        ),
    ]
    artifacts: Annotated[
        DynosimArtifactConfig,
        Field(default_factory=DynosimArtifactConfig, description="Backend-owned output artifacts."),
    ]
