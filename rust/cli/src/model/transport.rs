// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `transport` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_inline_transport`
//! (a discriminated union keyed by `type`, with only explicitly-set fields
//! present — the Pydantic `model_dump(exclude_unset=True, exclude_none=True)`).
//! The default HTTP transport projects to exactly `{"type":"http"}`.
//!
//! The DynoSim transports (`dynosim_offline` / `dynosim_online`) carry the full
//! typed [`DynosimConfig`] surface flat on the transport object (mirroring
//! `aiperf.config.execution.DynosimOfflineTransport`, whose fields sit directly
//! on the transport like a `PoissonPhase` lifts its own knobs). `exclude_unset`
//! is reproduced by modeling every field as `Option`: a field is present on the
//! wire iff the author set it in YAML, so a field left at its Python default is
//! absent — byte-identical to the Pydantic dump.

use serde::{Deserialize, Serialize};

/// The typed inline transport selection (discriminated by `type`).
///
/// Serde's internal tagging emits the `type` discriminator; the DynoSim newtype
/// variants flatten [`DynosimConfig`]'s set fields alongside it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Transport {
    /// Native HTTP/1.1 or HTTP/2 transport.
    Http,
    /// Native gRPC transport (KServe OIP / Riva).
    Grpc,
    /// Offline virtual-clock Dynamo replay (fields flat on the transport).
    DynosimOffline(DynosimConfig),
    /// Online wall-clock Dynamo replay (fields flat on the transport).
    DynosimOnline(DynosimConfig),
    /// Lightweight fake execution leaf: analytic-latency synthetic responses,
    /// zero network (fields flat on the transport).
    DryRun(DryRunConfig),
}

impl Transport {
    /// Whether this is one of the in-process Dynamo co-simulation transports.
    pub fn is_dynosim(&self) -> bool {
        matches!(
            self,
            Transport::DynosimOffline(_) | Transport::DynosimOnline(_)
        )
    }

    /// Whether this is the no-server fake `dry_run` transport.
    pub fn is_dry_run(&self) -> bool {
        matches!(self, Transport::DryRun(_))
    }
}

/// The typed `dry_run` transport surface (analytic latency knobs, flat on the
/// transport). Mirrors `aiperf_runtime::engine::dry_run::DryRunTransportConfigV2`.
/// Fields are `Option` so only author-set knobs serialize (`exclude_unset`); the
/// runtime applies its own defaults for any omitted field.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DryRunConfig {
    /// Base time-to-first-token in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Base inter-token latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_ms: Option<f64>,
    /// Prefill cost per input token (ms): `TTFT += this · ISL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_per_isl_token_ms: Option<f64>,
    /// Super-linear prefill contention (ms per inflight²).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_concurrency_quad_ms: Option<f64>,
    /// Decode cost per output token (ms): `ITL += this · OSL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_per_osl_token_ms: Option<f64>,
    /// Linear decode contention (ms per inflight).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_concurrency_lin_ms: Option<f64>,
    /// Lognormal TTFT jitter coefficient of variation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_jitter_cv: Option<f64>,
    /// Lognormal ITL jitter coefficient of variation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_jitter_cv: Option<f64>,
    /// Root seed for the per-request jitter draw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Analytic latency curve: `linear` or `aiconfigurator_polynomial`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_model: Option<String>,
    /// KV-cache utilization for the polynomial decode curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_utilization: Option<f64>,
    /// Clock driver: `real` (default) or `sim` (deterministic virtual time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<String>,
}

/// The typed `transport.config` field surface for the DynoSim transports.
///
/// Mirrors `aiperf.config.dynosim.DynosimTransportConfig` (`extra="forbid"`).
/// The `engine` / `prefill_engine` / `decode_engine` / `router` objects are
/// opaque `MockEngineArgs` / `KvRouterConfig` JSON preserved verbatim for Dynamo
/// to validate. Every field is `Option` so only author-set fields serialize
/// (the `exclude_unset` contract); the `#[serde(alias)]` on each multi-word key
/// accepts the Config-v2 camelCase spelling on input while emitting snake_case.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimConfig {
    /// JSON engine profile consumed by Dynamo's canonical parser.
    #[serde(
        default,
        alias = "engineProfile",
        skip_serializing_if = "Option::is_none"
    )]
    pub engine_profile: Option<String>,
    /// Inline aggregate/single `MockEngineArgs` object (passed through verbatim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<serde_json::Value>,
    /// Inline disaggregated prefill `MockEngineArgs` object.
    #[serde(
        default,
        alias = "prefillEngine",
        skip_serializing_if = "Option::is_none"
    )]
    pub prefill_engine: Option<serde_json::Value>,
    /// Inline disaggregated decode `MockEngineArgs` object.
    #[serde(
        default,
        alias = "decodeEngine",
        skip_serializing_if = "Option::is_none"
    )]
    pub decode_engine: Option<serde_json::Value>,
    /// Inline `KvRouterConfig` object (passed through verbatim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<serde_json::Value>,
    /// Startup router policy-family YAML path overriding the inline `router`.
    #[serde(
        default,
        alias = "routerPolicyConfig",
        skip_serializing_if = "Option::is_none"
    )]
    pub router_policy_config: Option<String>,
    /// Model selector for a multi-model router policy document.
    #[serde(
        default,
        alias = "routerModelName",
        skip_serializing_if = "Option::is_none"
    )]
    pub router_model_name: Option<String>,
    /// Optional structured AIConfigurator overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aic: Option<DynosimAic>,
    /// Capture backend per-request records even without a JSONL artifact.
    #[serde(
        default,
        alias = "capturePerRequest",
        skip_serializing_if = "Option::is_none"
    )]
    pub capture_per_request: Option<bool>,
    /// Canonical goodput thresholds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sla: Option<DynosimSla>,
    /// Deployment topology (`single` / `aggregated` / `disaggregated`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    /// Aggregate worker count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers: Option<u32>,
    /// Disaggregated prefill worker count.
    #[serde(
        default,
        alias = "prefillWorkers",
        skip_serializing_if = "Option::is_none"
    )]
    pub prefill_workers: Option<u32>,
    /// Disaggregated decode worker count.
    #[serde(
        default,
        alias = "decodeWorkers",
        skip_serializing_if = "Option::is_none"
    )]
    pub decode_workers: Option<u32>,
    /// Router policy for routed topologies (`round_robin` / `kv`).
    #[serde(default, alias = "routerMode", skip_serializing_if = "Option::is_none")]
    pub router_mode: Option<String>,
    /// Optional build capabilities the exact runner image must carry. Sorted +
    /// deduped at authoring time to match Python's `sorted(set(...))` dump.
    #[serde(
        default,
        alias = "requiredFeatures",
        skip_serializing_if = "Option::is_none"
    )]
    pub required_features: Option<Vec<String>>,
    /// Backend-owned output artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<DynosimArtifacts>,
}

impl DynosimConfig {
    /// Sort + dedup `required_features` so the wire order is Python's
    /// `sorted(set(...))`.
    pub fn normalize(&mut self) {
        if let Some(features) = self.required_features.as_mut() {
            features.sort();
            features.dedup();
        }
    }
}

/// Canonical goodput thresholds (`DynosimSlaConfig`; each bound optional).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimSla {
    /// Maximum time to first token (ms).
    #[serde(default, alias = "ttftMs", skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Maximum mean inter-token latency (ms).
    #[serde(default, alias = "itlMs", skip_serializing_if = "Option::is_none")]
    pub itl_ms: Option<f64>,
    /// Maximum end-to-end latency (ms). Pydantic's `to_camel("e2e_ms")` is
    /// `e2EMs` (its digit-boundary rule), so that is the accepted camelCase key.
    #[serde(default, alias = "e2EMs", skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<f64>,
}

/// Backend-owned Dynamo artifacts written after a successful run
/// (`DynosimArtifactConfig`). Paths are relative to the run artifact target.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimArtifacts {
    /// Canonical aggregate Dynamo JSON report path.
    #[serde(default, alias = "reportJson", skip_serializing_if = "Option::is_none")]
    pub report_json: Option<String>,
    /// Canonical per-request Dynamo JSONL path.
    #[serde(
        default,
        alias = "perRequestJsonl",
        skip_serializing_if = "Option::is_none"
    )]
    pub per_request_jsonl: Option<String>,
    /// Timed worker/request/KV artifact JSON for trace workloads.
    #[serde(
        default,
        alias = "workerArtifactsJson",
        skip_serializing_if = "Option::is_none"
    )]
    pub worker_artifacts_json: Option<String>,
    /// Pass-start/pass-end KV visibility override.
    #[serde(
        default,
        alias = "kvEventVisibility",
        skip_serializing_if = "Option::is_none"
    )]
    pub kv_event_visibility: Option<String>,
}

/// Structured AIConfigurator overrides (`DynosimAicConfig`; all optional).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimAic {
    /// AIC serving backend identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// AIC GPU system identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Performance-database backend version.
    #[serde(
        default,
        alias = "backendVersion",
        skip_serializing_if = "Option::is_none"
    )]
    pub backend_version: Option<String>,
    /// Hugging Face model path for AIC.
    #[serde(default, alias = "modelPath", skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// Tensor-parallel degree.
    #[serde(default, alias = "tpSize", skip_serializing_if = "Option::is_none")]
    pub tp_size: Option<u32>,
    /// MoE tensor-parallel degree.
    #[serde(default, alias = "moeTpSize", skip_serializing_if = "Option::is_none")]
    pub moe_tp_size: Option<u32>,
    /// MoE expert-parallel degree.
    #[serde(default, alias = "moeEpSize", skip_serializing_if = "Option::is_none")]
    pub moe_ep_size: Option<u32>,
    /// Attention data-parallel degree.
    #[serde(
        default,
        alias = "attentionDpSize",
        skip_serializing_if = "Option::is_none"
    )]
    pub attention_dp_size: Option<u32>,
    /// Speculative (MTP) draft-token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nextn: Option<u32>,
    /// GEMM quantization override.
    #[serde(default, alias = "gemmDtype", skip_serializing_if = "Option::is_none")]
    pub gemm_dtype: Option<String>,
    /// MoE quantization override.
    #[serde(default, alias = "moeDtype", skip_serializing_if = "Option::is_none")]
    pub moe_dtype: Option<String>,
    /// Attention/FMHA quantization override.
    #[serde(default, alias = "fmhaDtype", skip_serializing_if = "Option::is_none")]
    pub fmha_dtype: Option<String>,
    /// KV-cache quantization override.
    #[serde(
        default,
        alias = "kvCacheDtype",
        skip_serializing_if = "Option::is_none"
    )]
    pub kv_cache_dtype: Option<String>,
    /// Collective/comm quantization override.
    #[serde(default, alias = "commDtype", skip_serializing_if = "Option::is_none")]
    pub comm_dtype: Option<String>,
    /// Comma-separated conditional draft acceptance rates.
    #[serde(
        default,
        alias = "nextnAcceptRates",
        skip_serializing_if = "Option::is_none"
    )]
    pub nextn_accept_rates: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_projects_type_only() {
        assert_eq!(
            serde_json::to_value(Transport::Http).unwrap(),
            serde_json::json!({"type": "http"})
        );
    }

    #[test]
    fn dynosim_offline_flattens_set_fields_only() {
        let cfg = DynosimConfig {
            topology: Some("single".into()),
            workers: Some(1),
            router_mode: Some("round_robin".into()),
            engine: Some(serde_json::json!({"block_size": 16})),
            sla: Some(DynosimSla {
                ttft_ms: Some(500.0),
                itl_ms: Some(20.0),
                e2e_ms: None,
            }),
            ..Default::default()
        };
        let v = serde_json::to_value(Transport::DynosimOffline(cfg)).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "type": "dynosim_offline",
                "topology": "single",
                "workers": 1,
                "router_mode": "round_robin",
                "engine": {"block_size": 16},
                "sla": {"ttft_ms": 500.0, "itl_ms": 20.0},
            })
        );
    }

    #[test]
    fn required_features_sorted_and_deduped() {
        let mut cfg = DynosimConfig {
            required_features: Some(vec![
                "dynamo-zmq-events".into(),
                "dynamo-router-runtime".into(),
                "dynamo-zmq-events".into(),
            ]),
            ..Default::default()
        };
        cfg.normalize();
        assert_eq!(
            cfg.required_features.unwrap(),
            vec!["dynamo-router-runtime", "dynamo-zmq-events"]
        );
    }
}
