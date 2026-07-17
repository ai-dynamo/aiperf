// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native AIC runtime construction for AIPerf's DynoSim co-simulation path.
//!
//! Builds an `aiconfigurator` timing engine and installs it on the mocker's
//! `MockEngineArgs::perf_model`, keeping the embedded-Python AIC bridge in the
//! consumer rather than in the pure-Rust simulator. The callback follows
//! `dynamo/lib/bindings/python/rust/llm/aic_callback.rs`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use aiconfigurator_core::{AicEngine, build_aic_engine};
use anyhow::{Context, Result, anyhow, ensure};
use dynamo_kv_router::PrefillLoadEstimator;
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use dynamo_mocker::common::perf_model::{AicCallback, PerfModel};
use dynamo_mocker::common::protocols::MockEngineArgs;

// `(short field label, aiconfigurator quant-mode enum class)`, ordered to match
// `normalize_quant_modes`' `values` array. The label appears in error messages;
// the enum class is looked up on `aiconfigurator.sdk.common` to resolve a mode.
const AIC_QUANT_FIELDS: [(&str, &str); 5] = [
    ("gemm", "GEMMQuantMode"),
    ("moe", "MoEQuantMode"),
    ("fmha", "FMHAQuantMode"),
    ("kvcache", "KVCacheQuantMode"),
    ("comm", "CommQuantMode"),
];
const DEFAULT_AIC_SYSTEM: &str = "h200_sxm";
const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.9;
const DEFAULT_MEM_FRACTION_STATIC: f64 = 0.88;
const DEFAULT_FREE_GPU_MEMORY_FRACTION: f64 = 0.9;

fn normalize_quant_mode(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    if value.is_empty()
        || matches!(
            value.to_ascii_lowercase().as_str(),
            "auto" | "none" | "null"
        )
    {
        return None;
    }
    Some(if value == "int4" { "int4_wo" } else { value }.to_string())
}

fn resolve_backend_version<'a>(backend: &str, authored: Option<&'a str>) -> &'a str {
    authored.unwrap_or(match backend {
        "sglang" => "0.5.10",
        "trtllm" => "1.3.0rc10",
        _ => "0.19.0",
    })
}

/// Pure-Rust hot-path callback over one compiled AIC engine.
pub struct NativeAicCallback {
    engine: Arc<AicEngine>,
}

// The `AicCallback` contract returns a bare latency (no error channel) and feeds
// every scheduler timing, so a failed AIC estimate is an unrecoverable misconfig:
// fail loudly rather than fabricate a latency that would corrupt the simulation.
// The `PrefillLoadEstimator` impl below has a fallible caller and returns `Result`.
impl AicCallback for NativeAicCallback {
    fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64 {
        self.engine
            .prefill_latency_ms(
                batch_size as u32,
                effective_isl.saturating_add(prefix) as u32,
                prefix as u32,
            )
            .unwrap_or_else(|error| panic!("AIC predict_prefill failed: {error}"))
    }

    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64 {
        self.engine
            .decode_latency_ms(batch_size as u32, isl as u32, osl as u32)
            .unwrap_or_else(|error| panic!("AIC predict_decode failed: {error}"))
    }
}

impl PrefillLoadEstimator for NativeAicCallback {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> Result<Duration> {
        let latency_ms = self
            .engine
            .prefill_latency_ms(
                batch_size as u32,
                effective_isl.saturating_add(prefix) as u32,
                prefix as u32,
            )
            .map_err(|error| anyhow!("AIC predict_prefill failed: {error}"))?;
        Ok(Duration::from_secs_f64(latency_ms / 1_000.0))
    }
}

fn normalize_quant_modes(args: &MockEngineArgs) -> Result<[Option<String>; 5]> {
    let values = [
        args.aic_gemm_dtype.as_deref(),
        args.aic_moe_dtype.as_deref(),
        args.aic_fmha_dtype.as_deref(),
        args.aic_kv_cache_dtype.as_deref(),
        args.aic_comm_dtype.as_deref(),
    ];
    Python::with_gil(|py| -> PyResult<[Option<String>; 5]> {
        let common = py.import("aiconfigurator.sdk.common")?;
        let mut normalized: [Option<String>; 5] = Default::default();
        for (index, ((field, enum_name), value)) in AIC_QUANT_FIELDS.iter().zip(values).enumerate()
        {
            let Some(value) = normalize_quant_mode(value) else {
                continue;
            };
            let enum_type = common.getattr(*enum_name)?;
            let mode = enum_type.get_item(&value).map_err(|_| {
                PyValueError::new_err(format!("unsupported AIC {field} quant mode {value:?}"))
            })?;
            normalized[index] = Some(mode.getattr("name")?.extract()?);
        }
        Ok(normalized)
    })
    .map_err(|error| anyhow!("failed to normalize AIC quantization fields: {error}"))
}

fn parse_accept_rates(value: Option<&str>) -> Result<Option<Vec<f64>>> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .split(',')
                .map(|item| {
                    item.trim()
                        .parse::<f64>()
                        .with_context(|| format!("invalid aic_nextn_accept_rates value {item:?}"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()
}

fn build_engine(args: &MockEngineArgs) -> Result<Arc<AicEngine>> {
    let backend = args
        .aic_backend
        .as_deref()
        .context("AIC requires aic_backend")?;
    let system = args.aic_system.as_deref().unwrap_or(DEFAULT_AIC_SYSTEM);
    let model = args
        .aic_model_path
        .as_deref()
        .context("AIC requires aic_model_path")?;
    let backend_version = resolve_backend_version(backend, args.aic_backend_version.as_deref());
    let quant = normalize_quant_modes(args)?;
    let nextn = args.aic_nextn.unwrap_or(0) as u32;
    let accept_rates = parse_accept_rates(args.undiscounted_aic_accept_rates().as_deref())?;
    let tp_size = args.aic_tp_size.unwrap_or(1);
    let attention_dp_size = args.aic_attention_dp_size.unwrap_or(1);

    static CACHE: OnceLock<Mutex<HashMap<String, Arc<AicEngine>>>> = OnceLock::new();
    let key = format!(
        "{backend}|{system}|{:?}|{model}|{tp_size}|{:?}|{:?}|{attention_dp_size}|{quant:?}|{nextn}|{accept_rates:?}",
        backend_version, args.aic_moe_tp_size, args.aic_moe_ep_size
    );
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(engine) = cache.lock().get(&key) {
        return Ok(Arc::clone(engine));
    }

    Python::with_gil(|py| {
        if let Err(error) = py
            .import("aiconfigurator.sdk.rust_engine_step")
            .and_then(|module| module.call_method0("_configure_default_data_roots"))
        {
            tracing::warn!(%error, "AIC data-root auto-configuration failed");
        }
    });
    let engine = build_aic_engine(
        model,
        system,
        backend,
        Some(backend_version),
        tp_size as u32,
        1,
        attention_dp_size as u32,
        args.aic_moe_tp_size.map(|value| value as u32),
        args.aic_moe_ep_size.map(|value| value as u32),
        quant[0].as_deref(),
        quant[1].as_deref(),
        quant[3].as_deref(),
        quant[2].as_deref(),
        quant[4].as_deref(),
        nextn,
        accept_rates,
        None,
        None,
    )
    .map_err(|error| {
        anyhow!("failed to build AIC engine for {model} / {system} / {backend}: {error}")
    })?;
    let engine = Arc::new(engine);
    cache.lock().insert(key, Arc::clone(&engine));
    Ok(engine)
}

/// Estimate the engine-wide offline KV capacity through the same Python helper
/// used by canonical replay.
///
/// The helper returns a rank-local count. Offline replay owns one global engine
/// pool, so attention-DP scales the result exactly as in
/// `lib/bindings/python/rust/llm/replay.rs:1668-1715`.
fn estimate_engine_num_gpu_blocks(args: &MockEngineArgs) -> Result<usize> {
    let backend = args
        .aic_backend
        .as_deref()
        .context("AIC requires aic_backend")?;
    let system = args.aic_system.as_deref().unwrap_or(DEFAULT_AIC_SYSTEM);
    let model = args
        .aic_model_path
        .as_deref()
        .context("AIC KV cache capacity estimation requires aic_model_path")?;

    let (backend_version, memory_fraction_kind, memory_fraction_value) = match backend {
        "vllm" => (
            resolve_backend_version(backend, args.aic_backend_version.as_deref()),
            "of_total",
            args.gpu_memory_utilization
                .unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION),
        ),
        "sglang" => (
            resolve_backend_version(backend, args.aic_backend_version.as_deref()),
            "of_total",
            args.mem_fraction_static
                .unwrap_or(DEFAULT_MEM_FRACTION_STATIC),
        ),
        "trtllm" => (
            resolve_backend_version(backend, args.aic_backend_version.as_deref()),
            "of_free",
            args.free_gpu_memory_fraction
                .unwrap_or(DEFAULT_FREE_GPU_MEMORY_FRACTION),
        ),
        other => {
            return Err(anyhow!(
                "AIC KV cache capacity estimation does not support backend {other:?}; \
                 supported backends: sglang, trtllm, vllm. Set num_gpu_blocks explicitly \
                 for this backend."
            ));
        }
    };
    let quant = normalize_quant_modes(args)?;

    let per_rank = Python::with_gil(|py| -> PyResult<usize> {
        let module = py.import("aiconfigurator.sdk.memory")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("backend_version", backend_version)?;
        kwargs.set_item("scheduler_block_size", args.block_size)?;
        kwargs.set_item(
            "max_num_tokens",
            args.max_num_batched_tokens
                .unwrap_or(DEFAULT_MAX_NUM_BATCHED_TOKENS),
        )?;
        kwargs.set_item("max_batch_size", 1)?;
        kwargs.set_item("memory_fraction_kind", memory_fraction_kind)?;
        kwargs.set_item("memory_fraction_value", memory_fraction_value)?;
        kwargs.set_item("tp_size", args.aic_tp_size.unwrap_or(1))?;
        kwargs.set_item("attention_dp_size", args.aic_attention_dp_size.unwrap_or(1))?;
        kwargs.set_item("moe_tp_size", args.aic_moe_tp_size)?;
        kwargs.set_item("moe_ep_size", args.aic_moe_ep_size)?;
        kwargs.set_item("gemm_quant_mode", quant[0].as_deref())?;
        kwargs.set_item("moe_quant_mode", quant[1].as_deref())?;
        kwargs.set_item("fmha_quant_mode", quant[2].as_deref())?;
        kwargs.set_item("kvcache_quant_mode", quant[3].as_deref())?;
        kwargs.set_item("comm_quant_mode", quant[4].as_deref())?;
        module
            .call_method(
                "estimate_num_gpu_blocks",
                (model, system, backend),
                Some(&kwargs),
            )?
            .extract()
    })
    .map_err(|error| anyhow!("failed to estimate AIC KV cache capacity: {error}"))?;

    Ok(per_rank.saturating_mul(args.aic_attention_dp_size.unwrap_or(1).max(1)))
}

/// Populate offload byte sizing from the model only when the user omitted it.
///
/// This deliberately preserves the canonical helper's `None` result rather
/// than inventing a fallback size.
fn populate_missing_offload_kv_bytes_per_token(args: &mut MockEngineArgs) -> Result<()> {
    if args.kv_bytes_per_token.is_some() {
        return Ok(());
    }
    let offload_requested = args.num_g2_blocks.unwrap_or_default() > 0
        || args.num_g3_blocks.unwrap_or_default() > 0
        || args.enable_g4_storage;
    if !offload_requested {
        return Ok(());
    }
    let Some(model_path) = args.aic_model_path.as_deref() else {
        return Ok(());
    };

    let kv_cache_dtype = normalize_quant_mode(args.aic_kv_cache_dtype.as_deref())
        .unwrap_or_else(|| "auto".to_string());
    let kv_bytes_per_token = match Python::with_gil(|py| -> PyResult<usize> {
        let auto_config = py.import("transformers")?.getattr("AutoConfig")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("trust_remote_code", false)?;
        let config = auto_config.call_method("from_pretrained", (model_path,), Some(&kwargs))?;
        let num_layers: usize = config.getattr("num_hidden_layers")?.extract()?;
        let num_attention_heads: usize = config.getattr("num_attention_heads")?.extract()?;
        let hidden_size: usize = config.getattr("hidden_size")?.extract()?;
        let num_kv_heads: usize = if let Ok(value) = config.getattr("num_key_value_heads") {
            value.extract()?
        } else if let Ok(value) = config.getattr("num_kv_heads") {
            value.extract()?
        } else {
            num_attention_heads
        };
        let dtype = if kv_cache_dtype == "auto" {
            config
                .getattr("dtype")
                .ok()
                .and_then(|value| value.str().ok())
                .and_then(|value| value.to_str().ok().map(str::to_string))
                .unwrap_or_else(|| "float16".to_string())
                .trim_start_matches("torch.")
                .to_string()
        } else {
            kv_cache_dtype.clone()
        };
        let dtype_bytes = match dtype.as_str() {
            "float16" | "bfloat16" => 2usize,
            "float32" => 4,
            "float8_e4m3fn" | "float8_e5m2" | "fp8" | "fp8_ds_mla" | "fp8_e4m3" | "fp8_inc"
            | "int8" => 1,
            _ => 2,
        };
        let head_dim = hidden_size
            .checked_div(num_attention_heads)
            .ok_or_else(|| PyValueError::new_err("num_attention_heads must be positive"))?;
        num_layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(num_kv_heads))
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(dtype_bytes))
            .ok_or_else(|| PyValueError::new_err("KV bytes per token overflow"))
    }) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::warn!(%error, %model_path, "could not compute kv_bytes_per_token from model config");
            None
        }
    };
    if let Some(kv_bytes_per_token) = kv_bytes_per_token {
        args.kv_bytes_per_token = Some(kv_bytes_per_token);
    }
    Ok(())
}

/// Activate native AIC timing on `args` and return the matching router
/// prefill-load estimator. Returns `None` when no AIC fields were requested.
pub fn configure_aic_runtime(
    args: &mut MockEngineArgs,
) -> Result<Option<Arc<dyn PrefillLoadEstimator>>> {
    populate_missing_offload_kv_bytes_per_token(args)?;

    // Canonical engine materialization treats `aic_backend` as the timing-model
    // switch. Other AIC metadata may still be present solely to size offload KV.
    if args.aic_backend.is_none() {
        return Ok(None);
    }
    ensure!(
        args.aic_model_path.is_some(),
        "AIC replay modeling requires aic_model_path"
    );
    if !args.num_gpu_blocks_explicit() {
        args.num_gpu_blocks = estimate_engine_num_gpu_blocks(args)?;
    }
    let engine = build_engine(args)?;
    let callback = Arc::new(NativeAicCallback { engine });
    let timing: Arc<dyn AicCallback> = callback.clone();
    let estimator: Arc<dyn PrefillLoadEstimator> = callback;
    args.perf_model = Arc::new(PerfModel::from_aic_callback_with_attention_dp(
        timing,
        args.aic_attention_dp_size.unwrap_or(1).max(1),
    ));
    Ok(Some(estimator))
}
