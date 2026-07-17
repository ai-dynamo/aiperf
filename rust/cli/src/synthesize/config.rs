// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Agentic Code synthesis configuration.
//!
//! Defaults and `mu`/`sigma` derivation are part of the seeded sampling
//! contract. Unknown JSON keys are ignored.

use std::path::Path;

use serde_json::Value;

/// Lognormal parameters with real-space summary statistics.
///
/// `mu`/`sigma` are resolved
/// from `mean`/`median` when absent: `mu = ln(median)`,
/// `sigma = sqrt(2 * ln(mean/median))` if `mean/median > 1` else `0`.
#[derive(Clone, Debug)]
pub struct LognormalParams {
    pub mu: f64,
    pub sigma: f64,
    pub mean: f64,
    pub median: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl LognormalParams {
    /// Construct from real-space `mean` and `median`.
    pub fn from_mean_median(mean: f64, median: f64) -> Self {
        Self::new(mean, median, None, None, None, None)
    }

    /// Resolve missing `mu` or `sigma` from `mean` and `median`.
    pub fn new(
        mean: f64,
        median: f64,
        mu: Option<f64>,
        sigma: Option<f64>,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Self {
        let (mu, sigma) = match (mu, sigma) {
            (Some(mu), Some(sigma)) => (mu, sigma),
            _ => {
                let mu = median.ln();
                let ratio = mean / median;
                let sigma = if ratio > 1.0 {
                    (2.0 * ratio.ln()).sqrt()
                } else {
                    0.0
                };
                (mu, sigma)
            }
        };
        Self {
            mu,
            sigma,
            mean,
            median,
            min,
            max,
        }
    }

    /// Parse from a JSON object, honoring an explicit `mu`/`sigma` pair.
    fn from_json(v: &Value) -> anyhow::Result<Self> {
        let mean = v
            .get("mean")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("lognormal params require 'mean'"))?;
        let median = v
            .get("median")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("lognormal params require 'median'"))?;
        let mu = v.get("mu").and_then(Value::as_f64);
        let sigma = v.get("sigma").and_then(Value::as_f64);
        let min = v.get("min").and_then(Value::as_f64);
        let max = v.get("max").and_then(Value::as_f64);
        Ok(Self::new(mean, median, mu, sigma, min, max))
    }
}

/// Lognormal config for new tokens per turn with truncation-bias correction.
#[derive(Clone, Debug)]
pub struct NewTokensPerTurnConfig {
    pub params: LognormalParams,
    pub bias: f64,
}

/// Two-component mixture model for inter-turn delays.
#[derive(Clone, Debug)]
pub struct MixtureDelayConfig {
    pub agentic_fraction: f64,
    pub agentic_delay: LognormalParams,
    pub human_delay: LognormalParams,
    pub max: Option<f64>,
}

/// Context-dependent reset probability config.
#[derive(Clone, Debug)]
pub struct ResetConfig {
    pub base_probability: f64,
    pub context_scaling: f64,
}

/// Group assignment for L1.5 cache sharing via a Zipf distribution.
#[derive(Clone, Debug)]
pub struct Layer15GroupConfig {
    pub num_groups: usize,
    pub zipf_alpha: f64,
}

/// Token sizes for the KV cache prefix model.
#[derive(Clone, Debug)]
pub struct CacheLayerConfig {
    pub layer1_tokens: i64,
    pub layer1_5_tokens: i64,
    pub layer2: LognormalParams,
    pub layer1_5_groups: Layer15GroupConfig,
}

/// Explicit target distribution for turns per session.
#[derive(Clone, Debug)]
pub struct TurnCountConfig {
    pub mean: i64,
    pub median: i64,
    pub min: i64,
    pub max: i64,
    pub allow_truncation: bool,
    pub max_session_attempts: Option<i64>,
}

impl TurnCountConfig {
    /// Return bounded lognormal parameters for integer turn sampling.
    pub fn to_lognormal(&self) -> LognormalParams {
        LognormalParams::new(
            self.mean as f64,
            self.median as f64,
            None,
            None,
            Some(self.min as f64),
            Some(self.max as f64),
        )
    }
}

/// Configuration for synthesizing Agentic Code sessions.
#[derive(Clone, Debug)]
pub struct SessionDistributionConfig {
    pub new_tokens_per_turn: NewTokensPerTurnConfig,
    pub generation_length: LognormalParams,
    pub inter_turn_delay: MixtureDelayConfig,
    pub reset: Option<ResetConfig>,
    pub turns: Option<TurnCountConfig>,
    pub max_prompt_tokens: i64,
    pub block_size: i64,
    pub cache: CacheLayerConfig,
    pub restart_initial_probability: f64,
    pub restart_turn_range: [i64; 2],
}

impl Default for SessionDistributionConfig {
    fn default() -> Self {
        Self {
            new_tokens_per_turn: NewTokensPerTurnConfig {
                params: LognormalParams::from_mean_median(3500.0, 1800.0),
                bias: 1.0,
            },
            generation_length: LognormalParams::from_mean_median(500.0, 300.0),
            inter_turn_delay: MixtureDelayConfig {
                agentic_fraction: 0.7,
                agentic_delay: LognormalParams::from_mean_median(2500.0, 1800.0),
                human_delay: LognormalParams::from_mean_median(40000.0, 25000.0),
                max: None,
            },
            reset: Some(ResetConfig {
                base_probability: 0.02,
                context_scaling: 2.0,
            }),
            turns: None,
            max_prompt_tokens: 200_000,
            block_size: 512,
            cache: CacheLayerConfig {
                layer1_tokens: 32_000,
                layer1_5_tokens: 20_000,
                layer2: LognormalParams::from_mean_median(10_000.0, 5_000.0),
                layer1_5_groups: Layer15GroupConfig {
                    num_groups: 50,
                    zipf_alpha: 1.2,
                },
            },
            restart_initial_probability: 0.0,
            restart_turn_range: [5, 15],
        }
    }
}

impl SessionDistributionConfig {
    /// Load raw config JSON or a manifest containing `generation_params`.
    pub fn load(path_or_name: &str) -> anyhow::Result<Self> {
        if path_or_name == "default" && !Path::new(path_or_name).is_file() {
            // A named preset must resolve to a file; omitted config uses `Default`.
        }
        let p = Path::new(path_or_name);
        if p.is_file() {
            let bytes = std::fs::read(p)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", p.display()))?;
            let mut data: Value = serde_json::from_slice(&bytes)
                .map_err(|e| anyhow::anyhow!("config {} is not valid JSON: {e}", p.display()))?;
            if data.get("generation_params").is_some() {
                data = data
                    .get("generation_params")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("generation_params must be an object"))?;
            }
            return Self::from_value(&data);
        }
        anyhow::bail!(
            "Config '{path_or_name}' not found. Provide a path to a config or manifest JSON."
        )
    }

    /// Build from JSON, applying defaults and supported aliases.
    pub fn from_value(v: &Value) -> anyhow::Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("config must be a JSON object"))?;
        let mut cfg = Self::default();

        if let Some(ntp) = obj.get("new_tokens_per_turn") {
            let params = LognormalParams::from_json(ntp)?;
            let bias = ntp.get("bias").and_then(Value::as_f64).unwrap_or(1.0);
            cfg.new_tokens_per_turn = NewTokensPerTurnConfig { params, bias };
        }
        if let Some(gl) = obj.get("generation_length") {
            cfg.generation_length = LognormalParams::from_json(gl)?;
        }
        if let Some(itd) = obj.get("inter_turn_delay") {
            cfg.inter_turn_delay = MixtureDelayConfig {
                agentic_fraction: itd
                    .get("agentic_fraction")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.7),
                agentic_delay: match itd.get("agentic_delay") {
                    Some(a) => LognormalParams::from_json(a)?,
                    None => LognormalParams::from_mean_median(2500.0, 1800.0),
                },
                human_delay: match itd.get("human_delay") {
                    Some(h) => LognormalParams::from_json(h)?,
                    None => LognormalParams::from_mean_median(40000.0, 25000.0),
                },
                max: itd.get("max").and_then(Value::as_f64),
            };
        }
        // Explicit turn targets disable probabilistic reset unless reset is authored.
        let turns_present = obj.get("turns").is_some_and(|t| !t.is_null());
        if let Some(r) = obj.get("reset") {
            cfg.reset = if r.is_null() {
                None
            } else {
                Some(ResetConfig {
                    base_probability: r
                        .get("base_probability")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.02),
                    context_scaling: r
                        .get("context_scaling")
                        .and_then(Value::as_f64)
                        .unwrap_or(2.0),
                })
            };
        } else if turns_present {
            cfg.reset = None;
        }
        if turns_present {
            let t = &obj["turns"];
            let allow_truncation = t
                .get("allow_truncation")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let max_session_attempts = t.get("max_session_attempts").and_then(Value::as_i64);
            cfg.turns = Some(TurnCountConfig {
                mean: t.get("mean").and_then(Value::as_i64).unwrap_or(0),
                median: t.get("median").and_then(Value::as_i64).unwrap_or(0),
                min: t.get("min").and_then(Value::as_i64).unwrap_or(0),
                max: t.get("max").and_then(Value::as_i64).unwrap_or(0),
                allow_truncation,
                max_session_attempts: if allow_truncation {
                    None
                } else {
                    Some(max_session_attempts.unwrap_or(100))
                },
            });
        }
        if let Some(mpt) = obj.get("max_prompt_tokens").and_then(Value::as_i64) {
            cfg.max_prompt_tokens = mpt;
        }
        if let Some(bs) = obj.get("block_size").and_then(Value::as_i64) {
            cfg.block_size = bs;
        }
        if let Some(cache) = obj.get("cache") {
            let default_cache = CacheLayerConfig {
                layer1_tokens: 32_000,
                layer1_5_tokens: 20_000,
                layer2: LognormalParams::from_mean_median(10_000.0, 5_000.0),
                layer1_5_groups: Layer15GroupConfig {
                    num_groups: 50,
                    zipf_alpha: 1.2,
                },
            };
            cfg.cache = CacheLayerConfig {
                layer1_tokens: cache
                    .get("layer1_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(default_cache.layer1_tokens),
                layer1_5_tokens: cache
                    .get("layer1_5_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(default_cache.layer1_5_tokens),
                layer2: match cache.get("layer2") {
                    Some(l2) => LognormalParams::from_json(l2)?,
                    None => default_cache.layer2,
                },
                layer1_5_groups: match cache.get("layer1_5_groups") {
                    Some(g) => Layer15GroupConfig {
                        num_groups: g.get("num_groups").and_then(Value::as_u64).unwrap_or(50)
                            as usize,
                        zipf_alpha: g.get("zipf_alpha").and_then(Value::as_f64).unwrap_or(1.2),
                    },
                    None => default_cache.layer1_5_groups,
                },
            };
        }
        if let Some(rf) = obj.get("restart_fraction").and_then(Value::as_f64) {
            cfg.restart_initial_probability = rf;
        }
        if let Some(rip) = obj
            .get("restart_initial_probability")
            .and_then(Value::as_f64)
        {
            cfg.restart_initial_probability = rip;
        }
        if let Some(rtr) = obj.get("restart_turn_range").and_then(Value::as_array)
            && rtr.len() == 2
        {
            cfg.restart_turn_range = [rtr[0].as_i64().unwrap_or(5), rtr[1].as_i64().unwrap_or(15)];
        }
        Ok(cfg)
    }
}
