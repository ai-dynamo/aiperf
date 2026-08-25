// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Agentic Code session synthesis.
//!
//! Random draw order is part of the serialized dataset contract. The two modes
//! are restart-split and explicit-turn.

use aiperf_runtime::rng::compat::numpy_generator::NumpyGenerator;

use crate::synthesize::config::{LognormalParams, SessionDistributionConfig};
use crate::synthesize::dist::{sample_lognormal, sample_mixture_delay};
use crate::synthesize::prefix::PrefixAllocator;

const OUTPUT_MIN: i64 = 30;

/// Why a session ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionEndReason {
    ForcedRetire,
    ProbabilisticReset,
    RestartSplit,
    TargetTurnCount,
}

/// A synthesized turn.
#[derive(Clone, Debug)]
pub struct SynthesizedTurn {
    pub turn_index: i64,
    pub input_length: i64,
    pub output_length: i64,
    pub new_tokens: i64,
    pub delay_ms: f64,
    pub timestamp_ms: f64,
    pub hash_ids: Vec<i64>,
}

/// A synthesized multi-turn session.
#[derive(Clone, Debug)]
pub struct SynthesizedSession {
    pub session_id: String,
    pub group_id: i64,
    pub turns: Vec<SynthesizedTurn>,
    #[allow(dead_code)]
    pub end_reason: SessionEndReason,
    pub is_restart_continuation: bool,
}

/// Synthesizes multi-turn sessions from distribution config.
pub struct SessionSynthesizer<'a> {
    config: &'a SessionDistributionConfig,
    rng: NumpyGenerator,
    allocator: PrefixAllocator,
    session_counter: i64,
    group_weights: Vec<f64>,
    fixed_prefix: i64,
    output_min: i64,
    new_tokens_params: LognormalParams,
}

impl<'a> SessionSynthesizer<'a> {
    /// Construct with precomputed Zipf weights and bias-corrected token parameters.
    pub fn new(config: &'a SessionDistributionConfig, seed: u64) -> anyhow::Result<Self> {
        let allocator = PrefixAllocator::new(&config.cache, config.block_size)?;

        let ng = config.cache.layer1_5_groups.num_groups;
        let alpha = config.cache.layer1_5_groups.zipf_alpha;
        let mut weights: Vec<f64> = (1..=ng).map(|k| 1.0 / (k as f64).powf(alpha)).collect();
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        let fixed_prefix = allocator.prefix_tokens();

        let output_min = match config.generation_length.max {
            Some(gen_max) => OUTPUT_MIN.min(gen_max as i64),
            None => OUTPUT_MIN,
        };

        let ntp = &config.new_tokens_per_turn;
        let new_tokens_params = if ntp.bias != 1.0 {
            let shifted_mu = ntp.params.mu + ntp.bias.ln();
            LognormalParams {
                mu: shifted_mu,
                sigma: ntp.params.sigma,
                mean: ntp.params.mean * ntp.bias,
                median: ntp.params.median * ntp.bias,
                min: ntp.params.min,
                max: ntp.params.max,
            }
        } else {
            ntp.params.clone()
        };

        Ok(Self {
            config,
            rng: NumpyGenerator::from_seed(seed),
            allocator,
            session_counter: 0,
            group_weights: weights,
            fixed_prefix,
            output_min,
            new_tokens_params,
        })
    }

    fn next_session_index(&mut self) -> i64 {
        let idx = self.session_counter;
        self.session_counter += 1;
        idx
    }

    /// Always consume a random draw when reset configuration is present.
    fn should_reset(&mut self, input_length: i64) -> bool {
        let Some(cfg) = &self.config.reset else {
            return false;
        };
        let ratio = input_length as f64 / self.config.max_prompt_tokens as f64;
        let p = cfg.base_probability * (1.0 + (cfg.context_scaling - 1.0) * ratio);
        self.rng.random() < p
    }

    fn sample_group_id(&mut self) -> i64 {
        self.rng.choice_weighted(&self.group_weights) as i64
    }

    fn sample_initial_context(&mut self) -> i64 {
        let l2 = sample_lognormal(&self.config.cache.layer2, &mut self.rng, None, 100) as i64;
        let l2 = l2.max(1);
        (self.fixed_prefix + l2).min(self.config.max_prompt_tokens - 1)
    }

    fn sample_output_length(&mut self) -> i64 {
        sample_lognormal(
            &self.config.generation_length,
            &mut self.rng,
            Some(self.output_min as f64),
            100,
        ) as i64
    }

    fn sample_delay_ms(&mut self, prev_input: i64) -> f64 {
        let mut delay_ms = sample_mixture_delay(&self.config.inter_turn_delay, &mut self.rng);
        let context_ratio = prev_input as f64 / self.config.max_prompt_tokens as f64;
        delay_ms *= (1.0 - 0.8 * context_ratio).max(0.2);
        if let Some(m) = self.config.inter_turn_delay.max {
            delay_ms = delay_ms.min(m);
        }
        delay_ms
    }

    fn sample_new_tokens(&mut self) -> i64 {
        let nt = sample_lognormal(&self.new_tokens_params, &mut self.rng, None, 100) as i64;
        nt.max(1)
    }

    fn sample_turn_target(&mut self) -> i64 {
        let turns_cfg = self
            .config
            .turns
            .as_ref()
            .expect("explicit turn sampling requested without turns config");
        let sampled = sample_lognormal(&turns_cfg.to_lognormal(), &mut self.rng, None, 100);
        let target = banker_round(sampled);
        target.max(turns_cfg.min).min(turns_cfg.max)
    }

    /// Generate `sess-` followed by the first six RNG bytes as lowercase hex.
    fn session_id(&mut self) -> String {
        let b = self.rng.bytes(16);
        let mut s = String::with_capacity(5 + 12);
        s.push_str("sess-");
        for byte in &b[..6] {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    fn synthesize_explicit_turn_session(&mut self) -> anyhow::Result<SynthesizedSession> {
        let turns_cfg = self
            .config
            .turns
            .clone()
            .expect("explicit turn mode requested without turns config");

        let target_turns = self.sample_turn_target();
        let max_attempts = turns_cfg.max_session_attempts.unwrap_or(1).max(1);
        for _ in 0..max_attempts {
            let session_index = self.next_session_index();
            let session_id = self.session_id();
            let group_id = self.sample_group_id();

            let initial_ctx = self.sample_initial_context();
            if initial_ctx >= self.config.max_prompt_tokens {
                continue;
            }

            let output_len = self.sample_output_length();
            let mut timestamp_ms = 0.0_f64;
            let hash_ids =
                self.allocator
                    .turn_hash_ids(session_index, group_id, initial_ctx, None)?;
            let mut turns = vec![SynthesizedTurn {
                turn_index: 0,
                input_length: initial_ctx,
                output_length: output_len,
                new_tokens: initial_ctx,
                delay_ms: 0.0,
                timestamp_ms,
                hash_ids,
            }];

            let mut prev_input = initial_ctx;
            let mut prev_output = output_len;
            let mut realized = true;
            for turn_idx in 1..target_turns {
                let delay_ms = self.sample_delay_ms(prev_input);
                timestamp_ms += delay_ms;

                let new_tokens = self.sample_new_tokens();
                let input_length = prev_input + prev_output + new_tokens;
                if input_length >= self.config.max_prompt_tokens {
                    if turns_cfg.allow_truncation {
                        return Ok(SynthesizedSession {
                            session_id,
                            group_id,
                            turns,
                            end_reason: SessionEndReason::ForcedRetire,
                            is_restart_continuation: false,
                        });
                    }
                    realized = false;
                    break;
                }

                let output_len = self.sample_output_length();
                let prev_session = self
                    .allocator
                    .extract_session_ids(&turns.last().unwrap().hash_ids);
                let hash_ids = self.allocator.turn_hash_ids(
                    session_index,
                    group_id,
                    input_length,
                    Some(&prev_session),
                )?;
                turns.push(SynthesizedTurn {
                    turn_index: turn_idx,
                    input_length,
                    output_length: output_len,
                    new_tokens,
                    delay_ms,
                    timestamp_ms,
                    hash_ids,
                });
                prev_input = input_length;
                prev_output = output_len;
            }

            if realized {
                return Ok(SynthesizedSession {
                    session_id,
                    group_id,
                    turns,
                    end_reason: SessionEndReason::TargetTurnCount,
                    is_restart_continuation: false,
                });
            }
        }

        anyhow::bail!(
            "Failed to synthesize explicit-turn session for target_turns={target_turns} after {max_attempts} attempts with max_prompt_tokens={}",
            self.config.max_prompt_tokens
        )
    }

    fn synthesize_session(
        &mut self,
        inject_restart: bool,
    ) -> anyhow::Result<Vec<SynthesizedSession>> {
        if self.config.turns.is_some() {
            return Ok(vec![self.synthesize_explicit_turn_session()?]);
        }

        let session_index = self.next_session_index();
        let session_id = self.session_id();
        let mut turns: Vec<SynthesizedTurn> = Vec::new();

        let [lo, hi] = self.config.restart_turn_range;
        let restart_at_turn = if inject_restart {
            self.rng.integers(lo, hi)
        } else {
            -1
        };

        let group_id = self.sample_group_id();

        let initial_ctx = self.sample_initial_context();
        let output_len = self.sample_output_length();

        let mut timestamp_ms = 0.0_f64;
        let hash_ids = self
            .allocator
            .turn_hash_ids(session_index, group_id, initial_ctx, None)?;

        turns.push(SynthesizedTurn {
            turn_index: 0,
            input_length: initial_ctx,
            output_length: output_len,
            new_tokens: initial_ctx,
            delay_ms: 0.0,
            timestamp_ms,
            hash_ids,
        });

        let mut prev_input = initial_ctx;
        let mut prev_output = output_len;

        let mut turn_idx = 1_i64;
        let mut end_reason = SessionEndReason::ForcedRetire;
        loop {
            if turn_idx == restart_at_turn {
                let session_a = SynthesizedSession {
                    session_id: session_id.clone(),
                    group_id,
                    turns: turns.clone(),
                    end_reason: SessionEndReason::RestartSplit,
                    is_restart_continuation: false,
                };
                let prev_hash = turns.last().unwrap().hash_ids.clone();
                let session_b = self.synthesize_continuation(
                    session_index,
                    group_id,
                    prev_input,
                    prev_output,
                    &prev_hash,
                )?;
                return Ok(vec![session_a, session_b]);
            }

            let delay_ms = self.sample_delay_ms(prev_input);
            timestamp_ms += delay_ms;

            let new_tokens = self.sample_new_tokens();

            let input_length = prev_input + prev_output + new_tokens;

            if input_length >= self.config.max_prompt_tokens {
                break;
            }

            if self.should_reset(input_length) {
                end_reason = SessionEndReason::ProbabilisticReset;
                break;
            }

            let output_len = self.sample_output_length();

            let prev_session = self
                .allocator
                .extract_session_ids(&turns.last().unwrap().hash_ids);
            let hash_ids = self.allocator.turn_hash_ids(
                session_index,
                group_id,
                input_length,
                Some(&prev_session),
            )?;

            turns.push(SynthesizedTurn {
                turn_index: turn_idx,
                input_length,
                output_length: output_len,
                new_tokens,
                delay_ms,
                timestamp_ms,
                hash_ids,
            });

            prev_input = input_length;
            prev_output = output_len;
            turn_idx += 1;
        }

        Ok(vec![SynthesizedSession {
            session_id,
            group_id,
            turns,
            end_reason,
            is_restart_continuation: false,
        }])
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_continuation(
        &mut self,
        session_index: i64,
        group_id: i64,
        prev_input: i64,
        prev_output: i64,
        prev_hash_ids: &[i64],
    ) -> anyhow::Result<SynthesizedSession> {
        let session_id = self.session_id();

        let initial_input = (prev_input + prev_output).min(self.config.max_prompt_tokens - 1);

        let output_len = self.sample_output_length();

        let prev_session_ids = self.allocator.extract_session_ids(prev_hash_ids);
        let hash_ids = self.allocator.turn_hash_ids(
            session_index,
            group_id,
            initial_input,
            Some(&prev_session_ids),
        )?;

        let mut turns = vec![SynthesizedTurn {
            turn_index: 0,
            input_length: initial_input,
            output_length: output_len,
            new_tokens: initial_input,
            delay_ms: 0.0,
            timestamp_ms: 0.0,
            hash_ids,
        }];

        let mut prev_input_b = initial_input;
        let mut prev_output_b = output_len;
        let mut turn_idx = 1_i64;
        let mut end_reason = SessionEndReason::ForcedRetire;

        loop {
            let delay_ms = self.sample_delay_ms(prev_input_b);
            let timestamp_ms = turns.last().unwrap().timestamp_ms + delay_ms;

            let new_tokens = self.sample_new_tokens();

            let input_length = prev_input_b + prev_output_b + new_tokens;

            if input_length >= self.config.max_prompt_tokens {
                break;
            }

            if self.should_reset(input_length) {
                end_reason = SessionEndReason::ProbabilisticReset;
                break;
            }

            let output_len = self.sample_output_length();

            let prev_session = self
                .allocator
                .extract_session_ids(&turns.last().unwrap().hash_ids);
            let hash_ids = self.allocator.turn_hash_ids(
                session_index,
                group_id,
                input_length,
                Some(&prev_session),
            )?;

            turns.push(SynthesizedTurn {
                turn_index: turn_idx,
                input_length,
                output_length: output_len,
                new_tokens,
                delay_ms,
                timestamp_ms,
                hash_ids,
            });

            prev_input_b = input_length;
            prev_output_b = output_len;
            turn_idx += 1;
        }

        Ok(SynthesizedSession {
            session_id,
            group_id,
            turns,
            end_reason,
            is_restart_continuation: true,
        })
    }

    pub fn synthesize_sessions(
        &mut self,
        num_sessions: usize,
    ) -> anyhow::Result<Vec<SynthesizedSession>> {
        if self.config.turns.is_some() {
            let mut out = Vec::with_capacity(num_sessions);
            for _ in 0..num_sessions {
                out.push(self.synthesize_session(false)?.remove(0));
            }
            return Ok(out);
        }

        let restart_probability = self.config.restart_initial_probability;
        let cutoff = 0.75_f64;
        let mut primary: Vec<SynthesizedSession> = Vec::new();
        let mut deferred: Vec<(SynthesizedSession, usize)> = Vec::new();
        for i in 0..num_sessions {
            let progress = i as f64 / (num_sessions.max(2) - 1) as f64;
            let p_restart = if progress >= cutoff {
                0.0
            } else {
                restart_probability * (1.0 - progress / cutoff)
            };
            let inject = self.rng.random() < p_restart;
            let mut result = self.synthesize_session(inject)?;
            let origin_index = primary.len();
            let first = result.remove(0);
            primary.push(first);
            for session in result {
                deferred.push((session, origin_index));
            }
        }

        if deferred.is_empty() {
            return Ok(primary);
        }

        let min_offset = ((num_sessions as f64 * 0.25) as i64).max(1) as usize;
        for (session_b, origin_index) in deferred {
            let low = (origin_index + min_offset).min(primary.len());
            let pos = if low >= primary.len() {
                primary.len()
            } else {
                self.rng.integers(low as i64, primary.len() as i64 + 1) as usize
            };
            primary.insert(pos, session_b);
        }

        Ok(primary)
    }

    pub fn block_size(&self) -> i64 {
        self.config.block_size
    }
}

/// Round half to even.
fn banker_round(x: f64) -> i64 {
    let r = x.round_ties_even();
    r as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesize::config::{Layer15GroupConfig, TurnCountConfig};

    #[test]
    fn initial_context_starts_after_whole_shared_prefix() {
        let mut config = SessionDistributionConfig::default();
        config.block_size = 64;
        config.max_prompt_tokens = 10_000;
        config.cache.layer1_tokens = 1_000;
        config.cache.layer1_5_tokens = 500;
        config.cache.layer2 = LognormalParams::from_mean_median(1.0, 1.0);
        config.cache.layer1_5_groups = Layer15GroupConfig {
            num_groups: 1,
            zipf_alpha: 1.0,
        };
        config.turns = Some(TurnCountConfig {
            mean: 1,
            median: 1,
            min: 1,
            max: 1,
            allow_truncation: false,
            max_session_attempts: Some(1),
        });
        config.reset = None;

        let mut synthesizer = SessionSynthesizer::new(&config, 42).unwrap();
        let session = synthesizer.synthesize_sessions(1).unwrap().remove(0);
        let first_turn = &session.turns[0];
        assert_eq!(first_turn.input_length, 1537);
        assert_eq!(first_turn.hash_ids.len(), 25);
    }
}
