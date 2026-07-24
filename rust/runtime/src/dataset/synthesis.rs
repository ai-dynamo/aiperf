// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native trace-synthesis policy.
//!
//! Loaders retain ownership of format-specific grouping and reconstruction; this
//! module owns the shared prefix-width/depth, timestamp, ISL, and OSL transforms.

use std::collections::{HashMap, HashSet};

use crate::rng::namespace::DATASET_SYNTHESIS_SYNTHESIZER;
use crate::rng::{ConfiguredRandomGenerator, RandomGenerator, RngRoot};

use crate::dataset::error::{DatasetError, Result};

/// Typed Config-v2 trace-synthesis parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSynthesisConfig {
    /// Timestamp divisor; values above one replay faster.
    pub speedup_ratio: f64,
    /// Shared-prefix depth multiplier.
    pub prefix_len_multiplier: f64,
    /// Number of independent prefix roots.
    pub prefix_root_multiplier: u64,
    /// Unique-prompt length multiplier.
    pub prompt_len_multiplier: f64,
    /// Output-length multiplier.
    pub output_len_multiplier: f64,
    /// Optional transformed-ISL cap.
    pub max_isl: Option<u64>,
    /// Optional transformed-OSL cap.
    pub max_osl: Option<u32>,
    /// Hash block size in tokens.
    pub block_size: usize,
}

impl Default for TraceSynthesisConfig {
    fn default() -> Self {
        Self {
            speedup_ratio: 1.0,
            prefix_len_multiplier: 1.0,
            prefix_root_multiplier: 1,
            prompt_len_multiplier: 1.0,
            output_len_multiplier: 1.0,
            max_isl: None,
            max_osl: None,
            block_size: 512,
        }
    }
}

impl TraceSynthesisConfig {
    /// Validate values before constructing a synthesis strategy.
    pub fn validate(&self) -> Result<()> {
        if !self.speedup_ratio.is_finite() || self.speedup_ratio <= 0.0 {
            return Err(invalid("speedup_ratio must be finite and positive"));
        }
        if !self.prefix_len_multiplier.is_finite() || self.prefix_len_multiplier <= 0.0 {
            return Err(invalid("prefix_len_multiplier must be finite and positive"));
        }
        if self.prefix_root_multiplier == 0 {
            return Err(invalid("prefix_root_multiplier must be positive"));
        }
        if !self.prompt_len_multiplier.is_finite() || self.prompt_len_multiplier <= 0.0 {
            return Err(invalid("prompt_len_multiplier must be finite and positive"));
        }
        if !self.output_len_multiplier.is_finite() || self.output_len_multiplier < 0.0 {
            return Err(invalid(
                "output_len_multiplier must be finite and non-negative",
            ));
        }
        if self.max_isl == Some(0) || self.max_osl == Some(0) || self.block_size == 0 {
            return Err(invalid(
                "synthesis caps and block_size must be positive when configured",
            ));
        }
        Ok(())
    }

    /// Whether the Python runtime invokes the structural synthesizer.
    ///
    /// ISL/OSL caps alone are loader/finalizer policy and do not trigger prefix
    /// rewriting.
    pub fn has_structural_transform(&self) -> bool {
        self.speedup_ratio != 1.0
            || self.prefix_len_multiplier != 1.0
            || self.prefix_root_multiplier != 1
            || self.prompt_len_multiplier != 1.0
            || self.output_len_multiplier != 1.0
    }
}

/// Format-neutral trace fields transformed by a [`TraceSynthesizer`].
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSynthesisRecord {
    /// Hash-block identifiers in authored prefix order.
    pub hash_ids: Vec<i64>,
    /// Input length in tokens.
    pub input_length: u64,
    /// Optional absolute schedule timestamp in milliseconds.
    pub timestamp_ms: Option<f64>,
    /// Optional output length.
    pub output_length: Option<u32>,
}

/// Extension seam for format-neutral trace transformations.
pub trait TraceSynthesizer {
    /// Mutate records in loader-defined grouped order.
    fn synthesize(&mut self, records: &mut [TraceSynthesisRecord]) -> Result<()>;
}

/// Prefix-pattern synthesizer.
pub struct PrefixTraceSynthesizer {
    config: TraceSynthesisConfig,
    rng: ConfiguredRandomGenerator,
}

impl PrefixTraceSynthesizer {
    /// Construct a reproducible synthesizer from the dataset RNG root.
    pub fn new(config: TraceSynthesisConfig, root: RngRoot) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            rng: root.derive_generator(DATASET_SYNTHESIS_SYNTHESIZER),
        })
    }

    fn structural_transform(&mut self, records: &mut [TraceSynthesisRecord]) -> Result<()> {
        let mut counts = HashMap::<i64, usize>::new();
        for hash in records.iter().flat_map(|record| record.hash_ids.iter()) {
            *counts.entry(*hash).or_default() += 1;
        }
        let shared = counts
            .into_iter()
            .filter_map(|(hash, count)| (count > 1).then_some(hash))
            .collect::<HashSet<_>>();
        let max_shared = shared.iter().copied().max().unwrap_or(0);
        let integer_multiplier = if self.config.prefix_len_multiplier > 1.0 {
            trunc_f64_to_i64(self.config.prefix_len_multiplier, "prefix_len_multiplier")?
        } else {
            1
        };
        let mut max_hash_id = checked_i64_mul(
            checked_i64_add(max_shared, 1, "shared hash bound")?,
            integer_multiplier,
            "stretched shared hash bound",
        )?;
        let block_size = u64::try_from(self.config.block_size)
            .map_err(|_| invalid("synthesis block_size exceeds u64"))?;

        for record in records.iter_mut() {
            if record.hash_ids.is_empty() {
                continue;
            }
            let prefix_ids = record
                .hash_ids
                .iter()
                .copied()
                .take_while(|hash| shared.contains(hash))
                .collect::<Vec<_>>();
            let prefix_len = u64::try_from(prefix_ids.len())
                .ok()
                .and_then(|count| count.checked_mul(block_size))
                .ok_or_else(|| invalid("trace shared-prefix length overflow"))?;
            let prompt_len = record.input_length.checked_sub(prefix_len).ok_or_else(|| {
                invalid(format!(
                    "input_len ({}) < prefix_len ({prefix_len}): trace has fewer tokens than its shared prefix blocks",
                    record.input_length
                ))
            })?;

            let (mut stretched, new_prefix_len) = if self.config.prefix_len_multiplier > 1.0 {
                let multiplier = usize::try_from(integer_multiplier)
                    .map_err(|_| invalid("prefix multiplier exceeds usize"))?;
                let mut stretched = Vec::with_capacity(
                    prefix_ids
                        .len()
                        .checked_mul(multiplier)
                        .ok_or_else(|| invalid("stretched prefix allocation overflow"))?,
                );
                for hash in &prefix_ids {
                    for offset in 0..multiplier {
                        stretched.push(checked_i64_add(
                            checked_i64_mul(*hash, integer_multiplier, "stretched prefix hash")?,
                            i64::try_from(offset)
                                .map_err(|_| invalid("prefix offset exceeds i64"))?,
                            "stretched prefix hash",
                        )?);
                    }
                }
                let target = ceil_f64_to_usize(
                    prefix_ids.len() as f64 * self.config.prefix_len_multiplier,
                    "target prefix blocks",
                )?;
                let extra = target.saturating_sub(stretched.len());
                append_unique_hashes(&mut stretched, &mut max_hash_id, extra)?;
                let length = u64::try_from(stretched.len())
                    .ok()
                    .and_then(|count| count.checked_mul(block_size))
                    .ok_or_else(|| invalid("stretched prefix length overflow"))?;
                (stretched, length)
            } else if self.config.prefix_len_multiplier < 1.0 {
                let scaled = trunc_f64_to_usize(
                    prefix_ids.len() as f64 * self.config.prefix_len_multiplier,
                    "squeezed prefix blocks",
                )?;
                let blocks = scaled.max(1);
                let stretched = prefix_ids[..prefix_ids.len().min(blocks)].to_vec();
                let length = u64::try_from(blocks)
                    .ok()
                    .and_then(|count| count.checked_mul(block_size))
                    .ok_or_else(|| invalid("squeezed prefix length overflow"))?;
                (stretched, length)
            } else {
                (prefix_ids, prefix_len)
            };

            let new_prompt_len = trunc_f64_to_u64(
                prompt_len as f64 * self.config.prompt_len_multiplier,
                "scaled prompt length",
            )?;
            let prompt_blocks = new_prompt_len
                .checked_add(block_size - 1)
                .and_then(|value| value.checked_div(block_size))
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid("scaled prompt block count overflow"))?;
            append_unique_hashes(&mut stretched, &mut max_hash_id, prompt_blocks)?;
            record.hash_ids = stretched;
            record.input_length = new_prefix_len
                .checked_add(new_prompt_len)
                .ok_or_else(|| invalid("synthesized input length overflow"))?;
        }

        if self.config.prefix_root_multiplier > 1 {
            let high = i64::try_from(self.config.prefix_root_multiplier - 1)
                .map_err(|_| invalid("prefix_root_multiplier exceeds i64"))?;
            let offset_base = checked_i64_add(max_hash_id, 1, "prefix-root offset base")?;
            for record in records
                .iter_mut()
                .filter(|record| !record.hash_ids.is_empty())
            {
                let tree = self
                    .rng
                    .randint(0, high)
                    .map_err(|error| invalid(error.to_string()))?;
                if tree > 0 {
                    let offset = checked_i64_mul(tree, offset_base, "prefix-root offset")?;
                    for hash in &mut record.hash_ids {
                        *hash = checked_i64_add(*hash, offset, "prefix-root hash")?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl TraceSynthesizer for PrefixTraceSynthesizer {
    fn synthesize(&mut self, records: &mut [TraceSynthesisRecord]) -> Result<()> {
        if !self.config.has_structural_transform() {
            return Ok(());
        }
        self.structural_transform(records)?;
        for record in records {
            if let Some(cap) = self.config.max_isl {
                record.input_length = record.input_length.min(cap);
            }
            if let Some(timestamp) = record.timestamp_ms {
                let scaled = timestamp / self.config.speedup_ratio;
                if !scaled.is_finite() {
                    return Err(invalid("scaled trace timestamp is not finite"));
                }
                record.timestamp_ms = Some(scaled.trunc());
            }
            if let Some(output) = record.output_length {
                let scaled = f64::from(output) * self.config.output_len_multiplier;
                if !scaled.is_finite() || scaled > f64::from(u32::MAX) {
                    return Err(invalid("scaled output length exceeds u32"));
                }
                let mut output = (scaled.round_ties_even() as u32).max(1);
                if let Some(cap) = self.config.max_osl {
                    output = output.min(cap);
                }
                record.output_length = Some(output);
            }
        }
        Ok(())
    }
}

fn append_unique_hashes(target: &mut Vec<i64>, max_hash_id: &mut i64, count: usize) -> Result<()> {
    for _ in 0..count {
        *max_hash_id = checked_i64_add(*max_hash_id, 1, "unique prompt hash")?;
        target.push(*max_hash_id);
    }
    Ok(())
}

fn checked_i64_add(left: i64, right: i64, field: &str) -> Result<i64> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{field} overflowed i64")))
}

fn checked_i64_mul(left: i64, right: i64, field: &str) -> Result<i64> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format!("{field} overflowed i64")))
}

fn trunc_f64_to_i64(value: f64, field: &str) -> Result<i64> {
    if !value.is_finite() || value < i64::MIN as f64 || value >= i64::MAX as f64 {
        return Err(invalid(format!("{field} is outside i64 range")));
    }
    Ok(value.trunc() as i64)
}

fn trunc_f64_to_u64(value: f64, field: &str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
        return Err(invalid(format!("{field} is outside u64 range")));
    }
    Ok(value.trunc() as u64)
}

fn trunc_f64_to_usize(value: f64, field: &str) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value >= usize::MAX as f64 {
        return Err(invalid(format!("{field} is outside usize range")));
    }
    Ok(value.trunc() as usize)
}

fn ceil_f64_to_usize(value: f64, field: &str) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value >= usize::MAX as f64 {
        return Err(invalid(format!("{field} is outside usize range")));
    }
    Ok(value.ceil() as usize)
}

fn invalid(message: impl Into<String>) -> DatasetError {
    DatasetError::Validation(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_depth_prompt_output_and_time_follow_python_order() {
        let config = TraceSynthesisConfig {
            speedup_ratio: 2.0,
            prefix_len_multiplier: 2.0,
            prompt_len_multiplier: 1.5,
            output_len_multiplier: 1.5,
            block_size: 4,
            ..TraceSynthesisConfig::default()
        };
        let mut records = vec![
            TraceSynthesisRecord {
                hash_ids: vec![1, 2],
                input_length: 10,
                timestamp_ms: Some(100.0),
                output_length: Some(2),
            },
            TraceSynthesisRecord {
                hash_ids: vec![1, 3],
                input_length: 10,
                timestamp_ms: Some(201.0),
                output_length: Some(3),
            },
        ];

        PrefixTraceSynthesizer::new(config, RngRoot::new(Some(7)))
            .unwrap()
            .synthesize(&mut records)
            .unwrap();

        assert_eq!(records[0].hash_ids, vec![2, 3, 5, 6, 7]);
        assert_eq!(records[1].hash_ids, vec![2, 3, 8, 9, 10]);
        assert_eq!(records[0].input_length, 17);
        assert_eq!(records[0].timestamp_ms, Some(50.0));
        assert_eq!(records[1].timestamp_ms, Some(100.0));
        assert_eq!(records[0].output_length, Some(3));
        assert_eq!(records[1].output_length, Some(4));
    }

    #[test]
    fn caps_alone_do_not_rewrite_structural_fields() {
        let config = TraceSynthesisConfig {
            max_isl: Some(4),
            max_osl: Some(2),
            ..TraceSynthesisConfig::default()
        };
        let original = TraceSynthesisRecord {
            hash_ids: vec![1],
            input_length: 8,
            timestamp_ms: Some(9.0),
            output_length: Some(5),
        };
        let mut records = vec![original.clone()];
        PrefixTraceSynthesizer::new(config, RngRoot::new(Some(1)))
            .unwrap()
            .synthesize(&mut records)
            .unwrap();
        assert_eq!(records, vec![original]);
    }
}
