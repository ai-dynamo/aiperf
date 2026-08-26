// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reference-compatible random ISL/OSL range plans.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::dataset::error::{DatasetError, Result};
use crate::rng::compat::numpy_generator::NumpyGenerator;
use crate::rng::{ConfiguredRandomGenerator, RandomGenerator};

/// Reference benchmark whose random-dataset behavior is reproduced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomCorpusStyle {
    /// vLLM `benchmark_serving.py` / `RandomDataset` semantics.
    #[default]
    Vllm,
    /// SGLang `benchmark_serving.py --dataset-name random-ids` semantics.
    Sglang,
}

/// Public scalar or independent input/output ratio syntax.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RandomRangeRatioInput {
    /// One ratio applied to both windows.
    Same(f64),
    /// Independent vLLM input/output ratios.
    Split {
        /// Input window ratio.
        input: f64,
        /// Output window ratio.
        output: f64,
    },
}

impl<'de> Deserialize<'de> for RandomRangeRatioInput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum AuthoredRatio {
            Same(f64),
            Split(SplitRatio),
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct SplitRatio {
            input: f64,
            output: f64,
        }

        Ok(match AuthoredRatio::deserialize(deserializer)? {
            AuthoredRatio::Same(value) => Self::Same(value),
            AuthoredRatio::Split(value) => Self::Split {
                input: value.input,
                output: value.output,
            },
        })
    }
}

impl RandomRangeRatioInput {
    /// Lower the authored syntax into checked ratios for `style`.
    pub fn checked(self, style: RandomCorpusStyle) -> Result<RandomRangeRatio> {
        let ratio = match self {
            Self::Same(value) => RandomRangeRatio::same(value)?,
            Self::Split { input, output } => RandomRangeRatio::new(input, output)?,
        };
        ratio.validate(style)
    }
}

/// Independent input/output window ratios.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomRangeRatio {
    input: f64,
    output: f64,
}

impl RandomRangeRatio {
    /// Apply one ratio to both input and output windows.
    pub fn same(ratio: f64) -> Result<Self> {
        Self::new(ratio, ratio)
    }

    /// Construct independently authored input/output ratios.
    pub fn new(input: f64, output: f64) -> Result<Self> {
        if !input.is_finite() || !output.is_finite() || input < 0.0 || output < 0.0 {
            return Err(DatasetError::Validation(format!(
                "random range ratios must be finite and non-negative; got input={input}, output={output}"
            )));
        }
        Ok(Self { input, output })
    }

    /// Validate style-specific endpoints and shape.
    pub fn validate(self, style: RandomCorpusStyle) -> Result<Self> {
        match style {
            RandomCorpusStyle::Vllm if self.input >= 1.0 || self.output >= 1.0 => {
                Err(DatasetError::Validation(format!(
                    "vllm random range ratios must be within [0, 1); got input={}, output={}",
                    self.input, self.output
                )))
            }
            RandomCorpusStyle::Sglang
                if self.input > 1.0
                    || self.output > 1.0
                    || self.input.to_bits() != self.output.to_bits() =>
            {
                Err(DatasetError::Validation(format!(
                    "sglang random range ratio must be one value within [0, 1]; got input={}, output={}",
                    self.input, self.output
                )))
            }
            _ => Ok(self),
        }
    }
}

/// Checked style-specific bounds, before a run seed is applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomRangePlan {
    style: RandomCorpusStyle,
    input_bounds: (i64, i64),
    output_bounds: (i64, i64),
    special_tokens: i64,
}

impl RandomRangePlan {
    /// Build inclusive bounds from authored fixed means.
    pub fn new(
        style: RandomCorpusStyle,
        input_mean: i64,
        output_mean: i64,
        ratio: RandomRangeRatio,
        special_tokens: i64,
    ) -> Result<Self> {
        if input_mean < 0 || output_mean <= 0 || special_tokens < 0 {
            return Err(DatasetError::Validation(format!(
                "random range means require ISL >= 0, OSL > 0, and special tokens >= 0; got ISL={input_mean}, OSL={output_mean}, special={special_tokens}"
            )));
        }
        let ratio = ratio.validate(style)?;
        let bounds = |mean: i64, r: f64, floor: i64| {
            let low = ((mean as f64) * (1.0 - r)).floor() as i64;
            let high = ((mean as f64) * (1.0 + r)).ceil() as i64;
            (low.max(floor), high.max(floor))
        };
        let (input_bounds, output_bounds) = match style {
            RandomCorpusStyle::Vllm => (
                bounds((input_mean - special_tokens).max(0), ratio.input, 0),
                bounds(output_mean, ratio.output, 1),
            ),
            RandomCorpusStyle::Sglang => (
                (
                    (((input_mean as f64) * ratio.input).floor() as i64)
                        .max(1)
                        .min(input_mean),
                    input_mean,
                ),
                (
                    (((output_mean as f64) * ratio.output).floor() as i64)
                        .max(1)
                        .min(output_mean),
                    output_mean,
                ),
            ),
        };
        Ok(Self {
            style,
            input_bounds,
            output_bounds,
            special_tokens,
        })
    }

    /// Inclusive input bounds before SGLang's per-sample adjustment.
    pub const fn input_bounds(&self) -> (i64, i64) {
        self.input_bounds
    }

    /// Inclusive output bounds.
    pub const fn output_bounds(&self) -> (i64, i64) {
        self.output_bounds
    }

    /// Reject a vLLM window whose smallest prefix-plus-body input is empty.
    pub fn validate_minimum_input(&self, prefix_tokens: usize) -> Result<()> {
        if self.style == RandomCorpusStyle::Sglang {
            return Ok(());
        }
        let prefix_tokens = i64::try_from(prefix_tokens).map_err(|_| {
            DatasetError::Validation("synthetic prompt prefix length exceeds i64".into())
        })?;
        let minimum = prefix_tokens
            .checked_add(self.input_bounds.0)
            .ok_or_else(|| DatasetError::Validation("minimum synthetic input overflow".into()))?;
        if minimum < 1 {
            return Err(DatasetError::Validation(format!(
                "vllm random range produces a minimum input of {minimum} tokens after special-token adjustment; increase --isl or --prompt-prefix-length, or reduce --random-range-ratio"
            )));
        }
        Ok(())
    }

    /// Apply the style-specific post-draw input adjustment.
    pub fn adjust_input(&self, input: i64) -> i64 {
        match self.style {
            RandomCorpusStyle::Vllm => input,
            RandomCorpusStyle::Sglang => (input - self.special_tokens).max(1),
        }
    }

    /// Draw all inputs, then outputs, then offsets from one reference stream.
    pub fn preseed(
        &self,
        entries: usize,
        seed: u64,
        vocab_size: u32,
    ) -> Result<SeededRandomRangePlan> {
        if vocab_size == 0 {
            return Err(DatasetError::Validation(
                "random corpus requires a non-empty tokenizer vocabulary".into(),
            ));
        }
        if self.style == RandomCorpusStyle::Sglang && seed > u64::from(u32::MAX) {
            warn_folded_sglang_seed(seed);
        }
        let mut inputs = Vec::with_capacity(entries);
        let mut outputs = Vec::with_capacity(entries);
        let mut offsets = Vec::with_capacity(entries);
        let mut stream = ReferenceRandomStream::from_seed(self.style, seed);
        for _ in 0..entries {
            inputs.push(self.adjust_input(stream.draw_inclusive(self.input_bounds)?));
        }
        for _ in 0..entries {
            outputs.push(stream.draw_inclusive(self.output_bounds)?);
        }
        for _ in 0..entries {
            offsets.push(stream.draw_indices(vocab_size as usize, 1)?[0]);
        }
        Ok(SeededRandomRangePlan {
            policy: self.clone(),
            inputs,
            outputs,
            offsets,
            seed,
            vocab_size,
        })
    }
}

/// Cached reference draws for one synthetic dataset composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeededRandomRangePlan {
    policy: RandomRangePlan,
    inputs: Vec<i64>,
    outputs: Vec<i64>,
    offsets: Vec<usize>,
    seed: u64,
    vocab_size: u32,
}

impl SeededRandomRangePlan {
    /// Preseeded inputs in conversation order.
    pub fn inputs(&self) -> &[i64] {
        &self.inputs
    }
    /// Preseeded outputs in conversation order.
    pub fn outputs(&self) -> &[i64] {
        &self.outputs
    }
    /// Preseeded prompt offsets in conversation order.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Reference corpus style retained by this seeded plan.
    pub const fn style(&self) -> RandomCorpusStyle {
        self.policy.style
    }

    /// Recreate the shared reference stream immediately after the offset draws.
    pub(crate) fn continuation(&self) -> Result<ReferenceRandomStream> {
        let mut stream = ReferenceRandomStream::from_seed(self.policy.style, self.seed);
        for _ in 0..self.inputs.len() {
            stream.draw_inclusive(self.policy.input_bounds)?;
        }
        for _ in 0..self.outputs.len() {
            stream.draw_inclusive(self.policy.output_bounds)?;
        }
        for _ in 0..self.offsets.len() {
            stream.draw_indices(self.vocab_size as usize, 1)?;
        }
        Ok(stream)
    }

    /// Cached pair at one turn ordinal; `None` signals reference-cache exhaustion.
    pub fn pair(&self, ordinal: usize) -> Option<(i64, i64)> {
        Some((*self.inputs.get(ordinal)?, *self.outputs.get(ordinal)?))
    }

    /// Deterministic post-cache pair. Reference parity has already ended here.
    pub fn fallback_pair(&self, rng: &mut ConfiguredRandomGenerator) -> Result<(i64, i64)> {
        let input = rng
            .randint(self.policy.input_bounds.0, self.policy.input_bounds.1)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        let output = rng
            .randint(self.policy.output_bounds.0, self.policy.output_bounds.1)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        Ok((self.policy.adjust_input(input), output))
    }
}

fn fold_seed(seed: u64) -> u32 {
    seed as u32 ^ (seed >> 32) as u32
}

fn warn_folded_sglang_seed(seed: u64) {
    static WARNED: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let warned = WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    let is_first = warned
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(seed);
    if is_first {
        tracing::warn!(
            component = "dataset.random_range",
            seed,
            folded_seed = fold_seed(seed),
            "SGLang MT19937 folds seeds wider than u32; distinct authored seeds may alias"
        );
    }
}

pub(crate) struct ReferenceRandomStream {
    inner: ReferenceRandomStreamKind,
}

enum ReferenceRandomStreamKind {
    Vllm(NumpyGenerator),
    Sglang(NumpyRandomState),
}

impl ReferenceRandomStream {
    fn from_seed(style: RandomCorpusStyle, seed: u64) -> Self {
        let inner = match style {
            RandomCorpusStyle::Vllm => {
                ReferenceRandomStreamKind::Vllm(NumpyGenerator::from_seed(seed))
            }
            RandomCorpusStyle::Sglang => {
                ReferenceRandomStreamKind::Sglang(NumpyRandomState::from_seed(fold_seed(seed)))
            }
        };
        Self { inner }
    }

    fn draw_inclusive(&mut self, bounds: (i64, i64)) -> Result<i64> {
        let (low, high) = bounds;
        if low < 0 || high < low {
            return Err(DatasetError::Validation(format!(
                "random range bounds must satisfy 0 <= low <= high; got {low}..={high}"
            )));
        }
        match &mut self.inner {
            ReferenceRandomStreamKind::Vllm(rng) => {
                let high_exclusive = high.checked_add(1).ok_or_else(|| {
                    DatasetError::Validation("vllm random range upper bound exceeds i64".into())
                })?;
                let span = u64::try_from(high - low).map_err(|_| {
                    DatasetError::Validation("vllm random range span is invalid".into())
                })?;
                if span > u64::from(u32::MAX) {
                    return Err(DatasetError::Validation(format!(
                        "vllm random range span {span} exceeds the supported u32 interval"
                    )));
                }
                Ok(rng.integers(low, high_exclusive))
            }
            ReferenceRandomStreamKind::Sglang(rng) => {
                let low = u32::try_from(low).map_err(|_| {
                    DatasetError::Validation("sglang random range lower bound exceeds u32".into())
                })?;
                let high = u32::try_from(high).map_err(|_| {
                    DatasetError::Validation("sglang random range upper bound exceeds u32".into())
                })?;
                Ok(i64::from(rng.randint_inclusive(low, high)))
            }
        }
    }

    pub(crate) fn draw_indices(&mut self, upper: usize, count: usize) -> Result<Vec<usize>> {
        if upper == 0 {
            return Err(DatasetError::Validation(
                "reference random token pool cannot be empty".into(),
            ));
        }
        let upper = u32::try_from(upper).map_err(|_| {
            DatasetError::Validation("reference random token pool exceeds u32".into())
        })?;
        match &mut self.inner {
            ReferenceRandomStreamKind::Vllm(rng) => Ok((0..count)
                .map(|_| rng.integers(0, i64::from(upper)) as usize)
                .collect()),
            ReferenceRandomStreamKind::Sglang(rng) => Ok((0..count)
                .map(|_| rng.randint_inclusive(0, upper - 1) as usize)
                .collect()),
        }
    }
}

struct NumpyRandomState {
    state: [u32; 624],
    index: usize,
}

impl NumpyRandomState {
    fn from_seed(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253_u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut value = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    value ^= 0x9908_b0df;
                }
                self.state[i] = value;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn randint_inclusive(&mut self, low: u32, high: u32) -> u32 {
        if low == high {
            return low;
        }
        low + self.interval(high - low)
    }

    fn interval(&mut self, max: u32) -> u32 {
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_starts_after_all_lengths_and_offsets() {
        for (style, expected) in [
            (RandomCorpusStyle::Vllm, vec![4, 2, 2, 0, 0]),
            (RandomCorpusStyle::Sglang, vec![3, 3, 7, 3, 5]),
        ] {
            let ratio = RandomRangeRatio::same(match style {
                RandomCorpusStyle::Vllm => 0.0,
                RandomCorpusStyle::Sglang => 1.0,
            })
            .unwrap();
            let plan = RandomRangePlan::new(style, 12, 4, ratio, 0)
                .unwrap()
                .preseed(2, 0, 10)
                .unwrap();
            assert_eq!(
                plan.continuation().unwrap().draw_indices(9, 5).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn unsupported_reference_bounds_fail_instead_of_panicking() {
        let plan = RandomRangePlan::new(
            RandomCorpusStyle::Vllm,
            i64::MAX,
            1,
            RandomRangeRatio::same(0.0).unwrap(),
            0,
        )
        .unwrap();
        assert!(plan.preseed(1, 0, 10).is_err());

        let plan = RandomRangePlan::new(
            RandomCorpusStyle::Sglang,
            i64::from(u32::MAX) + 1,
            1,
            RandomRangeRatio::same(1.0).unwrap(),
            0,
        )
        .unwrap();
        assert!(plan.preseed(1, 0, 10).is_err());
    }

    #[test]
    fn vllm_empty_minimum_requires_an_additive_prefix() {
        let plan = RandomRangePlan::new(
            RandomCorpusStyle::Vllm,
            2,
            16,
            RandomRangeRatio::same(0.9).unwrap(),
            0,
        )
        .unwrap();
        let error = plan.validate_minimum_input(0).unwrap_err().to_string();
        assert!(error.contains("--isl"));
        assert!(error.contains("--prompt-prefix-length"));
        assert!(error.contains("--random-range-ratio"));
        plan.validate_minimum_input(20).unwrap();

        RandomRangePlan::new(
            RandomCorpusStyle::Sglang,
            2,
            16,
            RandomRangeRatio::same(0.9).unwrap(),
            0,
        )
        .unwrap()
        .validate_minimum_input(0)
        .unwrap();
    }

    #[test]
    fn ratio_input_rejects_bool_and_unknown_object_fields() {
        assert!(serde_json::from_str::<RandomRangeRatioInput>("true").is_err());
        assert!(
            serde_json::from_str::<RandomRangeRatioInput>(
                r#"{"input":0.2,"output":0.4,"unexpected":1}"#,
            )
            .is_err()
        );
    }
}
