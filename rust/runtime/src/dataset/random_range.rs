// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reference-compatible random ISL/OSL range plans.

use serde::{Deserialize, Serialize};

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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
        let mut inputs = Vec::with_capacity(entries);
        let mut outputs = Vec::with_capacity(entries);
        let mut offsets = Vec::with_capacity(entries);
        match self.style {
            RandomCorpusStyle::Vllm => {
                let mut rng = NumpyGenerator::from_seed(seed);
                for _ in 0..entries {
                    inputs.push(
                        self.adjust_input(
                            rng.integers(self.input_bounds.0, self.input_bounds.1 + 1),
                        ),
                    );
                }
                for _ in 0..entries {
                    outputs.push(rng.integers(self.output_bounds.0, self.output_bounds.1 + 1));
                }
                for _ in 0..entries {
                    offsets.push(rng.integers(0, i64::from(vocab_size)) as usize);
                }
            }
            RandomCorpusStyle::Sglang => {
                let mut rng = NumpyRandomState::from_seed(fold_seed(seed));
                for _ in 0..entries {
                    inputs.push(self.adjust_input(
                        i64::from(
                            rng.randint(self.input_bounds.0 as u32, self.input_bounds.1 as u32),
                        ),
                    ));
                }
                for _ in 0..entries {
                    outputs.push(i64::from(
                        rng.randint(self.output_bounds.0 as u32, self.output_bounds.1 as u32),
                    ));
                }
                for _ in 0..entries {
                    offsets.push(rng.randint(0, vocab_size - 1) as usize);
                }
            }
        }
        Ok(SeededRandomRangePlan {
            policy: self.clone(),
            inputs,
            outputs,
            offsets,
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

    fn randint(&mut self, low: u32, high: u32) -> u32 {
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
