// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Seeded structural renderer for the native coding corpus.
//!
//! [`TemplateRenderer`] owns the canonical [`RandomGenerator`] stream and the
//! shared draw helpers (`pick`/`number`/`sample`/`title_case`/`index`).
//! [`TemplateKind`] dispatches each category across its structural variants.

use crate::rng::{PythonRandomGenerator, RandomGenerator};

use super::{CodingCorpusError, corpus_error};
use super::{
    cicd_docs, errors_diff, go, json_blocks, ml, prompts_conv, python, rust_lang, sql, tool,
    typescript,
};

#[derive(Clone, Copy, Debug)]
pub(super) enum TemplateKind {
    Python,
    Go,
    Rust,
    TypeScript,
    MlTraining,
    MlInference,
    MlConfig,
    BashOutput,
    MlTrainingLog,
    JsonResponse,
    ErrorTraceback,
    CudaError,
    Sql,
    UserPrompt,
    ToolUse,
    Conversation,
    GitDiff,
    Cicd,
    Config,
    Markdown,
    TestOutput,
}

pub(super) struct TemplateRenderer<R: RandomGenerator = PythonRandomGenerator> {
    random: R,
}

impl TemplateRenderer<PythonRandomGenerator> {
    /// Build a renderer over the CPython-MT and NumPy child stream seed from
    /// `dataset.coding_content.template`.
    pub(super) fn new(seed: u64) -> Self {
        Self {
            random: PythonRandomGenerator::from_child_seed(seed),
        }
    }

    /// Dispatch one category to its structural variant. Lives on the concrete
    /// `PythonRandomGenerator` specialization (not the generic `impl<R>` block
    /// below) because every category module's `render`/helper functions are
    /// themselves written against the concrete, unparameterized
    /// `TemplateRenderer` (`= TemplateRenderer<PythonRandomGenerator>` via its
    /// default type parameter) — they draw exclusively through
    /// [`RandomGenerator`] but are not individually generic over it.
    pub(super) fn render(
        &mut self,
        kind: TemplateKind,
        _ordinal: usize,
    ) -> Result<String, CodingCorpusError> {
        match kind {
            TemplateKind::Python => python::render(self),
            TemplateKind::Go => go::render(self),
            TemplateKind::Rust => rust_lang::render(self),
            TemplateKind::TypeScript => typescript::render(self),
            TemplateKind::MlTraining => ml::training_code(self),
            TemplateKind::MlInference => ml::inference_code(self),
            TemplateKind::MlConfig => ml::config(self),
            TemplateKind::BashOutput => tool::bash_output(self),
            TemplateKind::MlTrainingLog => ml::training_log(self),
            TemplateKind::JsonResponse => json_blocks::render(self),
            TemplateKind::ErrorTraceback => errors_diff::error_traceback(self, None),
            TemplateKind::CudaError => ml::cuda_error(self),
            TemplateKind::Sql => sql::query(self),
            TemplateKind::UserPrompt => prompts_conv::user_prompt(self),
            TemplateKind::ToolUse => tool::tool_use_block(self),
            TemplateKind::Conversation => prompts_conv::coding_conversation(self),
            TemplateKind::GitDiff => errors_diff::git_diff(self, None),
            TemplateKind::Cicd => cicd_docs::cicd_output(self, None),
            TemplateKind::Config => cicd_docs::config_file(self, None),
            TemplateKind::Markdown => cicd_docs::markdown_doc(self),
            TemplateKind::TestOutput => cicd_docs::test_output(self),
        }
    }

    /// Python `f"{x:.{prec}e}"` scientific notation: Rust's `{:e}` omits the
    /// exponent sign and zero-padding, so post-process to CPython's form
    /// (`5.00e-04`, `1.23e+05`) — signed exponent, minimum two digits.
    pub(super) fn py_sci(x: f64, prec: usize) -> String {
        let raw = format!("{x:.prec$e}");
        let (mantissa, exp) = raw.split_once('e').expect("Rust {:e} always emits 'e'");
        let (sign, digits) = match exp.strip_prefix('-') {
            Some(rest) => ('-', rest),
            None => ('+', exp.strip_prefix('+').unwrap_or(exp)),
        };
        format!("{mantissa}e{sign}{digits:0>2}")
    }

    /// Python `str.title()`: uppercase the first letter of each alphabetic run,
    /// lowercase the rest (word boundaries are non-alphabetic characters).
    pub(super) fn title_case(value: &str) -> String {
        let mut output = String::with_capacity(value.len());
        let mut prev_alpha = false;
        for ch in value.chars() {
            if ch.is_alphabetic() {
                if prev_alpha {
                    output.extend(ch.to_lowercase());
                } else {
                    output.extend(ch.to_uppercase());
                }
                prev_alpha = true;
            } else {
                output.push(ch);
                prev_alpha = false;
            }
        }
        output
    }
}

/// Draw primitives generic over any [`RandomGenerator`] implementation — the
/// half of [`TemplateRenderer`] that genuinely doesn't care which backend
/// supplies the stream.
impl<R: RandomGenerator> TemplateRenderer<R> {
    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        self.random.shuffle(values);
    }

    /// Uniformly pick one `&'static str` from a vocabulary slice.
    pub(super) fn pick(
        &mut self,
        values: &'static [&'static str],
    ) -> Result<&'static str, CodingCorpusError> {
        self.random
            .choice(values)
            .copied()
            .map_err(|error| corpus_error(error.to_string()))
    }

    /// `k` distinct picks from a vocabulary slice (Python `random.sample`).
    pub(super) fn sample(
        &mut self,
        values: &'static [&'static str],
        k: usize,
    ) -> Result<Vec<&'static str>, CodingCorpusError> {
        self.random
            .sample(values, k)
            .map_err(|error| corpus_error(error.to_string()))
    }

    /// Uniform float draw in `[0.0, 1.0)` (Python `random.random`).
    pub(super) fn random(&mut self) -> f64 {
        self.random.random()
    }

    /// Uniform float draw in `[low, high)` (Python `random.uniform`):
    /// `low + (high - low) * random()`.
    pub(super) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        self.random.uniform(low, high)
    }

    /// Uniform pick from a runtime (non-`'static`) slice by value (Python
    /// `random.choice(seq)` = `seq[randbelow(len)]`).
    pub(super) fn choose<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T, CodingCorpusError> {
        self.random
            .choice(seq)
            .map_err(|error| corpus_error(error.to_string()))
    }

    /// Inclusive integer draw in `[low, high]` (Python `random.randint`).
    pub(super) fn number(&mut self, low: i64, high: i64) -> Result<i64, CodingCorpusError> {
        self.random
            .randint(low, high)
            .map_err(|error| corpus_error(error.to_string()))
    }

    /// A uniform variant index in `0..count` (Python `choice([fns])`).
    pub(super) fn index(&mut self, count: usize) -> Result<usize, CodingCorpusError> {
        let high = i64::try_from(count.saturating_sub(1))
            .map_err(|_| corpus_error("variant count exceeds i64 range"))?;
        let value = self
            .random
            .randint(0, high)
            .map_err(|error| corpus_error(error.to_string()))?;
        usize::try_from(value).map_err(|_| corpus_error("negative variant index"))
    }
}
