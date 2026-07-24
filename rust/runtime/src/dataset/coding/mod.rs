// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared procedural coding corpus generation.
//!
//! Synthetic prompt generation and recorded-trace reconstruction both consume the
//! same seeded coding corpus builder through this dataset-owned helper, so the
//! token pool and its template mix stay byte-identical across both paths.

mod cicd_docs;
mod conversations;
mod conversations_advanced;
mod errors_diff;
mod go;
mod json_blocks;
mod ml;
mod prompts_conv;
mod python;
mod rust_lang;
mod sql;
mod templates;
mod tool;
mod tool_long;
mod typescript;
mod vocab;

use super::tokenizer::TextTokenizer;
use crate::rng::PythonRandomGenerator;
use crate::rng::namespace;

use self::templates::{TemplateKind, TemplateRenderer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodingCorpusError(String);

impl std::fmt::Display for CodingCorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodingCorpusError {}

type CodingCorpusResult<T> = std::result::Result<T, CodingCorpusError>;

pub(crate) fn corpus_error(message: impl Into<String>) -> CodingCorpusError {
    CodingCorpusError(message.into())
}

/// Multiplier applied to every category's block count when building the pool
/// (`_pool_scale`). Fractional values scale down via truncation, mirroring the
/// recorded-trace contract.
const POOL_SCALE: f64 = 1.0;

const TOOL_POOL_BLOCK_COUNTS: &[(TemplateKind, usize)] = &[
    (TemplateKind::Python, 45),
    (TemplateKind::Go, 45),
    (TemplateKind::Rust, 45),
    (TemplateKind::TypeScript, 45),
    (TemplateKind::MlTraining, 30),
    (TemplateKind::MlInference, 25),
    (TemplateKind::MlConfig, 15),
    (TemplateKind::BashOutput, 130),
    (TemplateKind::MlTrainingLog, 20),
    (TemplateKind::JsonResponse, 80),
    (TemplateKind::ErrorTraceback, 45),
    (TemplateKind::CudaError, 20),
    (TemplateKind::Sql, 20),
    (TemplateKind::UserPrompt, 35),
    (TemplateKind::ToolUse, 25),
    (TemplateKind::Conversation, 90),
    (TemplateKind::GitDiff, 15),
    (TemplateKind::Cicd, 15),
    (TemplateKind::Config, 15),
    (TemplateKind::Markdown, 15),
    (TemplateKind::TestOutput, 15),
];

/// Build the shared coding corpus token stream for one tokenizer and seed.
pub(crate) fn build_coding_corpus(
    tokenizer: &dyn TextTokenizer,
    root_seed: u64,
) -> CodingCorpusResult<Vec<u32>> {
    build_scaled_corpus(tokenizer, root_seed, POOL_SCALE)
}

/// Build the corpus at an explicit pool scale. Tests use a fractional scale to
/// exercise seed stability without paying for the full token pool every time.
fn build_scaled_corpus(
    tokenizer: &dyn TextTokenizer,
    root_seed: u64,
    scale: f64,
) -> CodingCorpusResult<Vec<u32>> {
    let scaled = |count: usize| (count as f64 * scale) as usize;
    let template_seed = PythonRandomGenerator::derive_child_seed(
        root_seed,
        namespace::DATASET_CODING_CONTENT_TEMPLATE,
    );
    let mut renderer = TemplateRenderer::new(template_seed);
    let capacity: usize = TOOL_POOL_BLOCK_COUNTS
        .iter()
        .map(|&(_, count)| scaled(count))
        .sum();
    let mut blocks = Vec::with_capacity(capacity);
    for &(kind, count) in TOOL_POOL_BLOCK_COUNTS {
        for ordinal in 0..scaled(count) {
            blocks.push(renderer.render(kind, ordinal)?);
        }
    }
    renderer.shuffle(&mut blocks);
    tokenizer
        .encode(&blocks.join("\n\n"))
        .map_err(|error| corpus_error(error.to_string()))
}

#[cfg(test)]
mod tests {
    use crate::dataset::TiktokenTokenizer;

    use super::*;

    #[test]
    fn coding_corpus_is_seeded_and_reproducible() {
        let tokenizer = TiktokenTokenizer::builtin();
        let first = build_scaled_corpus(&tokenizer, 17, 0.25).expect("first corpus");
        let repeated = build_scaled_corpus(&tokenizer, 17, 0.25).expect("repeated corpus");
        let different = build_scaled_corpus(&tokenizer, 18, 0.25).expect("different corpus");
        assert!(!first.is_empty());
        assert_eq!(
            first, repeated,
            "same seed reproduces the corpus byte-for-byte"
        );
        assert_ne!(
            first, different,
            "a different seed yields a different corpus"
        );
    }

    #[test]
    fn every_template_kind_yields_structural_variety() {
        const KINDS: &[TemplateKind] = &[
            TemplateKind::Python,
            TemplateKind::Go,
            TemplateKind::Rust,
            TemplateKind::TypeScript,
            TemplateKind::MlTraining,
            TemplateKind::MlInference,
            TemplateKind::MlConfig,
            TemplateKind::BashOutput,
            TemplateKind::MlTrainingLog,
            TemplateKind::JsonResponse,
            TemplateKind::ErrorTraceback,
            TemplateKind::CudaError,
            TemplateKind::Sql,
            TemplateKind::UserPrompt,
            TemplateKind::ToolUse,
            TemplateKind::Conversation,
            TemplateKind::GitDiff,
            TemplateKind::Cicd,
            TemplateKind::Config,
            TemplateKind::Markdown,
            TemplateKind::TestOutput,
        ];
        let mut renderer = TemplateRenderer::new(2026);
        for kind in KINDS {
            let mut seen = std::collections::HashSet::new();
            for ordinal in 0..8 {
                seen.insert(renderer.render(*kind, ordinal).expect("render"));
            }
            assert!(
                seen.len() > 1,
                "template kind {kind:?} produced no variety across 8 renders"
            );
        }
    }
}
