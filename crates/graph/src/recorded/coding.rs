// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Procedural coding corpus used by recorded production traces.
//!
//! The source adapter only needs a stable, structurally varied token pool. This
//! native builder preserves the Python generator's category weights while its
//! random choices and final shuffle use AIPerf's canonical BLAKE3-derived PCG64
//! stream. Template expansion remains behind [`templates::TemplateRenderer`],
//! leaving the corpus vocabulary open to additional native renderers.

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

use aiperf_dataset::TextTokenizer;
use aiperf_rng::{RngRoot, namespace};

use self::templates::{TemplateKind, TemplateRenderer};
use super::RecordedTraceError;

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

pub(super) fn build_coding_corpus(
    tokenizer: &dyn TextTokenizer,
    root_seed: u64,
) -> Result<Vec<u32>, RecordedTraceError> {
    let template_seed = RngRoot::new(Some(root_seed))
        .derive_seed(namespace::DATASET_CODING_CONTENT_TEMPLATE)
        .expect("a seeded RNG root always derives a concrete stream seed");
    let mut renderer = TemplateRenderer::new(template_seed);
    let capacity = TOOL_POOL_BLOCK_COUNTS.iter().map(|(_, count)| count).sum();
    let mut blocks = Vec::with_capacity(capacity);
    for &(kind, count) in TOOL_POOL_BLOCK_COUNTS {
        for ordinal in 0..count {
            blocks.push(renderer.render(kind, ordinal)?);
        }
    }
    renderer.shuffle(&mut blocks);
    tokenizer
        .encode(&blocks.join("\n\n"))
        .map_err(|error| RecordedTraceError(error.to_string()))
}

#[cfg(test)]
mod tests {
    use aiperf_dataset::TiktokenTokenizer;

    use super::*;

    #[test]
    fn coding_corpus_is_nonempty_seeded_and_reproducible() {
        let tokenizer = TiktokenTokenizer::builtin();
        let first = build_coding_corpus(&tokenizer, 17).expect("first corpus");
        let repeated = build_coding_corpus(&tokenizer, 17).expect("repeated corpus");
        let different = build_coding_corpus(&tokenizer, 18).expect("different corpus");
        // The full multi-variant generator produces a much larger pool than the
        // single-template baseline (~50k tokens) — every category now fans out
        // across its whole family of structural variants.
        assert!(
            first.len() > 150_000,
            "full corpus should be large, got {} tokens",
            first.len()
        );
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
        use super::templates::{TemplateKind, TemplateRenderer};
        // Every top-level category dispatches across its Python variant family
        // (plus vocabulary fills), so repeated renders of one kind must differ —
        // proving the variant dispatch actually fires rather than one fixed shape.
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
