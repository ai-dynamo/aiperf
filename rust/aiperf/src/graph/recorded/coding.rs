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

use crate::dataset::TextTokenizer;
use crate::rng::PythonRandomGenerator;
use crate::rng::namespace;

use self::templates::{TemplateKind, TemplateRenderer};
use super::RecordedTraceError;

/// Multiplier applied to every category's block count when building the pool
/// (the native analogue of the Python generator's `_pool_scale`). Fractional
/// values scale down — `0.25` gives roughly a quarter-size pool — via truncation,
/// like Python's `int(count * scale)`. At `1.0` the pool is ~270k tokens; the
/// default `1.0` matches agentx's weka/trace path, which constructs
/// `CodingContentGenerator` with no `pool_tokens_target`, so `_pool_scale` clamps
/// to `1.0` (`max(1.0, 10_000_000 / 10_000_000)`). Byte-exact corpus parity
/// requires the SAME scale as agentx, so this is `1.0`, not a larger pool.
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

pub(super) fn build_coding_corpus(
    tokenizer: &dyn TextTokenizer,
    root_seed: u64,
) -> Result<Vec<u32>, RecordedTraceError> {
    build_scaled_corpus(tokenizer, root_seed, POOL_SCALE)
}

/// Build the corpus at an explicit pool scale. `build_coding_corpus` drives this
/// with [`POOL_SCALE`]; tests use a small fractional scale to exercise the
/// seeded-and-reproducible contract (a scale-independent property) without paying
/// for the full >1M-token pool three times over.
fn build_scaled_corpus(
    tokenizer: &dyn TextTokenizer,
    root_seed: u64,
    scale: f64,
) -> Result<Vec<u32>, RecordedTraceError> {
    // Truncating multiply, matching the Python generator's `int(count * scale)`.
    let scaled = |count: usize| (count as f64 * scale) as usize;
    // agentx derives `_template_rng` via `rng.derive("dataset.coding_content.template")`,
    // i.e. sha256(f"{root_seed}:{identifier}")[:8] — NOT the BLAKE3 RngRoot algebra.
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
        .map_err(|error| RecordedTraceError(error.to_string()))
}

#[cfg(test)]
mod tests {
    use crate::dataset::TiktokenTokenizer;

    use super::*;

    #[test]
    #[ignore]
    fn dump_corpus_for_parity() {
        // Temporary parity harness: build the scale-1.0 blocks (pre- and
        // post-shuffle) for root seed 42 and write them beside the Python
        // reference dumps for byte diffing. Run with `--ignored`.
        let template_seed = PythonRandomGenerator::derive_child_seed(
            42,
            namespace::DATASET_CODING_CONTENT_TEMPLATE,
        );
        let mut renderer = TemplateRenderer::new(template_seed);
        let names: &[(TemplateKind, &str)] = &[
            (TemplateKind::Python, "_gen_python_code"),
            (TemplateKind::Go, "_gen_go_code"),
            (TemplateKind::Rust, "_gen_rust_code"),
            (TemplateKind::TypeScript, "_gen_typescript_code"),
            (TemplateKind::MlTraining, "_gen_ml_training_code"),
            (TemplateKind::MlInference, "_gen_ml_inference_code"),
            (TemplateKind::MlConfig, "_gen_ml_config"),
            (TemplateKind::BashOutput, "_gen_bash_output"),
            (TemplateKind::MlTrainingLog, "_gen_ml_training_log"),
            (TemplateKind::JsonResponse, "_gen_json_response"),
            (TemplateKind::ErrorTraceback, "_gen_error_traceback"),
            (TemplateKind::CudaError, "_gen_cuda_error"),
            (TemplateKind::Sql, "_gen_sql_query"),
            (TemplateKind::UserPrompt, "_gen_user_prompt"),
            (TemplateKind::ToolUse, "_gen_tool_use_block"),
            (TemplateKind::Conversation, "_gen_coding_conversation"),
            (TemplateKind::GitDiff, "_gen_git_diff"),
            (TemplateKind::Cicd, "_gen_cicd_output"),
            (TemplateKind::Config, "_gen_config_file"),
            (TemplateKind::Markdown, "_gen_markdown_doc"),
            (TemplateKind::TestOutput, "_gen_test_output"),
        ];
        let mut blocks: Vec<String> = Vec::new();
        let mut first_per: Vec<(String, String)> = Vec::new();
        for &(kind, name) in names {
            let count = TOOL_POOL_BLOCK_COUNTS
                .iter()
                .find(|&&(k, _)| std::mem::discriminant(&k) == std::mem::discriminant(&kind))
                .map(|&(_, c)| c)
                .unwrap();
            let start = blocks.len();
            for ordinal in 0..count {
                blocks.push(renderer.render(kind, ordinal).expect("render"));
            }
            first_per.push((name.to_string(), blocks[start].clone()));
        }
        let _ = first_per;
        std::fs::write(
            "/tmp/rust_blocks.json",
            serde_json::to_string(&blocks).unwrap(),
        )
        .unwrap();
        eprintln!("dumped {} blocks", blocks.len());
    }

    #[test]
    fn coding_corpus_is_seeded_and_reproducible() {
        // Reproducibility and seed-sensitivity are scale-independent, so exercise
        // them at a small fractional scale (~65k tokens) instead of rebuilding the
        // full >1M pool three times.
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
