// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zero-shot lighteval-style benchmark ports with bare user prompts.
//!
//! Ported from inherited AIPerf:
//! `src/aiperf/accuracy/benchmarks/{aime24,aime25,math_500,gsm8k}.py`.

use serde_json::json;

use super::common::{
    finish_selection, generation, item_id, metadata, normalized_task, optional_string, problem,
    required_string, scalar_string,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

const REASONING_MAX_TOKENS: usize = 32_768;

/// Lighteval-aligned AIME 2024 (`HuggingFaceH4/aime_2024`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Aime24Benchmark;

/// Lighteval-aligned AIME 2025 (`yentinglin/aime_2025`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Aime25Benchmark;

/// Lighteval-aligned MATH-500 (`HuggingFaceH4/MATH-500`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Math500Benchmark;

/// Lighteval-aligned GSM8K (`openai/gsm8k`, subset `main`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Gsm8kBenchmark;

fn load_aime_year(
    benchmark: &str,
    task: &str,
    source: &dyn DatasetSource,
    config: &BenchmarkConfig,
) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
    let rows = source.load_rows(DatasetSplit::Train)?;
    let mut problems = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let prompt = required_string(row, "problem", index)?;
        let ground_truth = scalar_string(row, "answer", index)?;
        problems.push(problem(
            benchmark,
            item_id(row, index, &["id", "problem_id"]),
            task,
            vec![ChatMessage::user(prompt)],
            ground_truth,
            generation(
                config.max_tokens.unwrap_or(REASONING_MAX_TOKENS),
                Vec::new(),
            ),
            metadata([("generation_size", json!(REASONING_MAX_TOKENS))]),
        ));
    }
    finish_selection(benchmark, config, problems)
}

impl AccuracyBenchmark for Aime24Benchmark {
    fn name(&self) -> &'static str {
        "aime24"
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        load_aime_year(self.name(), "aime24", source, config)
    }
}

impl AccuracyBenchmark for Aime25Benchmark {
    fn name(&self) -> &'static str {
        "aime25"
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        load_aime_year(self.name(), "aime25", source, config)
    }
}

impl AccuracyBenchmark for Math500Benchmark {
    fn name(&self) -> &'static str {
        "math-500"
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        let rows = source.load_rows(DatasetSplit::Test)?;
        let mut problems = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let prompt = required_string(row, "problem", index)?;
            let solution = optional_string(row, "solution");
            let subject = optional_string(row, "subject");
            let subject = if subject.is_empty() {
                "math_500".to_string()
            } else {
                subject
            };
            problems.push(problem(
                self.name(),
                item_id(row, index, &["unique_id", "id", "problem_id"]),
                normalized_task("math_500", &subject),
                vec![ChatMessage::user(prompt)],
                solution,
                generation(
                    config.max_tokens.unwrap_or(REASONING_MAX_TOKENS),
                    Vec::new(),
                ),
                metadata([
                    ("subject", json!(subject)),
                    ("level", row.get("level").cloned().unwrap_or_default()),
                    ("generation_size", json!(REASONING_MAX_TOKENS)),
                ]),
            ));
        }
        finish_selection(self.name(), config, problems)
    }
}

impl AccuracyBenchmark for Gsm8kBenchmark {
    fn name(&self) -> &'static str {
        "gsm8k"
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        let rows = source.load_rows(DatasetSplit::Test)?;
        let mut problems = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let question = required_string(row, "question", index)?;
            let prompt = format!("Question: {question}\nAnswer:");
            problems.push(problem(
                self.name(),
                item_id(row, index, &["id", "question_id"]),
                "gsm8k",
                vec![ChatMessage::user(prompt)],
                required_string(row, "answer", index)?,
                generation(
                    config.max_tokens.unwrap_or(256),
                    vec![
                        "Question=".to_string(),
                        "Question".to_string(),
                        "=".to_string(),
                    ],
                ),
                metadata([("generation_size", json!(256))]),
            ));
        }
        finish_selection(self.name(), config, problems)
    }
}
