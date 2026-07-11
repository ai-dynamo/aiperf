// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepEval/trt-llm-aligned legacy AIME 2024 prompt builder.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/aime.py:1-236`.

use serde_json::json;

use super::common::{
    finish_selection, generation, item_id, metadata, optional_string, problem, required_string,
    scalar_string,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

/// Maximum number of reference-compatible few-shot examples.
pub const AIME_MAX_N_SHOTS: usize = 8;
/// Default system prompt used by the inherited plugin registration.
pub const AIME_SYSTEM_PROMPT: &str =
    "Please reason step by step, and put your final answer within \\boxed{}.";

const FEW_SHOT_HEADER: &str = "The following are problems from the American Invitational Mathematics Examination (AIME) 2024. AIME is a prestigious high school mathematics competition known for its challenging mathematical problems.\n\n";
const COT_SUFFIX: &str = "Let's think step-by-step.";
const NO_COT_SUFFIX: &str = "No explanation needed. Just return a number.";
const MAX_TOKENS: usize = 32_768;

#[derive(Debug, Clone)]
struct Example {
    problem: String,
    solution: String,
    answer: String,
}

/// Legacy combined AIME benchmark aligned with the trt-llm DeepEval path.
#[derive(Debug, Clone, Copy, Default)]
pub struct AimeBenchmark;

impl AccuracyBenchmark for AimeBenchmark {
    fn name(&self) -> &'static str {
        "aime"
    }

    fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        if config.n_shots > AIME_MAX_N_SHOTS {
            return Err(AccuracyError::UnsupportedConfiguration(format!(
                "aime accepts at most {AIME_MAX_N_SHOTS} shots, got {}",
                config.n_shots
            )));
        }
        Ok(())
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        self.validate_config(config)?;
        let rows = source.load_rows(DatasetSplit::Train)?;
        let shots = rows
            .iter()
            .take(config.n_shots)
            .enumerate()
            .map(|(index, row)| parse_example(row, index))
            .collect::<Result<Vec<_>, _>>()?;
        let mut problems = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let example = parse_example(row, index)?;
            let prompt = format_prompt(&example.problem, &shots, config.enable_cot);
            problems.push(problem(
                self.name(),
                item_id(row, index, &["ID", "id", "problem_id"]),
                "aime",
                vec![ChatMessage::user(prompt)],
                example.answer,
                generation(config.max_tokens.unwrap_or(MAX_TOKENS), Vec::new()),
                metadata([("generation_size", json!(MAX_TOKENS))]),
            ));
        }
        finish_selection(self.name(), config, problems)
    }
}

fn parse_example(row: &serde_json::Value, index: usize) -> Result<Example, AccuracyError> {
    Ok(Example {
        problem: required_string(row, "Problem", index)?,
        solution: optional_string(row, "Solution"),
        answer: scalar_string(row, "Answer", index)?,
    })
}

fn format_example(example: &Example, enable_cot: bool) -> String {
    let mut block = format!("**Problem**: {}\n", example.problem);
    if enable_cot {
        block.push_str("**Solution**: ");
        block.push_str(&example.solution);
        block.push('\n');
    }
    block.push_str("**Answer**: ");
    block.push_str(&example.answer);
    block
}

fn format_prompt(problem: &str, shots: &[Example], enable_cot: bool) -> String {
    let mut prompt = String::new();
    if !shots.is_empty() {
        prompt.push_str(FEW_SHOT_HEADER);
        for example in shots {
            prompt.push_str(&format_example(example, enable_cot));
            prompt.push_str("\n\n");
        }
    }
    prompt.push_str("**Problem**: ");
    prompt.push_str(problem);
    prompt.push_str("\n**Answer**: \n\n");
    prompt.push_str(if enable_cot {
        COT_SUFFIX
    } else {
        NO_COT_SUFFIX
    });
    prompt
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::InMemoryDatasetSource;

    #[test]
    fn renders_reference_prompt_and_caps_shots() {
        let rows = vec![json!({"Problem":"1+1?","Solution":"Add.","Answer":2})];
        let source = InMemoryDatasetSource::from_splits([(DatasetSplit::Train, rows)]);
        let problems = AimeBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec![],
                    n_shots: 1,
                    enable_cot: true,
                    max_problems: None,
                    max_tokens: None,
                },
            )
            .unwrap();
        let prompt = &problems[0].messages[0].content;
        assert!(prompt.starts_with(FEW_SHOT_HEADER));
        assert!(prompt.contains("**Solution**: Add."));
        assert!(prompt.ends_with(COT_SUFFIX));
    }
}
