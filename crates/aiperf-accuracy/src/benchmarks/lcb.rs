// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LiveCodeBench code-generation prompt and typed execution payload.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/lcb_codegeneration.py:1-293`.

use serde_json::{Value, json};

use super::common::{
    finish_selection, generation, item_id, metadata, optional_string, problem, required_string,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

const PREAMBLE: &str = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n";
const STARTER_INSTRUCTIONS: &str = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n";
const STDIN_INSTRUCTIONS: &str = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n";
const STDIN_SCAFFOLD: &str = "```python\n# YOUR CODE HERE\n```\n\n";
const MAX_TOKENS: usize = 32_768;

/// Native LiveCodeBench code-generation benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct LcbCodeGenerationBenchmark;

impl AccuracyBenchmark for LcbCodeGenerationBenchmark {
    fn name(&self) -> &'static str {
        "lcb-codegeneration"
    }

    fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        if !config.tasks.is_empty() {
            return Err(AccuracyError::UnsupportedConfiguration(
                "lcb-codegeneration does not support task filtering".to_string(),
            ));
        }
        if config.n_shots != 0 {
            return Err(AccuracyError::UnsupportedConfiguration(
                "lcb-codegeneration is reference-defined as zero-shot".to_string(),
            ));
        }
        if config.enable_cot {
            return Err(AccuracyError::UnsupportedConfiguration(
                "lcb-codegeneration has no separate CoT prompt toggle".to_string(),
            ));
        }
        Ok(())
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        self.validate_config(config)?;
        let rows = source.load_rows(DatasetSplit::Test)?;
        let mut problems = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let prompt = prepare_prompt(row, index)?;
            let payload = json!({
                "starter_code": row.get("starter_code").cloned().unwrap_or(Value::String(String::new())),
                "public_test_cases": row.get("public_test_cases").cloned().unwrap_or(Value::String(String::new())),
                "private_test_cases": row.get("private_test_cases").cloned().unwrap_or(Value::String(String::new())),
                "metadata": row.get("metadata").cloned().unwrap_or(Value::String(String::new())),
            });
            problems.push(problem(
                self.name(),
                item_id(row, index, &["question_id", "id"]),
                "lcb_codegeneration",
                vec![ChatMessage::user(prompt)],
                serde_json::to_string(&payload).map_err(|error| AccuracyError::InvalidRow {
                    question_id: None,
                    message: format!("row {index}: serializing LCB test payload: {error}"),
                })?,
                generation(config.max_tokens.unwrap_or(MAX_TOKENS), Vec::new()),
                metadata([
                    ("question_id", json!(optional_string(row, "question_id"))),
                    (
                        "question_title",
                        json!(optional_string(row, "question_title")),
                    ),
                    ("platform", json!(optional_string(row, "platform"))),
                    (
                        "difficulty",
                        json!(optional_string(row, "difficulty").to_ascii_lowercase()),
                    ),
                    ("generation_size", json!(MAX_TOKENS)),
                ]),
            ));
        }
        finish_selection(self.name(), config, problems)
    }
}

fn prepare_prompt(row: &Value, index: usize) -> Result<String, AccuracyError> {
    let content = required_string(row, "question_content", index)?;
    let starter = optional_string(row, "starter_code");
    let mut query = format!("{PREAMBLE}Question: {content}\n\n");
    if starter.is_empty() {
        query.push_str(STDIN_INSTRUCTIONS);
        query.push_str(STDIN_SCAFFOLD);
    } else {
        query.push_str(STARTER_INSTRUCTIONS);
        query.push_str("```python\n");
        query.push_str(&starter);
        query.push_str("\n```\n\n");
    }
    Ok(query)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::prepare_prompt;

    #[test]
    fn starter_and_stdin_scaffolds_match_reference() {
        let starter = prepare_prompt(
            &json!({"question_content":"Solve it","starter_code":"class Solution:"}),
            0,
        )
        .unwrap();
        assert!(starter.ends_with("```python\nclass Solution:\n```\n\n"));
        let stdin = prepare_prompt(&json!({"question_content":"Solve it"}), 0).unwrap();
        assert!(stdin.ends_with("```python\n# YOUR CODE HERE\n```\n\n"));
    }
}
