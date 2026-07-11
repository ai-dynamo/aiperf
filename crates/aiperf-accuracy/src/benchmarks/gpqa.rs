// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPQA-Diamond simple-evals prompt and deterministic answer shuffling.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/gpqa_diamond.py:1-219`.

use serde_json::json;
use sha2::{Digest, Sha256};

use super::common::{
    finish_selection, generation, item_id, metadata, normalized_task, optional_string, problem,
    required_string,
};
use super::mmlu::PythonRandom;
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

const PROMPT_TEMPLATE_PREFIX: &str = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n";
const MAX_TOKENS: usize = 32_768;

/// Native GPQA-Diamond benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpqaDiamondBenchmark;

impl AccuracyBenchmark for GpqaDiamondBenchmark {
    fn name(&self) -> &'static str {
        "gpqa-diamond"
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        let rows = source.load_rows(DatasetSplit::Train)?;
        let mut problems = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let question = required_string(row, "Question", index)?;
            let raw = [
                required_string(row, "Correct Answer", index)?,
                required_string(row, "Incorrect Answer 1", index)?,
                required_string(row, "Incorrect Answer 2", index)?,
                required_string(row, "Incorrect Answer 3", index)?,
            ];
            let order = seeded_shuffle_indices(&question);
            let choices = order.iter().map(|index| &raw[*index]).collect::<Vec<_>>();
            let gold_index = order
                .iter()
                .position(|index| *index == 0)
                .expect("permutation contains correct answer");
            let gold = char::from(b'A' + gold_index as u8).to_string();
            let prompt = format!(
                "{PROMPT_TEMPLATE_PREFIX}{question}\n\nA) {}\nB) {}\nC) {}\nD) {}",
                choices[0], choices[1], choices[2], choices[3]
            );
            let domain = optional_string(row, "High-level domain");
            let task = if domain.is_empty() {
                "gpqa_diamond".to_string()
            } else {
                normalized_task("gpqa_diamond", &domain)
            };
            problems.push(problem(
                self.name(),
                item_id(row, index, &["Record ID", "id", "question_id"]),
                task,
                vec![ChatMessage::user(prompt)],
                gold,
                generation(config.max_tokens.unwrap_or(MAX_TOKENS), Vec::new()),
                metadata([
                    ("domain", json!(domain)),
                    ("subdomain", json!(optional_string(row, "Subdomain"))),
                    ("generation_size", json!(MAX_TOKENS)),
                ]),
            ));
        }
        finish_selection(self.name(), config, problems)
    }
}

fn seeded_shuffle_indices(question: &str) -> [usize; 4] {
    let digest = Sha256::digest(question.as_bytes());
    let seed = u32::from_be_bytes(digest[28..32].try_into().expect("four-byte slice"));
    let mut indices = [0, 1, 2, 3];
    PythonRandom::new(seed).shuffle(&mut indices);
    indices
}

#[cfg(test)]
mod tests {
    use super::seeded_shuffle_indices;

    #[test]
    fn shuffle_matches_inherited_python_vectors() {
        assert_eq!(seeded_shuffle_indices("hello"), [3, 0, 1, 2]);
        assert_eq!(seeded_shuffle_indices("alpha"), [2, 0, 1, 3]);
        assert_eq!(seeded_shuffle_indices("beta"), [2, 1, 0, 3]);
    }
}
