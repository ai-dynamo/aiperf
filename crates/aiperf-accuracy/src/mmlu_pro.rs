// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MMLU-Pro benchmark loader and official CoT prompt construction.
//!
//! Source parity references:
//! - official `TIGER-AI-Lab/MMLU-Pro/evaluate_from_local.py`: `preprocess`,
//!   `format_cot_example`, and `generate_cot_prompt`;
//! - official `cot_prompt_lib/initial_prompt.txt` for the instruction;
//! - inherited AIPerf `src/aiperf/accuracy/benchmarks/mmlu.py:106-325` for the
//!   transport-neutral `BenchmarkProblem`/preformatted-message shape.

use std::collections::{BTreeMap, BTreeSet};

use aiperf_metrics::{CorrelationId, TaskId};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit, GenerationConfig,
};

/// Official Hugging Face dataset id.
pub const MMLU_PRO_DATASET: &str = "TIGER-Lab/MMLU-Pro";
/// Official leaderboard default number of validation examples per category.
pub const MMLU_PRO_DEFAULT_N_SHOTS: usize = 5;
/// Official local evaluator generation cap.
pub const MMLU_PRO_DEFAULT_MAX_TOKENS: usize = 2_048;
/// Official evaluator instruction from `cot_prompt_lib/initial_prompt.txt`.
pub const MMLU_PRO_INITIAL_PROMPT: &str = "The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.";
const CHOICES: &[u8] = b"ABCDEFGHIJ";

/// The 14 official MMLU-Pro categories.
pub const MMLU_PRO_CATEGORIES: &[&str] = &[
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
];

/// One row in the official MMLU-Pro dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmluProQuestion {
    /// Stable dataset id.
    pub question_id: u64,
    /// Question text.
    pub question: String,
    /// Up to ten answer choices; official preprocessing removes `N/A` sentinels.
    pub options: Vec<String>,
    /// Correct letter.
    pub answer: String,
    /// Correct option index.
    pub answer_index: usize,
    /// Reference chain-of-thought used only for validation/few-shot examples.
    pub cot_content: String,
    /// One of the 14 categories.
    pub category: String,
    /// Original dataset/source label.
    pub src: String,
}

/// Native MMLU-Pro benchmark implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmluProBenchmark;

impl AccuracyBenchmark for MmluProBenchmark {
    fn name(&self) -> &'static str {
        "mmlu-pro"
    }

    fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        resolve_tasks(&config.tasks).map(|_| ())
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        self.validate_config(config)?;
        let validation = parse_questions(source.load_rows(DatasetSplit::Validation)?)?;
        let test = parse_questions(source.load_rows(DatasetSplit::Test)?)?;
        let selected = resolve_tasks(&config.tasks)?;

        let mut validation_by_category = BTreeMap::<String, Vec<MmluProQuestion>>::new();
        for question in validation {
            validation_by_category
                .entry(normalize_task(&question.category))
                .or_default()
                .push(question);
        }

        let mut problems = Vec::new();
        for question in test {
            let category = normalize_task(&question.category);
            if !selected.contains(&category) {
                continue;
            }
            let shots = validation_by_category
                .get(&category)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let prompt = format_prompt(
                &question,
                &shots[..shots.len().min(config.n_shots)],
                config.enable_cot,
            );
            let task = format!("mmlu_pro.{}", category.replace(' ', "_"));
            let id = format!("mmlu-pro:{}", question.question_id);
            let mut metadata = BTreeMap::new();
            metadata.insert("question_id".to_string(), json!(question.question_id));
            metadata.insert("category".to_string(), json!(category));
            metadata.insert("source".to_string(), json!(question.src));
            metadata.insert("answer_index".to_string(), json!(question.answer_index));
            metadata.insert("option_count".to_string(), json!(question.options.len()));
            problems.push(BenchmarkProblem {
                correlation_id: CorrelationId::new(id.clone()),
                id,
                task: TaskId::new(task),
                messages: vec![ChatMessage::user(prompt)],
                ground_truth: question.answer,
                generation: GenerationConfig {
                    max_tokens: config.max_tokens.unwrap_or(MMLU_PRO_DEFAULT_MAX_TOKENS),
                    temperature: 0.0,
                    top_p: 1.0,
                    stop: vec!["Question:".to_string()],
                },
                metadata,
            });
            if config
                .max_problems
                .is_some_and(|limit| problems.len() >= limit)
            {
                break;
            }
        }
        if problems.is_empty() {
            return Err(AccuracyError::EmptySelection(format!(
                "benchmark=mmlu-pro tasks={:?}",
                config.tasks
            )));
        }
        Ok(problems)
    }
}

fn parse_questions(rows: Vec<Value>) -> Result<Vec<MmluProQuestion>, AccuracyError> {
    rows.into_iter()
        .map(|row| {
            let question_id = row.get("question_id").and_then(Value::as_u64);
            let mut question = serde_json::from_value::<MmluProQuestion>(row).map_err(|error| {
                AccuracyError::InvalidRow {
                    question_id,
                    message: error.to_string(),
                }
            })?;
            question.options.retain(|option| option != "N/A");
            validate_question(&question)?;
            Ok(question)
        })
        .collect()
}

fn validate_question(question: &MmluProQuestion) -> Result<(), AccuracyError> {
    let invalid = |message: String| AccuracyError::InvalidRow {
        question_id: Some(question.question_id),
        message,
    };
    if question.question.trim().is_empty() {
        return Err(invalid("question is empty".to_string()));
    }
    if question.options.len() < 2 || question.options.len() > CHOICES.len() {
        return Err(invalid(format!(
            "expected 2..=10 non-N/A options, found {}",
            question.options.len()
        )));
    }
    if question
        .options
        .iter()
        .any(|option| option.trim().is_empty())
    {
        return Err(invalid("an option is empty".to_string()));
    }
    if question.answer_index >= question.options.len() {
        return Err(invalid(format!(
            "answer_index {} is outside {} options",
            question.answer_index,
            question.options.len()
        )));
    }
    let expected = char::from(CHOICES[question.answer_index]).to_string();
    if question.answer != expected {
        return Err(invalid(format!(
            "answer {:?} disagrees with answer_index {} ({expected})",
            question.answer, question.answer_index
        )));
    }
    let category = normalize_task(&question.category);
    if !MMLU_PRO_CATEGORIES.contains(&category.as_str()) {
        return Err(invalid(format!("unknown category {:?}", question.category)));
    }
    Ok(())
}

fn resolve_tasks(tasks: &[String]) -> Result<BTreeSet<String>, AccuracyError> {
    if tasks.is_empty() || tasks.iter().any(|task| task.eq_ignore_ascii_case("all")) {
        return Ok(MMLU_PRO_CATEGORIES
            .iter()
            .map(|category| (*category).to_string())
            .collect());
    }
    let mut selected = BTreeSet::new();
    for task in tasks {
        let task = normalize_task(task);
        if !MMLU_PRO_CATEGORIES.contains(&task.as_str()) {
            return Err(AccuracyError::UnknownTask {
                task,
                available: MMLU_PRO_CATEGORIES
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
            });
        }
        selected.insert(task);
    }
    Ok(selected)
}

fn normalize_task(task: &str) -> String {
    task.trim()
        .to_ascii_lowercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_prompt(
    question: &MmluProQuestion,
    few_shots: &[MmluProQuestion],
    enable_cot: bool,
) -> String {
    let mut prompt = MMLU_PRO_INITIAL_PROMPT.replace("{$}", &normalize_task(&question.category));
    prompt.push('\n');
    for example in few_shots {
        prompt.push_str(&format_example(example, true, enable_cot));
    }
    prompt.push_str(&format_example(question, false, enable_cot));
    prompt
}

fn format_example(question: &MmluProQuestion, include_answer: bool, enable_cot: bool) -> String {
    let mut prompt = format!("Question:\n{}\nOptions:\n", question.question);
    for (index, option) in question.options.iter().enumerate() {
        prompt.push(char::from(CHOICES[index]));
        prompt.push_str(". ");
        prompt.push_str(option);
        prompt.push('\n');
    }
    if include_answer {
        if enable_cot {
            let cot = question.cot_content.replace(
                "A: Let's think step by step.",
                "Answer: Let's think step by step.",
            );
            prompt.push_str(&cot);
        } else {
            prompt.push_str("Answer: ");
            prompt.push_str(&question.answer);
        }
        prompt.push_str("\n\n");
    } else if enable_cot {
        prompt.push_str("Answer: Let's think step by step.");
    } else {
        prompt.push_str("Answer:");
    }
    prompt
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        AccuracyBenchmark, BenchmarkConfig, InMemoryDatasetSource, MMLU_PRO_INITIAL_PROMPT,
        MmluProBenchmark,
    };

    fn row(id: u64, category: &str, answer_index: usize, cot: &str) -> serde_json::Value {
        json!({
            "question_id": id,
            "question": format!("Question {id}?"),
            "options": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
            "answer": char::from(b'A' + answer_index as u8).to_string(),
            "answer_index": answer_index,
            "cot_content": cot,
            "category": category,
            "src": "fixture"
        })
    }

    #[test]
    fn builds_official_five_shot_cot_prompt_and_typed_problem() {
        let validation = (0..5)
            .map(|index| {
                row(
                    index,
                    "computer science",
                    index as usize,
                    &format!(
                        "A: Let's think step by step. rationale {index}. The answer is ({})",
                        char::from(b'A' + index as u8)
                    ),
                )
            })
            .collect();
        let source = InMemoryDatasetSource::new(
            validation,
            vec![row(99, "computer_science", 9, "unused test rationale")],
        );
        let problems = MmluProBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec!["computer_science".to_string()],
                    n_shots: 5,
                    enable_cot: true,
                    max_problems: None,
                    max_tokens: None,
                },
            )
            .unwrap();
        assert_eq!(problems.len(), 1);
        let problem = &problems[0];
        assert_eq!(problem.id, "mmlu-pro:99");
        assert_eq!(problem.correlation_id.as_str(), "mmlu-pro:99");
        assert_eq!(problem.task.as_str(), "mmlu_pro.computer_science");
        assert_eq!(problem.ground_truth, "J");
        assert_eq!(problem.generation.max_tokens, 2_048);
        assert_eq!(problem.generation.stop, ["Question:"]);
        let prompt = &problem.messages[0].content;
        assert!(prompt.starts_with(&MMLU_PRO_INITIAL_PROMPT.replace("{$}", "computer science")));
        assert_eq!(prompt.matches("Question:\n").count(), 6);
        assert!(prompt.ends_with("Answer: Let's think step by step."));
        assert!(!prompt.contains("A: Let's think step by step."));
    }

    #[test]
    fn removes_na_sentinels_and_rejects_answer_index_disagreement() {
        let mut valid = row(1, "math", 1, "A: rationale");
        valid["options"] = json!(["one", "two", "N/A"]);
        let source = InMemoryDatasetSource::new(vec![], vec![valid]);
        let problems = MmluProBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec![],
                    n_shots: 0,
                    enable_cot: false,
                    max_problems: None,
                    max_tokens: Some(7),
                },
            )
            .unwrap();
        assert_eq!(problems[0].metadata["option_count"], json!(2));
        assert_eq!(problems[0].generation.max_tokens, 7);

        let bad = row(2, "math", 1, "x");
        let mut bad = bad;
        bad["answer"] = json!("A");
        let source = InMemoryDatasetSource::new(vec![], vec![bad]);
        let error = MmluProBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec![],
                    n_shots: 0,
                    enable_cot: false,
                    max_problems: None,
                    max_tokens: None,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("disagrees with answer_index"));
    }
}
