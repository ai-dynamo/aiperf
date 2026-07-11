// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BigBench-Hard prompt builder over canonical CoT/shot prompt assets.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/bigbench.py:1-239`. Dataset
//! rows carry a `_task` discriminator and may carry `_cot_prompt` /
//! `_shot_prompt` assets sourced from the official BBH prompt repository.

use serde_json::json;

use super::common::{
    finish_selection, generation, item_id, metadata, normalized_task, optional_string, problem,
    scalar_string,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

/// The 27 BigBench-Hard subtasks.
pub const BIGBENCH_TASKS: &[&str] = &[
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
];

const CONFINEMENTS: &[(&str, &str)] = &[
    (
        "boolean_expressions",
        "\n\nOutput 'True' or 'False'. Full answer not needed.",
    ),
    (
        "causal_judgement",
        "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    ),
    (
        "date_understanding",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    ),
    (
        "disambiguation_qa",
        "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    ),
    (
        "dyck_languages",
        "\n\nOutput only the sequence of parentheses characters separated by white space. Full answer not needed.",
    ),
    (
        "formal_fallacies",
        "\n\nOutput 'invalid' or 'valid'. Full answer not needed.",
    ),
    (
        "geometric_shapes",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', or '(K)'. Full answer not needed.",
    ),
    (
        "hyperbaton",
        "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    ),
    (
        "logical_deduction_five_objects",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    ),
    (
        "logical_deduction_seven_objects",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    ),
    (
        "logical_deduction_three_objects",
        "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    ),
    (
        "movie_recommendation",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    ),
    (
        "multistep_arithmetic_two",
        "\n\nOutput the numerical answer. Full answer not needed.",
    ),
    (
        "navigate",
        "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    ),
    (
        "object_counting",
        "\n\nOutput the numerical answer. Full answer not needed.",
    ),
    (
        "penguins_in_a_table",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    ),
    (
        "reasoning_about_colored_objects",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', or '(R)'. Full answer not needed.",
    ),
    (
        "ruin_names",
        "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    ),
    (
        "salient_translation_error_detection",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    ),
    (
        "snarks",
        "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    ),
    (
        "sports_understanding",
        "\n\nOutput 'yes' or 'no'. Full answer not needed.",
    ),
    (
        "temporal_sequences",
        "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    ),
    (
        "tracking_shuffled_objects_five_objects",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    ),
    (
        "tracking_shuffled_objects_seven_objects",
        "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    ),
    (
        "tracking_shuffled_objects_three_objects",
        "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    ),
    (
        "web_of_lies",
        "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    ),
    (
        "word_sorting",
        "\n\nOutput only the sequence of words separated by white space. Full answer not needed.",
    ),
];

/// Native BigBench-Hard benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct BigBenchBenchmark;

impl AccuracyBenchmark for BigBenchBenchmark {
    fn name(&self) -> &'static str {
        "bigbench"
    }

    fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        if config.n_shots > 3 {
            return Err(AccuracyError::UnsupportedConfiguration(format!(
                "bigbench accepts at most 3 shots, got {}",
                config.n_shots
            )));
        }
        resolve_tasks(&config.tasks).map(|_| ())
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        self.validate_config(config)?;
        let selected = resolve_tasks(&config.tasks)?;
        let infer_single_task = selected.len() == 1;
        let rows = source.load_rows(DatasetSplit::Test)?;
        let mut problems = Vec::new();
        let mut task_occurrences = std::collections::BTreeMap::<String, usize>::new();
        for task in selected {
            let occurrence = task_occurrences.entry(task.clone()).or_default();
            for (index, row) in rows.iter().enumerate() {
                let row_task = row
                    .get("_task")
                    .or_else(|| row.get("task"))
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| infer_single_task.then_some(task.as_str()))
                    .ok_or_else(|| {
                        super::common::invalid_row(
                            index,
                            "BBH row has no _task discriminator".to_string(),
                        )
                    })?;
                if row_task != task {
                    continue;
                }
                let input = row
                    .get("input")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        super::common::invalid_row(
                            index,
                            "field \"input\" must be a string".to_string(),
                        )
                    })?;
                let asset = if config.enable_cot {
                    optional_string(row, "_cot_prompt")
                } else {
                    optional_string(row, "_shot_prompt")
                };
                if asset.is_empty() {
                    return Err(AccuracyError::UnsupportedConfiguration(format!(
                        "bigbench task {task} requires the canonical {} prompt asset",
                        if config.enable_cot { "CoT" } else { "non-CoT" },
                    )));
                }
                let mut prompt = render_prompt_asset(&asset, config.n_shots);
                prompt.push_str("\n\nQ: ");
                prompt.push_str(input);
                prompt.push_str("\nA: ");
                prompt.push_str(confinement(&task));
                let base_id = item_id(row, index, &["id", "question_id"]);
                let unique_id = if *occurrence == 0 {
                    base_id
                } else {
                    format!("{base_id}:repeat:{}", *occurrence)
                };
                problems.push(problem(
                    self.name(),
                    unique_id,
                    normalized_task("bigbench", &task),
                    vec![ChatMessage::user(prompt)],
                    scalar_string(row, "target", index)?,
                    generation(config.max_tokens.unwrap_or(1_024), Vec::new()),
                    metadata([
                        ("bbh_task", json!(task.clone())),
                        ("confinement", json!(confinement(&task))),
                        ("generation_size", json!(1_024)),
                    ]),
                ));
            }
            *occurrence += 1;
        }
        finish_selection(self.name(), config, problems)
    }
}

fn resolve_tasks(tasks: &[String]) -> Result<Vec<String>, AccuracyError> {
    if tasks.is_empty() {
        return Ok(BIGBENCH_TASKS
            .iter()
            .map(|task| (*task).to_string())
            .collect());
    }
    if tasks.iter().any(|task| task.eq_ignore_ascii_case("all")) {
        if tasks.len() != 1 {
            return Err(AccuracyError::UnsupportedConfiguration(
                "bigbench task 'all' cannot be mixed with other subtasks".to_string(),
            ));
        }
        return Ok(BIGBENCH_TASKS
            .iter()
            .map(|task| (*task).to_string())
            .collect());
    }
    let mut selected = Vec::with_capacity(tasks.len());
    for task in tasks {
        let resolved = BIGBENCH_TASKS
            .iter()
            .find(|candidate| **candidate == task)
            .or_else(|| {
                let enum_name = task.to_ascii_uppercase();
                BIGBENCH_TASKS
                    .iter()
                    .find(|candidate| candidate.to_ascii_uppercase() == enum_name)
            });
        let Some(resolved) = resolved else {
            return Err(AccuracyError::UnknownTask {
                task: task.clone(),
                available: BIGBENCH_TASKS
                    .iter()
                    .map(|task| (*task).to_string())
                    .collect(),
            });
        };
        selected.push((*resolved).to_string());
    }
    Ok(selected)
}

fn confinement(task: &str) -> &'static str {
    CONFINEMENTS
        .iter()
        .find_map(|(candidate, value)| (*candidate == task).then_some(*value))
        .unwrap_or("")
}

fn render_prompt_asset(asset: &str, n_shots: usize) -> String {
    format!(
        "Task description: {}",
        asset
            .split("\n\n")
            .take(n_shots + 1)
            .collect::<Vec<_>>()
            .join("\n\n")
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{BigBenchBenchmark, render_prompt_asset};
    use crate::{AccuracyBenchmark, BenchmarkConfig, DatasetSplit, InMemoryDatasetSource};

    fn config(tasks: &[&str], n_shots: usize, enable_cot: bool) -> BenchmarkConfig {
        BenchmarkConfig {
            tasks: tasks.iter().map(|task| (*task).to_string()).collect(),
            n_shots,
            enable_cot,
            max_problems: None,
            max_tokens: None,
        }
    }

    #[test]
    fn truncates_canonical_asset_to_requested_shots() {
        let asset = "Header\n\nQ: one\nA: 1\n\nQ: two\nA: 2\n\nQ: three\nA: 3";
        assert_eq!(
            render_prompt_asset(asset, 2),
            "Task description: Header\n\nQ: one\nA: 1\n\nQ: two\nA: 2"
        );
    }

    #[test]
    fn task_resolution_preserves_reference_validation_and_duplicates() {
        let tasks =
            super::resolve_tasks(&["navigate".to_string(), "NAVIGATE".to_string()]).unwrap();
        assert_eq!(tasks, ["navigate", "navigate"]);
        assert!(super::resolve_tasks(&[" navigate ".to_string()]).is_err());
        assert!(super::resolve_tasks(&["boolean-expressions".to_string()]).is_err());
    }

    #[test]
    fn prompt_is_deepeval_byte_shape_and_metadata_keeps_confinement() {
        let source = InMemoryDatasetSource::from_splits([(
            DatasetSplit::Test,
            vec![json!({
                "_task":"boolean_expressions",
                "input":"True and False is",
                "target":"False",
                "_cot_prompt":"Evaluate expressions.\n\nQ: one\nA: Let's reason. 1\n\nQ: two\nA: Let's reason. 2\n\nQ: three\nA: Let's reason. 3",
                "_shot_prompt":"Evaluate expressions.\n\nQ: one\nA: 1\n\nQ: two\nA: 2\n\nQ: three\nA: 3"
            })],
        )]);
        let cot = BigBenchBenchmark
            .load_problems(&source, &config(&["boolean_expressions"], 2, true))
            .unwrap();
        assert_eq!(
            cot[0].messages[0].content,
            "Task description: Evaluate expressions.\n\nQ: one\nA: Let's reason. 1\n\nQ: two\nA: Let's reason. 2\n\nQ: True and False is\nA: \n\nOutput 'True' or 'False'. Full answer not needed."
        );
        assert_eq!(
            cot[0].metadata["confinement"],
            "\n\nOutput 'True' or 'False'. Full answer not needed."
        );
        let zero = BigBenchBenchmark
            .load_problems(&source, &config(&["boolean_expressions"], 0, false))
            .unwrap();
        assert!(
            zero[0].messages[0]
                .content
                .starts_with("Task description: Evaluate expressions.\n\nQ:")
        );
        assert!(!zero[0].messages[0].content.contains("Q: one"));
    }

    #[test]
    fn requested_task_order_duplicates_and_numeric_targets_are_preserved() {
        let source = InMemoryDatasetSource::from_splits([(
            DatasetSplit::Test,
            vec![
                json!({"_task":"navigate","input":"N","target":"Yes","_cot_prompt":"Nav"}),
                json!({"_task":"object_counting","input":"O","target":42,"_cot_prompt":"Count"}),
            ],
        )]);
        let problems = BigBenchBenchmark
            .load_problems(
                &source,
                &config(&["object_counting", "navigate", "navigate"], 0, true),
            )
            .unwrap();
        assert_eq!(
            problems
                .iter()
                .map(|problem| problem.task.as_str())
                .collect::<Vec<_>>(),
            [
                "bigbench.object_counting",
                "bigbench.navigate",
                "bigbench.navigate"
            ]
        );
        assert_eq!(problems[0].ground_truth, "42");
        assert_ne!(problems[1].correlation_id, problems[2].correlation_id);
    }
}
