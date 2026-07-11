// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepEval-aligned HellaSwag sentence-completion prompts.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/hellaswag.py:1-328` and
//! the prompt shape pinned by `tests/unit/accuracy/test_hellaswag_benchmark.py`.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::common::{
    finish_selection, generation, integer, item_id, metadata, normalized_task, problem,
    required_string, string_array,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

/// DeepEval's default output-confinement instruction.
pub const HELLASWAG_CONFINEMENT: &str = "Output 'A', 'B', 'C', or 'D'. Full answer not needed.";
/// Maximum few-shot count accepted by DeepEval.
pub const HELLASWAG_MAX_N_SHOTS: usize = 15;

#[derive(Debug, Clone)]
struct Row {
    index: usize,
    activity: String,
    context: String,
    endings: Vec<String>,
    label: Option<usize>,
    source: Value,
}

/// Native HellaSwag benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct HellaSwagBenchmark;

impl AccuracyBenchmark for HellaSwagBenchmark {
    fn name(&self) -> &'static str {
        "hellaswag"
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        if config.n_shots > HELLASWAG_MAX_N_SHOTS {
            return Err(AccuracyError::UnsupportedConfiguration(format!(
                "hellaswag accepts at most {HELLASWAG_MAX_N_SHOTS} shots, got {}",
                config.n_shots
            )));
        }
        let train = parse_rows(source.load_rows(DatasetSplit::Train)?)?;
        let validation = parse_rows(source.load_rows(DatasetSplit::Validation)?)?;
        let available = train
            .iter()
            .chain(&validation)
            .map(|row| row.activity.clone())
            .collect::<BTreeSet<_>>();
        let selected = resolve_tasks(&config.tasks, &available)?;

        let mut seen = BTreeSet::new();
        let shots = train
            .iter()
            .filter(|row| seen.insert(row.activity.clone()))
            .take(config.n_shots)
            .collect::<Vec<_>>();

        let mut problems = Vec::new();
        let mut occurrences = BTreeMap::<String, usize>::new();
        for task in selected {
            let occurrence = occurrences.entry(task.clone()).or_default();
            for row in validation.iter().filter(|row| row.activity == task) {
                let Some(label) = row.label else {
                    continue;
                };
                if label >= 4 {
                    return Err(super::common::invalid_row(
                        row.index,
                        format!("HellaSwag label {label} is outside 0..=3"),
                    ));
                }
                let mut prompt = format!(
                    "The following are multiple choice questions (with answers) are sentence completion problems about {}.\n\n",
                    row.activity
                );
                for shot in &shots {
                    let Some(shot_label) = shot.label else {
                        continue;
                    };
                    prompt.push_str(&format_question(shot, Some(shot_label))?);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&format_question(row, None)?);
                prompt.push_str("\n\n");
                prompt.push_str(HELLASWAG_CONFINEMENT);
                let base_id = item_id(&row.source, row.index, &["ind", "id", "question_id"]);
                let unique_id = if *occurrence == 0 {
                    base_id
                } else {
                    format!("{base_id}:repeat:{}", *occurrence)
                };
                problems.push(problem(
                    self.name(),
                    unique_id,
                    normalized_task("hellaswag", &row.activity),
                    vec![ChatMessage::user(prompt)],
                    char::from(b'A' + label as u8).to_string(),
                    generation(config.max_tokens.unwrap_or(5), Vec::new()),
                    metadata([
                        ("activity_label", json!(row.activity)),
                        ("generation_size", json!(5)),
                    ]),
                ));
            }
            *occurrence += 1;
        }
        finish_selection(self.name(), config, problems)
    }
}

fn parse_rows(rows: Vec<Value>) -> Result<Vec<Row>, AccuracyError> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let label = match row.get("label") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) if value.is_empty() => None,
                _ => Some(integer(row, "label", index)? as usize),
            };
            let endings = string_array(row, "endings", index)?;
            if endings.len() != 4 {
                return Err(super::common::invalid_row(
                    index,
                    format!("HellaSwag expects four endings, found {}", endings.len()),
                ));
            }
            Ok(Row {
                index,
                activity: required_string(row, "activity_label", index)?,
                context: required_string(row, "ctx", index)?,
                endings,
                label,
                source: row.clone(),
            })
        })
        .collect()
}

fn resolve_tasks(
    requested: &[String],
    available: &BTreeSet<String>,
) -> Result<Vec<String>, AccuracyError> {
    let by_lower = available
        .iter()
        .map(|task| (task.to_ascii_lowercase(), task))
        .collect::<BTreeMap<_, _>>();
    if requested.is_empty() {
        return Ok(available.iter().cloned().collect());
    }
    let all_count = requested
        .iter()
        .filter(|task| task.eq_ignore_ascii_case("all"))
        .count();
    if all_count > 0 {
        if requested.len() != 1 {
            return Err(AccuracyError::UnsupportedConfiguration(
                "hellaswag task 'all' cannot be mixed with other activity labels".to_string(),
            ));
        }
        return Ok(available.iter().cloned().collect());
    }
    let mut selected = Vec::with_capacity(requested.len());
    for task in requested {
        let direct = by_lower.get(&task.to_ascii_lowercase()).copied();
        let enum_value = task.replace('_', " ").to_ascii_lowercase();
        let resolved = direct.or_else(|| by_lower.get(&enum_value).copied());
        let Some(resolved) = resolved else {
            return Err(AccuracyError::UnknownTask {
                task: task.clone(),
                available: available.iter().cloned().collect(),
            });
        };
        selected.push(resolved.clone());
    }
    Ok(selected)
}

fn format_question(row: &Row, answer: Option<usize>) -> Result<String, AccuracyError> {
    if answer.is_some_and(|answer| answer >= row.endings.len()) {
        return Err(super::common::invalid_row(
            row.index,
            "few-shot answer is outside ending range".to_string(),
        ));
    }
    let mut output = row.context.clone();
    for (index, ending) in row.endings.iter().enumerate() {
        output.push('\n');
        output.push(char::from(b'A' + index as u8));
        output.push_str(". ");
        output.push_str(ending);
    }
    output.push_str("\nAnswer:");
    if let Some(answer) = answer {
        output.push(' ');
        output.push(char::from(b'A' + answer as u8));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::InMemoryDatasetSource;

    #[test]
    fn renders_reference_question_and_unique_label_shots() {
        let row = |activity: &str, context: &str, label: usize| json!({"activity_label":activity,"ctx":context,"endings":["a","b","c","d"],"label":label});
        let source = InMemoryDatasetSource::from_splits([
            (DatasetSplit::Train, vec![row("Sailing", "shot", 1)]),
            (DatasetSplit::Validation, vec![row("Sailing", "query", 2)]),
        ]);
        let problems = HellaSwagBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec!["sailing".to_string()],
                    n_shots: 1,
                    enable_cot: false,
                    max_problems: None,
                    max_tokens: None,
                },
            )
            .unwrap();
        let prompt = &problems[0].messages[0].content;
        assert!(prompt.contains("shot\nA. a\nB. b\nC. c\nD. d\nAnswer: B"));
        assert!(prompt.ends_with(HELLASWAG_CONFINEMENT));
        assert_eq!(problems[0].ground_truth, "C");
    }
}
