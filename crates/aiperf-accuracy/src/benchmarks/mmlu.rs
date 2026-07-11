// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lighteval-aligned 57-subject MMLU prompt builder.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/mmlu.py:1-333`, including
//! balanced few-shot selection with Python-compatible MT19937 sampling.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::common::{
    finish_selection, generation, integer, item_id, metadata, normalized_task, problem,
    required_string, string_array,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

const LETTERS: &[u8] = b"ABCD";

/// The 57 lighteval MMLU subjects.
pub const MMLU_SUBJECTS: &[&str] = &[
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
];

#[derive(Debug, Clone)]
struct Question {
    row_index: usize,
    subject: String,
    question: String,
    choices: Vec<String>,
    answer_index: usize,
    source: Value,
}

/// Native 57-subject MMLU benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmluBenchmark;

impl AccuracyBenchmark for MmluBenchmark {
    fn name(&self) -> &'static str {
        "mmlu"
    }

    fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        resolve_subjects(&config.tasks).map(|_| ())
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        self.validate_config(config)?;
        let selected = resolve_subjects(&config.tasks)?;
        let dev_rows = source
            .load_rows(DatasetSplit::Dev)
            .or_else(|_| source.load_rows(DatasetSplit::Validation))?;
        let test_rows = source.load_rows(DatasetSplit::Test)?;
        let inferred_subject = (selected.len() == 1).then(|| selected[0].as_str());
        let dev = parse_questions(dev_rows, inferred_subject)?;
        let test = parse_questions(test_rows, inferred_subject)?;

        let mut dev_by_subject = BTreeMap::<String, Vec<Question>>::new();
        for question in dev {
            dev_by_subject
                .entry(question.subject.clone())
                .or_default()
                .push(question);
        }

        let mut problems = Vec::new();
        let mut occurrences = BTreeMap::<String, usize>::new();
        for subject in selected {
            let occurrence = occurrences.entry(subject.clone()).or_default();
            for question in test.iter().filter(|question| question.subject == subject) {
                let source_rows = dev_by_subject
                    .get(&question.subject)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let indices = balanced_sample_indices(source_rows, config.n_shots);
                let shots = indices
                    .into_iter()
                    .map(|index| &source_rows[index])
                    .collect::<Vec<_>>();
                let prompt = flat_prompt(question, &shots, config.enable_cot);
                let messages = chat_messages(question, &shots, config.enable_cot);
                let ground_truth = format!(" {}", char::from(LETTERS[question.answer_index]));
                let base_id = item_id(&question.source, question.row_index, &["id", "question_id"]);
                let unique_id = if *occurrence == 0 {
                    base_id
                } else {
                    format!("{base_id}:repeat:{}", *occurrence)
                };
                problems.push(problem(
                    self.name(),
                    unique_id,
                    normalized_task("mmlu", &question.subject),
                    messages,
                    ground_truth,
                    generation(config.max_tokens.unwrap_or(5), vec!["\n".to_string()]),
                    metadata([
                        ("subject", json!(question.subject)),
                        ("generation_size", json!(5)),
                        ("stop_sequence", json!(["\n"])),
                        ("flat_prompt", json!(prompt)),
                    ]),
                ));
            }
            *occurrence += 1;
        }
        finish_selection(self.name(), config, problems)
    }
}

fn resolve_subjects(tasks: &[String]) -> Result<Vec<String>, AccuracyError> {
    if tasks.is_empty() || tasks.iter().any(|task| task == "all") {
        return Ok(MMLU_SUBJECTS
            .iter()
            .map(|value| (*value).to_string())
            .collect());
    }
    let mut selected = Vec::with_capacity(tasks.len());
    for task in tasks {
        if !MMLU_SUBJECTS.contains(&task.as_str()) {
            return Err(AccuracyError::UnknownTask {
                task: task.clone(),
                available: MMLU_SUBJECTS
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
            });
        }
        selected.push(task.clone());
    }
    Ok(selected)
}

fn parse_questions(
    rows: Vec<Value>,
    inferred_subject: Option<&str>,
) -> Result<Vec<Question>, AccuracyError> {
    rows.iter()
        .enumerate()
        .map(|(row_index, row)| {
            let subject = row
                .get("_subject")
                .or_else(|| row.get("subject"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| inferred_subject.map(str::to_string))
                .ok_or_else(|| {
                    super::common::invalid_row(
                        row_index,
                        "MMLU row has no subject discriminator".to_string(),
                    )
                })?;
            if !MMLU_SUBJECTS.contains(&subject.as_str()) {
                return Err(super::common::invalid_row(
                    row_index,
                    format!("unknown MMLU row subject {subject:?}"),
                ));
            }
            let choices = string_array(row, "choices", row_index)?;
            if choices.len() != 4 {
                return Err(super::common::invalid_row(
                    row_index,
                    format!("MMLU expects four choices, found {}", choices.len()),
                ));
            }
            let answer_index = match row.get("answer") {
                Some(Value::String(answer)) if answer.len() == 1 => LETTERS
                    .iter()
                    .position(|letter| *letter == answer.as_bytes()[0])
                    .ok_or_else(|| {
                        super::common::invalid_row(
                            row_index,
                            format!("invalid MMLU answer {answer:?}"),
                        )
                    })?,
                _ => integer(row, "answer", row_index)? as usize,
            };
            if answer_index >= choices.len() {
                return Err(super::common::invalid_row(
                    row_index,
                    format!("MMLU answer index {answer_index} is outside four choices"),
                ));
            }
            Ok(Question {
                row_index,
                subject,
                question: required_string(row, "question", row_index)?,
                choices,
                answer_index,
                source: row.clone(),
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

fn query(question: &Question, enable_cot: bool) -> String {
    let mut output = format!("Question: {}", question.question);
    for (letter, choice) in LETTERS.iter().zip(&question.choices) {
        output.push('\n');
        output.push(char::from(*letter));
        output.push_str(". ");
        output.push_str(choice);
    }
    if enable_cot {
        output.push_str("\nLet's think step by step.\nAnswer:");
    } else {
        output.push_str("\nAnswer:");
    }
    output
}

fn instruction(subject: &str) -> String {
    format!(
        "The following are multiple choice questions (with answers) about {}.\n\n",
        subject.replace('_', " ")
    )
}

fn flat_prompt(question: &Question, shots: &[&Question], enable_cot: bool) -> String {
    let mut output = instruction(&question.subject);
    for shot in shots {
        output.push_str(&query(shot, false));
        output.push(' ');
        output.push(char::from(LETTERS[shot.answer_index]));
        output.push_str("\n\n");
    }
    output.push_str(&query(question, enable_cot));
    output
}

fn chat_messages(question: &Question, shots: &[&Question], enable_cot: bool) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(shots.len() * 2 + 1);
    for (index, shot) in shots.iter().enumerate() {
        let mut content = query(shot, false);
        if index == 0 {
            content.insert_str(0, &instruction(&question.subject));
        }
        messages.push(ChatMessage::user(content));
        messages.push(ChatMessage::assistant(format!(
            " {}",
            char::from(LETTERS[shot.answer_index])
        )));
    }
    let mut content = query(question, enable_cot);
    if shots.is_empty() {
        content.insert_str(0, &instruction(&question.subject));
    }
    messages.push(ChatMessage::user(content));
    messages
}

fn balanced_sample_indices(source: &[Question], n_shots: usize) -> Vec<usize> {
    if n_shots == 0 || source.is_empty() {
        return Vec::new();
    }
    let mut label_to_indices = BTreeMap::<String, Vec<usize>>::new();
    for (index, question) in source.iter().enumerate() {
        label_to_indices
            .entry(question.choices[question.answer_index].clone())
            .or_default()
            .push(index);
    }
    let mut count_to_labels = BTreeMap::<usize, Vec<String>>::new();
    for (label, indices) in &label_to_indices {
        count_to_labels
            .entry(indices.len())
            .or_default()
            .push(label.clone());
    }
    let mut rng = PythonRandom::new(0);
    let mut labels = Vec::new();
    for group in count_to_labels.values_mut().rev() {
        rng.shuffle(group);
        labels.extend(group.iter().cloned());
    }
    let mut output = Vec::new();
    let mut remaining = source.len().min(n_shots.saturating_add(1));
    let mut cursor = 0;
    while remaining > 0 && !labels.is_empty() {
        let label = &labels[cursor % labels.len()];
        cursor += 1;
        let pool = label_to_indices
            .get_mut(label)
            .expect("label came from map");
        if pool.is_empty() {
            if label_to_indices.values().all(Vec::is_empty) {
                break;
            }
            continue;
        }
        let selected = rng.randbelow(pool.len());
        output.push(pool.remove(selected));
        remaining -= 1;
    }
    output.truncate(n_shots);
    output
}

/// Minimal CPython-compatible MT19937 path used by `random.Random(seed)`'s
/// `shuffle`/`randrange` for byte-exact inherited sampling behavior.
pub(super) struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    pub(super) fn new(seed: u32) -> Self {
        let mut rng = Self {
            state: [0; 624],
            index: 624,
        };
        rng.init_by_array(&[seed]);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..624 {
            self.state[index] = 1_812_433_253u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624.max(key.len()) {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let y = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = (self.next_u32() >> (32 - bits)) as usize;
            if value < upper {
                return value;
            }
        }
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = self.randbelow(index + 1);
            values.swap(index, other);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::InMemoryDatasetSource;

    #[test]
    fn python_random_zero_shuffle_matches_cpython() {
        let mut values = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        PythonRandom::new(0).shuffle(&mut values);
        assert_eq!(values, [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]);
    }

    #[test]
    fn prompt_gold_and_requested_subject_order_match_reference() {
        let row = |subject: &str, question: &str, answer: Value| {
            json!({
                "_subject": subject,
                "question": question,
                "choices": ["zero", "one", "two", "three"],
                "answer": answer,
            })
        };
        let source = InMemoryDatasetSource::from_splits([
            (
                DatasetSplit::Dev,
                vec![
                    row("anatomy", "shot anatomy", json!(0)),
                    row("astronomy", "shot astronomy", json!(1)),
                ],
            ),
            (
                DatasetSplit::Test,
                vec![
                    row("anatomy", "test anatomy", json!("B")),
                    row("astronomy", "test astronomy", json!(2)),
                ],
            ),
        ]);
        let problems = MmluBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec![
                        "astronomy".to_string(),
                        "anatomy".to_string(),
                        "anatomy".to_string(),
                    ],
                    n_shots: 1,
                    enable_cot: false,
                    max_problems: None,
                    max_tokens: None,
                },
            )
            .unwrap();
        assert_eq!(
            problems
                .iter()
                .map(|problem| problem.task.as_str())
                .collect::<Vec<_>>(),
            ["mmlu.astronomy", "mmlu.anatomy", "mmlu.anatomy"]
        );
        assert_eq!(problems[0].ground_truth, " C");
        assert_eq!(problems[0].messages[1].content, " B");
        assert!(problems[0].messages[0]
            .content
            .starts_with("The following are multiple choice questions (with answers) about astronomy.\n\nQuestion: shot astronomy"));
        assert_ne!(problems[1].correlation_id, problems[2].correlation_id);
    }

    #[test]
    fn all_selection_keeps_inherited_exact_case_behavior() {
        assert_eq!(
            resolve_subjects(&["all".to_string(), "not_a_subject".to_string()])
                .unwrap()
                .len(),
            MMLU_SUBJECTS.len()
        );
        assert!(resolve_subjects(&["ALL".to_string()]).is_err());
        assert!(resolve_subjects(&[" anatomy".to_string()]).is_err());
    }
}
