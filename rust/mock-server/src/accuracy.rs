// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy-dataset response mode.
//!
//! AIPerf deliberately never sends ground truth to any inference server — it
//! lives only in the Python accuracy worker
//! (`src/aiperf/accuracy/worker.py`) and the Rust evaluator protocol strips it
//! (`aiperf::accuracy_core::protocol`). So to make the mock return *correct*
//! answers it must load the dataset itself and key on the request prompt. This
//! module owns that: it loads a JSONL `{prompt, ground_truth}` dataset, decides
//! (deterministically, from a per-prompt seed) whether a given request gets the
//! correct answer or a plausible wrong one, formats the answer to satisfy the
//! benchmark's real grader, and optionally renders chain-of-thought or one of
//! the parser-choking adversarial shapes drawn from real bug reports.
//!
//! ## Answer formats — must match `src/aiperf/accuracy/graders/`
//! - [`AccuracyFormat::Mmlu`]/[`AccuracyFormat::MmluPro`]: `multiple_choice.py`
//!   / `mmlu_pro.py` — `The answer is (B)` (clean tier-1) or a bare first-line
//!   letter.
//! - [`AccuracyFormat::Gsm8k`]: `gsm8k_grader.py` — `#### 42`.
//! - [`AccuracyFormat::Math`]: `math.py` — `\boxed{42}`.
//! - [`AccuracyFormat::ExactMatch`]: `exact_match.py` — strict, case-sensitive
//!   `pred.strip() == gold.strip()`; any prefix or case change fails.
//! - [`AccuracyFormat::Passthrough`]: gold verbatim.
//!
//! ## Adversarial catalog — reproduce real parser chokes
//! - [`Adversarial::ReasoningOnly`] — answer only in `reasoning_content`, empty
//!   `content` (github #1136: `get_text()` concatenates reasoning+content and
//!   the strict grader sees the whole CoT blob → 0%).
//! - [`Adversarial::LeadingWhitespace`] — `"\n\nThe answer is (B)"` (real
//!   content was `"\n\nTrue"`).
//! - [`Adversarial::WrongCase`] — lowercased vs a case-sensitive grader.
//! - [`Adversarial::TrailingProse`] — answer then hedging prose.
//! - [`Adversarial::BoxedWrap`] — `\boxed{}` wrap chokes exact-match.
//! - [`Adversarial::MultipleConflicting`] — two answers (tests take-LAST-match).
//! - [`Adversarial::Unicode`] — unicode suffix (SSE UTF-8 line-buffer path).
//! - [`Adversarial::NullObjectChunk`] — streaming-only final SSE frame with
//!   `"object": null` before `[DONE]` (github #1010: crashed
//!   `extract_chat_response_data`). The run must SURVIVE this, not crash.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use aiperf::rng::{RandomGenerator, derive_seed_parts};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Wrapping applied to a clean gold answer so the corresponding real grader
/// parses it. Selected globally via `--accuracy-format` or per dataset entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "snake_case")]
pub enum AccuracyFormat {
    /// Gold string emitted verbatim; matches any grader that compares raw text.
    #[default]
    Passthrough,
    /// MMLU multiple-choice (A–D): `The answer is (B)` / bare `B`.
    Mmlu,
    /// MMLU-Pro multiple-choice (A–J): `The answer is (B)` / bare `B`.
    MmluPro,
    /// GSM8K: `#### 42`.
    Gsm8k,
    /// AIME/MATH: `\boxed{42}`.
    Math,
    /// HellaSwag/BigBench strict exact match (case-sensitive).
    ExactMatch,
}

impl AccuracyFormat {
    /// Does this grader tolerate reasoning text *before* the answer in the same
    /// content field? Multiple-choice, GSM8K, and math extract with a cascade
    /// that skips leading prose; exact-match and passthrough compare the whole
    /// stripped string, so an inline CoT prefix would break correctness.
    fn tolerates_inline_prefix(self) -> bool {
        matches!(self, Self::Mmlu | Self::MmluPro | Self::Gsm8k | Self::Math)
    }
}

/// The adversarial response shapes, one per known parser failure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adversarial {
    LeadingWhitespace,
    TrailingProse,
    WrongCase,
    ReasoningOnly,
    BoxedWrap,
    MultipleConflicting,
    Unicode,
    NullObjectChunk,
}

impl Adversarial {
    const ALL: [Adversarial; 8] = [
        Adversarial::LeadingWhitespace,
        Adversarial::TrailingProse,
        Adversarial::WrongCase,
        Adversarial::ReasoningOnly,
        Adversarial::BoxedWrap,
        Adversarial::MultipleConflicting,
        Adversarial::Unicode,
        Adversarial::NullObjectChunk,
    ];
}

/// One dataset row: a prompt and its gold answer, plus optional per-row grader
/// format and multiple-choice option set.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Whitespace-normalized prompt, used as the lookup key.
    pub prompt_norm: String,
    /// The clean gold answer (`B`, `42`, a latex string, …).
    pub gold: String,
    /// Reporting task/subject, if present.
    pub task: Option<String>,
    /// Per-row format override (falls back to the dataset default).
    pub format: Option<AccuracyFormat>,
    /// Multiple-choice option letters, used to pick a plausible wrong answer.
    pub choices: Vec<String>,
}

/// The parsed dataset plus the seeded-decision knobs from config.
pub struct AccuracyDataset {
    exact: HashMap<String, Entry>,
    /// Entries sorted by descending prompt length, for substring fallback when
    /// a request wraps the dataset prompt (few-shot / system-prompt prefixes).
    entries: Vec<Entry>,
    default_format: AccuracyFormat,
    correct_rate: f64,
    cot_rate: f64,
    adversarial_rate: f64,
    reasoning_field: bool,
    seed: u64,
}

/// The rendered response for one matched request.
#[derive(Debug, Clone)]
pub struct AccuracyDecision {
    /// Assistant `content` string (may be empty for `ReasoningOnly`).
    pub content: String,
    /// Assistant `reasoning_content`, when CoT is rendered in the separate field.
    pub reasoning_content: Option<String>,
    /// When true, the streaming path emits one `{"object": null}` SSE frame
    /// before `[DONE]` (github #1010).
    pub null_object_chunk: bool,
    /// Whether this request was decided correct (for tests/introspection).
    pub correct: bool,
    /// Whether CoT was rendered.
    pub cot: bool,
    /// The adversarial variant applied, if any.
    pub adversarial: Option<Adversarial>,
}

/// Collapse all runs of whitespace to single spaces and trim, so lookup is
/// robust to formatting differences between the dataset and the wire prompt.
fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Read a string field trying each alias in order.
fn field<'a>(obj: &'a serde_json::Map<String, Value>, aliases: &[&str]) -> Option<&'a Value> {
    aliases.iter().find_map(|k| obj.get(*k))
}

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

impl AccuracyDataset {
    /// Load from a JSONL file. Each line is an object; `prompt`/`question`/
    /// `input`/`text` supplies the prompt key and `ground_truth`/`answer`/
    /// `gold`/`target` supplies the gold. Rows lacking either are skipped.
    pub fn load(path: &Path, cfg: &crate::config::MockServerConfig) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("accuracy dataset {}: {e}", path.display()))?;
        Self::from_jsonl(&text, cfg)
    }

    /// Parse the JSONL body directly (used by [`Self::load`] and tests).
    pub fn from_jsonl(body: &str, cfg: &crate::config::MockServerConfig) -> Result<Self, String> {
        let mut exact: HashMap<String, Entry> = HashMap::new();
        for (lineno, line) in body.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)
                .map_err(|e| format!("accuracy dataset line {}: {e}", lineno + 1))?;
            let Some(obj) = value.as_object() else {
                continue;
            };
            let Some(prompt) =
                field(obj, &["prompt", "question", "input", "text"]).and_then(value_to_string)
            else {
                continue;
            };
            let Some(gold) =
                field(obj, &["ground_truth", "answer", "gold", "target"]).and_then(value_to_string)
            else {
                continue;
            };
            let task = field(obj, &["task", "subject", "category"]).and_then(value_to_string);
            let format = obj
                .get("format")
                .or_else(|| obj.get("benchmark"))
                .and_then(Value::as_str)
                .and_then(parse_format);
            let choices = obj
                .get("choices")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(value_to_string).collect())
                .unwrap_or_default();
            let prompt_norm = normalize(&prompt);
            exact.insert(
                prompt_norm.clone(),
                Entry {
                    prompt_norm,
                    gold,
                    task,
                    format,
                    choices,
                },
            );
        }
        if exact.is_empty() {
            return Err("accuracy dataset has no usable rows (need a prompt field \
                 [prompt/question/input/text] and a gold field \
                 [ground_truth/answer/gold/target])"
                .to_string());
        }
        let mut entries: Vec<Entry> = exact.values().cloned().collect();
        // Longest prompt first so the most specific substring match wins.
        entries.sort_by_key(|e| std::cmp::Reverse(e.prompt_norm.len()));
        Ok(Self {
            exact,
            entries,
            default_format: cfg.accuracy_format,
            correct_rate: cfg.accuracy_correct_rate.clamp(0.0, 1.0),
            cot_rate: cfg.accuracy_cot_rate.clamp(0.0, 1.0),
            adversarial_rate: cfg.accuracy_adversarial_rate.clamp(0.0, 1.0),
            reasoning_field: cfg.accuracy_reasoning_field,
            seed: cfg.random_seed.unwrap_or(0),
        })
    }

    /// Number of loaded rows.
    pub fn len(&self) -> usize {
        self.exact.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exact.is_empty()
    }

    /// Find the entry whose prompt matches the request's user text: exact
    /// normalized match first, then the longest entry prompt contained in the
    /// request (handles few-shot / system-prompt wrapping).
    pub fn lookup(&self, request_text: &str) -> Option<&Entry> {
        let norm = normalize(request_text);
        if let Some(e) = self.exact.get(&norm) {
            return Some(e);
        }
        self.entries
            .iter()
            .find(|e| !e.prompt_norm.is_empty() && norm.contains(&e.prompt_norm))
    }

    /// Render the response for a matched request. Deterministic in
    /// `(random_seed, request_text)` — independent of arrival order — so a
    /// given prompt always gets the same verdict across a run and across runs.
    pub fn decide(&self, entry: &Entry, request_text: &str) -> AccuracyDecision {
        let seed = derive_seed_parts(&[
            self.seed.to_le_bytes().as_slice(),
            request_text.as_bytes(),
            "mock.accuracy".as_bytes(),
        ]);
        let mut rng = RandomGenerator::from_seed(Some(seed));
        // Fixed draw order — do NOT reorder, it defines the seeded stream.
        let correct = rng.random() < self.correct_rate;
        let cot = rng.random() < self.cot_rate;
        let adversarial_on = rng.random() < self.adversarial_rate;

        let fmt = entry.format.unwrap_or(self.default_format);
        let answer = if correct {
            format_correct(fmt, &entry.gold)
        } else {
            format_wrong(fmt, entry, &mut rng)
        };
        let reasoning_prose = if cot {
            Some(generate_cot(&mut rng, &answer))
        } else {
            None
        };

        // Compose base (content, reasoning_content) honoring CoT placement.
        let (mut content, mut reasoning_content) = match &reasoning_prose {
            None => (answer.clone(), None),
            Some(prose) => {
                if self.reasoning_field {
                    (answer.clone(), Some(prose.clone()))
                } else if fmt.tolerates_inline_prefix() {
                    (format!("{prose}\n{answer}"), None)
                } else {
                    // Exact-match/passthrough can't take an inline prefix
                    // without breaking; keep content clean, reasoning aside.
                    (answer.clone(), Some(prose.clone()))
                }
            }
        };

        let mut null_object_chunk = false;
        let adversarial = if adversarial_on {
            let variant = *rng.choice(&Adversarial::ALL).expect("non-empty catalog");
            apply_adversarial(
                variant,
                &answer,
                reasoning_prose.as_deref(),
                &mut content,
                &mut reasoning_content,
                &mut null_object_chunk,
            );
            Some(variant)
        } else {
            None
        };

        AccuracyDecision {
            content,
            reasoning_content,
            null_object_chunk,
            correct,
            cot,
            adversarial,
        }
    }
}

fn parse_format(s: &str) -> Option<AccuracyFormat> {
    match s.to_ascii_lowercase().replace('-', "_").as_str() {
        "passthrough" => Some(AccuracyFormat::Passthrough),
        "mmlu" => Some(AccuracyFormat::Mmlu),
        "mmlu_pro" => Some(AccuracyFormat::MmluPro),
        "gsm8k" => Some(AccuracyFormat::Gsm8k),
        "math" | "aime" => Some(AccuracyFormat::Math),
        "exact_match" | "exact" | "hellaswag" | "bigbench" => Some(AccuracyFormat::ExactMatch),
        _ => None,
    }
}

/// Format the correct answer in the grader's clean form.
fn format_correct(fmt: AccuracyFormat, gold: &str) -> String {
    let g = gold.trim();
    match fmt {
        AccuracyFormat::Passthrough | AccuracyFormat::ExactMatch => g.to_string(),
        AccuracyFormat::Mmlu | AccuracyFormat::MmluPro => format!("The answer is ({g})"),
        AccuracyFormat::Gsm8k => format!("#### {g}"),
        AccuracyFormat::Math => format!("\\boxed{{{g}}}"),
    }
}

/// Format a plausible *wrong* answer that the grader will score incorrect.
fn format_wrong(fmt: AccuracyFormat, entry: &Entry, rng: &mut RandomGenerator) -> String {
    let g = entry.gold.trim();
    match fmt {
        AccuracyFormat::Mmlu | AccuracyFormat::MmluPro => {
            let letter = wrong_letter(fmt, entry, rng);
            format!("The answer is ({letter})")
        }
        AccuracyFormat::Gsm8k => format!("#### {}", bump_number(g)),
        AccuracyFormat::Math => format!("\\boxed{{{}}}", bump_number(g)),
        AccuracyFormat::Passthrough | AccuracyFormat::ExactMatch => format!("{g}_wrong"),
    }
}

/// Pick a choice letter different from the gold letter.
fn wrong_letter(fmt: AccuracyFormat, entry: &Entry, rng: &mut RandomGenerator) -> String {
    let gold = entry.gold.trim().to_ascii_uppercase();
    let pool: Vec<String> = if !entry.choices.is_empty() {
        entry.choices.iter().map(|c| c.trim().to_string()).collect()
    } else {
        let range: &[char] = if matches!(fmt, AccuracyFormat::MmluPro) {
            &['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        } else {
            &['A', 'B', 'C', 'D']
        };
        range.iter().map(|c| c.to_string()).collect()
    };
    let alternatives: Vec<&String> = pool
        .iter()
        .filter(|c| c.to_ascii_uppercase() != gold)
        .collect();
    if alternatives.is_empty() {
        // Degenerate single-option set: shift the letter deterministically.
        let next = gold
            .chars()
            .next()
            .map(|c| ((c as u8) + 1) as char)
            .unwrap_or('Z');
        return next.to_string();
    }
    (*rng.choice(&alternatives).expect("non-empty alternatives")).clone()
}

/// Return a numeric string one greater than `s`, or a guaranteed-different
/// string when `s` isn't numeric.
fn bump_number(s: &str) -> String {
    let cleaned = s.replace(',', "");
    if let Ok(n) = cleaned.parse::<i64>() {
        (n + 1).to_string()
    } else if let Ok(f) = cleaned.parse::<f64>() {
        (f + 1.0).to_string()
    } else {
        format!("{s}1")
    }
}

/// A short, deterministic chain-of-thought whose last line points at `answer`.
fn generate_cot(rng: &mut RandomGenerator, answer: &str) -> String {
    let openers = [
        "Let me work through this step by step.",
        "First, let me analyze the problem carefully.",
        "Breaking this down into parts.",
        "I'll reason about each option in turn.",
    ];
    let middles = [
        "Considering the constraints, one path stands out.",
        "Eliminating the implausible cases narrows it down.",
        "The key relationship makes the conclusion clear.",
        "Cross-checking against the given facts confirms the direction.",
    ];
    let opener = rng.choice(&openers).expect("openers");
    let middle = rng.choice(&middles).expect("middles");
    format!("{opener} {middle} Therefore, {answer}")
}

/// Mutate the composed response into an adversarial shape in place.
fn apply_adversarial(
    variant: Adversarial,
    answer: &str,
    reasoning_prose: Option<&str>,
    content: &mut String,
    reasoning_content: &mut Option<String>,
    null_object_chunk: &mut bool,
) {
    match variant {
        Adversarial::LeadingWhitespace => {
            *content = format!("\n\n{content}");
        }
        Adversarial::TrailingProse => {
            content.push_str("\n\nActually, on reflection I am not fully certain.");
        }
        Adversarial::WrongCase => {
            *content = content.to_lowercase();
        }
        Adversarial::ReasoningOnly => {
            // Answer only in reasoning_content, empty content (github #1136).
            let prose = reasoning_prose.unwrap_or("");
            *reasoning_content = Some(if prose.is_empty() {
                answer.to_string()
            } else {
                format!("{prose}\n{answer}")
            });
            content.clear();
        }
        Adversarial::BoxedWrap => {
            *content = format!("\\boxed{{{content}}}");
        }
        Adversarial::MultipleConflicting => {
            // A decoy answer first; the real one last (tests take-LAST-match).
            *content = format!("The answer is (Z). Wait, reconsidering — {content}");
        }
        Adversarial::Unicode => {
            content.push_str(" ✓🎯—naïve");
        }
        Adversarial::NullObjectChunk => {
            *null_object_chunk = true;
        }
    }
}

/// Live tally of what the mock has actually answered so far this run. Updated
/// once per served, prompt-matched request (in `RequestCtx::build`), so
/// `correct / matched` is the accuracy the run is really achieving as it
/// progresses — an oracle to compare against what AIPerf's grader reports. This
/// is observed, not pre-computed: it counts real responses, including recycled
/// prompts and whatever subset of the dataset the run actually hit.
#[derive(Default)]
pub struct AccuracyLive {
    matched: AtomicU64,
    correct: AtomicU64,
    adversarial: AtomicU64,
    cot: AtomicU64,
    unmatched: AtomicU64,
    per_task: Mutex<BTreeMap<String, TaskCounts>>,
}

#[derive(Default, Clone, Copy)]
struct TaskCounts {
    matched: u64,
    correct: u64,
}

/// Per-task slice of the live tally.
#[derive(Debug, Clone, Serialize)]
pub struct TaskAccuracy {
    pub matched: u64,
    pub correct: u64,
    pub accuracy: f64,
}

/// A point-in-time copy of the live tally, safe to serialize.
#[derive(Debug, Clone, Serialize)]
pub struct AccuracyLiveSnapshot {
    /// Requests that matched a dataset prompt and were answered.
    pub matched: u64,
    /// Of those, how many were answered correctly.
    pub correct: u64,
    /// `matched - correct`.
    pub incorrect: u64,
    /// `correct / matched` (0.0 when nothing matched yet).
    pub accuracy: f64,
    /// Accuracy-enabled requests whose prompt did NOT match any dataset row.
    pub unmatched: u64,
    /// How many answered responses used an adversarial parser-choke shape.
    pub adversarial: u64,
    /// How many answered responses were rendered as chain-of-thought.
    pub cot: u64,
    /// Per-task breakdown, mirroring AIPerf's per-task accuracy table.
    pub tasks: BTreeMap<String, TaskAccuracy>,
}

fn ratio(correct: u64, matched: u64) -> f64 {
    if matched == 0 {
        0.0
    } else {
        correct as f64 / matched as f64
    }
}

impl AccuracyLive {
    /// Record one served, prompt-matched response.
    pub fn record(&self, decision: &AccuracyDecision, task: Option<&str>) {
        self.matched.fetch_add(1, Ordering::Relaxed);
        if decision.correct {
            self.correct.fetch_add(1, Ordering::Relaxed);
        }
        if decision.adversarial.is_some() {
            self.adversarial.fetch_add(1, Ordering::Relaxed);
        }
        if decision.cot {
            self.cot.fetch_add(1, Ordering::Relaxed);
        }
        let mut per_task = self.per_task.lock();
        let counts = per_task
            .entry(task.unwrap_or("unknown").to_string())
            .or_default();
        counts.matched += 1;
        if decision.correct {
            counts.correct += 1;
        }
    }

    /// Record one accuracy-enabled request whose prompt matched no dataset row.
    pub fn record_unmatched(&self) {
        self.unmatched.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a consistent snapshot of the current tally.
    pub fn snapshot(&self) -> AccuracyLiveSnapshot {
        let matched = self.matched.load(Ordering::Relaxed);
        let correct = self.correct.load(Ordering::Relaxed);
        let tasks = self
            .per_task
            .lock()
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    TaskAccuracy {
                        matched: v.matched,
                        correct: v.correct,
                        accuracy: ratio(v.correct, v.matched),
                    },
                )
            })
            .collect();
        AccuracyLiveSnapshot {
            matched,
            correct,
            incorrect: matched.saturating_sub(correct),
            accuracy: ratio(correct, matched),
            unmatched: self.unmatched.load(Ordering::Relaxed),
            adversarial: self.adversarial.load(Ordering::Relaxed),
            cot: self.cot.load(Ordering::Relaxed),
            tasks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MockServerConfig;

    fn cfg() -> MockServerConfig {
        MockServerConfig {
            random_seed: Some(1234),
            ..MockServerConfig::default()
        }
    }

    fn dataset(body: &str, mut mutate: impl FnMut(&mut MockServerConfig)) -> AccuracyDataset {
        let mut c = cfg();
        mutate(&mut c);
        AccuracyDataset::from_jsonl(body, &c).expect("dataset parses")
    }

    #[test]
    fn parses_aliases_and_normalizes() {
        let body = r#"{"question": "What is 2+2?", "answer": "4", "task": "math"}
{"text": "Capital of France?", "ground_truth": " B ", "subject": "geo"}"#;
        let ds = dataset(body, |_| {});
        assert_eq!(ds.len(), 2);
        let e = ds.lookup("What is 2+2?").expect("exact match");
        assert_eq!(e.gold, "4");
        assert_eq!(e.task.as_deref(), Some("math"));
    }

    #[test]
    fn lookup_substring_fallback_handles_wrapped_prompt() {
        let body = r#"{"prompt": "Capital of France?", "answer": "Paris"}"#;
        let ds = dataset(body, |_| {});
        let wrapped = "You are an expert.\n\nCapital of France?";
        assert!(ds.lookup(wrapped).is_some());
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let c = cfg();
        assert!(AccuracyDataset::from_jsonl("{\"foo\":1}\n", &c).is_err());
        assert!(AccuracyDataset::from_jsonl("", &c).is_err());
    }

    #[test]
    fn correct_answers_are_grader_formatted() {
        let body = r#"{"prompt":"q","answer":"B"}"#;
        let g = |f: AccuracyFormat, gold: &str| format_correct(f, gold);
        assert_eq!(g(AccuracyFormat::Mmlu, "B"), "The answer is (B)");
        assert_eq!(g(AccuracyFormat::MmluPro, "J"), "The answer is (J)");
        assert_eq!(g(AccuracyFormat::Gsm8k, "42"), "#### 42");
        assert_eq!(g(AccuracyFormat::Math, "42"), "\\boxed{42}");
        assert_eq!(g(AccuracyFormat::ExactMatch, "True"), "True");
        assert_eq!(g(AccuracyFormat::Passthrough, "hi"), "hi");
        // and correctness flows through decide()
        let ds = dataset(body, |c| {
            c.accuracy_format = AccuracyFormat::Mmlu;
            c.accuracy_correct_rate = 1.0;
        });
        let e = ds.lookup("q").unwrap();
        let d = ds.decide(e, "q");
        assert!(d.correct);
        assert_eq!(d.content, "The answer is (B)");
        assert!(d.reasoning_content.is_none());
    }

    #[test]
    fn wrong_answers_differ_from_gold() {
        let body = r#"{"prompt":"q","answer":"B"}"#;
        let ds = dataset(body, |c| {
            c.accuracy_format = AccuracyFormat::Mmlu;
            c.accuracy_correct_rate = 0.0;
        });
        let e = ds.lookup("q").unwrap();
        let d = ds.decide(e, "q");
        assert!(!d.correct);
        assert_ne!(d.content, "The answer is (B)");
        assert!(d.content.starts_with("The answer is ("));
    }

    #[test]
    fn gsm8k_wrong_is_a_different_number() {
        assert_eq!(bump_number("42"), "43");
        assert_eq!(bump_number("1,000"), "1001");
        assert_eq!(bump_number("3.5"), "4.5");
        assert_eq!(bump_number("abc"), "abc1");
    }

    #[test]
    fn decision_is_deterministic_per_prompt() {
        let body = r#"{"prompt":"q","answer":"B"}"#;
        let ds = dataset(body, |c| c.accuracy_correct_rate = 0.5);
        let e = ds.lookup("q").unwrap();
        let a = ds.decide(e, "q");
        let b = ds.decide(e, "q");
        assert_eq!(a.content, b.content);
        assert_eq!(a.correct, b.correct);
    }

    #[test]
    fn correct_rate_is_honored_across_prompts() {
        // 400 distinct prompts, rate 0.25 -> ~100 correct.
        let mut body = String::new();
        for i in 0..400 {
            body.push_str(&format!("{{\"prompt\":\"q{i}\",\"answer\":\"B\"}}\n"));
        }
        let ds = dataset(&body, |c| {
            c.accuracy_format = AccuracyFormat::Mmlu;
            c.accuracy_correct_rate = 0.25;
        });
        let mut correct = 0usize;
        for i in 0..400 {
            let key = format!("q{i}");
            let e = ds.lookup(&key).unwrap();
            if ds.decide(e, &key).correct {
                correct += 1;
            }
        }
        // Binomial(400, 0.25): mean 100, sd ~8.7; allow a wide band.
        assert!((70..=130).contains(&correct), "correct={correct}");
    }

    #[test]
    fn cot_uses_reasoning_field_by_default() {
        let body = r#"{"prompt":"q","answer":"B"}"#;
        let ds = dataset(body, |c| {
            c.accuracy_format = AccuracyFormat::Mmlu;
            c.accuracy_correct_rate = 1.0;
            c.accuracy_cot_rate = 1.0;
            c.accuracy_reasoning_field = true;
        });
        let e = ds.lookup("q").unwrap();
        let d = ds.decide(e, "q");
        assert!(d.cot);
        assert_eq!(d.content, "The answer is (B)");
        let r = d.reasoning_content.expect("reasoning present");
        assert!(r.contains("The answer is (B)"));
    }

    #[test]
    fn cot_inline_prefixes_answer_when_field_disabled() {
        let body = r#"{"prompt":"q","answer":"42"}"#;
        let ds = dataset(body, |c| {
            c.accuracy_format = AccuracyFormat::Gsm8k;
            c.accuracy_correct_rate = 1.0;
            c.accuracy_cot_rate = 1.0;
            c.accuracy_reasoning_field = false;
        });
        let e = ds.lookup("q").unwrap();
        let d = ds.decide(e, "q");
        assert!(d.reasoning_content.is_none());
        assert!(d.content.ends_with("#### 42"));
        assert!(d.content.contains("Therefore"));
    }

    #[test]
    fn exact_match_cot_never_pollutes_content_inline() {
        // Even with reasoning_field=false, exact-match content stays clean.
        let body = r#"{"prompt":"q","answer":"True"}"#;
        let ds = dataset(body, |c| {
            c.accuracy_format = AccuracyFormat::ExactMatch;
            c.accuracy_correct_rate = 1.0;
            c.accuracy_cot_rate = 1.0;
            c.accuracy_reasoning_field = false;
        });
        let e = ds.lookup("q").unwrap();
        let d = ds.decide(e, "q");
        assert_eq!(d.content, "True");
        assert!(d.reasoning_content.is_some());
    }

    #[test]
    fn adversarial_reasoning_only_empties_content() {
        // Force adversarial and find the ReasoningOnly draw across prompts.
        let mut found = false;
        for i in 0..64 {
            let key = format!("q{i}");
            let body = format!("{{\"prompt\":\"{key}\",\"answer\":\"B\"}}\n");
            let ds = dataset(&body, |c| {
                c.accuracy_format = AccuracyFormat::Mmlu;
                c.accuracy_correct_rate = 1.0;
                c.accuracy_adversarial_rate = 1.0;
            });
            let e = ds.lookup(&key).unwrap();
            let d = ds.decide(e, &key);
            assert!(d.adversarial.is_some());
            if d.adversarial == Some(Adversarial::ReasoningOnly) {
                assert!(d.content.is_empty());
                assert!(d.reasoning_content.unwrap().contains("The answer is (B)"));
                found = true;
                break;
            }
        }
        assert!(found, "never drew ReasoningOnly in 64 tries");
    }

    #[test]
    fn adversarial_null_object_chunk_sets_flag() {
        let mut found = false;
        for i in 0..64 {
            let key = format!("q{i}");
            let body = format!("{{\"prompt\":\"{key}\",\"answer\":\"B\"}}\n");
            let ds = dataset(&body, |c| {
                c.accuracy_format = AccuracyFormat::Mmlu;
                c.accuracy_adversarial_rate = 1.0;
            });
            let e = ds.lookup(&key).unwrap();
            let d = ds.decide(e, &key);
            if d.adversarial == Some(Adversarial::NullObjectChunk) {
                assert!(d.null_object_chunk);
                found = true;
                break;
            }
        }
        assert!(found, "never drew NullObjectChunk in 64 tries");
    }

    fn decision(correct: bool, cot: bool, adversarial: Option<Adversarial>) -> AccuracyDecision {
        AccuracyDecision {
            content: "x".into(),
            reasoning_content: None,
            null_object_chunk: false,
            correct,
            cot,
            adversarial,
        }
    }

    #[test]
    fn live_tally_counts_correct_incorrect_and_tasks() {
        let live = AccuracyLive::default();
        live.record(&decision(true, false, None), Some("demo"));
        live.record(&decision(false, false, None), Some("demo"));
        live.record(
            &decision(true, true, Some(Adversarial::WrongCase)),
            Some("other"),
        );
        live.record_unmatched();

        let s = live.snapshot();
        assert_eq!(s.matched, 3);
        assert_eq!(s.correct, 2);
        assert_eq!(s.incorrect, 1);
        assert!((s.accuracy - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.unmatched, 1);
        assert_eq!(s.adversarial, 1);
        assert_eq!(s.cot, 1);
        assert_eq!(s.tasks["demo"].matched, 2);
        assert_eq!(s.tasks["demo"].correct, 1);
        assert!((s.tasks["demo"].accuracy - 0.5).abs() < 1e-9);
        assert_eq!(s.tasks["other"].correct, 1);
        assert!((s.tasks["other"].accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn live_tally_empty_snapshot_is_zeroed() {
        let s = AccuracyLive::default().snapshot();
        assert_eq!(s.matched, 0);
        assert_eq!(s.accuracy, 0.0);
        assert!(s.tasks.is_empty());
    }

    #[test]
    fn per_entry_format_overrides_default() {
        let body = r#"{"prompt":"q","answer":"42","format":"gsm8k"}"#;
        let ds = dataset(body, |c| {
            c.accuracy_format = AccuracyFormat::Mmlu; // default differs
            c.accuracy_correct_rate = 1.0;
        });
        let e = ds.lookup("q").unwrap();
        assert_eq!(e.format, Some(AccuracyFormat::Gsm8k));
        assert_eq!(ds.decide(e, "q").content, "#### 42");
    }
}
