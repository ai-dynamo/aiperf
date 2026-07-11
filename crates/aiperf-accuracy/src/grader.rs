// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy grader extension seam and MMLU-Pro grader.
//!
//! MMLU-Pro extraction follows the official repository's three tiers:
//! `evaluate_from_api.py:extract_answer`, `extract_again`, and `extract_final`,
//! plus the A-J choice range in `compute_accuracy.py`.

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use regex::Regex;

use crate::AccuracyError;

mod code_execution;
mod exact;
mod extractive;
mod gsm8k;
mod math;
mod multiple_choice;

pub use code_execution::{
    BubblewrapPythonExecutor, CodeExecutionGrader, CodeExecutionOutcome, CodeExecutionRequest,
    CodeExecutor, CodeTestCase,
};
pub use exact::ExactMatchGrader;
pub use extractive::{ExpressionGrader, GpqaGrader, LatexGrader};
pub use gsm8k::Gsm8kGrader;
pub use math::MathGrader;
pub use multiple_choice::MultipleChoiceGrader;

/// Grades generated text against a benchmark ground truth.
#[async_trait(?Send)]
pub trait Grader {
    /// Stable grader name.
    fn name(&self) -> &'static str;
    /// Validate optional runtime/backend prerequisites before dispatch begins.
    fn check_available(&self) -> Result<(), AccuracyError> {
        Ok(())
    }
    /// Grade one response while retaining extraction diagnostics.
    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError>;
}

/// Official-style MMLU-Pro A-J answer grader.
#[derive(Debug)]
pub struct MmluProGrader {
    primary: Regex,
    answer_label: Regex,
    standalone: Regex,
}

impl MmluProGrader {
    /// Builds the precompiled grader.
    pub fn new() -> Self {
        Self {
            primary: Regex::new(r"answer is \(?([A-J])\)?").expect("static regex"),
            answer_label: Regex::new(r"[aA]nswer:\s*([A-J])").expect("static regex"),
            standalone: Regex::new(r"\b[A-J]\b").expect("static regex"),
        }
    }

    /// Extracts `(answer, fallback_tier)`. Tier 0 is the canonical phrase, tier
    /// 1 is an `Answer:` label, and tier 2 is the last standalone A-J letter.
    pub fn extract_answer(&self, response_text: &str) -> (Option<String>, Option<u8>) {
        let cleaned = response_text.replace("**", "");
        if let Some(answer) = self
            .primary
            .captures(&cleaned)
            .and_then(|captures| captures.get(1))
        {
            return (Some(answer.as_str().to_string()), Some(0));
        }
        if let Some(answer) = self
            .answer_label
            .captures_iter(&cleaned)
            .last()
            .and_then(|captures| captures.get(1))
        {
            return (Some(answer.as_str().to_string()), Some(1));
        }
        let answer = self.standalone.find_iter(&cleaned).last();
        (
            answer.map(|matched| matched.as_str().to_string()),
            answer.map(|_| 2),
        )
    }
}

impl Default for MmluProGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl Grader for MmluProGrader {
    fn name(&self) -> &'static str {
        "mmlu-pro"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let gold = ground_truth.trim();
        if gold.len() != 1 || !matches!(gold.as_bytes()[0], b'A'..=b'J') {
            return Err(AccuracyError::InvalidGroundTruth(format!(
                "MMLU-Pro expects one letter A-J, received {ground_truth:?}"
            )));
        }
        let (extracted, tier) = self.extract_answer(response_text);
        let correct = extracted.as_deref() == Some(gold);
        let unparsed = tier != Some(0);
        let reasoning = match tier {
            Some(0) => format!("canonical 'answer is (X)' extraction; expected {gold}"),
            Some(1) => format!("fallback 'Answer: X' extraction; expected {gold}"),
            Some(2) => format!("fallback last standalone A-J extraction; expected {gold}"),
            None => format!("no A-J answer could be extracted; expected {gold}"),
            Some(_) => unreachable!("grader only emits tiers 0..=2"),
        };
        Ok(GradingResult {
            correct,
            unparsed,
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted,
            ground_truth: gold.to_string(),
            reasoning: Some(reasoning),
        })
    }
}

#[cfg(test)]
mod tests {
    use aiperf_metrics::GradingResult;

    use super::{Grader, MmluProGrader};

    async fn grade(text: &str, gold: &str) -> GradingResult {
        MmluProGrader::new().grade(text, gold).await.unwrap()
    }

    #[tokio::test]
    async fn grades_canonical_and_all_official_fallback_tiers() {
        let primary = grade("Reasoning. The answer is (J)", "J").await;
        assert!(primary.correct);
        assert!(!primary.unparsed);

        let labelled = grade("work\nAnswer: C", "C").await;
        assert!(labelled.correct);
        assert!(labelled.unparsed);

        let final_letter = grade("A was considered, but finally J", "J").await;
        assert!(final_letter.correct);
        assert!(final_letter.unparsed);
        assert_eq!(final_letter.extracted.as_deref(), Some("J"));

        // The official last-letter fallback would interpret a standalone pronoun
        // "I" as option I, so use text with no standalone A-J token here.
        let missing = grade("Unable to determine", "A").await;
        assert!(!missing.correct);
        assert!(missing.unparsed);
        assert_eq!(missing.extracted, None);
    }

    #[tokio::test]
    async fn canonical_pattern_matches_official_case_sensitivity() {
        let result = grade("The Answer Is (B)", "B").await;
        assert!(result.correct);
        assert!(
            result.unparsed,
            "non-canonical case reaches the standalone fallback"
        );
    }
}
