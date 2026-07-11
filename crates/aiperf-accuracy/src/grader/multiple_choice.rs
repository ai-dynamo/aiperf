// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A-D multiple-choice extraction and exact comparison.
//!
//! Ported from `src/aiperf/accuracy/graders/multiple_choice.py:1-77`.

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use regex::Regex;

use super::Grader;
use crate::AccuracyError;

/// First-line A-D grader with a lone-letter regex fallback.
#[derive(Debug)]
pub struct MultipleChoiceGrader {
    letter: Regex,
}

impl MultipleChoiceGrader {
    /// Builds the precompiled grader.
    pub fn new() -> Self {
        Self {
            letter: Regex::new(r"(?:^|[^\p{L}\p{N}_])([A-D])(?:$|[^\p{L}\p{N}_])")
                .expect("static regex"),
        }
    }

    /// Returns `(answer, used_fallback)`.
    pub fn extract_answer(&self, response_text: &str) -> (String, bool) {
        let first_line = response_text
            .split_once('\n')
            .map_or(response_text, |(line, _)| line)
            .trim();
        if matches!(first_line, "A" | "B" | "C" | "D") {
            return (first_line.to_string(), false);
        }
        if let Some(answer) = self
            .letter
            .captures(first_line)
            .and_then(|captures| captures.get(1))
        {
            return (answer.as_str().to_string(), true);
        }
        (first_line.to_string(), true)
    }
}

impl Default for MultipleChoiceGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl Grader for MultipleChoiceGrader {
    fn name(&self) -> &'static str {
        "multiple-choice"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let (prediction, fallback) = self.extract_answer(response_text);
        let gold = ground_truth.trim();
        if !matches!(gold, "A" | "B" | "C" | "D") {
            return Err(AccuracyError::InvalidGroundTruth(format!(
                "multiple-choice expects A-D, received {ground_truth:?}"
            )));
        }
        let correct = !prediction.is_empty() && prediction == gold;
        Ok(GradingResult {
            correct,
            unparsed: fallback,
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted: Some(prediction.clone()),
            ground_truth: gold.to_string(),
            reasoning: Some(format!(
                "first-line answer {prediction:?}{} vs gold {gold:?}; match={correct}",
                if fallback { " (regex fallback)" } else { "" }
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn follows_first_line_and_fallback_contract() {
        let grader = MultipleChoiceGrader::new();
        let exact = grader.grade("A\nreason", "A").await.unwrap();
        assert!(exact.correct);
        assert!(!exact.unparsed);
        let fallback = grader.grade("The answer is B.", "B").await.unwrap();
        assert!(fallback.correct);
        assert!(fallback.unparsed);
        assert_eq!(fallback.extracted.as_deref(), Some("B"));
    }
}
