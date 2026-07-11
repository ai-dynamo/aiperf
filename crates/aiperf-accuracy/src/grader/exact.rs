// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict DeepEval-compatible exact-match grader.
//!
//! Ported from `src/aiperf/accuracy/graders/exact_match.py:1-77`.

use aiperf_metrics::GradingResult;
use async_trait::async_trait;

use super::Grader;
use crate::AccuracyError;

/// Strict, case-sensitive `trim(prediction) == trim(gold)` grader.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactMatchGrader;

#[async_trait(?Send)]
impl Grader for ExactMatchGrader {
    fn name(&self) -> &'static str {
        "exact-match"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let prediction = response_text.trim();
        let gold = ground_truth.trim();
        let unparsed = prediction.is_empty() && !gold.is_empty();
        let correct = !prediction.is_empty() && prediction == gold;
        Ok(GradingResult {
            correct,
            unparsed,
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted: Some(prediction.to_string()),
            ground_truth: gold.to_string(),
            reasoning: Some(format!(
                "strict equality: stripped prediction {prediction:?} vs gold {gold:?}; match={correct}{}",
                if unparsed { " (empty response)" } else { "" }
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn strict_case_and_whitespace_contract() {
        let grader = ExactMatchGrader;
        assert!(grader.grade(" A ", "A").await.unwrap().correct);
        assert!(!grader.grade("a", "A").await.unwrap().correct);
        assert!(!grader.grade("A.", "A").await.unwrap().correct);
        assert!(grader.grade("", "A").await.unwrap().unparsed);
        assert!(!grader.grade("", "").await.unwrap().unparsed);
    }
}
