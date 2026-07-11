// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GSM8K quasi-exact-match grader with the inherited chat fallback.
//!
//! Ported from `src/aiperf/accuracy/graders/gsm8k_grader.py:1-176`.

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use regex::Regex;

use super::Grader;
use crate::AccuracyError;

const INVALID: &str = "[invalid]";

/// Native GSM8K grader.
#[derive(Debug)]
pub struct Gsm8kGrader {
    marker: Regex,
    number: Regex,
}

impl Gsm8kGrader {
    /// Builds the precompiled grader.
    pub fn new() -> Self {
        Self {
            marker: Regex::new(r"#### (-?[0-9.,]+)").expect("static regex"),
            number: Regex::new(r"-?[0-9][0-9,]*(?:\.[0-9]+)?").expect("static regex"),
        }
    }

    /// Lighteval-compatible `#### <number>` normalization.
    pub fn normalize_gold(&self, text: &str) -> String {
        self.marker
            .captures(text)
            .and_then(|captures| captures.get(1))
            .map(|value| value.as_str().trim().replace(',', ""))
            .unwrap_or_else(|| INVALID.to_string())
    }

    /// Returns `(answer, used_last_number_fallback)`.
    pub fn extract_answer(&self, text: &str) -> (String, bool) {
        let marked = self.normalize_gold(text);
        if marked != INVALID {
            return (marked, false);
        }
        self.number
            .find_iter(text)
            .last()
            .map(|value| (value.as_str().replace(',', ""), true))
            .unwrap_or_else(|| (INVALID.to_string(), true))
    }
}

impl Default for Gsm8kGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl Grader for Gsm8kGrader {
    fn name(&self) -> &'static str {
        "gsm8k"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let gold = self.normalize_gold(ground_truth.trim());
        let (prediction, fallback) = self.extract_answer(response_text.trim());
        let parseable = prediction != INVALID;
        let correct = parseable && gold != INVALID && numbers_match(&gold, &prediction);
        Ok(GradingResult {
            correct,
            unparsed: fallback || !parseable,
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted: Some(prediction.clone()),
            ground_truth: gold.clone(),
            reasoning: Some(format!(
                "gsm8k quasi-exact-match: gold {gold:?} vs prediction {prediction:?}; match={correct}{}",
                if fallback {
                    " (last-number fallback)"
                } else {
                    ""
                }
            )),
        })
    }
}

fn numbers_match(gold: &str, prediction: &str) -> bool {
    match (gold.parse::<f64>(), prediction.parse::<f64>()) {
        (Ok(gold), Ok(prediction)) => (gold - prediction).abs() <= 1e-6,
        _ => gold == prediction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn marker_and_chat_fallback_are_distinguished() {
        let grader = Gsm8kGrader::new();
        let gold = "work\n#### 24";
        let canonical = grader.grade("#### 24", gold).await.unwrap();
        assert!(canonical.correct);
        assert!(!canonical.unparsed);
        let chat = grader.grade("The answer is 24.0", gold).await.unwrap();
        assert!(chat.correct);
        assert!(chat.unparsed);
        assert_eq!(grader.normalize_gold("#### 1,234"), "1234");
    }
}
