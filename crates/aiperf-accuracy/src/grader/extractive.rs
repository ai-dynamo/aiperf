// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native equivalents of the inherited lighteval extractive graders.
//!
//! Source configurations: `src/aiperf/accuracy/graders/lighteval_grader.py:1-295`.

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use regex::Regex;

use super::math::extract_last_boxed;
use super::{Grader, MathGrader};
use crate::AccuracyError;

/// Expression-extraction grader used by AIME24/AIME25.
#[derive(Debug, Default)]
pub struct ExpressionGrader {
    math: MathGrader,
}

/// LaTeX boxed-expression grader used by MATH-500.
#[derive(Debug, Default)]
pub struct LatexGrader {
    math: MathGrader,
}

/// Native-letter extractive grader used by GPQA-Diamond.
#[derive(Debug)]
pub struct GpqaGrader {
    labelled: Regex,
    standalone: Regex,
}

impl GpqaGrader {
    /// Builds the precompiled GPQA extractor.
    pub fn new() -> Self {
        Self {
            labelled: Regex::new(r"(?i)Answer:\s*\$?([A-D])\b").expect("static regex"),
            standalone: Regex::new(r"\b([A-D])\b").expect("static regex"),
        }
    }
}

impl Default for GpqaGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl Grader for ExpressionGrader {
    fn name(&self) -> &'static str {
        "expression"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let (prediction, _) = self.math.extract_answer(response_text);
        let correct = !prediction.is_empty() && self.math.equivalent(&prediction, ground_truth);
        Ok(GradingResult {
            correct,
            unparsed: prediction.is_empty(),
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted: (!prediction.is_empty()).then_some(prediction.clone()),
            ground_truth: ground_truth.trim().to_string(),
            reasoning: Some(format!(
                "native expression extraction: prediction {prediction:?}; match={correct}"
            )),
        })
    }
}

#[async_trait(?Send)]
impl Grader for LatexGrader {
    fn name(&self) -> &'static str {
        "latex"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let gold =
            extract_last_boxed(ground_truth).unwrap_or_else(|| ground_truth.trim().to_string());
        let (prediction, _) = self.math.extract_answer(response_text);
        let correct = !prediction.is_empty() && self.math.equivalent(&prediction, &gold);
        Ok(GradingResult {
            correct,
            unparsed: prediction.is_empty(),
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted: (!prediction.is_empty()).then_some(prediction.clone()),
            ground_truth: gold.clone(),
            reasoning: Some(format!(
                "native LaTeX extraction: prediction {prediction:?}, boxed gold {gold:?}; match={correct}"
            )),
        })
    }
}

#[async_trait(?Send)]
impl Grader for GpqaGrader {
    fn name(&self) -> &'static str {
        "gpqa"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let gold = ground_truth.trim().to_ascii_uppercase();
        if !matches!(gold.as_str(), "A" | "B" | "C" | "D") {
            return Err(AccuracyError::InvalidGroundTruth(format!(
                "GPQA expects A-D, received {ground_truth:?}"
            )));
        }
        let labelled = self
            .labelled
            .captures_iter(response_text)
            .last()
            .and_then(|captures| captures.get(1));
        let (extracted, fallback) = if let Some(answer) = labelled {
            (Some(answer.as_str().to_ascii_uppercase()), false)
        } else {
            (
                self.standalone
                    .captures_iter(response_text)
                    .last()
                    .and_then(|captures| captures.get(1))
                    .map(|answer| answer.as_str().to_ascii_uppercase()),
                true,
            )
        };
        let correct = extracted.as_deref() == Some(gold.as_str());
        Ok(GradingResult {
            correct,
            unparsed: fallback,
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted,
            ground_truth: gold,
            reasoning: Some(format!(
                "GPQA native-letter extraction{}; match={correct}",
                if fallback {
                    " (standalone fallback)"
                } else {
                    ""
                }
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grades_expression_latex_and_gpqa_shapes() {
        assert!(
            ExpressionGrader::default()
                .grade("work \\boxed{42}", "42")
                .await
                .unwrap()
                .correct
        );
        assert!(
            LatexGrader::default()
                .grade("\\boxed{1/2}", "solution \\boxed{\\frac{1}{2}}")
                .await
                .unwrap()
                .correct
        );
        let gpqa = GpqaGrader::new()
            .grade("reason\nAnswer: $C", "C")
            .await
            .unwrap();
        assert!(gpqa.correct);
        assert!(!gpqa.unparsed);
    }
}
