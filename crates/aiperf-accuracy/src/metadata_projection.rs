// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory-owned projection of provider metadata into public evaluation reports.
//!
//! Provider case labels, numeric-metric names, and aggregate definitions are
//! arbitrary strings and therefore restricted by default. A registered
//! provider factory may expose only exact values covered by this immutable,
//! fingerprinted projection. Future providers can implement
//! [`PublicEvaluationMetadataProjector`] directly; the stock implementation is
//! a deterministic declarative allowlist suitable for strict manifest decode.

use std::collections::BTreeMap;
use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::canonical::{
    CanonicalJson, is_sha256, redact_diagnostic, validate_no_secret_control_value,
};

/// Canonical schema ID for factory-owned public evaluation metadata rules.
pub const PUBLIC_EVALUATION_METADATA_SCHEMA_V1: &str = "aiperf-public-evaluation-metadata-v1";

/// One exact provider case-label mapping approved for public output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCaseMetadataRule {
    /// Provider-authored task label matched exactly.
    pub provider_task: String,
    /// Provider-authored source label matched exactly.
    pub provider_source: String,
    /// Factory-owned public task label.
    pub public_task: String,
    /// Factory-owned public source label.
    pub public_source: String,
}

/// One exact provider numeric-metric name approved for public output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicNumericMetricRule {
    /// Provider-authored metric name matched exactly.
    pub provider_name: String,
    /// Factory-owned public metric name.
    pub public_name: String,
}

/// One exact provider aggregate definition approved for public output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicAggregateMetadataRule {
    /// Provider-authored scorer identity matched exactly.
    pub provider_scorer: String,
    /// Provider-authored reducer identity matched exactly.
    pub provider_reducer: String,
    /// Provider-authored metric identity matched exactly.
    pub provider_metric: String,
    /// Factory-owned public scorer label.
    pub public_scorer: String,
    /// Factory-owned public reducer label.
    pub public_reducer: String,
    /// Factory-owned public metric label.
    pub public_metric: String,
    /// Exact canonical provider aggregate definition approved for public output.
    pub definition: CanonicalJson,
}

/// Factory-owned public case-label projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicCaseMetadataProjection {
    /// Reviewed public task label.
    pub task: String,
    /// Reviewed public source label.
    pub source: String,
}

/// Factory-owned public aggregate-label projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicAggregateMetadataProjection {
    /// Reviewed public scorer label.
    pub scorer: String,
    /// Reviewed public reducer label.
    pub reducer: String,
    /// Reviewed public metric label.
    pub metric: String,
}

#[derive(Clone, PartialEq, Eq)]
struct FrozenPublicAggregateMetadataRule {
    projection: PublicAggregateMetadataProjection,
    definition: CanonicalJson,
}

/// Replaceable factory validator/projector for public evaluator metadata.
pub trait PublicEvaluationMetadataProjector: Send + Sync {
    /// Exact canonical fingerprint of the executable projection policy.
    fn schema_sha256(&self) -> &str;

    /// Project one exact provider case label, or keep it restricted.
    fn project_case(
        &self,
        task: &str,
        source: &str,
    ) -> Result<Option<PublicCaseMetadataProjection>, PublicMetadataProjectionError>;

    /// Project one exact provider numeric-metric name, or keep it restricted.
    fn project_numeric_metric(
        &self,
        name: &str,
    ) -> Result<Option<String>, PublicMetadataProjectionError>;

    /// Project one exact provider aggregate definition, or keep it restricted.
    fn project_aggregate(
        &self,
        scorer: &str,
        reducer: &str,
        metric: &str,
        definition: &CanonicalJson,
    ) -> Result<Option<PublicAggregateMetadataProjection>, PublicMetadataProjectionError>;
}

/// Deterministic declarative public-metadata policy used by stock factories.
#[derive(Clone, PartialEq, Eq)]
pub struct FrozenPublicEvaluationMetadataPolicy {
    schema_sha256: String,
    cases: BTreeMap<(String, String), PublicCaseMetadataProjection>,
    numeric_metrics: BTreeMap<String, String>,
    aggregates: BTreeMap<(String, String, String), FrozenPublicAggregateMetadataRule>,
}

impl fmt::Debug for FrozenPublicEvaluationMetadataPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenPublicEvaluationMetadataPolicy")
            .field("schema_sha256", &self.schema_sha256)
            .field("case_rule_count", &self.cases.len())
            .field("numeric_metric_rule_count", &self.numeric_metrics.len())
            .field("aggregate_rule_count", &self.aggregates.len())
            .finish()
    }
}

impl FrozenPublicEvaluationMetadataPolicy {
    /// Validate, canonicalize, and freeze exact manifest-declared rules.
    pub fn new(
        case_rules: Vec<PublicCaseMetadataRule>,
        numeric_metric_rules: Vec<PublicNumericMetricRule>,
        aggregate_rules: Vec<PublicAggregateMetadataRule>,
    ) -> Result<Self, PublicMetadataProjectionError> {
        let mut cases = BTreeMap::new();
        let mut public_cases = std::collections::BTreeSet::new();
        for rule in &case_rules {
            validate_label(&rule.provider_task, "provider task")?;
            validate_label(&rule.provider_source, "provider source")?;
            validate_label(&rule.public_task, "public task")?;
            validate_label(&rule.public_source, "public source")?;
            if !public_cases.insert((rule.public_task.clone(), rule.public_source.clone()))
                || cases
                    .insert(
                        (rule.provider_task.clone(), rule.provider_source.clone()),
                        PublicCaseMetadataProjection {
                            task: rule.public_task.clone(),
                            source: rule.public_source.clone(),
                        },
                    )
                    .is_some()
            {
                return Err(PublicMetadataProjectionError::Policy(
                    "public case metadata rule was duplicated".to_string(),
                ));
            }
        }

        let mut numeric_metrics = BTreeMap::new();
        let mut public_numeric_metrics = std::collections::BTreeSet::new();
        for rule in &numeric_metric_rules {
            validate_label(&rule.provider_name, "provider numeric metric")?;
            validate_label(&rule.public_name, "public numeric metric")?;
            if !public_numeric_metrics.insert(rule.public_name.clone())
                || numeric_metrics
                    .insert(rule.provider_name.clone(), rule.public_name.clone())
                    .is_some()
            {
                return Err(PublicMetadataProjectionError::Policy(
                    "public numeric-metric rule was duplicated".to_string(),
                ));
            }
        }

        let mut aggregates = BTreeMap::new();
        let mut public_aggregates = std::collections::BTreeSet::new();
        for rule in &aggregate_rules {
            for (value, label) in [
                (&rule.provider_scorer, "provider aggregate scorer"),
                (&rule.provider_reducer, "provider aggregate reducer"),
                (&rule.provider_metric, "provider aggregate metric"),
                (&rule.public_scorer, "public aggregate scorer"),
                (&rule.public_reducer, "public aggregate reducer"),
                (&rule.public_metric, "public aggregate metric"),
            ] {
                validate_label(value, label)?;
            }
            validate_no_secret_control_value(&rule.definition)
                .map_err(|error| PublicMetadataProjectionError::Policy(error.to_string()))?;
            if !public_aggregates.insert((
                rule.public_scorer.clone(),
                rule.public_reducer.clone(),
                rule.public_metric.clone(),
            )) || aggregates
                .insert(
                    (
                        rule.provider_scorer.clone(),
                        rule.provider_reducer.clone(),
                        rule.provider_metric.clone(),
                    ),
                    FrozenPublicAggregateMetadataRule {
                        projection: PublicAggregateMetadataProjection {
                            scorer: rule.public_scorer.clone(),
                            reducer: rule.public_reducer.clone(),
                            metric: rule.public_metric.clone(),
                        },
                        definition: rule.definition.clone(),
                    },
                )
                .is_some()
            {
                return Err(PublicMetadataProjectionError::Policy(
                    "public aggregate metadata rule was duplicated".to_string(),
                ));
            }
        }

        let canonical_case_rules = cases
            .iter()
            .map(|((provider_task, provider_source), projection)| {
                json!({
                    "provider_source": provider_source,
                    "provider_task": provider_task,
                    "public_source": projection.source,
                    "public_task": projection.task,
                })
            })
            .collect::<Vec<_>>();
        let canonical_numeric_rules = numeric_metrics
            .iter()
            .map(|(provider_name, public_name)| {
                json!({
                    "provider_name": provider_name,
                    "public_name": public_name,
                })
            })
            .collect::<Vec<_>>();
        let canonical_aggregate_rules = aggregates
            .iter()
            .map(
                |((provider_scorer, provider_reducer, provider_metric), rule)| {
                    json!({
                        "provider_metric": provider_metric,
                        "provider_reducer": provider_reducer,
                        "provider_scorer": provider_scorer,
                        "public_metric": rule.projection.metric,
                        "public_reducer": rule.projection.reducer,
                        "public_scorer": rule.projection.scorer,
                        "definition": rule.definition.value(),
                    })
                },
            )
            .collect::<Vec<_>>();
        let schema_sha256 = CanonicalJson::new(json!({
            "aggregate_rules": canonical_aggregate_rules,
            "case_rules": canonical_case_rules,
            "numeric_metric_rules": canonical_numeric_rules,
            "schema": PUBLIC_EVALUATION_METADATA_SCHEMA_V1,
        }))
        .map_err(|error| PublicMetadataProjectionError::Policy(error.to_string()))?
        .normalized_result_sha256();
        debug_assert!(is_sha256(&schema_sha256));
        Ok(Self {
            schema_sha256,
            cases,
            numeric_metrics,
            aggregates,
        })
    }

    /// Construct an explicit policy that keeps all provider metadata restricted.
    pub fn restricted_only() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new())
            .expect("the empty public metadata policy is valid")
    }

    /// Verify a manifest fingerprint exactly matches executable rules.
    pub fn validate_schema_sha256(
        &self,
        expected: &str,
    ) -> Result<(), PublicMetadataProjectionError> {
        if !is_sha256(expected) || expected != self.schema_sha256 {
            return Err(PublicMetadataProjectionError::Policy(
                "public evaluation metadata fingerprint did not match executable rules".to_string(),
            ));
        }
        Ok(())
    }

    /// Whether this policy exposes no provider-authored metadata.
    pub fn is_restricted_only(&self) -> bool {
        self.cases.is_empty() && self.numeric_metrics.is_empty() && self.aggregates.is_empty()
    }
}

impl PublicEvaluationMetadataProjector for FrozenPublicEvaluationMetadataPolicy {
    fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    fn project_case(
        &self,
        task: &str,
        source: &str,
    ) -> Result<Option<PublicCaseMetadataProjection>, PublicMetadataProjectionError> {
        Ok(self
            .cases
            .get(&(task.to_string(), source.to_string()))
            .cloned())
    }

    fn project_numeric_metric(
        &self,
        name: &str,
    ) -> Result<Option<String>, PublicMetadataProjectionError> {
        Ok(self.numeric_metrics.get(name).cloned())
    }

    fn project_aggregate(
        &self,
        scorer: &str,
        reducer: &str,
        metric: &str,
        definition: &CanonicalJson,
    ) -> Result<Option<PublicAggregateMetadataProjection>, PublicMetadataProjectionError> {
        let Some(rule) =
            self.aggregates
                .get(&(scorer.to_string(), reducer.to_string(), metric.to_string()))
        else {
            return Ok(None);
        };
        if &rule.definition != definition {
            return Err(PublicMetadataProjectionError::rejected(
                "provider aggregate definition drifted from the factory-reviewed canonical value",
            ));
        }
        Ok(Some(rule.projection.clone()))
    }
}

/// Factory policy or provider metadata projection failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicMetadataProjectionError {
    /// Invalid, duplicated, or fingerprint-mismatched factory policy.
    Policy(String),
    /// Provider metadata failed one executable projection validator.
    Projection(String),
}

impl PublicMetadataProjectionError {
    /// Construct a content-redacted provider projection rejection.
    pub fn rejected(message: impl AsRef<str>) -> Self {
        Self::Projection(redact_diagnostic(message.as_ref()))
    }
}

impl Display for PublicMetadataProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(message) => write!(formatter, "public metadata policy: {message}"),
            Self::Projection(message) => {
                write!(
                    formatter,
                    "public metadata projection was rejected: {message}"
                )
            }
        }
    }
}

impl std::error::Error for PublicMetadataProjectionError {}

fn validate_label(value: &str, label: &str) -> Result<(), PublicMetadataProjectionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(PublicMetadataProjectionError::Policy(format!(
            "{label} was empty, oversized, padded, or contained control text"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> CanonicalJson {
        CanonicalJson::new(json!({
            "exclude_cancelled": true,
            "exclude_infrastructure": true,
        }))
        .unwrap()
    }

    fn policy() -> FrozenPublicEvaluationMetadataPolicy {
        FrozenPublicEvaluationMetadataPolicy::new(
            vec![PublicCaseMetadataRule {
                provider_task: "provider_gsm8k".to_string(),
                provider_source: "provider_gsm8k_source".to_string(),
                public_task: "gsm8k".to_string(),
                public_source: "openai/gsm8k:main:test".to_string(),
            }],
            vec![PublicNumericMetricRule {
                provider_name: "provider_reward".to_string(),
                public_name: "accuracy".to_string(),
            }],
            vec![PublicAggregateMetadataRule {
                provider_scorer: "provider_scorer".to_string(),
                provider_reducer: "provider_reducer".to_string(),
                provider_metric: "provider_metric".to_string(),
                public_scorer: "accuracy".to_string(),
                public_reducer: "mean".to_string(),
                public_metric: "accuracy".to_string(),
                definition: definition(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn exact_rules_project_and_unknown_metadata_stays_restricted() {
        let policy = policy();
        let case = policy
            .project_case("provider_gsm8k", "provider_gsm8k_source")
            .unwrap()
            .unwrap();
        assert_eq!(case.task, "gsm8k");
        assert_eq!(
            policy.project_numeric_metric("provider_reward").unwrap(),
            Some("accuracy".to_string())
        );
        let aggregate = policy
            .project_aggregate(
                "provider_scorer",
                "provider_reducer",
                "provider_metric",
                &definition(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(aggregate.scorer, "accuracy");
        assert_eq!(aggregate.reducer, "mean");
        assert_eq!(aggregate.metric, "accuracy");
        assert!(policy.project_case("private", "hidden").unwrap().is_none());
        assert!(
            policy
                .project_numeric_metric("answer=42")
                .unwrap()
                .is_none()
        );
        assert!(
            policy
                .project_aggregate(
                    "provider_scorer",
                    "provider_reducer",
                    "provider_metric",
                    &CanonicalJson::new(json!({"exclude_infrastructure": false})).unwrap(),
                )
                .is_err()
        );
        assert!(
            policy
                .project_aggregate(
                    "secret",
                    "mean",
                    "accuracy",
                    &CanonicalJson::new(json!({"private": true})).unwrap(),
                )
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn fingerprints_are_order_independent_and_exact() {
        let first = policy();
        let second = FrozenPublicEvaluationMetadataPolicy::new(
            vec![PublicCaseMetadataRule {
                provider_task: "provider_gsm8k".to_string(),
                provider_source: "provider_gsm8k_source".to_string(),
                public_task: "gsm8k".to_string(),
                public_source: "openai/gsm8k:main:test".to_string(),
            }],
            vec![PublicNumericMetricRule {
                provider_name: "provider_reward".to_string(),
                public_name: "accuracy".to_string(),
            }],
            vec![PublicAggregateMetadataRule {
                provider_scorer: "provider_scorer".to_string(),
                provider_reducer: "provider_reducer".to_string(),
                provider_metric: "provider_metric".to_string(),
                public_scorer: "accuracy".to_string(),
                public_reducer: "mean".to_string(),
                public_metric: "accuracy".to_string(),
                definition: definition(),
            }],
        )
        .unwrap();
        assert_eq!(first.schema_sha256(), second.schema_sha256());
        first.validate_schema_sha256(first.schema_sha256()).unwrap();
        assert!(first.validate_schema_sha256(&"0".repeat(64)).is_err());
    }

    #[test]
    fn stock_metadata_fingerprints_match_exact_task_manifest_rules() {
        let source = "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866";
        let case_rules = || {
            vec![PublicCaseMetadataRule {
                provider_task: "gsm8k".to_string(),
                provider_source: source.to_string(),
                public_task: "gsm8k".to_string(),
                public_source: source.to_string(),
            }]
        };
        let nemo = FrozenPublicEvaluationMetadataPolicy::new(
            case_rules(),
            vec![PublicNumericMetricRule {
                provider_name: "reward".to_string(),
                public_name: "accuracy".to_string(),
            }],
            vec![PublicAggregateMetadataRule {
                provider_scorer: "nemo_evaluator.gsm8k_scorer".to_string(),
                provider_reducer: "mean".to_string(),
                provider_metric: "reward".to_string(),
                public_scorer: "accuracy".to_string(),
                public_reducer: "mean".to_string(),
                public_metric: "accuracy".to_string(),
                definition: CanonicalJson::new(json!({
                    "exclude_cancelled": true,
                    "exclude_infrastructure": true,
                }))
                .unwrap(),
            }],
        )
        .unwrap();
        assert_eq!(
            nemo.schema_sha256(),
            "6f410c2015a7139126bfd76e892da26a7e08ce45a28379da72f9299472b0234b"
        );

        let openbench = FrozenPublicEvaluationMetadataPolicy::new(
            case_rules(),
            vec![PublicNumericMetricRule {
                provider_name: "grade_school_math_scorer".to_string(),
                public_name: "accuracy".to_string(),
            }],
            vec![PublicAggregateMetadataRule {
                provider_scorer: "grade_school_math_scorer".to_string(),
                provider_reducer: "identity".to_string(),
                provider_metric: "accuracy".to_string(),
                public_scorer: "accuracy".to_string(),
                public_reducer: "mean".to_string(),
                public_metric: "accuracy".to_string(),
                definition: CanonicalJson::new(json!({
                    "metric_params": {},
                    "params": {},
                    "score_name": "grade_school_math_scorer",
                }))
                .unwrap(),
            }],
        )
        .unwrap();
        assert_eq!(
            openbench.schema_sha256(),
            "41dec64f73bd428308c6f8b39bd78b7fd11424768043d0ffa6384fb3df20b360"
        );
    }

    #[test]
    fn duplicate_or_unsafe_rules_fail_closed() {
        let duplicate = PublicNumericMetricRule {
            provider_name: "accuracy".to_string(),
            public_name: "accuracy".to_string(),
        };
        assert!(
            FrozenPublicEvaluationMetadataPolicy::new(
                Vec::new(),
                vec![duplicate.clone(), duplicate],
                Vec::new(),
            )
            .is_err()
        );
        assert!(
            FrozenPublicEvaluationMetadataPolicy::new(
                Vec::new(),
                vec![
                    PublicNumericMetricRule {
                        provider_name: "provider_accuracy".to_string(),
                        public_name: "accuracy".to_string(),
                    },
                    PublicNumericMetricRule {
                        provider_name: "provider_reward".to_string(),
                        public_name: "accuracy".to_string(),
                    },
                ],
                Vec::new(),
            )
            .is_err()
        );
        assert!(
            FrozenPublicEvaluationMetadataPolicy::new(
                vec![PublicCaseMetadataRule {
                    provider_task: "gsm8k".to_string(),
                    provider_source: "source".to_string(),
                    public_task: "hidden\nanswer".to_string(),
                    public_source: "source".to_string(),
                }],
                Vec::new(),
                Vec::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn debug_and_projection_never_expose_private_tuple_or_definition() {
        let sentinel = "HIDDEN_EXPECTED_ANSWER_SENTINEL";
        let policy = FrozenPublicEvaluationMetadataPolicy::new(
            Vec::new(),
            Vec::new(),
            vec![PublicAggregateMetadataRule {
                provider_scorer: "private_provider_scorer".to_string(),
                provider_reducer: "private_provider_reducer".to_string(),
                provider_metric: "private_provider_metric".to_string(),
                public_scorer: "accuracy".to_string(),
                public_reducer: "mean".to_string(),
                public_metric: "accuracy".to_string(),
                definition: CanonicalJson::new(json!({"private_note": sentinel})).unwrap(),
            }],
        )
        .unwrap();
        let debug = format!("{policy:?}");
        assert!(!debug.contains(sentinel));
        assert!(!debug.contains("private_provider_scorer"));
        let projection = policy
            .project_aggregate(
                "private_provider_scorer",
                "private_provider_reducer",
                "private_provider_metric",
                &CanonicalJson::new(json!({"private_note": sentinel})).unwrap(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(projection.scorer, "accuracy");
        assert_eq!(projection.reducer, "mean");
        assert_eq!(projection.metric, "accuracy");
    }
}
