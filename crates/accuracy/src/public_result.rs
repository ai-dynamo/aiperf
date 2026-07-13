// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory-owned closure for public evaluator results.
//!
//! Provider-native score trees, aggregate labels, definitions, and auxiliary
//! metrics stay restricted. A case score becomes public only after an exact
//! factory-registered schema validator accepts it. An aggregate becomes public
//! only after a separate factory validator recognizes its provider tuple,
//! validates its complete definition and denominator, and proves its value
//! against those already validated case projections.
//!
//! The stock binary projection follows the pinned provider semantics rather
//! than recreating either grader: NeMo Evaluator produces its numeric reward in
//! `nemo_evaluator/environments/custom.py:227-249`, while OpenBench produces an
//! exact `0.0` or `1.0` in
//! `openbench/scorers/grade_school_math.py:11-38`. Inspect's owning accuracy
//! metric is the arithmetic mean in
//! `inspect_ai/scorer/_metrics/accuracy.py:14-35`; its reducer, score filtering,
//! and count construction remain Python-owned in
//! `inspect_ai/_eval/task/results.py:120-165,237-245,272-330`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::canonical::{
    CanonicalJson, is_sha256, redact_diagnostic, validate_no_secret_control_value,
};
use crate::provider_protocol::{
    AggregateMetric, CaseOutcome, CaseOutcomeKind, EvaluationCaseId, FiniteF64,
};
use crate::score_projection::{
    FINITE_BINARY_NUMBER_SCHEMA_SHA256, PublicScoreProjectionPolicy, finite_binary_number,
};

/// Exact stock NeMo Evaluator GSM8K score name.
pub const NEMO_GSM8K_PUBLIC_SCORE_NAME: &str = "reward";

/// Exact stock OpenBench GSM8K score name.
pub const OPENBENCH_GSM8K_PUBLIC_SCORE_NAME: &str = "grade_school_math_scorer";

/// Schema ID for an exact provider aggregate projected as public mean accuracy.
pub const EXACT_BINARY_MEAN_AGGREGATE_SCHEMA_V1: &str = "aiperf-exact-binary-mean-aggregate-v1";

/// Stable stock registration ID for the reviewed public mean accuracy projection.
pub const STOCK_ACCURACY_MEAN_PROJECTION_ID: &str = "accuracy_mean";

/// Exact executable aggregate-rule fingerprint for stock NeMo Evaluator GSM8K.
pub const NEMO_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256: &str =
    "d523fa29449c207508f94e50fbe0540d5a0b50a5ba48fe66cc6540d9086c5f4b";

/// Exact executable aggregate-rule fingerprint for stock OpenBench GSM8K.
pub const OPENBENCH_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256: &str =
    "fa74629fee52533d6f210b2f2c2a4a8b7d9b48bd328b31aa90494be5d04e39d5";

/// Schema ID for the audited stock OpenBench uniform-epoch accuracy projection.
pub const OPENBENCH_GSM8K_AGGREGATE_SCHEMA_V1: &str =
    "aiperf-openbench-gsm8k-uniform-epoch-mean-v1";

/// Maximum flat-reference ULP distance audited for stock OpenBench GSM8K.
pub const OPENBENCH_GSM8K_MAX_MEAN_ULPS: u64 = 2;

const PUBLIC_ACCURACY_SCORER: &str = "accuracy";
const PUBLIC_ACCURACY_REDUCER: &str = "mean";
const PUBLIC_ACCURACY_METRIC: &str = "accuracy";

/// One case score whose complete public value passed its registered validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPublicScoreProjection {
    value: CanonicalJson,
    schema_sha256: String,
}

impl ValidatedPublicScoreProjection {
    /// Borrow the validated canonical public value.
    pub fn value(&self) -> &CanonicalJson {
        &self.value
    }

    /// Return the exact executable schema fingerprint that accepted the value.
    pub fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValidatedPublicCaseKind {
    Completed {
        scores: BTreeMap<String, ValidatedPublicScoreProjection>,
    },
    InfrastructureError,
    Cancelled,
}

/// Exact terminal case set with only registered public score projections retained.
///
/// The private terminal variants prevent callers from attaching a score to an
/// infrastructure or cancellation outcome. Future aggregate validators consume
/// this same invariant-preserving view through its query methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPublicCaseProjections {
    cases: BTreeMap<EvaluationCaseId, ValidatedPublicCaseKind>,
}

impl ValidatedPublicCaseProjections {
    /// Validate every provider-authored public case projection and freeze outcomes.
    pub fn new(
        outcomes: &[CaseOutcome],
        policy: &PublicScoreProjectionPolicy,
    ) -> Result<Self, PublicResultProjectionError> {
        let mut cases = BTreeMap::new();
        for outcome in outcomes {
            let kind = match &outcome.outcome {
                CaseOutcomeKind::Completed { completed } => {
                    let mut scores = BTreeMap::new();
                    for (name, score) in &completed.scores {
                        let Some(value) = &score.public_projection else {
                            continue;
                        };
                        let schema_sha256 = policy
                            .validate(name, value)
                            .map_err(|error| PublicResultProjectionError::Score(error.to_string()))?
                            .to_string();
                        scores.insert(
                            name.clone(),
                            ValidatedPublicScoreProjection {
                                value: value.clone(),
                                schema_sha256,
                            },
                        );
                    }
                    ValidatedPublicCaseKind::Completed { scores }
                }
                CaseOutcomeKind::InfrastructureError { .. } => {
                    ValidatedPublicCaseKind::InfrastructureError
                }
                CaseOutcomeKind::Cancelled { .. } => ValidatedPublicCaseKind::Cancelled,
            };
            if cases.insert(outcome.case_id.clone(), kind).is_some() {
                return Err(PublicResultProjectionError::Projection(
                    "public case projection set contained duplicate case identities".to_string(),
                ));
            }
        }
        Ok(Self { cases })
    }

    /// Number of case terminals in the exact projection set.
    pub fn case_count(&self) -> usize {
        self.cases.len()
    }

    /// Number of semantically completed cases, including valid zero scores.
    pub fn completed_count(&self) -> usize {
        self.cases
            .values()
            .filter(|kind| matches!(kind, ValidatedPublicCaseKind::Completed { .. }))
            .count()
    }

    /// Number of cases excluded because evaluator infrastructure failed.
    pub fn infrastructure_error_count(&self) -> usize {
        self.cases
            .values()
            .filter(|kind| matches!(kind, ValidatedPublicCaseKind::InfrastructureError))
            .count()
    }

    /// Number of cases excluded because evaluation was cancelled.
    pub fn cancelled_count(&self) -> usize {
        self.cases
            .values()
            .filter(|kind| matches!(kind, ValidatedPublicCaseKind::Cancelled))
            .count()
    }

    /// Borrow the complete registered public score map for one completed case.
    pub fn scores(
        &self,
        case_id: &EvaluationCaseId,
    ) -> Option<&BTreeMap<String, ValidatedPublicScoreProjection>> {
        match self.cases.get(case_id) {
            Some(ValidatedPublicCaseKind::Completed { scores }) => Some(scores),
            Some(
                ValidatedPublicCaseKind::InfrastructureError | ValidatedPublicCaseKind::Cancelled,
            )
            | None => None,
        }
    }

    /// Iterate every completed case and its optional projection under one exact name.
    pub fn completed_scores<'a>(
        &'a self,
        score_name: &'a str,
    ) -> impl Iterator<
        Item = (
            &'a EvaluationCaseId,
            Option<&'a ValidatedPublicScoreProjection>,
        ),
    > + 'a {
        self.cases.iter().filter_map(move |(case_id, kind)| {
            let ValidatedPublicCaseKind::Completed { scores } = kind else {
                return None;
            };
            Some((case_id, scores.get(score_name)))
        })
    }
}

/// Factory-validated public aggregate with provider-native identity removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPublicAggregateProjection {
    scorer: String,
    reducer: String,
    metric: String,
    value: FiniteF64,
    scored_count: u64,
    unscored_count: u64,
    schema_sha256: String,
}

impl ValidatedPublicAggregateProjection {
    /// Factory-reviewed stable public scorer label.
    pub fn scorer(&self) -> &str {
        &self.scorer
    }

    /// Factory-reviewed stable public reducer label.
    pub fn reducer(&self) -> &str {
        &self.reducer
    }

    /// Factory-reviewed stable public metric label.
    pub fn metric(&self) -> &str {
        &self.metric
    }

    /// Provider-computed finite value after consistency validation.
    pub fn value(&self) -> f64 {
        self.value.get()
    }

    /// Exact number of factory-validated aggregation units in the denominator.
    pub fn scored_count(&self) -> u64 {
        self.scored_count
    }

    /// Exact number of factory-validated units excluded from the denominator.
    pub fn unscored_count(&self) -> u64 {
        self.unscored_count
    }

    /// Exact executable aggregate-rule fingerprint.
    pub fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }
}

/// Replaceable factory validator for one provider aggregate projection.
///
/// Candidate recognition deliberately remains inside the implementation, so
/// provider tuple and definition strings never become generic report labels.
pub trait PublicAggregateProjectionValidator: Send + Sync {
    /// Exact fingerprint of the immutable executable aggregate rule.
    fn schema_sha256(&self) -> &str;

    /// Whether this validator owns the candidate's exact provider tuple.
    fn recognizes(&self, candidate: &AggregateMetric) -> bool;

    /// Validate the complete candidate against the exact public case set.
    ///
    /// `None` means the tuple and counts were valid but no completed projected
    /// case exists, so there is no semantic aggregate to publish.
    fn validate(
        &self,
        candidate: &AggregateMetric,
        cases: &ValidatedPublicCaseProjections,
        safe_config: &CanonicalJson,
    ) -> Result<Option<ValidatedPublicAggregateProjection>, PublicResultProjectionError>;
}

#[derive(Clone)]
struct PublicAggregateProjectionRule {
    validator: Arc<dyn PublicAggregateProjectionValidator>,
}

impl fmt::Debug for PublicAggregateProjectionRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicAggregateProjectionRule")
            .field("schema_sha256", &self.validator.schema_sha256())
            .finish()
    }
}

/// Deterministic factory registry for public aggregate validators.
#[derive(Clone, Default)]
pub struct PublicAggregateProjectionPolicy {
    rules: BTreeMap<String, PublicAggregateProjectionRule>,
}

impl PublicAggregateProjectionPolicy {
    /// Construct a policy under which every provider aggregate stays restricted.
    pub fn restricted_only() -> Self {
        Self::default()
    }

    /// Register one immutable aggregate projection ID and executable validator.
    pub fn register(
        &mut self,
        projection_id: impl Into<String>,
        validator: Arc<dyn PublicAggregateProjectionValidator>,
    ) -> Result<&mut Self, PublicResultProjectionError> {
        let projection_id = projection_id.into();
        validate_label(&projection_id).map_err(PublicResultProjectionError::Policy)?;
        if !is_sha256(validator.schema_sha256()) || self.rules.contains_key(&projection_id) {
            return Err(PublicResultProjectionError::Policy(
                "public aggregate projection ID/schema was invalid or duplicated".to_string(),
            ));
        }
        self.rules
            .insert(projection_id, PublicAggregateProjectionRule { validator });
        Ok(self)
    }

    /// Validate registered candidates and return only reviewed public projections.
    ///
    /// Unrecognized candidates remain restricted. Every registered validator
    /// must recognize exactly one candidate, and no candidate may be claimed by
    /// more than one validator.
    pub fn project(
        &self,
        candidates: &[AggregateMetric],
        cases: &ValidatedPublicCaseProjections,
        safe_config: &CanonicalJson,
    ) -> Result<Vec<ValidatedPublicAggregateProjection>, PublicResultProjectionError> {
        let mut claimed = BTreeSet::new();
        let mut public_keys = BTreeSet::new();
        let mut projections = Vec::with_capacity(self.rules.len());
        for rule in self.rules.values() {
            let matches = candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| rule.validator.recognizes(candidate))
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(PublicResultProjectionError::Projection(
                    "registered aggregate validator did not recognize exactly one candidate"
                        .to_string(),
                ));
            }
            let (index, candidate) = matches[0];
            if !claimed.insert(index) {
                return Err(PublicResultProjectionError::Policy(
                    "multiple public aggregate validators recognized one provider candidate"
                        .to_string(),
                ));
            }
            let Some(projection) = rule.validator.validate(candidate, cases, safe_config)? else {
                continue;
            };
            let public_key = (
                projection.scorer.clone(),
                projection.reducer.clone(),
                projection.metric.clone(),
            );
            if !public_keys.insert(public_key) {
                return Err(PublicResultProjectionError::Policy(
                    "public aggregate validators produced duplicate reviewed labels".to_string(),
                ));
            }
            projections.push(projection);
        }
        Ok(projections)
    }

    /// Return immutable fingerprints in deterministic projection-ID order.
    pub fn schema_fingerprints(&self) -> BTreeMap<String, String> {
        self.rules
            .iter()
            .map(|(name, rule)| (name.clone(), rule.validator.schema_sha256().to_string()))
            .collect()
    }

    /// Prove descriptor metadata exactly matches executable aggregate validators.
    pub fn validate_descriptor_fingerprints(
        &self,
        fingerprints: &BTreeMap<String, String>,
    ) -> Result<(), PublicResultProjectionError> {
        if &self.schema_fingerprints() != fingerprints {
            return Err(PublicResultProjectionError::Policy(
                "provider aggregate schema descriptors did not match factory validators"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Whether no provider aggregate is eligible for public projection.
    pub fn is_restricted_only(&self) -> bool {
        self.rules.is_empty()
    }
}

impl fmt::Debug for PublicAggregateProjectionPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicAggregateProjectionPolicy")
            .field("schema_fingerprints", &self.schema_fingerprints())
            .finish()
    }
}

impl PartialEq for PublicAggregateProjectionPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.schema_fingerprints() == other.schema_fingerprints()
    }
}

impl Eq for PublicAggregateProjectionPolicy {}

/// Exact binary-case mean validator used by the stock accuracy providers.
///
/// Python remains the scorer and aggregator. This validator only closes the
/// public boundary: it recognizes one private provider tuple, requires its
/// complete canonical definition and counts, verifies the returned value
/// against the direct binary case projections, and emits stable
/// `accuracy`/`mean`/`accuracy` labels.
#[derive(Clone, PartialEq, Eq)]
pub struct ExactBinaryMeanAggregateValidator {
    provider_scorer: String,
    provider_reducer: String,
    provider_metric: String,
    provider_definition: CanonicalJson,
    source_score_name: String,
    schema_sha256: String,
}

impl ExactBinaryMeanAggregateValidator {
    /// Construct one exact factory-owned provider-to-public aggregate rule.
    pub fn accuracy(
        provider_scorer: impl Into<String>,
        provider_reducer: impl Into<String>,
        provider_metric: impl Into<String>,
        provider_definition: CanonicalJson,
        source_score_name: impl Into<String>,
    ) -> Result<Self, PublicResultProjectionError> {
        let provider_scorer = provider_scorer.into();
        let provider_reducer = provider_reducer.into();
        let provider_metric = provider_metric.into();
        let source_score_name = source_score_name.into();
        for value in [
            &provider_scorer,
            &provider_reducer,
            &provider_metric,
            &source_score_name,
        ] {
            validate_label(value).map_err(PublicResultProjectionError::Policy)?;
        }
        validate_no_secret_control_value(&provider_definition).map_err(|error| {
            PublicResultProjectionError::Policy(redact_diagnostic(&error.to_string()))
        })?;
        let schema_sha256 = CanonicalJson::new(json!({
            "comparison": {
                "max_ulps": 0,
                "reference": "flat_binary_case_mean",
            },
            "denominator": {
                "cancelled": "unscored",
                "completed": "scored",
                "infrastructure_error": "unscored",
            },
            "empty_completed_projection": "absent",
            "provider": {
                "definition": provider_definition.value(),
                "metric": provider_metric,
                "reducer": provider_reducer,
                "scorer": provider_scorer,
            },
            "public": {
                "metric": PUBLIC_ACCURACY_METRIC,
                "reducer": PUBLIC_ACCURACY_REDUCER,
                "scorer": PUBLIC_ACCURACY_SCORER,
            },
            "schema": EXACT_BINARY_MEAN_AGGREGATE_SCHEMA_V1,
            "source_score": {
                "name": source_score_name,
                "schema_sha256": FINITE_BINARY_NUMBER_SCHEMA_SHA256,
            },
        }))
        .map_err(|error| PublicResultProjectionError::Policy(error.to_string()))?
        .normalized_result_sha256();
        Ok(Self {
            provider_scorer,
            provider_reducer,
            provider_metric,
            provider_definition,
            source_score_name,
            schema_sha256,
        })
    }
}

impl fmt::Debug for ExactBinaryMeanAggregateValidator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExactBinaryMeanAggregateValidator")
            .field("schema_sha256", &self.schema_sha256)
            .finish_non_exhaustive()
    }
}

impl PublicAggregateProjectionValidator for ExactBinaryMeanAggregateValidator {
    fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    fn recognizes(&self, candidate: &AggregateMetric) -> bool {
        candidate.scorer == self.provider_scorer
            && candidate.reducer == self.provider_reducer
            && candidate.metric == self.provider_metric
    }

    fn validate(
        &self,
        candidate: &AggregateMetric,
        cases: &ValidatedPublicCaseProjections,
        _safe_config: &CanonicalJson,
    ) -> Result<Option<ValidatedPublicAggregateProjection>, PublicResultProjectionError> {
        if !self.recognizes(candidate) || candidate.definition != self.provider_definition {
            return Err(PublicResultProjectionError::rejected(
                "provider aggregate tuple or definition did not match its factory rule",
            ));
        }

        let completed_count = u64::try_from(cases.completed_count()).map_err(|_| {
            PublicResultProjectionError::Projection(
                "completed public case count exceeded the aggregate protocol domain".to_string(),
            )
        })?;
        let infrastructure_error_count = u64::try_from(cases.infrastructure_error_count())
            .map_err(|_| {
                PublicResultProjectionError::Projection(
                    "infrastructure case count exceeded the aggregate protocol domain".to_string(),
                )
            })?;
        let cancelled_count = u64::try_from(cases.cancelled_count()).map_err(|_| {
            PublicResultProjectionError::Projection(
                "cancelled case count exceeded the aggregate protocol domain".to_string(),
            )
        })?;
        let unscored_count = infrastructure_error_count
            .checked_add(cancelled_count)
            .ok_or_else(|| {
                PublicResultProjectionError::Projection(
                    "unscored public case count overflowed".to_string(),
                )
            })?;
        if candidate.scored_count != completed_count || candidate.unscored_count != unscored_count {
            return Err(PublicResultProjectionError::rejected(
                "provider aggregate counts did not match completed and excluded case projections",
            ));
        }

        let mut ones = 0_u64;
        let mut projected_count = 0_u64;
        for (_, projection) in cases.completed_scores(&self.source_score_name) {
            let projection = projection.ok_or_else(|| {
                PublicResultProjectionError::rejected(
                    "a completed case omitted the aggregate source projection",
                )
            })?;
            if projection.schema_sha256() != FINITE_BINARY_NUMBER_SCHEMA_SHA256 {
                return Err(PublicResultProjectionError::rejected(
                    "aggregate source projection used the wrong executable schema",
                ));
            }
            let value = finite_binary_number(projection.value())
                .map_err(PublicResultProjectionError::rejected)?;
            ones = ones.checked_add(value as u64).ok_or_else(|| {
                PublicResultProjectionError::Projection(
                    "binary public score total overflowed".to_string(),
                )
            })?;
            projected_count = projected_count.checked_add(1).ok_or_else(|| {
                PublicResultProjectionError::Projection(
                    "public score projection count overflowed".to_string(),
                )
            })?;
        }
        if projected_count != completed_count {
            return Err(PublicResultProjectionError::rejected(
                "aggregate source projection set did not match completed cases",
            ));
        }
        if completed_count == 0 {
            return Ok(None);
        }

        let expected = ones as f64 / completed_count as f64;
        if candidate.value.get() != expected {
            return Err(PublicResultProjectionError::rejected(
                "provider aggregate value was inconsistent with public case projections",
            ));
        }
        Ok(Some(ValidatedPublicAggregateProjection {
            scorer: PUBLIC_ACCURACY_SCORER.to_string(),
            reducer: PUBLIC_ACCURACY_REDUCER.to_string(),
            metric: PUBLIC_ACCURACY_METRIC.to_string(),
            value: candidate.value,
            scored_count: completed_count,
            unscored_count,
            schema_sha256: self.schema_sha256.clone(),
        }))
    }
}

/// Factory-owned validator for stock OpenBench GSM8K epoch reduction.
///
/// The pinned Inspect implementation first applies `statistics.mean` per
/// sample across epochs (`inspect_ai/scorer/_reducer/reducer.py:39-54,332-354`)
/// and then its accuracy metric sums those reduced sample values in order
/// (`inspect_ai/scorer/_metrics/accuracy.py:14-35`). The frozen stock config
/// permits one through five samples and one through eight epochs. Exhaustive
/// enumeration proves the resulting value stays within two ULPs of the flat
/// binary-case mean. Any incomplete case set remains restricted because
/// partial epoch groups have different weighting semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenBenchGsm8kAggregateValidator;

impl OpenBenchGsm8kAggregateValidator {
    /// Construct the immutable stock OpenBench aggregate validator.
    pub const fn new() -> Self {
        Self
    }
}

impl PublicAggregateProjectionValidator for OpenBenchGsm8kAggregateValidator {
    fn schema_sha256(&self) -> &str {
        OPENBENCH_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256
    }

    fn recognizes(&self, candidate: &AggregateMetric) -> bool {
        candidate.scorer == "grade_school_math_scorer"
            && candidate.reducer == "identity"
            && candidate.metric == "accuracy"
    }

    fn validate(
        &self,
        candidate: &AggregateMetric,
        cases: &ValidatedPublicCaseProjections,
        safe_config: &CanonicalJson,
    ) -> Result<Option<ValidatedPublicAggregateProjection>, PublicResultProjectionError> {
        if !self.recognizes(candidate)
            || candidate.definition.value()
                != &json!({
                    "metric_params": {},
                    "params": {},
                    "score_name": "grade_school_math_scorer",
                })
        {
            return Err(PublicResultProjectionError::rejected(
                "OpenBench aggregate tuple or definition did not match its factory rule",
            ));
        }
        let (sample_count, epochs) = openbench_gsm8k_shape(safe_config)?;
        let planned_case_count = sample_count.checked_mul(epochs).ok_or_else(|| {
            PublicResultProjectionError::Projection(
                "OpenBench configured case count overflowed".to_string(),
            )
        })?;
        let actual_case_count = u64::try_from(cases.case_count()).map_err(|_| {
            PublicResultProjectionError::Projection(
                "OpenBench public case count exceeded the aggregate protocol domain".to_string(),
            )
        })?;
        if actual_case_count != planned_case_count {
            return Err(PublicResultProjectionError::rejected(
                "OpenBench public cases did not match the factory-reviewed limit and epochs",
            ));
        }
        if u64::try_from(cases.completed_count()).ok() != Some(planned_case_count) {
            return Ok(None);
        }
        if cases.infrastructure_error_count() != 0 || cases.cancelled_count() != 0 {
            return Err(PublicResultProjectionError::rejected(
                "complete OpenBench projection contained an excluded terminal",
            ));
        }
        if candidate.scored_count != sample_count || candidate.unscored_count != 0 {
            return Err(PublicResultProjectionError::rejected(
                "OpenBench aggregate counts did not match configured samples",
            ));
        }

        let mut ones = 0_u64;
        let mut projected_count = 0_u64;
        for (_, projection) in cases.completed_scores(OPENBENCH_GSM8K_PUBLIC_SCORE_NAME) {
            let projection = projection.ok_or_else(|| {
                PublicResultProjectionError::rejected(
                    "a completed OpenBench case omitted its public score projection",
                )
            })?;
            if projection.schema_sha256() != FINITE_BINARY_NUMBER_SCHEMA_SHA256 {
                return Err(PublicResultProjectionError::rejected(
                    "OpenBench score projection used the wrong executable schema",
                ));
            }
            let value = finite_binary_number(projection.value())
                .map_err(PublicResultProjectionError::rejected)?;
            ones = ones.checked_add(value as u64).ok_or_else(|| {
                PublicResultProjectionError::Projection(
                    "OpenBench binary score total overflowed".to_string(),
                )
            })?;
            projected_count = projected_count.checked_add(1).ok_or_else(|| {
                PublicResultProjectionError::Projection(
                    "OpenBench public score count overflowed".to_string(),
                )
            })?;
        }
        if projected_count != planned_case_count {
            return Err(PublicResultProjectionError::rejected(
                "OpenBench score projection set did not match planned cases",
            ));
        }
        let flat_reference = ones as f64 / planned_case_count as f64;
        let provider_value = candidate.value.get();
        let consistent = if flat_reference == 0.0 || flat_reference == 1.0 {
            provider_value == flat_reference
        } else {
            (0.0..=1.0).contains(&provider_value)
                && provider_value.to_bits().abs_diff(flat_reference.to_bits())
                    <= OPENBENCH_GSM8K_MAX_MEAN_ULPS
        };
        if !consistent {
            return Err(PublicResultProjectionError::rejected(
                "OpenBench aggregate exceeded its audited binary-mean ULP bound",
            ));
        }
        Ok(Some(ValidatedPublicAggregateProjection {
            scorer: PUBLIC_ACCURACY_SCORER.to_string(),
            reducer: PUBLIC_ACCURACY_REDUCER.to_string(),
            metric: PUBLIC_ACCURACY_METRIC.to_string(),
            value: candidate.value,
            scored_count: sample_count,
            unscored_count: 0,
            schema_sha256: OPENBENCH_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256.to_string(),
        }))
    }
}

fn openbench_gsm8k_shape(
    safe_config: &CanonicalJson,
) -> Result<(u64, u64), PublicResultProjectionError> {
    let Some(config) = safe_config.value().as_object() else {
        return Err(PublicResultProjectionError::rejected(
            "OpenBench safe config was not an exact object",
        ));
    };
    let exact_fields = ["epochs", "limit", "task", "task_args"];
    if config.len() != exact_fields.len()
        || exact_fields
            .iter()
            .any(|field| !config.contains_key(*field))
        || config.get("task").and_then(Value::as_str) != Some("gsm8k")
        || config
            .get("task_args")
            .and_then(Value::as_object)
            .is_none_or(|value| !value.is_empty())
    {
        return Err(PublicResultProjectionError::rejected(
            "OpenBench safe config drifted from the reviewed GSM8K shape",
        ));
    }
    let sample_count = config.get("limit").and_then(Value::as_u64);
    let epochs = config.get("epochs").and_then(Value::as_u64);
    match (sample_count, epochs) {
        (Some(sample_count @ 1..=5), Some(epochs @ 1..=8)) => Ok((sample_count, epochs)),
        _ => Err(PublicResultProjectionError::rejected(
            "OpenBench safe config exceeded reviewed limit or epoch bounds",
        )),
    }
}

/// Factory policy or provider public-result projection failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicResultProjectionError {
    /// Invalid, duplicated, or descriptor-mismatched factory policy.
    Policy(String),
    /// Registered per-case score validation failed.
    Score(String),
    /// Provider result did not satisfy the registered aggregate closure.
    Projection(String),
}

impl PublicResultProjectionError {
    /// Construct a content-redacted provider result rejection.
    pub fn rejected(message: impl AsRef<str>) -> Self {
        Self::Projection(redact_diagnostic(message.as_ref()))
    }
}

impl Display for PublicResultProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(message) => write!(formatter, "public result policy: {message}"),
            Self::Score(message) => write!(formatter, "public score validation: {message}"),
            Self::Projection(message) => {
                write!(
                    formatter,
                    "public aggregate projection was rejected: {message}"
                )
            }
        }
    }
}

impl std::error::Error for PublicResultProjectionError {}

fn validate_label(value: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(
            "public result policy label was empty, oversized, padded, or contained control text"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{CanonicalJsonLimits, sha256_hex};
    use crate::provider_protocol::{
        CompletedCaseOutcome, EvaluationError, EvaluationStage, ProviderScore,
    };
    use crate::score_projection::{
        FiniteBinaryNumberProjectionValidator, PublicScoreProjectionValidator,
    };

    fn score_policy(name: &str) -> PublicScoreProjectionPolicy {
        let mut policy = PublicScoreProjectionPolicy::restricted_only();
        policy
            .register(name, Arc::new(FiniteBinaryNumberProjectionValidator))
            .unwrap();
        policy
    }

    fn completed(case_id: &str, score_name: &str, value: Option<Value>) -> CaseOutcome {
        CaseOutcome {
            case_id: EvaluationCaseId::new(case_id).unwrap(),
            outcome: CaseOutcomeKind::Completed {
                completed: CompletedCaseOutcome {
                    scores: BTreeMap::from([(
                        score_name.to_string(),
                        ProviderScore {
                            value: CanonicalJson::new(json!({"native": "restricted"})).unwrap(),
                            public_projection: value
                                .map(|value| CanonicalJson::new(value).unwrap()),
                        },
                    )]),
                    numeric_metrics: BTreeMap::new(),
                    primary_score: Some(score_name.to_string()),
                    annotations: None,
                },
            },
            artifact_refs: Vec::new(),
        }
    }

    fn infrastructure(case_id: &str) -> CaseOutcome {
        CaseOutcome {
            case_id: EvaluationCaseId::new(case_id).unwrap(),
            outcome: CaseOutcomeKind::InfrastructureError {
                error: EvaluationError::new(
                    EvaluationStage::new("provider").unwrap(),
                    "provider_error",
                    false,
                    "provider failed",
                )
                .unwrap(),
            },
            artifact_refs: Vec::new(),
        }
    }

    fn cancelled(case_id: &str) -> CaseOutcome {
        CaseOutcome {
            case_id: EvaluationCaseId::new(case_id).unwrap(),
            outcome: CaseOutcomeKind::Cancelled {
                stage: EvaluationStage::new("provider").unwrap(),
                reason: "cancelled".to_string(),
            },
            artifact_refs: Vec::new(),
        }
    }

    fn definition() -> CanonicalJson {
        CanonicalJson::new(json!({
            "exclude_cancelled": true,
            "exclude_infrastructure": true,
        }))
        .unwrap()
    }

    fn openbench_definition() -> CanonicalJson {
        CanonicalJson::new(json!({
            "metric_params": {},
            "params": {},
            "score_name": "grade_school_math_scorer",
        }))
        .unwrap()
    }

    fn safe_config() -> CanonicalJson {
        CanonicalJson::new(json!({})).unwrap()
    }

    fn aggregate(value: f64, scored_count: u64, unscored_count: u64) -> AggregateMetric {
        AggregateMetric {
            scorer: "nemo_evaluator.gsm8k_scorer".to_string(),
            reducer: "mean".to_string(),
            metric: "reward".to_string(),
            value: FiniteF64::new(value).unwrap(),
            scored_count,
            unscored_count,
            definition: definition(),
        }
    }

    fn aggregate_policy() -> PublicAggregateProjectionPolicy {
        let validator = ExactBinaryMeanAggregateValidator::accuracy(
            "nemo_evaluator.gsm8k_scorer",
            "mean",
            "reward",
            definition(),
            NEMO_GSM8K_PUBLIC_SCORE_NAME,
        )
        .unwrap();
        let mut policy = PublicAggregateProjectionPolicy::restricted_only();
        policy
            .register(STOCK_ACCURACY_MEAN_PROJECTION_ID, Arc::new(validator))
            .unwrap();
        policy
    }

    fn openbench_safe_config(sample_count: u64, epochs: u64) -> CanonicalJson {
        CanonicalJson::new(json!({
            "epochs": epochs,
            "limit": sample_count,
            "task": "gsm8k",
            "task_args": {},
        }))
        .unwrap()
    }

    fn openbench_aggregate(value: f64, scored_count: u64, unscored_count: u64) -> AggregateMetric {
        AggregateMetric {
            scorer: "grade_school_math_scorer".to_string(),
            reducer: "identity".to_string(),
            metric: "accuracy".to_string(),
            value: FiniteF64::new(value).unwrap(),
            scored_count,
            unscored_count,
            definition: openbench_definition(),
        }
    }

    fn openbench_outcomes(correct_per_sample: &[u64], epochs: u64) -> Vec<CaseOutcome> {
        let mut outcomes = Vec::new();
        for epoch in 0..epochs {
            for (sample, correct) in correct_per_sample.iter().enumerate() {
                outcomes.push(completed(
                    &format!("case-epoch-{epoch}-sample-{sample}"),
                    OPENBENCH_GSM8K_PUBLIC_SCORE_NAME,
                    Some(json!(u64::from(epoch < *correct))),
                ));
            }
        }
        outcomes
    }

    fn inspect_epoch_mean(correct_per_sample: &[u64], epochs: u64) -> f64 {
        let mut total = 0.0;
        for correct in correct_per_sample {
            total += *correct as f64 / epochs as f64;
        }
        total / correct_per_sample.len() as f64
    }

    #[test]
    fn direct_binary_schema_fingerprint_and_executable_domain_are_exact() {
        let schema = CanonicalJson::new(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "enum": [0, 1],
            "type": "number",
        }))
        .unwrap();
        assert_eq!(
            sha256_hex(&schema.to_bytes()),
            FINITE_BINARY_NUMBER_SCHEMA_SHA256
        );
        let validator = FiniteBinaryNumberProjectionValidator;
        for value in [json!(0), json!(1), json!(0.0), json!(1.0), json!(-0.0)] {
            validator
                .validate(&CanonicalJson::new(value).unwrap())
                .unwrap();
        }
        for value in [
            json!(-1),
            json!(0.5),
            json!(2),
            json!(true),
            json!("1"),
            json!(null),
            json!([]),
            json!({}),
        ] {
            assert!(
                validator
                    .validate(&CanonicalJson::new(value).unwrap())
                    .is_err()
            );
        }
        for bytes in [b"NaN".as_slice(), b"Infinity", b"1e400"] {
            assert!(CanonicalJson::from_slice(bytes, CanonicalJsonLimits::default()).is_err());
        }
    }

    #[test]
    fn case_projection_set_preserves_zero_and_terminal_exclusions() {
        let outcomes = vec![
            completed("case-zero", NEMO_GSM8K_PUBLIC_SCORE_NAME, Some(json!(0.0))),
            completed("case-one", NEMO_GSM8K_PUBLIC_SCORE_NAME, Some(json!(1))),
            infrastructure("case-infra"),
            cancelled("case-cancelled"),
        ];
        let projections = ValidatedPublicCaseProjections::new(
            &outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert_eq!(projections.case_count(), 4);
        assert_eq!(projections.completed_count(), 2);
        assert_eq!(projections.infrastructure_error_count(), 1);
        assert_eq!(projections.cancelled_count(), 1);
        assert_eq!(
            projections
                .scores(&EvaluationCaseId::new("case-zero").unwrap())
                .unwrap()[NEMO_GSM8K_PUBLIC_SCORE_NAME]
                .value()
                .value()
                .as_f64(),
            Some(0.0)
        );
    }

    #[test]
    fn exact_binary_mean_projects_only_reviewed_labels_and_counts() {
        let outcomes = vec![
            completed("case-0", NEMO_GSM8K_PUBLIC_SCORE_NAME, Some(json!(1))),
            completed("case-1", NEMO_GSM8K_PUBLIC_SCORE_NAME, Some(json!(0))),
            completed("case-2", NEMO_GSM8K_PUBLIC_SCORE_NAME, Some(json!(1.0))),
            infrastructure("case-3"),
            cancelled("case-4"),
        ];
        let cases = ValidatedPublicCaseProjections::new(
            &outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        let candidates = vec![
            aggregate(2.0 / 3.0, 3, 2),
            AggregateMetric {
                scorer: "grade_school_math_scorer".to_string(),
                reducer: "identity".to_string(),
                metric: "stderr".to_string(),
                value: FiniteF64::new(0.123).unwrap(),
                scored_count: 3,
                unscored_count: 2,
                definition: CanonicalJson::new(json!({"private": "restricted"})).unwrap(),
            },
        ];
        let projected = aggregate_policy()
            .project(&candidates, &cases, &safe_config())
            .unwrap();
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].scorer(), "accuracy");
        assert_eq!(projected[0].reducer(), "mean");
        assert_eq!(projected[0].metric(), "accuracy");
        assert_eq!(projected[0].value(), 2.0 / 3.0);
        assert_eq!(projected[0].scored_count(), 3);
        assert_eq!(projected[0].unscored_count(), 2);
        assert!(is_sha256(projected[0].schema_sha256()));
    }

    #[test]
    fn completed_zero_is_public_but_an_empty_denominator_is_absent() {
        let zero_outcomes = vec![completed(
            "case-zero",
            NEMO_GSM8K_PUBLIC_SCORE_NAME,
            Some(json!(0)),
        )];
        let zero_cases = ValidatedPublicCaseProjections::new(
            &zero_outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        let zero = aggregate_policy()
            .project(&[aggregate(0.0, 1, 0)], &zero_cases, &safe_config())
            .unwrap();
        assert_eq!(zero.len(), 1);
        assert_eq!(zero[0].value(), 0.0);

        let excluded_outcomes = vec![infrastructure("case-infra"), cancelled("case-cancel")];
        let excluded_cases = ValidatedPublicCaseProjections::new(
            &excluded_outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert!(
            aggregate_policy()
                .project(&[aggregate(0.0, 0, 2)], &excluded_cases, &safe_config(),)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn aggregate_closure_rejects_definition_count_value_and_case_set_drift() {
        let outcomes = vec![
            completed("case-0", NEMO_GSM8K_PUBLIC_SCORE_NAME, Some(json!(1))),
            infrastructure("case-1"),
        ];
        let cases = ValidatedPublicCaseProjections::new(
            &outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        let policy = aggregate_policy();

        let mut wrong_definition = aggregate(1.0, 1, 1);
        wrong_definition.definition = CanonicalJson::new(json!({"reducer": "mean"})).unwrap();
        assert!(
            policy
                .project(&[wrong_definition], &cases, &safe_config())
                .is_err()
        );
        assert!(
            policy
                .project(&[aggregate(1.0, 2, 0)], &cases, &safe_config())
                .is_err()
        );
        assert!(
            policy
                .project(&[aggregate(0.0, 1, 1)], &cases, &safe_config())
                .is_err()
        );

        let missing_outcomes = vec![completed(
            "case-missing",
            NEMO_GSM8K_PUBLIC_SCORE_NAME,
            None,
        )];
        let missing = ValidatedPublicCaseProjections::new(
            &missing_outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert!(
            policy
                .project(&[aggregate(0.0, 1, 0)], &missing, &safe_config())
                .is_err()
        );
    }

    #[test]
    fn aggregate_policy_fingerprints_are_exact_and_candidates_are_unambiguous() {
        let policy = aggregate_policy();
        let fingerprints = policy.schema_fingerprints();
        assert_eq!(fingerprints.len(), 1);
        assert_eq!(
            fingerprints[STOCK_ACCURACY_MEAN_PROJECTION_ID],
            NEMO_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256
        );
        policy
            .validate_descriptor_fingerprints(&fingerprints)
            .unwrap();
        assert!(
            policy
                .validate_descriptor_fingerprints(&BTreeMap::new())
                .is_err()
        );

        let outcomes = vec![completed(
            "case-0",
            NEMO_GSM8K_PUBLIC_SCORE_NAME,
            Some(json!(1)),
        )];
        let cases = ValidatedPublicCaseProjections::new(
            &outcomes,
            &score_policy(NEMO_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert!(
            policy
                .project(
                    &[aggregate(1.0, 1, 0), aggregate(1.0, 1, 0)],
                    &cases,
                    &safe_config(),
                )
                .is_err()
        );
    }

    #[test]
    fn both_stock_aggregate_rules_have_exact_executable_fingerprints() {
        let nemo = ExactBinaryMeanAggregateValidator::accuracy(
            "nemo_evaluator.gsm8k_scorer",
            "mean",
            "reward",
            definition(),
            NEMO_GSM8K_PUBLIC_SCORE_NAME,
        )
        .unwrap();
        assert_eq!(nemo.schema_sha256(), NEMO_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256);

        let openbench = OpenBenchGsm8kAggregateValidator::new();
        assert_eq!(
            openbench.schema_sha256(),
            OPENBENCH_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256
        );
        let openbench_schema = CanonicalJson::new(json!({
            "candidate_counts": {
                "scored_count": "config.limit",
                "unscored_count": 0,
            },
            "comparison": {
                "endpoints": "exact",
                "max_ulps": OPENBENCH_GSM8K_MAX_MEAN_ULPS,
                "reference": "flat_binary_case_mean",
            },
            "configuration": {
                "epochs": {"field": "epochs", "maximum": 8, "minimum": 1},
                "sample_count": {"field": "limit", "maximum": 5, "minimum": 1},
                "task": {"field": "task", "value": "gsm8k"},
                "task_args": {"field": "task_args", "value": {}},
            },
            "denominator": {
                "cancelled": "absent",
                "completed": "all_required",
                "infrastructure_error": "absent",
            },
            "provider": {
                "definition": openbench_definition().value(),
                "metric": "accuracy",
                "reducer": "identity",
                "scorer": "grade_school_math_scorer",
            },
            "public": {
                "metric": "accuracy",
                "reducer": "mean",
                "scorer": "accuracy",
            },
            "schema": OPENBENCH_GSM8K_AGGREGATE_SCHEMA_V1,
            "source_score": {
                "name": OPENBENCH_GSM8K_PUBLIC_SCORE_NAME,
                "schema_sha256": FINITE_BINARY_NUMBER_SCHEMA_SHA256,
            },
        }))
        .unwrap();
        assert_eq!(
            sha256_hex(&openbench_schema.to_bytes()),
            OPENBENCH_GSM8K_ACCURACY_MEAN_SCHEMA_SHA256
        );
    }

    #[test]
    fn openbench_validator_uses_configured_sample_count_and_complete_epoch_topology() {
        let correct_per_sample = [2, 3];
        let epochs = 3;
        let outcomes = openbench_outcomes(&correct_per_sample, epochs);
        let cases = ValidatedPublicCaseProjections::new(
            &outcomes,
            &score_policy(OPENBENCH_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        let config = openbench_safe_config(2, epochs);
        let value = inspect_epoch_mean(&correct_per_sample, epochs);
        let candidate = openbench_aggregate(value, 2, 0);
        let mut policy = PublicAggregateProjectionPolicy::restricted_only();
        policy
            .register(
                STOCK_ACCURACY_MEAN_PROJECTION_ID,
                Arc::new(OpenBenchGsm8kAggregateValidator::new()),
            )
            .unwrap();
        let projected = policy
            .project(std::slice::from_ref(&candidate), &cases, &config)
            .unwrap();
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].value(), value);
        assert_eq!(projected[0].scored_count(), 2);
        assert_eq!(projected[0].unscored_count(), 0);

        let wrong_divisor_count = openbench_aggregate(value, 3, 0);
        assert!(
            policy
                .project(&[wrong_divisor_count], &cases, &config)
                .is_err()
        );
        let flat = 5.0_f64 / 6.0;
        let beyond_bound = f64::from_bits(flat.to_bits() + OPENBENCH_GSM8K_MAX_MEAN_ULPS + 1);
        assert!(
            policy
                .project(&[openbench_aggregate(beyond_bound, 2, 0)], &cases, &config,)
                .is_err()
        );

        let mut incomplete = outcomes;
        incomplete.pop();
        incomplete.push(infrastructure("case-infrastructure"));
        let incomplete = ValidatedPublicCaseProjections::new(
            &incomplete,
            &score_policy(OPENBENCH_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert!(
            policy
                .project(&[candidate], &incomplete, &config)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn openbench_binary_mean_endpoints_are_exact_not_ulp_neighborhoods() {
        let mut policy = PublicAggregateProjectionPolicy::restricted_only();
        policy
            .register(
                STOCK_ACCURACY_MEAN_PROJECTION_ID,
                Arc::new(OpenBenchGsm8kAggregateValidator::new()),
            )
            .unwrap();
        let config = openbench_safe_config(1, 1);

        let zero_outcomes = openbench_outcomes(&[0], 1);
        let zero_cases = ValidatedPublicCaseProjections::new(
            &zero_outcomes,
            &score_policy(OPENBENCH_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert!(
            policy
                .project(&[openbench_aggregate(0.0, 1, 0)], &zero_cases, &config)
                .is_ok()
        );
        assert!(
            policy
                .project(
                    &[openbench_aggregate(f64::from_bits(1), 1, 0)],
                    &zero_cases,
                    &config,
                )
                .is_err()
        );

        let one_outcomes = openbench_outcomes(&[1], 1);
        let one_cases = ValidatedPublicCaseProjections::new(
            &one_outcomes,
            &score_policy(OPENBENCH_GSM8K_PUBLIC_SCORE_NAME),
        )
        .unwrap();
        assert!(
            policy
                .project(&[openbench_aggregate(1.0, 1, 0)], &one_cases, &config)
                .is_ok()
        );
        assert!(
            policy
                .project(
                    &[openbench_aggregate(
                        f64::from_bits(1.0_f64.to_bits() - 1),
                        1,
                        0,
                    )],
                    &one_cases,
                    &config,
                )
                .is_err()
        );
    }

    #[test]
    fn openbench_two_ulp_bound_is_exhaustive_over_stock_config_domain() {
        fn enumerate(counts: &mut [u64], index: usize, epochs: u64, maximum: &mut u64) {
            if index == counts.len() {
                let inspect = inspect_epoch_mean(counts, epochs);
                let flat =
                    counts.iter().sum::<u64>() as f64 / (counts.len() as u64 * epochs) as f64;
                let distance = inspect.to_bits().abs_diff(flat.to_bits());
                assert!(
                    distance <= OPENBENCH_GSM8K_MAX_MEAN_ULPS,
                    "sample_count={}, epochs={epochs}, counts={counts:?}, distance={distance}",
                    counts.len(),
                );
                *maximum = (*maximum).max(distance);
                return;
            }
            for correct in 0..=epochs {
                counts[index] = correct;
                enumerate(counts, index + 1, epochs, maximum);
            }
        }

        let mut maximum = 0;
        for epochs in 1..=8 {
            for sample_count in 1..=5 {
                enumerate(&mut vec![0; sample_count], 0, epochs, &mut maximum);
            }
        }
        assert_eq!(maximum, OPENBENCH_GSM8K_MAX_MEAN_ULPS);
    }
}
