// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic paired-bootstrap statistics for native-plugin parity gates.

use std::{collections::BTreeMap, error::Error, fmt};

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};

const NORMATIVE_BOOTSTRAP_RESAMPLES: usize = 100_000;
const NORMATIVE_RETAINED_PAIRS: usize = 30;
const NORMATIVE_MAX_REPLACEMENTS: usize = 5;
const NORMATIVE_MAX_EXPERIMENT_ATTEMPTS: u8 = 3;
const NORMATIVE_MAX_CV: f64 = 0.02;
const NORMATIVE_MAX_REGRESSION: f64 = 0.01;
const NORMATIVE_CONFIDENCE: f64 = 0.95;

/// One canonical JSONL member measurement.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedSample {
    /// Frozen inventory scenario name.
    pub scenario: String,
    /// Stable identifier shared by the two paired members.
    pub pair_id: String,
    /// Static or dynamic artifact member.
    pub variant: Variant,
    /// Canonical metric name.
    pub metric: String,
    /// Finite, positive member summary.
    pub value: f64,
    /// Canonical metric unit.
    pub unit: String,
    /// Exact source commit used to build the member.
    pub commit: String,
    /// Digest of the measured artifact.
    pub artifact_digest: String,
    /// Digest of the complete experiment authority that produced this member.
    pub experiment_identity_digest: String,
}

/// Artifact member in a paired comparison.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Variant {
    /// Test-only monolithic static comparator.
    Static,
    /// Native-plugin distribution.
    Dynamic,
}

/// Configuration for a one-sided paired non-inferiority gate.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NonInferiorityGate {
    /// Metric evaluated by the gate.
    pub metric: String,
    /// Largest permitted relative regression.
    pub max_relative_regression: f64,
    /// One-sided confidence level.
    pub confidence: f64,
}

impl NonInferiorityGate {
    /// Construct the normative 1%/95% gate for a canonical metric.
    pub fn standard(metric: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
            max_relative_regression: NORMATIVE_MAX_REGRESSION,
            confidence: NORMATIVE_CONFIDENCE,
        }
    }
}

/// Deterministic report for one metric's paired gate.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GateReport {
    /// Canonical metric name.
    pub metric: String,
    /// Direction used to form every positive pair ratio.
    pub ratio_direction: RatioDirection,
    /// Ratios in lexicographic pair-ID order.
    pub paired_ratios: Vec<f64>,
    /// Arithmetic mean of paired ratios.
    pub observed_ratio: f64,
    /// Deterministic paired-bootstrap means.
    pub bootstrap_distribution: Vec<f64>,
    /// Hyndman-Fan type-7 one-sided lower endpoint.
    pub lower_confidence_bound: f64,
    /// Required lower endpoint.
    pub threshold: f64,
    /// Whether the one-sided lower endpoint meets the threshold.
    pub passed: bool,
}

/// Numerator and denominator used for a canonical performance ratio.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RatioDirection {
    /// Throughput is dynamic divided by static.
    DynamicOverStatic,
    /// Latency, CPU, and exporter duration are static divided by dynamic.
    StaticOverDynamic,
}

/// One normative metric and its fixed comparison direction.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormativeMetric {
    /// Canonical metric name.
    pub metric: String,
    /// Direction fixed by the performance contract.
    pub direction: RatioDirection,
}

/// Complete normative metric inventory for one benchmark case.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormativeCase {
    /// Canonical benchmark case name.
    pub scenario: String,
    /// The one legal primary metric for this case.
    pub primary_metric: String,
    /// Exact complete metric set, sorted by canonical metric name.
    pub metrics: Vec<NormativeMetric>,
}

/// Authenticated complete `(component, case, metric, direction)` inventory.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormativeInventory {
    /// Inventory schema version.
    pub schema_version: u32,
    /// Migrated component governed by this inventory.
    pub component: String,
    /// Exact complete case set, sorted by scenario.
    pub cases: Vec<NormativeCase>,
    /// BLAKE3 digest of the preceding canonical fields.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub digest: String,
}

impl NormativeInventory {
    /// Construct and authenticate a canonical complete inventory.
    pub fn new(
        component: impl Into<String>,
        mut cases: Vec<NormativeCase>,
    ) -> Result<Self, PluginStatsError> {
        cases.sort_by(|left, right| left.scenario.cmp(&right.scenario));
        for case in &mut cases {
            case.metrics
                .sort_by(|left, right| left.metric.cmp(&right.metric));
        }
        let mut inventory = Self {
            schema_version: 1,
            component: component.into(),
            cases,
            digest: String::new(),
        };
        validate_inventory_shape(&inventory)?;
        inventory.digest = inventory.computed_digest()?;
        Ok(inventory)
    }

    fn computed_digest(&self) -> Result<String, PluginStatsError> {
        let mut canonical = self.clone();
        canonical.digest.clear();
        canonical_blake3(&canonical, "normative inventory")
    }
}

/// One exact retained pair and its seeded member order.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairSchedule {
    /// Canonical pair identifier.
    pub pair_id: String,
    /// Exact AB/BA order for this pair.
    pub member_order: [Variant; 2],
}

/// Complete authenticated identity for one parity experiment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentIdentity {
    /// Identity schema version.
    pub schema_version: u32,
    /// Exact source commit shared by both compared artifacts.
    pub source_commit: String,
    /// Complete source-tree digest.
    pub source_tree_digest: String,
    /// Exact Cargo.lock digest.
    pub cargo_lock_digest: String,
    /// Exact compiler identity.
    pub rustc: String,
    /// Exact compiler sysroot digest.
    pub sysroot_digest: String,
    /// Rust compilation target.
    pub target: String,
    /// Optimized profile identity.
    pub profile: String,
    /// Expected static comparator artifact digest.
    pub static_artifact_digest: String,
    /// Expected dynamic plugin artifact digest.
    pub dynamic_artifact_digest: String,
    /// Benchmark harness artifact digest.
    pub harness_artifact_digest: String,
    /// Mock-server artifact digest.
    pub mock_server_artifact_digest: String,
    /// Authenticated complete normative inventory digest.
    pub inventory_digest: String,
    /// CPU model identity.
    pub cpu_model: String,
    /// CPU stepping identity.
    pub cpu_stepping: String,
    /// CPU microcode identity.
    pub microcode: String,
    /// Core topology identity.
    pub core_topology: String,
    /// Memory topology identity.
    pub memory_topology: String,
    /// Firmware identity.
    pub firmware: String,
    /// Kernel identity.
    pub kernel: String,
    /// Allocator/provider identity.
    pub allocator_provider: String,
    /// CPU frequency/governor identity.
    pub frequency_governor: String,
    /// Affinity and isolation identity.
    pub affinity_isolation: String,
    /// Mock-server placement identity.
    pub mock_server_placement: String,
    /// Every environment value admitted by the harness.
    pub environment: BTreeMap<String, String>,
    /// Seed governing both the exact pair schedule and bootstrap.
    pub bootstrap_seed: u64,
    /// Complete exact 30-pair AB/BA schedule.
    pub pair_schedule: Vec<PairSchedule>,
    /// BLAKE3 digest of every preceding identity field.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub identity_digest: String,
}

impl ExperimentIdentity {
    /// Validate, digest, and seal a complete experiment identity.
    pub fn seal(mut self) -> Result<Self, PluginStatsError> {
        validate_experiment_identity_shape(&self)?;
        self.identity_digest = self.computed_digest()?;
        Ok(self)
    }

    fn computed_digest(&self) -> Result<String, PluginStatsError> {
        let mut canonical = self.clone();
        canonical.identity_digest.clear();
        canonical_blake3(&canonical, "experiment identity")
    }
}

/// One authoritative simultaneous-gate input document.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimultaneousGateInput {
    /// Complete experiment and artifact identity.
    pub experiment_identity: ExperimentIdentity,
    /// Exact samples for every inventory case and metric.
    pub cases: Vec<PairedCase>,
}

/// All retained measurements and invalidated raw attempts for one scenario.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedCase {
    /// Frozen inventory scenario name.
    pub scenario: String,
    /// The scenario's legal primary metric.
    pub primary_metric: String,
    /// Retained static/dynamic members for primary and secondary metrics.
    pub samples: Vec<PairedSample>,
    /// Raw infrastructure-invalid pairs replaced before retention.
    pub invalidation_attempts: Vec<InvalidationAttempt>,
}

/// One raw pair discarded only by the frozen infrastructure classifier.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InvalidationAttempt {
    /// Retained replacement pair identifier.
    pub pair_id: String,
    /// Complete experiment attempt containing this pair.
    pub experiment_attempt: u8,
    /// One-based replacement ordinal within the case.
    pub replacement_ordinal: usize,
    /// Original member order that the replacement must preserve.
    pub member_order: [Variant; 2],
    /// Both raw member measurements from the discarded pair.
    pub members: Vec<PairedSample>,
    /// Blinded infrastructure-classifier reason.
    pub reason: String,
    /// Classification proving this was not a product failure.
    pub disposition: AttemptDisposition,
}

/// Classification of an attempted pair.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    /// Harness-owned infrastructure failure eligible for replacement.
    InfrastructureInvalid,
    /// Valid product failure that cannot be retried away.
    ProductFailure,
}

/// Normative simultaneous-gate settings.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimultaneousGatePolicy {
    confidence: f64,
    max_relative_regression: f64,
    bootstrap_resamples: usize,
    retained_pairs: usize,
    max_coefficient_of_variation: f64,
    max_replacement_pairs: usize,
    max_experiment_attempts: u8,
}

impl SimultaneousGatePolicy {
    /// Return the immutable generation-1 performance policy.
    pub fn normative() -> Self {
        Self {
            confidence: NORMATIVE_CONFIDENCE,
            max_relative_regression: NORMATIVE_MAX_REGRESSION,
            bootstrap_resamples: NORMATIVE_BOOTSTRAP_RESAMPLES,
            retained_pairs: NORMATIVE_RETAINED_PAIRS,
            max_coefficient_of_variation: NORMATIVE_MAX_CV,
            max_replacement_pairs: NORMATIVE_MAX_REPLACEMENTS,
            max_experiment_attempts: NORMATIVE_MAX_EXPERIMENT_ATTEMPTS,
        }
    }
}

impl Default for SimultaneousGatePolicy {
    fn default() -> Self {
        Self::normative()
    }
}

/// Per-case/per-metric vectors and simultaneous lower endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimultaneousMetricReport {
    /// Scenario owning this metric.
    pub scenario: String,
    /// Canonical metric name.
    pub metric: String,
    /// Ratio direction fixed by the metric contract.
    pub ratio_direction: RatioDirection,
    /// Thirty static member summaries.
    pub static_summaries: Vec<f64>,
    /// Thirty dynamic member summaries.
    pub dynamic_summaries: Vec<f64>,
    /// Thirty positive paired ratios.
    pub positive_paired_ratios: Vec<f64>,
    /// Bessel-corrected static-member coefficient of variation.
    pub static_coefficient_of_variation: f64,
    /// Bessel-corrected dynamic-member coefficient of variation.
    pub dynamic_coefficient_of_variation: f64,
    /// Bessel-corrected paired-ratio coefficient of variation.
    pub ratio_coefficient_of_variation: f64,
    /// Arithmetic mean of the paired ratios.
    pub observed_ratio: f64,
    /// Simultaneous one-sided lower endpoint.
    pub lower_confidence_bound: f64,
    /// Required lower endpoint.
    pub threshold: f64,
    /// Whether this bound meets the threshold.
    pub passed: bool,
}

/// Joint report across the complete case/metric matrix.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimultaneousGateReport {
    /// Component whose complete normative inventory was evaluated.
    pub component: String,
    /// Authenticated normative inventory digest.
    pub inventory_digest: String,
    /// Complete experiment identity digest shared by every sample.
    pub experiment_identity_digest: String,
    /// Seed used to generate the schedule and joint bootstrap.
    pub bootstrap_seed: u64,
    /// Complete exact retained AB/BA schedule.
    pub pair_schedule: Vec<PairSchedule>,
    /// Expected static comparator artifact digest.
    pub static_artifact_digest: String,
    /// Expected dynamic plugin artifact digest.
    pub dynamic_artifact_digest: String,
    /// Reports in scenario then metric order.
    pub metric_reports: Vec<SimultaneousMetricReport>,
    /// Deterministic maximum-degradation value for every paired resample.
    pub maximum_degradation_bootstrap_distribution: Vec<f64>,
    /// Every replaced raw pair, including both members and its reason.
    pub invalidation_attempts: Vec<InvalidationAttempt>,
    /// True only when a noise or protocol rule invalidates the attempt.
    pub is_invalid: bool,
    /// Stable invalidation diagnosis when the attempt is invalid.
    pub invalidation_reason: Option<String>,
    /// True only when the valid joint gate passes every simultaneous bound.
    pub passed: bool,
}

/// Outcome of one complete experiment attempt.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentAttempt {
    /// One-based attempt ordinal.
    pub ordinal: u8,
    /// Terminal experiment outcome.
    pub outcome: ExperimentOutcome,
    /// Required diagnosis for invalid or valid-failure attempts.
    pub reason: Option<String>,
}

impl ExperimentAttempt {
    /// Construct an infrastructure-invalid attempt eligible for a retry.
    pub fn invalid(ordinal: u8, reason: impl Into<String>) -> Self {
        Self {
            ordinal,
            outcome: ExperimentOutcome::Invalid,
            reason: Some(reason.into()),
        }
    }

    /// Construct an authoritative valid failure.
    pub fn valid_failure(ordinal: u8, reason: impl Into<String>) -> Self {
        Self {
            ordinal,
            outcome: ExperimentOutcome::ValidFailure,
            reason: Some(reason.into()),
        }
    }

    /// Construct an authoritative valid pass.
    pub fn valid_pass(ordinal: u8) -> Self {
        Self {
            ordinal,
            outcome: ExperimentOutcome::ValidPass,
            reason: None,
        }
    }
}

/// Terminal classification of a complete experiment attempt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentOutcome {
    /// Noise-protocol invalidation; retry is permitted within the cap.
    Invalid,
    /// Statistically valid authoritative pass.
    ValidPass,
    /// Statistically valid authoritative failure.
    ValidFailure,
}

/// Frozen exporter member construction parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExporterSampleContract {
    /// Records in the one deterministic corpus.
    pub corpus_records: u64,
    /// Sequential exact-pass repetitions in one retained member.
    pub sample_repetitions: usize,
    /// Total processed records used as the duration divisor.
    pub processed_records: u64,
    /// Records in the sole retained artifact.
    pub retained_artifact_records: u64,
}

impl ExporterSampleContract {
    /// Return the immutable generation-1 exporter sample contract.
    pub fn normative() -> Self {
        Self {
            corpus_records: 100_000,
            sample_repetitions: 16,
            processed_records: 1_600_000,
            retained_artifact_records: 100_000,
        }
    }
}

impl Default for ExporterSampleContract {
    fn default() -> Self {
        Self::normative()
    }
}

/// Receipt for one active exporter pass.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExporterRepetition {
    /// Zero-based sequential repetition ordinal.
    pub ordinal: usize,
    /// Records emitted by this repetition.
    pub emitted_records: u64,
    /// Digest of the complete output from this repetition.
    pub output_digest: String,
    /// Active write-and-flush duration, excluding all gaps.
    pub active_duration_nanoseconds: u64,
}

/// Validated exporter member summary.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExporterSampleSummary {
    /// Sum of the sixteen active repetition durations.
    pub active_duration_nanoseconds: u64,
    /// Exact processed-record divisor.
    pub processed_records: u64,
    /// Records in the sole retained artifact.
    pub retained_artifact_records: u64,
    /// Active nanoseconds divided by processed records.
    pub exporter_nanoseconds_per_record: f64,
    /// Common output digest across all repetitions.
    pub output_digest: String,
    /// Validated per-repetition receipts.
    pub repetitions: Vec<ExporterRepetition>,
}

/// Typed refusal from the canonical statistical harness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginStatsError {
    message: String,
}

impl PluginStatsError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PluginStatsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for PluginStatsError {}

/// Encode samples as compact JSONL in their recorded member order.
pub fn encode_samples_jsonl(samples: &[PairedSample]) -> Result<Vec<u8>, PluginStatsError> {
    let mut output = Vec::new();
    for sample in samples {
        validate_sample_identity(sample)?;
        serde_json::to_writer(&mut output, sample).map_err(|error| {
            PluginStatsError::new(format!("cannot serialize canonical sample: {error}"))
        })?;
        output.push(b'\n');
    }
    Ok(output)
}

/// Decode canonical JSONL and reject blank or noncanonical records.
pub fn decode_samples_jsonl(bytes: &[u8]) -> Result<Vec<PairedSample>, PluginStatsError> {
    if bytes.is_empty() {
        return Err(PluginStatsError::new("sample JSONL is empty"));
    }
    let mut samples = Vec::new();
    let lines = bytes.split(|byte| *byte == b'\n').collect::<Vec<_>>();
    for (line_index, line) in lines.iter().enumerate() {
        if line.is_empty() {
            if line_index + 1 == lines.len() {
                continue;
            }
            return Err(PluginStatsError::new(
                "sample JSONL contains a blank record",
            ));
        }
        let sample: PairedSample = serde_json::from_slice(line).map_err(|error| {
            PluginStatsError::new(format!("invalid sample JSONL record: {error}"))
        })?;
        validate_sample_identity(&sample)?;
        let canonical = serde_json::to_vec(&sample).map_err(|error| {
            PluginStatsError::new(format!("cannot canonicalize sample JSONL record: {error}"))
        })?;
        if canonical != *line {
            return Err(PluginStatsError::new(
                "sample JSONL record is not in canonical field order or encoding",
            ));
        }
        samples.push(sample);
    }
    if samples.is_empty() {
        return Err(PluginStatsError::new("sample JSONL contains no records"));
    }
    Ok(samples)
}

#[derive(Clone)]
struct MetricVector {
    scenario: String,
    metric: String,
    direction: RatioDirection,
    static_values: Vec<f64>,
    dynamic_values: Vec<f64>,
    ratios: Vec<f64>,
    pair_ids: Vec<String>,
    pair_orders: Vec<[Variant; 2]>,
}

/// Generate the reproducible balanced AB/BA order for thirty retained pairs.
pub fn balanced_pair_orders(bootstrap_seed: u64) -> Vec<[Variant; 2]> {
    let mut orders = Vec::with_capacity(NORMATIVE_RETAINED_PAIRS);
    orders.extend(std::iter::repeat_n([Variant::Static, Variant::Dynamic], 15));
    orders.extend(std::iter::repeat_n([Variant::Dynamic, Variant::Static], 15));
    orders.shuffle(&mut Pcg64Mcg::seed_from_u64(bootstrap_seed));
    orders
}

/// Evaluate one metric with a deterministic one-sided paired bootstrap.
pub fn evaluate_paired_gate(
    samples: &[PairedSample],
    gate: &NonInferiorityGate,
    bootstrap_seed: u64,
) -> Result<GateReport, PluginStatsError> {
    validate_gate(gate)?;
    let direction = metric_direction(&gate.metric)?.0;
    let vector = collect_metric(samples, &gate.metric, direction)?;
    let observed_ratio = arithmetic_mean(&vector.ratios)?;
    let bootstrap_distribution = bootstrap_means(
        &vector.ratios,
        NORMATIVE_BOOTSTRAP_RESAMPLES,
        bootstrap_seed,
    )?;
    let lower_confidence_bound = type_7_quantile(&bootstrap_distribution, 1.0 - gate.confidence)?;
    let threshold = 1.0 - gate.max_relative_regression;
    Ok(GateReport {
        metric: gate.metric.clone(),
        ratio_direction: direction,
        paired_ratios: vector.ratios,
        observed_ratio,
        bootstrap_distribution,
        lower_confidence_bound,
        threshold,
        passed: lower_confidence_bound >= threshold,
    })
}

/// Evaluate the complete case/metric matrix with one max-degradation bootstrap.
pub fn evaluate_simultaneous_gate(
    input: &SimultaneousGateInput,
    inventory: &NormativeInventory,
    expected_inventory_digest: &str,
    policy: &SimultaneousGatePolicy,
) -> Result<SimultaneousGateReport, PluginStatsError> {
    validate_policy(policy)?;
    validate_inventory(inventory, expected_inventory_digest)?;
    validate_experiment_identity(&input.experiment_identity, inventory)?;
    if input
        .cases
        .iter()
        .flat_map(|case| &case.samples)
        .any(|sample| {
            sample.experiment_identity_digest != input.experiment_identity.identity_digest
        })
    {
        return Err(PluginStatsError::new(
            "a sample row is bound to a different experiment identity",
        ));
    }

    let cases_by_scenario = input
        .cases
        .iter()
        .map(|case| (case.scenario.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    if cases_by_scenario.len() != input.cases.len() {
        return Err(PluginStatsError::new(
            "simultaneous gate contains a duplicate scenario",
        ));
    }
    let expected_scenarios = inventory
        .cases
        .iter()
        .map(|case| case.scenario.as_str())
        .collect::<Vec<_>>();
    if cases_by_scenario.keys().copied().collect::<Vec<_>>() != expected_scenarios {
        return Err(PluginStatsError::new(
            "supplied case set differs from the authenticated normative inventory",
        ));
    }

    let mut vectors = Vec::new();
    let mut invalidation_attempts = Vec::new();
    for normative_case in &inventory.cases {
        let case = cases_by_scenario[normative_case.scenario.as_str()];
        if case.primary_metric != normative_case.primary_metric {
            return Err(PluginStatsError::new(format!(
                "case {} primary metric differs from the authenticated inventory",
                case.scenario
            )));
        }
        if case
            .samples
            .iter()
            .any(|sample| sample.scenario != case.scenario)
        {
            return Err(PluginStatsError::new(format!(
                "case {} contains a sample from another scenario",
                case.scenario
            )));
        }
        let mut metrics = case
            .samples
            .iter()
            .map(|sample| sample.metric.clone())
            .collect::<Vec<_>>();
        metrics.sort();
        metrics.dedup();
        let expected_metrics = normative_case
            .metrics
            .iter()
            .map(|metric| metric.metric.clone())
            .collect::<Vec<_>>();
        if metrics != expected_metrics {
            return Err(PluginStatsError::new(format!(
                "case {} metric set differs from the authenticated normative inventory",
                case.scenario
            )));
        }
        for sample in &case.samples {
            validate_sample_against_identity(sample, &input.experiment_identity)?;
        }
        for normative_metric in &normative_case.metrics {
            let vector = collect_metric(
                &case.samples,
                &normative_metric.metric,
                normative_metric.direction,
            )?;
            validate_exact_schedule(&vector, &input.experiment_identity.pair_schedule)?;
            vectors.push(vector);
        }
        validate_invalidations(case, policy, &input.experiment_identity)?;
        invalidation_attempts.extend(case.invalidation_attempts.iter().cloned());
    }
    vectors.sort_by(|left, right| {
        (&left.scenario, &left.metric).cmp(&(&right.scenario, &right.metric))
    });

    let observed = vectors
        .iter()
        .map(|vector| arithmetic_mean(&vector.ratios))
        .collect::<Result<Vec<_>, _>>()?;
    let reference_pair_ids = &vectors[0].pair_ids;
    let reference_pair_orders = &vectors[0].pair_orders;
    if vectors.iter().any(|vector| {
        vector.pair_ids != *reference_pair_ids || vector.pair_orders != *reference_pair_orders
    }) {
        return Err(PluginStatsError::new(
            "all normative cases and metrics must share the exact retained pair schedule",
        ));
    }

    let mut rng = Pcg64Mcg::seed_from_u64(input.experiment_identity.bootstrap_seed);
    let mut maximum_degradation_bootstrap_distribution =
        Vec::with_capacity(policy.bootstrap_resamples);
    let mut resampled_totals = vec![0.0; vectors.len()];
    for _ in 0..policy.bootstrap_resamples {
        resampled_totals.fill(0.0);
        let mut maximum_degradation = 0.0_f64;
        let pair_count = vectors[0].ratios.len();
        for _ in 0..pair_count {
            let pair_index = rng.random_range(0..pair_count);
            for (index, vector) in vectors.iter().enumerate() {
                resampled_totals[index] += vector.ratios[pair_index];
            }
        }
        for index in 0..vectors.len() {
            let resampled_ratio = resampled_totals[index] / pair_count as f64;
            if !resampled_ratio.is_finite() {
                return Err(PluginStatsError::new(
                    "simultaneous bootstrap produced a non-finite ratio",
                ));
            }
            maximum_degradation = maximum_degradation.max(observed[index] - resampled_ratio);
        }
        maximum_degradation_bootstrap_distribution.push(maximum_degradation);
    }
    let critical_degradation = type_7_quantile(
        &maximum_degradation_bootstrap_distribution,
        policy.confidence,
    )?;
    let threshold = 1.0 - policy.max_relative_regression;
    let mut noise_reasons = Vec::new();
    let mut metric_reports = Vec::with_capacity(vectors.len());
    for (vector, observed_ratio) in vectors.into_iter().zip(observed) {
        let static_cv = coefficient_of_variation(&vector.static_values)?;
        let dynamic_cv = coefficient_of_variation(&vector.dynamic_values)?;
        let ratio_cv = coefficient_of_variation(&vector.ratios)?;
        for (label, value) in [
            ("static", static_cv),
            ("dynamic", dynamic_cv),
            ("paired ratio", ratio_cv),
        ] {
            if value > policy.max_coefficient_of_variation {
                noise_reasons.push(format!(
                    "{}/{} {label} coefficient of variation {value:.9} exceeds {:.9}",
                    vector.scenario, vector.metric, policy.max_coefficient_of_variation
                ));
            }
        }
        let lower_confidence_bound = observed_ratio - critical_degradation;
        metric_reports.push(SimultaneousMetricReport {
            scenario: vector.scenario,
            metric: vector.metric,
            ratio_direction: vector.direction,
            static_summaries: vector.static_values,
            dynamic_summaries: vector.dynamic_values,
            positive_paired_ratios: vector.ratios,
            static_coefficient_of_variation: static_cv,
            dynamic_coefficient_of_variation: dynamic_cv,
            ratio_coefficient_of_variation: ratio_cv,
            observed_ratio,
            lower_confidence_bound,
            threshold,
            passed: lower_confidence_bound >= threshold,
        });
    }
    let is_invalid = !noise_reasons.is_empty();
    let passed = !is_invalid && metric_reports.iter().all(|report| report.passed);
    Ok(SimultaneousGateReport {
        component: inventory.component.clone(),
        inventory_digest: inventory.digest.clone(),
        experiment_identity_digest: input.experiment_identity.identity_digest.clone(),
        bootstrap_seed: input.experiment_identity.bootstrap_seed,
        pair_schedule: input.experiment_identity.pair_schedule.clone(),
        static_artifact_digest: input.experiment_identity.static_artifact_digest.clone(),
        dynamic_artifact_digest: input.experiment_identity.dynamic_artifact_digest.clone(),
        metric_reports,
        maximum_degradation_bootstrap_distribution,
        invalidation_attempts,
        is_invalid,
        invalidation_reason: is_invalid.then(|| noise_reasons.join("; ")),
        passed,
    })
}

/// Enforce the three-attempt rule and first-valid-attempt authority.
pub fn validate_experiment_attempts(
    attempts: &[ExperimentAttempt],
) -> Result<(), PluginStatsError> {
    if attempts.is_empty() {
        return Err(PluginStatsError::new("experiment attempt history is empty"));
    }
    if attempts.len() > usize::from(NORMATIVE_MAX_EXPERIMENT_ATTEMPTS) {
        return Err(PluginStatsError::new("more than three experiment attempts"));
    }
    let mut terminal_valid = false;
    for (index, attempt) in attempts.iter().enumerate() {
        let expected = u8::try_from(index + 1)
            .map_err(|_| PluginStatsError::new("experiment attempt ordinal overflow"))?;
        if attempt.ordinal != expected {
            return Err(PluginStatsError::new(
                "experiment attempts are not contiguous",
            ));
        }
        if terminal_valid {
            return Err(PluginStatsError::new(
                "a valid pass or failure cannot be replaced by a later attempt",
            ));
        }
        match attempt.outcome {
            ExperimentOutcome::Invalid => {
                if attempt.reason.as_deref().is_none_or(str::is_empty) {
                    return Err(PluginStatsError::new(
                        "an invalid experiment attempt requires a diagnosis",
                    ));
                }
            }
            ExperimentOutcome::ValidFailure => {
                if attempt.reason.as_deref().is_none_or(str::is_empty) {
                    return Err(PluginStatsError::new("a valid failure requires a reason"));
                }
                terminal_valid = true;
            }
            ExperimentOutcome::ValidPass => {
                if attempt.reason.is_some() {
                    return Err(PluginStatsError::new(
                        "a valid pass cannot carry a retry reason",
                    ));
                }
                terminal_valid = true;
            }
        }
    }
    Ok(())
}

/// Validate and summarize the exact 16-pass exporter member construction.
pub fn evaluate_exporter_sample(
    contract: &ExporterSampleContract,
    repetitions: &[ExporterRepetition],
) -> Result<ExporterSampleSummary, PluginStatsError> {
    if contract != &ExporterSampleContract::normative() {
        return Err(PluginStatsError::new(
            "exporter sample parameters are an immutable performance contract",
        ));
    }
    if repetitions.len() != contract.sample_repetitions {
        return Err(PluginStatsError::new(
            "exporter member must contain 16 repetitions",
        ));
    }
    let first_digest = repetitions
        .first()
        .map(|repetition| repetition.output_digest.as_str())
        .ok_or_else(|| PluginStatsError::new("exporter repetitions are empty"))?;
    if first_digest.is_empty() {
        return Err(PluginStatsError::new("exporter output digest is empty"));
    }
    if !is_blake3_digest(first_digest) {
        return Err(PluginStatsError::new(
            "exporter output digest is not canonical BLAKE3",
        ));
    }
    let mut active_duration_nanoseconds = 0_u64;
    for (ordinal, repetition) in repetitions.iter().enumerate() {
        if repetition.ordinal != ordinal {
            return Err(PluginStatsError::new(
                "exporter repetitions are not sequential",
            ));
        }
        if repetition.emitted_records != contract.corpus_records {
            return Err(PluginStatsError::new(
                "each exporter repetition must emit exactly 100000 records",
            ));
        }
        if repetition.output_digest != first_digest {
            return Err(PluginStatsError::new(
                "exporter repetition output digests differ",
            ));
        }
        if repetition.active_duration_nanoseconds == 0 {
            return Err(PluginStatsError::new(
                "exporter active repetition duration must be positive",
            ));
        }
        active_duration_nanoseconds = active_duration_nanoseconds
            .checked_add(repetition.active_duration_nanoseconds)
            .ok_or_else(|| PluginStatsError::new("exporter active duration overflow"))?;
    }
    if active_duration_nanoseconds < 30_000_000_000 {
        return Err(PluginStatsError::new(
            "summed active exporter duration is less than 30 seconds",
        ));
    }
    let exporter_nanoseconds_per_record =
        active_duration_nanoseconds as f64 / contract.processed_records as f64;
    Ok(ExporterSampleSummary {
        active_duration_nanoseconds,
        processed_records: contract.processed_records,
        retained_artifact_records: contract.retained_artifact_records,
        exporter_nanoseconds_per_record,
        output_digest: first_digest.to_owned(),
        repetitions: repetitions.to_vec(),
    })
}

fn validate_gate(gate: &NonInferiorityGate) -> Result<(), PluginStatsError> {
    metric_direction(&gate.metric)?;
    if !gate.max_relative_regression.is_finite()
        || !(0.0..1.0).contains(&gate.max_relative_regression)
    {
        return Err(PluginStatsError::new("invalid maximum relative regression"));
    }
    if !gate.confidence.is_finite() || !(0.0..1.0).contains(&gate.confidence) {
        return Err(PluginStatsError::new("invalid confidence"));
    }
    Ok(())
}

fn validate_policy(policy: &SimultaneousGatePolicy) -> Result<(), PluginStatsError> {
    if policy != &SimultaneousGatePolicy::normative() {
        return Err(PluginStatsError::new(
            "simultaneous policy differs from the immutable generation-1 contract",
        ));
    }
    Ok(())
}

fn canonical_blake3<T: Serialize>(value: &T, label: &str) -> Result<String, PluginStatsError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| PluginStatsError::new(format!("cannot encode {label}: {error}")))?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

fn validate_inventory_shape(inventory: &NormativeInventory) -> Result<(), PluginStatsError> {
    if inventory.schema_version != 1 || inventory.component.is_empty() || inventory.cases.is_empty()
    {
        return Err(PluginStatsError::new(
            "normative inventory identity is incomplete",
        ));
    }
    if inventory
        .cases
        .windows(2)
        .any(|pair| pair[0].scenario >= pair[1].scenario)
    {
        return Err(PluginStatsError::new(
            "normative inventory cases are not uniquely sorted",
        ));
    }
    for case in &inventory.cases {
        if case.scenario.is_empty() || case.primary_metric.is_empty() || case.metrics.is_empty() {
            return Err(PluginStatsError::new(
                "normative inventory case is incomplete",
            ));
        }
        if case
            .metrics
            .windows(2)
            .any(|pair| pair[0].metric >= pair[1].metric)
        {
            return Err(PluginStatsError::new(
                "normative inventory metrics are not uniquely sorted",
            ));
        }
        let mut has_primary = false;
        for metric in &case.metrics {
            let (expected_direction, is_primary) = metric_direction(&metric.metric)?;
            if expected_direction != metric.direction {
                return Err(PluginStatsError::new(format!(
                    "inventory direction for {}/{} differs from the metric contract",
                    case.scenario, metric.metric
                )));
            }
            if metric.metric == case.primary_metric {
                if !is_primary {
                    return Err(PluginStatsError::new(format!(
                        "{} is a secondary metric and cannot be primary",
                        metric.metric
                    )));
                }
                has_primary = true;
            }
        }
        if !has_primary {
            return Err(PluginStatsError::new(format!(
                "case {} omits its primary metric {}",
                case.scenario, case.primary_metric
            )));
        }
    }
    Ok(())
}

fn validate_inventory(
    inventory: &NormativeInventory,
    expected_inventory_digest: &str,
) -> Result<(), PluginStatsError> {
    validate_inventory_shape(inventory)?;
    if !is_blake3_digest(expected_inventory_digest)
        || !is_blake3_digest(&inventory.digest)
        || inventory.computed_digest()? != inventory.digest
        || inventory.digest != expected_inventory_digest
    {
        return Err(PluginStatsError::new(
            "normative inventory does not match the independently bound expected digest",
        ));
    }
    Ok(())
}

fn validate_experiment_identity_shape(
    identity: &ExperimentIdentity,
) -> Result<(), PluginStatsError> {
    let required_strings = [
        identity.source_commit.as_str(),
        identity.rustc.as_str(),
        identity.target.as_str(),
        identity.profile.as_str(),
        identity.cpu_model.as_str(),
        identity.cpu_stepping.as_str(),
        identity.microcode.as_str(),
        identity.core_topology.as_str(),
        identity.memory_topology.as_str(),
        identity.firmware.as_str(),
        identity.kernel.as_str(),
        identity.allocator_provider.as_str(),
        identity.frequency_governor.as_str(),
        identity.affinity_isolation.as_str(),
        identity.mock_server_placement.as_str(),
    ];
    if identity.schema_version != 1
        || required_strings.iter().any(|value| value.is_empty())
        || !is_lower_hex(&identity.source_commit, 40)
    {
        return Err(PluginStatsError::new(
            "experiment identity has an omitted or invalid scalar field",
        ));
    }
    for digest in [
        &identity.source_tree_digest,
        &identity.cargo_lock_digest,
        &identity.sysroot_digest,
        &identity.static_artifact_digest,
        &identity.dynamic_artifact_digest,
        &identity.harness_artifact_digest,
        &identity.mock_server_artifact_digest,
        &identity.inventory_digest,
    ] {
        if !is_blake3_digest(digest) {
            return Err(PluginStatsError::new(
                "experiment identity has a noncanonical digest",
            ));
        }
    }
    if identity.environment.is_empty()
        || identity.environment.iter().any(|(name, _)| name.is_empty())
    {
        return Err(PluginStatsError::new(
            "experiment identity omits the admitted environment",
        ));
    }
    let expected_orders = balanced_pair_orders(identity.bootstrap_seed);
    if identity.pair_schedule.len() != NORMATIVE_RETAINED_PAIRS
        || identity
            .pair_schedule
            .iter()
            .zip(expected_orders)
            .enumerate()
            .any(|(pair, (scheduled, expected_order))| {
                scheduled.pair_id != format!("pair-{pair:02}")
                    || scheduled.member_order != expected_order
            })
    {
        return Err(PluginStatsError::new(
            "experiment identity does not contain the seeded exact 30-pair schedule",
        ));
    }
    Ok(())
}

fn validate_experiment_identity(
    identity: &ExperimentIdentity,
    inventory: &NormativeInventory,
) -> Result<(), PluginStatsError> {
    validate_experiment_identity_shape(identity)?;
    if identity.inventory_digest != inventory.digest {
        return Err(PluginStatsError::new(
            "experiment identity is bound to a different normative inventory",
        ));
    }
    if !is_blake3_digest(&identity.identity_digest)
        || identity.computed_digest()? != identity.identity_digest
    {
        return Err(PluginStatsError::new(
            "experiment identity digest does not authenticate its complete contents",
        ));
    }
    Ok(())
}

fn validate_sample_against_identity(
    sample: &PairedSample,
    identity: &ExperimentIdentity,
) -> Result<(), PluginStatsError> {
    validate_sample_identity(sample)?;
    let expected_artifact = match sample.variant {
        Variant::Static => &identity.static_artifact_digest,
        Variant::Dynamic => &identity.dynamic_artifact_digest,
    };
    if sample.commit != identity.source_commit
        || sample.artifact_digest != *expected_artifact
        || sample.experiment_identity_digest != identity.identity_digest
    {
        return Err(PluginStatsError::new(
            "sample does not match the complete experiment or variant artifact identity",
        ));
    }
    Ok(())
}

fn metric_direction(metric: &str) -> Result<(RatioDirection, bool), PluginStatsError> {
    match metric {
        "successful_requests_per_second" | "output_tokens_per_second" => {
            Ok((RatioDirection::DynamicOverStatic, true))
        }
        "cpu_nanoseconds_per_successful_request" | "exporter_nanoseconds_per_record" => {
            Ok((RatioDirection::StaticOverDynamic, true))
        }
        "ttft_p50" | "ttft_p90" | "ttft_p99" | "itl_p50" | "itl_p90" | "itl_p99" => {
            Ok((RatioDirection::StaticOverDynamic, false))
        }
        _ => Err(PluginStatsError::new(format!(
            "unsupported plugin parity metric {metric}"
        ))),
    }
}

fn collect_metric(
    samples: &[PairedSample],
    metric: &str,
    direction: RatioDirection,
) -> Result<MetricVector, PluginStatsError> {
    let selected = samples
        .iter()
        .filter(|sample| sample.metric == metric)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(PluginStatsError::new(format!(
            "no samples for metric {metric}"
        )));
    }
    let scenario = selected[0].scenario.clone();
    let unit = selected[0].unit.as_str();
    let commit = selected[0].commit.as_str();
    let mut pairs: BTreeMap<&str, Vec<&PairedSample>> = BTreeMap::new();
    for sample in selected {
        if sample.scenario != scenario {
            return Err(PluginStatsError::new("paired metric spans scenarios"));
        }
        if sample.unit != unit || sample.commit != commit {
            return Err(PluginStatsError::new(
                "paired metric changes unit or source commit",
            ));
        }
        validate_sample_identity(sample)?;
        pairs.entry(&sample.pair_id).or_default().push(sample);
    }
    if pairs.len() < 2 {
        return Err(PluginStatsError::new(
            "paired bootstrap requires at least two pairs",
        ));
    }

    let mut static_values = Vec::with_capacity(pairs.len());
    let mut dynamic_values = Vec::with_capacity(pairs.len());
    let mut ratios = Vec::with_capacity(pairs.len());
    let mut pair_orders = Vec::with_capacity(pairs.len());
    let mut pair_ids = Vec::with_capacity(pairs.len());
    for (pair_id, members) in pairs {
        if members.len() != 2 || members[0].variant == members[1].variant {
            return Err(PluginStatsError::new(format!(
                "pair {pair_id} must contain exactly one static and one dynamic member"
            )));
        }
        let (static_value, dynamic_value) = match (members[0].variant, members[1].variant) {
            (Variant::Static, Variant::Dynamic) => (members[0].value, members[1].value),
            (Variant::Dynamic, Variant::Static) => (members[1].value, members[0].value),
            _ => {
                return Err(PluginStatsError::new(format!(
                    "pair {pair_id} has duplicate variants"
                )));
            }
        };
        let (numerator, denominator) = match direction {
            RatioDirection::DynamicOverStatic => (dynamic_value, static_value),
            RatioDirection::StaticOverDynamic => (static_value, dynamic_value),
        };
        let ratio = zero_aware_ratio(numerator, denominator);
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(PluginStatsError::new(format!(
                "pair {pair_id} produced an invalid positive ratio"
            )));
        }
        static_values.push(static_value);
        dynamic_values.push(dynamic_value);
        ratios.push(ratio);
        pair_ids.push(pair_id.to_owned());
        pair_orders.push([members[0].variant, members[1].variant]);
    }
    Ok(MetricVector {
        scenario,
        metric: metric.to_owned(),
        direction,
        static_values,
        dynamic_values,
        ratios,
        pair_ids,
        pair_orders,
    })
}

fn validate_sample_identity(sample: &PairedSample) -> Result<(), PluginStatsError> {
    if !sample.value.is_finite() || sample.value < 0.0 {
        return Err(PluginStatsError::new(format!(
            "{} contains a non-finite or negative value",
            sample.metric
        )));
    }
    if sample.scenario.is_empty()
        || sample.pair_id.is_empty()
        || sample.metric.is_empty()
        || sample.unit.is_empty()
        || !is_lower_hex(&sample.commit, 40)
        || !is_blake3_digest(&sample.artifact_digest)
        || !is_blake3_digest(&sample.experiment_identity_digest)
    {
        return Err(PluginStatsError::new("sample identity is incomplete"));
    }
    Ok(())
}

fn zero_aware_ratio(numerator: f64, denominator: f64) -> f64 {
    match (numerator, denominator) {
        (0.0, 0.0) => 1.0,
        (0.0, _) => f64::EPSILON,
        (_, 0.0) => 1.0 / f64::EPSILON,
        _ => numerator / denominator,
    }
}

fn is_blake3_digest(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|digest| is_lower_hex(digest, 64))
}

fn is_lower_hex(value: &str, expected_length: usize) -> bool {
    value.len() == expected_length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_exact_schedule(
    vector: &MetricVector,
    pair_schedule: &[PairSchedule],
) -> Result<(), PluginStatsError> {
    if vector.ratios.len() != pair_schedule.len() {
        return Err(PluginStatsError::new(format!(
            "{}/{} retains {} pairs instead of {}",
            vector.scenario,
            vector.metric,
            vector.ratios.len(),
            pair_schedule.len()
        )));
    }
    if vector
        .pair_ids
        .iter()
        .zip(&vector.pair_orders)
        .zip(pair_schedule)
        .any(|((pair_id, member_order), planned)| {
            pair_id != &planned.pair_id || member_order != &planned.member_order
        })
    {
        return Err(PluginStatsError::new(format!(
            "{}/{} differs from the seeded exact pair schedule",
            vector.scenario, vector.metric
        )));
    }
    Ok(())
}

fn validate_invalidations(
    case: &PairedCase,
    policy: &SimultaneousGatePolicy,
    identity: &ExperimentIdentity,
) -> Result<(), PluginStatsError> {
    if case.invalidation_attempts.len() > policy.max_replacement_pairs {
        return Err(PluginStatsError::new(format!(
            "case {} exceeds the five-pair replacement cap",
            case.scenario
        )));
    }
    let retained_orders = case
        .samples
        .iter()
        .filter(|sample| sample.metric == case.primary_metric)
        .fold(
            BTreeMap::<&str, Vec<Variant>>::new(),
            |mut orders, sample| {
                orders
                    .entry(&sample.pair_id)
                    .or_default()
                    .push(sample.variant);
                orders
            },
        );
    for (index, attempt) in case.invalidation_attempts.iter().enumerate() {
        if attempt.replacement_ordinal != index + 1 {
            return Err(PluginStatsError::new(
                "replacement ordinals must be contiguous and one-based",
            ));
        }
        if attempt.experiment_attempt == 0
            || attempt.experiment_attempt > policy.max_experiment_attempts
        {
            return Err(PluginStatsError::new(
                "pair invalidation exceeds the three-attempt limit",
            ));
        }
        if attempt.disposition != AttemptDisposition::InfrastructureInvalid {
            return Err(PluginStatsError::new(
                "a product failure cannot be replaced as infrastructure noise",
            ));
        }
        if attempt.reason.is_empty() || attempt.members.is_empty() {
            return Err(PluginStatsError::new(
                "invalidated pairs must retain members and a reason",
            ));
        }
        if attempt.members.iter().any(|sample| {
            sample.scenario != case.scenario
                || sample.pair_id != attempt.pair_id
                || validate_sample_against_identity(sample, identity).is_err()
        }) {
            return Err(PluginStatsError::new(
                "invalidated raw members have inconsistent identity or value",
            ));
        }
        let retained = retained_orders
            .get(attempt.pair_id.as_str())
            .ok_or_else(|| PluginStatsError::new("invalidated pair has no retained replacement"))?;
        let planned = identity
            .pair_schedule
            .iter()
            .find(|planned| planned.pair_id == attempt.pair_id)
            .ok_or_else(|| PluginStatsError::new("invalidated pair is absent from the schedule"))?;
        if retained.as_slice() != planned.member_order
            || attempt.member_order != planned.member_order
        {
            return Err(PluginStatsError::new(
                "replacement changed the invalidated pair's seeded member order",
            ));
        }
        let raw_order = attempt
            .members
            .iter()
            .filter(|sample| sample.metric == case.primary_metric)
            .map(|sample| sample.variant)
            .collect::<Vec<_>>();
        if raw_order.as_slice() != attempt.member_order {
            return Err(PluginStatsError::new(
                "retained invalidation members do not match the recorded member order",
            ));
        }
    }
    Ok(())
}

fn bootstrap_means(
    values: &[f64],
    resamples: usize,
    seed: u64,
) -> Result<Vec<f64>, PluginStatsError> {
    if values.is_empty() || resamples == 0 {
        return Err(PluginStatsError::new("bootstrap input is empty"));
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut distribution = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.random_range(0..values.len())];
        }
        let mean = total / values.len() as f64;
        if !mean.is_finite() {
            return Err(PluginStatsError::new(
                "paired bootstrap produced a non-finite mean",
            ));
        }
        distribution.push(mean);
    }
    Ok(distribution)
}

fn arithmetic_mean(values: &[f64]) -> Result<f64, PluginStatsError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(PluginStatsError::new("mean input is empty or non-finite"));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !mean.is_finite() {
        return Err(PluginStatsError::new("mean is non-finite"));
    }
    Ok(mean)
}

fn coefficient_of_variation(values: &[f64]) -> Result<f64, PluginStatsError> {
    if values.len() < 2 {
        return Err(PluginStatsError::new(
            "coefficient of variation requires two samples",
        ));
    }
    let mean = arithmetic_mean(values)?;
    if mean == 0.0 {
        return if values.iter().all(|value| *value == 0.0) {
            Ok(0.0)
        } else {
            Err(PluginStatsError::new(
                "coefficient of variation has a zero mean",
            ))
        };
    }
    let squared_deviations = values
        .iter()
        .map(|value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<f64>();
    let standard_deviation = (squared_deviations / (values.len() - 1) as f64).sqrt();
    let cv = standard_deviation / mean.abs();
    if !cv.is_finite() {
        return Err(PluginStatsError::new(
            "coefficient of variation is non-finite",
        ));
    }
    Ok(cv)
}

fn type_7_quantile(values: &[f64], probability: f64) -> Result<f64, PluginStatsError> {
    if values.is_empty()
        || values.iter().any(|value| !value.is_finite())
        || !probability.is_finite()
        || !(0.0..=1.0).contains(&probability)
    {
        return Err(PluginStatsError::new("invalid type-7 quantile input"));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }
    let index = probability * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;
    Ok(sorted[lower] + fraction * (sorted[upper] - sorted[lower]))
}
