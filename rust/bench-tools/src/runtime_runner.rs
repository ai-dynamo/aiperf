// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Same-process authority for the complete runtime parity matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::build_pair::{BuildPairReportV1, validate_authoritative_build_report_v1};
use crate::exporter_policy::parse_exporter_observable_policy;
use crate::exporter_runner::ExporterHarnessRunner;
use crate::plugin_stats::{
    AuthoritativeIdentityInput, ControlledAttemptDecision, ControlledAttemptRecord,
    ControlledMeasurementEvaluator, FrozenCasePlan, MemberTerminalOutcome, PairAttemptDecision,
    PairedCase, PairedSample, RawMemberTerminalRecord, RawPairTerminalRecord,
    SimultaneousGateInput, SimultaneousGateReport, Variant, acquire_authoritative_identity,
    checked_in_case_plans, checked_in_inventory_digest,
};

const OUTPUT_SCHEMA_V1: &[u8] = b"plugin_runtime_member_output/v1;closed-jcs-line;scenario,pair_id,variant,experiment_identity_blake3,completed_budget,active_duration_nanoseconds,metrics";
const POLICY_BYTES: &[u8] = include_bytes!("../../benchmarks/exporter-observable-policy.json");
const TASKSET: &str = "/usr/bin/taskset";

/// Failure while sealing, executing, or evaluating a controlled runtime matrix.
#[derive(Debug)]
pub struct ControlledRuntimeError(String);

impl ControlledRuntimeError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ControlledRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ControlledRuntimeError {}

/// Complete result owned by one non-resettable controlled runner invocation.
#[derive(Clone, Debug, Serialize)]
pub struct ControlledRuntimeReportV1 {
    /// Digest of the one sealed experiment identity used by every child row.
    pub experiment_identity_blake3: String,
    /// Canonical bytes of the sealed experiment identity.
    pub experiment_identity_bytes: Vec<u8>,
    /// Terminal controller decision.
    pub decision: ControlledAttemptDecision,
    /// Statistical report, present only after a complete valid matrix.
    pub statistical_report: Option<SimultaneousGateReport>,
    /// Append-only attempt ledger for this invocation.
    pub attempt_history: Vec<ControlledAttemptRecord>,
    /// Paired build record bound into the experiment identity.
    pub paired_build_record_blake3: String,
    /// Checked-in exporter observable policy bound into the identity.
    pub observable_policy_blake3: String,
    /// Closed terminal-output schema bound into the identity.
    pub output_schema_blake3: String,
    /// Complete checked-in workload matrix bound into the identity.
    pub workload_contract_blake3: String,
    /// Deterministic exporter corpus bound into the identity.
    pub corpus_blake3: String,
    /// Number of checked-in scenarios evaluated.
    pub scenario_count: usize,
    /// Number of retained scenario/pair combinations.
    pub retained_pair_count: usize,
    /// Number of real static/dynamic child processes executed, including warmups.
    pub executed_member_count: usize,
    /// Digests of every exact terminal stdout in execution order.
    pub terminal_output_blake3: Vec<String>,
    /// Canonical evidence binding identity, ledger, report, and raw outputs.
    pub runtime_evidence_bytes: Vec<u8>,
    /// Digest of the canonical runtime evidence bytes.
    pub runtime_evidence_blake3: String,
}

#[derive(Serialize)]
struct RuntimeEvidenceV1<'a> {
    schema_version: u8,
    experiment_identity_blake3: &'a str,
    decision: ControlledAttemptDecision,
    statistical_report: Option<&'a SimultaneousGateReport>,
    attempt_history: &'a [ControlledAttemptRecord],
    paired_build_record_blake3: &'a str,
    observable_policy_blake3: &'a str,
    output_schema_blake3: &'a str,
    workload_contract_blake3: &'a str,
    corpus_blake3: &'a str,
    scenario_count: usize,
    retained_pair_count: usize,
    executed_member_count: usize,
    terminal_output_blake3: &'a [String],
}

#[derive(Serialize)]
struct RuntimeAuthorityContractV1<'a> {
    schema_version: u8,
    paired_build_record_blake3: &'a str,
    member_build_receipt_blake3: [&'a str; 2],
    member_source_identity_blake3: [&'a str; 2],
    member_cargo_lock_blake3: [&'a str; 2],
    member_artifact_blake3: [&'a str; 2],
    observable_policy_blake3: &'a str,
    output_schema_blake3: &'a str,
    workload_contract_blake3: &'a str,
    corpus_blake3: &'a str,
    inventory_blake3: &'a str,
    admitted_environment_blake3: &'a str,
    taskset_blake3: &'a str,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeMemberOutputV1 {
    active_duration_nanoseconds: u64,
    completed_budget: u64,
    experiment_identity_blake3: String,
    metrics: BTreeMap<String, f64>,
    pair_id: String,
    scenario: String,
    schema_version: u8,
    variant: Variant,
}

struct MemberExecution {
    outcome: MemberTerminalOutcome,
    samples: Vec<PairedSample>,
    stdout_blake3: Option<String>,
}

/// Execute both build-bound members across the complete checked-in matrix.
pub fn run_controlled_runtime_v1(
    build_report: &BuildPairReportV1,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    let artifact_paths = validate_authoritative_build_report_v1(build_report).map_err(|error| {
        ControlledRuntimeError::new(format!("invalid paired build authority: {error}"))
    })?;
    let cases =
        checked_in_case_plans().map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let inventory_blake3 = checked_in_inventory_digest()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let policy =
        parse_exporter_observable_policy(POLICY_BYTES, &BTreeSet::new()).map_err(|error| {
            ControlledRuntimeError::new(format!("invalid checked-in exporter policy: {error}"))
        })?;
    let observable_policy_blake3 = policy
        .canonical_blake3()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let output_schema_blake3 = digest(OUTPUT_SCHEMA_V1);
    let workload_contract_blake3 = canonical_digest(&cases, "workload contract")?;
    let corpus_blake3 = ExporterHarnessRunner::new(policy)
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?
        .corpus_blake3()
        .to_owned();
    let taskset_blake3 = validate_taskset()?;
    let inherited_environment = capture_environment()?;
    let admitted_environment = inherited_environment
        .iter()
        .map(|(name, value)| (name.clone(), Some(value.clone())))
        .collect::<BTreeMap<_, _>>();
    let admitted_environment_blake3 =
        canonical_digest(&admitted_environment, "admitted environment")?;
    let authority_contract_blake3 = canonical_digest(
        &RuntimeAuthorityContractV1 {
            schema_version: 1,
            paired_build_record_blake3: &build_report.pair_record_blake3,
            member_build_receipt_blake3: [
                &build_report.members[0].build_receipt_blake3,
                &build_report.members[1].build_receipt_blake3,
            ],
            member_source_identity_blake3: [
                &build_report.members[0].source_identity_blake3,
                &build_report.members[1].source_identity_blake3,
            ],
            member_cargo_lock_blake3: [
                &build_report.members[0].cargo_lock_blake3,
                &build_report.members[1].cargo_lock_blake3,
            ],
            member_artifact_blake3: [
                &build_report.members[0].artifact_blake3,
                &build_report.members[1].artifact_blake3,
            ],
            observable_policy_blake3: &observable_policy_blake3,
            output_schema_blake3: &output_schema_blake3,
            workload_contract_blake3: &workload_contract_blake3,
            corpus_blake3: &corpus_blake3,
            inventory_blake3: &inventory_blake3,
            admitted_environment_blake3: &admitted_environment_blake3,
            taskset_blake3: &taskset_blake3,
        },
        "runtime authority contract",
    )?;
    let rustc = String::from_utf8(build_report.rustc_verbose_version.clone())
        .map_err(|_| ControlledRuntimeError::new("paired build rustc identity is not UTF-8"))?;
    let current_executable = std::env::current_exe().map_err(|error| {
        ControlledRuntimeError::new(format!(
            "cannot resolve runtime harness executable: {error}"
        ))
    })?;
    let harness_artifact_blake3 = digest_file(&current_executable, "runtime harness")?;
    let source_tree_digest = canonical_digest(
        &[
            &build_report.members[0].source_identity_blake3,
            &build_report.members[1].source_identity_blake3,
        ],
        "paired source identity",
    )?;
    let cargo_lock_digest = canonical_digest(
        &[
            &build_report.members[0].cargo_lock_blake3,
            &build_report.members[1].cargo_lock_blake3,
        ],
        "paired Cargo.lock identity",
    )?;
    let observed = acquire_authoritative_identity(AuthoritativeIdentityInput {
        source_commit: build_report.source_commit.clone(),
        source_tree_digest,
        cargo_lock_digest,
        rustc,
        sysroot_digest: build_report.sysroot_identity_blake3.clone(),
        profile: build_report.profile.clone(),
        static_artifact_digest: build_report.members[0].artifact_blake3.clone(),
        dynamic_artifact_digest: build_report.members[1].artifact_blake3.clone(),
        harness_artifact_digest: harness_artifact_blake3,
        authority_contract_digest: authority_contract_blake3,
        environment: admitted_environment,
    })
    .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let experiment_identity_blake3 = observed.identity_digest().to_owned();
    let experiment_identity_bytes = observed
        .canonical_identity_bytes()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;

    let mut evaluator = ControlledMeasurementEvaluator::new()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    evaluator
        .begin_attempt()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let schedule = evaluator.pair_schedule().to_vec();
    let mut executed_member_count = 0_usize;
    let mut terminal_output_blake3 = Vec::new();
    let mut measured_cases = Vec::with_capacity(cases.len());

    for case in &cases {
        for warmup in 0..case.warmups {
            let pair_id = format!("warmup-{warmup:02}");
            for variant in [Variant::Static, Variant::Dynamic] {
                let execution = execute_member(
                    case,
                    &pair_id,
                    variant,
                    artifact_for(variant, &artifact_paths),
                    build_report,
                    &experiment_identity_blake3,
                    &inherited_environment,
                )?;
                executed_member_count += 1;
                if let Some(stdout_blake3) = execution.stdout_blake3 {
                    terminal_output_blake3.push(stdout_blake3);
                }
                if execution.outcome != MemberTerminalOutcome::Completed {
                    evaluator
                        .finish_authoritative_product_failure(format!(
                            "warmup {pair_id} for {} {:?} failed: {:?}",
                            case.scenario, variant, execution.outcome
                        ))
                        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
                    return runtime_report(
                        &evaluator,
                        experiment_identity_blake3,
                        experiment_identity_bytes,
                        build_report,
                        observable_policy_blake3,
                        output_schema_blake3,
                        workload_contract_blake3,
                        corpus_blake3,
                        cases.len(),
                        executed_member_count,
                        terminal_output_blake3,
                    );
                }
            }
        }

        let mut samples = Vec::new();
        for scheduled in &schedule {
            let mut member_records = Vec::with_capacity(2);
            let mut pair_samples = Vec::new();
            for variant in scheduled.member_order {
                let execution = execute_member(
                    case,
                    &scheduled.pair_id,
                    variant,
                    artifact_for(variant, &artifact_paths),
                    build_report,
                    &experiment_identity_blake3,
                    &inherited_environment,
                )?;
                executed_member_count += 1;
                if let Some(stdout_blake3) = execution.stdout_blake3 {
                    terminal_output_blake3.push(stdout_blake3);
                }
                member_records.push(RawMemberTerminalRecord {
                    variant,
                    outcome: execution.outcome,
                });
                pair_samples.extend(execution.samples);
            }
            let decision = evaluator
                .record_pair(RawPairTerminalRecord {
                    scenario: case.scenario.clone(),
                    pair_id: scheduled.pair_id.clone(),
                    member_order: scheduled.member_order,
                    members: member_records,
                    asserted_reason: None,
                    asserted_disposition: None,
                })
                .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
            if decision != PairAttemptDecision::RetainPair {
                return runtime_report(
                    &evaluator,
                    experiment_identity_blake3,
                    experiment_identity_bytes,
                    build_report,
                    observable_policy_blake3,
                    output_schema_blake3,
                    workload_contract_blake3,
                    corpus_blake3,
                    cases.len(),
                    executed_member_count,
                    terminal_output_blake3,
                );
            }
            samples.extend(pair_samples);
        }
        measured_cases.push(PairedCase {
            scenario: case.scenario.clone(),
            primary_metric: case.primary_metric.clone(),
            samples,
            invalidation_attempts: Vec::new(),
        });
    }

    evaluator
        .finish_authoritative_measurements(
            &SimultaneousGateInput {
                cases: measured_cases,
            },
            observed,
        )
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    runtime_report(
        &evaluator,
        experiment_identity_blake3,
        experiment_identity_bytes,
        build_report,
        observable_policy_blake3,
        output_schema_blake3,
        workload_contract_blake3,
        corpus_blake3,
        cases.len(),
        executed_member_count,
        terminal_output_blake3,
    )
}

fn execute_member(
    case: &FrozenCasePlan,
    pair_id: &str,
    variant: Variant,
    artifact_path: &Path,
    build_report: &BuildPairReportV1,
    experiment_identity_blake3: &str,
    inherited_environment: &BTreeMap<String, String>,
) -> Result<MemberExecution, ControlledRuntimeError> {
    let artifact_blake3 = match variant {
        Variant::Static => &build_report.members[0].artifact_blake3,
        Variant::Dynamic => &build_report.members[1].artifact_blake3,
    };
    if digest_file(artifact_path, "runtime member artifact")? != *artifact_blake3 {
        return Err(ControlledRuntimeError::new(
            "runtime member artifact changed after identity sealing",
        ));
    }
    if case.command.len() < 5 || case.command[0] != "taskset" || case.command[1] != "-c" {
        return Err(ControlledRuntimeError::new(
            "checked-in command is not an executable runtime template",
        ));
    }
    let mut command = Command::new(TASKSET);
    command
        .args(["-c", case.command[2].as_str()])
        .arg(artifact_path)
        .args(&case.command[4..])
        .env_clear();
    for (name, value) in inherited_environment {
        command.env(name, value);
    }
    command
        .env(
            "AIPERF_PARITY_EXPERIMENT_IDENTITY",
            experiment_identity_blake3,
        )
        .env("AIPERF_PARITY_METRICS", case.measured_metrics.join(","))
        .env("AIPERF_PARITY_PAIR_ID", pair_id)
        .env(
            "AIPERF_PARITY_REQUEST_BUDGET",
            case.request_budget.to_string(),
        )
        .env("AIPERF_PARITY_SCENARIO", &case.scenario)
        .env(
            "AIPERF_PARITY_VARIANT",
            match variant {
                Variant::Static => "static",
                Variant::Dynamic => "dynamic",
            },
        );
    let output = command.output().map_err(|error| {
        ControlledRuntimeError::new(format!("cannot execute controlled runtime member: {error}"))
    })?;
    if !output.status.success() {
        return Ok(MemberExecution {
            outcome: MemberTerminalOutcome::Crash(format!("process exited with {}", output.status)),
            samples: Vec::new(),
            stdout_blake3: Some(digest(&output.stdout)),
        });
    }
    let stdout_blake3 = digest(&output.stdout);
    let decoded = match decode_member_output(
        &output.stdout,
        case,
        pair_id,
        variant,
        experiment_identity_blake3,
    ) {
        Ok(decoded) => decoded,
        Err(error) => {
            return Ok(MemberExecution {
                outcome: MemberTerminalOutcome::MalformedOutput(error.to_string()),
                samples: Vec::new(),
                stdout_blake3: Some(stdout_blake3),
            });
        }
    };
    let samples = decoded
        .metrics
        .into_iter()
        .map(|(metric, value)| PairedSample {
            scenario: case.scenario.clone(),
            pair_id: pair_id.to_owned(),
            variant,
            unit: metric_unit(&metric).to_owned(),
            metric,
            value,
            commit: build_report.source_commit.clone(),
            artifact_digest: artifact_blake3.clone(),
            experiment_identity_digest: experiment_identity_blake3.to_owned(),
        })
        .collect();
    Ok(MemberExecution {
        outcome: MemberTerminalOutcome::Completed,
        samples,
        stdout_blake3: Some(stdout_blake3),
    })
}

fn decode_member_output(
    bytes: &[u8],
    case: &FrozenCasePlan,
    pair_id: &str,
    variant: Variant,
    experiment_identity_blake3: &str,
) -> Result<RuntimeMemberOutputV1, ControlledRuntimeError> {
    let decoded: RuntimeMemberOutputV1 = serde_json::from_slice(bytes).map_err(|error| {
        ControlledRuntimeError::new(format!("member output is not schema-1 JSON: {error}"))
    })?;
    let mut canonical = serde_json_canonicalizer::to_vec(&decoded).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize member output: {error}"))
    })?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(ControlledRuntimeError::new(
            "member output is not one exact canonical JCS line",
        ));
    }
    let metric_names = decoded.metrics.keys().cloned().collect::<Vec<_>>();
    let minimum_duration_nanoseconds = case
        .minimum_duration_seconds
        .checked_mul(1_000_000_000)
        .ok_or_else(|| ControlledRuntimeError::new("minimum duration overflow"))?;
    if decoded.schema_version != 1
        || decoded.scenario != case.scenario
        || decoded.pair_id != pair_id
        || decoded.variant != variant
        || decoded.experiment_identity_blake3 != experiment_identity_blake3
        || decoded.completed_budget != case.request_budget
        || decoded.active_duration_nanoseconds < minimum_duration_nanoseconds
        || metric_names != case.measured_metrics
        || decoded
            .metrics
            .values()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ControlledRuntimeError::new(
            "member output differs from its sealed scenario/member contract",
        ));
    }
    Ok(decoded)
}

fn runtime_report(
    evaluator: &ControlledMeasurementEvaluator,
    experiment_identity_blake3: String,
    experiment_identity_bytes: Vec<u8>,
    build_report: &BuildPairReportV1,
    observable_policy_blake3: String,
    output_schema_blake3: String,
    workload_contract_blake3: String,
    corpus_blake3: String,
    scenario_count: usize,
    executed_member_count: usize,
    terminal_output_blake3: Vec<String>,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    let decision = evaluator
        .history()
        .last()
        .map(|attempt| attempt.decision)
        .ok_or_else(|| {
            ControlledRuntimeError::new(
                "controlled runtime report requires one terminal attempt ledger entry",
            )
        })?;
    let statistical_report = evaluator.last_statistical_report().cloned();
    let attempt_history = evaluator.history().to_vec();
    let retained_pair_count = evaluator
        .raw_pair_history()
        .iter()
        .filter(|record| record.decision == PairAttemptDecision::RetainPair)
        .count();
    let runtime_evidence_bytes = serde_json_canonicalizer::to_vec(&RuntimeEvidenceV1 {
        schema_version: 1,
        experiment_identity_blake3: &experiment_identity_blake3,
        decision,
        statistical_report: statistical_report.as_ref(),
        attempt_history: &attempt_history,
        paired_build_record_blake3: &build_report.pair_record_blake3,
        observable_policy_blake3: &observable_policy_blake3,
        output_schema_blake3: &output_schema_blake3,
        workload_contract_blake3: &workload_contract_blake3,
        corpus_blake3: &corpus_blake3,
        scenario_count,
        retained_pair_count,
        executed_member_count,
        terminal_output_blake3: &terminal_output_blake3,
    })
    .map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize runtime evidence: {error}"))
    })?;
    let runtime_evidence_blake3 = digest(&runtime_evidence_bytes);
    Ok(ControlledRuntimeReportV1 {
        experiment_identity_blake3,
        experiment_identity_bytes,
        decision,
        statistical_report,
        attempt_history,
        paired_build_record_blake3: build_report.pair_record_blake3.clone(),
        observable_policy_blake3,
        output_schema_blake3,
        workload_contract_blake3,
        corpus_blake3,
        scenario_count,
        retained_pair_count,
        executed_member_count,
        terminal_output_blake3,
        runtime_evidence_bytes,
        runtime_evidence_blake3,
    })
}

fn artifact_for(variant: Variant, paths: &[PathBuf; 2]) -> &Path {
    match variant {
        Variant::Static => &paths[0],
        Variant::Dynamic => &paths[1],
    }
}

fn metric_unit(metric: &str) -> &'static str {
    match metric {
        "successful_requests_per_second" => "requests_per_second",
        "output_tokens_per_second" => "tokens_per_second",
        "cpu_nanoseconds_per_successful_request" | "exporter_nanoseconds_per_record" => {
            "nanoseconds"
        }
        "allocated_bytes_per_successful_request" => "bytes",
        "allocation_count_per_successful_request" => "allocations",
        _ => "milliseconds",
    }
}

fn validate_taskset() -> Result<String, ControlledRuntimeError> {
    let path = Path::new(TASKSET);
    let canonical = fs::canonicalize(path).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot resolve controlled taskset: {error}"))
    })?;
    if canonical != path {
        return Err(ControlledRuntimeError::new(
            "controlled taskset path is not canonical",
        ));
    }
    digest_file(path, "taskset")
}

fn capture_environment() -> Result<BTreeMap<String, String>, ControlledRuntimeError> {
    std::env::vars_os()
        .map(|(name, value)| {
            let name = name.into_string().map_err(|_| {
                ControlledRuntimeError::new("runtime environment has a non-UTF-8 name")
            })?;
            let value = value.into_string().map_err(|_| {
                ControlledRuntimeError::new(format!(
                    "runtime environment value for {name} is not UTF-8"
                ))
            })?;
            Ok((name, value))
        })
        .collect()
}

fn canonical_digest<T: Serialize>(
    value: &T,
    label: &str,
) -> Result<String, ControlledRuntimeError> {
    let bytes = serde_json_canonicalizer::to_vec(value).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize {label}: {error}"))
    })?;
    Ok(digest(&bytes))
}

fn digest_file(path: &Path, label: &str) -> Result<String, ControlledRuntimeError> {
    let bytes = fs::read(path).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot read {label} {}: {error}", path.display()))
    })?;
    Ok(digest(&bytes))
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}
