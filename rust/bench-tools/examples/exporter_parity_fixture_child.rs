// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conforming artifact-bound exporter member child.
//!
//! The controlled runner admits an exporter sample only when the child returns
//! one canonical artifact-bound line whose sealed evidence reproduces the
//! controller expectation it was handed. Every digest in that line is chained
//! off controller-owned values that exist only at child run time, so no
//! pre-baked fixture file can stand in for a real member. This example is the
//! smallest program that satisfies the protocol end to end and is used by the
//! controlled-runtime integration tests as the exporter member artifact.

use std::collections::BTreeSet;
use std::io::Write as _;
use std::process::ExitCode;

use aiperf_bench_tools::exporter_policy::{
    ProvenanceBindingV1, apply_exporter_observable_policy_v1, parse_exporter_observable_policy,
};
use aiperf_bench_tools::plugin_stats::{
    ArtifactBoundExporterMemberV1, ExporterEvidenceMode, ExporterMember, ExporterMemberBinding,
    ExporterMemberEvidence, ExporterMemberRecord, ExporterRepetitionReceipt,
    ExporterSampleContract, RetainedExporterEvidence, Variant,
};
use aiperf_bench_tools::runtime_runner::{
    ExporterChildExpectationV1, ExporterMemberChildOutputV1,
};

/// Checked-in policy the controlled runner binds into the exporter identity.
const PAIRED_POLICY_BYTES: &[u8] =
    include_bytes!("../../benchmarks/exporter-paired-runtime-policy.json");

/// Artifact tree both members publish; parity requires byte-identical output.
const RAW_OBSERVABLE: &[u8] = b"[{\"kind\":\"regular_file\",\"path\":\"records.json\"}]\n";

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}

fn canonical_line<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json_canonicalizer::to_vec(value)
        .map_err(|error| format!("cannot canonicalize child output: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn emit() -> Result<Vec<u8>, String> {
    let expectation_json = std::env::var("AIPERF_PARITY_EXPORTER_EXPECTATION")
        .map_err(|_| "controller expectation is absent from the member environment".to_owned())?;
    let expectation: ExporterChildExpectationV1 = serde_json::from_str(&expectation_json)
        .map_err(|error| format!("controller expectation is not schema-1: {error}"))?;
    let policy = parse_exporter_observable_policy(PAIRED_POLICY_BYTES, &BTreeSet::new())
        .map_err(|error| format!("checked-in paired policy is invalid: {error}"))?;
    let contract = ExporterSampleContract::normative();

    let binding = ExporterMemberBinding {
        mode: ExporterEvidenceMode::Paired,
        experiment_identity_blake3: expectation.experiment_identity_blake3.clone(),
        attempt_ordinal: expectation.attempt_ordinal,
        scenario_id: expectation.scenario_id.clone(),
        pair_id: expectation.pair_id.clone(),
        member: expectation.member,
        corpus_blake3: expectation.corpus_blake3.clone(),
        observable_kind: expectation.observable_kind,
        observable_policy_blake3: expectation.observable_policy_blake3.clone(),
        build_artifact_blake3: expectation.build_artifact_blake3.clone(),
        build_receipt_blake3: expectation.build_receipt_blake3.clone(),
    };

    // The frozen budget is the floor for the whole member, so each of the 16
    // repetitions carries a rounded-up share of it.
    let repetition_ns = expectation
        .minimum_active_duration_ns
        .div_ceil(contract.sample_repetitions as u64)
        .max(1);

    let mut receipts = Vec::with_capacity(contract.sample_repetitions);
    let mut retained_provenance = Vec::new();
    for ordinal in 0..contract.sample_repetitions as u64 {
        let applied = apply_exporter_observable_policy_v1(
            &policy,
            &ProvenanceBindingV1 {
                experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
                attempt_ordinal: binding.attempt_ordinal,
                scenario_id: binding.scenario_id.clone(),
                pair_id: binding.pair_id.clone(),
                member: binding.member,
                repetition_ordinal: ordinal,
            },
            RAW_OBSERVABLE,
            &[],
        )
        .map_err(|error| format!("policy application failed: {error}"))?;
        if ordinal == 0 {
            retained_provenance = applied.provenance_receipt_bytes.clone();
        }
        receipts.push(ExporterRepetitionReceipt {
            schema_version: 1,
            experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
            attempt_ordinal: binding.attempt_ordinal,
            scenario_id: binding.scenario_id.clone(),
            pair_id: binding.pair_id.clone(),
            member: binding.member,
            repetition_ordinal: ordinal,
            corpus_blake3: binding.corpus_blake3.clone(),
            processed_records: contract.corpus_records,
            observable_kind: binding.observable_kind,
            raw_observable_blake3: applied.raw_observable_blake3,
            comparison_observable_blake3: applied.comparison_observable_blake3,
            provenance_receipt_blake3: applied.provenance_receipt_blake3,
            active_duration_ns: repetition_ns,
            build_artifact_blake3: binding.build_artifact_blake3.clone(),
            build_receipt_blake3: binding.build_receipt_blake3.clone(),
        });
    }

    let evidence = ExporterMemberEvidence {
        repetition_receipt_bytes: canonical_line(&receipts)?,
        retained: RetainedExporterEvidence {
            repetition_ordinal: 0,
            raw_observable_bytes: RAW_OBSERVABLE.to_vec(),
            comparison_observable_bytes: RAW_OBSERVABLE.to_vec(),
            provenance_receipt_bytes: retained_provenance,
        },
    };
    let active_duration_ns = repetition_ns * contract.sample_repetitions as u64;
    let record = ExporterMemberRecord {
        schema_version: 1,
        experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
        attempt_ordinal: binding.attempt_ordinal,
        scenario_id: binding.scenario_id.clone(),
        pair_id: binding.pair_id.clone(),
        member: binding.member,
        active_duration_ns,
        processed_records: contract.processed_records,
        retained_artifact_records: contract.retained_artifact_records,
        comparison_observable_blake3: digest(RAW_OBSERVABLE),
        repetition_receipts_blake3: digest(&evidence.repetition_receipt_bytes),
        retained_repetition_ordinal: 0,
        retained_raw_observable_blake3: digest(RAW_OBSERVABLE),
        retained_comparison_observable_blake3: digest(RAW_OBSERVABLE),
        retained_provenance_receipt_blake3: digest(&evidence.retained.provenance_receipt_bytes),
        observable_policy_blake3: binding.observable_policy_blake3.clone(),
        build_artifact_blake3: binding.build_artifact_blake3.clone(),
        build_receipt_blake3: binding.build_receipt_blake3.clone(),
    };
    let record_bytes = canonical_line(&record)?;

    canonical_line(&ExporterMemberChildOutputV1 {
        artifact_bound: ArtifactBoundExporterMemberV1 {
            binding,
            evidence,
            backing_payloads: Vec::new(),
            record_bytes,
            receiver_protocol: None,
            receiver_protocol_authority_blake3: None,
        },
        experiment_identity_blake3: expectation.experiment_identity_blake3,
        pair_id: expectation.pair_id,
        scenario: expectation.scenario_id,
        schema_version: 1,
        variant: match expectation.member {
            ExporterMember::Static => Variant::Static,
            ExporterMember::Dynamic => Variant::Dynamic,
        },
    })
}

fn main() -> ExitCode {
    match emit() {
        Ok(line) => {
            if std::io::stdout().write_all(&line).is_err() {
                return ExitCode::from(75);
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            let _ = writeln!(std::io::stderr(), "{message}");
            ExitCode::from(74)
        }
    }
}
