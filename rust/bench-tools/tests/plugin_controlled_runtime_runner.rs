// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production-authority boundary for complete runtime parity measurement.

use std::path::{Path, PathBuf};

use aiperf_bench_tools::build_pair::{
    BuildLtoV1, BuildPairMemberV1, BuildPairPlanV1, BuildPairReportV1, run_paired_build_v1,
};
use aiperf_bench_tools::exporter_policy::parse_exporter_observable_policy;
use aiperf_bench_tools::exporter_runner::{
    ExporterHarnessError, ExporterHarnessRunner, ExporterRecordStream, ExporterWorkload,
    HostExporterCapture,
};
use aiperf_bench_tools::plugin_stats::{
    ControlledAttemptDecision, ExporterMember, PairAttemptDecision, Variant,
};
use aiperf_bench_tools::runtime_runner::{
    ControlledExporterWorkloadFactory, ExporterWorkloadAcquisitionError, ExporterWorkloadRequest,
    ControlledRuntimeReportV1, HostLivenessSourceV1, controlled_attempt_ledger_path,
    run_controlled_runtime_with_exporters_v1, run_controlled_runtime_with_ledger_v1,
    run_controlled_runtime_with_liveness_v1,
};

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const PREBUILD_IDENTITY: &str =
    "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}

fn write_executable(path: &Path, bytes: &[u8]) {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::write(path, bytes).expect("fixture executable is written");
    let mut permissions = std::fs::metadata(path)
        .expect("fixture executable metadata is available")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).expect("fixture executable is executable");
}

fn runtime_artifact(label: &str) -> Vec<u8> {
    format!(
        "#!/bin/sh\n# {label}\nset -eu\nmetrics=\nold_ifs=$IFS\nIFS=,\nfor metric in $AIPERF_PARITY_METRICS; do\n  value=100\n  if [ \"$metric\" = exporter_nanoseconds_per_record ]; then\n    if [ \"$AIPERF_PARITY_VARIANT\" = static ]; then value=1; else value=1000; fi\n  fi\n  if [ -n \"$metrics\" ]; then metrics=\"$metrics,\"; fi\n  metrics=\"$metrics\\\"$metric\\\":$value\"\ndone\nIFS=$old_ifs\nprintf '{{\"active_duration_nanoseconds\":30000000000,\"completed_budget\":%s,\"experiment_identity_blake3\":\"%s\",\"metrics\":{{%s}},\"pair_id\":\"%s\",\"scenario\":\"%s\",\"schema_version\":1,\"variant\":\"%s\"}}\\n' \"$AIPERF_PARITY_REQUEST_BUDGET\" \"$AIPERF_PARITY_EXPERIMENT_IDENTITY\" \"$metrics\" \"$AIPERF_PARITY_PAIR_ID\" \"$AIPERF_PARITY_SCENARIO\" \"$AIPERF_PARITY_VARIANT\"\n"
    )
    .into_bytes()
}

fn runtime_artifact_rejecting_exporter(label: &str) -> Vec<u8> {
    let script = String::from_utf8(runtime_artifact(label)).expect("fixture script is UTF-8");
    script
        .replacen(
            "set -eu\n",
            "set -eu\nif [ \"$AIPERF_PARITY_SCENARIO\" = exporter_100k ]; then exit 71; fi\n",
            1,
        )
        .into_bytes()
}

fn runtime_artifact_with_persistent_affinity_loss(label: &str) -> Vec<u8> {
    let script = String::from_utf8(runtime_artifact(label)).expect("fixture script is UTF-8");
    script
        .replacen(
            "set -eu\n",
            "set -eu\ncase \"$AIPERF_PARITY_PAIR_ID\" in pair-00) if [ \"$AIPERF_PARITY_SCENARIO\" = http_non_streaming_c1 ]; then sleep 0.05; /usr/bin/taskset -pc 8 $$ >/dev/null; sleep 0.1; fi;; esac\n",
            1,
        )
        .into_bytes()
}

fn runtime_artifact_with_one_affinity_loss(label: &str, marker: &Path) -> Vec<u8> {
    let script = String::from_utf8(runtime_artifact(label)).expect("fixture script is UTF-8");
    script
        .replacen(
            "set -eu\n",
            &format!(
                "set -eu\nif [ \"$AIPERF_PARITY_SCENARIO\" = http_non_streaming_c1 ] && [ \"$AIPERF_PARITY_PAIR_ID\" = pair-00 ] && [ ! -e '{}' ]; then touch '{}'; sleep 0.05; /usr/bin/taskset -pc 8 $$ >/dev/null; sleep 0.1; fi\n",
                marker.display(),
                marker.display()
            ),
            1,
        )
        .into_bytes()
}

fn runtime_artifact_with_one_reboot(label: &str, boot_identity: &Path, marker: &Path) -> Vec<u8> {
    let script = String::from_utf8(runtime_artifact(label)).expect("fixture script is UTF-8");
    script
        .replacen(
            "set -eu\n",
            &format!(
                "set -eu\nif [ \"$AIPERF_PARITY_SCENARIO\" = http_non_streaming_c1 ] && [ \"$AIPERF_PARITY_PAIR_ID\" = pair-00 ] && [ ! -e '{marker}' ]; then touch '{marker}'; printf '%s' \"reboot-$(date +%s%N)\" > '{boot}'; fi\n",
                marker = marker.display(),
                boot = boot_identity.display()
            ),
            1,
        )
        .into_bytes()
}

fn runtime_artifact_rebooting_every_pair(label: &str, boot_identity: &Path) -> Vec<u8> {
    let script = String::from_utf8(runtime_artifact(label)).expect("fixture script is UTF-8");
    script
        .replacen(
            "set -eu\n",
            &format!(
                "set -eu\ncase \"$AIPERF_PARITY_PAIR_ID\" in pair-*) printf '%s' \"reboot-$(date +%s%N)-$$\" > '{boot}';; esac\n",
                boot = boot_identity.display()
            ),
            1,
        )
        .into_bytes()
}

fn runtime_artifact_killing_the_mock_once(label: &str, mock_pid: &Path, marker: &Path) -> Vec<u8> {
    let script = String::from_utf8(runtime_artifact(label)).expect("fixture script is UTF-8");
    script
        .replacen(
            "set -eu\n",
            &format!(
                "set -eu\nif [ \"$AIPERF_PARITY_SCENARIO\" = http_non_streaming_c1 ] && [ \"$AIPERF_PARITY_PAIR_ID\" = pair-00 ] && [ ! -e '{marker}' ]; then touch '{marker}'; kill -9 \"$(cat '{pid}')\" 2>/dev/null || true; fi\n",
                marker = marker.display(),
                pid = mock_pid.display()
            ),
            1,
        )
        .into_bytes()
}

#[derive(Clone, Copy)]
enum FakeExporterMode {
    AcquisitionFailure,
    ProductFailure,
}

struct FakeExporterFactory {
    mode: FakeExporterMode,
}

struct FakeExporterWorkload {
    member: ExporterMember,
    mode: FakeExporterMode,
}

impl ControlledExporterWorkloadFactory for FakeExporterFactory {
    fn acquire(
        &mut self,
        request: ExporterWorkloadRequest<'_>,
    ) -> Result<Box<dyn ExporterWorkload>, ExporterWorkloadAcquisitionError> {
        if matches!(self.mode, FakeExporterMode::AcquisitionFailure) {
            return Err(ExporterWorkloadAcquisitionError::new(
                "fixture adapter acquisition failed",
            ));
        }
        Ok(Box::new(FakeExporterWorkload {
            member: request.member(),
            mode: self.mode,
        }))
    }
}

impl ExporterWorkload for FakeExporterWorkload {
    fn export(
        &mut self,
        _repetition_ordinal: u64,
        records: &mut ExporterRecordStream<'_>,
        capture: &mut HostExporterCapture,
    ) -> Result<(), ExporterHarnessError> {
        if matches!(self.mode, FakeExporterMode::ProductFailure) {
            return Err(ExporterHarnessError::product(
                "fixture exporter product failure",
            ));
        }
        let rounds = match self.member {
            ExporterMember::Static => 8,
            ExporterMember::Dynamic => 1,
        };
        let mut accumulator = 0_u64;
        for record in records {
            for round in 0..rounds {
                accumulator = std::hint::black_box(
                    accumulator
                        .wrapping_add(record.ordinal())
                        .wrapping_add(round),
                );
            }
        }
        std::hint::black_box(accumulator);
        capture.write_artifact("records.json", b"{\"status\":\"complete\"}")
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    cargo: PathBuf,
    rustc: PathBuf,
    sysroot: PathBuf,
    static_source: PathBuf,
    dynamic_source: PathBuf,
    static_target: PathBuf,
    dynamic_target: PathBuf,
    cargo_bytes: Vec<u8>,
    rustc_bytes: Vec<u8>,
    rustc_verbose_version: Vec<u8>,
    static_artifact: Vec<u8>,
    dynamic_artifact: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("temporary fixture directory");
        let cargo = directory.path().join("cargo");
        let cargo_bytes = b"#!/bin/sh\nset -eu\ncp artifact-source \"$CARGO_TARGET_DIR/artifact.bin\"\nchmod 755 \"$CARGO_TARGET_DIR/artifact.bin\"\n".to_vec();
        write_executable(&cargo, &cargo_bytes);

        let sysroot = directory.path().join("sysroot");
        std::fs::create_dir_all(&sysroot).expect("sysroot exists");
        std::fs::write(sysroot.join("sysroot.identity"), b"sysroot identity\n")
            .expect("sysroot identity is written");
        let rustc = directory.path().join("rustc");
        let rustc_verbose_version = b"rustc fixture\ncommit-hash: feedface\n".to_vec();
        let rustc_bytes = format!(
            "#!/bin/sh\nset -eu\ncase \"$1\" in\n  -vV) printf '%b' 'rustc fixture\\ncommit-hash: feedface\\n' ;;\n  --print) [ \"$2\" = sysroot ]; printf '%s\\n' '{}' ;;\n  *) exit 64 ;;\nesac\n",
            sysroot.display()
        )
        .into_bytes();
        write_executable(&rustc, &rustc_bytes);

        let static_source = directory.path().join("static-source");
        let dynamic_source = directory.path().join("dynamic-source");
        std::fs::create_dir_all(&static_source).expect("static source exists");
        std::fs::create_dir_all(&dynamic_source).expect("dynamic source exists");
        let static_artifact = runtime_artifact("static authority fixture");
        let dynamic_artifact = runtime_artifact("dynamic authority fixture");
        for (source, identity, lock, artifact) in [
            (
                &static_source,
                b"static source identity\n".as_slice(),
                b"static lock\n".as_slice(),
                static_artifact.as_slice(),
            ),
            (
                &dynamic_source,
                b"dynamic source identity\n".as_slice(),
                b"dynamic lock\n".as_slice(),
                dynamic_artifact.as_slice(),
            ),
        ] {
            std::fs::write(source.join("source.identity"), identity)
                .expect("source identity is written");
            std::fs::write(source.join("Cargo.lock"), lock).expect("lock is written");
            write_executable(&source.join("artifact-source"), artifact);
        }

        let static_target = directory.path().join("static-target");
        let dynamic_target = directory.path().join("dynamic-target");
        Self {
            cargo,
            rustc,
            sysroot,
            static_source,
            dynamic_source,
            static_target,
            dynamic_target,
            cargo_bytes,
            rustc_bytes,
            rustc_verbose_version,
            static_artifact,
            dynamic_artifact,
            _directory: directory,
        }
    }

    fn member(&self, variant: Variant) -> BuildPairMemberV1 {
        let (source, target, identity, lock, artifact) = match variant {
            Variant::Static => (
                &self.static_source,
                &self.static_target,
                b"static source identity\n".as_slice(),
                b"static lock\n".as_slice(),
                self.static_artifact.as_slice(),
            ),
            Variant::Dynamic => (
                &self.dynamic_source,
                &self.dynamic_target,
                b"dynamic source identity\n".as_slice(),
                b"dynamic lock\n".as_slice(),
                self.dynamic_artifact.as_slice(),
            ),
        };
        BuildPairMemberV1 {
            variant,
            source_root: source.clone(),
            source_identity_receipt: PathBuf::from("source.identity"),
            source_identity_blake3: digest(identity),
            cargo_lock_blake3: digest(lock),
            target_root: target.clone(),
            artifact_relative_path: PathBuf::from("artifact.bin"),
            expected_artifact_blake3: digest(artifact),
        }
    }

    fn build_report(&self) -> BuildPairReportV1 {
        run_paired_build_v1(&BuildPairPlanV1 {
            scenario: "runtime-full-matrix".to_owned(),
            pair_id: "build-pair".to_owned(),
            source_commit: COMMIT.to_owned(),
            experiment_identity_blake3: PREBUILD_IDENTITY.to_owned(),
            cargo_executable: self.cargo.clone(),
            cargo_executable_blake3: digest(&self.cargo_bytes),
            rustc_executable: self.rustc.clone(),
            rustc_executable_blake3: digest(&self.rustc_bytes),
            rustc_verbose_version: self.rustc_verbose_version.clone(),
            sysroot_root: self.sysroot.clone(),
            sysroot_identity_receipt: PathBuf::from("sysroot.identity"),
            sysroot_identity_blake3: digest(b"sysroot identity\n"),
            baseline_arguments: vec!["build".to_owned(), "--locked".to_owned()],
            profile: "release".to_owned(),
            features: vec!["full".to_owned()],
            lto: BuildLtoV1::Fat,
            static_member: self.member(Variant::Static),
            dynamic_member: self.member(Variant::Dynamic),
        })
        .expect("paired runtime artifacts build under one authority")
    }
}

#[test]
fn bare_exact_artifact_exporter_metric_is_a_product_failure() {
    let fixture = Fixture::new();
    let build_report = fixture.build_report();

    let report = run_controlled_runtime_with_ledger_v1(
        &build_report,
        &fixture._directory.path().join("raw-member-attempts.jsonl"),
    )
    .expect("controlled runtime matrix executes and evaluates");

    assert_eq!(report.decision, ControlledAttemptDecision::ValidFailure);
    assert_eq!(report.scenario_count, 12);
    assert_eq!(report.retained_pair_count, 330);
    assert_eq!(report.executed_member_count, 771);
    assert_eq!(report.attempt_history.len(), 1);
    assert_eq!(
        report.attempt_history[0].decision,
        ControlledAttemptDecision::ValidFailure
    );
    assert_eq!(
        report.paired_build_record_blake3,
        build_report.pair_record_blake3
    );
    assert!(report.experiment_identity_blake3.starts_with("blake3:"));
    assert!(report.observable_policy_blake3.starts_with("blake3:"));
    assert!(report.output_schema_blake3.starts_with("blake3:"));
    assert!(report.workload_contract_blake3.starts_with("blake3:"));
    let policy = parse_exporter_observable_policy(
        include_bytes!("../../benchmarks/exporter-observable-policy.json"),
        &std::collections::BTreeSet::new(),
    )
    .expect("checked-in exporter policy validates");
    let exporter_runner =
        ExporterHarnessRunner::new(policy).expect("exporter harness owns its exact corpus");
    assert_eq!(report.corpus_blake3, exporter_runner.corpus_blake3());
    assert_eq!(
        digest(&report.runtime_evidence_bytes),
        report.runtime_evidence_blake3
    );
    let evidence: serde_json::Value = serde_json::from_slice(&report.runtime_evidence_bytes)
        .expect("runtime evidence is canonical JSON");
    assert_eq!(
        evidence["terminal_output_blake3"]
            .as_array()
            .expect("terminal output digests are retained")
            .len(),
        771
    );
    assert!(report.statistical_report.is_none());
    assert!(
        report.attempt_history[0]
            .reason
            .as_deref()
            .expect("product failure retains its reason")
            .contains("bare exporter metric")
    );
}

#[test]
fn unrelated_in_process_exporter_workload_is_rejected_before_measurement() {
    let mut fixture = Fixture::new();
    fixture.static_artifact = runtime_artifact_rejecting_exporter("static authority fixture");
    fixture.dynamic_artifact = runtime_artifact_rejecting_exporter("dynamic authority fixture");
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &fixture.static_artifact,
    );
    write_executable(
        &fixture.dynamic_source.join("artifact-source"),
        &fixture.dynamic_artifact,
    );
    let build_report = fixture.build_report();
    let mut factory = FakeExporterFactory {
        mode: FakeExporterMode::ProductFailure,
    };

    let error = run_controlled_runtime_with_exporters_v1(&build_report, &mut factory)
        .expect_err("an unrelated in-process workload cannot acquire artifact authority");

    assert!(
        error
            .to_string()
            .contains("unrelated in-process exporter workload")
    );
}

#[test]
fn unrelated_exporter_adapter_modes_are_refused_before_measurement() {
    for mode in [
        FakeExporterMode::AcquisitionFailure,
        FakeExporterMode::ProductFailure,
    ] {
        let fixture = Fixture::new();
        let build_report = fixture.build_report();
        let mut factory = FakeExporterFactory { mode };

        let error = run_controlled_runtime_with_exporters_v1(&build_report, &mut factory)
            .expect_err("unrelated adapters cannot acquire artifact authority");

        assert!(
            error
                .to_string()
                .contains("unrelated in-process exporter workload")
        );
    }
}

#[test]
fn standalone_caller_authored_evaluation_remains_refused() {
    let directory = tempfile::tempdir().expect("temporary refusal fixture");
    let input = directory.path().join("forged.json");
    std::fs::write(&input, b"{\"cases\":[]}").expect("forged input is written");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_plugin_runtime_bench"))
        .args(["evaluate", input.to_str().expect("fixture path is UTF-8")])
        .output()
        .expect("standalone refusal executes");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("same-process controlled measurement capability")
    );
}

#[test]
fn terminal_failure_is_retained_once_with_its_exact_empty_output_digest() {
    let mut fixture = Fixture::new();
    let failing_artifact = b"#!/bin/sh\nexit 9\n".to_vec();
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &failing_artifact,
    );
    fixture.static_artifact = failing_artifact;
    let build_report = fixture.build_report();

    let report = run_controlled_runtime_with_ledger_v1(
        &build_report,
        &fixture._directory.path().join("terminal-attempts.jsonl"),
    )
    .expect("terminal product failure is retained as an authoritative outcome");

    assert_eq!(report.decision, ControlledAttemptDecision::ValidFailure);
    assert_eq!(report.attempt_history.len(), 1);
    assert_eq!(report.executed_member_count, 1);
    assert_eq!(report.terminal_output_blake3, [digest(b"")]);
    assert!(report.statistical_report.is_none());
}

#[test]
fn valid_terminal_attempt_refuses_a_second_runner_invocation() {
    let mut fixture = Fixture::new();
    let failing_artifact = b"#!/bin/sh\nexit 9\n".to_vec();
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &failing_artifact,
    );
    fixture.static_artifact = failing_artifact;
    let build_report = fixture.build_report();
    let ledger = fixture._directory.path().join("attempt-ledger.jsonl");

    let first = run_controlled_runtime_with_ledger_v1(&build_report, &ledger)
        .expect("first invocation records its terminal attempt");
    assert_eq!(first.decision, ControlledAttemptDecision::ValidFailure);

    let error = run_controlled_runtime_with_ledger_v1(&build_report, &ledger)
        .expect_err("the first valid attempt is authoritative across invocations");
    assert!(
        error
            .to_string()
            .contains("first valid experiment attempt is authoritative")
    );
    assert_eq!(
        std::fs::read(controlled_attempt_ledger_path(
            fixture._directory.path(),
            &first.experiment_identity_blake3
        ))
        .expect("attempt ledger is retained")
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .count(),
        1
    );
}

#[test]
fn controller_affinity_monitor_replaces_the_whole_pair_in_seeded_order() {
    let mut fixture = Fixture::new();
    let marker = fixture._directory.path().join("affinity-loss-once");
    fixture.static_artifact =
        runtime_artifact_with_one_affinity_loss("static authority fixture", &marker);
    fixture.dynamic_artifact =
        runtime_artifact_with_one_affinity_loss("dynamic authority fixture", &marker);
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &fixture.static_artifact,
    );
    write_executable(
        &fixture.dynamic_source.join("artifact-source"),
        &fixture.dynamic_artifact,
    );
    let build_report = fixture.build_report();

    let report = run_controlled_runtime_with_ledger_v1(
        &build_report,
        &fixture._directory.path().join("replacement-attempts.jsonl"),
    )
    .expect("controller completes after one infrastructure replacement");

    let pair_attempts = report
        .raw_pair_history
        .iter()
        .filter(|record| {
            record.raw.scenario == "http_non_streaming_c1" && record.raw.pair_id == "pair-00"
        })
        .collect::<Vec<_>>();
    assert_eq!(pair_attempts.len(), 2);
    assert!(matches!(
        pair_attempts[0].decision,
        PairAttemptDecision::ReplaceWholePair {
            replacement_ordinal: 1,
            ..
        }
    ));
    assert_eq!(pair_attempts[0].derived_reason, "affinity_loss");
    assert_eq!(
        pair_attempts[0].raw.member_order,
        pair_attempts[1].raw.member_order
    );
    assert_eq!(pair_attempts[0].raw.members.len(), 2);
    for member in &pair_attempts[0].raw.members {
        assert!(!member.samples.is_empty());
        let terminal = &report.terminal_member_evidence[member
            .terminal_evidence_index
            .expect("raw member references bounded terminal evidence")];
        assert_eq!(terminal.scenario, pair_attempts[0].raw.scenario);
        assert_eq!(terminal.pair_id, pair_attempts[0].raw.pair_id);
        assert_eq!(terminal.variant, member.variant);
    }
    assert_eq!(pair_attempts[1].decision, PairAttemptDecision::RetainPair);
    // The sealed matrix ends on the exporter scenario, which the checked-in
    // calibration-mode observable policy cannot admit as a pair, so the attempt
    // is a product failure rather than a complete statistical attempt. The
    // replacement itself stays fully observable in retained raw evidence.
    let replacements = report
        .raw_pair_history
        .iter()
        .filter(|record| {
            matches!(
                record.decision,
                PairAttemptDecision::ReplaceWholePair { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(replacements.len(), 1);
    assert_eq!(replacements[0].derived_reason, "affinity_loss");
    assert_eq!(
        digest(&report.attempt_evidence_tree_bytes),
        report.attempt_evidence_tree_blake3
    );
    assert_eq!(
        report.attempt_evidence_tree_blake3,
        report.attempt_history[0].evidence_tree_blake3
    );
    let runtime_evidence: serde_json::Value =
        serde_json::from_slice(&report.runtime_evidence_bytes)
            .expect("runtime evidence is canonical JSON");
    assert_eq!(
        runtime_evidence["attempt_evidence_tree_blake3"],
        report.attempt_evidence_tree_blake3
    );
    assert_eq!(
        runtime_evidence["raw_pair_history"]
            .as_array()
            .expect("raw pair history is retained")
            .len(),
        report.raw_pair_history.len()
    );
}

#[test]
fn one_experiment_identity_owns_one_ledger_across_requested_output_paths() {
    let mut fixture = Fixture::new();
    let failing_artifact = b"#!/bin/sh\nexit 9\n".to_vec();
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &failing_artifact,
    );
    fixture.static_artifact = failing_artifact;
    let build_report = fixture.build_report();
    let first_request = fixture._directory.path().join("first-ledger.jsonl");
    let second_request = fixture._directory.path().join("second-ledger.jsonl");

    let first = run_controlled_runtime_with_ledger_v1(&build_report, &first_request)
        .expect("first invocation records its terminal attempt");
    assert_eq!(first.decision, ControlledAttemptDecision::ValidFailure);

    let error = run_controlled_runtime_with_ledger_v1(&build_report, &second_request)
        .expect_err("a second requested filename cannot restart the same identity");
    assert!(
        error
            .to_string()
            .contains("first valid experiment attempt is authoritative"),
        "unexpected error: {error}"
    );

    let derived = controlled_attempt_ledger_path(
        fixture._directory.path(),
        &first.experiment_identity_blake3,
    );
    assert_eq!(
        std::fs::read(&derived)
            .expect("identity-derived ledger is retained")
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .count(),
        1
    );
    assert!(!first_request.exists());
    assert!(!second_request.exists());
}

#[test]
fn three_invalid_attempts_block_the_identity_across_requested_output_paths() {
    let mut fixture = Fixture::new();
    fixture.static_artifact = runtime_artifact_with_persistent_affinity_loss("static authority fixture");
    fixture.dynamic_artifact =
        runtime_artifact_with_persistent_affinity_loss("dynamic authority fixture");
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &fixture.static_artifact,
    );
    write_executable(
        &fixture.dynamic_source.join("artifact-source"),
        &fixture.dynamic_artifact,
    );
    let build_report = fixture.build_report();

    let mut identity = String::new();
    for attempt in 0..3 {
        let requested = fixture
            ._directory
            .path()
            .join(format!("attempt-{attempt}.jsonl"));
        let report = run_controlled_runtime_with_ledger_v1(&build_report, &requested)
            .expect("each invalid attempt is recorded");
        assert_eq!(report.decision, ControlledAttemptDecision::Invalid);
        identity = report.experiment_identity_blake3.clone();
    }

    let error = run_controlled_runtime_with_ledger_v1(
        &build_report,
        &fixture._directory.path().join("attempt-3.jsonl"),
    )
    .expect_err("three invalid attempts block another invocation");
    assert!(
        error
            .to_string()
            .contains("three invalid attempts block the experiment"),
        "unexpected error: {error}"
    );

    let derived = controlled_attempt_ledger_path(fixture._directory.path(), &identity);
    assert_eq!(
        std::fs::read(&derived)
            .expect("identity-derived ledger is retained")
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .count(),
        3
    );
}

#[test]
fn controller_observed_host_reboot_replaces_the_whole_pair_in_seeded_order() {
    let mut fixture = Fixture::new();
    let boot_identity = fixture._directory.path().join("boot-identity");
    std::fs::write(&boot_identity, "boot-original").expect("boot identity fixture is written");
    let marker = fixture._directory.path().join("reboot-once");
    fixture.static_artifact =
        runtime_artifact_with_one_reboot("static authority fixture", &boot_identity, &marker);
    fixture.dynamic_artifact =
        runtime_artifact_with_one_reboot("dynamic authority fixture", &boot_identity, &marker);
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &fixture.static_artifact,
    );
    write_executable(
        &fixture.dynamic_source.join("artifact-source"),
        &fixture.dynamic_artifact,
    );
    let build_report = fixture.build_report();

    let report = run_controlled_runtime_with_liveness_v1(
        &build_report,
        &fixture._directory.path().join("reboot-attempts.jsonl"),
        &HostLivenessSourceV1::new(boot_identity.clone(), None),
    )
    .expect("controller completes after one observed reboot");

    let attempts = report
        .raw_pair_history
        .iter()
        .filter(|record| {
            record.raw.scenario == "http_non_streaming_c1" && record.raw.pair_id == "pair-00"
        })
        .collect::<Vec<_>>();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].derived_reason, "host_reboot");
    assert!(matches!(
        attempts[0].decision,
        PairAttemptDecision::ReplaceWholePair {
            replacement_ordinal: 1,
            ..
        }
    ));
    assert_eq!(attempts[0].raw.member_order, attempts[1].raw.member_order);
    assert_eq!(attempts[1].decision, PairAttemptDecision::RetainPair);
}

#[test]
fn controller_observed_mock_death_replaces_the_whole_pair_in_seeded_order() {
    let mut fixture = Fixture::new();
    let boot_identity = fixture._directory.path().join("boot-identity");
    std::fs::write(&boot_identity, "boot-stable").expect("boot identity fixture is written");
    let mock_pid_path = fixture._directory.path().join("mock-server.pid");
    let mut mock = std::process::Command::new("/bin/sh")
        .args(["-c", "sleep 300"])
        .spawn()
        .expect("mock stand-in starts");
    std::fs::write(&mock_pid_path, mock.id().to_string()).expect("mock pid file is written");
    let marker = fixture._directory.path().join("mock-death-once");
    fixture.static_artifact =
        runtime_artifact_killing_the_mock_once("static authority fixture", &mock_pid_path, &marker);
    fixture.dynamic_artifact = runtime_artifact_killing_the_mock_once(
        "dynamic authority fixture",
        &mock_pid_path,
        &marker,
    );
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &fixture.static_artifact,
    );
    write_executable(
        &fixture.dynamic_source.join("artifact-source"),
        &fixture.dynamic_artifact,
    );
    let build_report = fixture.build_report();

    let report = run_controlled_runtime_with_liveness_v1(
        &build_report,
        &fixture._directory.path().join("mock-death-attempts.jsonl"),
        &HostLivenessSourceV1::new(boot_identity.clone(), Some(mock_pid_path.clone())),
    )
    .expect("controller completes after one observed mock death");
    let _ = mock.kill();
    let _ = mock.wait();

    let attempts = report
        .raw_pair_history
        .iter()
        .filter(|record| {
            record.raw.scenario == "http_non_streaming_c1" && record.raw.pair_id == "pair-00"
        })
        .collect::<Vec<_>>();
    assert_eq!(attempts.len(), 2);
    assert_eq!(
        attempts[0].derived_reason,
        "mock_death_unrelated_to_member"
    );
    assert!(matches!(
        attempts[0].decision,
        PairAttemptDecision::ReplaceWholePair {
            replacement_ordinal: 1,
            ..
        }
    ));
    assert_eq!(attempts[0].raw.member_order, attempts[1].raw.member_order);
}

fn run_until_the_replacement_cap(
    build_report: &BuildPairReportV1,
    boot_identity: &Path,
    requested: &Path,
) -> ControlledRuntimeReportV1 {
    run_controlled_runtime_with_liveness_v1(
        build_report,
        requested,
        &HostLivenessSourceV1::new(boot_identity.to_path_buf(), None),
    )
    .expect("controller reaches a terminal decision under continuous reboots")
}

#[test]
fn continuous_reboots_exhaust_the_five_pair_replacement_cap_and_resume_diagnoses_the_reboot() {
    let mut fixture = Fixture::new();
    let boot_identity = fixture._directory.path().join("boot-identity");
    std::fs::write(&boot_identity, "boot-original").expect("boot identity fixture is written");
    fixture.static_artifact =
        runtime_artifact_rebooting_every_pair("static authority fixture", &boot_identity);
    fixture.dynamic_artifact =
        runtime_artifact_rebooting_every_pair("dynamic authority fixture", &boot_identity);
    write_executable(
        &fixture.static_source.join("artifact-source"),
        &fixture.static_artifact,
    );
    write_executable(
        &fixture.dynamic_source.join("artifact-source"),
        &fixture.dynamic_artifact,
    );
    let build_report = fixture.build_report();
    let first_request = fixture._directory.path().join("cap-attempt-1.jsonl");
    let second_request = fixture._directory.path().join("cap-attempt-2.jsonl");

    let first = run_until_the_replacement_cap(&build_report, &boot_identity, &first_request);

    assert_eq!(first.decision, ControlledAttemptDecision::Invalid);
    let replacements = first
        .raw_pair_history
        .iter()
        .filter(|record| {
            matches!(
                record.decision,
                PairAttemptDecision::ReplaceWholePair { .. }
            )
        })
        .count();
    assert_eq!(replacements, 5);
    assert!(
        first
            .raw_pair_history
            .last()
            .is_some_and(|record| record.decision == PairAttemptDecision::AttemptInvalid)
    );

    // A reboot between invocations is diagnosed from the persisted pair-start
    // context, not from anything the interrupted children reported.
    std::fs::write(&boot_identity, "boot-after-restart").expect("restart boot identity is written");
    let second = run_until_the_replacement_cap(&build_report, &boot_identity, &second_request);

    let resumed = second
        .resumed_pair_context
        .as_ref()
        .expect("the restart diagnoses the interrupted pair");
    assert_eq!(resumed.attempt_ordinal, 1);
    assert_eq!(resumed.scenario, "http_non_streaming_c1");
    assert_eq!(resumed.pair_id, "pair-00");
    assert_ne!(resumed.observed.boot_identity, "boot-after-restart");
}
