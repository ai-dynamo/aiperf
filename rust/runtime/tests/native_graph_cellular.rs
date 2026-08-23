// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned NativeGraph cellular placement and fold contracts.
#![cfg(feature = "engine")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    num::NonZeroUsize,
};

use aiperf_runtime::{
    cellular::{CellPartition, ModuloCellPartition},
    eval::{
        AgentVariantRef, ArtifactDigest, CellularFoldError, EpisodeComparability, EpisodeExecution,
        EpisodeIntegrity, EpisodeResult, EpisodeScoreState, EvidenceEvent, EvidenceKind,
        FrozenAttemptBundle, ModelIdentity, NativeGraphAttemptAuthority,
        NativeGraphCellLeaseAuthority, NativeGraphCellLeaseError, NativeGraphCellResultAuthority,
        NativeGraphCellResultReceipt, NativeGraphCellularPlan, NativeGraphCellularReceiptError,
        NativeGraphCellularReceiptLimits, NativeGraphCompletedAttempt, NativeGraphSuiteManifest,
        NativeSourceAcquirer, PolicyIdentity, RegradeRequest, ResolvedEpisodeTrial,
        ResourceLeaseRequest, ResourceLimits, RewardDocument, RuntimeIdentity, ScoreVersion,
        SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec, VerifierResult, regrade,
    },
};

#[test]
fn controller_mints_modulo_placements_with_a_complete_disjoint_union() {
    let suite = resolved_suite(b"cellular-union");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 3).expect("three cells are valid");

    assert_eq!(plan.cell_count(), 3);
    assert_eq!(plan.placements().len(), suite.trials().len());
    for placement in plan.placements() {
        let partition = ModuloCellPartition::new(placement.cell_id(), plan.cell_count())
            .expect("cell is valid");
        assert!(partition.owns(placement.output_index() as u64));
        assert_eq!(
            placement.assignment_id(),
            suite.trials()[placement.output_index()].assignment_id()
        );
        assert_eq!(
            placement.attempt_id(),
            suite.trials()[placement.output_index()].attempt_id()
        );
    }

    let mut assigned_by_cell = BTreeMap::<u32, BTreeSet<_>>::new();
    for placement in plan.placements() {
        assigned_by_cell
            .entry(placement.cell_id())
            .or_default()
            .insert(placement.assignment_id().as_str());
    }
    let assigned = assigned_by_cell
        .values()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let expected = suite
        .trials()
        .iter()
        .map(|trial| trial.assignment_id().as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(assigned, expected);
    assert_eq!(
        assigned_by_cell.values().map(BTreeSet::len).sum::<usize>(),
        assigned.len(),
        "modulo cell assignment must be disjoint"
    );
}

#[test]
fn controller_fold_returns_cell_receipts_in_output_order() {
    let plan = NativeGraphCellularPlan::from_suite(&resolved_suite(b"cellular-order"), 3)
        .expect("three cells are valid");
    let mut fold = plan.begin_fold();

    for placement in plan.placements().iter().rev() {
        fold.accept(
            placement.cell_id(),
            placement.assignment_id(),
            placement.output_index(),
        )
        .expect("assigned receipt is accepted");
    }

    assert_eq!(
        fold.finish().expect("complete fold is ordered"),
        (0..plan.placements().len()).collect::<Vec<_>>()
    );
}

#[test]
fn controller_fold_rejects_an_unknown_assignment() {
    let plan = NativeGraphCellularPlan::from_suite(&resolved_suite(b"cellular-known"), 2)
        .expect("two cells are valid");
    let foreign = NativeGraphCellularPlan::from_suite(
        &resolved_suite_with_seeds(b"cellular-foreign", [13, 17, 19, 23]),
        2,
    )
    .expect("two cells are valid");
    let placement = &foreign.placements()[0];

    let error = plan
        .begin_fold()
        .accept(placement.cell_id(), placement.assignment_id(), ())
        .expect_err("a controller fold cannot accept a foreign assignment");
    assert!(matches!(error, CellularFoldError::UnknownAssignment { .. }));
}

#[test]
fn controller_fold_rejects_a_receipt_from_the_wrong_cell() {
    let plan = NativeGraphCellularPlan::from_suite(&resolved_suite(b"cellular-cell"), 3)
        .expect("three cells are valid");
    let placement = &plan.placements()[0];
    let wrong_cell = (placement.cell_id() + 1) % plan.cell_count();

    let error = plan
        .begin_fold()
        .accept(wrong_cell, placement.assignment_id(), ())
        .expect_err("a receipt must arrive from its controller-minted cell");
    assert!(matches!(error, CellularFoldError::WrongCell { .. }));
}

#[test]
fn controller_fold_rejects_a_duplicate_assignment() {
    let plan = NativeGraphCellularPlan::from_suite(&resolved_suite(b"cellular-duplicate"), 2)
        .expect("two cells are valid");
    let placement = &plan.placements()[0];
    let mut fold = plan.begin_fold();
    fold.accept(placement.cell_id(), placement.assignment_id(), ())
        .expect("first receipt is accepted");

    let error = fold
        .accept(placement.cell_id(), placement.assignment_id(), ())
        .expect_err("the assignment identity is append-only");
    assert!(matches!(
        error,
        CellularFoldError::DuplicateAssignment { .. }
    ));
}

#[test]
fn controller_fold_rejects_missing_assignments_before_aggregation() {
    let plan = NativeGraphCellularPlan::from_suite(&resolved_suite(b"cellular-missing"), 2)
        .expect("two cells are valid");
    let placement = &plan.placements()[0];
    let mut fold = plan.begin_fold();
    fold.accept(placement.cell_id(), placement.assignment_id(), ())
        .expect("first receipt is accepted");

    let error = fold
        .finish()
        .expect_err("a partial cellular fold cannot reach aggregation");
    assert!(matches!(error, CellularFoldError::MissingAssignment { .. }));
}

#[test]
fn controller_leases_bound_all_cells_and_fold_completed_receipts_in_output_order() {
    let suite = resolved_suite(b"cellular-global-capacity");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");
    let mut authority = NativeGraphCellLeaseAuthority::new(
        plan.clone(),
        ResourceLimits::new(1, 1, 64, BTreeMap::new()).expect("limits are valid"),
    )
    .expect("every planned request fits before any lease is issued");
    let first = &plan.placements()[0];
    let blocked = &plan.placements()[1];
    let first_lease = authority
        .issue_for_cell(first.cell_id())
        .expect("known cell is valid")
        .expect("first placement is admitted");

    assert_eq!(first_lease.assignment_id(), first.assignment_id());
    assert_eq!(first_lease.attempt_id(), first.attempt_id());
    assert!(
        authority
            .issue_for_cell(blocked.cell_id())
            .expect("known cell is valid")
            .is_none(),
        "one controller-owned capacity pool must bound all cells"
    );

    authority
        .complete_from_cell(first.cell_id(), first_lease, first.output_index())
        .expect("completing a valid lease releases the global capacity");
    for placement in plan.placements().iter().skip(1) {
        let lease = authority
            .issue_for_cell(placement.cell_id())
            .expect("known cell is valid")
            .expect("the released capacity admits the next placement");
        assert_eq!(lease.assignment_id(), placement.assignment_id());
        authority
            .complete_from_cell(placement.cell_id(), lease, placement.output_index())
            .expect("controller accepts exactly its issued lease");
    }

    assert_eq!(
        authority.finish().expect("every planned receipt completed"),
        (0..plan.placements().len()).collect::<Vec<_>>()
    );
}

#[test]
fn controller_lease_rejects_wrong_cell_and_replay() {
    let suite = resolved_suite(b"cellular-lease-replay");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");
    let mut authority = NativeGraphCellLeaseAuthority::<()>::new(
        plan.clone(),
        ResourceLimits::new(1, 1, 64, BTreeMap::new()).expect("limits are valid"),
    )
    .expect("every planned request fits");
    let placement = &plan.placements()[0];
    let lease = authority
        .issue_for_cell(placement.cell_id())
        .expect("known cell is valid")
        .expect("placement is admitted");
    let wrong_cell = (placement.cell_id() + 1) % plan.cell_count();

    let error = authority
        .complete_from_cell(wrong_cell, lease.clone(), ())
        .expect_err("a lease cannot be completed by another cell");
    assert!(matches!(error, NativeGraphCellLeaseError::WrongCell { .. }));

    authority
        .complete_from_cell(placement.cell_id(), lease.clone(), ())
        .expect("the issuing cell completes the lease once");
    let error = authority
        .complete_from_cell(placement.cell_id(), lease, ())
        .expect_err("a completed lease cannot be replayed");
    assert!(matches!(
        error,
        NativeGraphCellLeaseError::ReplayedLease { .. }
    ));
}

#[test]
fn controller_abort_releases_capacity_without_accepting_a_receipt() {
    let suite = resolved_suite(b"cellular-lease-abort");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");
    let mut authority = NativeGraphCellLeaseAuthority::<()>::new(
        plan.clone(),
        ResourceLimits::new(1, 1, 64, BTreeMap::new()).expect("limits are valid"),
    )
    .expect("every planned request fits");
    let first = &plan.placements()[0];
    let second = &plan.placements()[1];
    let lease = authority
        .issue_for_cell(first.cell_id())
        .expect("known cell is valid")
        .expect("first placement is admitted");

    assert!(
        authority
            .issue_for_cell(second.cell_id())
            .expect("known cell is valid")
            .is_none(),
        "the second cell must wait for the controller-owned slot"
    );
    authority
        .abort(lease)
        .expect("aborting an issued lease returns its reservation");

    assert!(
        authority
            .issue_for_cell(second.cell_id())
            .expect("known cell is valid")
            .is_some(),
        "an aborted lease must not strand global capacity"
    );
}

#[test]
fn controller_preflights_every_planned_resource_request_before_issuing() {
    let suite = resolved_suite(b"cellular-lease-preflight");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");

    let result = NativeGraphCellLeaseAuthority::<()>::new(
        plan,
        ResourceLimits::new(1, 1, 63, BTreeMap::new()).expect("limits are valid"),
    );

    assert!(matches!(
        result,
        Err(NativeGraphCellLeaseError::InvalidResourceRequest {
            output_index: 0,
            ..
        })
    ));
}

#[test]
fn sealed_cell_receipts_fold_in_controller_order_and_preserve_zero_score() {
    let suite = resolved_suite(b"cellular-sealed-order");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");
    let mut authority = NativeGraphCellResultAuthority::new(
        plan.clone(),
        ResourceLimits::new(4, 4, 256, BTreeMap::new()).expect("limits are valid"),
        receipt_limits(1),
    )
    .expect("controller receipt authority initializes");
    let mut assignments = plan
        .placements()
        .iter()
        .map(|placement| {
            authority
                .issue_for_cell(placement.cell_id())
                .expect("known cell is valid")
                .expect("capacity admits every placement")
        })
        .collect::<Vec<_>>();
    assignments.reverse();

    for assignment in assignments {
        let trial = &suite.trials()[assignment.output_index()];
        let reward = assignment.output_index() as f64;
        let completed = completed_attempt(trial, reward);
        let result = completed_result(&completed, reward);
        let receipt = NativeGraphCellResultReceipt::from_completed(
            &assignment,
            &completed,
            result,
            &receipt_limits(1),
        )
        .expect("the sealed attempt binds the exact assignment");
        authority
            .complete_from_cell(assignment.cell_id(), receipt)
            .expect("controller accepts the exact issued receipt");
    }

    let results = authority.finish().expect("every issued receipt folded");
    assert_eq!(results.len(), plan.placements().len());
    assert_eq!(
        results
            .iter()
            .map(|result| result.attempt_id())
            .collect::<Vec<_>>(),
        suite
            .trials()
            .iter()
            .map(|trial| trial.attempt_id())
            .collect::<Vec<_>>(),
        "arrival order must not change controller order"
    );
    assert_eq!(results[0].verified_reward(), Some(0.0));
}

#[test]
fn sealed_cell_receipt_refuses_a_completed_attempt_from_another_assignment() {
    let suite = resolved_suite(b"cellular-sealed-identity");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");
    let mut authority = NativeGraphCellResultAuthority::new(
        plan.clone(),
        ResourceLimits::new(2, 2, 128, BTreeMap::new()).expect("limits are valid"),
        receipt_limits(1),
    )
    .expect("controller receipt authority initializes");
    let assignment = authority
        .issue_for_cell(plan.placements()[0].cell_id())
        .expect("known cell is valid")
        .expect("first assignment is issued");
    let foreign = completed_attempt(&suite.trials()[1], 1.0);
    let foreign_result = completed_result(&foreign, 1.0);

    let error = NativeGraphCellResultReceipt::from_completed(
        &assignment,
        &foreign,
        foreign_result,
        &receipt_limits(1),
    )
    .expect_err("a cell cannot relabel another sealed completion");

    assert!(matches!(
        error,
        NativeGraphCellularReceiptError::CompletedTrialMismatch { .. }
            | NativeGraphCellularReceiptError::CompletedAttemptMismatch { .. }
    ));
}

#[test]
fn controller_receipt_boundary_rejects_excess_evidence_wrong_cell_and_replay() {
    let suite = resolved_suite(b"cellular-sealed-adversarial");
    let plan = NativeGraphCellularPlan::from_suite(&suite, 2).expect("two cells are valid");
    let mut authority = NativeGraphCellResultAuthority::new(
        plan.clone(),
        ResourceLimits::new(1, 1, 64, BTreeMap::new()).expect("limits are valid"),
        receipt_limits(1),
    )
    .expect("controller receipt authority initializes");
    let assignment = authority
        .issue_for_cell(plan.placements()[0].cell_id())
        .expect("known cell is valid")
        .expect("assignment is issued");
    let completed = completed_attempt(&suite.trials()[0], 0.0);
    let mut excessive = completed_result(&completed, 0.0);
    excessive = EpisodeResult::new(
        excessive.trial_digest().clone(),
        excessive.attempt_id().clone(),
        excessive.integrity(),
        excessive.execution(),
        excessive.score(),
        excessive.comparability(),
        vec![
            completed.frozen_attempt().identity_digest(),
            ArtifactDigest::from_bytes(b"extra-cell-evidence"),
        ],
    )
    .expect("finite fixture result is valid");
    let error = NativeGraphCellResultReceipt::from_completed(
        &assignment,
        &completed,
        excessive,
        &receipt_limits(1),
    )
    .expect_err("receipt evidence must respect the controller-selected cap");
    assert!(matches!(
        error,
        NativeGraphCellularReceiptError::EvidenceLimitExceeded { .. }
    ));

    let error = NativeGraphCellResultReceipt::from_completed(
        &assignment,
        &completed,
        completed_result(&completed, 0.0),
        &NativeGraphCellularReceiptLimits::new(
            4_096,
            assignment.attempt_id().as_str().len() - 1,
            1,
        )
        .expect("fixture attempt limit is valid"),
    )
    .expect_err("receipt attempt identity must respect the controller-selected cap");
    assert!(matches!(
        error,
        NativeGraphCellularReceiptError::AttemptIdLimitExceeded { .. }
    ));

    let error = NativeGraphCellResultReceipt::from_completed(
        &assignment,
        &completed,
        completed_result(&completed, 0.0),
        &NativeGraphCellularReceiptLimits::new(1, 256, 1).expect("fixture byte limit is valid"),
    )
    .expect_err("receipt retained identity bytes must be bounded before the fold");
    assert!(matches!(
        error,
        NativeGraphCellularReceiptError::ReceiptByteLimitExceeded { .. }
    ));

    let receipt = NativeGraphCellResultReceipt::from_completed(
        &assignment,
        &completed,
        completed_result(&completed, 0.0),
        &receipt_limits(1),
    )
    .expect("valid sealed completion becomes one receipt");
    let wrong_cell = (assignment.cell_id() + 1) % plan.cell_count();
    let error = authority
        .complete_from_cell(wrong_cell, receipt.clone())
        .expect_err("only the controller-minted cell may complete its receipt");
    assert!(matches!(
        error,
        NativeGraphCellularReceiptError::WrongCell { .. }
    ));

    authority
        .complete_from_cell(assignment.cell_id(), receipt.clone())
        .expect("the issuing cell completes once");
    let error = authority
        .complete_from_cell(assignment.cell_id(), receipt)
        .expect_err("a consumed receipt grant is never reusable");
    assert!(matches!(
        error,
        NativeGraphCellularReceiptError::ReplayedGrant { .. }
    ));
}

fn resolved_suite(run: &[u8]) -> aiperf_runtime::eval::ResolvedNativeGraphSuite {
    resolved_suite_with_seeds(run, [3, 5, 7, 11])
}

fn resolved_suite_with_seeds(
    run: &[u8],
    seeds: [u64; 4],
) -> aiperf_runtime::eval::ResolvedNativeGraphSuite {
    let task = native_task_fixture();
    let source = aiperf_runtime::eval::HarborSource::local(task.path().to_string_lossy())
        .expect("fixture source is valid");
    let imported = aiperf_runtime::eval::HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture task imports");
    let resources = ResourceLeaseRequest::new(1, 64, BTreeMap::new()).expect("lease is valid");
    NativeGraphSuiteManifest::new(
        seeds
            .into_iter()
            .map(|seed| {
                SuiteTrialSpec::from_imported(
                    imported.clone(),
                    trial(imported.task.clone(), seed),
                    NonZeroUsize::new(1).expect("one repetition is nonzero"),
                    resources.clone(),
                )
                .expect("trial is valid")
            })
            .collect(),
    )
    .expect("manifest is valid")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(run)))
    .expect("suite resolves")
}

fn trial(task: aiperf_runtime::eval::EvalTaskRef, seed: u64) -> TrialSpec {
    TrialSpec::new(
        task,
        AgentVariantRef::new("native-graph").expect("variant is valid"),
        ModelIdentity::new("provider-default", "example-model").expect("model is valid"),
        seed,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).expect("budget is valid"),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").expect("runtime is valid"),
    )
    .expect("trial is valid")
}

fn receipt_limits(max_evidence_digests: usize) -> NativeGraphCellularReceiptLimits {
    NativeGraphCellularReceiptLimits::new(4_096, 256, max_evidence_digests)
        .expect("fixture receipt limits are valid")
}

fn completed_attempt(trial: &ResolvedEpisodeTrial, reward: f64) -> NativeGraphCompletedAttempt {
    let authority = NativeGraphAttemptAuthority::from_resolved_trial(trial);
    NativeGraphCompletedAttempt::freeze(&authority, frozen_harbor_attempt(&authority, reward), None)
        .expect("fixture completion freezes")
}

fn frozen_harbor_attempt(
    authority: &NativeGraphAttemptAuthority,
    reward: f64,
) -> FrozenAttemptBundle {
    let attempt = authority.attempt_id().clone();
    let verifier = VerifierResult::new(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"cellular-verifier"),
        vec![ArtifactDigest::from_bytes(b"cellular-declared-artifact")],
        RewardDocument::parse(Some(format!(r#"{{"reward":{reward}}}"#).as_bytes()), None)
            .expect("fixture reward document is valid"),
        ArtifactDigest::from_bytes(b"cellular-rationale"),
    )
    .expect("fixture verifier result is valid");
    let initial = ScoreVersion::initial(
        attempt.clone(),
        verifier.verifier.clone(),
        verifier.evidence.clone(),
        "reward",
        reward,
        ArtifactDigest::from_bytes(b"cellular-initial-score"),
    )
    .expect("fixture initial score is valid");
    let rescored = regrade(
        RegradeRequest::new(initial.clone(), verifier.clone(), "reward")
            .expect("fixture regrade request is valid"),
    )
    .expect("fixture regrade is valid");
    FrozenAttemptBundle::new(
        authority.trial_digest().clone(),
        verifier,
        vec![EvidenceEvent::new(
            attempt,
            0,
            EvidenceKind::Evaluator,
            ArtifactDigest::from_bytes(b"cellular-lifecycle"),
            None,
        )],
        vec![initial, rescored],
    )
    .expect("fixture attempt freezes")
}

fn completed_result(completed: &NativeGraphCompletedAttempt, reward: f64) -> EpisodeResult {
    EpisodeResult::new(
        completed.frozen_attempt().trial_digest().clone(),
        completed.frozen_attempt().attempt().clone(),
        EpisodeIntegrity::Valid,
        EpisodeExecution::Completed,
        EpisodeScoreState::Verified { reward },
        EpisodeComparability::Scored,
        vec![completed.frozen_attempt().identity_digest()],
    )
    .expect("fixture result is valid")
}

fn native_task_fixture() -> tempfile::TempDir {
    let task = tempfile::tempdir().expect("temporary task directory exists");
    fs::create_dir_all(task.path().join("environment")).expect("environment directory exists");
    fs::create_dir_all(task.path().join("tests")).expect("test directory exists");
    fs::create_dir_all(task.path().join("tools")).expect("tool directory exists");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("Dockerfile writes");
    fs::write(task.path().join("instruction.md"), b"Do work.\n").expect("instruction writes");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("test writes");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-cellular"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("task manifest writes");
    fs::write(task.path().join("agent_graph.json"), b"{}\n").expect("program writes");
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[model_bindings.generation]
"#,
    )
    .expect("model bindings write");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.py"]
executable = "tools/adapter.py"
"#,
    )
    .expect("adapter manifest writes");
    fs::write(task.path().join("tools/adapter.py"), b"print('adapter')\n").expect("adapter writes");
    task
}
