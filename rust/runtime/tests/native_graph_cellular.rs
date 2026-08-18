// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned NativeGraph cellular placement and fold contracts.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    num::NonZeroUsize,
};

use aiperf_runtime::{
    cellular::{CellPartition, ModuloCellPartition},
    eval::{
        AgentVariantRef, ArtifactDigest, CellularFoldError, ModelIdentity,
        NativeGraphCellLeaseAuthority, NativeGraphCellLeaseError, NativeGraphCellularPlan,
        NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity, ResourceLeaseRequest,
        ResourceLimits, RuntimeIdentity, SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec,
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
