// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, fs, num::NonZeroUsize};

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, HarborImporter, HarborSource, ModelIdentity,
    NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity, ResourceLeaseRequest,
    RuntimeIdentity, SuiteError, SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec,
    parse_native_graph_suite_toml,
};

#[test]
fn resolving_a_suite_expands_repetitions_in_authored_trial_order() {
    let first_task = native_task_fixture();
    let first_source = HarborSource::local(first_task.path().to_string_lossy()).unwrap();
    let first_imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&first_source)
        .unwrap();
    let second_task = native_task_fixture();
    fs::write(
        second_task.path().join("agent_graph.json"),
        b"{\"second\":true}\n",
    )
    .unwrap();
    let second_source = HarborSource::local(second_task.path().to_string_lossy()).unwrap();
    let second_imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&second_source)
        .unwrap();
    fs::write(
        first_task.path().join("agent_graph.json"),
        b"{\"mutated-after-import\":true}\n",
    )
    .unwrap();
    let lease = ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap();
    let first_trial = trial(first_imported.task.clone(), 7);
    let second_trial = trial(second_imported.task.clone(), 11);
    let manifest = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            first_imported,
            first_trial,
            NonZeroUsize::new(2).unwrap(),
            lease.clone(),
        )
        .unwrap(),
        SuiteTrialSpec::from_imported(
            second_imported,
            second_trial,
            NonZeroUsize::new(1).unwrap(),
            lease,
        )
        .unwrap(),
    ])
    .unwrap();

    let resolved = manifest
        .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(b"run-one")))
        .unwrap();
    let rerun = manifest
        .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(b"run-two")))
        .unwrap();

    assert_eq!(resolved.manifest_digest(), &manifest.identity_digest());
    assert_eq!(
        resolved
            .trials()
            .iter()
            .map(|trial| (
                trial.manifest_index(),
                trial.repetition_index(),
                trial.trial().seed
            ))
            .collect::<Vec<_>>(),
        vec![(0, 0, 7), (0, 1, 7), (1, 0, 11)]
    );
    assert_eq!(
        resolved
            .trials()
            .iter()
            .map(|trial| trial.assignment_id().as_str())
            .collect::<Vec<_>>(),
        rerun
            .trials()
            .iter()
            .map(|trial| trial.assignment_id().as_str())
            .collect::<Vec<_>>()
    );
    assert_ne!(
        resolved.trials()[0].attempt_id(),
        rerun.trials()[0].attempt_id()
    );
    assert_eq!(
        resolved.trials()[0]
            .package()
            .native_graph()
            .unwrap()
            .program_source()
            .unwrap()
            .bytes(),
        b"{}\n"
    );
    assert_ne!(
        resolved.trials()[0]
            .package()
            .native_graph()
            .unwrap()
            .program_source()
            .unwrap()
            .digest(),
        resolved.trials()[2]
            .package()
            .native_graph()
            .unwrap()
            .program_source()
            .unwrap()
            .digest()
    );
}

#[test]
fn strict_suite_toml_resolves_ordered_two_task_axes_with_resource_limits() {
    let first_task = native_task_fixture();
    let second_task = native_task_fixture_with_model_binding("secondary", "secondary-model");
    let first_source = HarborSource::local(first_task.path().to_string_lossy()).unwrap();
    let second_source = HarborSource::local(second_task.path().to_string_lossy()).unwrap();
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first_imported = importer.import(&first_source).unwrap();
    let second_imported = importer.import(&second_source).unwrap();
    let suite_toml = two_task_suite_toml(
        first_task.path().to_string_lossy().as_ref(),
        &first_imported.task,
        second_task.path().to_string_lossy().as_ref(),
        &second_imported.task,
        "secondary",
        64,
    );
    let suite = parse_native_graph_suite_toml(suite_toml.as_bytes())
        .unwrap()
        .resolve(&importer)
        .unwrap();
    fs::write(
        first_task.path().join("agent_graph.json"),
        b"{\"mutated-after-suite-import\":true}\n",
    )
    .unwrap();

    assert_eq!(suite.resource_limits().episode_slots(), 2);
    assert_eq!(suite.resource_limits().cpu_units(), 3);
    assert_eq!(suite.resource_limits().memory_bytes(), 128);
    let primary_capacity = aiperf_runtime::eval::ModelCapacityKey::from_task_binding(
        &first_imported.task,
        &first_imported
            .package
            .native_graph()
            .unwrap()
            .model_bindings()[0],
    );
    let secondary_capacity = aiperf_runtime::eval::ModelCapacityKey::from_task_binding(
        &second_imported.task,
        &second_imported
            .package
            .native_graph()
            .unwrap()
            .model_bindings()[0],
    );
    assert_eq!(
        suite
            .resource_limits()
            .model_binding_units()
            .get(&primary_capacity),
        Some(&1)
    );
    assert_eq!(
        suite
            .resource_limits()
            .model_binding_units()
            .get(&secondary_capacity),
        Some(&2)
    );
    assert_eq!(suite.manifest().trials().len(), 5);
    let resolved = suite
        .manifest()
        .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
            b"suite-toml-run",
        )))
        .unwrap();
    assert_eq!(resolved.trials().len(), 5);
    let policy_a = ArtifactDigest::from_bytes(b"policy-a");
    let policy_b = ArtifactDigest::from_bytes(b"policy-b");
    let policy_c = ArtifactDigest::from_bytes(b"policy-c");
    assert_eq!(
        resolved
            .trials()
            .iter()
            .map(|trial| (
                trial.trial().agent.as_str(),
                trial.trial().model.model.as_str(),
                trial.trial().policy.digest().as_str(),
                trial.trial().seed,
            ))
            .collect::<Vec<_>>(),
        vec![
            ("graph-a", "example-model", policy_a.as_str(), 17,),
            ("graph-a", "example-model", policy_b.as_str(), 17,),
            ("graph-b", "example-model", policy_a.as_str(), 17,),
            ("graph-b", "example-model", policy_b.as_str(), 17,),
            ("graph-c", "secondary-model", policy_c.as_str(), 23,),
        ]
    );
    assert_eq!(
        resolved.trials()[0]
            .paired_factors()
            .get("prompt_set")
            .map(String::as_str),
        Some("paired-a")
    );
    assert_eq!(
        resolved.trials()[4]
            .paired_factors()
            .get("prompt_set")
            .map(String::as_str),
        Some("paired-b")
    );
    assert_eq!(
        (
            resolved.trials()[0].resources().cpu_units(),
            resolved.trials()[0].resources().memory_bytes(),
        ),
        (1, 64)
    );
    assert_eq!(
        (
            resolved.trials()[4].resources().cpu_units(),
            resolved.trials()[4].resources().memory_bytes(),
        ),
        (2, 32)
    );
    assert_eq!(
        resolved.trials()[0]
            .resources()
            .model_binding_units()
            .get(&primary_capacity),
        Some(&1)
    );
    assert_eq!(
        resolved.trials()[4]
            .resources()
            .model_binding_units()
            .get(&secondary_capacity),
        Some(&1)
    );
    assert_eq!(
        resolved.trials()[0]
            .selected_model_binding()
            .binding_id()
            .as_str(),
        "primary"
    );
    assert_eq!(
        resolved.trials()[4]
            .selected_model_binding()
            .binding_id()
            .as_str(),
        "secondary"
    );
    assert_ne!(
        resolved.trials()[0]
            .selected_model_binding()
            .identity_digest(),
        resolved.trials()[4]
            .selected_model_binding()
            .identity_digest()
    );
    assert_eq!(
        resolved.trials()[0]
            .resources()
            .model_binding_units()
            .get(resolved.trials()[0].selected_model_binding().capacity_key()),
        Some(&1)
    );
    assert_eq!(
        resolved.trials()[0]
            .package()
            .native_graph()
            .unwrap()
            .program_source()
            .unwrap()
            .bytes(),
        b"{}\n"
    );
}

#[test]
fn strict_suite_toml_rejects_unknown_fields_mismatched_task_and_unbounded_expansion() {
    let task = native_task_fixture();
    let source = HarborSource::local(task.path().to_string_lossy()).unwrap();
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let imported = importer.import(&source).unwrap();
    let unknown_field = format!(
        "{}\nunknown = true\n",
        suite_toml(task.path().to_string_lossy().as_ref(), &imported.task, 64)
    );
    assert!(parse_native_graph_suite_toml(unknown_field.as_bytes()).is_err());

    let mismatched_toml = suite_toml(
        task.path().to_string_lossy().as_ref(),
        &aiperf_runtime::eval::EvalTaskRef::new(
            imported.task.id.as_str(),
            ArtifactDigest::from_bytes(b"wrong-task"),
        )
        .unwrap(),
        64,
    );
    let mismatched = parse_native_graph_suite_toml(mismatched_toml.as_bytes())
        .unwrap()
        .resolve(&importer)
        .unwrap_err();
    assert_eq!(
        mismatched,
        SuiteError::TaskReferenceMismatch {
            task_index: 0,
            expected: aiperf_runtime::eval::EvalTaskRef::new(
                imported.task.id.as_str(),
                ArtifactDigest::from_bytes(b"wrong-task"),
            )
            .unwrap(),
            actual: imported.task.clone(),
        }
    );

    let expansion_toml = suite_toml(task.path().to_string_lossy().as_ref(), &imported.task, 15);
    let expansion = parse_native_graph_suite_toml(expansion_toml.as_bytes()).unwrap_err();
    assert_eq!(
        expansion,
        SuiteError::TrialExpansionLimitExceeded {
            requested: 16,
            limit: 15,
        }
    );

    let oversized = vec![b' '; 1024 * 1024 + 1];
    assert!(matches!(
        parse_native_graph_suite_toml(&oversized),
        Err(SuiteError::DocumentTooLarge { actual, .. }) if actual == oversized.len()
    ));
}

#[test]
fn strict_suite_task_resource_binding_must_exist_in_that_task_snapshot() {
    let first_task = native_task_fixture();
    let second_task = native_task_fixture_with_model_binding("secondary", "secondary-model");
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first_imported = importer
        .import(&HarborSource::local(first_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let second_imported = importer
        .import(&HarborSource::local(second_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let document = two_task_suite_toml(
        first_task.path().to_string_lossy().as_ref(),
        &first_imported.task,
        second_task.path().to_string_lossy().as_ref(),
        &second_imported.task,
        "primary",
        64,
    );

    assert_eq!(
        parse_native_graph_suite_toml(document.as_bytes())
            .unwrap()
            .resolve(&importer)
            .unwrap_err(),
        SuiteError::MissingResourceBinding {
            field: "tasks.resources.model_binding_units",
            binding: "primary".to_owned(),
        }
    );
}

#[test]
fn strict_suite_rejects_same_name_binding_aliases_with_distinct_runtime_identity() {
    let first_task = native_task_fixture();
    let second_task = native_task_fixture_with_same_name_different_runtime();
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first_imported = importer
        .import(&HarborSource::local(first_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let second_imported = importer
        .import(&HarborSource::local(second_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let document = aliased_two_task_suite_toml(
        first_task.path().to_string_lossy().as_ref(),
        &first_imported.task,
        second_task.path().to_string_lossy().as_ref(),
        &second_imported.task,
    );

    assert_eq!(
        parse_native_graph_suite_toml(document.as_bytes())
            .unwrap()
            .resolve(&importer)
            .unwrap_err(),
        SuiteError::CrossTaskModelBindingAlias {
            binding: "primary".to_owned(),
        }
    );
}

#[test]
fn model_binding_identity_covers_url_transport_tokenizer_and_generation() {
    let baseline_task = native_task_fixture();
    let baseline = imported_binding_identities(&baseline_task);

    for change in ["url", "transport", "tokenizer", "generation"] {
        let changed_task = native_task_fixture_with_runtime_change(change);
        let changed = imported_binding_identities(&changed_task);
        assert_ne!(
            baseline.0, changed.0,
            "{change} must affect binding identity"
        );
        assert_ne!(baseline.1, changed.1, "{change} must affect capacity key");
    }
}

#[test]
fn resolved_suite_identity_includes_all_scheduler_resource_limits() {
    let first_task = native_task_fixture();
    let second_task = native_task_fixture_with_model_binding("secondary", "secondary-model");
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first_imported = importer
        .import(&HarborSource::local(first_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let second_imported = importer
        .import(&HarborSource::local(second_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let document = two_task_suite_toml(
        first_task.path().to_string_lossy().as_ref(),
        &first_imported.task,
        second_task.path().to_string_lossy().as_ref(),
        &second_imported.task,
        "secondary",
        64,
    );
    let baseline = parse_native_graph_suite_toml(document.as_bytes())
        .unwrap()
        .resolve(&importer)
        .unwrap()
        .identity_digest();

    for altered in [
        document.replacen("parallelism = 2", "parallelism = 1", 1),
        document.replacen("cpu_units = 3", "cpu_units = 4", 1),
        document.replacen("memory_bytes = 128", "memory_bytes = 129", 1),
        document.replacen("secondary = 2", "secondary = 3", 1),
    ] {
        let identity = parse_native_graph_suite_toml(altered.as_bytes())
            .unwrap()
            .resolve(&importer)
            .unwrap()
            .identity_digest();
        assert_ne!(baseline, identity);
    }
}

#[test]
fn definition_resolution_includes_resource_limits_in_suite_and_assignment_identity() {
    let first_task = native_task_fixture();
    let second_task = native_task_fixture_with_model_binding("secondary", "secondary-model");
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first_imported = importer
        .import(&HarborSource::local(first_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let second_imported = importer
        .import(&HarborSource::local(second_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let document = two_task_suite_toml(
        first_task.path().to_string_lossy().as_ref(),
        &first_imported.task,
        second_task.path().to_string_lossy().as_ref(),
        &second_imported.task,
        "secondary",
        64,
    );
    let changed_document = document.replacen("memory_bytes = 128", "memory_bytes = 129", 1);
    let run_id = SuiteRunId::new(ArtifactDigest::from_bytes(b"limit-sensitive-run"));
    let baseline = parse_native_graph_suite_toml(document.as_bytes())
        .unwrap()
        .resolve(&importer)
        .unwrap()
        .resolve(run_id.clone())
        .unwrap();
    let changed = parse_native_graph_suite_toml(changed_document.as_bytes())
        .unwrap()
        .resolve(&importer)
        .unwrap()
        .resolve(run_id)
        .unwrap();

    assert_ne!(baseline.suite_digest(), changed.suite_digest());
    assert_ne!(
        baseline.trials()[0].assignment_id(),
        changed.trials()[0].assignment_id()
    );
}

#[test]
fn programmatic_trial_rejects_a_foreign_task_capacity_key() {
    let first_task = native_task_fixture();
    let second_task = native_task_fixture_with_model_binding("secondary", "secondary-model");
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first_imported = importer
        .import(&HarborSource::local(first_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let second_imported = importer
        .import(&HarborSource::local(second_task.path().to_string_lossy()).unwrap())
        .unwrap();
    let foreign_key = aiperf_runtime::eval::ModelCapacityKey::from_task_binding(
        &second_imported.task,
        &second_imported
            .package
            .native_graph()
            .unwrap()
            .model_bindings()[0],
    );
    let mut weights = BTreeMap::new();
    weights.insert(foreign_key.clone(), 1);

    assert_eq!(
        SuiteTrialSpec::from_imported(
            first_imported.clone(),
            trial(first_imported.task.clone(), 7),
            NonZeroUsize::new(1).unwrap(),
            ResourceLeaseRequest::new(1, 64, weights).unwrap(),
        )
        .unwrap_err(),
        SuiteError::ForeignResourceCapacityKey {
            key: foreign_key.digest().clone(),
        }
    );
}

#[test]
fn suite_identity_changes_for_paired_factors_and_resource_weights() {
    let task = native_task_fixture();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task.path().to_string_lossy()).unwrap())
        .unwrap();
    let repetitions = NonZeroUsize::new(1).unwrap();
    let baseline = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported_with_factors(
            imported.clone(),
            trial(imported.task.clone(), 7),
            repetitions,
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
            BTreeMap::from([("prompt_set".to_owned(), "paired-a".to_owned())]),
        )
        .unwrap(),
    ])
    .unwrap();
    let factor_changed = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported_with_factors(
            imported.clone(),
            trial(imported.task.clone(), 7),
            repetitions,
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
            BTreeMap::from([("prompt_set".to_owned(), "paired-b".to_owned())]),
        )
        .unwrap(),
    ])
    .unwrap();
    let resource_changed = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported_with_factors(
            imported.clone(),
            trial(imported.task.clone(), 7),
            repetitions,
            ResourceLeaseRequest::new(2, 64, BTreeMap::new()).unwrap(),
            BTreeMap::from([("prompt_set".to_owned(), "paired-a".to_owned())]),
        )
        .unwrap(),
    ])
    .unwrap();

    assert_ne!(baseline.identity_digest(), factor_changed.identity_digest());
    assert_ne!(
        baseline.identity_digest(),
        resource_changed.identity_digest()
    );
}

#[test]
fn large_repetition_expansion_reuses_one_imported_snapshot_per_trial_axis() {
    let task = native_task_fixture();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task.path().to_string_lossy()).unwrap())
        .unwrap();
    let manifest = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial(imported.task.clone(), 7),
            NonZeroUsize::new(10_000).unwrap(),
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let resolved = manifest
        .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
            b"large-expansion",
        )))
        .unwrap();
    assert_eq!(resolved.trials().len(), 10_000);
    assert_eq!(
        resolved.trials()[0].package().identity_digest(),
        resolved.trials()[9_999].package().identity_digest()
    );
}

fn trial(task: aiperf_runtime::eval::EvalTaskRef, seed: u64) -> TrialSpec {
    TrialSpec::new(
        task,
        AgentVariantRef::new("native-graph").unwrap(),
        ModelIdentity::new("provider-default", "example-model").unwrap(),
        seed,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).unwrap(),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").unwrap(),
    )
    .unwrap()
}

fn suite_toml(
    path: &str,
    task: &aiperf_runtime::eval::EvalTaskRef,
    expansion_limit: usize,
) -> String {
    format!(
        r#"[defaults]
runtime = "native"
execution_seconds = 30.0
verifier_seconds = 30.0
environment = "{environment}"
verifier = "{verifier}"

[limits]
parallelism = 2
cpu_units = 2
memory_bytes = 128
max_expanded_trials = {expansion_limit}

[limits.model_binding_units]
primary = 1

[[tasks]]
source = {{ kind = "local", path = {path:?} }}
task_id = "{task_id}"
task_digest = "{task_digest}"
graph_axes = ["graph-a", "graph-b"]
model_axes = ["primary"]
policy_axes = ["{first_policy}", "{second_policy}"]
seeds = [17, 19]
repetitions = 2
paired_factors = {{ prompt_set = "paired-a" }}

[tasks.resources]
cpu_units = 1
memory_bytes = 64
model_binding_units = {{ primary = 1 }}
"#,
        environment = ArtifactDigest::from_bytes(b"environment").as_str(),
        verifier = ArtifactDigest::from_bytes(b"verifier").as_str(),
        task_id = task.id.as_str(),
        task_digest = task.digest.as_str(),
        first_policy = ArtifactDigest::from_bytes(b"policy-a").as_str(),
        second_policy = ArtifactDigest::from_bytes(b"policy-b").as_str(),
    )
}

#[allow(clippy::too_many_arguments)]
fn two_task_suite_toml(
    first_path: &str,
    first_task: &aiperf_runtime::eval::EvalTaskRef,
    second_path: &str,
    second_task: &aiperf_runtime::eval::EvalTaskRef,
    second_resource_binding: &str,
    expansion_limit: usize,
) -> String {
    format!(
        r#"[defaults]
runtime = "native"
execution_seconds = 30.0
verifier_seconds = 30.0
environment = "{environment}"
verifier = "{verifier}"

[limits]
parallelism = 2
cpu_units = 3
memory_bytes = 128
max_expanded_trials = {expansion_limit}

[limits.model_binding_units]
primary = 1
secondary = 2

[[tasks]]
source = {{ kind = "local", path = {first_path:?} }}
task_id = "{first_task_id}"
task_digest = "{first_task_digest}"
graph_axes = ["graph-a", "graph-b"]
model_axes = ["primary"]
policy_axes = ["{policy_a}", "{policy_b}"]
seeds = [17]
repetitions = 1
paired_factors = {{ prompt_set = "paired-a" }}
resources = {{ cpu_units = 1, memory_bytes = 64, model_binding_units = {{ primary = 1 }} }}

[[tasks]]
source = {{ kind = "local", path = {second_path:?} }}
task_id = "{second_task_id}"
task_digest = "{second_task_digest}"
graph_axes = ["graph-c"]
model_axes = ["secondary"]
policy_axes = ["{policy_c}"]
seeds = [23]
repetitions = 1
paired_factors = {{ prompt_set = "paired-b" }}
resources = {{ cpu_units = 2, memory_bytes = 32, model_binding_units = {{ {second_resource_binding} = 1 }} }}
"#,
        environment = ArtifactDigest::from_bytes(b"environment").as_str(),
        verifier = ArtifactDigest::from_bytes(b"verifier").as_str(),
        first_task_id = first_task.id.as_str(),
        first_task_digest = first_task.digest.as_str(),
        second_task_id = second_task.id.as_str(),
        second_task_digest = second_task.digest.as_str(),
        policy_a = ArtifactDigest::from_bytes(b"policy-a").as_str(),
        policy_b = ArtifactDigest::from_bytes(b"policy-b").as_str(),
        policy_c = ArtifactDigest::from_bytes(b"policy-c").as_str(),
    )
}

fn aliased_two_task_suite_toml(
    first_path: &str,
    first_task: &aiperf_runtime::eval::EvalTaskRef,
    second_path: &str,
    second_task: &aiperf_runtime::eval::EvalTaskRef,
) -> String {
    two_task_suite_toml(
        first_path,
        first_task,
        second_path,
        second_task,
        "primary",
        64,
    )
    .replace("secondary = 2\n", "")
    .replace("model_axes = [\"secondary\"]", "model_axes = [\"primary\"]")
}

fn native_task_fixture_with_model_binding(binding_id: &str, model_id: &str) -> tempfile::TempDir {
    let task = native_task_fixture();
    let model_path = task.path().join("models.toml");
    let model_manifest = fs::read_to_string(&model_path)
        .unwrap()
        .replace("id = \"primary\"", &format!("id = \"{binding_id}\""))
        .replace(
            "model = \"example-model\"",
            &format!("model = \"{model_id}\""),
        );
    fs::write(model_path, model_manifest).unwrap();
    task
}

fn native_task_fixture_with_same_name_different_runtime() -> tempfile::TempDir {
    let task = native_task_fixture();
    let model_path = task.path().join("models.toml");
    let model_manifest = fs::read_to_string(&model_path)
        .unwrap()
        .replace(
            "urls = [\"https://provider.example/v1\"]",
            "urls = [\"https://alternate.example/v1\"]",
        )
        .replace(
            "transport_factory_id = \"http\"",
            "transport_factory_id = \"grpc\"",
        )
        .replace("revision = \"main\"", "revision = \"alternate\"")
        .replace(
            "[model_bindings.generation]\n",
            "[model_bindings.generation]\ntemperature = 0.25\n",
        );
    fs::write(model_path, model_manifest).unwrap();
    task
}

fn native_task_fixture_with_runtime_change(change: &str) -> tempfile::TempDir {
    let task = native_task_fixture();
    let model_path = task.path().join("models.toml");
    let model_manifest = fs::read_to_string(&model_path).unwrap();
    let model_manifest = match change {
        "url" => model_manifest.replace(
            "urls = [\"https://provider.example/v1\"]",
            "urls = [\"https://alternate.example/v1\"]",
        ),
        "transport" => model_manifest.replace(
            "transport_factory_id = \"http\"",
            "transport_factory_id = \"grpc\"",
        ),
        "tokenizer" => model_manifest.replace("revision = \"main\"", "revision = \"alternate\""),
        "generation" => model_manifest.replace(
            "[model_bindings.generation]\n",
            "[model_bindings.generation]\ntemperature = 0.25\n",
        ),
        _ => panic!("unsupported binding runtime change {change}"),
    };
    fs::write(model_path, model_manifest).unwrap();
    task
}

fn imported_binding_identities(task: &tempfile::TempDir) -> (ArtifactDigest, ArtifactDigest) {
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task.path().to_string_lossy()).unwrap())
        .unwrap();
    let binding = &imported.package.native_graph().unwrap().model_bindings()[0];
    (
        binding.identity_digest(),
        aiperf_runtime::eval::ModelCapacityKey::from_task_binding(&imported.task, binding)
            .digest()
            .clone(),
    )
}

fn native_task_fixture() -> tempfile::TempDir {
    let task = tempfile::tempdir().unwrap();
    fs::create_dir_all(task.path().join("environment")).unwrap();
    fs::create_dir_all(task.path().join("tests")).unwrap();
    fs::create_dir_all(task.path().join("tools")).unwrap();
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .unwrap();
    fs::write(task.path().join("instruction.md"), b"Do work.\n").unwrap();
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").unwrap();
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-suite"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .unwrap();
    fs::write(task.path().join("agent_graph.json"), b"{}\n").unwrap();
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
    .unwrap();
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.py"]
executable = "tools/adapter.py"
"#,
    )
    .unwrap();
    fs::write(task.path().join("tools/adapter.py"), b"print('adapter')\n").unwrap();
    task
}
