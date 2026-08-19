// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable suite expansion for NativeGraph episode trials.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    num::NonZeroUsize,
    rc::Rc,
};

use serde::Deserialize;

use super::matrix::ResourceLimits;
use crate::eval::{
    AgentVariantRef, ArtifactDigest, AttemptId, EvalTaskRef, HarborImporter, HarborSource,
    HarborTaskPackage, ImportedTask, ModelBindingId, ModelBindingSpec, ModelIdentity,
    PolicyIdentity, RuntimeIdentity, TrialBudget, TrialSpec,
};

const MAX_SUITE_TOML_BYTES: usize = 1024 * 1024;
const MAX_SUITE_TASKS: usize = 256;
const MAX_AXIS_VALUES: usize = 128;
const MAX_SEEDS: usize = 512;
const MAX_REPETITIONS: usize = 256;
const MAX_PAIRED_FACTORS: usize = 64;
const MAX_FACTOR_TEXT_BYTES: usize = 256;
const MAX_EXPANDED_TRIALS: usize = 10_000;

/// Immutable caller-owned namespace for one append-only suite execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuiteRunId(ArtifactDigest);

impl SuiteRunId {
    /// Creates one caller-owned run namespace from an immutable digest.
    pub fn new(digest: ArtifactDigest) -> Self {
        Self(digest)
    }

    /// Borrows the immutable run namespace digest.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.0
    }
}

/// Stable identity for one expanded suite position across independently named runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeAssignmentId(ArtifactDigest);

impl EpisodeAssignmentId {
    /// Borrows the canonical assignment identity.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Borrows the canonical assignment digest.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.0
    }
}

/// Immutable scheduler capacity namespace for one selected task model binding.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ModelCapacityKey(ArtifactDigest);

impl ModelCapacityKey {
    /// Derives a capacity key from the imported task and every binding runtime field.
    pub fn from_task_binding(task: &EvalTaskRef, binding: &ModelBindingSpec) -> Self {
        let binding_identity = binding.identity_digest();
        let mut material = Vec::new();
        append_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-model-capacity-v1",
        );
        append_field(&mut material, "task-id", task.id.as_str().as_bytes());
        append_field(
            &mut material,
            "task-digest",
            task.digest.as_str().as_bytes(),
        );
        append_field(
            &mut material,
            "binding-identity",
            binding_identity.as_str().as_bytes(),
        );
        Self(ArtifactDigest::from_bytes(&material))
    }

    /// Borrows the opaque immutable scheduler capacity identity.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.0
    }
}

/// Full immutable selection for the binding an expanded trial will execute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedModelBinding {
    binding_id: ModelBindingId,
    identity_digest: ArtifactDigest,
    capacity_key: ModelCapacityKey,
}

impl SelectedModelBinding {
    fn from_task_binding(task: &EvalTaskRef, binding: &ModelBindingSpec) -> Self {
        Self {
            binding_id: binding.id.clone(),
            identity_digest: binding.identity_digest(),
            capacity_key: ModelCapacityKey::from_task_binding(task, binding),
        }
    }

    /// Borrows the logical binding identifier within its imported task snapshot.
    pub fn binding_id(&self) -> &ModelBindingId {
        &self.binding_id
    }

    /// Borrows the identity covering every selected binding runtime field.
    pub fn identity_digest(&self) -> &ArtifactDigest {
        &self.identity_digest
    }

    /// Borrows the package-scoped scheduler capacity key.
    pub fn capacity_key(&self) -> &ModelCapacityKey {
        &self.capacity_key
    }
}

/// Resource units one resolved episode must lease before it can execute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceLeaseRequest {
    cpu_units: u32,
    memory_bytes: u64,
    model_binding_units: BTreeMap<ModelCapacityKey, u32>,
}

impl ResourceLeaseRequest {
    /// Creates a finite positive resource request with nonzero model-binding weights.
    pub fn new(
        cpu_units: u32,
        memory_bytes: u64,
        model_binding_units: BTreeMap<ModelCapacityKey, u32>,
    ) -> Result<Self, SuiteError> {
        if cpu_units == 0 {
            return Err(SuiteError::ZeroResourceRequest("cpu_units"));
        }
        if memory_bytes == 0 {
            return Err(SuiteError::ZeroResourceRequest("memory_bytes"));
        }
        if model_binding_units.values().any(|units| *units == 0) {
            return Err(SuiteError::ZeroResourceRequest("model_binding_units"));
        }
        Ok(Self {
            cpu_units,
            memory_bytes,
            model_binding_units,
        })
    }

    /// Returns the CPU admission weight.
    pub const fn cpu_units(&self) -> u32 {
        self.cpu_units
    }

    /// Returns the memory admission weight in bytes.
    pub const fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Returns package-scoped per-model admission weights in deterministic key order.
    pub fn model_binding_units(&self) -> &BTreeMap<ModelCapacityKey, u32> {
        &self.model_binding_units
    }
}

/// One authored trial axis and its repetition count within a suite manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct SuiteTrialSpec {
    imported: Rc<ImportedTask>,
    trial: TrialSpec,
    repetitions: NonZeroUsize,
    resources: Rc<ResourceLeaseRequest>,
    paired_factors: Rc<BTreeMap<String, String>>,
    selected_model_binding: Option<SelectedModelBinding>,
}

impl SuiteTrialSpec {
    /// Combines an imported task snapshot with an exactly matching trial identity.
    pub fn from_imported(
        imported: ImportedTask,
        trial: TrialSpec,
        repetitions: NonZeroUsize,
        resources: ResourceLeaseRequest,
    ) -> Result<Self, SuiteError> {
        Self::from_imported_with_factors(imported, trial, repetitions, resources, BTreeMap::new())
    }

    /// Combines an imported task snapshot with paired-comparison factor labels.
    pub fn from_imported_with_factors(
        imported: ImportedTask,
        trial: TrialSpec,
        repetitions: NonZeroUsize,
        resources: ResourceLeaseRequest,
        paired_factors: BTreeMap<String, String>,
    ) -> Result<Self, SuiteError> {
        Self::from_snapshot_with_factors(
            Rc::new(imported),
            trial,
            repetitions,
            Rc::new(resources),
            Rc::new(paired_factors),
            None,
        )
    }

    fn from_snapshot_with_factors(
        imported: Rc<ImportedTask>,
        trial: TrialSpec,
        repetitions: NonZeroUsize,
        resources: Rc<ResourceLeaseRequest>,
        paired_factors: Rc<BTreeMap<String, String>>,
        selected_model_binding: Option<SelectedModelBinding>,
    ) -> Result<Self, SuiteError> {
        let Some(native_graph) = imported.package.native_graph() else {
            return Err(SuiteError::NotNativeGraphTask);
        };
        if trial.task != imported.task {
            return Err(SuiteError::TrialTaskMismatch);
        }
        let declared_capacity_keys = native_graph
            .model_bindings()
            .iter()
            .map(|binding| ModelCapacityKey::from_task_binding(&imported.task, binding))
            .collect::<BTreeSet<_>>();
        if let Some(foreign_key) = resources
            .model_binding_units()
            .keys()
            .find(|key| !declared_capacity_keys.contains(*key))
        {
            return Err(SuiteError::ForeignResourceCapacityKey {
                key: foreign_key.digest().clone(),
            });
        }
        let selected_model_binding = match (native_graph.profile(), selected_model_binding) {
            (super::NativeGraphProfile::ExternallyDriven, None) => None,
            (super::NativeGraphProfile::ExternallyDriven, Some(_)) => {
                return Err(SuiteError::ExternalTrialModelBinding);
            }
            (super::NativeGraphProfile::NativeGraph, Some(binding)) => Some(binding),
            (super::NativeGraphProfile::NativeGraph, None) => Some(select_trial_model_binding(
                &imported.task,
                native_graph.model_bindings(),
                &trial,
            )?),
        };
        Ok(Self {
            imported,
            trial,
            repetitions,
            resources,
            paired_factors,
            selected_model_binding,
        })
    }

    /// Borrows the importer-owned package snapshot used for every repetition.
    pub fn package(&self) -> &HarborTaskPackage {
        &self.imported.package
    }

    /// Borrows the complete immutable imported-task snapshot used by this trial.
    pub fn imported(&self) -> &ImportedTask {
        &self.imported
    }

    /// Borrows the content-addressed task identity derived during import.
    pub fn task(&self) -> &EvalTaskRef {
        &self.imported.task
    }

    /// Borrows the immutable trial identity.
    pub fn trial(&self) -> &TrialSpec {
        &self.trial
    }

    /// Returns the authored repetition count.
    pub const fn repetitions(&self) -> NonZeroUsize {
        self.repetitions
    }

    /// Borrows the resources required by every repetition.
    pub fn resources(&self) -> &ResourceLeaseRequest {
        &self.resources
    }

    /// Borrows paired-comparison factor labels in canonical key order.
    pub fn paired_factors(&self) -> &BTreeMap<String, String> {
        &self.paired_factors
    }

    /// Borrows the complete immutable model-binding selection for this trial axis.
    pub fn selected_model_binding(&self) -> Option<&SelectedModelBinding> {
        self.selected_model_binding.as_ref()
    }
}

/// Ordered NativeGraph suite definition retained before environment provisioning.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphSuiteManifest {
    trials: Vec<SuiteTrialSpec>,
}

impl NativeGraphSuiteManifest {
    /// Creates an ordered suite of independently acquired NativeGraph task references.
    pub fn new(trials: Vec<SuiteTrialSpec>) -> Result<Self, SuiteError> {
        if trials.is_empty() {
            return Err(SuiteError::EmptyManifest);
        }
        if trials.len() > MAX_EXPANDED_TRIALS {
            return Err(SuiteError::ManifestTrialLimitExceeded {
                requested: trials.len(),
                limit: MAX_EXPANDED_TRIALS,
            });
        }
        let external_trials = trials
            .iter()
            .filter(|trial| {
                trial.package().native_graph().is_some_and(|plan| {
                    plan.profile() == super::NativeGraphProfile::ExternallyDriven
                })
            })
            .count();
        if external_trials != 0 && external_trials != trials.len() {
            return Err(SuiteError::MixedNativeGraphProfiles);
        }
        if external_trials > 1 {
            return Err(SuiteError::ExternalManifestTrialAxes {
                requested: external_trials,
            });
        }
        if external_trials == 1 && trials[0].repetitions().get() != 1 {
            return Err(SuiteError::ExternalTrialRepetitionCount {
                requested: trials[0].repetitions().get(),
            });
        }
        Ok(Self { trials })
    }

    /// Borrows the authored trial axes in manifest order.
    pub fn trials(&self) -> &[SuiteTrialSpec] {
        &self.trials
    }

    /// Computes the immutable identity of this resolved suite definition.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_field(&mut material, "domain", b"aiperf-native-graph-suite-v1");
        append_field(
            &mut material,
            "trial-count",
            self.trials.len().to_string().as_bytes(),
        );
        for (index, trial) in self.trials.iter().enumerate() {
            append_field(&mut material, "trial-index", index.to_string().as_bytes());
            append_field(
                &mut material,
                "task-id",
                trial.task().id.as_str().as_bytes(),
            );
            append_field(
                &mut material,
                "task-digest",
                trial.task().digest.as_str().as_bytes(),
            );
            append_field(
                &mut material,
                "package-digest",
                trial.package().identity_digest().as_str().as_bytes(),
            );
            append_field(
                &mut material,
                "trial-digest",
                trial.trial.identity_digest().as_str().as_bytes(),
            );
            append_field(
                &mut material,
                "repetitions",
                trial.repetitions.get().to_string().as_bytes(),
            );
            if let Some(binding) = &trial.selected_model_binding {
                append_field(
                    &mut material,
                    "selected-model-binding-identity",
                    binding.identity_digest().as_str().as_bytes(),
                );
                append_field(
                    &mut material,
                    "selected-model-capacity-key",
                    binding.capacity_key().digest().as_str().as_bytes(),
                );
            } else {
                append_field(&mut material, "selected-model-binding-identity", b"none");
                append_field(&mut material, "selected-model-capacity-key", b"none");
            }
            append_resource_request(&mut material, trial.resources());
            append_paired_factors(&mut material, trial.paired_factors());
        }
        ArtifactDigest::from_bytes(&material)
    }

    /// Expands repetitions into deterministic resolved trials without executing them.
    pub fn resolve(&self, run_id: SuiteRunId) -> Result<ResolvedNativeGraphSuite, SuiteError> {
        let manifest_digest = self.identity_digest();
        self.resolve_with_suite_digest(run_id, manifest_digest)
    }

    fn resolve_with_suite_digest(
        &self,
        run_id: SuiteRunId,
        suite_digest: ArtifactDigest,
    ) -> Result<ResolvedNativeGraphSuite, SuiteError> {
        let manifest_digest = self.identity_digest();
        let total = self.trials.iter().try_fold(0usize, |total, trial| {
            total
                .checked_add(trial.repetitions.get())
                .ok_or(SuiteError::TrialExpansionOverflow)
        })?;
        if total > MAX_EXPANDED_TRIALS {
            return Err(SuiteError::TrialExpansionLimitExceeded {
                requested: total,
                limit: MAX_EXPANDED_TRIALS,
            });
        }
        let mut resolved = Vec::with_capacity(total);
        for (manifest_index, entry) in self.trials.iter().enumerate() {
            let trial_digest = entry.trial.identity_digest();
            let specification = Rc::new(entry.clone());
            for repetition_index in 0..entry.repetitions.get() {
                let resolved_digest = ArtifactDigest::from_bytes(
                    format!(
                        "suite={}\u{1f}trial={}\u{1f}repetition={}\u{1f}trial-digest={}",
                        suite_digest.as_str(),
                        manifest_index,
                        repetition_index,
                        trial_digest.as_str(),
                    )
                    .as_bytes(),
                );
                let assignment_id = EpisodeAssignmentId(resolved_digest.clone());
                let attempt_id = AttemptId::new(format!(
                    "native-graph:{}:{}",
                    run_id.digest().as_str(),
                    assignment_id.as_str(),
                ))
                .map_err(|error| SuiteError::InvalidAttemptId(error.to_string()))?;
                resolved.push(ResolvedEpisodeTrial {
                    manifest_index,
                    repetition_index,
                    specification: specification.clone(),
                    trial_digest: trial_digest.clone(),
                    resolved_digest,
                    assignment_id,
                    attempt_id,
                });
            }
        }
        Ok(ResolvedNativeGraphSuite {
            manifest_digest,
            suite_digest,
            trials: resolved,
        })
    }
}

/// Fully expanded immutable suite ready for matrix admission.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedNativeGraphSuite {
    manifest_digest: ArtifactDigest,
    suite_digest: ArtifactDigest,
    trials: Vec<ResolvedEpisodeTrial>,
}

impl ResolvedNativeGraphSuite {
    /// Borrows the immutable manifest identity used to derive all attempt identifiers.
    pub fn manifest_digest(&self) -> &ArtifactDigest {
        &self.manifest_digest
    }

    /// Borrows the full resolved-suite identity used for attempt assignment.
    pub fn suite_digest(&self) -> &ArtifactDigest {
        &self.suite_digest
    }

    /// Borrows resolved trials in their stable manifest order.
    pub fn trials(&self) -> &[ResolvedEpisodeTrial] {
        &self.trials
    }

    pub(crate) fn into_trials(self) -> Vec<ResolvedEpisodeTrial> {
        self.trials
    }
}

/// One immutable, repetition-expanded episode trial ready for exactly one attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedEpisodeTrial {
    manifest_index: usize,
    repetition_index: usize,
    specification: Rc<SuiteTrialSpec>,
    trial_digest: ArtifactDigest,
    resolved_digest: ArtifactDigest,
    assignment_id: EpisodeAssignmentId,
    attempt_id: AttemptId,
}

impl ResolvedEpisodeTrial {
    /// Returns the authored trial position within the suite manifest.
    pub const fn manifest_index(&self) -> usize {
        self.manifest_index
    }

    /// Returns the zero-based repetition within the authored trial axis.
    pub const fn repetition_index(&self) -> usize {
        self.repetition_index
    }

    /// Borrows the immutable resolved trial inputs.
    pub fn trial(&self) -> &TrialSpec {
        self.specification.trial()
    }

    /// Borrows the importer-owned task snapshot selected for this exact attempt.
    pub fn imported(&self) -> &ImportedTask {
        self.specification.imported()
    }

    /// Borrows the importer-owned package snapshot used for this attempt.
    pub fn package(&self) -> &HarborTaskPackage {
        self.specification.package()
    }

    /// Borrows the immutable trial identity.
    pub fn trial_digest(&self) -> &ArtifactDigest {
        &self.trial_digest
    }

    /// Borrows the immutable identity of this expanded repetition.
    pub fn resolved_digest(&self) -> &ArtifactDigest {
        &self.resolved_digest
    }

    /// Borrows the stable expanded-position identity shared by all run attempts.
    pub fn assignment_id(&self) -> &EpisodeAssignmentId {
        &self.assignment_id
    }

    /// Borrows this repetition's deterministic attempt identifier.
    pub fn attempt_id(&self) -> &AttemptId {
        &self.attempt_id
    }

    /// Borrows the resources required by this attempt.
    pub fn resources(&self) -> &ResourceLeaseRequest {
        self.specification.resources()
    }

    /// Borrows paired-comparison factor labels in canonical key order.
    pub fn paired_factors(&self) -> &BTreeMap<String, String> {
        self.specification.paired_factors()
    }

    /// Borrows the complete immutable binding selection used by this attempt.
    pub fn selected_model_binding(&self) -> Option<&SelectedModelBinding> {
        self.specification.selected_model_binding()
    }

    pub(crate) fn resource_handle(&self) -> Rc<ResourceLeaseRequest> {
        self.specification.resources.clone()
    }
}

/// Strict authored NativeGraph suite input before caller-controlled sources are acquired.
#[derive(Clone, Debug)]
pub struct AuthoredNativeGraphSuite {
    document: SuiteTomlDto,
}

impl AuthoredNativeGraphSuite {
    /// Resolves sources once through the injected importer into immutable package snapshots.
    pub fn resolve(
        &self,
        importer: &HarborImporter<'_>,
    ) -> Result<NativeGraphSuiteDefinition, SuiteError> {
        let expanded_trials = expanded_trial_count(&self.document)?;
        let mut imported_tasks = Vec::with_capacity(self.document.tasks.len());
        let mut global_binding_keys = BTreeMap::new();

        for (task_index, task) in self.document.tasks.iter().enumerate() {
            let source = task.source.to_harbor_source()?;
            let imported = importer
                .import(&source)
                .map_err(|error| SuiteError::TaskImport {
                    task_index,
                    message: error.to_string(),
                })?;
            let expected = EvalTaskRef::new(task.task_id.clone(), task.task_digest.clone())
                .map_err(|error| SuiteError::InvalidTaskReference(error.to_string()))?;
            if imported.task != expected {
                return Err(SuiteError::TaskReferenceMismatch {
                    task_index,
                    expected,
                    actual: imported.task,
                });
            }
            let native_graph = imported
                .package
                .native_graph()
                .ok_or(SuiteError::NotNativeGraphTask)?;
            for binding in native_graph.model_bindings() {
                let capacity_key = ModelCapacityKey::from_task_binding(&imported.task, binding);
                match global_binding_keys.get(binding.id.as_str()) {
                    Some(existing) if existing != &capacity_key => {
                        return Err(SuiteError::CrossTaskModelBindingAlias {
                            binding: binding.id.as_str().to_owned(),
                        });
                    }
                    Some(_) => {}
                    None => {
                        global_binding_keys.insert(binding.id.as_str().to_owned(), capacity_key);
                    }
                }
            }
            imported_tasks.push(Rc::new(imported));
        }

        let limit_weights = resolve_model_weights(
            &self.document.limits.model_binding_units,
            &global_binding_keys,
            "limits.model_binding_units",
        )?;
        let resource_limits = ResourceLimits::new(
            self.document.limits.parallelism,
            self.document.limits.cpu_units,
            self.document.limits.memory_bytes,
            limit_weights,
        )
        .map_err(|error| SuiteError::InvalidLimits(error.to_string()))?;
        let budget = TrialBudget::new(
            self.document.defaults.execution_seconds,
            self.document.defaults.verifier_seconds,
        )
        .map_err(|error| SuiteError::InvalidTrialBudget(error.to_string()))?;
        let runtime = RuntimeIdentity::new(self.document.defaults.runtime.clone())
            .map_err(|error| SuiteError::InvalidRuntime(error.to_string()))?;
        let mut trials = Vec::with_capacity(expanded_trials);

        for (task_index, (authored, imported)) in
            self.document.tasks.iter().zip(imported_tasks).enumerate()
        {
            let native_graph = imported
                .package
                .native_graph()
                .ok_or(SuiteError::NotNativeGraphTask)?;
            let task_bindings = native_graph
                .model_bindings()
                .iter()
                .map(|binding| {
                    (
                        binding.id.as_str().to_owned(),
                        SelectedModelBinding::from_task_binding(&imported.task, binding),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let resource_weights = resolve_model_weights(
                &authored.resources.model_binding_units,
                &task_bindings
                    .iter()
                    .map(|(id, binding)| (id.clone(), binding.capacity_key().clone()))
                    .collect(),
                "tasks.resources.model_binding_units",
            )?;
            let resources = Rc::new(ResourceLeaseRequest::new(
                authored.resources.cpu_units,
                authored.resources.memory_bytes,
                resource_weights,
            )?);
            let paired_factors = Rc::new(authored.paired_factors.clone());
            let repetitions = NonZeroUsize::new(authored.repetitions)
                .ok_or(SuiteError::ZeroRepetitions { task_index })?;

            for graph_axis in &authored.graph_axes {
                let agent = AgentVariantRef::new(graph_axis.clone()).map_err(|error| {
                    SuiteError::InvalidAxis {
                        axis: "graph_axes",
                        message: error.to_string(),
                    }
                })?;
                for model_axis in &authored.model_axes {
                    let binding = native_graph
                        .model_bindings()
                        .iter()
                        .find(|binding| binding.id.as_str() == model_axis)
                        .ok_or_else(|| SuiteError::MissingModelAxisBinding {
                            task_index,
                            binding: model_axis.clone(),
                        })?;
                    let selected_model_binding =
                        task_bindings.get(model_axis).ok_or_else(|| {
                            SuiteError::MissingModelAxisBinding {
                                task_index,
                                binding: model_axis.clone(),
                            }
                        })?;
                    let model = ModelIdentity::new(
                        binding.endpoint_profile_id.clone(),
                        binding.model.clone(),
                    )
                    .map_err(|error| SuiteError::InvalidAxis {
                        axis: "model_axes",
                        message: error.to_string(),
                    })?;
                    for policy_axis in &authored.policy_axes {
                        for &seed in &authored.seeds {
                            let trial = TrialSpec::new(
                                imported.task.clone(),
                                agent.clone(),
                                model.clone(),
                                seed,
                                PolicyIdentity::new(policy_axis.clone()),
                                budget.clone(),
                                self.document.defaults.environment.clone(),
                                self.document.defaults.verifier.clone(),
                                runtime.clone(),
                            )
                            .map_err(|error| SuiteError::InvalidTrialBudget(error.to_string()))?;
                            trials.push(SuiteTrialSpec::from_snapshot_with_factors(
                                imported.clone(),
                                trial,
                                repetitions,
                                resources.clone(),
                                paired_factors.clone(),
                                Some(selected_model_binding.clone()),
                            )?);
                        }
                    }
                }
            }
        }

        Ok(NativeGraphSuiteDefinition {
            manifest: NativeGraphSuiteManifest::new(trials)?,
            resource_limits,
        })
    }
}

/// Resolved immutable suite inputs ready for a selected local scheduler capability.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphSuiteDefinition {
    manifest: NativeGraphSuiteManifest,
    resource_limits: ResourceLimits,
}

impl NativeGraphSuiteDefinition {
    /// Borrows the immutable ordered suite manifest.
    pub fn manifest(&self) -> &NativeGraphSuiteManifest {
        &self.manifest
    }

    /// Borrows the scheduler-wide finite resource capacities.
    pub fn resource_limits(&self) -> &ResourceLimits {
        &self.resource_limits
    }

    /// Computes the identity of both the manifest and its scheduler-wide limits.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-suite-definition-v1",
        );
        append_field(
            &mut material,
            "manifest",
            self.manifest.identity_digest().as_str().as_bytes(),
        );
        append_resource_limits(&mut material, &self.resource_limits);
        ArtifactDigest::from_bytes(&material)
    }

    /// Expands this definition so assignment identities include scheduler limits.
    pub fn resolve(&self, run_id: SuiteRunId) -> Result<ResolvedNativeGraphSuite, SuiteError> {
        self.manifest
            .resolve_with_suite_digest(run_id, self.identity_digest())
    }
}

/// Parses a byte-capped strict `suite.toml` document without acquiring task sources.
pub fn parse_native_graph_suite_toml(bytes: &[u8]) -> Result<AuthoredNativeGraphSuite, SuiteError> {
    if bytes.len() > MAX_SUITE_TOML_BYTES {
        return Err(SuiteError::DocumentTooLarge {
            actual: bytes.len(),
            limit: MAX_SUITE_TOML_BYTES,
        });
    }
    let text =
        std::str::from_utf8(bytes).map_err(|error| SuiteError::InvalidToml(error.to_string()))?;
    let document = toml::from_str::<SuiteTomlDto>(text)
        .map_err(|error| SuiteError::InvalidToml(error.to_string()))?;
    validate_suite_document(&document)?;
    Ok(AuthoredNativeGraphSuite { document })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteTomlDto {
    defaults: SuiteDefaultsDto,
    limits: SuiteLimitsDto,
    tasks: Vec<SuiteTaskDto>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteDefaultsDto {
    runtime: String,
    execution_seconds: f64,
    verifier_seconds: f64,
    environment: ArtifactDigest,
    verifier: ArtifactDigest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteLimitsDto {
    parallelism: usize,
    cpu_units: u32,
    memory_bytes: u64,
    max_expanded_trials: usize,
    model_binding_units: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteTaskDto {
    source: SuiteSourceDto,
    task_id: String,
    task_digest: ArtifactDigest,
    graph_axes: Vec<String>,
    model_axes: Vec<String>,
    policy_axes: Vec<ArtifactDigest>,
    seeds: Vec<u64>,
    repetitions: usize,
    #[serde(default)]
    paired_factors: BTreeMap<String, String>,
    resources: SuiteTaskResourceDto,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum SuiteSourceDto {
    Local {
        path: String,
    },
    PinnedGit {
        repository: String,
        revision: String,
        package_path: String,
    },
    Registry {
        reference: String,
    },
}

impl SuiteSourceDto {
    fn to_harbor_source(&self) -> Result<HarborSource, SuiteError> {
        match self {
            Self::Local { path } => HarborSource::local(path.clone())
                .map_err(|error| SuiteError::InvalidSource(error.to_string())),
            Self::PinnedGit {
                repository,
                revision,
                package_path,
            } => {
                HarborSource::pinned_git(repository.clone(), revision.clone(), package_path.clone())
                    .map_err(|error| SuiteError::InvalidSource(error.to_string()))
            }
            Self::Registry { reference } => {
                if reference.trim().is_empty() {
                    return Err(SuiteError::InvalidSource(
                        "registry reference must not be empty".to_owned(),
                    ));
                }
                Ok(HarborSource::Registry(reference.clone()))
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteTaskResourceDto {
    cpu_units: u32,
    memory_bytes: u64,
    model_binding_units: BTreeMap<String, u32>,
}

fn validate_suite_document(document: &SuiteTomlDto) -> Result<(), SuiteError> {
    if document.tasks.is_empty() {
        return Err(SuiteError::EmptyManifest);
    }
    if document.tasks.len() > MAX_SUITE_TASKS {
        return Err(SuiteError::TaskLimitExceeded {
            requested: document.tasks.len(),
            limit: MAX_SUITE_TASKS,
        });
    }
    validate_bounded_text(&document.defaults.runtime, "defaults.runtime")?;
    TrialBudget::new(
        document.defaults.execution_seconds,
        document.defaults.verifier_seconds,
    )
    .map_err(|error| SuiteError::InvalidTrialBudget(error.to_string()))?;
    if document.limits.parallelism == 0 {
        return Err(SuiteError::InvalidLimits(
            "limits.parallelism must be positive".to_owned(),
        ));
    }
    if document.limits.cpu_units == 0 || document.limits.memory_bytes == 0 {
        return Err(SuiteError::InvalidLimits(
            "limits.cpu_units and limits.memory_bytes must be positive".to_owned(),
        ));
    }
    if document.limits.max_expanded_trials == 0
        || document.limits.max_expanded_trials > MAX_EXPANDED_TRIALS
    {
        return Err(SuiteError::InvalidExpansionLimit {
            requested: document.limits.max_expanded_trials,
            limit: MAX_EXPANDED_TRIALS,
        });
    }
    validate_weight_map(
        &document.limits.model_binding_units,
        "limits.model_binding_units",
    )?;
    for (task_index, task) in document.tasks.iter().enumerate() {
        validate_task_document(task_index, task)?;
    }
    expanded_trial_count(document)?;
    Ok(())
}

fn validate_task_document(task_index: usize, task: &SuiteTaskDto) -> Result<(), SuiteError> {
    validate_bounded_text(&task.task_id, "tasks.task_id")?;
    match &task.source {
        SuiteSourceDto::Local { path } => validate_bounded_text(path, "tasks.source.path")?,
        SuiteSourceDto::PinnedGit {
            repository,
            revision,
            package_path,
        } => {
            validate_bounded_text(repository, "tasks.source.repository")?;
            validate_bounded_text(revision, "tasks.source.revision")?;
            validate_bounded_text(package_path, "tasks.source.package_path")?;
        }
        SuiteSourceDto::Registry { reference } => {
            validate_bounded_text(reference, "tasks.source.reference")?;
        }
    }
    validate_axis(&task.graph_axes, "graph_axes")?;
    validate_axis(&task.model_axes, "model_axes")?;
    if task.policy_axes.is_empty() || task.policy_axes.len() > MAX_AXIS_VALUES {
        return Err(SuiteError::AxisLimitExceeded {
            axis: "policy_axes",
            requested: task.policy_axes.len(),
            limit: MAX_AXIS_VALUES,
        });
    }
    if task.policy_axes.iter().collect::<BTreeSet<_>>().len() != task.policy_axes.len() {
        return Err(SuiteError::DuplicateAxisValue("policy_axes"));
    }
    if task.seeds.is_empty() || task.seeds.len() > MAX_SEEDS {
        return Err(SuiteError::SeedLimitExceeded {
            task_index,
            requested: task.seeds.len(),
            limit: MAX_SEEDS,
        });
    }
    if task.seeds.iter().collect::<BTreeSet<_>>().len() != task.seeds.len() {
        return Err(SuiteError::DuplicateSeed { task_index });
    }
    if task.repetitions == 0 {
        return Err(SuiteError::ZeroRepetitions { task_index });
    }
    if task.repetitions > MAX_REPETITIONS {
        return Err(SuiteError::RepetitionLimitExceeded {
            task_index,
            requested: task.repetitions,
            limit: MAX_REPETITIONS,
        });
    }
    if task.paired_factors.len() > MAX_PAIRED_FACTORS {
        return Err(SuiteError::PairedFactorLimitExceeded {
            task_index,
            requested: task.paired_factors.len(),
            limit: MAX_PAIRED_FACTORS,
        });
    }
    for (key, value) in &task.paired_factors {
        validate_bounded_text(key, "tasks.paired_factors key")?;
        validate_bounded_text(value, "tasks.paired_factors value")?;
    }
    if task.resources.cpu_units == 0 || task.resources.memory_bytes == 0 {
        return Err(SuiteError::InvalidTaskResources { task_index });
    }
    validate_weight_map(
        &task.resources.model_binding_units,
        "tasks.resources.model_binding_units",
    )
}

fn validate_axis(axis: &[String], name: &'static str) -> Result<(), SuiteError> {
    if axis.is_empty() || axis.len() > MAX_AXIS_VALUES {
        return Err(SuiteError::AxisLimitExceeded {
            axis: name,
            requested: axis.len(),
            limit: MAX_AXIS_VALUES,
        });
    }
    for value in axis {
        validate_bounded_text(value, name)?;
    }
    if axis.iter().collect::<BTreeSet<_>>().len() != axis.len() {
        return Err(SuiteError::DuplicateAxisValue(name));
    }
    Ok(())
}

fn validate_weight_map(
    weights: &BTreeMap<String, u32>,
    field: &'static str,
) -> Result<(), SuiteError> {
    for (binding, units) in weights {
        validate_bounded_text(binding, field)?;
        if *units == 0 {
            return Err(SuiteError::ZeroResourceRequest(field));
        }
    }
    Ok(())
}

fn validate_bounded_text(value: &str, field: &'static str) -> Result<(), SuiteError> {
    if value.trim().is_empty() || value.len() > MAX_FACTOR_TEXT_BYTES {
        return Err(SuiteError::InvalidBoundedText(field));
    }
    Ok(())
}

fn expanded_trial_count(document: &SuiteTomlDto) -> Result<usize, SuiteError> {
    let mut total = 0usize;
    for task in &document.tasks {
        let task_total = task
            .graph_axes
            .len()
            .checked_mul(task.model_axes.len())
            .and_then(|count| count.checked_mul(task.policy_axes.len()))
            .and_then(|count| count.checked_mul(task.seeds.len()))
            .and_then(|count| count.checked_mul(task.repetitions))
            .ok_or(SuiteError::TrialExpansionOverflow)?;
        total = total
            .checked_add(task_total)
            .ok_or(SuiteError::TrialExpansionOverflow)?;
    }
    let limit = document.limits.max_expanded_trials.min(MAX_EXPANDED_TRIALS);
    if total > limit {
        return Err(SuiteError::TrialExpansionLimitExceeded {
            requested: total,
            limit,
        });
    }
    Ok(total)
}

fn resolve_model_weights(
    weights: &BTreeMap<String, u32>,
    binding_keys: &BTreeMap<String, ModelCapacityKey>,
    field: &'static str,
) -> Result<BTreeMap<ModelCapacityKey, u32>, SuiteError> {
    let mut resolved = BTreeMap::new();
    for (binding, units) in weights {
        let capacity_key =
            binding_keys
                .get(binding)
                .ok_or_else(|| SuiteError::MissingResourceBinding {
                    field,
                    binding: binding.clone(),
                })?;
        resolved.insert(capacity_key.clone(), *units);
    }
    Ok(resolved)
}

fn select_trial_model_binding(
    task: &EvalTaskRef,
    bindings: &[ModelBindingSpec],
    trial: &TrialSpec,
) -> Result<SelectedModelBinding, SuiteError> {
    let mut matches = bindings.iter().filter(|binding| {
        binding.endpoint_profile_id == trial.model.provider && binding.model == trial.model.model
    });
    let Some(binding) = matches.next() else {
        return Err(SuiteError::MissingTrialModelBinding);
    };
    if matches.next().is_some() {
        return Err(SuiteError::AmbiguousTrialModelBinding);
    }
    Ok(SelectedModelBinding::from_task_binding(task, binding))
}

/// Failed suite construction or pure expansion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SuiteError {
    /// A manifest had no authored trial axis.
    EmptyManifest,
    /// An authored resource weight was zero.
    ZeroResourceRequest(&'static str),
    /// Repetition expansion exceeded addressable local memory.
    TrialExpansionOverflow,
    /// The resolved repetition count exceeded an authored or hard suite bound.
    TrialExpansionLimitExceeded {
        /// Number of expanded trials requested by the authored axes.
        requested: usize,
        /// Maximum expanded trials allowed for this suite.
        limit: usize,
    },
    /// The programmatic manifest contained too many authored trial axes.
    ManifestTrialLimitExceeded {
        /// Number of authored trial axes supplied by the caller.
        requested: usize,
        /// Maximum authored trial axes accepted by the local scheduler contract.
        limit: usize,
    },
    /// A strict suite manifest was larger than the parser input bound.
    DocumentTooLarge {
        /// Number of source bytes received.
        actual: usize,
        /// Largest supported strict suite manifest in bytes.
        limit: usize,
    },
    /// Strict TOML parsing rejected the authored suite document.
    InvalidToml(String),
    /// A strict suite document exceeded its task-reference limit.
    TaskLimitExceeded {
        /// Number of ordered task references requested.
        requested: usize,
        /// Maximum task references accepted.
        limit: usize,
    },
    /// An authored expansion limit was zero or exceeded the hard cap.
    InvalidExpansionLimit {
        /// Authored expansion cap.
        requested: usize,
        /// Hard expansion cap.
        limit: usize,
    },
    /// An axis was empty or exceeded its bounded cardinality.
    AxisLimitExceeded {
        /// Name of the rejected axis.
        axis: &'static str,
        /// Authored cardinality.
        requested: usize,
        /// Maximum supported cardinality.
        limit: usize,
    },
    /// An axis repeated an authored value.
    DuplicateAxisValue(&'static str),
    /// A seed schedule was empty or exceeded its bounded cardinality.
    SeedLimitExceeded {
        /// Ordered task-reference index.
        task_index: usize,
        /// Authored seed count.
        requested: usize,
        /// Maximum supported seed count.
        limit: usize,
    },
    /// A seed appeared more than once in one authored schedule.
    DuplicateSeed {
        /// Ordered task-reference index.
        task_index: usize,
    },
    /// An authored repetition count was zero.
    ZeroRepetitions {
        /// Ordered task-reference index.
        task_index: usize,
    },
    /// An authored repetition count exceeded its bounded cardinality.
    RepetitionLimitExceeded {
        /// Ordered task-reference index.
        task_index: usize,
        /// Authored repetition count.
        requested: usize,
        /// Maximum supported repetition count.
        limit: usize,
    },
    /// A task supplied too many paired-comparison factor labels.
    PairedFactorLimitExceeded {
        /// Ordered task-reference index.
        task_index: usize,
        /// Authored factor count.
        requested: usize,
        /// Maximum supported factor count.
        limit: usize,
    },
    /// A bounded manifest text field was empty or too long.
    InvalidBoundedText(&'static str),
    /// An authored task resource request was not finite and positive.
    InvalidTaskResources {
        /// Ordered task-reference index.
        task_index: usize,
    },
    /// A suite source could not be converted into a validated importer reference.
    InvalidSource(String),
    /// The importer could not acquire or normalize an authored suite task.
    TaskImport {
        /// Ordered task-reference index.
        task_index: usize,
        /// Source acquisition or normalization diagnostic.
        message: String,
    },
    /// An authored task reference was malformed.
    InvalidTaskReference(String),
    /// An imported snapshot did not match the exact authored task reference.
    TaskReferenceMismatch {
        /// Ordered task-reference index.
        task_index: usize,
        /// Immutable task reference authored in the suite.
        expected: EvalTaskRef,
        /// Immutable task reference derived from the importer-owned snapshot.
        actual: EvalTaskRef,
    },
    /// One authored model axis did not select a binding declared by its task snapshot.
    MissingModelAxisBinding {
        /// Ordered task-reference index.
        task_index: usize,
        /// Logical model binding identifier.
        binding: String,
    },
    /// One binding name would select conflicting package-scoped model runtimes.
    CrossTaskModelBindingAlias {
        /// Ambiguous textual model-binding identifier.
        binding: String,
    },
    /// A resource map referenced no binding declared by the resolved suite tasks.
    MissingResourceBinding {
        /// Authored resource map containing the missing binding.
        field: &'static str,
        /// Logical model binding identifier.
        binding: String,
    },
    /// A programmatic resource request named a capacity from another task snapshot.
    ForeignResourceCapacityKey {
        /// Immutable capacity key not declared by the imported task.
        key: ArtifactDigest,
    },
    /// An axis value failed its typed identity validation.
    InvalidAxis {
        /// Authored axis containing the value.
        axis: &'static str,
        /// Typed identity validation diagnostic.
        message: String,
    },
    /// The strict suite defaults carried an invalid execution or verifier budget.
    InvalidTrialBudget(String),
    /// The strict suite defaults carried an invalid runtime identity.
    InvalidRuntime(String),
    /// The strict suite limits could not construct a bounded scheduler capacity.
    InvalidLimits(String),
    /// An internally derived deterministic attempt identifier was rejected.
    InvalidAttemptId(String),
    /// An imported task did not select the strict NativeGraph profile.
    NotNativeGraphTask,
    /// A trial's task identity was not derived from the imported package snapshot.
    TrialTaskMismatch,
    /// A programmatic trial selected no binding in its imported package snapshot.
    MissingTrialModelBinding,
    /// An externally driven trial attempted to select a model binding.
    ExternalTrialModelBinding,
    /// An externally driven manifest requested more than its one admitted repetition.
    ExternalTrialRepetitionCount {
        /// Authored repetition count.
        requested: usize,
    },
    /// An externally driven manifest contained more than one authored trial axis.
    ExternalManifestTrialAxes {
        /// Authored externally driven axis count.
        requested: usize,
    },
    /// One manifest mixed native and externally driven profiles.
    MixedNativeGraphProfiles,
    /// A programmatic trial selected more than one binding in its imported package snapshot.
    AmbiguousTrialModelBinding,
}

impl fmt::Display for SuiteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyManifest => formatter.write_str("native graph suite must contain a trial"),
            Self::ZeroResourceRequest(field) => {
                write!(
                    formatter,
                    "native graph resource request {field} must be positive"
                )
            }
            Self::TrialExpansionOverflow => {
                formatter.write_str("native graph suite trial expansion overflowed")
            }
            Self::TrialExpansionLimitExceeded { requested, limit } => write!(
                formatter,
                "native graph suite expands to {requested} trials, above limit {limit}"
            ),
            Self::ManifestTrialLimitExceeded { requested, limit } => write!(
                formatter,
                "native graph suite contains {requested} authored trial axes, above limit {limit}"
            ),
            Self::DocumentTooLarge { actual, limit } => write!(
                formatter,
                "native graph suite document is {actual} bytes, above limit {limit}"
            ),
            Self::InvalidToml(error) => {
                write!(formatter, "invalid native graph suite TOML: {error}")
            }
            Self::TaskLimitExceeded { requested, limit } => write!(
                formatter,
                "native graph suite contains {requested} tasks, above limit {limit}"
            ),
            Self::InvalidExpansionLimit { requested, limit } => write!(
                formatter,
                "native graph suite expansion limit {requested} must be between 1 and {limit}"
            ),
            Self::AxisLimitExceeded {
                axis,
                requested,
                limit,
            } => write!(
                formatter,
                "native graph suite axis {axis} contains {requested} values, outside 1..={limit}"
            ),
            Self::DuplicateAxisValue(axis) => {
                write!(formatter, "native graph suite axis {axis} repeats a value")
            }
            Self::SeedLimitExceeded {
                task_index,
                requested,
                limit,
            } => write!(
                formatter,
                "native graph suite task {task_index} has {requested} seeds, outside 1..={limit}"
            ),
            Self::DuplicateSeed { task_index } => write!(
                formatter,
                "native graph suite task {task_index} repeats a seed"
            ),
            Self::ZeroRepetitions { task_index } => write!(
                formatter,
                "native graph suite task {task_index} repetitions must be positive"
            ),
            Self::RepetitionLimitExceeded {
                task_index,
                requested,
                limit,
            } => write!(
                formatter,
                "native graph suite task {task_index} has {requested} repetitions, above limit {limit}"
            ),
            Self::PairedFactorLimitExceeded {
                task_index,
                requested,
                limit,
            } => write!(
                formatter,
                "native graph suite task {task_index} has {requested} paired factors, above limit {limit}"
            ),
            Self::InvalidBoundedText(field) => write!(
                formatter,
                "native graph suite field {field} must be nonempty and at most {MAX_FACTOR_TEXT_BYTES} bytes"
            ),
            Self::InvalidTaskResources { task_index } => write!(
                formatter,
                "native graph suite task {task_index} resource weights must be positive"
            ),
            Self::InvalidSource(error) => {
                write!(formatter, "invalid native graph suite source: {error}")
            }
            Self::TaskImport {
                task_index,
                message,
            } => write!(
                formatter,
                "native graph suite task {task_index} could not be imported: {message}"
            ),
            Self::InvalidTaskReference(error) => {
                write!(
                    formatter,
                    "invalid native graph suite task reference: {error}"
                )
            }
            Self::TaskReferenceMismatch {
                task_index,
                expected,
                actual,
            } => write!(
                formatter,
                "native graph suite task {task_index} resolved to {}@{}, expected {}@{}",
                actual.id.as_str(),
                actual.digest.as_str(),
                expected.id.as_str(),
                expected.digest.as_str(),
            ),
            Self::MissingModelAxisBinding {
                task_index,
                binding,
            } => write!(
                formatter,
                "native graph suite task {task_index} does not declare model binding {binding:?}"
            ),
            Self::CrossTaskModelBindingAlias { binding } => write!(
                formatter,
                "native graph suite model binding {binding:?} names conflicting task-scoped runtimes"
            ),
            Self::MissingResourceBinding { field, binding } => write!(
                formatter,
                "native graph suite {field} does not resolve model binding {binding:?}"
            ),
            Self::ForeignResourceCapacityKey { key } => write!(
                formatter,
                "native graph suite resource request references foreign model capacity key {}",
                key.as_str()
            ),
            Self::InvalidAxis { axis, message } => {
                write!(
                    formatter,
                    "invalid native graph suite {axis} value: {message}"
                )
            }
            Self::InvalidTrialBudget(error) => {
                write!(
                    formatter,
                    "invalid native graph suite trial budget: {error}"
                )
            }
            Self::InvalidRuntime(error) => {
                write!(formatter, "invalid native graph suite runtime: {error}")
            }
            Self::InvalidLimits(error) => {
                write!(formatter, "invalid native graph suite limits: {error}")
            }
            Self::InvalidAttemptId(error) => {
                write!(
                    formatter,
                    "invalid derived native graph attempt id: {error}"
                )
            }
            Self::NotNativeGraphTask => {
                formatter.write_str("native graph suite task did not select a native graph package")
            }
            Self::TrialTaskMismatch => formatter.write_str(
                "native graph suite trial task identity does not match imported package snapshot",
            ),
            Self::MissingTrialModelBinding => formatter.write_str(
                "native graph suite trial model does not select an imported package binding",
            ),
            Self::ExternalTrialModelBinding => {
                formatter.write_str("externally driven trial must not select a model binding")
            }
            Self::ExternalTrialRepetitionCount { requested } => write!(
                formatter,
                "externally driven suite trial has {requested} repetitions; exactly one is required"
            ),
            Self::ExternalManifestTrialAxes { requested } => write!(
                formatter,
                "externally driven suite has {requested} trial axes; exactly one is required"
            ),
            Self::MixedNativeGraphProfiles => formatter
                .write_str("native graph suite must not mix native and externally driven profiles"),
            Self::AmbiguousTrialModelBinding => formatter.write_str(
                "native graph suite trial model selects multiple imported package bindings",
            ),
        }
    }
}

impl std::error::Error for SuiteError {}

fn append_resource_request(material: &mut Vec<u8>, request: &ResourceLeaseRequest) {
    append_field(
        material,
        "resource-cpu-units",
        request.cpu_units.to_string().as_bytes(),
    );
    append_field(
        material,
        "resource-memory-bytes",
        request.memory_bytes.to_string().as_bytes(),
    );
    append_field(
        material,
        "resource-model-count",
        request.model_binding_units.len().to_string().as_bytes(),
    );
    for (binding, units) in &request.model_binding_units {
        append_field(
            material,
            "resource-model-capacity-key",
            binding.digest().as_str().as_bytes(),
        );
        append_field(
            material,
            "resource-model-units",
            units.to_string().as_bytes(),
        );
    }
}

fn append_resource_limits(material: &mut Vec<u8>, limits: &ResourceLimits) {
    append_field(
        material,
        "limit-episode-slots",
        limits.episode_slots().to_string().as_bytes(),
    );
    append_field(
        material,
        "limit-cpu-units",
        limits.cpu_units().to_string().as_bytes(),
    );
    append_field(
        material,
        "limit-memory-bytes",
        limits.memory_bytes().to_string().as_bytes(),
    );
    append_field(
        material,
        "limit-model-count",
        limits.model_binding_units().len().to_string().as_bytes(),
    );
    for (binding, units) in limits.model_binding_units() {
        append_field(
            material,
            "limit-model-capacity-key",
            binding.digest().as_str().as_bytes(),
        );
        append_field(material, "limit-model-units", units.to_string().as_bytes());
    }
}

fn append_paired_factors(material: &mut Vec<u8>, paired_factors: &BTreeMap<String, String>) {
    append_field(
        material,
        "paired-factor-count",
        paired_factors.len().to_string().as_bytes(),
    );
    for (key, value) in paired_factors {
        append_field(material, "paired-factor-key", key.as_bytes());
        append_field(material, "paired-factor-value", value.as_bytes());
    }
}

fn append_field(material: &mut Vec<u8>, name: &str, value: &[u8]) {
    material.extend_from_slice(name.len().to_string().as_bytes());
    material.push(b':');
    material.extend_from_slice(name.as_bytes());
    material.push(b'=');
    material.extend_from_slice(value.len().to_string().as_bytes());
    material.push(b':');
    material.extend_from_slice(value);
    material.push(0x1f);
}
