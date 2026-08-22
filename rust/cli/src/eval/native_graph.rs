// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI composition for scored NativeGraph episodes, including sealed live rollouts.

//! A task invocation becomes a one-trial suite. A suite document is accepted
//! only when its already-resolved shape has exactly one lifecycle-addressable
//! trial; the future suite lifecycle schema will carry independent provenance
//! for larger matrices rather than letting this boundary invent it.

use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    rc::Rc,
};

use aiperf_runtime::{
    engine::{application::Application, distribution_identity::current_distribution_id},
    eval::{
        CompatibilityFidelity, DockerExternallyDrivenEpisodeExecutor,
        DockerNativeGraphEpisodeExecutor, DockerProcessSandbox, EngineNativeGraphEpisodeCallback,
        EnvName, EpisodeFidelity, EvalNodeRecordArtifact, HarborEvaluationCoordinator,
        HarborImporter, HarborLifecycleAgentContract, HarborLifecycleRequest, HarborSandboxRecipe,
        ImportedTask, ModelCapacityKey, ModelRuntimeConfig, NativeGraphEpisodeRunner,
        NativeGraphProfile, NativeGraphSuiteManifest, NativeSourceAcquirer,
        PreparedExternalDriverCapability, ResourceLeaseRequest, ResourceLimits, SecretProvider,
        SecretValue, SuiteRunId, SuiteTrialSpec, VerifierMode, parse_native_graph_suite_toml,
        run_resolved_suite, select_native_graph_external_driver, select_native_graph_scheduler,
    },
};
use anyhow::Context as _;
use serde::Serialize;
use tokio::task::LocalSet;

use super::{DEFAULT_SANDBOX_IMAGE, SandboxFlag};

const MAX_MODEL_RUNTIME_BYTES: u64 = 1024 * 1024;
const NATIVE_GRAPH_SCHEDULER: &str = "local";
const NATIVE_GRAPH_EVALUATOR: &str = "harbor";

/// Immutable CLI selections needed before a NativeGraph episode may provision Docker.
pub(super) struct NativeGraphCliOptions {
    pub(super) image: Option<String>,
    pub(super) workdir: Option<String>,
    pub(super) sandbox: SandboxFlag,
    pub(super) requested_verifier_mode: Option<VerifierMode>,
    pub(super) has_external_agent_command: bool,
    pub(super) lifecycle_output_explicit: bool,
    pub(super) records_output: Option<PathBuf>,
}

/// Runs one imported NativeGraph task as exactly one resolved suite trial.
pub(super) fn run_task(
    imported: ImportedTask,
    model_runtime_path: Option<&Path>,
    lifecycle: &HarborLifecycleRequest,
    options: NativeGraphCliOptions,
) -> anyhow::Result<i32> {
    match native_profile(&imported)? {
        NativeGraphProfile::NativeGraph => {
            let runtime = read_required_model_runtime(model_runtime_path)?;
            validate_native_graph_invocation(&imported, lifecycle, &options)?;
            let trial = HarborEvaluationCoordinator::resolve_trial(&imported, lifecycle)?;
            let (resolved, limits) = one_trial_suite(imported.clone(), trial)?;
            run_resolved_native_graph_suite(imported, runtime, lifecycle, options, resolved, limits)
        }
        NativeGraphProfile::ExternallyDriven => {
            validate_native_graph_invocation(&imported, lifecycle, &options)?;
            if model_runtime_path.is_some() {
                anyhow::bail!(
                    "externally driven NativeGraph evaluation does not accept --model-runtime"
                );
            }
            let trial = HarborEvaluationCoordinator::resolve_trial(&imported, lifecycle)?;
            let (resolved, limits) = one_trial_suite(imported.clone(), trial)?;
            let native = imported.package.native_graph().ok_or_else(|| {
                anyhow::anyhow!(
                    "NativeGraph task snapshot disappeared before external-driver preflight"
                )
            })?;
            let dist_id = current_distribution_id()
                .context("deriving native graph distribution identity from the current binary")?;
            let application = Application::stock(dist_id)?;
            let factory =
                select_native_graph_external_driver(application.product_registry(), native)?;
            let resolved_trial = resolved
                .trials()
                .first()
                .ok_or_else(|| anyhow::anyhow!("external suite resolved no trial"))?;
            let prepared_driver = factory.prepare(&imported.package, resolved_trial)?;
            run_resolved_external_suite(
                imported,
                lifecycle,
                options,
                resolved,
                limits,
                &application,
                prepared_driver,
            )
        }
    }
}

/// Runs a one-lifecycle-addressable NativeGraph suite through the shared matrix path.
pub(super) fn run_suite(
    suite_path: &Path,
    model_runtime_path: &Path,
    lifecycle: &HarborLifecycleRequest,
    options: NativeGraphCliOptions,
) -> anyhow::Result<i32> {
    let runtime = read_model_runtime(model_runtime_path)?;
    let bytes = read_regular_file_bounded(suite_path, MAX_MODEL_RUNTIME_BYTES, "suite")?;
    let authored = parse_native_graph_suite_toml(&bytes)?;
    let definition = authored.resolve(&HarborImporter::new(&NativeSourceAcquirer))?;
    let resolved = definition.resolve(SuiteRunId::new(definition.identity_digest()))?;
    let [trial] = resolved.trials() else {
        anyhow::bail!(
            "NativeGraph --suite requires exactly one lifecycle-addressable trial; multi-trial lifecycle provenance is deferred"
        );
    };
    let imported = trial.imported().clone();
    validate_native_graph_invocation(&imported, lifecycle, &options)?;
    let lifecycle_trial = HarborEvaluationCoordinator::resolve_trial(&imported, lifecycle)?;
    if lifecycle_trial.identity_digest() != trial.trial().identity_digest() {
        anyhow::bail!(
            "NativeGraph --suite trial does not match the supplied lifecycle request; multi-lifecycle suite provenance is deferred"
        );
    }
    run_resolved_native_graph_suite(
        imported,
        runtime,
        lifecycle,
        options,
        resolved,
        definition.resource_limits().clone(),
    )
}

fn validate_native_graph_invocation(
    imported: &ImportedTask,
    lifecycle: &HarborLifecycleRequest,
    options: &NativeGraphCliOptions,
) -> anyhow::Result<()> {
    if matches!(options.sandbox, SandboxFlag::Local) {
        anyhow::bail!("NativeGraph evaluation requires the Docker sandbox backend");
    }
    if options.has_external_agent_command {
        anyhow::bail!("NativeGraph lifecycle contracts do not accept --agent-command");
    }
    if options.lifecycle_output_explicit {
        anyhow::bail!(
            "--lifecycle-output is not available for NativeGraph matrix results; retain the emitted scored episode result instead"
        );
    }
    let native = imported
        .package
        .native_graph()
        .ok_or_else(|| anyhow::anyhow!("NativeGraph task snapshot disappeared before execution"))?;
    match native.profile() {
        NativeGraphProfile::NativeGraph => {
            if lifecycle.agent_contract != HarborLifecycleAgentContract::NativeGraph {
                anyhow::bail!("NativeGraph evaluation requires a native_graph lifecycle contract");
            }
            if !native.adapters().is_empty() && native.rollout().is_none() {
                anyhow::bail!("NativeGraph adapters require one sealed rollout selection");
            }
        }
        NativeGraphProfile::ExternallyDriven => {
            if options.records_output.is_some() {
                anyhow::bail!(
                    "--records-output is available only for schema-1.1 NativeGraph evaluation"
                );
            }
            if lifecycle.agent_contract != HarborLifecycleAgentContract::ExternallyDriven {
                anyhow::bail!(
                    "externally driven NativeGraph evaluation requires an externally_driven lifecycle contract"
                );
            }
            let driver = native.driver_adapter().ok_or_else(|| {
                anyhow::anyhow!(
                    "externally driven NativeGraph task snapshot has no selected driver adapter"
                )
            })?;
            if lifecycle.command != driver.argv {
                anyhow::bail!("lifecycle command provenance disagrees with the manifest driver");
            }
        }
    }
    if !imported.package.is_standard_directory() {
        anyhow::bail!("NativeGraph evaluation requires a standard task directory");
    }
    if let Some(requested) = options.requested_verifier_mode
        && requested != imported.package.execution_plan().verifier().mode()
    {
        anyhow::bail!("--verifier-mode conflicts with the NativeGraph task verifier environment");
    }
    Ok(())
}

fn native_profile(imported: &ImportedTask) -> anyhow::Result<NativeGraphProfile> {
    imported
        .package
        .native_graph()
        .map(|native| native.profile())
        .ok_or_else(|| anyhow::anyhow!("NativeGraph task snapshot disappeared before execution"))
}

fn read_required_model_runtime(path: Option<&Path>) -> anyhow::Result<ModelRuntimeConfig> {
    let path = path.ok_or_else(|| {
        anyhow::anyhow!("--model-runtime is required for schema-1.1 NativeGraph evaluation")
    })?;
    read_model_runtime(path)
}

fn one_trial_suite(
    imported: ImportedTask,
    trial: aiperf_runtime::eval::TrialSpec,
) -> anyhow::Result<(
    aiperf_runtime::eval::ResolvedNativeGraphSuite,
    ResourceLimits,
)> {
    let native = imported.package.native_graph().ok_or_else(|| {
        anyhow::anyhow!("NativeGraph task snapshot disappeared before suite expansion")
    })?;
    let model_binding_units = native
        .model_bindings()
        .iter()
        .map(|binding| {
            (
                ModelCapacityKey::from_task_binding(&imported.task, binding),
                1,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let resources = ResourceLeaseRequest::new(1, 1, model_binding_units.clone())?;
    let manifest = NativeGraphSuiteManifest::new(vec![SuiteTrialSpec::from_imported(
        imported,
        trial,
        NonZeroUsize::new(1).ok_or_else(|| anyhow::anyhow!("one NativeGraph repetition"))?,
        resources,
    )?])?;
    let limits = ResourceLimits::new(1, 1, 1, model_binding_units)?;
    let resolved = manifest.resolve(SuiteRunId::new(manifest.identity_digest()))?;
    Ok((resolved, limits))
}

fn run_resolved_native_graph_suite(
    imported: ImportedTask,
    model_runtime: ModelRuntimeConfig,
    lifecycle: &HarborLifecycleRequest,
    options: NativeGraphCliOptions,
    suite: aiperf_runtime::eval::ResolvedNativeGraphSuite,
    limits: ResourceLimits,
) -> anyhow::Result<i32> {
    let record_artifact = options
        .records_output
        .as_deref()
        .map(EvalNodeRecordArtifact::open)
        .transpose()?;
    let image = options
        .image
        .unwrap_or_else(|| DEFAULT_SANDBOX_IMAGE.to_owned());
    let recipe = HarborSandboxRecipe::for_standard_task(image, options.workdir)?;
    let dist_id = current_distribution_id()
        .context("deriving native graph distribution identity from the current binary")?;
    let application = Rc::new(Application::stock(dist_id)?);
    let native = imported
        .package
        .native_graph()
        .ok_or_else(|| anyhow::anyhow!("NativeGraph task snapshot disappeared before preflight"))?;
    let secrets = Rc::new(HostEnvironmentSecrets);
    preflight_registered_native_graph_seams(application.as_ref(), native)?;
    let _callback = EngineNativeGraphEpisodeCallback::new(
        application.as_ref(),
        native,
        &model_runtime,
        secrets.as_ref(),
        record_artifact.clone(),
    )?;
    let scheduler = select_native_graph_scheduler(
        application.product_registry(),
        NATIVE_GRAPH_SCHEDULER,
        limits,
    )?;
    let executor = Rc::new(DockerNativeGraphEpisodeExecutor::new(
        DockerProcessSandbox::new(),
        recipe,
        imported.clone(),
        lifecycle.clone(),
        application.clone(),
        model_runtime,
        secrets,
        record_artifact.clone(),
    )?);
    let runner = Rc::new(NativeGraphEpisodeRunner::with_registered_evaluator(
        executor,
        application.product_registry(),
        NATIVE_GRAPH_EVALUATOR,
    )?);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = LocalSet::new();
    let results = local.block_on(
        &runtime,
        run_resolved_suite(scheduler.as_ref(), suite, runner),
    )?;
    if let Some(record_artifact) = &record_artifact {
        record_artifact.finish()?;
    }
    emit_results(imported.task.id.as_str(), lifecycle, &results)?;
    Ok(0)
}

fn run_resolved_external_suite(
    imported: ImportedTask,
    lifecycle: &HarborLifecycleRequest,
    options: NativeGraphCliOptions,
    suite: aiperf_runtime::eval::ResolvedNativeGraphSuite,
    limits: ResourceLimits,
    application: &Application,
    prepared_driver: PreparedExternalDriverCapability,
) -> anyhow::Result<i32> {
    let image = options
        .image
        .unwrap_or_else(|| DEFAULT_SANDBOX_IMAGE.to_owned());
    let recipe = HarborSandboxRecipe::for_standard_task(image, options.workdir)?;
    let scheduler = select_native_graph_scheduler(
        application.product_registry(),
        NATIVE_GRAPH_SCHEDULER,
        limits,
    )?;
    let task = imported.task.id.as_str().to_owned();
    let executor = Rc::new(DockerExternallyDrivenEpisodeExecutor::new(
        DockerProcessSandbox::new(),
        recipe,
        imported,
        lifecycle.clone(),
        prepared_driver,
        Rc::new(HostEnvironmentSecrets),
    )?);
    let runner = Rc::new(NativeGraphEpisodeRunner::with_registered_evaluator(
        executor,
        application.product_registry(),
        NATIVE_GRAPH_EVALUATOR,
    )?);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = LocalSet::new();
    let results = local.block_on(
        &runtime,
        run_resolved_suite(scheduler.as_ref(), suite, runner),
    )?;
    emit_results(&task, lifecycle, &results)?;
    Ok(0)
}

fn preflight_registered_native_graph_seams(
    application: &Application,
    native: &aiperf_runtime::eval::NativeGraphPackagePlan,
) -> anyhow::Result<()> {
    let registry = application.product_registry();
    let required = [
        (
            "lowerer",
            registry.native_graph_lowerer("native_graph").is_some(),
        ),
        (
            "scheduler",
            registry
                .native_graph_scheduler(NATIVE_GRAPH_SCHEDULER)
                .is_some(),
        ),
        (
            "evaluator",
            registry
                .native_graph_evaluator(NATIVE_GRAPH_EVALUATOR)
                .is_some(),
        ),
        (
            "external driver",
            registry.native_graph_external_driver("refuse").is_some(),
        ),
        (
            "fidelity observer",
            registry.native_graph_fidelity_observer("exact").is_some(),
        ),
        (
            "provider recovery",
            registry
                .native_graph_provider_recovery("confirmed")
                .is_some(),
        ),
    ];
    if let Some((seam, _)) = required.iter().find(|(_, available)| !available) {
        anyhow::bail!("no linked NativeGraph {seam} factory is available");
    }
    if let Some(rollout) = native.rollout() {
        let environment = rollout.environment();
        let selected = [
            (
                "rollout adapter protocol",
                registry
                    .native_graph_protocol(environment.protocol_factory_id().as_str())
                    .is_some(),
            ),
            (
                "rollout adapter runtime",
                registry
                    .native_graph_adapter_runtime(environment.runtime_provider_id().as_str())
                    .is_some(),
            ),
            (
                "rollout environment stepper",
                registry
                    .native_graph_environment_stepper(environment.stepper_factory_id().as_str())
                    .is_some(),
            ),
            (
                "rollout action encoder",
                registry
                    .native_graph_action_encoder(environment.action_encoder_id().as_str())
                    .is_some(),
            ),
        ];
        if let Some((seam, _)) = selected.iter().find(|(_, available)| !available) {
            anyhow::bail!("no linked NativeGraph {seam} factory is available");
        }
    }
    if native.program_source().is_none() {
        anyhow::bail!("NativeGraph task has no immutable program source");
    }
    Ok(())
}

fn read_model_runtime(path: &Path) -> anyhow::Result<ModelRuntimeConfig> {
    let bytes = read_regular_file_bounded(path, MAX_MODEL_RUNTIME_BYTES, "model runtime")?;
    toml::from_str(std::str::from_utf8(&bytes)?)
        .map_err(|error| anyhow::anyhow!("invalid model runtime {}: {error}", path.display()))
}

fn read_regular_file_bounded(path: &Path, limit: u64, kind: &str) -> anyhow::Result<Vec<u8>> {
    let metadata = fs::metadata(path)
        .map_err(|error| anyhow::anyhow!("unable to read {kind} {}: {error}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{kind} {} is not a regular file", path.display());
    }
    if metadata.len() > limit {
        anyhow::bail!("{kind} {} exceeds {limit} bytes", path.display());
    }
    fs::read(path)
        .map_err(|error| anyhow::anyhow!("unable to read {kind} {}: {error}", path.display()))
}

struct HostEnvironmentSecrets;

impl SecretProvider for HostEnvironmentSecrets {
    fn resolve(
        &self,
        name: &EnvName,
    ) -> Result<SecretValue, aiperf_runtime::eval::EvalExecutionError> {
        std::env::var(name)
            .map(SecretValue::new)
            .map_err(|_| aiperf_runtime::eval::EvalExecutionError::MissingSecret(name.clone()))
    }
}

#[derive(Serialize)]
struct NativeGraphEvalOutput<'a> {
    task: &'a str,
    artifacts: [(); 0],
    reward: BTreeMap<&'a str, f64>,
    episodes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<ExternalEvalScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fidelity: Option<ExternalEvalFidelity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lifecycle_evidence: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ExternalEvalScore {
    state: &'static str,
    reward: f64,
}

#[derive(Serialize)]
struct ExternalEvalFidelity {
    profile: &'static str,
    capture: &'static str,
}

fn emit_results(
    task: &str,
    lifecycle: &HarborLifecycleRequest,
    results: &[aiperf_runtime::eval::EpisodeResult],
) -> anyhow::Result<()> {
    let reward = results
        .first()
        .and_then(aiperf_runtime::eval::EpisodeResult::verified_reward)
        .ok_or_else(|| anyhow::anyhow!("NativeGraph episode completed without a verified score"))?;
    let external = results.first().and_then(|result| match result.fidelity() {
        EpisodeFidelity::ExternallyDriven(fidelity) => Some((result, fidelity)),
        EpisodeFidelity::Legacy | EpisodeFidelity::NativeGraph => None,
    });
    let score = external.map(|_| ExternalEvalScore {
        state: "verified",
        reward,
    });
    let fidelity = external.map(|(_, fidelity)| ExternalEvalFidelity {
        profile: "externally_driven",
        capture: match fidelity {
            CompatibilityFidelity::ObservedProxy => "observed_proxy",
            CompatibilityFidelity::Partial => "partial",
            CompatibilityFidelity::Missing => "missing",
        },
    });
    let lifecycle_evidence = external
        .map(|(result, _)| {
            result
                .compatibility_lifecycle_evidence()
                .map(|evidence| vec![evidence.digest().as_str().to_owned()])
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "externally driven result omitted compatibility lifecycle evidence"
                    )
                })
        })
        .transpose()?;
    let output = NativeGraphEvalOutput {
        task,
        artifacts: [],
        reward: BTreeMap::from([(lifecycle.regrade.metric.as_str(), reward)]),
        episodes: results.len(),
        score,
        fidelity,
        lifecycle_evidence,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
