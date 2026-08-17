// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI composition for the first scored, acyclic NativeGraph episode.

//! A task invocation becomes a one-trial suite. A suite document is accepted
//! only when its already-resolved shape has exactly one lifecycle-addressable
//! trial; the future suite lifecycle schema will carry independent provenance
//! for larger matrices rather than letting this boundary invent it.

use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, rc::Rc};

use aiperf_runtime::{
    engine::application::Application,
    eval::{
        DockerNativeGraphEpisodeExecutor, DockerProcessSandbox, EngineNativeGraphEpisodeCallback,
        EnvName, HarborEvaluationCoordinator, HarborImporter, HarborLifecycleAgentContract,
        HarborLifecycleRequest, HarborSandboxRecipe, ImportedTask, ModelCapacityKey,
        ModelRuntimeConfig, NativeGraphEpisodeRunner, NativeGraphProfile, NativeGraphSuiteManifest,
        NativeSourceAcquirer, ResourceLeaseRequest, ResourceLimits, SecretProvider, SecretValue,
        SuiteRunId, SuiteTrialSpec, VerifierMode, parse_native_graph_suite_toml,
        run_resolved_suite, select_native_graph_scheduler,
    },
};
use serde::Serialize;

use super::SandboxFlag;

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
}

/// Runs one imported NativeGraph task as exactly one resolved suite trial.
pub(super) fn run_task(
    imported: ImportedTask,
    model_runtime_path: &Path,
    lifecycle: &HarborLifecycleRequest,
    options: NativeGraphCliOptions,
) -> anyhow::Result<i32> {
    let runtime = read_model_runtime(model_runtime_path)?;
    let trial = validate_native_graph_invocation(&imported, lifecycle, &options)?;
    let (resolved, limits) = one_trial_suite(imported.clone(), trial)?;
    run_resolved_native_graph_suite(imported, runtime, lifecycle, options, resolved, limits)
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
    let lifecycle_trial = validate_native_graph_invocation(&imported, lifecycle, &options)?;
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
) -> anyhow::Result<aiperf_runtime::eval::TrialSpec> {
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
    if lifecycle.agent_contract != HarborLifecycleAgentContract::NativeGraph {
        anyhow::bail!("NativeGraph evaluation requires a native_graph lifecycle contract");
    }
    let native = imported
        .package
        .native_graph()
        .ok_or_else(|| anyhow::anyhow!("NativeGraph task snapshot disappeared before execution"))?;
    if native.profile() != NativeGraphProfile::NativeGraph {
        anyhow::bail!(
            "externally driven NativeGraph packages are not enabled by the acyclic model slice"
        );
    }
    if !native.adapters().is_empty() {
        anyhow::bail!(
            "NativeGraph adapters are not enabled by the acyclic model slice; the task must declare no adapters"
        );
    }
    if !imported.package.is_standard_directory() {
        anyhow::bail!("NativeGraph evaluation requires a standard task directory");
    }
    if let Some(requested) = options.requested_verifier_mode
        && requested != imported.package.execution_plan().verifier().mode()
    {
        anyhow::bail!("--verifier-mode conflicts with the NativeGraph task verifier environment");
    }
    Ok(HarborEvaluationCoordinator::resolve_trial(
        imported, lifecycle,
    )?)
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
    let image = options.image.unwrap_or_else(|| {
        "sha256:0000000000000000000000000000000000000000000000000000000000".to_owned()
    });
    let recipe = HarborSandboxRecipe::for_standard_task(image, options.workdir)?;
    let application = Rc::new(Application::stock(format!(
        "aiperf-cli-native-graph:{}",
        imported.package.identity_digest().as_str()
    ))?);
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
    )?);
    let runner = Rc::new(NativeGraphEpisodeRunner::with_registered_evaluator(
        executor,
        application.product_registry(),
        NATIVE_GRAPH_EVALUATOR,
    )?);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let results = runtime.block_on(run_resolved_suite(scheduler.as_ref(), suite, runner))?;
    emit_results(imported.task.id.as_str(), lifecycle, &results)?;
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
            "adapter protocol",
            registry.native_graph_protocol("jsonl").is_some(),
        ),
        (
            "adapter runtime",
            registry.native_graph_adapter_runtime("strict").is_some(),
        ),
        (
            "environment stepper",
            registry
                .native_graph_environment_stepper("refuse")
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
    let output = NativeGraphEvalOutput {
        task,
        artifacts: [],
        reward: BTreeMap::from([(lifecycle.regrade.metric.as_str(), reward)]),
        episodes: results.len(),
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
