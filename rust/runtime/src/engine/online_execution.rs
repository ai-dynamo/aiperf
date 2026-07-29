// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 executable workloads for the native online transports.
//!
//! Workload factories lower authored input into `NativeRunSpec` and resolve
//! execution and readiness policy from the validated transport.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::content_server::ContentServerMediaPublisher;
use crate::dataset::{
    DatasetFetcher, HttpDatasetFetcher, MaterializedTracePromptStorage,
    NativeSyntheticMediaGeneratorFactory, SyntheticMediaGeneratorFactory, TiktokenEncoding,
    download_hugging_face_tokenizer, find_tiktoken_model_file,
};
use crate::endpoints::Modality;
use crate::failure::OnFailure;
use crate::metrics_core::{ReportGraphRunInfo, ReportPairRunFacts};
use crate::rng::RngRoot;
use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::{Value, value::RawValue};
use url::Url;

use crate::engine::dataset_input::DatasetInputContext;
use crate::engine::execute::{
    NativeDatasetPlan, NativeEndpointPlan, NativeGraphDatasetPlan, NativeRunSpec,
    NativeSidecarPlan, NativeStaticAccuracyEvaluatorFactory, NativeStaticAccuracyPlan,
    StaticAccuracyEvaluatorFactory, StaticAccuracyEvaluatorProcessSpec,
    execute_prepared_native_plan_uncommitted_selected, load_tokenizer,
};
use crate::engine::execution_factories::ExecutionFactories;
use crate::engine::graph_input::GraphInputContext;
use crate::engine::protocol::{ArtifactSpec, PhaseSpec, TokenizerSpec};
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::readiness::{
    PreparedOnlineReadiness, ReadinessEndpointProfile, ReadinessPlanInput,
};
use crate::engine::registry::{
    GRAPH_WORKLOAD_DESCRIPTOR, GraphWorkloadConfigV2, NativeTransportExecution, PreparedRunOutcome,
    PreparedRunnerOperation, RunContext, SCHEDULED_WORKLOAD_DESCRIPTOR,
    STATIC_ACCURACY_WORKLOAD_DESCRIPTOR, ScheduledWorkloadConfigV2, StaticAccuracyWorkloadConfigV2,
    ValidatedTransportConfig, ValidatedWorkloadConfig, WorkloadDescriptor, WorkloadFactory,
    WorkloadRequirements, inference_workload_requirements, strict_decode, validate_common_workload,
};
use crate::engine::turn_execution::RequestExecutorFactory;

/// Native execution binding for the built-in `http` transport.
#[derive(Clone)]
pub struct HttpNativeExecution {
    executor: Arc<dyn RequestExecutorFactory>,
}

impl HttpNativeExecution {
    pub fn new(executor: Arc<dyn RequestExecutorFactory>) -> Self {
        Self { executor }
    }
}

impl NativeTransportExecution for HttpNativeExecution {
    fn executor_factory(&self) -> Arc<dyn RequestExecutorFactory> {
        self.executor.clone()
    }

    fn readiness_enabled(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn build_graph_dispatcher(
        &self,
        clock: Rc<dyn crate::clock::Clock>,
        run_origin_ns: i64,
        urls: &[String],
        model: &str,
        transport_config: crate::transport::http::TransportSinkConfig,
        endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
        // The HTTP sink's compatibility record already carries `Bytes` bodies
        // (cheap refcount clones), so it is built unconditionally regardless of
        // raw capture; the flag only matters to the gRPC sink.
        _capture_raw: bool,
    ) -> Result<Rc<dyn crate::transport::core::Dispatcher>> {
        Ok(Rc::new(
            crate::transport::http::TransportSink::new_multi_configured(
                clock,
                run_origin_ns,
                urls,
                model,
                transport_config,
            )?
            .with_prepared_endpoints(endpoints),
        ))
    }

    fn graph_transport_label(&self) -> &'static str {
        "http"
    }

    fn validate_run(&self, _run: &AuthoredRunSpecV2, context: &RunContext) -> Result<()> {
        validate_online_run(context)
    }

    fn run_metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }
}

/// Resolve the native execution binding for a validated transport.
///
/// Returns `Ok(None)` for a transport whose execution mode is not the
/// `RequestExecutor` seam, such as virtual-clock co-simulation.
fn resolve_native_execution(
    context: &RunContext,
    transport: &dyn ValidatedTransportConfig,
    transport_id: &str,
) -> Result<Option<Arc<dyn NativeTransportExecution>>> {
    let factory = context
        .product_registry()
        .transport_factory(transport_id)
        .ok_or_else(|| anyhow::anyhow!("transport {transport_id:?} is not registered"))?;
    factory.native_execution(transport, context)
}

use crate::engine::sidecar_input::{CONTENT_SERVER_SIDECAR_ID, ContentServerSpec};

/// Dispatch to dynosim or reject it when the feature is disabled.
///
/// `$unused` keeps cfg-disabled arguments live without requiring dynosim symbols.
macro_rules! dynosim_or_unsupported {
    ($transport_id:expr, $workload:literal, $dispatch:expr, $unused:expr $(,)?) => {{
        #[cfg(feature = "dynosim")]
        {
            return $dispatch;
        }
        #[cfg(not(feature = "dynosim"))]
        {
            let _ = $unused;
            anyhow::bail!(
                "transport {:?} does not support the {} workload",
                $transport_id,
                $workload,
            );
        }
    }};
}

/// Emit the `Debug` impl every built-in workload factory shares: the bare type
/// name, since these factories carry only injected collaborators with no
/// inspectable state worth formatting.
macro_rules! workload_factory_debug {
    ($ty:ident) => {
        impl fmt::Debug for $ty {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(stringify!($ty))
            }
        }
    };
}

/// Register the built-in executable workloads (`scheduled`, `graph`).
pub fn register_online_workloads(registry: &mut crate::extensions::AIPerfRegistry) -> Result<()> {
    let tokenizers: Arc<dyn OnlineTokenizerSourceResolver> =
        Arc::new(HfHubOnlineTokenizerSourceResolver::default());
    registry.register_workload(Arc::new(ScheduledWorkloadFactoryV2 {
        tokenizers: tokenizers.clone(),
    }))?;
    registry.register_workload(Arc::new(GraphWorkloadFactoryV2 { tokenizers }))?;
    Ok(())
}

/// Register the HTTP-only static-accuracy workload.
pub fn register_http_static_accuracy_workload(
    registry: &mut crate::extensions::AIPerfRegistry,
) -> Result<()> {
    register_http_static_accuracy_workload_with_factories(
        registry,
        Arc::new(HfHubOnlineTokenizerSourceResolver::default()),
        Arc::new(NativeStaticAccuracyEvaluatorFactory),
    )
}

/// Register static accuracy with distribution-selected preparation factories.
pub fn register_http_static_accuracy_workload_with_factories(
    registry: &mut crate::extensions::AIPerfRegistry,
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
    evaluator_factory: Arc<dyn StaticAccuracyEvaluatorFactory>,
) -> Result<()> {
    registry.register_workload(Arc::new(StaticAccuracyWorkloadFactoryV2 {
        tokenizers,
        evaluator_factory,
    }))
}

/// Built-in scheduled workload for native and dynosim transports.
struct ScheduledWorkloadFactoryV2 {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

workload_factory_debug!(ScheduledWorkloadFactoryV2);

impl WorkloadFactory for ScheduledWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static WorkloadDescriptor {
        &SCHEDULED_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config =
            strict_decode::<ScheduledWorkloadConfigV2>(authored, "scheduled workload config")?;
        validate_common_workload(
            config.worker_count,
            &config.dataset,
            &config.tokenizer,
            &config.phases,
        )?;
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        Ok(inference_workload_requirements())
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
        transport_id: &str,
    ) -> Result<()> {
        let workload = workload_config::<ScheduledWorkloadConfigV2>(workload, "scheduled")?;
        match resolve_native_execution(context, transport, transport_id)? {
            Some(binding) => binding.validate_run(run, context)?,
            None => dynosim_or_unsupported!(
                transport_id,
                "scheduled",
                crate::engine::offline_execution::dynosim_scheduled_validate_run(
                    run, context, transport, workload,
                ),
                run,
            ),
        }
        validate_authored_tokenizer(&workload.tokenizer)
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        transport: Box<dyn ValidatedTransportConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
        transport_id: &str,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        match resolve_native_execution(context, transport.as_ref(), transport_id)? {
            Some(binding) => {
                let workload =
                    workload_config::<ScheduledWorkloadConfigV2>(workload.as_ref(), "scheduled")?;
                let plan = lower_scheduled(run, context, workload, self.tokenizers.as_ref())?;
                prepare_native_operation(run, context, plan, binding)
            }
            None => dynosim_or_unsupported!(
                transport_id,
                "scheduled",
                crate::engine::offline_execution::prepare_dynosim_scheduled(
                    run,
                    context,
                    transport,
                    workload,
                    self.tokenizers.clone(),
                ),
                (run, context, workload),
            ),
        }
    }
}

/// Whether a graph weka run selects the legacy AgentX agentic pipeline.
///
/// `None`/`graph-ir` (any spelling) uses the graph-ir path; `legacy`/`agentx`
/// selects the byte-exact legacy path; any other value is rejected. Kept
/// feature-independent so a lean build still parses and rejects the selector.
fn weka_wants_legacy(semantics: Option<&str>) -> Result<bool> {
    match semantics.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("graph-ir") | Some("graphir") | Some("graph_ir") => Ok(false),
        Some("legacy") | Some("agentx") => Ok(true),
        Some(other) => Err(anyhow!(
            "unknown weka semantics {other:?}; expected 'legacy' or 'graph-ir'"
        )),
    }
}

/// Built-in Graph-IR workload for native and dynosim transports.
struct GraphWorkloadFactoryV2 {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

workload_factory_debug!(GraphWorkloadFactoryV2);

impl WorkloadFactory for GraphWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static WorkloadDescriptor {
        &GRAPH_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config = strict_decode::<GraphWorkloadConfigV2>(authored, "graph workload config")?;
        validate_common_workload(
            config.worker_count,
            &config.dataset,
            &config.tokenizer,
            &config.phases,
        )?;
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        Ok(inference_workload_requirements())
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
        transport_id: &str,
    ) -> Result<()> {
        let workload = workload_config::<GraphWorkloadConfigV2>(workload, "graph")?;
        match resolve_native_execution(context, transport, transport_id)? {
            Some(binding) => {
                binding.validate_run(run, context)?;
                ensure!(
                    run.sidecars.live_streaming.is_none(),
                    "protocol-v2 graph execution supports the content-server and GPU/network/server telemetry side-channels but not the live-streaming record extension"
                );
                ensure!(
                    run.models.items.len() == 1,
                    "online graph execution requires exactly one default model"
                );
                validate_authored_tokenizer(&workload.tokenizer)?;
                // The legacy AgentX path owns its own loader (weka reconstruction),
                // so it does not go through the graph-input identity validation.
                if weka_wants_legacy(workload.weka_semantics.as_deref())? {
                    return Ok(());
                }
                context
                    .graph_inputs()
                    .validate_identity(&workload.dataset)?;
                Ok(())
            }
            None => dynosim_or_unsupported!(
                transport_id,
                "graph",
                crate::engine::offline_execution::dynosim_graph_validate_run(
                    run, context, transport, workload,
                ),
                run,
            ),
        }
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        transport: Box<dyn ValidatedTransportConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
        transport_id: &str,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        match resolve_native_execution(context, transport.as_ref(), transport_id)? {
            Some(binding) => {
                let workload =
                    workload_config::<GraphWorkloadConfigV2>(workload.as_ref(), "graph")?;
                // Legacy AgentX agentic weka path: a separate loader+runtime,
                // selected by `--weka-semantics legacy` (default under an
                // agentic-replay scenario). Graph-ir stays the fall-through.
                if weka_wants_legacy(workload.weka_semantics.as_deref())? {
                    let plan = lower_legacy_agentic(
                        run,
                        context,
                        workload,
                        self.tokenizers.as_ref(),
                        binding.clone(),
                    )?;
                    return prepare_native_operation(run, context, plan, binding);
                }
                let plan = lower_graph(
                    run,
                    context,
                    workload,
                    self.tokenizers.as_ref(),
                    binding.clone(),
                )?;
                prepare_native_operation(run, context, plan, binding)
            }
            None => dynosim_or_unsupported!(
                transport_id,
                "graph",
                crate::engine::offline_execution::prepare_dynosim_graph(
                    run,
                    context,
                    transport,
                    workload,
                    self.tokenizers.clone(),
                ),
                (run, context, workload),
            ),
        }
    }
}

/// Built-in static-accuracy workload (HTTP only).
struct StaticAccuracyWorkloadFactoryV2 {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
    evaluator_factory: Arc<dyn StaticAccuracyEvaluatorFactory>,
}

workload_factory_debug!(StaticAccuracyWorkloadFactoryV2);

impl WorkloadFactory for StaticAccuracyWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static WorkloadDescriptor {
        &STATIC_ACCURACY_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config = strict_decode::<StaticAccuracyWorkloadConfigV2>(
            authored,
            "static accuracy workload config",
        )?;
        validate_common_workload(
            config.worker_count,
            &config.dataset,
            &config.tokenizer,
            &config.phases,
        )?;
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        Ok(inference_workload_requirements())
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        _transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
        transport_id: &str,
    ) -> Result<()> {
        ensure!(
            transport_id == "http",
            "static accuracy execution runs only over the http transport"
        );
        validate_online_run(context)?;
        validate_static_accuracy_endpoint(context)?;
        ensure!(
            run.models.items.len() == 1,
            "static accuracy execution requires exactly one model"
        );
        let workload =
            workload_config::<StaticAccuracyWorkloadConfigV2>(workload, "static_accuracy")?;
        validate_authored_tokenizer(&workload.tokenizer)?;
        StaticAccuracyConfigV2::decode(&workload.accuracy).map(drop)
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunContext,
        transport: Box<dyn ValidatedTransportConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
        transport_id: &str,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        ensure!(
            transport_id == "http",
            "static accuracy execution runs only over the http transport"
        );
        let binding = resolve_native_execution(context, transport.as_ref(), transport_id)?
            .ok_or_else(|| {
                anyhow::anyhow!("static accuracy execution runs only over the http transport")
            })?;
        let workload = workload_config::<StaticAccuracyWorkloadConfigV2>(
            workload.as_ref(),
            "static_accuracy",
        )?;
        let plan = lower_static_accuracy(
            run,
            context,
            workload,
            self.tokenizers.as_ref(),
            self.evaluator_factory.clone(),
        )?;
        prepare_native_operation(run, context, plan, binding)
    }
}

/// Build a prepared operation using its [`NativeTransportExecution`] binding.
fn prepare_native_operation(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    mut plan: NativeRunSpec,
    binding: Arc<dyn NativeTransportExecution>,
) -> Result<Box<dyn PreparedRunnerOperation>> {
    // The driver selects its clock from the resolved transport binding.
    plan.transport = Some(binding.clone());
    let report_facts = native_plan_report_facts(&plan)?;
    let request_executor = binding.executor_factory();
    let readiness = if binding.readiness_enabled() {
        Some(prepare_online_readiness(run, context)?)
    } else {
        None
    };
    let run_metadata = binding.run_metadata();
    Ok(Box::new(PreparedNativeOperation {
        plan,
        request_executor,
        execution_factories: context.execution_factories_handle(),
        product_registry: context.product_registry_handle(),
        readiness,
        report_facts,
        run_metadata,
    }))
}

fn workload_config<'a, T: 'static>(
    config: &'a dyn ValidatedWorkloadConfig,
    workload_id: &str,
) -> Result<&'a T> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("online {workload_id} workload received a different config type"))
}

fn validate_online_run(context: &RunContext) -> Result<()> {
    context.default_endpoint_profile()?;
    for (profile_id, profile) in context.endpoint_profiles() {
        ensure!(
            !profile.config.urls.is_empty(),
            "endpoint profile {profile_id:?} has no URL"
        );
        for url in &profile.config.urls {
            let scheme = Url::parse(url)
                .with_context(|| format!("parsing endpoint profile {profile_id:?} URL"))?
                .scheme()
                .to_string();
            ensure!(
                matches!(scheme.as_str(), "http" | "https"),
                "http endpoint profile {profile_id:?} requires http:// or https:// URLs, got {url:?}"
            );
        }
    }
    Ok(())
}

pub(crate) fn prepare_online_readiness(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
) -> Result<Box<dyn PreparedOnlineReadiness>> {
    let profiles = context
        .endpoint_profiles()
        .map(|(profile_id, profile)| {
            ReadinessEndpointProfile::new(
                profile_id,
                profile.endpoint_id.clone(),
                profile.config.clone(),
                profile.connection_reuse,
                profile.client.clone(),
            )
        })
        .collect::<Vec<_>>();
    let models = run
        .models
        .items
        .iter()
        .map(|model| model.name.clone())
        .collect::<Vec<_>>();
    context
        .execution_factories()
        .readiness_plans()
        .prepare(ReadinessPlanInput {
            endpoints: context.product_registry().endpoints(),
            profiles: &profiles,
            models: &models,
        })
        .context("preparing endpoint-owned readiness plan")
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoredTokenizerV2 {
    name: String,
    #[serde(default = "default_revision")]
    revision: String,
    #[serde(default)]
    trust_remote_code: bool,
    #[serde(default)]
    apply_chat_template: bool,
    /// Opt-in server-side tokenizer origin; offloads tokenization to the
    /// inference server's `/tokenize` and `/detokenize` endpoints.
    #[serde(default)]
    server_url: Option<String>,
}

fn default_revision() -> String {
    "main".into()
}

/// Actionable diagnostic for a tokenizer repository AIPerf cannot load natively.
///
/// The native tokenizer accepts `tokenizer.json` (HuggingFace fast), a native
/// `tiktoken.model` / `tokenizer.model` / `*.tiktoken` BPE vocab (Kimi/Qwen/
/// DeepSeek-class), or a built-in tiktoken encoding, and never executes
/// repository Python. A repository that ships none of these — only a custom
/// Python tokenizer with no vocab file AIPerf can parse — cannot be loaded.
/// Substituting a different tokenizer would silently report wrong token counts —
/// the worst failure for a benchmark tool — so this fails closed with guidance.
fn remote_code_tokenizer_error(name: &str) -> String {
    format!(
        "tokenizer {name:?} cannot be loaded by AIPerf's native tokenizer: the repository \
         provides neither a `tokenizer.json` nor a native tiktoken vocab file \
         (`tiktoken.model`, `tokenizer.model`, or `*.tiktoken`) and requires executing custom \
         repository code (`trust_remote_code`), which the native tokenizer never does. Loading \
         a substitute tokenizer would report incorrect token counts, so AIPerf refuses rather \
         than guess. Point the tokenizer at a repository or local directory that contains a \
         `tokenizer.json` or a tiktoken vocab file, or run with `--use-server-token-count` so \
         the endpoint's reported `usage` supplies authoritative token counts."
    )
}

/// Reject a resolved tokenizer directory that has no native-loadable tokenizer,
/// naming the model in an actionable error. A directory is native-loadable when
/// it holds either a `tokenizer.json` (HuggingFace fast path) or a native
/// tiktoken vocab file (`tiktoken.model` / `tokenizer.model` / `*.tiktoken`).
/// Shared by every resolver so a custom-code-only repository fails identically
/// instead of surfacing a bare "No such file" at [`load_tokenizer`] time.
fn ensure_native_tokenizer_loadable(name: &str, directory: &Path) -> Result<()> {
    ensure!(
        directory.join("tokenizer.json").is_file() || find_tiktoken_model_file(directory).is_some(),
        "{}",
        remote_code_tokenizer_error(name)
    );
    Ok(())
}

/// Preparation-time tokenizer acquisition for protocol-v2 online workloads.
///
/// The native implementation accepts built-in encodings and local paths
/// without IO, or resolves a Hugging Face repository revision to an immutable
/// commit and caches the exact tokenizer files through AIPerf's Clock-injected
/// HTTP fetcher. A distribution with an internal artifact store can replace
/// this implementation without changing workload or coordinator code.
pub trait OnlineTokenizerSourceResolver: Send + Sync {
    /// Resolve authored tokenizer identity to a built-in name or local source.
    fn resolve(&self, name: &str, revision: &str, trust_remote_code: bool) -> Result<String>;
}

/// Resolves built-in, local, and Hugging Face tokenizers.
pub struct NativeOnlineTokenizerSourceResolver {
    fetcher: Arc<dyn DatasetFetcher>,
    cache_directory: PathBuf,
    hugging_face_endpoint: String,
}

impl fmt::Debug for NativeOnlineTokenizerSourceResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeOnlineTokenizerSourceResolver")
            .field("cache_directory", &self.cache_directory)
            .field("hugging_face_endpoint", &self.hugging_face_endpoint)
            .finish_non_exhaustive()
    }
}

impl Default for NativeOnlineTokenizerSourceResolver {
    fn default() -> Self {
        let cache_directory = std::env::var_os("AIPERF_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".cache/aiperf"))
            .join("tokenizers-rust");
        Self {
            fetcher: Arc::new(HttpDatasetFetcher::new(cache_directory.join("downloads"))),
            cache_directory,
            hugging_face_endpoint: std::env::var("HF_ENDPOINT")
                .unwrap_or_else(|_| "https://huggingface.co".into()),
        }
    }
}

impl NativeOnlineTokenizerSourceResolver {
    pub fn new(fetcher: Arc<dyn DatasetFetcher>, cache_directory: PathBuf) -> Self {
        Self {
            fetcher,
            cache_directory,
            hugging_face_endpoint: "https://huggingface.co".into(),
        }
    }

    async fn resolve_remote(&self, name: &str, revision: &str) -> Result<PathBuf> {
        validate_repository_id(name)?;
        let token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
        let metadata_url = hugging_face_model_url(
            &self.hugging_face_endpoint,
            name,
            Some(("revision", revision)),
        )?;
        let metadata_key = format!("hf-tokenizer-metadata:{name}:{revision}");
        let metadata = self
            .fetcher
            .fetch(&metadata_url, &metadata_key, token.as_deref())
            .await
            .with_context(|| format!("resolving tokenizer repository {name:?}@{revision:?}"))?;
        let metadata: Value = serde_json::from_slice(&metadata)
            .context("decoding Hugging Face tokenizer repository metadata")?;
        let commit = metadata
            .get("sha")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Hugging Face tokenizer metadata has no commit sha"))?;
        ensure!(
            commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Hugging Face returned invalid tokenizer commit {commit:?}"
        );
        let resolved_directory = self.cache_directory.join(
            blake3::hash(format!("{name}\0{commit}").as_bytes())
                .to_hex()
                .as_str(),
        );
        let tokenizer_path = resolved_directory.join("tokenizer.json");
        if tokenizer_path.is_file() {
            return Ok(resolved_directory);
        }

        // Fetch `tokenizer.json` (HuggingFace fast path) best-effort. A repository
        // can resolve (commit sha present) yet ship no `tokenizer.json` because it
        // uses a tiktoken vocab (Kimi/Qwen/DeepSeek) or a custom Python tokenizer,
        // so its absence is not fatal here — the tiktoken fallback and the final
        // `ensure_native_tokenizer_loadable` guard decide loadability.
        if let Some(bytes) = self
            .fetch_optional_file(name, commit, "tokenizer.json", token.as_deref())
            .await
        {
            persist_tokenizer_file(&tokenizer_path, bytes.as_ref())?;
        } else {
            // No `tokenizer.json`: pull a native tiktoken vocab file if present,
            // so the loader can tokenize Kimi/Qwen-class models client-side.
            for vocab_name in ["tiktoken.model", "tokenizer.model"] {
                if let Some(bytes) = self
                    .fetch_optional_file(name, commit, vocab_name, token.as_deref())
                    .await
                {
                    persist_tokenizer_file(&resolved_directory.join(vocab_name), bytes.as_ref())?;
                    break;
                }
            }
        }

        // `tokenizer_config.json` (special tokens, BOS/EOS, chat template) and
        // `config.json` (`vocab_size`, `model_type` for the tiktoken regex) enrich
        // loading but are not universal; a missing optional file must not
        // invalidate an otherwise-complete tokenizer.
        for optional in ["tokenizer_config.json", "config.json"] {
            if let Some(bytes) = self
                .fetch_optional_file(name, commit, optional, token.as_deref())
                .await
            {
                persist_tokenizer_file(&resolved_directory.join(optional), bytes.as_ref())?;
            }
        }
        Ok(resolved_directory)
    }

    /// Fetch one repository file at `commit`, returning `None` when it is absent
    /// (any fetch error) rather than failing the whole resolution.
    async fn fetch_optional_file(
        &self,
        name: &str,
        commit: &str,
        filename: &str,
        token: Option<&str>,
    ) -> Option<bytes::Bytes> {
        let url =
            hugging_face_model_url(&self.hugging_face_endpoint, name, Some(("resolve", commit)))
                .ok()?
                + &format!("/{filename}");
        self.fetcher
            .fetch(
                &url,
                &format!("hf-tokenizer-file:{name}:{commit}:{filename}"),
                token,
            )
            .await
            .ok()
    }
}

impl OnlineTokenizerSourceResolver for NativeOnlineTokenizerSourceResolver {
    fn resolve(&self, name: &str, revision: &str, trust_remote_code: bool) -> Result<String> {
        if let Some(local) = resolve_builtin_or_local(name, trust_remote_code)? {
            return Ok(local);
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating tokenizer preparation runtime")?;
        let local = tokio::task::LocalSet::new();
        let resolved = local.block_on(&runtime, self.resolve_remote(name, revision))?;
        ensure_native_tokenizer_loadable(name, &resolved)?;
        tokenizer_directory_to_string(&resolved)
    }
}

/// Built-in encodings and local filesystem tokenizers resolve without any IO.
///
/// Returns `Ok(Some(source))` when `name` is a tiktoken encoding or an existing
/// local path, and `Ok(None)` when a remote Hugging Face download is required.
/// Shared by every resolver so the no-network short-circuit stays identical
/// regardless of download backend. `trust_remote_code` is intentionally ignored
/// here: the native tokenizer never executes repository code, so the flag is
/// inert (it is warned about once at policy-decode time, not refused).
fn resolve_builtin_or_local(name: &str, _trust_remote_code: bool) -> Result<Option<String>> {
    let path = Path::new(name);
    if path.is_dir() || path.is_file() || name.parse::<TiktokenEncoding>().is_ok() {
        return Ok(Some(name.to_owned()));
    }
    Ok(None)
}

/// Render a resolved tokenizer cache directory as a UTF-8 source string.
fn tokenizer_directory_to_string(directory: &Path) -> Result<String> {
    directory
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("resolved tokenizer cache path is not valid UTF-8"))
}

/// Hugging Face tokenizer resolver backed by the `hf-hub` crate.
///
/// Delegates repository download, caching, and offline handling to `hf-hub`
/// (via [`download_hugging_face_tokenizer`]), which follows the xet CDN `302`
/// redirect, reuses the standard `~/.cache/huggingface` cache across runs, and
/// honors `HF_HUB_OFFLINE`. Built-in encodings and local paths short-circuit
/// without IO. `hf-hub` acquires the `main` revision, so a pinned non-`main`
/// revision falls back to the exact-commit HTTP resolver.
#[derive(Default)]
pub struct HfHubOnlineTokenizerSourceResolver {
    pinned_revision_fallback: NativeOnlineTokenizerSourceResolver,
}

impl fmt::Debug for HfHubOnlineTokenizerSourceResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HfHubOnlineTokenizerSourceResolver")
            .finish_non_exhaustive()
    }
}

impl OnlineTokenizerSourceResolver for HfHubOnlineTokenizerSourceResolver {
    fn resolve(&self, name: &str, revision: &str, trust_remote_code: bool) -> Result<String> {
        if let Some(local) = resolve_builtin_or_local(name, trust_remote_code)? {
            return Ok(local);
        }
        // hf-hub resolves branch tips, so pinned revisions require the
        // exact-commit HTTP resolver.
        if revision != default_revision() {
            return self
                .pinned_revision_fallback
                .resolve(name, revision, trust_remote_code);
        }
        validate_repository_id(name)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating tokenizer preparation runtime")?;
        let directory = runtime
            .block_on(download_hugging_face_tokenizer(name))
            .with_context(|| format!("resolving tokenizer repository {name:?}"))?;
        // `download_hugging_face_tokenizer` accepts a repository whose only
        // tokenizer artifact is `tokenizer_config.json` (a Kimi-class custom-code
        // repo), returning a directory with no native-loadable `tokenizer.json`.
        // Fail here with actionable guidance rather than a bare "No such file"
        // from `from_directory`, and never substitute a wrong tokenizer.
        ensure_native_tokenizer_loadable(name, &directory)?;
        tokenizer_directory_to_string(&directory)
    }
}

fn validate_repository_id(value: &str) -> Result<()> {
    // Hugging Face repository IDs are either `namespace/name` or a bare model
    // ID (`gpt2`, `bert-base-uncased`, `t5-small`). Both are valid `hf-hub`
    // repositories, so accept one or more
    // segments; only reject empty segments and path-traversal components (local
    // paths and tiktoken encodings are already resolved before this point).
    let segments = value.split('/').collect::<Vec<_>>();
    ensure!(
        !value.is_empty()
            && segments
                .iter()
                .all(|segment| { !segment.is_empty() && *segment != "." && *segment != ".." }),
        "remote tokenizer name {value:?} must be a Hugging Face repository ID"
    );
    Ok(())
}

fn hugging_face_model_url(
    endpoint: &str,
    repository: &str,
    suffix: Option<(&str, &str)>,
) -> Result<String> {
    let mut url = Url::parse(endpoint)?;
    {
        let mut base = url
            .path_segments_mut()
            .map_err(|_| anyhow!("Hugging Face endpoint cannot accept path segments"))?;
        base.pop_if_empty();
        base.extend(["api", "models"]);
    }
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| anyhow!("Hugging Face model URL cannot accept path segments"))?;
        path.pop_if_empty();
        path.extend(repository.split('/'));
        if let Some((kind, value)) = suffix {
            if kind == "resolve" {
                drop(path);
                let mut file = Url::parse(endpoint)?;
                {
                    let mut file_path = file.path_segments_mut().map_err(|_| {
                        anyhow!("Hugging Face file URL cannot accept path segments")
                    })?;
                    file_path.pop_if_empty();
                    file_path.extend(repository.split('/').chain([kind, value]));
                }
                return Ok(file.into());
            }
            path.extend([kind, value]);
        }
    }
    Ok(url.into())
}

fn persist_tokenizer_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("tokenizer cache file {} has no parent", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

impl AuthoredTokenizerV2 {
    fn decode(raw: &RawValue) -> Result<Self> {
        let config: Self = serde_json::from_str(raw.get())
            .context("decoding protocol-v2 tokenizer acquisition policy")?;
        ensure!(
            !config.name.trim().is_empty() && config.name.trim() == config.name,
            "tokenizer.name must be non-empty and contain no surrounding whitespace"
        );
        ensure!(
            !config.revision.trim().is_empty(),
            "tokenizer.revision must not be empty"
        );
        if config.trust_remote_code {
            // The native `tokenizers` library loads tokenizer artifacts directly
            // and never executes repository Python, so `trust_remote_code=true`
            // is inert rather than a capability. Accept it and warn instead of
            // failing closed so command lines that pass the flag unconditionally
            // (e.g. Hugging Face / SGLang benchmark harnesses) still run; the
            // tokenizer loads exactly as if the flag were false.
            tracing::warn!(
                tokenizer = %config.name,
                "tokenizer.trust_remote_code=true has no effect: the native tokenizer never \
                 executes repository code; loading the tokenizer normally"
            );
        }
        Ok(config)
    }

    fn lower(&self, resolver: &dyn OnlineTokenizerSourceResolver) -> Result<TokenizerSpec> {
        // A server-side tokenizer needs no local source resolution: the origin
        // owns the vocabulary and `name` is only the model selector forwarded to
        // it. Skip the (possibly network / gated-model) resolve in that case.
        if let Some(server_url) = self.server_url.clone() {
            return Ok(TokenizerSpec {
                name: self.name.clone(),
                apply_chat_template: self.apply_chat_template,
                server_url: Some(server_url),
            });
        }
        Ok(TokenizerSpec {
            name: resolver.resolve(&self.name, &self.revision, self.trust_remote_code)?,
            apply_chat_template: self.apply_chat_template,
            server_url: None,
        })
    }

    /// Use a built-in tokenizer for endpoints with no token metrics.
    ///
    /// This avoids resolving an unused model repository. Chat templates are
    /// disabled because applying one would require tokenization.
    fn lower_builtin(&self) -> TokenizerSpec {
        TokenizerSpec {
            name: "builtin".into(),
            apply_chat_template: false,
            server_url: None,
        }
    }
}

/// Whether the selected endpoint descriptor requires a real tokenizer.
///
/// AIPerf only tokenizes when the endpoint tokenizes its input or produces
/// output tokens; `base_metrics_processor` filters token metrics by the same
/// two flags, so a descriptor with both false has no token metrics and needs no
/// tokenizer. Descriptor-driven gating avoids enumerating endpoint IDs.
fn endpoint_needs_tokenizer(descriptor: &crate::endpoints::EndpointDescriptor) -> bool {
    descriptor.tokenizes_input || descriptor.produces_tokens
}

/// Decode and validate the authored tokenizer, then resolve its source only
/// when the selected endpoint descriptor requires tokenization. Validation is
/// unconditional so a malformed policy still fails; source resolution (and its
/// network / gated-model IO) is gated on [`endpoint_needs_tokenizer`].
fn lower_tokenizer_for_endpoint(
    raw: &RawValue,
    resolver: &dyn OnlineTokenizerSourceResolver,
    descriptor: &crate::endpoints::EndpointDescriptor,
) -> Result<TokenizerSpec> {
    let authored = AuthoredTokenizerV2::decode(raw)?;
    if endpoint_needs_tokenizer(descriptor) {
        authored.lower(resolver)
    } else {
        Ok(authored.lower_builtin())
    }
}

/// Validate an authored tokenizer policy without resolving its source.
pub(crate) fn validate_authored_tokenizer(raw: &RawValue) -> Result<()> {
    AuthoredTokenizerV2::decode(raw).map(|_| ())
}

/// Resolve an authored tokenizer through an injected source resolver.
pub(crate) fn lower_authored_tokenizer(
    raw: &RawValue,
    resolver: &dyn OnlineTokenizerSourceResolver,
) -> Result<TokenizerSpec> {
    AuthoredTokenizerV2::decode(raw)?.lower(resolver)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StaticAccuracyConfigV2 {
    benchmark: String,
    #[serde(default)]
    tasks: Option<Vec<String>>,
    #[serde(default)]
    n_shots: Option<usize>,
    #[serde(default)]
    enable_cot: Option<bool>,
    #[serde(default)]
    grader: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    python_executable: PathBuf,
    #[serde(default = "default_accuracy_worker_module")]
    worker_module: String,
    #[serde(default)]
    verbose: bool,
}

fn default_accuracy_worker_module() -> String {
    "aiperf.accuracy.worker".into()
}

impl StaticAccuracyConfigV2 {
    fn decode(raw: &RawValue) -> Result<Self> {
        let config: Self = serde_json::from_str(raw.get())
            .context("decoding protocol-v2 static accuracy policy")?;
        ensure!(
            !config.benchmark.trim().is_empty(),
            "accuracy.benchmark must not be empty"
        );
        ensure!(
            config.python_executable.is_absolute(),
            "accuracy.python_executable must be absolute"
        );
        ensure!(
            !config.worker_module.trim().is_empty(),
            "accuracy.worker_module must not be empty"
        );
        ensure!(
            !config.verbose,
            "accuracy.verbose is presentation policy and is not accepted by the native evaluator adapter"
        );
        Ok(config)
    }

    fn lower(
        self,
        evaluator_factory: Arc<dyn StaticAccuracyEvaluatorFactory>,
    ) -> NativeStaticAccuracyPlan {
        NativeStaticAccuracyPlan {
            benchmark: self.benchmark,
            tasks: self.tasks,
            n_shots: self.n_shots,
            enable_cot: self.enable_cot,
            grader: self.grader,
            system_prompt: self.system_prompt,
            process: StaticAccuracyEvaluatorProcessSpec {
                python_executable: self.python_executable,
                worker_module: self.worker_module,
            },
            evaluator_factory,
        }
    }
}

pub(crate) fn lower_scheduled(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    workload: &ScheduledWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunSpec> {
    // Endpoint capabilities determine whether tokenizer resolution may perform
    // remote or gated-model I/O.
    let profile = context.default_endpoint_profile()?;
    let prepared_endpoint = context
        .product_registry()
        .endpoints()
        .prepare(&profile.endpoint_id, profile.config.clone())
        .context("preparing the default endpoint for linear dataset composition")?;
    let endpoint_descriptor = prepared_endpoint.descriptor();
    let tokenizer =
        lower_tokenizer_for_endpoint(&workload.tokenizer, tokenizers, endpoint_descriptor)?;
    let tokenizer_impl = load_tokenizer(Some(&tokenizer.name))?;
    // The rankings composer synthesizes text query/passage rows and is correct
    // only for text-input rerankers. `image_retrieval` also advertises a
    // `Rankings` output but consumes image input, so keying solely on the output
    // modality would misroute it to the text composer and leave every turn
    // imageless (the request builder then rejects it). Restrict the rankings
    // composer to endpoints whose input is not image-based so image-input
    // reranking flows through the media-generating synthetic composer.
    let rankings = endpoint_descriptor
        .output_modalities
        .contains(&Modality::Rankings)
        && !endpoint_descriptor
            .input_modalities
            .contains(&Modality::Image);
    let prepare_context = DatasetInputContext {
        registry: context.product_registry(),
        models: &run.models,
        run_rng_root: RngRoot::new(run.identity.random_seed),
        tokenizer: tokenizer_impl.as_ref(),
        rankings,
        endpoint_descriptor,
        trace_prompt_storage: Arc::new(MaterializedTracePromptStorage),
        media_generator_factory: media_generator_factory(context)?,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating linear dataset preparation runtime")?;
    let local = tokio::task::LocalSet::new();
    let dataset = local
        .block_on(
            &runtime,
            context
                .dataset_inputs()
                .load(&workload.dataset, &prepare_context),
        )
        .context("loading and validating authored linear dataset")?;
    build_common_plan(
        run,
        workload.worker_count,
        NativeDatasetPlan::PreparedLinear(dataset),
        tokenizer,
        &workload.phases,
        NativeEndpointPlan::Prepared(context.endpoint_profiles_handle()),
        NativeSidecarPlan::Prepared(context.sidecar_inputs_handle()),
        workload.failure_policy,
        None,
    )
}

fn media_generator_factory(
    context: &RunContext,
) -> Result<Arc<dyn SyntheticMediaGeneratorFactory>> {
    let Some(spec) = context
        .sidecar_inputs()
        .get::<ContentServerSpec>(CONTENT_SERVER_SIDECAR_ID)?
    else {
        return Ok(Arc::new(NativeSyntheticMediaGeneratorFactory::default()));
    };
    let Some(content_dir) = &spec.content_dir else {
        // Without a content directory, generated media cannot be published
        // through the content server.
        return Ok(Arc::new(NativeSyntheticMediaGeneratorFactory::default()));
    };
    let publisher = ContentServerMediaPublisher::new(content_dir, spec.base_url())
        .context("preparing synthetic media for the content server")?;
    Ok(Arc::new(NativeSyntheticMediaGeneratorFactory::new(
        Arc::new(publisher),
    )))
}

fn lower_graph(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    workload: &GraphWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
    transport: Arc<dyn crate::engine::registry::NativeTransportExecution>,
) -> Result<NativeRunSpec> {
    let tokenizer = lower_authored_tokenizer(&workload.tokenizer, tokenizers)?;
    let tokenizer_impl = load_tokenizer(Some(&tokenizer.name))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating direct Graph-IR preparation runtime")?;
    let local = tokio::task::LocalSet::new();
    let prepared = local
        .block_on(
            &runtime,
            context.graph_inputs().load(
                &workload.dataset,
                &GraphInputContext {
                    tokenizer: tokenizer_impl.as_ref(),
                    run_random_seed: run.identity.random_seed,
                },
            ),
        )
        .context("loading direct authored Graph-IR input")?;
    validate_graph_endpoint_profile_references(&prepared.bundle, context)?;
    let dataset = NativeGraphDatasetPlan {
        input: Arc::new(prepared.bundle),
        random_seed: prepared.random_seed,
        default_output_tokens: prepared.default_output_tokens,
        allow_dataset_wrap: prepared.allow_dataset_wrap,
        t_star_window: prepared.t_star_window,
        cache_bust_target: prepared.cache_bust_target,
        ignore_trace_delays: workload.ignore_trace_delays,
        system_idle_gap_cap_seconds: workload.system_idle_gap_cap_seconds,
    };
    build_common_plan(
        run,
        workload.worker_count,
        NativeDatasetPlan::Graph(Box::new(dataset)),
        tokenizer,
        &workload.phases,
        NativeEndpointPlan::Prepared(context.endpoint_profiles_handle()),
        NativeSidecarPlan::Prepared(context.sidecar_inputs_handle()),
        workload.failure_policy,
        Some(transport),
    )
}

/// Recover the authored accelerated cache-warmup duration
/// (`--agentic-cache-warmup-duration`) from a legacy-weka run's authored phases.
///
/// The CLI stamps the flag onto a phase's `agentic_cache_warmup_duration`; the
/// legacy path synthesizes its own WARMUP phase and replaces the authored ones,
/// so the value is recovered here (first authored phase carrying it) and threaded
/// onto the synthesized warmup phase. Returns `None` when unset.
fn warmup_agentic_cache_duration(phases: &[PhaseSpec]) -> Option<f64> {
    phases
        .iter()
        .find_map(|phase| phase.common().agentic_cache_warmup_duration)
}

/// Default upper bound for accelerated-cache agentic warmup grace (Python
/// `_AGENTIC_CACHE_WARMUP_DEFAULT_GRACE_PERIOD_SEC`).
const AGENTIC_CACHE_WARMUP_DEFAULT_GRACE_PERIOD_SEC: f64 = 300.0;

/// Resolve synthesized agentic-warmup grace from the profiling phase common.
///
/// `None` leaves the phase on the warmup default (infinite grace).
fn resolve_agentic_warmup_grace(
    profiling: &crate::engine::protocol::PhaseCommonSpec,
) -> Option<f64> {
    if let Some(grace) = profiling.agentic_warmup_grace_period {
        return Some(grace);
    }
    let cache_warmup = profiling.agentic_cache_warmup_duration?;
    let default_grace = cache_warmup.min(AGENTIC_CACHE_WARMUP_DEFAULT_GRACE_PERIOD_SEC);
    Some(match profiling.grace_period {
        None => default_grace,
        Some(benchmark_grace) => benchmark_grace.max(default_grace),
    })
}

/// Lower a legacy-AgentX weka run into a scheduled `NativeRunSpec` driven by the
/// `agentic_replay` timing mode. Reconstructs the WEKA trajectories with the
/// byte-exact AgentX loader, composes them into a verbatim-replay linear dataset,
/// translates the phases to `agentic_replay`, and reuses the shared scheduled
/// execution (`prepare_native_operation`) for transport/metrics/records/report.
fn lower_legacy_agentic(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    workload: &GraphWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
    transport: Arc<dyn crate::engine::registry::NativeTransportExecution>,
) -> Result<NativeRunSpec> {
    use crate::agentx::config::WekaConfig;
    use crate::agentx::corpus::CorpusTokenSynth;
    use crate::agentx::hf_dataset::{HfDatasetRef, load_hf_weka_traces};
    use crate::agentx::loader::{MainReconstructOptions, convert_traces_parallel};
    use crate::agentx::weka_dataset::{WekaComposeOptions, compose_weka_agentic_dataset};
    use crate::rng::compat::python_random::PythonRandomGenerator;
    use std::collections::HashMap;

    // Defense-in-depth, fail-closed: agentic_replay's join-gating + cross-lane
    // alignment assume a single global issuance order. The CLI already forces
    // global-hop dispatch (and non-cellular execution) for this workload, but do
    // not trust that guard here — reject any run that reaches lowering with a
    // different dispatch mode. (Cell count is not readable at this lowering point
    // — it lives in the `cfg.runtime.cells` envelope, not on `AuthoredRunSpecV2`
    // or `RunContext` — so the non-cellular half stays enforced at the CLI.)
    ensure!(
        run.dispatch == crate::engine::protocol::DispatchMode::GlobalHop,
        "agentic_replay requires global-hop dispatch (runtime.dispatch = global-hop)"
    );

    let tokenizer = lower_authored_tokenizer(&workload.tokenizer, tokenizers)?;
    let tokenizer_impl = load_tokenizer(Some(&tokenizer.name))?;

    // Decode the HuggingFace weka source from the authored dataset descriptor.
    let dataset_json: serde_json::Value = serde_json::from_str(workload.dataset.get())
        .context("decoding legacy weka dataset descriptor")?;
    let source = dataset_json
        .get("source")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let hf_name = source
        .get("dataset")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("legacy weka requires a hugging_face dataset source"))?;
    let subset = source
        .get("subset")
        .and_then(|v| v.as_str())
        .filter(|s| *s != "default")
        .map(str::to_string);
    let split = source
        .get("split")
        .and_then(|v| v.as_str())
        .unwrap_or("train")
        .to_string();
    let revision = source
        .get("revision")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let entries = dataset_json
        .get("entries")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);

    // Reconstruct: HF load (async) → agentx byte-exact reconstruction.
    let root_seed = run.identity.random_seed.unwrap_or(0);
    let hash_base_seed = PythonRandomGenerator::derive_child_seed(
        root_seed,
        crate::rng::namespace::DATASET_CODING_CONTENT_CORPUS,
    );
    let corpus = crate::dataset::coding::build_coding_corpus(tokenizer_impl.as_ref(), root_seed)
        .map_err(|error| anyhow!(error.to_string()))?;
    let corpus = std::sync::Arc::new(corpus);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating legacy weka reconstruction runtime")?;
    let local = tokio::task::LocalSet::new();
    let (traces, _stats) = local
        .block_on(
            &rt,
            load_hf_weka_traces(
                HfDatasetRef {
                    name: hf_name.to_string(),
                    subset,
                    split,
                    revision,
                    max_rows: entries,
                },
                entries,
                dataset_json
                    .get("synthesis")
                    .and_then(|s| s.get("max_context_length"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|n| n as i64),
                None,
            ),
        )
        .map_err(|error| anyhow!(error))?;

    // The Python oracle enables WEKA_SPLIT_FLATTENED_AGENTS by default (True),
    // so it carves interleaved worker/flat chains out of the main conversation
    // via hash-LCP detection (removing e.g. a small-context reset request from
    // the main normals into its own `::fa:`/`::wg:` lane). Use the Rust default
    // (also true) so the main-conversation turn sequence matches byte-for-byte.
    let cfg = WekaConfig::default();
    // Prefer authored synthesis idle-gap / think-time knobs; preserve the
    // historical agentic default of 10s when unset.
    let synth = dataset_json.get("synthesis");
    let idle_gap_cap_seconds = synth
        .and_then(|s| s.get("idle_gap_cap_seconds"))
        .and_then(serde_json::Value::as_f64)
        .or(Some(10.0));
    let think_time_only = synth
        .and_then(|s| s.get("use_think_time_only"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let burst_phase_starts = synth
        .and_then(|s| s.get("burst_phase_starts"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let cache_bust_target = match synth
        .and_then(|s| s.get("cache_bust_target"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            dataset_json
                .get("cache_bust")
                .and_then(|c| c.get("target"))
                .and_then(serde_json::Value::as_str)
        }) {
        Some("none") => crate::agentx::cache_bust::CacheBustTarget::None,
        Some("system_prefix") => crate::agentx::cache_bust::CacheBustTarget::SystemPrefix,
        Some("system_suffix") => crate::agentx::cache_bust::CacheBustTarget::SystemSuffix,
        Some("first_turn_suffix") => crate::agentx::cache_bust::CacheBustTarget::FirstTurnSuffix,
        Some("first_turn_prefix") | None => {
            crate::agentx::cache_bust::CacheBustTarget::FirstTurnPrefix
        }
        Some(_) => crate::agentx::cache_bust::CacheBustTarget::FirstTurnPrefix,
    };
    let opts = MainReconstructOptions {
        idle_gap_cap_seconds,
        think_time_only,
        ..MainReconstructOptions::default()
    };
    let results =
        convert_traces_parallel(&traces, &HashMap::new(), &cfg, &opts, |tid: &str, bs| {
            let tok = tokenizer_impl.clone();
            CorpusTokenSynth::new(
                (*corpus).clone(),
                bs,
                hash_base_seed,
                tid,
                move |t: &[u32]| tok.decode(t).unwrap_or_default(),
            )
        });
    let convs: Vec<_> = results
        .into_iter()
        .filter_map(Result::ok)
        .flatten()
        .collect();
    ensure!(
        !convs.is_empty(),
        "legacy weka reconstruction produced no conversations"
    );
    // Apply the per-lane t\* snapshot slice (Python `AgenticReplayStrategy`): sample
    // t\* once over each lane's full recorded span, EXCLUDE history (turns before
    // the profiling resume point), and rebase each retained turn's `timestamp_ms`
    // to its t\*-relative dispatch offset. The `agentic_replay` workload then owns
    // only the cross-lane phase-start alignment + dispatch.
    let convs = crate::agentx::weka_dataset::slice_trajectories_at_tstar(
        convs,
        root_seed,
        0.0,
        1.0,
        Some(10_000.0),
    );
    ensure!(
        !convs.is_empty(),
        "legacy weka trajectories produced no profiling turns at t*"
    );

    // Compose the delta-message linear dataset (history-accumulating dispatch).
    let count_tokens = |text: &str| tokenizer_impl.as_ref().count(text).unwrap_or(0);
    let dataset = compose_weka_agentic_dataset(
        &convs,
        &WekaComposeOptions {
            streaming: true,
            ignore_eos: true,
            benchmark_id: run.identity.benchmark_id.clone(),
            cache_bust_target,
        },
        &count_tokens,
    )?;
    // Build the DAG-free subagent join-gate side channel from the SLICED
    // profiling conversations: the slice preserves each surviving turn's
    // `join_prerequisite` and reindexes turn indices to their profiling
    // (post-t\*, history-excluded) position, so `build_tree_specs` reads the
    // exact index the workload gate consults — no manual remap needed. Empty
    // for a root-only run (the workload gate then stays a pass-through).
    let agentic_trees = std::sync::Arc::new(crate::agentic_replay::build_tree_specs(&convs));

    // Install a live cross-phase accelerated cache-warmup carrier only when the
    // run authored `--agentic-cache-warmup-duration`; otherwise an empty carrier
    // keeps profiling on the exact non-accelerated path. The typed carrier is
    // stored type-erased so the non-`agentx` phase-plan plumbing can thread it.
    let warmup_handoff: crate::agentic_tree::WarmupHandoffCarrierAny =
        if warmup_agentic_cache_duration(&workload.phases).is_some() {
            crate::agentic_replay::new_warmup_handoff_carrier()
        } else {
            crate::agentic_tree::empty_warmup_handoff_carrier()
        };
    let prepared = crate::engine::dataset_input::PreparedDatasetInput {
        dataset,
        random_seed: run.identity.random_seed,
        default_output_tokens: 1,
        agentic_trees,
        warmup_handoff,
    };

    // Translate each authored phase to the agentic_replay timing mode, and
    // prepend a WARMUP phase (Python runs warmup as a separate barrier before
    // profiling) so the turn-(n-1) primes dispatch and drain first. The warmup
    // phase excludes itself from results and carries no stop bound (it drains).
    let profiling_common = workload
        .phases
        .first()
        .map(|p| p.common().clone())
        .ok_or_else(|| anyhow!("legacy weka run has no authored phase to profile"))?;
    let mut warmup_common = profiling_common.clone();
    warmup_common.name = "warmup".to_string();
    warmup_common.kind = Some(crate::engine::protocol::PhaseRoleSpec::Warmup);
    warmup_common.exclude_from_results = true;
    // Thread the accelerated cache-warmup duration onto the synthesized WARMUP
    // phase (`--agentic-cache-warmup-duration`). The CLI stamps it on an authored
    // phase's common; this synthesized warmup replaces those, so recover the
    // authored value from any authored phase and carry it forward so
    // `dataset_build` can populate `AgenticReplayConfig.cache_warmup_duration_s`.
    warmup_common.agentic_cache_warmup_duration = warmup_agentic_cache_duration(&workload.phases);
    // Warmup barrier grace is independent of profiling `--grace-period`. Explicit
    // `--agentic-warmup-grace-period` wins; otherwise accelerated cache-warmup
    // gets a bounded default; plain snapshot warmup keeps infinite grace.
    let mut grace_source = profiling_common.clone();
    if grace_source.agentic_cache_warmup_duration.is_none() {
        grace_source.agentic_cache_warmup_duration = warmup_common.agentic_cache_warmup_duration;
    }
    warmup_common.grace_period = resolve_agentic_warmup_grace(&grace_source);
    warmup_common.agentic_warmup_grace_period = None;
    warmup_common.failed_request_threshold = None;
    // The warmup phase dispatches each warmup conversation exactly once (no
    // recycle), so its record count is known. Set `requests` to that count so the
    // per-phase record-ordinal base of the following PROFILING phase is offset past
    // the warmup range (`compute_phase_ordinal_bases` strides by `requests`);
    // otherwise both phases get base 0 and their records collide at absolute slot 0
    // under the striding issuer (global-hop / cellular). `enforce_stop=false` means
    // this does not gate warmup dispatch — it only offsets the ordinal space.
    let warmup_count = convs
        .iter()
        .filter(|c| {
            c.session_id
                .ends_with(crate::agentx::weka_dataset::WARMUP_SUFFIX)
        })
        .count();
    warmup_common.requests = Some(warmup_count as u64);
    warmup_common.sessions = None;
    warmup_common.duration = None;
    let agentic_replay_phase =
        |common: crate::engine::protocol::PhaseCommonSpec| PhaseSpec::AgenticReplay {
            common,
            start_min_ratio: 0.0,
            start_max_ratio: 1.0,
            idle_gap_cap_seconds,
            system_idle_gap_cap_seconds: workload.system_idle_gap_cap_seconds,
            burst_phase_starts,
        };
    let mut agentic_phases: Vec<PhaseSpec> = vec![agentic_replay_phase(warmup_common)];
    // Extend with the authored PROFILING phases only. `--agentic-cache-warmup-duration`
    // makes the CLI author its own warmup phase (named "warmup"); the synthesized
    // warmup above already represents the warmup barrier and carries the recovered
    // duration, so re-emitting an authored warmup here would produce a duplicate
    // phase id "warmup". Drop any authored warmup-role phase; keep profiling.
    agentic_phases.extend(
        workload
            .phases
            .iter()
            .filter(|p| !p.common().is_warmup())
            .map(|p| agentic_replay_phase(p.common().clone())),
    );

    // agentic_replay runs under `global-hop` (forced at CLI resolution): ONE
    // coordinator scheduling loop is the single central driver, and `worker_count`
    // is the size of the transport-thread pool it hops requests to. Cellular
    // (`cells > 1`) and non-global-hop dispatch are rejected at CLI resolution.
    build_common_plan(
        run,
        workload.worker_count,
        NativeDatasetPlan::PreparedLinear(prepared),
        tokenizer,
        &agentic_phases,
        NativeEndpointPlan::Prepared(context.endpoint_profiles_handle()),
        NativeSidecarPlan::Prepared(context.sidecar_inputs_handle()),
        workload.failure_policy,
        Some(transport),
    )
}

fn lower_static_accuracy(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    workload: &StaticAccuracyWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
    evaluator_factory: Arc<dyn StaticAccuracyEvaluatorFactory>,
) -> Result<NativeRunSpec> {
    validate_static_accuracy_endpoint(context)?;
    let accuracy = StaticAccuracyConfigV2::decode(&workload.accuracy)?.lower(evaluator_factory);
    let tokenizer = lower_authored_tokenizer(&workload.tokenizer, tokenizers)?;
    build_common_plan(
        run,
        workload.worker_count,
        NativeDatasetPlan::StaticAccuracy(accuracy),
        tokenizer,
        &workload.phases,
        NativeEndpointPlan::Prepared(context.endpoint_profiles_handle()),
        NativeSidecarPlan::Prepared(context.sidecar_inputs_handle()),
        // Static accuracy uses the resilient default failure policy.
        None,
        None,
    )
}

fn validate_static_accuracy_endpoint(context: &RunContext) -> Result<()> {
    let profile = context.default_endpoint_profile()?;
    let descriptor = context
        .product_registry()
        .endpoints()
        .resolve_factory(&profile.endpoint_id)
        .context("resolving static-accuracy endpoint")?
        .descriptor();
    ensure!(
        !descriptor.requires_raw_token_ids,
        "static accuracy endpoint {:?} requires raw token IDs, but evaluator-authored problems provide semantic text",
        descriptor.id
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_common_plan(
    run: &AuthoredRunSpecV2,
    workers: usize,
    dataset: NativeDatasetPlan,
    tokenizer: TokenizerSpec,
    phases: &[PhaseSpec],
    endpoint: NativeEndpointPlan,
    sidecars: NativeSidecarPlan,
    failure_policy: Option<OnFailure>,
    transport: Option<Arc<dyn crate::engine::registry::NativeTransportExecution>>,
) -> Result<NativeRunSpec> {
    Ok(NativeRunSpec {
        benchmark_id: run.identity.benchmark_id.clone(),
        random_seed: run.identity.random_seed,
        workers,
        artifact_dir: run.artifact_target.clone(),
        models: run.models.clone(),
        endpoint,
        dataset,
        tokenizer,
        phases: phases.to_vec(),
        metrics: run.metrics.clone(),
        artifacts: ArtifactSpec {
            records_path: run.artifacts.records_path.clone(),
            records_parquet_path: run.artifacts.records_parquet_path.clone(),
            records_csv_path: run.artifacts.records_csv_path.clone(),
            raw_path: run.artifacts.raw_path.clone(),
            outputs_path: run.artifacts.outputs_path.clone(),
            inputs_path: run.artifacts.inputs_path.clone(),
            trace: run.artifacts.trace,
            dataset_analysis_path: run.artifacts.dataset_analysis_path.clone(),
            dataset_analysis_block_size: run.artifacts.dataset_analysis_block_size,
            dataset_analysis_cache_blocks: run.artifacts.dataset_analysis_cache_blocks,
            dataset_analysis_per_conversation: run.artifacts.dataset_analysis_per_conversation,
        },
        sidecars,
        user_files: run.artifacts.user_files.clone(),
        failure_policy,
        native_otel_enabled: run.export.otel.enabled && run.export.otel.endpoint.is_some(),
        transport,
        dispatch_mode: run.dispatch,
        hop_routing: run.hop_routing,
    })
}

fn validate_graph_endpoint_profile_references(
    bundle: &crate::graph::input::GraphInputBundle,
    context: &RunContext,
) -> Result<()> {
    context.default_endpoint_profile()?;
    for plan in &bundle.plans {
        for (node_id, node) in &plan.graph.nodes {
            if let Some(profile_id) = node.metadata.get("endpoint").and_then(Value::as_str) {
                context.endpoint_profile(profile_id).with_context(|| {
                    format!(
                        "Graph-IR trace {:?} node {node_id:?} endpoint profile",
                        plan.trace.id
                    )
                })?;
            }
        }
    }
    Ok(())
}

fn native_plan_report_facts(plan: &NativeRunSpec) -> Result<ReportPairRunFacts> {
    let NativeDatasetPlan::Graph(graph) = &plan.dataset else {
        return Ok(ReportPairRunFacts::new());
    };
    let graph = ReportGraphRunInfo::new(
        graph.input.metadata.format.clone(),
        graph.input.metadata.root_count,
        graph.input.metadata.node_count,
        plan.workers,
        plan.phases.len(),
    )?;
    Ok(ReportPairRunFacts::new().with_graph(graph))
}

/// Prepared native operation with resolved execution and readiness policy.
struct PreparedNativeOperation {
    plan: NativeRunSpec,
    request_executor: Arc<dyn RequestExecutorFactory>,
    execution_factories: ExecutionFactories,
    product_registry: Arc<crate::extensions::AIPerfRegistry>,
    readiness: Option<Box<dyn PreparedOnlineReadiness>>,
    report_facts: ReportPairRunFacts,
    run_metadata: BTreeMap<String, String>,
}

impl fmt::Debug for PreparedNativeOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedNativeOperation")
            .field("benchmark_id", &self.plan.benchmark_id)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedNativeOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let native_report = execute_prepared_native_plan_uncommitted_selected(
            self.plan,
            self.request_executor,
            &self.execution_factories,
            self.product_registry.as_ref(),
            self.readiness,
        )?;
        Ok(PreparedRunOutcome {
            native_report,
            report_facts: self.report_facts,
            run_metadata: self.run_metadata,
            report_commit: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use bytes::Bytes;

    use super::*;

    struct NeverEvaluatorFactory;

    #[async_trait(?Send)]
    impl StaticAccuracyEvaluatorFactory for NeverEvaluatorFactory {
        async fn spawn(
            &self,
            _process: &StaticAccuracyEvaluatorProcessSpec,
        ) -> Result<Box<dyn crate::accuracy_core::AccuracyEvaluator>> {
            panic!("config lowering must retain, not invoke, the selected evaluator factory")
        }
    }

    #[derive(Default)]
    struct FixtureFetcher {
        urls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl DatasetFetcher for FixtureFetcher {
        async fn fetch(
            &self,
            url: &str,
            _cache_key: &str,
            _bearer_token: Option<&str>,
        ) -> crate::dataset::Result<Bytes> {
            self.urls.lock().unwrap().push(url.to_owned());
            if url.contains("/api/models/") {
                return Ok(Bytes::from(
                    serde_json::json!({"sha": "a".repeat(40)}).to_string(),
                ));
            }
            if url.ends_with("/tokenizer.json") {
                return Ok(Bytes::from_static(br#"{"version":"1.0"}"#));
            }
            Err(crate::dataset::DatasetError::Validation(
                "optional fixture file is absent".into(),
            ))
        }
    }

    #[test]
    fn warmup_agentic_cache_duration_recovers_authored_value() {
        // The CLI stamps `--agentic-cache-warmup-duration` onto an authored
        // phase's common; the legacy path recovers it to thread onto its
        // synthesized WARMUP phase so `dataset_build` populates
        // `AgenticReplayConfig.cache_warmup_duration_s`.
        let phases: Vec<PhaseSpec> = serde_json::from_str(
            r#"[
                {
                    "type": "concurrency",
                    "name": "warmup",
                    "exclude_from_results": true,
                    "concurrency": 1,
                    "agentic_cache_warmup_duration": 5.0
                },
                {
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "concurrency": 1
                }
            ]"#,
        )
        .unwrap();
        assert_eq!(warmup_agentic_cache_duration(&phases), Some(5.0));
    }

    #[test]
    fn warmup_agentic_cache_duration_absent_when_unset() {
        let phases: Vec<PhaseSpec> = serde_json::from_str(
            r#"[
                {
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "concurrency": 1
                }
            ]"#,
        )
        .unwrap();
        assert_eq!(warmup_agentic_cache_duration(&phases), None);
    }

    #[test]
    fn resolve_agentic_warmup_grace_prefers_explicit_then_cache_default() {
        let explicit: crate::engine::protocol::PhaseCommonSpec =
            serde_json::from_value(serde_json::json!({
                "name": "profiling",
                "exclude_from_results": false,
                "agentic_warmup_grace_period": 7.5,
                "agentic_cache_warmup_duration": 60.0,
                "grace_period": 1.0
            }))
            .unwrap();
        assert_eq!(resolve_agentic_warmup_grace(&explicit), Some(7.5));

        let cache_only: crate::engine::protocol::PhaseCommonSpec =
            serde_json::from_value(serde_json::json!({
                "name": "profiling",
                "exclude_from_results": false,
                "agentic_cache_warmup_duration": 60.0,
                "grace_period": 10.0
            }))
            .unwrap();
        // max(benchmark grace 10, min(cache 60, 300)) = 60
        assert_eq!(resolve_agentic_warmup_grace(&cache_only), Some(60.0));

        let plain: crate::engine::protocol::PhaseCommonSpec =
            serde_json::from_value(serde_json::json!({
                "name": "profiling",
                "exclude_from_results": false
            }))
            .unwrap();
        assert_eq!(resolve_agentic_warmup_grace(&plain), None);
    }

    #[test]
    fn tokenizer_policy_accepts_trust_remote_code_as_inert() {
        // The native tokenizer never executes repository code, so
        // `trust_remote_code=true` is inert rather than refused: decode must
        // succeed (a warning is emitted) so harnesses that pass the flag
        // unconditionally still run.
        let raw = RawValue::from_string(
            serde_json::json!({
                "name": "fixture/tokenizer",
                "revision": "locked",
                "trust_remote_code": true,
                "apply_chat_template": false
            })
            .to_string(),
        )
        .unwrap();
        let config = AuthoredTokenizerV2::decode(&raw).expect("trust_remote_code must be accepted");
        assert!(config.trust_remote_code);
        assert_eq!(config.name, "fixture/tokenizer");
    }

    #[test]
    fn builtin_tokenizer_resolves_with_trust_remote_code() {
        // A built-in (tiktoken) encoding must resolve identically whether or not
        // `trust_remote_code` is set; the flag must not preempt the load.
        let resolved = resolve_builtin_or_local("cl100k_base", true).expect("builtin must resolve");
        assert_eq!(resolved.as_deref(), Some("cl100k_base"));
    }

    #[test]
    fn local_path_tokenizer_resolves_with_trust_remote_code() {
        // A local filesystem tokenizer path must also resolve with the flag set.
        let dir = std::env::temp_dir();
        let resolved =
            resolve_builtin_or_local(dir.to_str().unwrap(), true).expect("local path must resolve");
        assert_eq!(resolved.as_deref(), dir.to_str());
    }

    #[test]
    fn accuracy_verbose_fails_closed_in_execution_adapter() {
        let raw = RawValue::from_string(
            serde_json::json!({
                "benchmark": "fixture",
                "python_executable": if cfg!(windows) { "C:\\python.exe" } else { "/usr/bin/python3" },
                "verbose": true
            })
            .to_string(),
        )
        .unwrap();
        assert!(
            StaticAccuracyConfigV2::decode(&raw)
                .unwrap_err()
                .to_string()
                .contains("presentation policy")
        );
    }

    #[test]
    fn static_accuracy_lowers_directly_with_selected_evaluator_factory() {
        let raw = RawValue::from_string(
            serde_json::json!({
                "benchmark": "fixture",
                "tasks": ["task-a"],
                "python_executable": if cfg!(windows) { "C:\\python.exe" } else { "/usr/bin/python3" },
                "worker_module": "fixture.worker"
            })
            .to_string(),
        )
        .unwrap();
        let factory: Arc<dyn StaticAccuracyEvaluatorFactory> = Arc::new(NeverEvaluatorFactory);

        let plan = StaticAccuracyConfigV2::decode(&raw)
            .unwrap()
            .lower(factory.clone());

        assert_eq!(plan.benchmark, "fixture");
        assert_eq!(plan.tasks.as_deref().unwrap(), ["task-a"]);
        assert_eq!(plan.process.worker_module, "fixture.worker");
        assert!(Arc::ptr_eq(&plan.evaluator_factory, &factory));
    }

    #[test]
    fn registration_adds_only_real_online_workloads() {
        let mut registry = crate::extensions::AIPerfRegistry::builtin().unwrap();
        register_online_workloads(&mut registry).unwrap();
        register_http_static_accuracy_workload(&mut registry).unwrap();
    }

    #[test]
    fn remote_tokenizer_resolves_revision_to_commit_and_caches_local_source() {
        let fetcher = Arc::new(FixtureFetcher::default());
        let cache = tempfile::tempdir().unwrap();
        let resolver =
            NativeOnlineTokenizerSourceResolver::new(fetcher.clone(), cache.path().to_path_buf());
        let resolved = resolver
            .resolve("fixture/model", "locked-revision", false)
            .unwrap();
        assert!(Path::new(&resolved).join("tokenizer.json").is_file());
        let urls = fetcher.urls.lock().unwrap();
        assert!(
            urls.iter()
                .any(|url| url.ends_with("/api/models/fixture/model/revision/locked-revision"))
        );
        assert!(urls.iter().any(|url| url.ends_with(&format!(
            "/fixture/model/resolve/{}/tokenizer.json",
            "a".repeat(40)
        ))));
    }

    /// A repository whose only tokenizer artifact is a custom Python tokenizer
    /// (no `tokenizer.json`). The metadata request resolves a commit, but the
    /// `tokenizer.json` fetch fails, mirroring a Kimi-K2-class remote-code repo.
    #[derive(Default)]
    struct RemoteCodeOnlyFetcher {
        urls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl DatasetFetcher for RemoteCodeOnlyFetcher {
        async fn fetch(
            &self,
            url: &str,
            _cache_key: &str,
            _bearer_token: Option<&str>,
        ) -> crate::dataset::Result<Bytes> {
            self.urls.lock().unwrap().push(url.to_owned());
            if url.contains("/api/models/") {
                return Ok(Bytes::from(
                    serde_json::json!({"sha": "b".repeat(40)}).to_string(),
                ));
            }
            // No `tokenizer.json` in the repository: the native tokenizer cannot
            // load a custom-code tokenizer, so this download 404s upstream.
            Err(crate::dataset::DatasetError::Validation(
                "tokenizer.json is absent (custom-code-only repository)".into(),
            ))
        }
    }

    #[test]
    fn remote_code_only_tokenizer_fails_with_actionable_error_not_silent_substitution() {
        // Invariant: a Kimi-class repository that ships only a custom Python
        // tokenizer must NOT silently resolve to a substitute (builtin/gpt2/
        // cl100k) tokenizer. It must fail loudly with guidance that names the
        // `--use-server-token-count` escape hatch. Wrong token counts are the
        // worst failure mode for a benchmark tool.
        let fetcher = Arc::new(RemoteCodeOnlyFetcher::default());
        let cache = tempfile::tempdir().unwrap();
        let resolver =
            NativeOnlineTokenizerSourceResolver::new(fetcher.clone(), cache.path().to_path_buf());
        let error = resolver
            .resolve("moonshotai/Kimi-K2", "locked-revision", true)
            .expect_err("a custom-code-only tokenizer must not resolve to a substitute");
        let message = format!("{error:#}");
        // Names the model, explains why, and points to the honest escape hatch.
        assert!(message.contains("Kimi-K2"), "message: {message}");
        assert!(
            message.contains("--use-server-token-count"),
            "message: {message}"
        );
        assert!(
            message.contains("native tokenizer") && message.contains("tokenizer.json"),
            "message: {message}"
        );
        // Never leaked a substitute encoding name as a resolved source.
        assert!(!message.contains("cl100k") && !message.contains("gpt2"));
    }

    #[test]
    fn resolved_directory_without_tokenizer_json_is_rejected() {
        // The shared guard both online resolvers call: a resolved directory that
        // has no `tokenizer.json` (a custom-code repo whose config-only files
        // downloaded successfully) is rejected before it reaches `load_tokenizer`,
        // where it would otherwise surface a bare "No such file" error.
        let directory = tempfile::tempdir().unwrap();
        let error = ensure_native_tokenizer_loadable("moonshotai/Kimi-K2", directory.path())
            .expect_err("a directory without tokenizer.json must be rejected");
        let message = format!("{error:#}");
        assert!(message.contains("Kimi-K2"));
        assert!(message.contains("--use-server-token-count"));

        // A directory that DOES contain tokenizer.json passes the guard.
        std::fs::write(directory.path().join("tokenizer.json"), b"{}").unwrap();
        ensure_native_tokenizer_loadable("moonshotai/Kimi-K2", directory.path())
            .expect("a directory with tokenizer.json is natively loadable");
    }

    #[test]
    fn resolved_directory_with_tiktoken_model_is_natively_loadable() {
        // A Kimi/Qwen-class repo ships a `tiktoken.model` BPE vocab instead of a
        // `tokenizer.json`. The guard must accept it so the native tiktoken loader
        // runs, rather than firing the actionable remote-code error.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("tiktoken.model"), b"AAAA 0\n").unwrap();
        ensure_native_tokenizer_loadable("moonshotai/Kimi-K2", directory.path())
            .expect("a directory with tiktoken.model is natively loadable");

        // `tokenizer.model` (Llama-3 / some Qwen) and `*.tiktoken` (older Qwen)
        // are equally acceptable.
        let alt = tempfile::tempdir().unwrap();
        std::fs::write(alt.path().join("tokenizer.model"), b"AAAA 0\n").unwrap();
        ensure_native_tokenizer_loadable("Qwen/Qwen", alt.path()).unwrap();
    }
}
