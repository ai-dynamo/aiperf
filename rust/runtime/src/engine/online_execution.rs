// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 executable workloads for the native online transports.
//!
//! The scheduled, graph, and static-accuracy [`WorkloadFactory`]
//! implementations own authored-source lowering into the protocol-neutral
//! [`NativeRunSpec`] and resolve the transport's turn-placement/graph-placement
//! factory and readiness policy from the validated transport (http/grpc) by
//! type — there is no per-transport pair object, and dynosim transports are
//! prepared through their own module. Direct `dag_jsonl` input remains an
//! authored graph program: the common coordinator passes it once to the selected
//! runner-owned authored-input adapter, which returns `GraphTracePlan`s and one
//! frozen segment store.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::content_server::ContentServerMediaPublisher;
use crate::dataset::{
    DatasetFetcher, HttpDatasetFetcher, MaterializedTracePromptStorage,
    NativeSyntheticMediaGeneratorFactory, SyntheticMediaGeneratorFactory, TiktokenEncoding,
    download_hugging_face_tokenizer,
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
    PreparedRunnerOperation, RunContext, WorkloadDescriptor, WorkloadFactory,
    SCHEDULED_WORKLOAD_DESCRIPTOR, STATIC_ACCURACY_WORKLOAD_DESCRIPTOR, ScheduledWorkloadConfigV2,
    StaticAccuracyWorkloadConfigV2, ValidatedTransportConfig, ValidatedWorkloadConfig,
    WorkloadRequirements, inference_workload_requirements, strict_decode, validate_common_workload,
};
use crate::engine::turn_execution::RequestExecutorFactory;

/// Native execution binding for the built-in `http` transport.
///
/// HTTP is the base transport: it drives the shared hyper turn-placement factory
/// (injected through [`ExecutionFactories`], so a remote HTTP placement can
/// replace it) and is the only native transport that performs a model-readiness
/// probe before profiling.
///
/// [`ExecutionFactories`]: crate::engine::execution_factories::ExecutionFactories
#[derive(Clone)]
pub struct HttpNativeExecution {
    executor: Arc<dyn RequestExecutorFactory>,
}

impl HttpNativeExecution {
    /// Bind the HTTP turn-placement factory resolved from the process context.
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

    fn build_graph_dispatcher(
        &self,
        clock: Rc<dyn crate::clock::Clock>,
        run_origin_ns: i64,
        urls: &[String],
        model: &str,
        transport_config: crate::transport::http::TransportSinkConfig,
        endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
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

    fn provenance(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }
}

/// Resolve the native execution binding for a validated transport by resolving
/// its registered factory by id and asking it for the binding.
///
/// Returns `Ok(None)` for a transport whose execution mode is not the
/// `RequestExecutor` seam (dynosim's virtual-clock co-simulation), which the
/// caller prepares through the offline module instead. This is the one place the
/// native path consults the transport registry; there is no per-transport
/// `match` in any workload.
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

/// Register the built-in executable workloads (`scheduled`, `graph`).
///
/// Each is one [`WorkloadFactory`] that owns its authored-source lowering
/// and resolves the transport's turn-placement/graph-placement factory from the
/// validated transport by id/type. Any registered transport (http, grpc,
/// dynosim) can drive them; there is no per-transport pair object and no
/// compatibility predicate.
pub fn register_online_workloads(registry: &mut crate::extensions::AIPerfRegistry) -> Result<()> {
    let tokenizers: Arc<dyn OnlineTokenizerSourceResolver> =
        Arc::new(HfHubOnlineTokenizerSourceResolver::default());
    registry.register_workload(Arc::new(ScheduledWorkloadFactoryV2 {
        tokenizers: tokenizers.clone(),
    }))?;
    registry.register_workload(Arc::new(GraphWorkloadFactoryV2 { tokenizers }))?;
    Ok(())
}

/// Register the static-accuracy workload (HTTP only) after sidecar parity or an
/// exact frontend gate is present in the same distribution.
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
///
/// Both factories are retained by the workload factory and used exactly once
/// when an authored run becomes its prepared operation. This is the compile-time
/// extension point for non-Hugging-Face tokenizer stores or non-local evaluator
/// processes.
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

/// Built-in scheduled workload: runs over any native (http/grpc) or dynosim
/// transport. Transport-specific endpoint validation, turn-placement factory,
/// and readiness policy are resolved from the validated transport by id/type.
struct ScheduledWorkloadFactoryV2 {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for ScheduledWorkloadFactoryV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ScheduledWorkloadFactoryV2")
    }
}

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
            None => {
                #[cfg(feature = "dynosim")]
                return crate::engine::offline_execution::dynosim_scheduled_validate_run(
                    run, context, transport, workload,
                );
                #[cfg(not(feature = "dynosim"))]
                {
                    let _ = run;
                    anyhow::bail!(
                        "transport {transport_id:?} does not support the scheduled workload"
                    );
                }
            }
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
                // A `dry_run` transport with `clock: sim` runs the deterministic
                // virtual-time path: the SAME native execution driven under
                // `drive_sim` on a `SimClock`, so `RunCapture`/per-record/report
                // all flow through the normal path — only the clock differs.
                if crate::engine::dry_run::sim_params_for(transport.as_ref()).is_some() {
                    return crate::engine::execute::prepare_dry_run_sim_scheduled(
                        plan,
                        binding.executor_factory(),
                        context,
                    );
                }
                prepare_native_operation(run, context, plan, binding)
            }
            None => {
                #[cfg(feature = "dynosim")]
                return crate::engine::offline_execution::prepare_dynosim_scheduled(
                    run,
                    context,
                    transport,
                    workload,
                    self.tokenizers.clone(),
                );
                #[cfg(not(feature = "dynosim"))]
                {
                    let _ = (run, context, workload);
                    anyhow::bail!(
                        "transport {transport_id:?} does not support the scheduled workload"
                    );
                }
            }
        }
    }
}

/// Built-in direct Graph-IR workload: runs over any native (http/grpc) or
/// dynosim transport. The resolved transport selects the graph placement's
/// graph dispatcher (`build_graph_dispatcher`) plus readiness policy.
struct GraphWorkloadFactoryV2 {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for GraphWorkloadFactoryV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GraphWorkloadFactoryV2")
    }
}

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
                // Every native transport binding builds a graph dispatcher
                // (`build_graph_dispatcher`), so graph support is not a separate
                // gate. Run-level validation is dispatched to the binding.
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
                context
                    .graph_inputs()
                    .validate_identity(&workload.dataset)?;
                Ok(())
            }
            None => {
                #[cfg(feature = "dynosim")]
                return crate::engine::offline_execution::dynosim_graph_validate_run(
                    run, context, transport, workload,
                );
                #[cfg(not(feature = "dynosim"))]
                {
                    let _ = run;
                    anyhow::bail!("transport {transport_id:?} does not support the graph workload");
                }
            }
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
                let plan = lower_graph(
                    run,
                    context,
                    workload,
                    self.tokenizers.as_ref(),
                    binding.clone(),
                )?;
                // A `dry_run` transport with `clock: sim` runs the graph replay in
                // virtual time (SimClock + drive_sim), so a 30-minute agentic
                // trace finishes at ~startup wall-speed — the same normal
                // execution, only the clock differs.
                if crate::engine::dry_run::sim_params_for(transport.as_ref()).is_some() {
                    return crate::engine::execute::prepare_dry_run_sim_scheduled(
                        plan,
                        binding.executor_factory(),
                        context,
                    );
                }
                prepare_native_operation(run, context, plan, binding)
            }
            None => {
                #[cfg(feature = "dynosim")]
                return crate::engine::offline_execution::prepare_dynosim_graph(
                    run,
                    context,
                    transport,
                    workload,
                    self.tokenizers.clone(),
                );
                #[cfg(not(feature = "dynosim"))]
                {
                    let _ = (run, context, workload);
                    anyhow::bail!("transport {transport_id:?} does not support the graph workload");
                }
            }
        }
    }
}

/// Built-in static-accuracy workload (HTTP only).
struct StaticAccuracyWorkloadFactoryV2 {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
    evaluator_factory: Arc<dyn StaticAccuracyEvaluatorFactory>,
}

impl fmt::Debug for StaticAccuracyWorkloadFactoryV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StaticAccuracyWorkloadFactoryV2")
    }
}

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
            "static accuracy execution currently requires exactly one model"
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

/// Build the prepared native operation for one lowered plan, driving the
/// transport's turn-placement factory, readiness policy, and provenance entirely
/// through its [`NativeTransportExecution`] binding.
///
/// There is no per-transport branch here: the binding is the transport, so
/// adding a native transport changes nothing in this function.
fn prepare_native_operation(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    plan: NativeRunSpec,
    binding: Arc<dyn NativeTransportExecution>,
) -> Result<Box<dyn PreparedRunnerOperation>> {
    let report_facts = native_plan_report_facts(&plan)?;
    let request_executor = binding.executor_factory();
    let readiness = if binding.readiness_enabled() {
        Some(prepare_online_readiness(run, context)?)
    } else {
        None
    };
    let provenance = binding.provenance();
    Ok(Box::new(PreparedNativeOperation {
        plan,
        request_executor,
        execution_factories: context.execution_factories_handle(),
        product_registry: context.product_registry_handle(),
        readiness,
        report_facts,
        provenance,
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
}

fn default_revision() -> String {
    "main".into()
}

/// Preparation-time tokenizer acquisition seam for protocol-v2 online pairs.
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

/// Native Hugging Face/local/built-in tokenizer source resolver.
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
    /// Construct with an injected byte fetcher and cache root.
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

        let tokenizer_url =
            hugging_face_model_url(&self.hugging_face_endpoint, name, Some(("resolve", commit)))?
                + "/tokenizer.json";
        let tokenizer = self
            .fetcher
            .fetch(
                &tokenizer_url,
                &format!("hf-tokenizer-file:{name}:{commit}:tokenizer.json"),
                token.as_deref(),
            )
            .await
            .with_context(|| format!("downloading tokenizer.json for {name:?}@{commit}"))?;
        persist_tokenizer_file(&tokenizer_path, tokenizer.as_ref())?;

        // `tokenizer_config.json` enriches BOS/EOS and chat-template policy but
        // is not universal. A missing optional file must not invalidate a
        // complete tokenizer.json; malformed present files are rejected later
        // by `HuggingFaceTokenizer::from_directory`.
        let config_url =
            hugging_face_model_url(&self.hugging_face_endpoint, name, Some(("resolve", commit)))?
                + "/tokenizer_config.json";
        if let Ok(config) = self
            .fetcher
            .fetch(
                &config_url,
                &format!("hf-tokenizer-file:{name}:{commit}:tokenizer_config.json"),
                token.as_deref(),
            )
            .await
        {
            persist_tokenizer_file(
                &resolved_directory.join("tokenizer_config.json"),
                config.as_ref(),
            )?;
        }
        Ok(resolved_directory)
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
        tokenizer_directory_to_string(&resolved)
    }
}

/// Built-in encodings and local filesystem tokenizers resolve without any IO.
///
/// Returns `Ok(Some(source))` when `name` is a tiktoken encoding or an existing
/// local path, and `Ok(None)` when a remote Hugging Face download is required.
/// Shared by every resolver so the no-network short-circuit and the
/// `trust_remote_code` refusal stay identical regardless of download backend.
fn resolve_builtin_or_local(name: &str, trust_remote_code: bool) -> Result<Option<String>> {
    ensure!(
        !trust_remote_code,
        "native tokenizers never execute repository code; set tokenizer.trust_remote_code=false"
    );
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
/// revision falls back to the exact-commit HTTP resolver. Selected through the
/// [`OnlineTokenizerSourceResolver`] seam so a distribution with an internal
/// artifact store can still swap the whole mechanism.
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
        // hf-hub resolves the branch tip; an authored pinned commit/branch needs
        // the exact-commit HTTP path, which the fallback still performs.
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
        tokenizer_directory_to_string(&directory)
    }
}

fn validate_repository_id(value: &str) -> Result<()> {
    // Hugging Face repository IDs are either `namespace/name` or a bare
    // canonical/legacy model id with no namespace (`gpt2`, `bert-base-uncased`,
    // `t5-small`). Both are valid `hf-hub` repos, so accept one or more
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
                // File URLs do not use the API prefix.
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
        ensure!(
            !config.trust_remote_code,
            "native tokenizers never execute repository code; set tokenizer.trust_remote_code=false"
        );
        Ok(config)
    }

    fn lower(&self, resolver: &dyn OnlineTokenizerSourceResolver) -> Result<TokenizerSpec> {
        Ok(TokenizerSpec {
            name: resolver.resolve(&self.name, &self.revision, self.trust_remote_code)?,
            apply_chat_template: self.apply_chat_template,
        })
    }

    /// Lower to a harmless built-in tokenizer without resolving the authored
    /// source. Selected for endpoints that neither tokenize their input nor
    /// produce output tokens: the authored `tokenizer.name` is the primary
    /// model repository regardless of endpoint (Python projects it verbatim),
    /// so resolving it would download or gate-fail a tokenizer the run never
    /// consults. A chat template would tokenize input, so it is dropped here.
    fn lower_builtin(&self) -> TokenizerSpec {
        TokenizerSpec {
            name: "builtin".into(),
            apply_chat_template: false,
        }
    }
}

/// Whether the selected endpoint descriptor requires a real tokenizer.
///
/// AIPerf only tokenizes when the endpoint tokenizes its input or produces
/// output tokens; `base_metrics_processor` filters token metrics by the same
/// two flags, so a descriptor with both false has no token metrics and needs no
/// tokenizer. Descriptor-driven so any future non-tokenizing endpoint is
/// covered without enumerating endpoint ids.
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

/// Strictly validate one authored tokenizer policy without resolving its
/// source. Workload factories use this during side-effect-free validation and
/// call [`lower_authored_tokenizer`] exactly once during preparation.
pub(crate) fn validate_authored_tokenizer(raw: &RawValue) -> Result<()> {
    AuthoredTokenizerV2::decode(raw).map(|_| ())
}

/// Decode and resolve one authored tokenizer through an injected source
/// resolver. Offline and future remote transports reuse this preparation without
/// inheriting the online HTTP workload adapter.
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
    // Prepare the endpoint before resolving the tokenizer: its descriptor
    // decides whether a real tokenizer is needed at all. Non-tokenizing
    // endpoints (e.g. image_retrieval, riva_asr) must not trigger a remote /
    // gated-model tokenizer download for a tokenizer the run never consults.
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
        // Scheduled resolves its transport through the injected
        // RequestExecutorFactory; the graph binding is unused on the linear path.
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
        // The Python feature starts a temporary server when CONTENT_DIR is
        // empty but activates file writing only when both values are present.
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
        // The resolved transport builds the graph dispatcher (`build_graph_dispatcher`).
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
        // Static accuracy has no failure knob today; it inherits the scheduled
        // default (resilient) via `None`.
        None,
        // Static accuracy runs over HTTP and does not use the graph runtime.
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
        },
        sidecars,
        user_files: run.artifacts.user_files.clone(),
        failure_policy,
        native_otel_enabled: run.export.otel.enabled && run.export.otel.endpoint.is_some(),
        transport,
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

/// Prepared native (http/grpc, scheduled/graph/static-accuracy) operation.
///
/// The workload resolved the transport's turn-placement factory and readiness
/// policy at prepare time; execution is transport-blind from here — the same
/// entry point drives scheduled turns (through the selected request executor)
/// and graph traces (through the shared graph placement with the plan's
/// `transport_kind` arm).
struct PreparedNativeOperation {
    plan: NativeRunSpec,
    request_executor: Arc<dyn RequestExecutorFactory>,
    execution_factories: ExecutionFactories,
    product_registry: Arc<crate::extensions::AIPerfRegistry>,
    readiness: Option<Box<dyn PreparedOnlineReadiness>>,
    report_facts: ReportPairRunFacts,
    provenance: BTreeMap<String, String>,
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
            provenance: self.provenance,
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
    fn tokenizer_policy_rejects_repository_code_execution() {
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
        assert!(
            AuthoredTokenizerV2::decode(&raw)
                .unwrap_err()
                .to_string()
                .contains("never execute")
        );
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
        // Registration proves the executable scheduled/graph/static-accuracy
        // workloads compose into the one unified registry without a per-transport
        // pair object; any workload runs over any transport.
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
}
