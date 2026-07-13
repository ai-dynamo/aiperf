// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 adapters for the native online HTTP transport.
//!
//! Authored protocol-v2 values lower directly into the protocol-neutral
//! [`NativeRunPlan`] consumed by the same coordinator as protocol v1. The
//! adapter never constructs or serializes a protocol-v1 `RunRequest`. Direct
//! `dag_jsonl` input remains an authored graph program: the common coordinator
//! passes it once to the selected runner-owned authored-input adapter, which
//! returns `GraphTracePlan`s and one frozen segment store.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aiperf_content_server::ContentServerMediaPublisher;
use aiperf_dataset::{
    DatasetFetcher, HttpDatasetFetcher, MaterializedTracePromptStorage,
    NativeSyntheticMediaGeneratorFactory, SyntheticMediaGeneratorFactory, TiktokenEncoding,
};
use aiperf_endpoints::Modality;
use aiperf_metrics::{NativeReport, ReportGraphRunInfo, ReportPairRunFacts};
use aiperf_rng::RngRoot;
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, value::RawValue};
use url::Url;

use crate::dataset_input::RunnerDatasetInputContext;
use crate::execute::{
    NativeDatasetPlan, NativeEndpointPlan, NativeGraphDatasetPlan, NativeRunPlan, NativeRunSpec,
    NativeSidecarPlan, execute_prepared_native_plan_uncommitted_with_execution_factories,
    load_tokenizer,
};
use crate::execution_factories::RunnerExecutionFactories;
use crate::graph_input::RunnerGraphInputContext;
use crate::protocol::{ArtifactSpec, PhaseSpec, TokenizerSpec};
use crate::protocol_v2::AuthoredRunSpecV2;
use crate::readiness::{PreparedOnlineReadiness, ReadinessEndpointProfile, ReadinessPlanInput};
use crate::registry::{
    GraphWorkloadConfigV2, OnlineHttpTransportConfigV2, PreparedRunOutcome,
    PreparedRunnerOperation, RunnerPairFactory, RunnerRegistryBuilder, RunnerRunContext,
    ScheduledWorkloadConfigV2, ValidatedTransportConfig, ValidatedWorkloadConfig,
};
use crate::sidecar_input::{CONTENT_SERVER_SIDECAR_ID, ContentServerSpec};

const TRANSPORT_ID: &str = "http";

/// Register online HTTP pairs whose entire authored Config-v2 surface is
/// executable without falling back to protocol v1.
///
/// Keeping registration in this module means capability publication changes
/// only when a real adapter and its execution proof are both linked.
pub fn register_http_pairs(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    let tokenizers: Arc<dyn OnlineTokenizerSourceResolver> =
        Arc::new(NativeOnlineTokenizerSourceResolver::default());
    builder.register_pair(Arc::new(OnlineHttpPairFactory::new(Arc::new(
        OnlineGraphAdapter { tokenizers },
    ))))
}

/// Register the scheduled adapter after the distribution also supplies a
/// capability-aware frontend gate or complete public-dataset and sidecar
/// preparation.
///
/// The adapter is real and subprocess-tested, but the base registry must not
/// advertise it while ordinary v1 scheduled configs could be routed to a
/// narrower v2 surface merely because the pair ID matches.
pub fn register_http_scheduled_pair(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_pair(Arc::new(OnlineHttpPairFactory::new(Arc::new(
        OnlineScheduledAdapter {
            tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
        },
    ))))
}

#[derive(Clone)]
struct OnlineHttpPairFactory {
    adapter: Arc<dyn OnlineWorkloadAdapter>,
}

impl OnlineHttpPairFactory {
    fn new(adapter: Arc<dyn OnlineWorkloadAdapter>) -> Self {
        Self { adapter }
    }
}

impl fmt::Debug for OnlineHttpPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OnlineHttpPairFactory")
            .field("workload_id", &self.adapter.workload_id())
            .finish_non_exhaustive()
    }
}

/// Direct authored-workload adapter selected by one registered online pair.
///
/// Each implementation owns strict type recovery, workload validation, and
/// its single source-to-prepared-harness transition. The generic pair and
/// registry coordinator never branch on workload IDs or inspect workload
/// configuration.
trait OnlineWorkloadAdapter: fmt::Debug + Send + Sync {
    /// Registry identity consumed by this adapter.
    fn workload_id(&self) -> &'static str;

    /// Check that the selected workload factory returned this adapter's exact
    /// validated configuration type.
    fn validate_workload(&self, workload: &dyn ValidatedWorkloadConfig) -> Result<()>;

    /// Check run-level invariants without acquisition, socket IO, or artifact
    /// creation.
    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()>;

    /// Load the selected authored source exactly once and retain its canonical
    /// prepared harness through execution.
    fn prepare(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedOnlineHarness>>;
}

impl RunnerPairFactory for OnlineHttpPairFactory {
    fn transport_id(&self) -> &'static str {
        TRANSPORT_ID
    }

    fn workload_id(&self) -> &'static str {
        self.adapter.workload_id()
    }

    fn validate_pair(
        &self,
        transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        online_transport(transport)?;
        self.adapter.validate_workload(workload)
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        transport: &dyn ValidatedTransportConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(transport, workload)?;
        self.adapter.validate_run(run, context, workload)
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _transport: Box<dyn ValidatedTransportConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        bail!(
            "{TRANSPORT_ID} + {} preparation requires the coordinator-owned RunnerRunContext",
            self.adapter.workload_id()
        )
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        transport: Box<dyn ValidatedTransportConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        online_transport(transport.as_ref())?;
        self.adapter.validate_workload(workload.as_ref())?;
        let readiness = prepare_online_readiness(run, context)?;
        let harness = self.adapter.prepare(run, context, workload)?;
        Ok(Box::new(PreparedOnlineOperation {
            workload_id: self.adapter.workload_id(),
            harness,
            product_registry: context.product_registry_handle(),
            execution_factories: context.execution_factories_handle(),
            readiness,
        }))
    }
}

/// Retained workload-specific execution harness produced by one direct adapter
/// load. Implementations own their canonical prepared state; the registry and
/// pair coordinator do not inspect it.
trait PreparedOnlineHarness: fmt::Debug {
    /// Execute through the selected transport and return an uncommitted report.
    fn execute(
        self: Box<Self>,
        product_registry: &aiperf_extensions::AiperfRegistry,
        execution_factories: &RunnerExecutionFactories,
        readiness: Box<dyn PreparedOnlineReadiness>,
    ) -> Result<(NativeReport, ReportPairRunFacts)>;
}

#[derive(Clone)]
struct OnlineScheduledAdapter {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for OnlineScheduledAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OnlineScheduledAdapter")
    }
}

#[derive(Clone)]
struct OnlineGraphAdapter {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for OnlineGraphAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OnlineGraphAdapter")
    }
}

impl OnlineWorkloadAdapter for OnlineScheduledAdapter {
    fn workload_id(&self) -> &'static str {
        "scheduled"
    }

    fn validate_workload(&self, workload: &dyn ValidatedWorkloadConfig) -> Result<()> {
        workload_config::<ScheduledWorkloadConfigV2>(workload, self.workload_id()).map(drop)
    }

    fn validate_run(
        &self,
        _run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        validate_online_run(context)?;
        let workload = workload_config::<ScheduledWorkloadConfigV2>(workload, self.workload_id())?;
        validate_authored_tokenizer(&workload.tokenizer)
    }

    fn prepare(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedOnlineHarness>> {
        let workload =
            workload_config::<ScheduledWorkloadConfigV2>(workload.as_ref(), self.workload_id())?;
        let plan = lower_scheduled(run, context, workload, self.tokenizers.as_ref())?;
        Ok(Box::new(NativePlanHarness { plan }))
    }
}

impl OnlineWorkloadAdapter for OnlineGraphAdapter {
    fn workload_id(&self) -> &'static str {
        "graph"
    }

    fn validate_workload(&self, workload: &dyn ValidatedWorkloadConfig) -> Result<()> {
        workload_config::<GraphWorkloadConfigV2>(workload, self.workload_id()).map(drop)
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        validate_online_run(context)?;
        ensure!(
            run.sidecars.gpu_telemetry.is_none()
                && run.sidecars.network_latency.is_none()
                && run.sidecars.server_metrics.is_none()
                && run.sidecars.live_streaming.is_none(),
            "protocol-v2 graph execution supports only the content-server sidecar"
        );
        ensure!(
            run.models.items.len() == 1,
            "online graph execution requires exactly one default model"
        );
        let workload = workload_config::<GraphWorkloadConfigV2>(workload, self.workload_id())?;
        validate_authored_tokenizer(&workload.tokenizer)?;
        context.graph_inputs().validate_identity(&workload.dataset)
    }

    fn prepare(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedOnlineHarness>> {
        let workload =
            workload_config::<GraphWorkloadConfigV2>(workload.as_ref(), self.workload_id())?;
        let plan = lower_graph(run, context, workload, self.tokenizers.as_ref())?;
        Ok(Box::new(NativePlanHarness { plan }))
    }
}

fn online_transport(config: &dyn ValidatedTransportConfig) -> Result<&OnlineHttpTransportConfigV2> {
    ValidatedTransportConfig::as_any(config)
        .downcast_ref::<OnlineHttpTransportConfigV2>()
        .ok_or_else(|| anyhow!("online HTTP pair received a different transport config type"))
}

fn workload_config<'a, T: 'static>(
    config: &'a dyn ValidatedWorkloadConfig,
    workload_id: &str,
) -> Result<&'a T> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("online {workload_id} pair received a different config type"))
}

fn validate_online_run(context: &RunnerRunContext) -> Result<()> {
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
        if profile.config.wait_for_model_timeout > 0.0 {
            ensure!(
                profile.config.wait_for_model_mode == "models",
                "endpoint profile {profile_id:?} readiness mode {:?} is not executable yet; only endpoint-owned models/health readiness is implemented",
                profile.config.wait_for_model_mode
            );
        }
    }
    Ok(())
}

pub(crate) fn prepare_online_readiness(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
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
        ensure!(
            !trust_remote_code,
            "native tokenizers never execute repository code; set tokenizer.trust_remote_code=false"
        );
        let path = Path::new(name);
        if path.is_dir() || path.is_file() || name.parse::<TiktokenEncoding>().is_ok() {
            return Ok(name.to_owned());
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating tokenizer preparation runtime")?;
        let local = tokio::task::LocalSet::new();
        let resolved = local.block_on(&runtime, self.resolve_remote(name, revision))?;
        resolved
            .to_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("resolved tokenizer cache path is not valid UTF-8"))
    }
}

fn validate_repository_id(value: &str) -> Result<()> {
    let segments = value.split('/').collect::<Vec<_>>();
    ensure!(
        segments.len() >= 2
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
}

/// Strictly validate one authored tokenizer policy without resolving its
/// source. Pair adapters use this during side-effect-free validation and call
/// [`lower_authored_tokenizer`] exactly once during preparation.
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

pub(crate) fn lower_scheduled(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
    workload: &ScheduledWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunPlan> {
    let tokenizer = lower_authored_tokenizer(&workload.tokenizer, tokenizers)?;
    let tokenizer_impl = load_tokenizer(Some(&tokenizer.name))?;
    let profile = context.default_endpoint_profile()?;
    let prepared_endpoint = context
        .product_registry()
        .endpoints()
        .prepare(&profile.endpoint_id, profile.config.clone())
        .context("preparing the default endpoint for linear dataset composition")?;
    let rankings = prepared_endpoint
        .descriptor()
        .output_modalities
        .contains(&Modality::Rankings);
    let endpoint_descriptor = prepared_endpoint.descriptor();
    let prepare_context = RunnerDatasetInputContext {
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
    )
}

fn media_generator_factory(
    context: &RunnerRunContext,
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
    context: &RunnerRunContext,
    workload: &GraphWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunPlan> {
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
                &RunnerGraphInputContext {
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
    };
    build_common_plan(
        run,
        workload.worker_count,
        NativeDatasetPlan::Graph(Box::new(dataset)),
        tokenizer,
        &workload.phases,
        NativeEndpointPlan::Prepared(context.endpoint_profiles_handle()),
        NativeSidecarPlan::Prepared(context.sidecar_inputs_handle()),
    )
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
) -> Result<NativeRunPlan> {
    Ok(NativeRunPlan {
        run: NativeRunSpec {
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
                raw_path: run.artifacts.raw_path.clone(),
                outputs_path: run.artifacts.outputs_path.clone(),
                trace: run.artifacts.trace,
            },
            sidecars,
            user_files: run.artifacts.user_files.clone(),
        },
    })
}

fn validate_graph_endpoint_profile_references(
    bundle: &aiperf_graph::input::GraphInputBundle,
    context: &RunnerRunContext,
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

struct NativePlanHarness {
    plan: NativeRunPlan,
}

impl fmt::Debug for NativePlanHarness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativePlanHarness")
            .field("benchmark_id", &self.plan.run.benchmark_id)
            .finish_non_exhaustive()
    }
}

impl PreparedOnlineHarness for NativePlanHarness {
    fn execute(
        self: Box<Self>,
        product_registry: &aiperf_extensions::AiperfRegistry,
        execution_factories: &RunnerExecutionFactories,
        readiness: Box<dyn PreparedOnlineReadiness>,
    ) -> Result<(NativeReport, ReportPairRunFacts)> {
        let report_facts = native_plan_report_facts(&self.plan)?;
        let native_report = execute_prepared_native_plan_uncommitted_with_execution_factories(
            self.plan,
            execution_factories,
            product_registry,
            readiness,
        )?;
        Ok((native_report, report_facts))
    }
}

fn native_plan_report_facts(plan: &NativeRunPlan) -> Result<ReportPairRunFacts> {
    let NativeDatasetPlan::Graph(graph) = &plan.run.dataset else {
        return Ok(ReportPairRunFacts::new());
    };
    let graph = ReportGraphRunInfo::new(
        graph.input.metadata.format.clone(),
        graph.input.metadata.root_count,
        graph.input.metadata.node_count,
        plan.run.workers,
        plan.run.phases.len(),
    )?;
    Ok(ReportPairRunFacts::new().with_graph(graph))
}

struct PreparedOnlineOperation {
    workload_id: &'static str,
    harness: Box<dyn PreparedOnlineHarness>,
    product_registry: Arc<aiperf_extensions::AiperfRegistry>,
    execution_factories: RunnerExecutionFactories,
    readiness: Box<dyn PreparedOnlineReadiness>,
}

impl fmt::Debug for PreparedOnlineOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedOnlineOperation")
            .field("workload_id", &self.workload_id)
            .field("harness", &self.harness)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedOnlineOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let (native_report, report_facts) = self.harness.execute(
            self.product_registry.as_ref(),
            &self.execution_factories,
            self.readiness,
        )?;
        Ok(PreparedRunOutcome {
            native_report,
            report_facts,
            provenance: BTreeMap::new(),
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
        ) -> aiperf_dataset::Result<Bytes> {
            self.urls.lock().unwrap().push(url.to_owned());
            if url.contains("/api/models/") {
                return Ok(Bytes::from(
                    serde_json::json!({"sha": "a".repeat(40)}).to_string(),
                ));
            }
            if url.ends_with("/tokenizer.json") {
                return Ok(Bytes::from_static(br#"{"version":"1.0"}"#));
            }
            Err(aiperf_dataset::DatasetError::Validation(
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
    fn registration_adds_only_real_online_pairs() {
        let mut builder = RunnerRegistryBuilder::new();
        // The builder intentionally rejects dangling pairs at freeze time;
        // registration itself proves the function has no hidden mode switch.
        register_http_pairs(&mut builder).unwrap();
        register_http_scheduled_pair(&mut builder).unwrap();
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
