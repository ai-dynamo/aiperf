// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 adapters for the native online HTTP backend.
//!
//! Authored protocol-v2 values lower directly into the protocol-neutral
//! [`NativeRunPlan`] consumed by the same coordinator as protocol v1. The
//! adapter never constructs or serializes a protocol-v1 `RunRequest`. Direct
//! `dag_jsonl` input remains an authored graph program: the common coordinator
//! passes it once to the selected [`aiperf_graph::input::GraphInputAdapter`],
//! which returns `GraphTracePlan`s and one frozen segment store.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aiperf_dataset::{
    DatasetFetcher, DatasetSource, HttpDatasetFetcher, LoadConfig, TiktokenEncoding,
};
use aiperf_endpoints::EndpointType;
use aiperf_graph::input::GraphInputAdapterRegistry;
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};
use url::Url;

use crate::execute::{
    NativeDatasetPlan, NativeEndpointPlan, NativeGraphDatasetPlan, NativeGraphInputPlan,
    NativePreparedEndpointPlan, NativePreparedEndpointProfile, NativeRunPlan, NativeRunSpec,
    distribution, execute_native_plan_with_factories, load_tokenizer,
};
use crate::graph_execution::NativeRunnerGraphPlacementFactory;
use crate::protocol::{
    AccuracySpec, ArtifactSpec, DatasetSpec, DistributionSpec, EndpointSpec, FixedDistributionSpec,
    PhaseSpec, SyntheticDatasetSpec, TokenizerSpec,
};
use crate::protocol_v2::AuthoredRunSpecV2;
use crate::registry::{
    GraphWorkloadConfigV2, OnlineHttpBackendConfigV2, PreparedRunOutcome, PreparedRunnerOperation,
    RunnerPairFactory, RunnerRegistryBuilder, RunnerRunContext, ScheduledWorkloadConfigV2,
    StaticAccuracyWorkloadConfigV2, ValidatedBackendConfig, ValidatedEndpointProfileV2,
    ValidatedWorkloadConfig,
};
use crate::turn_execution::NativeHttpExecutionBackendFactory;

const BACKEND_ID: &str = "online_http";

/// Register online HTTP pairs whose entire authored Config-v2 surface is
/// executable without falling back to protocol v1.
///
/// Keeping registration in this module means capability publication changes
/// only when a real adapter and its execution proof are both linked.
pub fn register_online_http_pairs(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    let tokenizers: Arc<dyn OnlineTokenizerSourceResolver> =
        Arc::new(NativeOnlineTokenizerSourceResolver::default());
    builder.register_pair(Arc::new(OnlineGraphPairFactory { tokenizers }))
}

/// Register static accuracy after sidecar parity or an exact frontend gate is
/// present in the same distribution.
pub fn register_online_http_static_accuracy_pair(
    builder: &mut RunnerRegistryBuilder,
) -> Result<()> {
    builder.register_pair(Arc::new(OnlineStaticAccuracyPairFactory {
        tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
    }))
}

/// Register the scheduled adapter after the distribution also supplies a
/// capability-aware frontend gate or complete public-dataset and sidecar
/// preparation.
///
/// The adapter is real and subprocess-tested, but the base registry must not
/// advertise it while ordinary v1 scheduled configs could be routed to a
/// narrower v2 surface merely because the pair ID matches.
pub fn register_online_http_scheduled_pair(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_pair(Arc::new(OnlineScheduledPairFactory {
        tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
    }))
}

#[derive(Clone)]
struct OnlineScheduledPairFactory {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for OnlineScheduledPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OnlineScheduledPairFactory")
    }
}

#[derive(Clone)]
struct OnlineGraphPairFactory {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for OnlineGraphPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OnlineGraphPairFactory")
    }
}

#[derive(Clone)]
struct OnlineStaticAccuracyPairFactory {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for OnlineStaticAccuracyPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OnlineStaticAccuracyPairFactory")
    }
}

macro_rules! impl_online_pair {
    ($factory:ty, $workload_id:literal, $config:ty, $kind:expr) => {
        impl RunnerPairFactory for $factory {
            fn backend_id(&self) -> &'static str {
                BACKEND_ID
            }

            fn workload_id(&self) -> &'static str {
                $workload_id
            }

            fn validate_pair(
                &self,
                backend: &dyn ValidatedBackendConfig,
                workload: &dyn ValidatedWorkloadConfig,
            ) -> Result<()> {
                online_backend(backend)?;
                workload_config::<$config>(workload, $workload_id)?;
                Ok(())
            }

            fn validate_run(
                &self,
                run: &AuthoredRunSpecV2,
                context: &RunnerRunContext,
                backend: &dyn ValidatedBackendConfig,
                workload: &dyn ValidatedWorkloadConfig,
            ) -> Result<()> {
                self.validate_pair(backend, workload)?;
                validate_online_run(run, context, $kind)
            }

            fn prepare(
                &self,
                _run: &AuthoredRunSpecV2,
                _backend: Box<dyn ValidatedBackendConfig>,
                _workload: Box<dyn ValidatedWorkloadConfig>,
            ) -> Result<Box<dyn PreparedRunnerOperation>> {
                bail!(
                    "{BACKEND_ID} + {} preparation requires the coordinator-owned RunnerRunContext",
                    $workload_id
                )
            }

            fn prepare_with_context(
                &self,
                run: &AuthoredRunSpecV2,
                context: &RunnerRunContext,
                backend: Box<dyn ValidatedBackendConfig>,
                workload: Box<dyn ValidatedWorkloadConfig>,
            ) -> Result<Box<dyn PreparedRunnerOperation>> {
                online_backend(backend.as_ref())?;
                let workload = workload_config::<$config>(workload.as_ref(), $workload_id)?;
                let plan = ($kind.lower)(run, context, workload, self.tokenizers.as_ref())?;
                Ok(Box::new(PreparedOnlineOperation {
                    workload_id: $workload_id,
                    plan,
                    product_registry: context.product_registry_handle(),
                }))
            }
        }
    };
}

#[derive(Clone, Copy)]
struct OnlineWorkloadKind<T> {
    lower: fn(
        &AuthoredRunSpecV2,
        &RunnerRunContext,
        &T,
        &dyn OnlineTokenizerSourceResolver,
    ) -> Result<NativeRunPlan>,
    graph: bool,
    accuracy: bool,
}

impl<T> fmt::Debug for OnlineWorkloadKind<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OnlineWorkloadKind")
            .field("graph", &self.graph)
            .field("accuracy", &self.accuracy)
            .finish_non_exhaustive()
    }
}

const SCHEDULED_KIND: OnlineWorkloadKind<ScheduledWorkloadConfigV2> = OnlineWorkloadKind {
    lower: lower_scheduled,
    graph: false,
    accuracy: false,
};
const GRAPH_KIND: OnlineWorkloadKind<GraphWorkloadConfigV2> = OnlineWorkloadKind {
    lower: lower_graph,
    graph: true,
    accuracy: false,
};
const STATIC_ACCURACY_KIND: OnlineWorkloadKind<StaticAccuracyWorkloadConfigV2> =
    OnlineWorkloadKind {
        lower: lower_static_accuracy,
        graph: false,
        accuracy: true,
    };

impl_online_pair!(
    OnlineScheduledPairFactory,
    "scheduled",
    ScheduledWorkloadConfigV2,
    SCHEDULED_KIND
);
impl_online_pair!(
    OnlineGraphPairFactory,
    "graph",
    GraphWorkloadConfigV2,
    GRAPH_KIND
);
impl_online_pair!(
    OnlineStaticAccuracyPairFactory,
    "static_accuracy",
    StaticAccuracyWorkloadConfigV2,
    STATIC_ACCURACY_KIND
);

fn online_backend(config: &dyn ValidatedBackendConfig) -> Result<&OnlineHttpBackendConfigV2> {
    ValidatedBackendConfig::as_any(config)
        .downcast_ref::<OnlineHttpBackendConfigV2>()
        .ok_or_else(|| anyhow!("online HTTP pair received a different backend config type"))
}

fn workload_config<'a, T: 'static>(
    config: &'a dyn ValidatedWorkloadConfig,
    workload_id: &str,
) -> Result<&'a T> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("online {workload_id} pair received a different config type"))
}

fn validate_online_run<T>(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
    kind: OnlineWorkloadKind<T>,
) -> Result<()> {
    context.default_endpoint_profile()?;
    for (profile_id, profile) in context.endpoint_profiles() {
        ensure!(
            !profile.config.urls.is_empty(),
            "endpoint profile {profile_id:?} has no URL"
        );
        ensure!(
            profile.config.wait_for_model_timeout <= 0.0,
            "protocol-v2 online execution does not yet implement endpoint readiness probes; endpoint profile {profile_id:?} enables one"
        );
    }
    ensure!(
        run.artifacts.user_files.is_empty(),
        "protocol-v2 online execution does not yet materialize artifacts.user_files"
    );
    ensure!(
        run.sidecars.gpu_telemetry.is_none()
            && run.sidecars.network_latency.is_none()
            && run.sidecars.server_metrics.is_none()
            && run.sidecars.live_streaming.is_none(),
        "protocol-v2 online execution has no registered sidecar adapter"
    );
    if kind.graph {
        ensure!(
            run.models.items.len() == 1,
            "online graph execution requires exactly one default model"
        );
    }
    if kind.accuracy {
        ensure!(
            run.models.items.len() == 1,
            "static accuracy execution currently requires exactly one model"
        );
    }
    Ok(())
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

/// Decode and resolve one authored tokenizer through an injected source
/// resolver. Offline and future remote backends reuse this preparation without
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

    fn lower(self) -> AccuracySpec {
        AccuracySpec {
            benchmark: self.benchmark,
            tasks: self.tasks,
            n_shots: self.n_shots,
            enable_cot: self.enable_cot,
            grader: self.grader,
            system_prompt: self.system_prompt,
            python_executable: self.python_executable,
            worker_module: self.worker_module,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectGraphDatasetV2 {
    #[serde(rename = "type")]
    source_type: String,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    records: Option<Value>,
    format: String,
    #[serde(default = "default_sequential")]
    sampling: String,
    #[serde(default)]
    synthesis: Option<Value>,
    #[serde(default)]
    entries: Option<usize>,
    #[serde(default)]
    random_seed: Option<u64>,
    #[serde(default)]
    osl: Option<DistributionSpec>,
    #[serde(default)]
    options: Map<String, Value>,
}

fn default_sequential() -> String {
    "sequential".into()
}

impl DirectGraphDatasetV2 {
    fn decode(raw: &RawValue) -> Result<Self> {
        let config: Self =
            serde_json::from_str(raw.get()).context("decoding direct protocol-v2 graph input")?;
        ensure!(
            config.source_type == "file",
            "online graph execution requires dataset.type=\"file\""
        );
        ensure!(
            config.format == "dag_jsonl",
            "online graph execution requires dataset.format=\"dag_jsonl\""
        );
        ensure!(
            config.path.is_some() ^ config.records.is_some(),
            "direct graph input requires exactly one of path or records"
        );
        ensure!(
            config.sampling.eq_ignore_ascii_case("sequential"),
            "direct Graph-IR input currently requires sequential root selection"
        );
        ensure!(
            config.synthesis.is_none(),
            "direct Graph-IR input does not accept linear trace synthesis"
        );
        ensure!(
            config.entries != Some(0),
            "direct graph root limit must be positive when configured"
        );
        for name in config.options.keys() {
            ensure!(
                name == "inter_turn_delay_cap_seconds",
                "dag_jsonl graph input does not accept option {name:?}"
            );
        }
        if let Some(delay) = config
            .options
            .get("inter_turn_delay_cap_seconds")
            .and_then(Value::as_f64)
        {
            ensure!(
                delay.is_finite() && delay >= 0.0,
                "inter_turn_delay_cap_seconds must be finite and non-negative"
            );
        }
        Ok(config)
    }

    fn lower(self) -> Result<NativeGraphDatasetPlan> {
        let source = match (self.path, self.records) {
            (Some(path), None) => DatasetSource::Path(path),
            (None, Some(records)) => DatasetSource::Inline(records),
            _ => unreachable!("source exclusivity validated"),
        };
        let mut load = LoadConfig::new(source);
        load.options = self.options;
        let default_output_tokens = self
            .osl
            .as_ref()
            .map(distribution)
            .transpose()?
            .map(|value| value.expected_value().ceil())
            .unwrap_or(1.0);
        ensure!(
            default_output_tokens.is_finite()
                && default_output_tokens > 0.0
                && default_output_tokens <= usize::MAX as f64,
            "graph dataset.osl expected value is outside the native usize range"
        );
        Ok(NativeGraphDatasetPlan {
            input: NativeGraphInputPlan::Authored {
                adapter_name: self.format,
                input: Box::new(aiperf_graph::input::GraphInputConfig {
                    load,
                    root_limit: self.entries,
                }),
            },
            random_seed: self.random_seed,
            default_output_tokens: default_output_tokens as usize,
        })
    }
}

fn lower_scheduled(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
    workload: &ScheduledWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunPlan> {
    let dataset = serde_json::from_str::<DatasetSpec>(workload.dataset.get())
        .context("decoding protocol-v2 linear dataset plan")?;
    ensure!(
        !matches!(&dataset, DatasetSpec::File(spec) if spec.format == "dag_jsonl")
            && !matches!(&dataset, DatasetSpec::Public(spec) if spec.format == "dag_jsonl"),
        "scheduled workloads cannot consume a direct dag_jsonl graph program"
    );
    lower_common(
        run,
        context,
        workload.worker_count,
        NativeDatasetPlan::Linear(dataset),
        &workload.tokenizer,
        &workload.phases,
        None,
        tokenizers,
    )
}

fn lower_graph(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
    workload: &GraphWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunPlan> {
    let dataset = DirectGraphDatasetV2::decode(&workload.dataset)?.lower()?;
    let tokenizer = lower_authored_tokenizer(&workload.tokenizer, tokenizers)?;
    let dataset = prepare_graph_input(dataset, &tokenizer, context)?;
    build_common_plan(
        run,
        workload.worker_count,
        NativeDatasetPlan::Graph(Box::new(dataset)),
        tokenizer,
        &workload.phases,
        None,
        NativeEndpointPlan::Prepared(lower_prepared_endpoint_plan(context)),
    )
}

fn lower_static_accuracy(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
    workload: &StaticAccuracyWorkloadConfigV2,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunPlan> {
    let accuracy = StaticAccuracyConfigV2::decode(&workload.accuracy)?.lower();
    lower_common(
        run,
        context,
        workload.worker_count,
        NativeDatasetPlan::Linear(accuracy_placeholder_dataset()),
        &workload.tokenizer,
        &workload.phases,
        Some(accuracy),
        tokenizers,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_common(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
    workers: usize,
    dataset: NativeDatasetPlan,
    tokenizer: &RawValue,
    phases: &[PhaseSpec],
    accuracy: Option<AccuracySpec>,
    tokenizers: &dyn OnlineTokenizerSourceResolver,
) -> Result<NativeRunPlan> {
    let tokenizer = lower_authored_tokenizer(tokenizer, tokenizers)?;
    let endpoint = lower_endpoint(context.default_endpoint_profile()?, context)?;
    build_common_plan(
        run,
        workers,
        dataset,
        tokenizer,
        phases,
        accuracy,
        NativeEndpointPlan::Legacy(Box::new(endpoint)),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_common_plan(
    run: &AuthoredRunSpecV2,
    workers: usize,
    dataset: NativeDatasetPlan,
    tokenizer: TokenizerSpec,
    phases: &[PhaseSpec],
    accuracy: Option<AccuracySpec>,
    endpoint: NativeEndpointPlan,
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
            accuracy,
            gpu_telemetry: None,
            network_latency: None,
            server_metrics: None,
            live_streaming: None,
        },
    })
}

fn prepare_graph_input(
    mut graph: NativeGraphDatasetPlan,
    tokenizer: &TokenizerSpec,
    context: &RunnerRunContext,
) -> Result<NativeGraphDatasetPlan> {
    let NativeGraphInputPlan::Authored {
        adapter_name,
        input,
    } = graph.input
    else {
        bail!("protocol-v2 graph preparation received an already prepared input bundle")
    };
    let adapters = GraphInputAdapterRegistry::with_builtin_adapters();
    let adapter = adapters
        .get(&adapter_name)
        .with_context(|| format!("resolving direct Graph-IR adapter {adapter_name:?}"))?;
    let tokenizer_impl = load_tokenizer(Some(&tokenizer.name))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating direct Graph-IR preparation runtime")?;
    let local = tokio::task::LocalSet::new();
    let bundle = local
        .block_on(&runtime, adapter.load(*input, tokenizer_impl.as_ref()))
        .context("loading and validating direct authored Graph-IR input")?;
    ensure!(
        !bundle.plans.is_empty(),
        "authored Graph-IR input contains no root traces after root limiting"
    );
    ensure!(
        bundle.metadata.format == adapter_name,
        "Graph-IR adapter {:?} returned bundle format {:?}",
        adapter_name,
        bundle.metadata.format
    );
    validate_graph_endpoint_profile_references(&bundle, context)?;
    graph.input = NativeGraphInputPlan::Prepared(Arc::new(bundle));
    Ok(graph)
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

fn lower_prepared_endpoint_plan(context: &RunnerRunContext) -> NativePreparedEndpointPlan {
    NativePreparedEndpointPlan {
        default_profile_id: "default".into(),
        profiles: context
            .endpoint_profiles()
            .map(|(profile_id, profile)| NativePreparedEndpointProfile {
                profile_id: profile_id.to_owned(),
                endpoint_id: profile.endpoint_id.clone(),
                config: profile.config.clone(),
                connection_reuse: profile.connection_reuse,
                http2: profile.http2,
                session_header: profile.session_header.clone(),
            })
            .collect(),
    }
}

fn accuracy_placeholder_dataset() -> DatasetSpec {
    // Static evaluators author every prompt and generation control. The normal
    // dataset field is deliberately not loaded or converted; this inert value
    // supplies only the common coordinator's pre-evaluator RNG namespace.
    DatasetSpec::Synthetic(Box::new(SyntheticDatasetSpec {
        entries: 1,
        random_seed: None,
        sampling: "sequential".into(),
        prompts: None,
        prefix_prompts: None,
        turns: DistributionSpec::Fixed(FixedDistributionSpec {
            value: 1.0,
            min: None,
            max: None,
        }),
        turn_delay_ms: DistributionSpec::Fixed(FixedDistributionSpec {
            value: 0.0,
            min: None,
            max: None,
        }),
        turn_delay_ratio: 1.0,
        images: None,
        audio: None,
        video: None,
        rankings: None,
    }))
}

fn lower_endpoint(
    profile: &ValidatedEndpointProfileV2,
    context: &RunnerRunContext,
) -> Result<EndpointSpec> {
    let legacy = context
        .product_registry()
        .endpoints()
        .legacy_endpoint(&profile.endpoint_id)
        .context("resolving the selected endpoint's native execution adapter")?;
    let endpoint_type: EndpointType = legacy.metadata().endpoint_type;
    let raw = profile.config.clone();
    Ok(EndpointSpec {
        urls: raw.urls,
        endpoint_type,
        path: raw.path,
        streaming: raw.streaming,
        use_legacy_max_tokens: raw.use_legacy_max_tokens,
        use_server_token_count: raw.use_server_token_count,
        timeout_seconds: raw.timeout_seconds,
        connection_reuse: profile.connection_reuse,
        request_content_type: raw.request_content_type,
        download_video_content: raw.download_video_content,
        template: raw.template,
        response_field: raw.response_field,
        extra: raw.extra.unwrap_or_default(),
        headers: raw.headers,
        api_key: raw.api_key,
        session_header: profile.session_header.clone(),
        http2: profile.http2,
    })
}

struct PreparedOnlineOperation {
    workload_id: &'static str,
    plan: NativeRunPlan,
    product_registry: Arc<aiperf_extensions::AiperfRegistry>,
}

impl fmt::Debug for PreparedOnlineOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedOnlineOperation")
            .field("workload_id", &self.workload_id)
            .field("benchmark_id", &self.plan.run.benchmark_id)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedOnlineOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let graph_inputs = GraphInputAdapterRegistry::with_builtin_adapters();
        let report_path = execute_native_plan_with_factories(
            self.plan,
            &NativeHttpExecutionBackendFactory,
            &graph_inputs,
            &NativeRunnerGraphPlacementFactory,
            self.product_registry.as_ref(),
        )?;
        Ok(PreparedRunOutcome {
            report_path,
            provenance: BTreeMap::from([
                ("backend".into(), BACKEND_ID.into()),
                ("workload".into(), self.workload_id.into()),
            ]),
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
    fn registration_adds_only_real_online_pairs() {
        let mut builder = RunnerRegistryBuilder::new();
        // The builder intentionally rejects dangling pairs at freeze time;
        // registration itself proves the function has no hidden mode switch.
        register_online_http_pairs(&mut builder).unwrap();
        register_online_http_static_accuracy_pair(&mut builder).unwrap();
        register_online_http_scheduled_pair(&mut builder).unwrap();
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
