// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached remote-dataset clients for native accuracy benchmarks.
//!
//! Remote retrieval is a concrete control-plane implementation behind
//! `aiperf_accuracy::DatasetSource`: benchmark parsing itself remains independent of
//! HTTP and can consume explicit JSON/JSONL files in offline or hermetic runs. The
//! official MMLU-Pro path downloads its two Parquet shards directly; the rows API
//! remains available for mirrors and small test fixtures.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use aiperf_accuracy::{
    BIGBENCH_TASKS, DatasetSplit, JsonDatasetSource, MMLU_PRO_DATASET, MMLU_SUBJECTS,
};
use aiperf_clock::Clock;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{RequestConfig, RequestRecord, Response};
use aiperf_transport::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::Deserialize;
use serde_json::Value;
use url::Url;

/// Default Hugging Face datasets-server rows endpoint.
pub const HUGGING_FACE_ROWS_ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";

/// Default Hugging Face dataset repository base URL.
pub const HUGGING_FACE_DATASET_BASE: &str = "https://huggingface.co/datasets";

/// Hugging Face Hub API base used to enumerate auto-converted Parquet shards.
pub const HUGGING_FACE_API_BASE: &str = "https://huggingface.co/api/datasets";

/// Reviewed MMLU-Pro dataset revision used for reproducible benchmark inputs.
pub const MMLU_PRO_DATASET_REVISION: &str = "b189ec765aa7ed75c8acfea42df31fdae71f97be";

async fn download_binary(
    transport: &HttpTransport,
    initial_url: &str,
    token: Option<&str>,
) -> anyhow::Result<bytes::Bytes> {
    let mut url = initial_url.to_string();
    for _ in 0..=8 {
        let mut config = RequestConfig::new(&url);
        if let Some(token) = token.filter(|_| is_hugging_face_host(&url)) {
            config = config.header("Authorization", format!("Bearer {token}"));
        }
        let record = transport.get(&config).await;
        if matches!(record.status, Some(301 | 302 | 303 | 307 | 308)) {
            let location = record
                .response_headers
                .get("location")
                .ok_or_else(|| anyhow::anyhow!("redirect from {url} had no Location header"))?;
            url = Url::parse(&url)
                .and_then(|base| base.join(location))
                .map_err(|error| anyhow::anyhow!("resolving redirect from {url}: {error}"))?
                .into();
            continue;
        }
        ensure_success(&record, &url)?;
        return record
            .responses
            .into_iter()
            .find_map(|response| match response {
                Response::Text(text) => Some(text.body),
                Response::Sse(_) => None,
            })
            .ok_or_else(|| anyhow::anyhow!("download from {url} had no response body"));
    }
    anyhow::bail!("download from {initial_url} exceeded the redirect limit")
}

fn ensure_success(record: &RequestRecord, url: &str) -> anyhow::Result<()> {
    if let Some(error) = &record.error {
        anyhow::bail!("download from {url} failed: {error:?}");
    }
    Ok(())
}

fn is_hugging_face_host(url: &str) -> bool {
    Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "huggingface.co" || host.ends_with(".huggingface.co"))
}

fn decode_parquet(body: bytes::Bytes, source: &str) -> anyhow::Result<Vec<Value>> {
    let reader = SerializedFileReader::new(body)
        .map_err(|error| anyhow::anyhow!("opening {source} as Parquet: {error}"))?;
    let rows = reader
        .get_row_iter(None)
        .map_err(|error| anyhow::anyhow!("reading {source} Parquet rows: {error}"))?;
    rows.map(|row| {
        row.map(|row| row.to_json_value())
            .map_err(|error| anyhow::anyhow!("decoding {source} Parquet row: {error}"))
    })
    .collect()
}

/// Generic Hugging Face auto-converted Parquet client.
pub struct HuggingFaceHubClient {
    transport: HttpTransport,
    api_base: String,
    token: Option<String>,
}

impl HuggingFaceHubClient {
    /// Builds the public Hub client and discovers an optional token from
    /// `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` for gated datasets.
    pub fn new(clock: Rc<dyn Clock>) -> Self {
        Self {
            transport: HttpTransport::new(clock, ClientConfig::default()),
            api_base: HUGGING_FACE_API_BASE.to_string(),
            token: std::env::var("HF_TOKEN")
                .ok()
                .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok()),
        }
    }

    /// Overrides the API base for mirrors and hermetic tests.
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = api_base.into();
        self
    }

    /// Overrides bearer-token discovery.
    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    /// Fetches every Parquet shard for one repository/config/split.
    pub async fn fetch_split(
        &self,
        repository: &str,
        config: &str,
        split: &str,
    ) -> anyhow::Result<Vec<Value>> {
        let list_url = format!(
            "{}/{repository}/parquet/{config}/{split}",
            self.api_base.trim_end_matches('/')
        );
        let body = download_binary(&self.transport, &list_url, self.token.as_deref()).await?;
        let urls: Vec<String> = serde_json::from_slice(&body).map_err(|error| {
            anyhow::anyhow!(
                "parsing Hugging Face Parquet shard list for {repository}/{config}/{split}: {error}"
            )
        })?;
        if urls.is_empty() {
            anyhow::bail!(
                "Hugging Face returned no Parquet shards for {repository}/{config}/{split}"
            );
        }
        let mut rows = Vec::new();
        for (index, url) in urls.iter().enumerate() {
            let body = download_binary(&self.transport, url, self.token.as_deref()).await?;
            rows.extend(decode_parquet(
                body,
                &format!("{repository}/{config}/{split} shard {index}"),
            )?);
        }
        Ok(rows)
    }

    /// Downloads a UTF-8 control-plane asset through the same transport.
    pub async fn fetch_text(&self, url: &str) -> anyhow::Result<String> {
        let body = download_binary(&self.transport, url, self.token.as_deref()).await?;
        String::from_utf8(body.to_vec())
            .map_err(|error| anyhow::anyhow!("downloaded text from {url} is not UTF-8: {error}"))
    }
}

/// Remote split retrieval seam used by the cache installer.
///
/// Implementations may use direct artifacts, a dataset service, or an internal
/// mirror without coupling benchmark parsing to a particular control plane.
#[async_trait(?Send)]
pub trait RemoteDatasetClient {
    /// Fetch every decoded row in `split`.
    async fn fetch_split(&self, split: DatasetSplit) -> anyhow::Result<Vec<Value>>;
}

/// Direct Parquet-artifact client over AIPerf's Clock-injected HTTP stack.
pub struct HuggingFaceParquetClient {
    transport: HttpTransport,
    base: String,
    dataset: String,
    revision: String,
}

impl HuggingFaceParquetClient {
    /// Builds the pinned official MMLU-Pro artifact client.
    pub fn mmlu_pro(clock: Rc<dyn Clock>) -> Self {
        Self::new(
            clock,
            HUGGING_FACE_DATASET_BASE,
            MMLU_PRO_DATASET,
            MMLU_PRO_DATASET_REVISION,
        )
    }

    /// Builds a direct-artifact client with injectable repository coordinates.
    pub fn new(
        clock: Rc<dyn Clock>,
        base: impl Into<String>,
        dataset: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        Self {
            transport: HttpTransport::new(clock, ClientConfig::default()),
            base: base.into(),
            dataset: dataset.into(),
            revision: revision.into(),
        }
    }

    fn split_url(&self, split: DatasetSplit) -> String {
        format!(
            "{}/{}/resolve/{}/data/{}-00000-of-00001.parquet?download=true",
            self.base.trim_end_matches('/'),
            self.dataset,
            self.revision,
            split.as_str()
        )
    }

    async fn download(&self, split: DatasetSplit) -> anyhow::Result<bytes::Bytes> {
        download_binary(&self.transport, &self.split_url(split), None).await
    }
}

#[async_trait(?Send)]
impl RemoteDatasetClient for HuggingFaceParquetClient {
    async fn fetch_split(&self, split: DatasetSplit) -> anyhow::Result<Vec<Value>> {
        let body = self.download(split).await?;
        decode_parquet(body, &format!("{} {}", self.dataset, split.as_str()))
    }
}

/// Hugging Face rows-API client over AIPerf's Clock-injected HTTP stack.
pub struct HuggingFaceRowsClient {
    transport: HttpTransport,
    endpoint: String,
    dataset: String,
    config: String,
    page_size: usize,
}

impl HuggingFaceRowsClient {
    /// Builds a client for the public MMLU-Pro dataset.
    pub fn mmlu_pro(clock: Rc<dyn Clock>) -> Self {
        Self::new(
            clock,
            HUGGING_FACE_ROWS_ENDPOINT,
            MMLU_PRO_DATASET,
            "default",
        )
    }

    /// Builds a client with an injectable endpoint for mirrors and tests.
    pub fn new(
        clock: Rc<dyn Clock>,
        endpoint: impl Into<String>,
        dataset: impl Into<String>,
        config: impl Into<String>,
    ) -> Self {
        Self {
            transport: HttpTransport::new(clock, ClientConfig::default()),
            endpoint: endpoint.into(),
            dataset: dataset.into(),
            config: config.into(),
            page_size: 100,
        }
    }

    /// Overrides rows per request. Values are clamped to the server's 1..=100 range.
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size.clamp(1, 100);
        self
    }

    async fn fetch_rows(&self, split: DatasetSplit) -> anyhow::Result<Vec<Value>> {
        let mut rows = Vec::new();
        let mut total = None;
        loop {
            if total.is_some_and(|expected| rows.len() >= expected) {
                break;
            }
            let cfg = RequestConfig::new(&self.endpoint)
                .param("dataset", &self.dataset)
                .param("config", &self.config)
                .param("split", split.as_str())
                .param("offset", rows.len().to_string())
                .param("length", self.page_size.to_string());
            let record = self.transport.get(&cfg).await;
            if let Some(error) = record.error {
                anyhow::bail!(
                    "fetching {} split {} from {} failed: {error:?}",
                    self.dataset,
                    split.as_str(),
                    self.endpoint
                );
            }
            let text = record
                .responses
                .iter()
                .find_map(|response| match response {
                    Response::Text(text) => Some(text.text.as_str()),
                    Response::Sse(_) => None,
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Hugging Face rows response for {} {} had no text body",
                        self.dataset,
                        split.as_str()
                    )
                })?;
            let page: RowsPage = serde_json::from_str(text).map_err(|error| {
                anyhow::anyhow!(
                    "parsing Hugging Face rows response for {} {} at offset {}: {error}",
                    self.dataset,
                    split.as_str(),
                    rows.len()
                )
            })?;
            match total {
                Some(expected) if expected != page.num_rows_total => anyhow::bail!(
                    "Hugging Face rows total changed during download: {expected} -> {}",
                    page.num_rows_total
                ),
                None => total = Some(page.num_rows_total),
                Some(_) => {}
            }
            if page.rows.is_empty() && rows.len() < page.num_rows_total {
                anyhow::bail!(
                    "Hugging Face rows pagination stopped at {} of {} for {} {}",
                    rows.len(),
                    page.num_rows_total,
                    self.dataset,
                    split.as_str()
                );
            }
            rows.extend(page.rows.into_iter().map(|row| row.row));
            if rows.len() > page.num_rows_total {
                anyhow::bail!(
                    "Hugging Face rows returned {} rows but declared {} for {} {}",
                    rows.len(),
                    page.num_rows_total,
                    self.dataset,
                    split.as_str()
                );
            }
        }
        Ok(rows)
    }
}

#[async_trait(?Send)]
impl RemoteDatasetClient for HuggingFaceRowsClient {
    async fn fetch_split(&self, split: DatasetSplit) -> anyhow::Result<Vec<Value>> {
        self.fetch_rows(split).await
    }
}

#[derive(Debug, Deserialize)]
struct RowsPage {
    rows: Vec<RowEnvelope>,
    num_rows_total: usize,
}

#[derive(Debug, Deserialize)]
struct RowEnvelope {
    row: Value,
}

/// Ensure cached `validation.json` and `test.json` files exist and return a
/// filesystem dataset source. `refresh` replaces both files from the remote source.
pub async fn prepare_mmlu_pro_dataset(
    client: &dyn RemoteDatasetClient,
    cache_directory: impl AsRef<Path>,
    refresh: bool,
) -> anyhow::Result<JsonDatasetSource> {
    let cache_directory = cache_directory.as_ref();
    std::fs::create_dir_all(cache_directory).map_err(|error| {
        anyhow::anyhow!(
            "creating accuracy dataset cache {}: {error}",
            cache_directory.display()
        )
    })?;
    for split in [DatasetSplit::Validation, DatasetSplit::Test] {
        let path = cache_directory.join(format!("{}.json", split.as_str()));
        if refresh || !path.is_file() {
            let rows = client.fetch_split(split).await?;
            write_json_atomically(&path, &rows)?;
        }
    }
    Ok(JsonDatasetSource::from_directory(cache_directory))
}

/// Remote benchmark-dataset preparation seam.
#[async_trait(?Send)]
pub trait BenchmarkDatasetProvider {
    /// Canonical benchmark name served by this provider.
    fn benchmark(&self) -> &'static str;
    /// Populate/reuse a cache directory and return its filesystem source.
    async fn prepare(
        &self,
        clock: Rc<dyn Clock>,
        cache_root: &Path,
        refresh: bool,
    ) -> anyhow::Result<JsonDatasetSource>;
}

#[derive(Debug, Clone)]
struct FetchSpec {
    repository: &'static str,
    config: String,
    upstream_split: &'static str,
    local_split: DatasetSplit,
    discriminator: Option<(&'static str, String)>,
}

struct HuggingFacePlanProvider {
    benchmark: &'static str,
    fetches: Vec<FetchSpec>,
}

#[async_trait(?Send)]
impl BenchmarkDatasetProvider for HuggingFacePlanProvider {
    fn benchmark(&self) -> &'static str {
        self.benchmark
    }

    async fn prepare(
        &self,
        clock: Rc<dyn Clock>,
        cache_root: &Path,
        refresh: bool,
    ) -> anyhow::Result<JsonDatasetSource> {
        let directory = cache_root.join(self.benchmark);
        let required = self
            .fetches
            .iter()
            .map(|fetch| fetch.local_split)
            .collect::<std::collections::BTreeSet<_>>();
        if !refresh
            && required
                .iter()
                .all(|split| directory.join(format!("{}.json", split.as_str())).is_file())
        {
            return Ok(JsonDatasetSource::from_directory(directory));
        }
        let client = HuggingFaceHubClient::new(clock);
        let mut split_rows = BTreeMap::<DatasetSplit, Vec<Value>>::new();
        for fetch in &self.fetches {
            let mut rows = client
                .fetch_split(fetch.repository, &fetch.config, fetch.upstream_split)
                .await?;
            if let Some((field, value)) = &fetch.discriminator {
                for row in &mut rows {
                    row.as_object_mut()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "{} {}/{} row is not an object",
                                fetch.repository,
                                fetch.config,
                                fetch.upstream_split
                            )
                        })?
                        .insert((*field).to_string(), Value::String(value.clone()));
                }
            }
            split_rows
                .entry(fetch.local_split)
                .or_default()
                .extend(rows);
        }
        install_splits(&directory, &split_rows)?;
        Ok(JsonDatasetSource::from_directory(directory))
    }
}

struct MmluProDatasetProvider;

#[async_trait(?Send)]
impl BenchmarkDatasetProvider for MmluProDatasetProvider {
    fn benchmark(&self) -> &'static str {
        "mmlu-pro"
    }

    async fn prepare(
        &self,
        clock: Rc<dyn Clock>,
        cache_root: &Path,
        refresh: bool,
    ) -> anyhow::Result<JsonDatasetSource> {
        let client = HuggingFaceParquetClient::mmlu_pro(clock);
        prepare_mmlu_pro_dataset(&client, cache_root.join(self.benchmark()), refresh).await
    }
}

struct BigBenchDatasetProvider;

/// Immutable DeepEval revision supplying its byte-reference BBH prompt assets.
const DEEPEVAL_BBH_PROMPT_REVISION: &str = "625814c0c7f3fe88abd2dd7cf96944b2b4d9ed68";

#[async_trait(?Send)]
impl BenchmarkDatasetProvider for BigBenchDatasetProvider {
    fn benchmark(&self) -> &'static str {
        "bigbench"
    }

    async fn prepare(
        &self,
        clock: Rc<dyn Clock>,
        cache_root: &Path,
        refresh: bool,
    ) -> anyhow::Result<JsonDatasetSource> {
        let directory = cache_root.join(self.benchmark());
        let path = directory.join("test.json");
        if !refresh && path.is_file() {
            return Ok(JsonDatasetSource::from_directory(directory));
        }
        let client = HuggingFaceHubClient::new(clock);
        let mut all_rows = Vec::new();
        for task in BIGBENCH_TASKS {
            let cot_url = format!(
                "https://raw.githubusercontent.com/confident-ai/deepeval/{DEEPEVAL_BBH_PROMPT_REVISION}/deepeval/benchmarks/big_bench_hard/cot_prompts/{task}.txt"
            );
            let shot_url = format!(
                "https://raw.githubusercontent.com/confident-ai/deepeval/{DEEPEVAL_BBH_PROMPT_REVISION}/deepeval/benchmarks/big_bench_hard/shot_prompts/{task}.txt"
            );
            let cot_prompt = client.fetch_text(&cot_url).await?;
            let shot_prompt = client.fetch_text(&shot_url).await?;
            let mut rows = client.fetch_split("lukaemon/bbh", task, "test").await?;
            for row in &mut rows {
                let object = row.as_object_mut().ok_or_else(|| {
                    anyhow::anyhow!("lukaemon/bbh {task} test row is not an object")
                })?;
                object.insert("_task".to_string(), Value::String((*task).to_string()));
                object.insert("_cot_prompt".to_string(), Value::String(cot_prompt.clone()));
                object.insert(
                    "_shot_prompt".to_string(),
                    Value::String(shot_prompt.clone()),
                );
            }
            all_rows.extend(rows);
        }
        install_splits(
            &directory,
            &BTreeMap::from([(DatasetSplit::Test, all_rows)]),
        )?;
        Ok(JsonDatasetSource::from_directory(directory))
    }
}

/// Registry of remote dataset providers. Benchmark parsing remains behind
/// `DatasetSource`; this registry only owns acquisition/cache policy.
pub struct AccuracyDatasetRegistry {
    providers: BTreeMap<&'static str, Rc<dyn BenchmarkDatasetProvider>>,
}

impl AccuracyDatasetRegistry {
    /// Builds providers for every in-tree benchmark.
    pub fn builtin() -> Self {
        let mut providers: Vec<Rc<dyn BenchmarkDatasetProvider>> = vec![
            Rc::new(MmluProDatasetProvider),
            Rc::new(BigBenchDatasetProvider),
            Rc::new(plan_provider(
                "aime",
                "Maxwell-Jia/AIME_2024",
                "default",
                &[("train", DatasetSplit::Train)],
            )),
            Rc::new(plan_provider(
                "hellaswag",
                "Rowan/hellaswag",
                "default",
                &[
                    ("train", DatasetSplit::Train),
                    ("validation", DatasetSplit::Validation),
                ],
            )),
            Rc::new(plan_provider(
                "aime24",
                "HuggingFaceH4/aime_2024",
                "default",
                &[("train", DatasetSplit::Train)],
            )),
            Rc::new(plan_provider(
                "aime25",
                "yentinglin/aime_2025",
                "default",
                &[("train", DatasetSplit::Train)],
            )),
            Rc::new(plan_provider(
                "math-500",
                "HuggingFaceH4/MATH-500",
                "default",
                &[("test", DatasetSplit::Test)],
            )),
            Rc::new(plan_provider(
                "gsm8k",
                "openai/gsm8k",
                "main",
                &[("test", DatasetSplit::Test)],
            )),
            Rc::new(plan_provider(
                "gpqa-diamond",
                "Idavidrein/gpqa",
                "gpqa_diamond",
                &[("train", DatasetSplit::Train)],
            )),
            Rc::new(plan_provider(
                "lcb-codegeneration",
                "livecodebench/code_generation_lite",
                &std::env::var("AIPERF_ACCURACY_LCB_RELEASE_TAG")
                    .unwrap_or_else(|_| "v4_v5".to_string()),
                &[("test", DatasetSplit::Test)],
            )),
        ];

        let mut mmlu_fetches = Vec::with_capacity(MMLU_SUBJECTS.len() * 2);
        for subject in MMLU_SUBJECTS {
            for (upstream_split, local_split) in
                [("dev", DatasetSplit::Dev), ("test", DatasetSplit::Test)]
            {
                mmlu_fetches.push(FetchSpec {
                    repository: "lighteval/mmlu",
                    config: (*subject).to_string(),
                    upstream_split,
                    local_split,
                    discriminator: Some(("_subject", (*subject).to_string())),
                });
            }
        }
        providers.push(Rc::new(HuggingFacePlanProvider {
            benchmark: "mmlu",
            fetches: mmlu_fetches,
        }));
        Self {
            providers: providers
                .into_iter()
                .map(|provider| (provider.benchmark(), provider))
                .collect(),
        }
    }

    /// Prepare one canonical benchmark's remote cache.
    pub async fn prepare(
        &self,
        benchmark: &str,
        clock: Rc<dyn Clock>,
        cache_root: &Path,
        refresh: bool,
    ) -> anyhow::Result<JsonDatasetSource> {
        let provider = self.providers.get(benchmark).ok_or_else(|| {
            anyhow::anyhow!(
                "no remote dataset provider registered for {benchmark}; use --accuracy-dataset"
            )
        })?;
        provider.prepare(clock, cache_root, refresh).await
    }
}

impl Default for AccuracyDatasetRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

fn plan_provider(
    benchmark: &'static str,
    repository: &'static str,
    config: &str,
    splits: &[(&'static str, DatasetSplit)],
) -> HuggingFacePlanProvider {
    HuggingFacePlanProvider {
        benchmark,
        fetches: splits
            .iter()
            .map(|(upstream_split, local_split)| FetchSpec {
                repository,
                config: config.to_string(),
                upstream_split,
                local_split: *local_split,
                discriminator: None,
            })
            .collect(),
    }
}

fn install_splits(
    directory: &Path,
    split_rows: &BTreeMap<DatasetSplit, Vec<Value>>,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(directory).map_err(|error| {
        anyhow::anyhow!(
            "creating accuracy dataset cache {}: {error}",
            directory.display()
        )
    })?;
    for (split, rows) in split_rows {
        write_json_atomically(&directory.join(format!("{}.json", split.as_str())), rows)?;
    }
    Ok(())
}

fn write_json_atomically(path: &Path, rows: &[Value]) -> anyhow::Result<()> {
    let temp = temporary_path(path);
    let bytes = serde_json::to_vec(rows)?;
    std::fs::write(&temp, bytes)
        .map_err(|error| anyhow::anyhow!("writing dataset cache {}: {error}", temp.display()))?;
    std::fs::rename(&temp, path).map_err(|error| {
        let _ = std::fs::remove_file(&temp);
        anyhow::anyhow!(
            "installing dataset cache {} -> {}: {error}",
            temp.display(),
            path.display()
        )
    })
}

fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dataset.json");
    path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()))
}

/// Resolve the default native accuracy cache root.
pub fn default_accuracy_cache_root() -> PathBuf {
    if let Some(root) = std::env::var_os("AIPERF_CACHE_DIR") {
        return PathBuf::from(root).join("datasets");
    }
    if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(root).join("aiperf/datasets");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache/aiperf/datasets");
    }
    std::env::temp_dir().join("aiperf/datasets")
}

/// Resolve the backwards-compatible MMLU-Pro cache directory.
pub fn default_accuracy_cache_directory() -> PathBuf {
    default_accuracy_cache_root().join("mmlu-pro")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use aiperf_accuracy::DatasetSource;
    use aiperf_clock::RealClock;
    use axum::{Json, Router, extract::Query, routing::get};
    use serde_json::json;

    use super::*;

    async fn rows_handler(Query(query): Query<HashMap<String, String>>) -> Json<Value> {
        let split = query["split"].as_str();
        let offset = query["offset"].parse::<usize>().unwrap();
        let length = query["length"].parse::<usize>().unwrap();
        let total = if split == "validation" { 3 } else { 5 };
        let rows = (offset..(offset + length).min(total))
            .map(|index| json!({"row_idx": index, "row": {"split": split, "index": index}}))
            .collect::<Vec<_>>();
        Json(json!({"rows": rows, "num_rows_total": total}))
    }

    #[tokio::test]
    async fn get_paginates_and_installs_both_cache_splits() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let app = Router::new().route("/rows", get(rows_handler));
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                let clock: Rc<dyn Clock> = RealClock::new();
                let client = HuggingFaceRowsClient::new(
                    clock,
                    format!("http://{address}/rows"),
                    "fixture/dataset",
                    "default",
                )
                .with_page_size(2);
                let directory = std::env::temp_dir().join(format!(
                    "aiperf_accuracy_cache_{}_{}",
                    std::process::id(),
                    address.port()
                ));
                let source = prepare_mmlu_pro_dataset(&client, &directory, true)
                    .await
                    .unwrap();
                assert_eq!(source.load_rows(DatasetSplit::Validation).unwrap().len(), 3);
                assert_eq!(source.load_rows(DatasetSplit::Test).unwrap().len(), 5);
                std::fs::remove_dir_all(directory).unwrap();
            })
            .await;
    }
}
