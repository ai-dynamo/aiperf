// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native sidecar resource factories, media ingestion, and run-artifact creation.

use super::*;

/// Startup seam for native sidecar resources.
///
/// Preparation runs on the coordinator's `LocalSet`, may supervise extension
/// workers, and must return the exact Clock/anchor later given to scheduling
/// and HTTP execution. A distribution can replace resource construction
/// without changing artifact ownership or phase execution.
#[async_trait(?Send)]
pub(crate) trait NativeSidecarResourceFactory: std::fmt::Debug + Send + Sync {
    /// Prepare the complete run-owned bundle without creating local artifacts.
    ///
    /// The `clock` and `real_clock_anchor` are constructed once at the native
    /// driver layer (real vs virtual chosen there) and threaded in, so the
    /// bundle returns the exact clock scheduling and HTTP execution will use;
    /// the factory does not create a separate clock.
    async fn prepare(
        &self,
        run: &NativeRunSpec,
        clock: Rc<dyn Clock>,
        real_clock_anchor: RealClockAnchor,
    ) -> Result<PreparedNativeSidecarResources>;
}

/// Built-in native sidecar resource composition.
#[derive(Debug)]
pub(crate) struct BuiltinNativeSidecarResourceFactory;

/// Resources prepared before the exclusive artifact target is created.
///
/// The bundle owns cleanup order and retains every path/fact derived during
/// preparation so execution never reopens the authored sidecar configuration.
pub(crate) struct PreparedNativeSidecarResources {
    pub(crate) real_clock_anchor: RealClockAnchor,
    pub(crate) clock: Rc<dyn Clock>,
    pub(crate) content_server: Option<Box<dyn ContentServerRuntime>>,
    pub(crate) gpu_telemetry: Option<GpuTelemetryRun>,
    pub(crate) network_latency: Option<NetworkLatencyRun>,
    pub(crate) server_metrics: Option<ServerMetricsRun>,
    pub(crate) server_profiler: Option<Rc<crate::engine::control_hooks::ServerProfilerCoordinator>>,
    pub(crate) live_streaming: Option<PythonLiveStreamingRun>,
    pub(crate) gpu_records_path: Option<PathBuf>,
    pub(crate) network_latency_records_path: Option<PathBuf>,
    pub(crate) server_metrics_jsonl_path: Option<PathBuf>,
    pub(crate) server_metrics_parquet_wire_path: Option<PathBuf>,
    /// Background task folding content records into media-fetch metrics. Joined
    /// at the finalize tail after the content server (its record sender) is
    /// dropped.
    pub(crate) media_handle: Option<tokio::task::JoinHandle<MediaMetricsSummary>>,
    pub(crate) media_record_sender: Option<crate::content_server::ContentRecordSender>,
}

/// Artifact filename for per-fetch media records.
pub(crate) const MEDIA_RECORDS_FILENAME: &str = "media_records.jsonl";

/// The content-server origin to tag media URLs with, or `None` when no server
/// publishes files this run (media stays inline, nothing to correlate).
pub(crate) fn content_server_media_base(run: &NativeRunSpec) -> Result<Option<Arc<str>>> {
    Ok(run
        .sidecars
        .content_server()?
        .filter(|spec| spec.content_dir.is_some())
        .map(|spec| Arc::from(spec.base_url())))
}

/// Ingest one content record into the aggregator and stream its row. Ingestion
/// (and thus metric folding) always happens; the row is written only when the
/// artifact writer is available.
pub(crate) fn ingest_media_record(
    aggregator: &mut MediaFetchAggregator,
    writer: Option<&mut MediaRecordWriter>,
    record: &ContentRequestRecord,
) {
    if let Some(media_record) = aggregator.ingest(record)
        && let Some(writer) = writer
        && let Err(error) = writer.write(&media_record)
    {
        tracing::warn!(error = %error, "writing media_records line failed");
    }
}

#[async_trait(?Send)]
impl NativeSidecarResourceFactory for BuiltinNativeSidecarResourceFactory {
    async fn prepare(
        &self,
        run: &NativeRunSpec,
        clock: Rc<dyn Clock>,
        real_clock_anchor: RealClockAnchor,
    ) -> Result<PreparedNativeSidecarResources> {
        let endpoint_urls = run.endpoint.default_urls()?;
        let content_server_spec = run.sidecars.content_server()?;
        let gpu_spec = run.sidecars.gpu_telemetry()?;
        let network_spec = run.sidecars.network_latency()?;
        let server_spec = run.sidecars.server_metrics()?;
        let live_spec = run.sidecars.live_streaming()?;

        // These constructors and path checks cannot start phase tasks. Finish
        // every fallible local step before supervising a GPU/live child.
        let network_latency = network_spec
            .map(|spec| {
                NetworkLatencyRun::new(&run.benchmark_id, spec, endpoint_urls, clock.clone())
            })
            .transpose()?;
        let server_metrics = server_spec
            .map(|spec| ServerMetricsRun::new(spec, clock.clone()))
            .transpose()?;
        let gpu_records_path = gpu_spec
            .map(|spec| {
                artifact_path(
                    &run.artifact_dir,
                    &spec.records_path,
                    "gpu_telemetry.records_path",
                )
            })
            .transpose()?;
        let network_latency_records_path = network_spec
            .and_then(|spec| spec.probe.as_ref())
            .map(|probe| {
                artifact_path(
                    &run.artifact_dir,
                    &probe.records_path,
                    "network_latency.probe.records_path",
                )
            })
            .transpose()?;
        let server_metrics_jsonl_path = server_spec
            .and_then(|spec| spec.jsonl_path.as_ref())
            .map(|path| artifact_path(&run.artifact_dir, path, "server_metrics.jsonl_path"))
            .transpose()?;
        let server_metrics_parquet_wire_path = server_spec
            .and_then(|spec| spec.parquet_wire_path.as_ref())
            .map(|path| artifact_path(&run.artifact_dir, path, "server_metrics.parquet_wire_path"))
            .transpose()?;
        let live_metrics_config = live_spec
            .is_some()
            .then(|| metrics_config(&run.metrics, run.endpoint.use_server_token_count()))
            .transpose()?;

        let mut media_handle = None;
        let mut media_record_sender = None;
        let content_server = match content_server_spec {
            Some(spec) => {
                // Wire media-fetch metrics only when the server publishes files
                // (a content dir is set); otherwise media stays inline, no URLs
                // are fetched, and there is nothing to correlate.
                let record_sink = if spec.content_dir.is_some() {
                    let path = artifact_path(
                        &run.artifact_dir,
                        Path::new(MEDIA_RECORDS_FILENAME),
                        "media_records",
                    )?;
                    let (record_tx, mut record_rx) =
                        tokio::sync::mpsc::channel::<ContentRequestRecord>(256);
                    let record_sender = crate::content_server::ContentRecordSender::new(record_tx);
                    let handle = tokio::spawn(async move {
                        let mut aggregator = MediaFetchAggregator::new();
                        let mut writer = match MediaRecordWriter::create(&path) {
                            Ok(writer) => Some(writer),
                            Err(error) => {
                                tracing::warn!(error = %error, "media_records artifact unavailable");
                                None
                            }
                        };
                        // Drains until every sender is dropped, which happens when
                        // the content server is shut down at the finalize tail (all
                        // fetches have completed by then, so none are lost). Writes
                        // go through a BufWriter, so disk syscalls are infrequent and
                        // confined to this dedicated task.
                        while let Some(record) = record_rx.recv().await {
                            ingest_media_record(&mut aggregator, writer.as_mut(), &record);
                        }
                        if let Some(mut writer) = writer {
                            let _ = writer.flush();
                        }
                        aggregator.finish()
                    });
                    media_handle = Some(handle);
                    media_record_sender = Some(record_sender.clone());
                    Some(record_sender)
                } else {
                    None
                };
                Some(
                    NativeContentServerFactory::default()
                        .start(ContentServerConfig {
                            host: spec.host.clone(),
                            port: spec.port,
                            content_dir: spec.content_dir.clone(),
                            max_tracked_records: spec.max_tracked_records,
                            record_sink,
                        })
                        .await
                        .context("starting native content server")?,
                )
            }
            None => None,
        };

        let gpu_telemetry = match gpu_spec {
            Some(spec) => Some(GpuTelemetryRun::new(spec, clock.clone()).await?),
            None => None,
        };
        let live_streaming = if live_spec.is_some() {
            match PythonLiveStreamingRun::spawn(
                run,
                live_metrics_config.expect("present live spec prepared its metrics config"),
            )
            .await
            {
                Ok(worker) => Some(worker),
                Err(error) => {
                    tracing::warn!(
                        error = format!("{error:#}"),
                        "live telemetry extension failed to start"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(PreparedNativeSidecarResources {
            real_clock_anchor,
            clock,
            content_server,
            gpu_telemetry,
            network_latency,
            server_metrics,
            server_profiler: None,
            live_streaming,
            gpu_records_path,
            network_latency_records_path,
            server_metrics_jsonl_path,
            server_metrics_parquet_wire_path,
            media_handle,
            media_record_sender,
        })
    }
}

impl PreparedNativeSidecarResources {
    /// Shut the content server down (releasing the record sender) so the drain
    /// task reaches channel-close, then collect its finalized distributions.
    /// Returns an empty summary when there was no media wiring. Idempotent.
    pub(crate) async fn finalize_media_metrics(&mut self) -> Result<MediaMetricsSummary> {
        let Some(handle) = self.media_handle.take() else {
            return Ok(MediaMetricsSummary::default());
        };
        // Dropping the server releases the tracker's sender; the drain task then
        // sees channel-close and finalizes. All fetches have completed by the
        // finalize tail, so this loses none. `shutdown_run_resources` later finds
        // the server already taken and skips it.
        if let Some(mut content_server) = self.content_server.take()
            && let Err(error) = content_server.shutdown().await
        {
            tracing::warn!(error = %error, "content server shutdown during media finalize failed");
        }
        let overflowed = self
            .media_record_sender
            .take()
            .is_some_and(|sender| sender.overflowed());
        match handle.await {
            Ok(summary) => {
                tracing::info!(
                    total_fetches = summary.total_fetches,
                    unmatched = summary.unmatched,
                    negative_ttmf = summary.negative_ttmf,
                    "media-fetch metrics finalized"
                );
                if overflowed {
                    anyhow::bail!("media record queue overflowed")
                }
                Ok(summary)
            }
            Err(error) => {
                tracing::warn!(error = %error, "media aggregator task failed to join");
                Ok(MediaMetricsSummary::default())
            }
        }
    }

    pub(crate) fn live_sink(&self) -> Option<Rc<dyn LiveResultsSink>> {
        self.live_streaming
            .as_ref()
            .map(PythonLiveStreamingRun::sink)
    }

    pub(crate) async fn activate_live_streaming(&mut self) {
        let activation = match self.live_streaming.as_mut() {
            Some(worker) => worker.activate().await,
            None => return,
        };
        if let Err(error) = activation {
            tracing::warn!(
                error = format!("{error:#}"),
                "live telemetry extension failed to activate"
            );
            self.live_streaming.take();
        }
    }

    pub(crate) async fn shutdown_run_resources(&mut self) {
        if let Some(profiler) = self.server_profiler.take()
            && let Err(error) = profiler.force_stop().await
        {
            tracing::warn!(
                error = %error,
                "server profiler failed to stop during run shutdown"
            );
        }
        if let Some(worker) = self.live_streaming.take()
            && let Err(error) = worker.shutdown().await
        {
            tracing::warn!(
                error = format!("{error:#}"),
                "live telemetry extension failed to shut down cleanly"
            );
        }

        // Server-metrics tasks belong to phase sidecars and have already
        // drained. Drop that retained source graph before supervised GPU
        // workers, matching the explicit run-owned cleanup order.
        self.server_metrics.take();
        if let Some(gpu_telemetry) = self.gpu_telemetry.take() {
            gpu_telemetry.shutdown().await;
        }
        self.network_latency.take();
        if let Some(mut content_server) = self.content_server.take()
            && let Err(error) = content_server.shutdown().await
        {
            tracing::warn!(
                error = format!("{error:#}"),
                "content server failed to shut down cleanly"
            );
        }
    }
}

pub(crate) fn create_run_artifacts(run: &NativeRunSpec) -> Result<()> {
    std::fs::create_dir_all(&run.artifact_dir).with_context(|| {
        format!(
            "creating run artifact directory {}",
            run.artifact_dir.display()
        )
    })?;
    crate::engine::phase_manifest::write_phase_manifest(&run.artifact_dir, &run.phases)?;
    materialize_user_files(&run.artifact_dir, &run.user_files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use crate::content_server::ContentRecordSender;

    /// A served content record whose query string carries a parseable media tag
    /// (`rid`/`mi`/`td`), so the aggregator joins it into a `MediaRecord`.
    fn tagged_record() -> ContentRequestRecord {
        serde_json::from_value(serde_json::json!({
            "timestamp_ns": 1_300u64,
            "method": "GET",
            "path": "/content/images/x.png",
            "query_string": "rid=A&mi=0&td=1000",
            "status_code": 200,
            "body_bytes": 128u64,
            "latency_ns": 40u64,
        }))
        .unwrap()
    }

    /// A served content record with no media tag; the aggregator counts it as
    /// unmatched and yields no `MediaRecord`, so no artifact row is written.
    fn untagged_record() -> ContentRequestRecord {
        serde_json::from_value(serde_json::json!({
            "timestamp_ns": 9_999u64,
            "method": "GET",
            "path": "/content/images/y.png",
            "query_string": "garbage=1",
            "status_code": 200,
            "body_bytes": 16u64,
            "latency_ns": 10u64,
        }))
        .unwrap()
    }

    #[test]
    fn ingest_media_record_writes_only_matched_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(MEDIA_RECORDS_FILENAME);
        let mut aggregator = MediaFetchAggregator::new();
        let mut writer = MediaRecordWriter::create(&path).unwrap();

        // An untagged record is ingested (counted unmatched) but must not emit a
        // JSONL row; a tagged record both counts and writes exactly one row.
        ingest_media_record(&mut aggregator, Some(&mut writer), &untagged_record());
        ingest_media_record(&mut aggregator, Some(&mut writer), &tagged_record());
        writer.flush().unwrap();

        let lines = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            lines.lines().count(),
            1,
            "only the tag-matched fetch writes a media_records row"
        );
        let summary = aggregator.finish();
        assert_eq!(summary.total_fetches, 1);
        assert_eq!(summary.unmatched, 1);
    }

    #[test]
    fn ingest_media_record_without_writer_still_aggregates() {
        // A run that publishes files but requests no media_records artifact still
        // folds fetch metrics: ingestion always happens, only the row write is
        // gated on a present writer.
        let mut aggregator = MediaFetchAggregator::new();
        ingest_media_record(&mut aggregator, None, &tagged_record());
        ingest_media_record(&mut aggregator, None, &tagged_record());
        assert_eq!(aggregator.finish().total_fetches, 2);
    }

    fn empty_resources() -> PreparedNativeSidecarResources {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        PreparedNativeSidecarResources {
            real_clock_anchor: RealClockAnchor::now(),
            clock,
            content_server: None,
            gpu_telemetry: None,
            network_latency: None,
            server_metrics: None,
            server_profiler: None,
            live_streaming: None,
            gpu_records_path: None,
            network_latency_records_path: None,
            server_metrics_jsonl_path: None,
            server_metrics_parquet_wire_path: None,
            media_handle: None,
            media_record_sender: None,
        }
    }

    #[tokio::test]
    async fn empty_resources_finalize_media_metrics_is_default_and_idempotent() {
        let mut resources = empty_resources();
        // No media wiring: finalize returns the empty summary and is safe to call
        // more than once (the handle is taken on the first call).
        let first = resources.finalize_media_metrics().await.unwrap();
        assert_eq!(first.total_fetches, 0);
        assert_eq!(first.unmatched, 0);
        let second = resources.finalize_media_metrics().await.unwrap();
        assert_eq!(second.total_fetches, 0);
        assert!(resources.live_sink().is_none());
    }

    #[tokio::test]
    async fn media_queue_overflow_fails_sidecar_finalization() {
        let (queue, _receiver) = tokio::sync::mpsc::channel(1);
        let sender = ContentRecordSender::new(queue);
        sender.try_send(tagged_record()).unwrap();
        assert!(sender.try_send(tagged_record()).is_err());

        let mut resources = empty_resources();
        resources.media_record_sender = Some(sender);
        resources.media_handle = Some(tokio::spawn(async { MediaMetricsSummary::default() }));

        let error = resources
            .finalize_media_metrics()
            .await
            .expect_err("a media queue overflow must prevent a successful metrics report");
        assert!(error.to_string().contains("media record queue overflowed"));
    }

    #[tokio::test]
    async fn empty_resources_shutdown_is_a_noop() {
        // Shutting down a bundle that owns no live sidecars must not panic and must
        // leave every optional resource cleared.
        let mut resources = empty_resources();
        resources.shutdown_run_resources().await;
        assert!(resources.content_server.is_none());
        assert!(resources.gpu_telemetry.is_none());
        assert!(resources.network_latency.is_none());
        assert!(resources.server_metrics.is_none());
        assert!(resources.live_streaming.is_none());
    }
}
