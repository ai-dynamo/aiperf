// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected server-metrics HTTP source.
//!
//! JSON rejection, OpenMetrics-to-classic parser fallback, the one-shot
//! `/prometheus/metrics` compatibility path, and terminal incompatibility
//! classification port `src/aiperf/server_metrics/data_collector.py:133-361`
//! plus `common/mixins/base_metrics_collector_mixin.py:441-590`. The runtime
//! drives this source sequentially, so the inherited concurrent auto-disable
//! race and response-hash synchronization are intentionally absent.

use std::cell::RefCell;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

use aiperf_clock::Clock;
use aiperf_prometheus::ExpositionFormat;
use aiperf_transport_http::models::{RequestConfig, Response};
use aiperf_transport_http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use url::Url;

use crate::model::ServerMetricsRecord;
use crate::parser::{MetricsParseError, MetricsTextParser, PrometheusTextParser};

/// Whether one scrape is cadence-driven or a mandatory phase barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMetricsScrapeMode {
    /// Sequential continuous scrape within an active phase.
    Continuous,
    /// Synchronous phase start/end scrape.
    Boundary,
}

/// Successful scrape disposition, keeping an empty exposition distinct from disablement.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMetricsScrapeOutcome {
    /// One supported non-empty Prometheus snapshot.
    Record(ServerMetricsRecord),
    /// A valid empty or unsupported-only exposition; the source remains active.
    Empty,
    /// The source was previously classified as terminally incompatible.
    Disabled,
}

/// One server-metrics fetch or decode failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerMetricsError {
    /// Underlying HTTP transport failed.
    Transport(String),
    /// Endpoint returned a non-success status.
    HttpStatus(u16),
    /// Successful response omitted a text body.
    MissingBody,
    /// Body was structurally not Prometheus exposition.
    Incompatible(String),
    /// Prometheus exposition was malformed.
    Parse {
        /// One-based line, or zero for a body-level error.
        line: usize,
        /// Parser detail.
        message: String,
    },
}

impl ServerMetricsError {
    /// Whether this source is permanently incompatible with Prometheus scraping.
    pub fn is_incompatible(&self) -> bool {
        matches!(self, Self::Incompatible(_))
    }
}

impl Display for ServerMetricsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Transport(message) => {
                write!(formatter, "server metrics transport failed: {message}")
            }
            Self::HttpStatus(status) => {
                write!(formatter, "server metrics endpoint returned HTTP {status}")
            }
            Self::MissingBody => {
                formatter.write_str("server metrics endpoint returned no text body")
            }
            Self::Incompatible(message) => formatter.write_str(message),
            Self::Parse { line, message } if *line == 0 => formatter.write_str(message),
            Self::Parse { line, message } => {
                write!(formatter, "invalid metrics at line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for ServerMetricsError {}

impl From<MetricsParseError> for ServerMetricsError {
    fn from(error: MetricsParseError) -> Self {
        Self::Parse {
            line: error.line,
            message: error.message,
        }
    }
}

/// Object-safe source seam for HTTP, replay, or injected server telemetry.
#[async_trait(?Send)]
pub trait ServerMetricsSource {
    /// Current credential-free source identifier.
    fn endpoint_url(&self) -> String;

    /// Collect and parse one complete snapshot.
    async fn scrape(
        &self,
        mode: ServerMetricsScrapeMode,
    ) -> Result<ServerMetricsScrapeOutcome, ServerMetricsError>;
}

struct SourceState {
    endpoint_url: String,
    display_url: String,
    fallback_attempted: bool,
    disabled: bool,
    last_body: Option<String>,
}

/// Prometheus/OpenMetrics source backed by the shared native HTTP transport.
pub struct PrometheusHttpSource {
    clock: Rc<dyn Clock>,
    transport: Rc<HttpTransport>,
    parser: Rc<dyn MetricsTextParser>,
    state: RefCell<SourceState>,
}

impl PrometheusHttpSource {
    /// Build a source with the native exposition parser.
    pub fn new(
        clock: Rc<dyn Clock>,
        transport: Rc<HttpTransport>,
        endpoint_url: impl Into<String>,
    ) -> Self {
        Self::with_parser(
            clock,
            transport,
            endpoint_url,
            Rc::new(PrometheusTextParser),
        )
    }

    /// Build a source with an injected parser implementation.
    pub fn with_parser(
        clock: Rc<dyn Clock>,
        transport: Rc<HttpTransport>,
        endpoint_url: impl Into<String>,
        parser: Rc<dyn MetricsTextParser>,
    ) -> Self {
        let endpoint_url = normalize_metrics_url(&endpoint_url.into());
        let display_url = redact_url(&endpoint_url);
        Self {
            clock,
            transport,
            parser,
            state: RefCell::new(SourceState {
                endpoint_url,
                display_url,
                fallback_attempted: false,
                disabled: false,
                last_body: None,
            }),
        }
    }

    async fn fetch(
        &self,
        endpoint_url: &str,
        display_url: &str,
    ) -> Result<(Option<ServerMetricsRecord>, String), ServerMetricsError> {
        let record = self
            .transport
            .get(&RequestConfig::new(endpoint_url.to_string()))
            .await;
        if let Some(status) = record.status
            && !(200..300).contains(&status)
        {
            return Err(ServerMetricsError::HttpStatus(status));
        }
        if let Some(error) = record.error {
            return Err(ServerMetricsError::Transport(error.message));
        }
        let status = record.status.ok_or(ServerMetricsError::MissingBody)?;
        debug_assert!((200..300).contains(&status));
        let content_type = record
            .response_headers
            .get("content-type")
            .map(String::as_str);
        let declared_format = declared_exposition_format(content_type).map_err(|error| {
            ServerMetricsError::Incompatible(format!(
                "endpoint {display_url:?} returned an unsupported metrics Content-Type ({error})"
            ))
        })?;
        let body = record
            .responses
            .into_iter()
            .find_map(|response| match response {
                Response::Text(response) => Some(response.text),
                Response::Sse(_) => None,
            })
            .ok_or(ServerMetricsError::MissingBody)?;
        if body.trim().is_empty() {
            return Ok((None, body));
        }
        let metrics = self
            .parse_native_metrics(declared_format, body.as_bytes())
            .map_err(|error| {
            let preview = body.chars().take(200).collect::<String>();
            ServerMetricsError::Incompatible(format!(
                "endpoint did not return valid Prometheus exposition format ({error}); body sample: {preview:?}"
            ))
            })?;
        if metrics.is_empty() {
            return Ok((None, body));
        }
        let timestamp_ns = record.recv_start_ns.unwrap_or_else(|| self.clock.now_ns());
        let endpoint_latency_ns = record
            .end_ns
            .map(|end_ns| (end_ns - record.start_ns).max(0));
        Ok((
            Some(ServerMetricsRecord {
                endpoint_url: display_url.to_string(),
                timestamp_ns,
                endpoint_latency_ns,
                request_sent_ns: Some(record.start_ns),
                first_byte_ns: record.recv_start_ns,
                is_duplicate: false,
                benchmark_phase: None,
                metrics,
            }),
            body,
        ))
    }

    fn parse_native_metrics(
        &self,
        declared_format: ExpositionFormat,
        exact_body: &[u8],
    ) -> Result<std::collections::BTreeMap<String, crate::model::MetricFamily>, MetricsParseError>
    {
        match self.parser.parse_exposition(declared_format, exact_body) {
            Ok(exposition) => self.parser.project_native(&exposition),
            Err(strict_error) if declared_format == ExpositionFormat::OpenMetricsText100 => {
                let compatibility = self
                    .parser
                    .parse_exposition(ExpositionFormat::PrometheusText004, exact_body)
                    .and_then(|exposition| self.parser.project_native(&exposition));
                compatibility.map_err(|fallback_error| MetricsParseError {
                    line: strict_error.line,
                    message: format!(
                        "declared OpenMetrics strict parse failed ({strict_error}); named classic native-compatibility fallback also failed ({fallback_error})"
                    ),
                })
            }
            Err(error) => Err(error),
        }
    }

    fn should_try_fallback(state: &SourceState) -> bool {
        !state.fallback_attempted
            && state.endpoint_url.ends_with("/metrics")
            && !state.endpoint_url.ends_with("/prometheus/metrics")
    }

    async fn fallback(
        &self,
        original_url: &str,
        original_display: &str,
    ) -> Result<(Option<ServerMetricsRecord>, String, String, String), ServerMetricsError> {
        let candidate_url = format!(
            "{}/prometheus/metrics",
            original_url.trim_end_matches("/metrics")
        );
        let candidate_display = redact_url(&candidate_url);
        match self.fetch(&candidate_url, &candidate_display).await {
            Ok((record, body)) => Ok((record, body, candidate_url, candidate_display)),
            Err(error) => Err(ServerMetricsError::Incompatible(format!(
                "Prometheus fallback {candidate_display:?} also failed ({error}); original endpoint {original_display:?} returned non-Prometheus content"
            ))),
        }
    }

    fn commit_response(
        state: &mut SourceState,
        active_url: String,
        active_display: String,
        body: String,
    ) -> bool {
        if state.endpoint_url != active_url {
            state.last_body = None;
        }
        let duplicate = state.last_body.as_ref().is_some_and(|last| last == &body);
        state.endpoint_url = active_url;
        state.display_url = active_display;
        state.last_body = Some(body);
        duplicate
    }
}

#[async_trait(?Send)]
impl ServerMetricsSource for PrometheusHttpSource {
    fn endpoint_url(&self) -> String {
        self.state.borrow().display_url.clone()
    }

    async fn scrape(
        &self,
        _mode: ServerMetricsScrapeMode,
    ) -> Result<ServerMetricsScrapeOutcome, ServerMetricsError> {
        let (endpoint_url, display_url, disabled, should_try_fallback) = {
            let state = self.state.borrow();
            (
                state.endpoint_url.clone(),
                state.display_url.clone(),
                state.disabled,
                Self::should_try_fallback(&state),
            )
        };
        if disabled {
            return Ok(ServerMetricsScrapeOutcome::Disabled);
        }

        let fetched = match self.fetch(&endpoint_url, &display_url).await {
            Ok((record, body)) => Ok((record, body, endpoint_url, display_url)),
            Err(ServerMetricsError::Incompatible(_)) if should_try_fallback => {
                self.state.borrow_mut().fallback_attempted = true;
                self.fallback(&endpoint_url, &display_url).await
            }
            Err(error @ ServerMetricsError::Incompatible(_)) => Err(error),
            Err(error) => return Err(error),
        };

        let (record, body, active_url, active_display) = match fetched {
            Ok(value) => value,
            Err(error) => {
                self.state.borrow_mut().disabled = true;
                return Err(error);
            }
        };
        let mut state = self.state.borrow_mut();
        let duplicate = Self::commit_response(&mut state, active_url, active_display.clone(), body);
        let Some(mut record) = record else {
            return Ok(ServerMetricsScrapeOutcome::Empty);
        };
        record.endpoint_url = active_display;
        record.is_duplicate = duplicate;
        Ok(ServerMetricsScrapeOutcome::Record(record))
    }
}

/// Apply the inherited URL normalization rule exactly once.
pub fn normalize_metrics_url(value: &str) -> String {
    let mut normalized = if value.starts_with("http://") || value.starts_with("https://") {
        value.to_string()
    } else {
        format!("http://{value}")
    };
    while normalized.ends_with('/') {
        normalized.pop();
    }
    if !normalized.ends_with("/metrics") {
        normalized.push_str("/metrics");
    }
    normalized
}

fn redact_url(endpoint_url: &str) -> String {
    let Ok(mut parsed) = Url::parse(endpoint_url) else {
        return endpoint_url.to_string();
    };
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    parsed.to_string()
}

fn declared_exposition_format(
    content_type: Option<&str>,
) -> Result<ExpositionFormat, aiperf_prometheus::ContentTypeError> {
    content_type.map_or(Ok(ExpositionFormat::PrometheusText004), |value| {
        ExpositionFormat::from_content_type(value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use aiperf_clock::RealClock;
    use aiperf_prometheus::Exposition;
    use aiperf_transport_http::config::ClientConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::{JoinHandle, LocalSet};

    struct FixtureResponse {
        path: &'static str,
        status: &'static str,
        content_type: &'static str,
        body: &'static str,
    }

    async fn spawn_fixture_server(responses: Vec<FixtureResponse>) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::task::spawn_local(async move {
            for expected in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut chunk = [0_u8; 1024];
                    let read = stream.read(&mut chunk).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&chunk[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8(request).unwrap();
                assert!(request.starts_with(&format!("GET {} HTTP/1.1", expected.path)));
                let response = format!(
                    "HTTP/1.1 {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    expected.status,
                    expected.content_type,
                    expected.body.len(),
                    expected.body,
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        (format!("http://{address}"), task)
    }

    fn source_for(base_url: &str) -> PrometheusHttpSource {
        let clock: Rc<dyn Clock> = RealClock::new();
        let transport = Rc::new(HttpTransport::new(clock.clone(), ClientConfig::default()));
        PrometheusHttpSource::new(clock, transport, format!("{base_url}/metrics"))
    }

    struct CountingParser {
        calls: Rc<Cell<usize>>,
    }

    impl MetricsTextParser for CountingParser {
        fn parse_exposition(
            &self,
            format: ExpositionFormat,
            exact_body: &[u8],
        ) -> Result<Exposition, MetricsParseError> {
            self.calls.set(self.calls.get() + 1);
            PrometheusTextParser.parse_exposition(format, exact_body)
        }

        fn project_native(
            &self,
            exposition: &Exposition,
        ) -> Result<std::collections::BTreeMap<String, crate::model::MetricFamily>, MetricsParseError>
        {
            PrometheusTextParser.project_native(exposition)
        }
    }

    #[test]
    fn metrics_urls_normalize_and_credentials_never_reach_display() {
        assert_eq!(
            normalize_metrics_url("localhost:9400/"),
            "http://localhost:9400/metrics"
        );
        assert_eq!(
            normalize_metrics_url("http://host/v1/chat/completions"),
            "http://host/v1/chat/completions/metrics"
        );
        assert_eq!(
            redact_url("https://user:secret@host/metrics"),
            "https://host/metrics"
        );
    }

    #[test]
    fn fallback_is_one_shot_and_never_recurses() {
        let state = SourceState {
            endpoint_url: "http://host/metrics".to_string(),
            display_url: "http://host/metrics".to_string(),
            fallback_attempted: false,
            disabled: false,
            last_body: None,
        };
        assert!(PrometheusHttpSource::should_try_fallback(&state));
        let state = SourceState {
            endpoint_url: "http://host/prometheus/metrics".to_string(),
            ..state
        };
        assert!(!PrometheusHttpSource::should_try_fallback(&state));
    }

    #[test]
    fn swapping_to_fallback_resets_body_deduplication() {
        let mut state = SourceState {
            endpoint_url: "http://host/metrics".to_string(),
            display_url: "http://host/metrics".to_string(),
            fallback_attempted: true,
            disabled: false,
            last_body: Some("same body".to_string()),
        };

        assert!(!PrometheusHttpSource::commit_response(
            &mut state,
            "http://host/prometheus/metrics".to_string(),
            "http://host/prometheus/metrics".to_string(),
            "same body".to_string(),
        ));
        assert!(PrometheusHttpSource::commit_response(
            &mut state,
            "http://host/prometheus/metrics".to_string(),
            "http://host/prometheus/metrics".to_string(),
            "same body".to_string(),
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trt_json_routes_once_to_vllm_openmetrics_fallback() {
        LocalSet::new()
            .run_until(async {
                let openmetrics =
                    "# TYPE vllm:prompt_tokens counter\nvllm:prompt_tokens_total 12\n# EOF\n";
                let (base_url, server) = spawn_fixture_server(vec![
                    FixtureResponse {
                        path: "/metrics",
                        status: "200 OK",
                        content_type: "application/json",
                        body: "[{\"iteration_stats\":1}]",
                    },
                    FixtureResponse {
                        path: "/prometheus/metrics",
                        status: "200 OK",
                        content_type: "application/openmetrics-text; version=1.0.0; charset=utf-8",
                        body: openmetrics,
                    },
                ])
                .await;
                let source = source_for(&base_url);

                let outcome = source
                    .scrape(ServerMetricsScrapeMode::Boundary)
                    .await
                    .unwrap();

                let ServerMetricsScrapeOutcome::Record(record) = outcome else {
                    panic!("fallback record")
                };
                assert!(record.endpoint_url.ends_with("/prometheus/metrics"));
                assert_eq!(
                    record.metrics["vllm:prompt_tokens"].metric_type,
                    crate::model::PrometheusMetricType::Counter
                );
                assert!(source.endpoint_url().ends_with("/prometheus/metrics"));
                server.await.unwrap();
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_trt_fallback_auto_disables_without_a_third_request() {
        LocalSet::new()
            .run_until(async {
                let (base_url, server) = spawn_fixture_server(vec![
                    FixtureResponse {
                        path: "/metrics",
                        status: "200 OK",
                        content_type: "application/json",
                        body: "[{\"iteration_stats\":1}]",
                    },
                    FixtureResponse {
                        path: "/prometheus/metrics",
                        status: "404 Not Found",
                        content_type: "text/plain",
                        body: "missing",
                    },
                ])
                .await;
                let source = source_for(&base_url);

                let error = source
                    .scrape(ServerMetricsScrapeMode::Continuous)
                    .await
                    .unwrap_err();
                assert!(error.is_incompatible());
                assert_eq!(
                    source
                        .scrape(ServerMetricsScrapeMode::Continuous)
                        .await
                        .unwrap(),
                    ServerMetricsScrapeOutcome::Disabled
                );
                server.await.unwrap();
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn declared_openmetrics_uses_a_named_classic_native_fallback() {
        LocalSet::new()
            .run_until(async {
                let classic = "# TYPE requests_total counter\nrequests_total 2\n";
                let (base_url, server) = spawn_fixture_server(vec![FixtureResponse {
                    path: "/metrics",
                    status: "200 OK",
                    content_type: "application/openmetrics-text; version=1.0.0; charset=utf-8",
                    body: classic,
                }])
                .await;
                let source = source_for(&base_url);

                let outcome = source
                    .scrape(ServerMetricsScrapeMode::Continuous)
                    .await
                    .unwrap();
                let ServerMetricsScrapeOutcome::Record(record) = outcome else {
                    panic!("compatibility record")
                };
                assert_eq!(
                    record.metrics["requests"].metric_type,
                    crate::model::PrometheusMetricType::Counter
                );
                server.await.unwrap();
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn metric_looking_non_success_body_is_never_parsed() {
        LocalSet::new()
            .run_until(async {
                let (base_url, server) = spawn_fixture_server(vec![FixtureResponse {
                    path: "/metrics",
                    status: "500 Internal Server Error",
                    content_type: "text/plain; version=0.0.4; charset=utf-8",
                    body: "# TYPE should_not_parse gauge\nshould_not_parse 42\n",
                }])
                .await;
                let calls = Rc::new(Cell::new(0));
                let clock: Rc<dyn Clock> = RealClock::new();
                let transport = Rc::new(HttpTransport::new(clock.clone(), ClientConfig::default()));
                let source = PrometheusHttpSource::with_parser(
                    clock,
                    transport,
                    format!("{base_url}/metrics"),
                    Rc::new(CountingParser {
                        calls: calls.clone(),
                    }),
                );

                assert_eq!(
                    source.scrape(ServerMetricsScrapeMode::Continuous).await,
                    Err(ServerMetricsError::HttpStatus(500))
                );
                assert_eq!(calls.get(), 0);
                server.await.unwrap();
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unchanged_bodies_mark_duplicates_without_inventing_unique_updates() {
        LocalSet::new()
            .run_until(async {
                let body = "# TYPE queue gauge\nqueue 3\n";
                let responses = (0..2)
                    .map(|_| FixtureResponse {
                        path: "/metrics",
                        status: "200 OK",
                        content_type: "text/plain; version=0.0.4; charset=utf-8",
                        body,
                    })
                    .collect();
                let (base_url, server) = spawn_fixture_server(responses).await;
                let source = source_for(&base_url);

                let first = source
                    .scrape(ServerMetricsScrapeMode::Continuous)
                    .await
                    .unwrap();
                let second = source
                    .scrape(ServerMetricsScrapeMode::Continuous)
                    .await
                    .unwrap();
                let ServerMetricsScrapeOutcome::Record(first) = first else {
                    panic!("first record")
                };
                let ServerMetricsScrapeOutcome::Record(second) = second else {
                    panic!("second record")
                };
                assert!(!first.is_duplicate);
                assert!(second.is_duplicate);

                let endpoint = first.endpoint_url.clone();
                let mut accumulator = crate::accumulator::ServerMetricsAccumulator::new();
                accumulator.ingest_record(first);
                accumulator.ingest_record(second);
                let info = &accumulator.endpoint_info()[&endpoint];
                assert_eq!(info.total_fetches, 2);
                assert_eq!(info.unique_updates, 1);
                server.await.unwrap();
            })
            .await;
    }
}
