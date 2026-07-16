// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected server-metrics HTTP source.
//!
//! JSON rejection, OpenMetrics-to-classic parser fallback, the one-shot
//! `/prometheus/metrics` compatibility path, and terminal incompatibility
//! classification are implemented here. The runtime
//! drives this source sequentially, so the inherited concurrent auto-disable
//! race and response-hash synchronization are intentionally absent.

use std::cell::RefCell;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

use crate::clock::Clock;
use crate::transport::core::Response;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use url::Url;

use crate::server_metrics::model::ServerMetricsRecord;
use crate::server_metrics::parser::{MetricsParseError, MetricsTextParser, PrometheusTextParser};

const JSON_CONTENT_TYPE_PREFIX: &str = "application/json";
const OPENMETRICS_CONTENT_TYPE_PREFIX: &str = "application/openmetrics-text";

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
        if let Some(error) = record.error {
            return Err(ServerMetricsError::Transport(error.message));
        }
        let status = record.status.ok_or(ServerMetricsError::MissingBody)?;
        if !(200..300).contains(&status) {
            return Err(ServerMetricsError::HttpStatus(status));
        }
        let content_type = record
            .response_headers
            .get("content-type")
            .map(|value| value.to_ascii_lowercase());
        if content_type
            .as_deref()
            .is_some_and(|value| value.starts_with(JSON_CONTENT_TYPE_PREFIX))
        {
            return Err(ServerMetricsError::Incompatible(format!(
                "endpoint {display_url:?} returned non-Prometheus content-type {:?}; expected text/plain",
                content_type.as_deref().unwrap_or_default()
            )));
        }
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
        let metrics = if content_type
            .as_deref()
            .is_some_and(|value| value.starts_with(OPENMETRICS_CONTENT_TYPE_PREFIX))
        {
            self.parser
                .parse_openmetrics(&body)
                .or_else(|_| self.parser.parse_classic(&body))
        } else {
            self.parser.parse_classic(&body)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::RealClock;
    use crate::transport::http::config::ClientConfig;
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
                        content_type: "application/openmetrics-text; version=1.0.0",
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
                    crate::server_metrics::model::PrometheusMetricType::Counter
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
}
