// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected telemetry-source seam and DCGM HTTP implementation.
//!
//! Typed response metadata and DCGM's fetch/decode flow are implemented here.
//! Every successful scrape is retained so cadence observations remain complete
//! when the exporter body is stable between exporter updates.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

use crate::clock::Clock;
use crate::transport::core::Response;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use url::Url;

use crate::gpu_telemetry::model::GpuScrape;
use crate::gpu_telemetry::parser::{DcgmPrometheusDecoder, GpuTelemetryDecoder};

/// Whether a scrape is cadence-driven or a mandatory phase barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuScrapeMode {
    /// Cadence scrape.
    Continuous,
    /// Synchronous phase-boundary scrape.
    Boundary,
}

/// GPU telemetry collection or decoding failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuTelemetryError {
    /// Underlying transport failed before a usable response arrived.
    Transport(String),
    /// Endpoint returned a non-success status.
    HttpStatus(u16),
    /// Successful response did not carry a text body.
    MissingBody,
    /// A telemetry source violated the scrape contract.
    Protocol(String),
    /// Dedicated source worker violated its process, thread, or wire contract.
    Worker(String),
    /// Prometheus exposition was malformed.
    Parse {
        /// One-based source line.
        line: usize,
        /// Parser detail.
        message: String,
    },
    /// Phase boundary timestamps were reversed.
    InvalidBoundary {
        /// Start snapshot timestamp.
        start_ns: i64,
        /// End snapshot timestamp.
        end_ns: i64,
    },
}

impl Display for GpuTelemetryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Transport(message) => {
                write!(formatter, "GPU telemetry transport failed: {message}")
            }
            Self::HttpStatus(status) => {
                write!(formatter, "GPU telemetry endpoint returned HTTP {status}")
            }
            Self::MissingBody => {
                formatter.write_str("GPU telemetry endpoint returned no text body")
            }
            Self::Protocol(message) => {
                write!(
                    formatter,
                    "GPU telemetry source violated its contract: {message}"
                )
            }
            Self::Worker(message) => write!(formatter, "GPU telemetry worker failed: {message}"),
            Self::Parse { line, message } => {
                write!(formatter, "invalid DCGM metrics at line {line}: {message}")
            }
            Self::InvalidBoundary { start_ns, end_ns } => write!(
                formatter,
                "GPU telemetry boundary ends before it starts ({end_ns} < {start_ns})"
            ),
        }
    }
}

impl std::error::Error for GpuTelemetryError {}

/// Object-safe extension point for DCGM, NVML, AMDSMI, or replay sources.
#[async_trait(?Send)]
pub trait GpuTelemetrySource {
    /// Credential-free source identifier used in reports.
    fn endpoint_url(&self) -> &str;

    /// Collects one scrape.
    ///
    /// Sources return a scrape or a typed error. `None` remains reserved for
    /// compatibility with optional source implementations and is not emitted by
    /// the native DCGM, NVML, or AMD SMI sources.
    async fn scrape(&self, mode: GpuScrapeMode) -> Result<Option<GpuScrape>, GpuTelemetryError>;

    /// Releases source-owned process or device resources.
    async fn shutdown(&self) -> Result<(), GpuTelemetryError> {
        Ok(())
    }
}

/// DCGM Prometheus source backed by the shared Clock-injected HTTP transport.
pub struct DcgmTelemetrySource {
    clock: Rc<dyn Clock>,
    transport: Rc<HttpTransport>,
    request: RequestConfig,
    display_url: String,
    decoder: Rc<dyn GpuTelemetryDecoder>,
}

impl DcgmTelemetrySource {
    /// Builds a DCGM source with the native Prometheus decoder.
    pub fn new(
        clock: Rc<dyn Clock>,
        transport: Rc<HttpTransport>,
        endpoint_url: impl Into<String>,
    ) -> Self {
        Self::with_decoder(
            clock,
            transport,
            endpoint_url,
            Rc::new(DcgmPrometheusDecoder::new()),
        )
    }

    /// Builds a DCGM source with an injected decoder implementation.
    pub fn with_decoder(
        clock: Rc<dyn Clock>,
        transport: Rc<HttpTransport>,
        endpoint_url: impl Into<String>,
        decoder: Rc<dyn GpuTelemetryDecoder>,
    ) -> Self {
        let endpoint_url = normalize_metrics_url(endpoint_url.into());
        let display_url = redact_url(&endpoint_url);
        Self {
            clock,
            transport,
            request: RequestConfig::new(endpoint_url),
            display_url,
            decoder,
        }
    }
}

#[async_trait(?Send)]
impl GpuTelemetrySource for DcgmTelemetrySource {
    fn endpoint_url(&self) -> &str {
        &self.display_url
    }

    async fn scrape(&self, _mode: GpuScrapeMode) -> Result<Option<GpuScrape>, GpuTelemetryError> {
        let record = self.transport.get(&self.request).await;
        if let Some(error) = record.error {
            return Err(GpuTelemetryError::Transport(error.message));
        }
        let status = record.status.ok_or(GpuTelemetryError::MissingBody)?;
        if !(200..300).contains(&status) {
            return Err(GpuTelemetryError::HttpStatus(status));
        }
        let body = record
            .responses
            .into_iter()
            .find_map(|response| match response {
                Response::Text(response) => Some(response.text),
                Response::Sse(_) => None,
            })
            .ok_or(GpuTelemetryError::MissingBody)?;
        let timestamp_ns = self.clock.now_ns();
        self.decoder
            .decode(&self.display_url, timestamp_ns, &body)
            .map(Some)
    }
}

fn normalize_metrics_url(mut endpoint_url: String) -> String {
    while endpoint_url.ends_with('/') {
        endpoint_url.pop();
    }
    if !endpoint_url.ends_with("/metrics") {
        endpoint_url.push_str("/metrics");
    }
    endpoint_url
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
    use tokio::task::LocalSet;

    #[test]
    fn endpoint_normalization_and_redaction_are_artifact_safe() {
        assert_eq!(
            normalize_metrics_url("http://host:9400/".to_string()),
            "http://host:9400/metrics"
        );
        assert_eq!(
            redact_url("http://user:secret@host:9400/metrics"),
            "http://host:9400/metrics"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn continuous_scrapes_retain_identical_successful_bodies() {
        LocalSet::new()
            .run_until(async {
                let body = "DCGM_FI_DEV_POWER_USAGE{gpu=\"0\",UUID=\"GPU-a\",modelName=\"H100\"} 250\n";
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let server = tokio::task::spawn_local(async move {
                    for _ in 0..2 {
                        let (mut stream, _) = listener.accept().await.unwrap();
                        let mut request = [0_u8; 1024];
                        let _ = stream.read(&mut request).await.unwrap();
                        let response = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        stream.write_all(response.as_bytes()).await.unwrap();
                        stream.shutdown().await.unwrap();
                    }
                });
                let clock: Rc<dyn Clock> = RealClock::new();
                let transport = Rc::new(HttpTransport::new(clock.clone(), ClientConfig::default()));
                let source = DcgmTelemetrySource::new(
                    clock,
                    transport,
                    format!("http://{address}"),
                );

                let first = source.scrape(GpuScrapeMode::Continuous).await.unwrap();
                let second = source.scrape(GpuScrapeMode::Continuous).await.unwrap();

                assert_eq!(first.unwrap().records.len(), 1);
                assert_eq!(second.unwrap().records.len(), 1);
                server.await.unwrap();
            })
            .await;
    }
}
