// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable probe source and fresh-TCP implementation.

use std::cell::RefCell;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::rc::Rc;

use crate::clock::Clock;
use tokio::net::TcpStream;

use crate::network_latency::model::{
    NetworkLatencyErrorDetails, NetworkLatencySample, NetworkLatencyTarget,
};

/// Non-`Send` future used by a thread-local probe implementation.
pub type LocalProbeFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Source seam for a single target's network-latency samples.
pub trait NetworkLatencyProbe {
    /// Target identity used for deduplication and summaries.
    fn target(&self) -> &NetworkLatencyTarget;

    /// Resolve and cache one address before timed probes begin.
    ///
    /// Failure is diagnostic only: [`probe_once`](Self::probe_once) falls back
    /// to per-connect resolution.
    fn resolve(&self) -> LocalProbeFuture<'_, io::Result<()>>;

    /// Open one fresh connection and always return one success/failure sample.
    fn probe_once(&self, timeout_ns: i64) -> LocalProbeFuture<'_, NetworkLatencySample>;
}

/// Plain TCP-connect probe using an injected AIPerf clock.
pub struct TcpConnectProbe {
    clock: Rc<dyn Clock>,
    target: NetworkLatencyTarget,
    resolved_address: RefCell<Option<SocketAddr>>,
}

impl TcpConnectProbe {
    /// Build a fresh-connection source for one already parsed target.
    pub fn new(clock: Rc<dyn Clock>, target: NetworkLatencyTarget) -> Self {
        Self {
            clock,
            target,
            resolved_address: RefCell::new(None),
        }
    }

    async fn connect(&self) -> io::Result<TcpStream> {
        let resolved_address = { *self.resolved_address.borrow() };
        if let Some(address) = resolved_address {
            TcpStream::connect(address).await
        } else {
            TcpStream::connect((self.target.target_host.as_str(), self.target.target_port)).await
        }
    }

    fn failed_sample(
        &self,
        timestamp_ns: i64,
        error_type: &str,
        message: String,
        code: Option<i32>,
    ) -> NetworkLatencySample {
        NetworkLatencySample {
            timestamp_ns,
            target_url: self.target.target_url.clone(),
            target_host: self.target.target_host.clone(),
            target_port: self.target.target_port,
            probe_type: "tcp_connect",
            rtt_ns: None,
            success: false,
            error: Some(NetworkLatencyErrorDetails {
                code,
                error_type: Some(error_type.to_string()),
                message,
                cause: None,
                cause_chain: Some(vec![error_type.to_string()]),
            }),
        }
    }
}

impl NetworkLatencyProbe for TcpConnectProbe {
    fn target(&self) -> &NetworkLatencyTarget {
        &self.target
    }

    fn resolve(&self) -> LocalProbeFuture<'_, io::Result<()>> {
        Box::pin(async move {
            let mut addresses = tokio::net::lookup_host((
                self.target.target_host.as_str(),
                self.target.target_port,
            ))
            .await?;
            let address = addresses.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::AddrNotAvailable,
                    "DNS resolution returned no TCP addresses",
                )
            })?;
            *self.resolved_address.borrow_mut() = Some(address);
            Ok(())
        })
    }

    fn probe_once(&self, timeout_ns: i64) -> LocalProbeFuture<'_, NetworkLatencySample> {
        Box::pin(async move {
            let timestamp_ns = self.clock.now_ns();
            if timeout_ns <= 0 {
                return self.failed_sample(
                    timestamp_ns,
                    "TimeoutError",
                    "TCP connect deadline expired before probe issuance".to_string(),
                    None,
                );
            }

            let connect = self.connect();
            let timeout = self.clock.clone().sleep(timeout_ns);
            tokio::pin!(connect);
            tokio::pin!(timeout);
            tokio::select! {
                biased;
                result = &mut connect => match result {
                    Ok(stream) => {
                        let rtt_ns = (self.clock.now_ns() - timestamp_ns).max(0);
                        drop(stream);
                        NetworkLatencySample {
                            timestamp_ns,
                            target_url: self.target.target_url.clone(),
                            target_host: self.target.target_host.clone(),
                            target_port: self.target.target_port,
                            probe_type: "tcp_connect",
                            rtt_ns: Some(rtt_ns),
                            success: true,
                            error: None,
                        }
                    }
                    Err(error) => self.failed_sample(
                        timestamp_ns,
                        io_error_type(error.kind()),
                        error.to_string(),
                        error.raw_os_error(),
                    ),
                },
                () = &mut timeout => self.failed_sample(
                    timestamp_ns,
                    "TimeoutError",
                    format!("TCP connect timed out after {timeout_ns}ns"),
                    None,
                ),
            }
        })
    }
}

const fn io_error_type(kind: io::ErrorKind) -> &'static str {
    match kind {
        io::ErrorKind::ConnectionRefused => "ConnectionRefusedError",
        io::ErrorKind::ConnectionReset => "ConnectionResetError",
        io::ErrorKind::ConnectionAborted => "ConnectionAbortedError",
        io::ErrorKind::PermissionDenied => "PermissionError",
        io::ErrorKind::TimedOut => "TimeoutError",
        _ => "OSError",
    }
}
