// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Clock-injected gRPC transport on Tonic channels.
//!
//! Supports pooled, per-request, and sticky-session channels; normalized
//! metadata; bounded channel readiness; post-submission cancellation; unary,
//! server-streaming, and bidirectional calls; in-band stream errors; and
//! gRPC-to-HTTP status mapping. Deadlines use only [`crate::clock::Clock`].

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::rc::Rc;
use std::task::Poll;
use std::time::Duration;

use bytes::Bytes;
use futures::{FutureExt, select_biased};
use http::uri::PathAndQuery;
use serde_json::Value;
use tonic::client::Grpc;
use tonic::codec::Streaming;
use tonic::metadata::{Ascii, KeyAndValueRef, MetadataKey, MetadataMap, MetadataValue};
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use tonic::{Code, Request, Status};
use url::{Host, Url};

use crate::clock::Clock;

use crate::transport::grpc::binding::GrpcEndpointBinding;
use crate::transport::grpc::models::{
    ConnectionReuseStrategy, GrpcClientConfig, GrpcErrorDetails, GrpcErrorKind, GrpcRequestConfig,
    GrpcRequestRecord, GrpcResponse,
};
use crate::transport::grpc::raw_codec::RawBytesCodec;

/// gRPC setup or dispatch failure before it is recorded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrpcTransportError {
    details: GrpcErrorDetails,
}

impl GrpcTransportError {
    fn new(kind: GrpcErrorKind, message: impl Into<String>, code: u16) -> Self {
        Self {
            details: GrpcErrorDetails {
                kind,
                message: message.into(),
                code,
                grpc_status_code: None,
            },
        }
    }

    fn from_status(status: Status) -> Self {
        let grpc_status_code = code_to_i32(status.code());
        Self {
            details: GrpcErrorDetails {
                kind: GrpcErrorKind::Rpc,
                message: status.message().to_string(),
                code: grpc_status_to_http(status.code()),
                grpc_status_code: Some(grpc_status_code),
            },
        }
    }

    /// Structured failure details.
    pub fn details(&self) -> &GrpcErrorDetails {
        &self.details
    }
}

impl fmt::Display for GrpcTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.details.message)
    }
}

impl Error for GrpcTransportError {}

#[derive(Clone, Debug)]
struct GrpcTarget {
    authority: String,
    tonic_uri: String,
    secure: bool,
}

struct RpcDispatchContext<'a> {
    channel: Channel,
    method: PathAndQuery,
    metadata: MetadataMap,
    body: Bytes,
    binding: &'a dyn GrpcEndpointBinding,
    request: &'a GrpcRequestConfig,
    total_deadline: Option<i64>,
    record: &'a mut GrpcRequestRecord,
}

/// Worker-local native gRPC transport.
//
// `Channel` is Tonic's cheap cloneable multiplexing handle. The surrounding
// maps are local `Rc<RefCell<_>>`, preserving the thread-per-core ownership
// model without application-level locks.
pub struct GrpcTransport {
    clock: Rc<dyn Clock>,
    config: GrpcClientConfig,
    targets: Vec<GrpcTarget>,
    pooled: RefCell<BTreeMap<String, Channel>>,
    sticky: RefCell<BTreeMap<String, Channel>>,
    user_agent: String,
    session_header: Option<String>,
}

impl fmt::Debug for GrpcTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrpcTransport")
            .field(
                "targets",
                &self
                    .targets
                    .iter()
                    .map(|target| &target.authority)
                    .collect::<Vec<_>>(),
            )
            .field("config", &self.config)
            .field("pooled_channels", &self.pooled.borrow().len())
            .field("sticky_channels", &self.sticky.borrow().len())
            .finish()
    }
}

impl GrpcTransport {
    /// Validate configured targets and construct a local transport.
    pub fn new(
        clock: Rc<dyn Clock>,
        config: GrpcClientConfig,
        urls: impl IntoIterator<Item = String>,
    ) -> Result<Self, GrpcTransportError> {
        let targets = urls
            .into_iter()
            .map(|url| parse_target(&url))
            .collect::<Result<Vec<_>, _>>()?;
        if targets.is_empty() {
            return Err(GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                "gRPC transport requires at least one URL",
                400,
            ));
        }
        let secure = targets[0].secure;
        if targets.iter().any(|target| target.secure != secure) {
            return Err(GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                "all gRPC URLs must use the same grpc or grpcs scheme",
                400,
            ));
        }
        if config.max_receive_message_size == 0 || config.max_send_message_size == 0 {
            return Err(GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                "gRPC maximum message sizes must be positive",
                400,
            ));
        }
        if config.channel_ready_timeout_ns <= 0 {
            return Err(GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                "gRPC channel-ready timeout must be positive",
                400,
            ));
        }
        Ok(Self {
            clock,
            config,
            targets,
            pooled: RefCell::new(BTreeMap::new()),
            sticky: RefCell::new(BTreeMap::new()),
            user_agent: "aiperf-transport-grpc/0".to_string(),
            session_header: None,
        })
    }

    /// Override the user-agent metadata value.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Forward the correlation ID under an additional endpoint-specific name.
    pub fn with_session_header(mut self, name: impl Into<String>) -> Self {
        self.session_header = Some(name.into());
        self
    }

    /// Number of connected pooled target channels.
    pub fn pooled_channel_count(&self) -> usize {
        self.pooled.borrow().len()
    }

    /// Number of active sticky-session channel leases.
    pub fn sticky_channel_count(&self) -> usize {
        self.sticky.borrow().len()
    }

    /// Release one sticky-session channel lease.
    pub fn release_sticky_session(&self, correlation_id: &str) {
        self.sticky.borrow_mut().remove(correlation_id);
    }

    /// Release all local channels.
    pub fn close(&self) {
        self.pooled.borrow_mut().clear();
        self.sticky.borrow_mut().clear();
    }

    /// Dispatch canonical endpoint JSON through one prepared gRPC binding.
    pub async fn send_request(
        &self,
        binding: &dyn GrpcEndpointBinding,
        request: &GrpcRequestConfig,
        payload: &Value,
        first_response_filter: &mut dyn FnMut(i64, &Value) -> bool,
    ) -> GrpcRequestRecord {
        let start_ns = self.clock.now_ns();
        let mut record = GrpcRequestRecord::started(start_ns);
        let request_id = request.request_id.as_deref().unwrap_or_default();
        let bidi_method = request
            .streaming
            .then(|| binding.bidi_streaming_method())
            .flatten();
        let messages = match bidi_method {
            Some(_) => binding.encode_bidi_requests(payload, &request.model_name, request_id),
            None => binding
                .encode_request(payload, &request.model_name, request_id)
                .map(|message| vec![message]),
        };
        let messages = match messages {
            Ok(messages) if !messages.is_empty() => messages,
            Ok(_) => {
                finish_error(
                    &mut record,
                    self.clock.now_ns(),
                    GrpcTransportError::new(
                        GrpcErrorKind::InvalidRequest,
                        "encode gRPC request produced no protobuf messages",
                        400,
                    ),
                );
                return record;
            }
            Err(error) => {
                finish_error(
                    &mut record,
                    self.clock.now_ns(),
                    GrpcTransportError::new(
                        GrpcErrorKind::InvalidRequest,
                        format!("encode gRPC request: {error}"),
                        400,
                    ),
                );
                return record;
            }
        };
        if let Some(encoded) = messages
            .iter()
            .find(|message| message.len() > self.config.max_send_message_size)
        {
            finish_error(
                &mut record,
                self.clock.now_ns(),
                GrpcTransportError::new(
                    GrpcErrorKind::InvalidRequest,
                    format!(
                        "gRPC request is {} bytes, exceeding configured {}-byte maximum",
                        encoded.len(),
                        self.config.max_send_message_size
                    ),
                    400,
                ),
            );
            return record;
        }
        record.request_body = messages[0].clone();
        record.request_messages = messages.clone();
        let metadata = match self.build_metadata(request) {
            Ok(metadata) => metadata,
            Err(error) => {
                finish_error(&mut record, self.clock.now_ns(), error);
                return record;
            }
        };
        record.trace.request_metadata = metadata_to_map(&metadata);

        let total_deadline =
            positive_timeout(request.total_timeout_ns.or(self.config.total_timeout_ns))
                .map(|timeout| start_ns.saturating_add(timeout));
        let channel = match self
            .acquire_channel(request, total_deadline, &mut record)
            .await
        {
            Ok(channel) => channel,
            Err(error) => {
                finish_error(&mut record, self.clock.now_ns(), error);
                self.release_after_terminal(request, true);
                return record;
            }
        };

        record.trace.request_send_start_ns = Some(self.clock.now_ns());
        record.trace.request_headers_sent_ns = record.trace.request_send_start_ns;
        record.trace.request_chunks_count = u32::try_from(messages.len()).unwrap_or(u32::MAX);
        record.trace.request_bytes_total = messages.iter().fold(0_u64, |total, message| {
            total.saturating_add(message.len() as u64)
        });
        if self.config.trace_chunks {
            let at_ns = record.trace.request_send_start_ns.unwrap_or(start_ns);
            record
                .trace
                .request_chunks
                .extend(messages.iter().map(|message| (at_ns, message.len() as u64)));
        }

        let result = if request.streaming {
            if let Some(method) = bidi_method {
                self.dispatch_bidi_streaming(
                    channel,
                    method.clone(),
                    metadata,
                    messages,
                    binding,
                    request,
                    total_deadline,
                    &mut record,
                    first_response_filter,
                )
                .await
            } else if let Some(method) = binding.streaming_method() {
                self.dispatch_streaming(
                    RpcDispatchContext {
                        channel,
                        method: method.clone(),
                        metadata,
                        body: messages[0].clone(),
                        binding,
                        request,
                        total_deadline,
                        record: &mut record,
                    },
                    first_response_filter,
                )
                .await
            } else {
                Err(GrpcTransportError::new(
                    GrpcErrorKind::InvalidRequest,
                    format!(
                        "endpoint {} has no gRPC streaming method",
                        binding.endpoint_id()
                    ),
                    400,
                ))
            }
        } else {
            self.dispatch_unary(RpcDispatchContext {
                channel,
                method: binding.unary_method().clone(),
                metadata,
                body: messages[0].clone(),
                binding,
                request,
                total_deadline,
                record: &mut record,
            })
            .await
        };

        if let Err(error) = result {
            finish_error(&mut record, self.clock.now_ns(), error);
        }
        let terminal_failure = record.error.is_some() || record.cancellation_ns.is_some();
        self.release_after_terminal(request, terminal_failure);
        record
    }

    /// Query dialect-defined model readiness over the same native channel substrate.
    pub async fn model_ready(
        &self,
        binding: &dyn GrpcEndpointBinding,
        model_name: &str,
        url_index: Option<u32>,
        metadata: BTreeMap<String, String>,
    ) -> Result<bool, GrpcTransportError> {
        let start_ns = self.clock.now_ns();
        let method = binding.readiness_method().ok_or_else(|| {
            GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                format!(
                    "endpoint {} does not define a gRPC readiness method",
                    binding.endpoint_id()
                ),
                400,
            )
        })?;
        let request = GrpcRequestConfig {
            metadata,
            url_index,
            model_name: model_name.to_string(),
            ..GrpcRequestConfig::new(model_name)
        };
        let mut scratch = GrpcRequestRecord::started(start_ns);
        let total_deadline = positive_timeout(self.config.total_timeout_ns)
            .map(|timeout| start_ns.saturating_add(timeout));
        let channel = self
            .acquire_channel(&request, total_deadline, &mut scratch)
            .await?;
        let request_metadata = self.build_metadata(&request)?;
        let bytes = binding.encode_readiness_request(model_name);
        let response = await_deadline(
            self.clock.clone(),
            total_deadline.map(|at_ns| Deadline {
                at_ns,
                kind: DeadlineKind::Total,
            }),
            raw_unary(
                channel,
                method.clone(),
                request_metadata,
                bytes,
                self.config.max_send_message_size,
                self.config.max_receive_message_size,
            ),
        )
        .await
        .map_err(deadline_error)?
        .map_err(GrpcTransportError::from_status)?;
        binding
            .decode_readiness_response(&response.0)
            .map_err(|error| {
                GrpcTransportError::new(
                    GrpcErrorKind::Decode,
                    format!("decode gRPC readiness response: {error}"),
                    500,
                )
            })
    }

    async fn dispatch_unary(
        &self,
        context: RpcDispatchContext<'_>,
    ) -> Result<(), GrpcTransportError> {
        let RpcDispatchContext {
            channel,
            method,
            metadata,
            body,
            binding,
            request,
            total_deadline,
            record,
        } = context;
        let mut rpc = Box::pin(raw_unary(
            channel,
            method,
            metadata,
            body,
            self.config.max_send_message_size,
            self.config.max_receive_message_size,
        ));
        if let Some(result) = poll_once(rpc.as_mut()).await {
            record.trace.request_send_end_ns = Some(self.clock.now_ns());
            return self.finish_unary(result, binding, record);
        }
        let send_anchor = self.clock.now_ns();
        record.trace.request_send_end_ns = Some(send_anchor);
        let deadline = request_deadline(request, send_anchor, total_deadline);
        let result = await_deadline(self.clock.clone(), deadline, rpc)
            .await
            .map_err(deadline_error)?;
        self.finish_unary(result, binding, record)
    }

    fn finish_unary(
        &self,
        result: Result<(Bytes, MetadataMap), Status>,
        binding: &dyn GrpcEndpointBinding,
        record: &mut GrpcRequestRecord,
    ) -> Result<(), GrpcTransportError> {
        let (bytes, metadata) = result.map_err(GrpcTransportError::from_status)?;
        let perf_ns = self.clock.now_ns();
        let json = binding.decode_response(&bytes).map_err(|error| {
            GrpcTransportError::new(
                GrpcErrorKind::Decode,
                format!("decode gRPC response: {error}"),
                500,
            )
        })?;
        record.responses.push(GrpcResponse {
            perf_ns,
            json,
            wire_size: bytes.len(),
        });
        record.trace.response_metadata = metadata_to_map(&metadata);
        record.trace.response_receive_start_ns = Some(perf_ns);
        record.trace.response_headers_received_ns = Some(perf_ns);
        record.trace.response_receive_end_ns = Some(perf_ns);
        record.trace.response_chunks_count = 1;
        record.trace.response_bytes_total = bytes.len() as u64;
        if self.config.trace_chunks {
            record
                .trace
                .response_chunks
                .push((perf_ns, bytes.len() as u64));
        }
        finish_ok(record, perf_ns);
        Ok(())
    }

    async fn dispatch_streaming(
        &self,
        context: RpcDispatchContext<'_>,
        first_response_filter: &mut dyn FnMut(i64, &Value) -> bool,
    ) -> Result<(), GrpcTransportError> {
        let RpcDispatchContext {
            channel,
            method,
            metadata,
            body,
            binding,
            request,
            total_deadline,
            record,
        } = context;
        let mut rpc = Box::pin(raw_server_streaming(
            channel,
            method,
            metadata,
            body,
            self.config.max_send_message_size,
            self.config.max_receive_message_size,
        ));
        let immediate = poll_once(rpc.as_mut()).await;
        let send_anchor = self.clock.now_ns();
        record.trace.request_send_end_ns = Some(send_anchor);
        let deadline = request_deadline(request, send_anchor, total_deadline);
        let (mut stream, initial_metadata) = match immediate {
            Some(result) => result.map_err(GrpcTransportError::from_status)?,
            None => await_deadline(self.clock.clone(), deadline, rpc)
                .await
                .map_err(deadline_error)?
                .map_err(GrpcTransportError::from_status)?,
        };
        record.trace.response_metadata = metadata_to_map(&initial_metadata);
        record.trace.response_headers_received_ns = Some(self.clock.now_ns());
        let mut first_token_acquired = false;
        loop {
            let next = await_deadline(self.clock.clone(), deadline, stream.message())
                .await
                .map_err(deadline_error)?
                .map_err(GrpcTransportError::from_status)?;
            let Some(bytes) = next else {
                break;
            };
            let perf_ns = self.clock.now_ns();
            let chunk = binding.decode_stream_response(&bytes).map_err(|error| {
                GrpcTransportError::new(
                    GrpcErrorKind::Decode,
                    format!("decode gRPC stream response: {error}"),
                    500,
                )
            })?;
            record.trace.response_chunks_count =
                record.trace.response_chunks_count.saturating_add(1);
            record.trace.response_bytes_total = record
                .trace
                .response_bytes_total
                .saturating_add(chunk.response_size as u64);
            if self.config.trace_chunks {
                record
                    .trace
                    .response_chunks
                    .push((perf_ns, chunk.response_size as u64));
            }
            if let Some(error_message) = chunk.error_message {
                return Err(GrpcTransportError::new(
                    GrpcErrorKind::Stream,
                    error_message,
                    500,
                ));
            }
            let Some(json) = chunk.response else {
                continue;
            };
            record
                .trace
                .response_receive_start_ns
                .get_or_insert(perf_ns);
            if !first_token_acquired {
                first_token_acquired = first_response_filter(perf_ns - record.start_ns, &json);
            }
            record.responses.push(GrpcResponse {
                perf_ns,
                json,
                wire_size: chunk.response_size,
            });
        }
        let end_ns = self.clock.now_ns();
        record.trace.response_receive_end_ns = Some(end_ns);
        if let Ok(Some(trailers)) = stream.trailers().await {
            record
                .trace
                .response_metadata
                .extend(metadata_to_map(&trailers));
        }
        finish_ok(record, end_ns);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_bidi_streaming(
        &self,
        channel: Channel,
        method: PathAndQuery,
        metadata: MetadataMap,
        messages: Vec<Bytes>,
        binding: &dyn GrpcEndpointBinding,
        request: &GrpcRequestConfig,
        total_deadline: Option<i64>,
        record: &mut GrpcRequestRecord,
        first_response_filter: &mut dyn FnMut(i64, &Value) -> bool,
    ) -> Result<(), GrpcTransportError> {
        let mut rpc = Box::pin(raw_bidi_streaming(
            channel,
            method,
            metadata,
            messages,
            self.config.max_send_message_size,
            self.config.max_receive_message_size,
        ));
        let immediate = poll_once(rpc.as_mut()).await;
        let send_anchor = self.clock.now_ns();
        record.trace.request_send_end_ns = Some(send_anchor);
        let deadline = request_deadline(request, send_anchor, total_deadline);
        let (mut stream, initial_metadata) = match immediate {
            Some(result) => result.map_err(GrpcTransportError::from_status)?,
            None => await_deadline(self.clock.clone(), deadline, rpc)
                .await
                .map_err(deadline_error)?
                .map_err(GrpcTransportError::from_status)?,
        };
        record.trace.response_metadata = metadata_to_map(&initial_metadata);
        record.trace.response_headers_received_ns = Some(self.clock.now_ns());
        let mut first_response_acquired = false;
        loop {
            let next = await_deadline(self.clock.clone(), deadline, stream.message())
                .await
                .map_err(deadline_error)?
                .map_err(GrpcTransportError::from_status)?;
            let Some(bytes) = next else {
                break;
            };
            let perf_ns = self.clock.now_ns();
            let chunk = binding.decode_stream_response(&bytes).map_err(|error| {
                GrpcTransportError::new(
                    GrpcErrorKind::Decode,
                    format!("decode gRPC bidirectional response: {error}"),
                    500,
                )
            })?;
            record.trace.response_chunks_count =
                record.trace.response_chunks_count.saturating_add(1);
            record.trace.response_bytes_total = record
                .trace
                .response_bytes_total
                .saturating_add(chunk.response_size as u64);
            if self.config.trace_chunks {
                record
                    .trace
                    .response_chunks
                    .push((perf_ns, chunk.response_size as u64));
            }
            if let Some(error_message) = chunk.error_message {
                return Err(GrpcTransportError::new(
                    GrpcErrorKind::Stream,
                    error_message,
                    500,
                ));
            }
            let Some(json) = chunk.response else {
                continue;
            };
            record
                .trace
                .response_receive_start_ns
                .get_or_insert(perf_ns);
            if !first_response_acquired {
                first_response_acquired = first_response_filter(perf_ns - record.start_ns, &json);
            }
            record.responses.push(GrpcResponse {
                perf_ns,
                json,
                wire_size: chunk.response_size,
            });
        }
        let end_ns = self.clock.now_ns();
        record.trace.response_receive_end_ns = Some(end_ns);
        if let Ok(Some(trailers)) = stream.trailers().await {
            record
                .trace
                .response_metadata
                .extend(metadata_to_map(&trailers));
        }
        finish_ok(record, end_ns);
        Ok(())
    }

    async fn acquire_channel(
        &self,
        request: &GrpcRequestConfig,
        total_deadline: Option<i64>,
        record: &mut GrpcRequestRecord,
    ) -> Result<Channel, GrpcTransportError> {
        let target = self.selected_target(request.url_index)?.clone();
        match request.reuse {
            ConnectionReuseStrategy::Pooled => {
                if let Some(channel) = self.pooled.borrow().get(&target.authority).cloned() {
                    record.trace.channel_reused_ns = Some(self.clock.now_ns());
                    return Ok(channel);
                }
                let channel = self.connect_target(&target, total_deadline, record).await?;
                self.pooled
                    .borrow_mut()
                    .insert(target.authority.clone(), channel.clone());
                Ok(channel)
            }
            ConnectionReuseStrategy::Never => {
                self.connect_target(&target, total_deadline, record).await
            }
            ConnectionReuseStrategy::StickyUserSessions => {
                let correlation_id = request.correlation_id.as_deref().ok_or_else(|| {
                    GrpcTransportError::new(
                        GrpcErrorKind::InvalidRequest,
                        "sticky-user-sessions gRPC reuse requires a correlation ID",
                        400,
                    )
                })?;
                if let Some(channel) = self.sticky.borrow().get(correlation_id).cloned() {
                    record.trace.channel_reused_ns = Some(self.clock.now_ns());
                    return Ok(channel);
                }
                let channel = self.connect_target(&target, total_deadline, record).await?;
                self.sticky
                    .borrow_mut()
                    .insert(correlation_id.to_string(), channel.clone());
                Ok(channel)
            }
        }
    }

    async fn connect_target(
        &self,
        target: &GrpcTarget,
        total_deadline: Option<i64>,
        record: &mut GrpcRequestRecord,
    ) -> Result<Channel, GrpcTransportError> {
        let mut endpoint = Endpoint::from_shared(target.tonic_uri.clone()).map_err(|error| {
            GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                format!("invalid gRPC target URI: {error}"),
                400,
            )
        })?;
        endpoint = endpoint
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true);
        if target.secure {
            // `ssl_verify=false` installs a danger verifier that accepts any
            // certificate (self-signed / untrusted `grpcs` test servers),
            // sharing the exact rustls verifier the HTTP transport uses. tonic
            // rejects mixing roots with a custom verifier, so the verified path
            // sets roots and the insecure path sets neither.
            let configured = if self.config.ssl_verify {
                endpoint.tls_config(ClientTlsConfig::new().with_enabled_roots())
            } else {
                endpoint.tls_config_with_verifier(
                    ClientTlsConfig::new(),
                    crate::transport::http::client::connection::insecure_server_cert_verifier(),
                )
            };
            endpoint = configured.map_err(|error| {
                GrpcTransportError::new(
                    GrpcErrorKind::InvalidRequest,
                    format!("configure gRPC TLS: {error}"),
                    400,
                )
            })?;
        }
        let start_ns = self.clock.now_ns();
        record.trace.connect_start_ns = Some(start_ns);
        let ready_deadline = start_ns.saturating_add(self.config.channel_ready_timeout_ns);
        let (at_ns, kind) = match total_deadline {
            Some(total) if total < ready_deadline => (total, DeadlineKind::Total),
            _ => (ready_deadline, DeadlineKind::ChannelReady),
        };
        let channel = await_deadline(
            self.clock.clone(),
            Some(Deadline { at_ns, kind }),
            endpoint.connect(),
        )
        .await
        .map_err(deadline_error)?
        .map_err(|error| {
            GrpcTransportError::new(
                GrpcErrorKind::Other,
                format!("connect gRPC target {}: {error}", target.authority),
                503,
            )
        })?;
        record.trace.connect_end_ns = Some(self.clock.now_ns());
        Ok(channel)
    }

    fn selected_target(&self, url_index: Option<u32>) -> Result<&GrpcTarget, GrpcTransportError> {
        let index = url_index.unwrap_or(0) as usize;
        self.targets.get(index).ok_or_else(|| {
            GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                format!(
                    "gRPC URL index {index} is out of range for {} configured targets",
                    self.targets.len()
                ),
                400,
            )
        })
    }

    fn build_metadata(
        &self,
        request: &GrpcRequestConfig,
    ) -> Result<MetadataMap, GrpcTransportError> {
        let mut values = request
            .metadata
            .iter()
            .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        values.insert("user-agent".to_string(), self.user_agent.clone());
        if let Some(request_id) = &request.request_id {
            values.insert("x-request-id".to_string(), request_id.clone());
        }
        if let Some(correlation_id) = &request.correlation_id {
            values.insert("x-correlation-id".to_string(), correlation_id.clone());
            if let Some(session_header) = &self.session_header {
                values.insert(session_header.to_ascii_lowercase(), correlation_id.clone());
            }
        }
        let mut metadata = MetadataMap::with_capacity(values.len());
        for (name, value) in values {
            if name.ends_with("-bin") {
                return Err(GrpcTransportError::new(
                    GrpcErrorKind::InvalidRequest,
                    format!("binary gRPC metadata {name:?} requires byte-valued configuration"),
                    400,
                ));
            }
            let key = MetadataKey::<Ascii>::from_bytes(name.as_bytes()).map_err(|error| {
                GrpcTransportError::new(
                    GrpcErrorKind::InvalidRequest,
                    format!("invalid gRPC metadata key {name:?}: {error}"),
                    400,
                )
            })?;
            let value = MetadataValue::<Ascii>::try_from(value.as_str()).map_err(|error| {
                GrpcTransportError::new(
                    GrpcErrorKind::InvalidRequest,
                    format!("invalid gRPC metadata value for {name:?}: {error}"),
                    400,
                )
            })?;
            metadata.insert(key, value);
        }
        Ok(metadata)
    }

    fn release_after_terminal(&self, request: &GrpcRequestConfig, failed: bool) {
        if request.reuse == ConnectionReuseStrategy::StickyUserSessions
            && (request.is_final_turn || failed)
            && let Some(correlation_id) = request.correlation_id.as_deref()
        {
            self.release_sticky_session(correlation_id);
        }
    }
}

async fn raw_unary(
    channel: Channel,
    method: PathAndQuery,
    metadata: MetadataMap,
    body: Bytes,
    max_send_message_size: usize,
    max_receive_message_size: usize,
) -> Result<(Bytes, MetadataMap), Status> {
    let mut grpc = Grpc::new(channel)
        .max_encoding_message_size(max_send_message_size)
        .max_decoding_message_size(max_receive_message_size);
    grpc.ready()
        .await
        .map_err(|error| Status::unavailable(format!("gRPC channel is not ready: {error}")))?;
    let mut request = Request::new(body);
    *request.metadata_mut() = metadata;
    let response = grpc.unary(request, method, RawBytesCodec).await?;
    let metadata = response.metadata().clone();
    Ok((response.into_inner(), metadata))
}

async fn raw_server_streaming(
    channel: Channel,
    method: PathAndQuery,
    metadata: MetadataMap,
    body: Bytes,
    max_send_message_size: usize,
    max_receive_message_size: usize,
) -> Result<(Streaming<Bytes>, MetadataMap), Status> {
    let mut grpc = Grpc::new(channel)
        .max_encoding_message_size(max_send_message_size)
        .max_decoding_message_size(max_receive_message_size);
    grpc.ready()
        .await
        .map_err(|error| Status::unavailable(format!("gRPC channel is not ready: {error}")))?;
    let mut request = Request::new(body);
    *request.metadata_mut() = metadata;
    let response = grpc
        .server_streaming(request, method, RawBytesCodec)
        .await?;
    let metadata = response.metadata().clone();
    Ok((response.into_inner(), metadata))
}

async fn raw_bidi_streaming(
    channel: Channel,
    method: PathAndQuery,
    metadata: MetadataMap,
    messages: Vec<Bytes>,
    max_send_message_size: usize,
    max_receive_message_size: usize,
) -> Result<(Streaming<Bytes>, MetadataMap), Status> {
    let mut grpc = Grpc::new(channel)
        .max_encoding_message_size(max_send_message_size)
        .max_decoding_message_size(max_receive_message_size);
    grpc.ready()
        .await
        .map_err(|error| Status::unavailable(format!("gRPC channel is not ready: {error}")))?;
    let mut request = Request::new(tokio_stream::iter(messages));
    *request.metadata_mut() = metadata;
    let response = grpc.streaming(request, method, RawBytesCodec).await?;
    let metadata = response.metadata().clone();
    Ok((response.into_inner(), metadata))
}

async fn poll_once<F>(mut future: Pin<&mut F>) -> Option<F::Output>
where
    F: Future + ?Sized,
{
    poll_fn(|context| {
        Poll::Ready(match future.as_mut().poll(context) {
            Poll::Ready(output) => Some(output),
            Poll::Pending => None,
        })
    })
    .await
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeadlineKind {
    ChannelReady,
    Cancellation,
    Total,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Deadline {
    at_ns: i64,
    kind: DeadlineKind,
}

fn request_deadline(
    request: &GrpcRequestConfig,
    send_anchor: i64,
    total_deadline: Option<i64>,
) -> Option<Deadline> {
    let cancellation = request.cancel_after_ns.map(|delay| Deadline {
        at_ns: send_anchor.saturating_add(delay),
        kind: DeadlineKind::Cancellation,
    });
    match (cancellation, total_deadline) {
        (Some(cancellation), Some(total)) if cancellation.at_ns <= total => Some(cancellation),
        (Some(_), Some(total)) => Some(Deadline {
            at_ns: total,
            kind: DeadlineKind::Total,
        }),
        (Some(cancellation), None) => Some(cancellation),
        (None, Some(total)) => Some(Deadline {
            at_ns: total,
            kind: DeadlineKind::Total,
        }),
        (None, None) => None,
    }
}

async fn await_deadline<F>(
    clock: Rc<dyn Clock>,
    deadline: Option<Deadline>,
    future: F,
) -> Result<F::Output, DeadlineKind>
where
    F: Future,
{
    let Some(deadline) = deadline else {
        return Ok(future.await);
    };
    let remaining = deadline.at_ns.saturating_sub(clock.now_ns());
    if remaining <= 0 {
        clock.clone().sleep(0).await;
        return Err(deadline.kind);
    }
    let sleeper = clock.clone().sleep(remaining).fuse();
    let future = future.fuse();
    futures::pin_mut!(sleeper, future);
    select_biased! {
        _ = sleeper => Err(deadline.kind),
        output = future => Ok(output),
    }
}

fn deadline_error(kind: DeadlineKind) -> GrpcTransportError {
    match kind {
        DeadlineKind::ChannelReady => GrpcTransportError::new(
            GrpcErrorKind::RequestSendTimeout,
            "timed out waiting for gRPC channel to be ready",
            0,
        ),
        DeadlineKind::Cancellation => GrpcTransportError::new(
            GrpcErrorKind::RequestCancellation,
            "gRPC request cancelled after it was sent",
            499,
        ),
        DeadlineKind::Total => GrpcTransportError::new(
            GrpcErrorKind::RequestTimeout,
            "gRPC request exceeded its whole-request timeout",
            504,
        ),
    }
}

fn finish_ok(record: &mut GrpcRequestRecord, end_ns: i64) {
    record.end_ns = Some(end_ns);
    record.status = Some(200);
    record.trace.response_status_code = Some(200);
    record.trace.response_reason = Some("OK".to_string());
    record.trace.grpc_status_code = Some(0);
    record.trace.grpc_status_message = None;
}

fn finish_error(record: &mut GrpcRequestRecord, end_ns: i64, error: GrpcTransportError) {
    if error.details.kind == GrpcErrorKind::RequestCancellation {
        record.cancellation_ns = Some(end_ns);
    }
    if record.trace.response_receive_start_ns.is_some() {
        record.trace.response_receive_end_ns.get_or_insert(end_ns);
    }
    record.end_ns = Some(end_ns);
    record.status = (error.details.code != 0).then_some(error.details.code);
    record.trace.response_status_code = record.status;
    record.trace.error_timestamp_ns = Some(end_ns);
    record.trace.grpc_status_code = error.details.grpc_status_code;
    record.trace.grpc_status_message = Some(error.details.message.clone());
    record.trace.response_reason = Some(match error.details.kind {
        GrpcErrorKind::RequestCancellation => "CANCELLED".to_string(),
        GrpcErrorKind::RequestTimeout => "DEADLINE_EXCEEDED".to_string(),
        GrpcErrorKind::RequestSendTimeout => "CHANNEL_NOT_READY".to_string(),
        GrpcErrorKind::Rpc => error
            .details
            .grpc_status_code
            .map(grpc_code_name)
            .unwrap_or("UNKNOWN")
            .to_string(),
        GrpcErrorKind::Stream => "STREAM_ERROR".to_string(),
        _ => "ERROR".to_string(),
    });
    record.error = Some(error.details);
}

fn metadata_to_map(metadata: &MetadataMap) -> BTreeMap<String, String> {
    metadata
        .iter()
        .map(|entry| match entry {
            KeyAndValueRef::Ascii(key, value) => (
                key.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            ),
            KeyAndValueRef::Binary(key, value) => (
                key.as_str().to_string(),
                String::from_utf8_lossy(value.as_encoded_bytes()).into_owned(),
            ),
        })
        .collect()
}

fn parse_target(raw: &str) -> Result<GrpcTarget, GrpcTransportError> {
    let parsed = Url::parse(raw).map_err(|error| {
        GrpcTransportError::new(
            GrpcErrorKind::InvalidRequest,
            format!("invalid gRPC URL {raw:?}: {error}"),
            400,
        )
    })?;
    let secure = match parsed.scheme() {
        "grpc" => false,
        "grpcs" => true,
        scheme => {
            return Err(GrpcTransportError::new(
                GrpcErrorKind::InvalidRequest,
                format!("gRPC URL {raw:?} has unsupported scheme {scheme:?}"),
                400,
            ));
        }
    };
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(GrpcTransportError::new(
            GrpcErrorKind::InvalidRequest,
            "gRPC URLs must not contain user information",
            400,
        ));
    }
    let host = parsed.host().ok_or_else(|| {
        GrpcTransportError::new(
            GrpcErrorKind::InvalidRequest,
            format!("gRPC URL {raw:?} is missing a host"),
            400,
        )
    })?;
    let host = match host {
        Host::Domain(domain) => domain.to_string(),
        Host::Ipv4(address) => address.to_string(),
        Host::Ipv6(address) => format!("[{address}]"),
    };
    let port = parsed.port().unwrap_or(if secure { 443 } else { 80 });
    let authority = format!("{host}:{port}");
    Ok(GrpcTarget {
        tonic_uri: format!("{}://{authority}", if secure { "https" } else { "http" }),
        authority,
        secure,
    })
}

fn positive_timeout(timeout: Option<i64>) -> Option<i64> {
    timeout.filter(|timeout| *timeout > 0)
}

/// Convert a native gRPC status to the HTTP-equivalent status used by shared metrics.
pub const fn grpc_status_to_http(code: Code) -> u16 {
    match code {
        Code::Ok => 200,
        Code::Cancelled => 499,
        Code::Unknown => 500,
        Code::InvalidArgument => 400,
        Code::DeadlineExceeded => 504,
        Code::NotFound => 404,
        Code::AlreadyExists => 409,
        Code::PermissionDenied => 403,
        Code::ResourceExhausted => 429,
        Code::FailedPrecondition => 400,
        Code::Aborted => 409,
        Code::OutOfRange => 400,
        Code::Unimplemented => 501,
        Code::Internal => 500,
        Code::Unavailable => 503,
        Code::DataLoss => 500,
        Code::Unauthenticated => 401,
    }
}

const fn code_to_i32(code: Code) -> i32 {
    match code {
        Code::Ok => 0,
        Code::Cancelled => 1,
        Code::Unknown => 2,
        Code::InvalidArgument => 3,
        Code::DeadlineExceeded => 4,
        Code::NotFound => 5,
        Code::AlreadyExists => 6,
        Code::PermissionDenied => 7,
        Code::ResourceExhausted => 8,
        Code::FailedPrecondition => 9,
        Code::Aborted => 10,
        Code::OutOfRange => 11,
        Code::Unimplemented => 12,
        Code::Internal => 13,
        Code::Unavailable => 14,
        Code::DataLoss => 15,
        Code::Unauthenticated => 16,
    }
}

const fn grpc_code_name(code: i32) -> &'static str {
    match code {
        0 => "OK",
        1 => "CANCELLED",
        2 => "UNKNOWN",
        3 => "INVALID_ARGUMENT",
        4 => "DEADLINE_EXCEEDED",
        5 => "NOT_FOUND",
        6 => "ALREADY_EXISTS",
        7 => "PERMISSION_DENIED",
        8 => "RESOURCE_EXHAUSTED",
        9 => "FAILED_PRECONDITION",
        10 => "ABORTED",
        11 => "OUT_OF_RANGE",
        12 => "UNIMPLEMENTED",
        13 => "INTERNAL",
        14 => "UNAVAILABLE",
        15 => "DATA_LOSS",
        16 => "UNAUTHENTICATED",
        _ => "UNKNOWN",
    }
}
