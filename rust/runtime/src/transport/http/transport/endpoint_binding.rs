// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP binding for transport-agnostic endpoint payloads.
//!
//! Endpoint implementations own decoded request and response semantics. This
//! module lowers those decoded values into HTTP URL/body/lifecycle policy and
//! decodes HTTP response frames back into [`ServerResponse`]. A future gRPC or
//! WebSocket transport supplies its own binding without forking the endpoint.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::task::{Context, Poll};

use crate::clock::Clock;
use crate::endpoints::{
    EndpointConfig, EndpointDescriptor, PreparedEndpoint, RawEndpointConfig, RequestContentType,
    ServerResponse,
};
use bytes::Bytes;
use serde_json::Value;

use crate::transport::core::{ConnectionReuseStrategy, ErrorDetails, RequestRecord, Response};
use crate::transport::http::client::http_client::SseMessageFilter;
use crate::transport::http::models::{RequestConfig, SseMessage};
use crate::transport::http::transport::body::{
    JsonBodyEncoder, MultipartBodyEncoder, RequestBodyEncoder,
};
use crate::transport::http::transport::http_transport::HttpTransport;
use crate::transport::http::transport::inline_media::{
    HttpMediaFetcher, ImageDataUrlEncoder, inline_image_urls,
};
use crate::transport::http::transport::polling::{
    JsonVideoPollingProtocol, PollingOptions, submit_and_poll,
};
use crate::transport::http::transport::url::build_url;

/// Failure while binding a decoded endpoint request to HTTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpEndpointBindingError {
    message: String,
}

impl HttpEndpointBindingError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for HttpEndpointBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HttpEndpointBindingError {}

impl From<ErrorDetails> for HttpEndpointBindingError {
    fn from(error: ErrorDetails) -> Self {
        Self::new(error.message)
    }
}

/// Canonical endpoint request plus HTTP dispatch metadata.
#[derive(Debug, Clone)]
pub struct HttpEndpointRequest {
    /// Decoded endpoint body serialized as canonical JSON bytes.
    pub body: Bytes,
    /// Endpoint- and request-owned HTTP headers.
    pub headers: BTreeMap<String, String>,
    /// URL query parameters.
    pub parameters: BTreeMap<String, String>,
    /// Authored endpoint path or absolute URL override.
    pub endpoint_path: Option<String>,
    /// Whether the response is streamed.
    pub streaming: bool,
    /// Correlated session identifier used by sticky connection reuse.
    pub correlation_id: Option<String>,
    /// Stable request identifier forwarded on the wire.
    pub request_id: Option<String>,
    /// Whether this is the final turn in a correlated session.
    pub is_final_turn: bool,
    /// Cancellation delay armed after the request body is sent.
    pub cancel_after_ns: Option<i64>,
    /// Selected URL index.
    pub url_index: Option<u32>,
    /// HTTP connection reuse policy.
    pub reuse: ConnectionReuseStrategy,
}

/// HTTP policy selected by an endpoint–transport binding for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpEndpointPolicy {
    /// Fully resolved target URL.
    pub url: String,
    /// Wire content type and body encoder selection.
    pub content_type: RequestContentType,
    /// Whether remote media URLs must be fetched and inlined before encoding.
    pub inline_media: bool,
    /// Submit/poll/download lifecycle policy, when required.
    pub polling: Option<PollingOptions>,
}

/// HTTP-ready request produced by an [`HttpEndpointBinding`].
#[derive(Debug, Clone)]
pub struct PreparedHttpEndpointRequest {
    canonical_body: Bytes,
    wire_body: Bytes,
    request_config: RequestConfig,
    streaming: bool,
    polling: Option<PollingOptions>,
}

/// Backpressured consumer for one endpoint-decoded HTTP/SSE response frame.
pub trait HttpEndpointResponseFilter {
    /// Reserve downstream capacity for the next decoded frame.
    fn poll_ready(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<(), HttpEndpointBindingError>>;

    /// Observe one ready frame and report whether first-token search is done.
    fn start_send(
        &mut self,
        ttft_ns: i64,
        response: &ServerResponse,
    ) -> Result<bool, HttpEndpointBindingError>;
}

struct BindingSseMessageFilter<'a> {
    binding: &'a dyn HttpEndpointBinding,
    responses: &'a mut dyn HttpEndpointResponseFilter,
}

impl SseMessageFilter for BindingSseMessageFilter<'_> {
    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>> {
        self.responses
            .poll_ready(context)
            .map(|result| result.map_err(|error| ErrorDetails::other(error.to_string())))
    }

    fn start_send(&mut self, ttft_ns: i64, message: &SseMessage) -> Result<bool, ErrorDetails> {
        let Some(response) = self.binding.decode_sse_response(message) else {
            return Ok(false);
        };
        self.responses
            .start_send(ttft_ns, &response)
            .map_err(|error| ErrorDetails::other(error.to_string()))
    }
}

impl PreparedHttpEndpointRequest {
    /// Return the endpoint's canonical JSON bytes before HTTP-specific lowering.
    pub fn canonical_body(&self) -> &Bytes {
        &self.canonical_body
    }

    /// Return the resolved HTTP request configuration.
    pub fn request_config(&self) -> &RequestConfig {
        &self.request_config
    }

    /// Dispatch the prepared HTTP request and decode streaming frames through
    /// the binding before evaluating first-token significance.
    pub async fn dispatch(
        &self,
        transport: &HttpTransport,
        clock: Rc<dyn Clock>,
        binding: &dyn HttpEndpointBinding,
        first_response_filter: &mut dyn FnMut(i64, &ServerResponse) -> bool,
    ) -> RequestRecord {
        if let Some(options) = self.polling {
            return submit_and_poll(
                transport,
                clock,
                &self.request_config,
                self.wire_body.clone(),
                options,
                &JsonVideoPollingProtocol,
            )
            .await
            .record;
        }

        transport
            .send_request_bytes_with_first_token_filter(
                &self.request_config,
                self.wire_body.clone(),
                self.streaming,
                |ttft_ns, message| {
                    binding
                        .decode_sse_response(message)
                        .as_ref()
                        .is_some_and(|response| first_response_filter(ttft_ns, response))
                },
            )
            .await
    }

    /// Dispatch while awaiting bounded capacity for every decoded response.
    pub async fn dispatch_backpressured(
        &self,
        transport: &HttpTransport,
        clock: Rc<dyn Clock>,
        binding: &dyn HttpEndpointBinding,
        first_response_filter: &mut dyn HttpEndpointResponseFilter,
    ) -> RequestRecord {
        if let Some(options) = self.polling {
            return submit_and_poll(
                transport,
                clock,
                &self.request_config,
                self.wire_body.clone(),
                options,
                &JsonVideoPollingProtocol,
            )
            .await
            .record;
        }

        let mut filter = BindingSseMessageFilter {
            binding,
            responses: first_response_filter,
        };
        transport
            .send_request_bytes_with_sse_filter(
                &self.request_config,
                self.wire_body.clone(),
                self.streaming,
                &mut filter,
            )
            .await
    }
}

/// Object-safe translation seam between one endpoint dialect and HTTP.
///
/// Alternate HTTP bindings may supply a different body codec or response
/// envelope while preserving the same [`Endpoint`] implementation.
pub trait HttpEndpointBinding: fmt::Debug {
    /// Return the stable endpoint identity used in diagnostics.
    fn endpoint_id(&self) -> &str;

    /// Select URL, body, and lifecycle policy for one canonical request.
    fn request_policy(
        &self,
        endpoint_path: Option<&str>,
        streaming: bool,
        url_index: Option<u32>,
    ) -> Result<HttpEndpointPolicy, HttpEndpointBindingError>;

    /// Encode a decoded payload using the selected HTTP content type.
    fn encode_body(
        &self,
        payload: &Value,
        content_type: RequestContentType,
    ) -> Result<crate::transport::http::transport::body::EncodedRequestBody, HttpEndpointBindingError>
    {
        match content_type {
            RequestContentType::ApplicationJson => JsonBodyEncoder.encode(payload),
            RequestContentType::MultipartFormData => MultipartBodyEncoder.encode(payload),
        }
        .map_err(Into::into)
    }

    /// Decode one complete HTTP response into the endpoint-facing shape.
    fn decode_response(&self, response: &Response) -> Option<ServerResponse>;

    /// Decode one SSE frame into the endpoint-facing shape.
    fn decode_sse_response(&self, message: &SseMessage) -> Option<ServerResponse>;
}

/// Lower one endpoint request through a concrete HTTP binding.
///
/// This async operation is generic over the binding so the per-request hot path
/// remains monomorphized and does not allocate an `async_trait` future.
pub async fn prepare_request<B>(
    binding: &B,
    transport: &HttpTransport,
    request: HttpEndpointRequest,
) -> Result<PreparedHttpEndpointRequest, HttpEndpointBindingError>
where
    B: HttpEndpointBinding + ?Sized,
{
    let HttpEndpointRequest {
        body,
        mut headers,
        parameters,
        endpoint_path,
        streaming,
        correlation_id,
        request_id,
        is_final_turn,
        cancel_after_ns,
        url_index,
        reuse,
    } = request;
    let policy = binding.request_policy(endpoint_path.as_deref(), streaming, url_index)?;
    let canonical_body = body.clone();
    let wire_body = if policy.inline_media
        || matches!(policy.content_type, RequestContentType::MultipartFormData)
    {
        let mut payload = serde_json::from_slice::<Value>(&body).map_err(|error| {
            HttpEndpointBindingError::new(format!(
                "decode endpoint {:?} request before applying its HTTP lifecycle: {error}",
                binding.endpoint_id()
            ))
        })?;
        if policy.inline_media {
            inline_image_urls(
                &mut payload,
                &HttpMediaFetcher::new(transport),
                &ImageDataUrlEncoder,
            )
            .await?;
        }
        let encoded = binding.encode_body(&payload, policy.content_type)?;
        headers.retain(|name, _| !name.eq_ignore_ascii_case("content-type"));
        headers.insert("Content-Type".into(), encoded.content_type);
        encoded.bytes
    } else {
        body
    };

    let mut request_config = RequestConfig::new(policy.url);
    request_config.headers = headers;
    request_config.params = parameters;
    request_config.correlation_id = correlation_id;
    request_config.request_id = request_id;
    request_config.is_final_turn = is_final_turn;
    request_config.cancel_after_ns = cancel_after_ns;
    request_config.reuse = reuse;

    Ok(PreparedHttpEndpointRequest {
        canonical_body,
        wire_body,
        request_config,
        streaming,
        polling: policy.polling,
    })
}

/// Borrowed endpoint configuration view used by metadata-driven HTTP binding.
pub trait HttpEndpointConfigView: fmt::Debug {
    /// Profile-owned base URLs.
    fn urls(&self) -> &[String];
    /// Optional endpoint path override.
    fn path(&self) -> Option<&str>;
    /// Explicit or normalized content type.
    fn request_content_type(&self) -> Option<RequestContentType>;
    /// Whole polling lifecycle timeout.
    fn timeout_seconds(&self) -> f64;
    /// Poll cadence.
    fn polling_interval_seconds(&self) -> f64;
    /// Whether completed media bytes are downloaded.
    fn download_video_content(&self) -> bool;
}

macro_rules! impl_config_view {
    ($type:ty) => {
        impl HttpEndpointConfigView for $type {
            fn urls(&self) -> &[String] {
                &self.urls
            }

            fn path(&self) -> Option<&str> {
                self.path.as_deref()
            }

            fn request_content_type(&self) -> Option<RequestContentType> {
                self.request_content_type
            }

            fn timeout_seconds(&self) -> f64 {
                self.timeout_seconds
            }

            fn polling_interval_seconds(&self) -> f64 {
                self.polling_interval_seconds
            }

            fn download_video_content(&self) -> bool {
                self.download_video_content
            }
        }
    };
}

impl_config_view!(EndpointConfig);
impl_config_view!(RawEndpointConfig);

#[derive(Debug)]
pub struct MetadataHttpEndpointBinding<'a, C: HttpEndpointConfigView + ?Sized = EndpointConfig> {
    descriptor: &'static EndpointDescriptor,
    config: &'a C,
    default_base_urls: &'a [String],
    model_name: &'a str,
}

impl<'a> MetadataHttpEndpointBinding<'a, RawEndpointConfig> {
    /// Bind a worker-local prepared endpoint without reconstructing a legacy
    /// enum/configuration pair.
    pub fn from_prepared(
        endpoint: &'a dyn PreparedEndpoint,
        default_base_urls: &'a [String],
        model_name: &'a str,
    ) -> Self {
        Self {
            descriptor: endpoint.descriptor(),
            config: endpoint.config().as_raw(),
            default_base_urls,
            model_name,
        }
    }
}

impl<C: HttpEndpointConfigView + ?Sized> MetadataHttpEndpointBinding<'_, C> {
    fn endpoint_url(
        &self,
        authored_path: Option<&str>,
        streaming: bool,
        url_index: Option<u32>,
    ) -> Result<String, HttpEndpointBindingError> {
        let selected_index = url_index.unwrap_or(0) as usize;
        let base_urls = if self.config.urls().is_empty() {
            self.default_base_urls
        } else {
            self.config.urls()
        };
        let base_url = base_urls.get(selected_index).ok_or_else(|| {
            HttpEndpointBindingError::new(format!(
                "URL index {selected_index} is out of range for {} configured endpoints",
                base_urls.len()
            ))
        })?;
        let target = authored_path
            .or(self.config.path())
            .or_else(|| {
                streaming
                    .then_some(self.descriptor.streaming_path)
                    .flatten()
            })
            .or(self.descriptor.endpoint_path);
        // The source transport expands this
        // transport-owned placeholder in both custom and metadata paths before
        // applying URL-prefix de-duplication. Keep expansion at the binding
        // boundary so endpoint dialects remain transport agnostic.
        let target = target.map(|target| target.replace("{model_name}", self.model_name));
        match target.as_deref() {
            None => Ok(base_url.trim_end_matches('/').to_string()),
            Some(path) if path.starts_with('/') => build_url(base_url, path, &BTreeMap::new())
                .map_err(|error| {
                    HttpEndpointBindingError::new(format!(
                        "cannot append endpoint path {path:?} to {base_url:?}: {error}"
                    ))
                }),
            Some(url) if url::Url::parse(url).is_ok() => Ok(url.to_string()),
            Some(value) => Err(HttpEndpointBindingError::new(format!(
                "dataset endpoint target {value:?} must be an absolute path or URL"
            ))),
        }
    }

    fn polling_options(&self) -> Result<Option<PollingOptions>, HttpEndpointBindingError> {
        if !self.descriptor.requires_polling {
            return Ok(None);
        }
        Ok(Some(PollingOptions {
            timeout_ns: seconds_to_ns(self.config.timeout_seconds(), "timeout_seconds")?,
            interval_ns: seconds_to_ns(
                self.config.polling_interval_seconds(),
                "polling_interval_seconds",
            )?,
            download_content: self.config.download_video_content(),
        }))
    }
}

impl<C: HttpEndpointConfigView + ?Sized> HttpEndpointBinding
    for MetadataHttpEndpointBinding<'_, C>
{
    fn endpoint_id(&self) -> &str {
        self.descriptor.id
    }

    fn request_policy(
        &self,
        endpoint_path: Option<&str>,
        streaming: bool,
        url_index: Option<u32>,
    ) -> Result<HttpEndpointPolicy, HttpEndpointBindingError> {
        let content_type =
            self.config
                .request_content_type()
                .unwrap_or(if self.descriptor.requires_form_data {
                    RequestContentType::MultipartFormData
                } else {
                    RequestContentType::ApplicationJson
                });
        if self.descriptor.requires_form_data
            != matches!(content_type, RequestContentType::MultipartFormData)
        {
            return Err(HttpEndpointBindingError::new(format!(
                "endpoint {:?} requires_form_data={} but request content type is {:?}",
                self.descriptor.id, self.descriptor.requires_form_data, content_type
            )));
        }
        Ok(HttpEndpointPolicy {
            url: self.endpoint_url(endpoint_path, streaming, url_index)?,
            content_type,
            inline_media: self.descriptor.requires_inline_media,
            polling: self.polling_options()?,
        })
    }

    fn decode_response(&self, response: &Response) -> Option<ServerResponse> {
        decode_response(response)
    }

    fn decode_sse_response(&self, message: &SseMessage) -> Option<ServerResponse> {
        decode_sse_response(message)
    }
}

/// Decode one complete HTTP response into canonical endpoint data.
pub fn decode_response(response: &Response) -> Option<ServerResponse> {
    match response {
        Response::Sse(message) => decode_sse_response(message),
        Response::Text(response) => Some(ServerResponse {
            perf_ns: non_negative_ns(response.perf_ns),
            json: response.json(),
            raw: Some(response.text.clone()),
        }),
    }
}

/// Decode one SSE message into canonical endpoint data.
pub fn decode_sse_response(message: &SseMessage) -> Option<ServerResponse> {
    if message.is_done() {
        return None;
    }
    let raw = message.data()?.to_string();
    Some(ServerResponse {
        perf_ns: non_negative_ns(message.perf_ns),
        json: serde_json::from_str(&raw).ok(),
        raw: Some(raw),
    })
}

fn non_negative_ns(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

fn seconds_to_ns(seconds: f64, field: &str) -> Result<i64, HttpEndpointBindingError> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(HttpEndpointBindingError::new(format!(
            "{field} must be finite and non-negative, got {seconds}"
        )));
    }
    let nanos = seconds * 1_000_000_000.0;
    if nanos >= i64::MAX as f64 {
        return Err(HttpEndpointBindingError::new(format!(
            "{field} is too large to represent in nanoseconds"
        )));
    }
    Ok(nanos.round() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use crate::transport::http::models::SseMessage;

    /// Prepare a builtin endpoint by its open ID for the prepared HTTP binding.
    fn prepared(endpoint_name: &str) -> Box<dyn PreparedEndpoint> {
        crate::endpoints::EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &crate::endpoints::EndpointId::new(endpoint_name).unwrap(),
                RawEndpointConfig::default(),
            )
            .unwrap()
    }

    fn endpoint_request(body: Bytes) -> HttpEndpointRequest {
        HttpEndpointRequest {
            body,
            headers: BTreeMap::new(),
            parameters: BTreeMap::new(),
            endpoint_path: None,
            streaming: false,
            correlation_id: None,
            request_id: Some("request-1".into()),
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
            reuse: ConnectionReuseStrategy::Pooled,
        }
    }

    #[test]
    fn response_decoder_normalizes_text_and_sse() {
        let text = Response::Text(crate::transport::core::TextResponse {
            perf_ns: 11,
            content_type: Some("application/json".into()),
            text: "{\"ok\":true}".into(),
            body: Bytes::from_static(b"{\"ok\":true}"),
        });
        assert_eq!(decode_response(&text).unwrap().json.unwrap()["ok"], true);

        let sse = SseMessage::parse("data: {\"token\":\"x\"}", 12);
        assert_eq!(
            decode_sse_response(&sse).unwrap().json.unwrap()["token"],
            "x"
        );
    }

    #[test]
    fn seconds_validation_accepts_zero_and_rejects_non_finite_values() {
        assert_eq!(seconds_to_ns(0.0, "timeout").unwrap(), 0);
        assert!(seconds_to_ns(f64::NAN, "timeout").is_err());
        assert_eq!(seconds_to_ns(0.25, "timeout").unwrap(), 250_000_000);
    }

    #[test]
    fn endpoint_path_join_collapses_v1_and_full_suffix_overlap() {
        let chat = prepared("chat");
        let v1_base = vec!["http://host/v1".to_string()];
        let binding =
            MetadataHttpEndpointBinding::from_prepared(chat.as_ref(), &v1_base, "fixture-model");
        assert_eq!(
            binding
                .endpoint_url(Some("/v1/embeddings"), false, None)
                .unwrap(),
            "http://host/v1/embeddings"
        );

        let full_base = vec!["http://host/v1/images/generations".to_string()];
        let binding =
            MetadataHttpEndpointBinding::from_prepared(chat.as_ref(), &full_base, "fixture-model");
        assert_eq!(
            binding
                .endpoint_url(Some("/v1/images/generations"), false, None)
                .unwrap(),
            "http://host/v1/images/generations"
        );
    }

    #[test]
    fn endpoint_path_templates_expand_model_name_before_url_joining() {
        let chat = prepared("chat");
        let base_urls = vec!["http://host/v1".to_string()];
        let binding =
            MetadataHttpEndpointBinding::from_prepared(chat.as_ref(), &base_urls, "sklearn-iris");
        assert_eq!(
            binding
                .endpoint_url(Some("/v1/models/{model_name}:predict"), false, None)
                .unwrap(),
            "http://host/v1/models/sklearn-iris:predict"
        );
        assert_eq!(
            binding
                .endpoint_url(
                    Some("http://other/v2/models/{model_name}/infer"),
                    false,
                    None,
                )
                .unwrap(),
            "http://other/v2/models/sklearn-iris/infer"
        );
    }

    #[tokio::test]
    async fn metadata_binding_preserves_json_and_lowers_multipart_at_http_boundary() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let transport = HttpTransport::new(
            clock,
            crate::transport::http::config::ClientConfig::default(),
        );
        let base_urls = vec!["http://host/v1".to_string()];
        let chat = prepared("chat");
        let binding =
            MetadataHttpEndpointBinding::from_prepared(chat.as_ref(), &base_urls, "fixture-model");
        let body = Bytes::from_static(br#"{"model":"m","messages":[]}"#);
        let request = prepare_request(&binding, &transport, endpoint_request(body.clone()))
            .await
            .unwrap();
        assert_eq!(request.canonical_body, body);
        assert_eq!(request.wire_body, body);
        assert_eq!(
            request.request_config.url,
            "http://host/v1/chat/completions"
        );

        let image = prepared("image_edit");
        let binding =
            MetadataHttpEndpointBinding::from_prepared(image.as_ref(), &base_urls, "fixture-model");
        let body = Bytes::from_static(
            br#"{"prompt":"edit","image":{"b64_data":"aGVsbG8=","filename":"in.txt","content_type":"text/plain"}}"#,
        );
        let request = prepare_request(&binding, &transport, endpoint_request(body.clone()))
            .await
            .unwrap();
        assert_eq!(request.canonical_body, body);
        assert_ne!(request.wire_body, body);
        assert!(
            request.request_config.headers["Content-Type"]
                .starts_with("multipart/form-data; boundary=")
        );
    }
}
