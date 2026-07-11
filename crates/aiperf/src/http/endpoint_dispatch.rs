// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata-driven HTTP lifecycles for materialized endpoint requests.
//!
//! Endpoint adapters own decoded JSON while this module applies the four wire
//! policies identified by `specs/2026-07-11-aiperf-rust-endpoints-design.md`:
//! streaming-path selection, multipart encoding, inline-media preparation, and
//! Clock-paced submit/poll/download. New endpoint dialects therefore extend the
//! decoded [`Endpoint`] seam without adding another issuer or HTTP stack.

use std::cell::Cell;
use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, ensure};
use bytes::Bytes;
use serde_json::Value;

use aiperf_endpoints::{
    Endpoint, EndpointConfig, EndpointResult, EndpointType, ParsedResponse, RequestContentType,
    ResponseData, ServerResponse,
};
use aiperf_metrics::HttpTrace;
use aiperf_transport::models::{ErrorKind, RequestConfig, RequestRecord, Response, SseMessage};
use aiperf_transport::transport::body::{
    JsonBodyEncoder, MultipartBodyEncoder, RequestBodyEncoder,
};
use aiperf_transport::transport::inline_media::{
    HttpMediaFetcher, ImageDataUrlEncoder, inline_image_urls,
};
use aiperf_transport::transport::polling::{
    JsonVideoPollingProtocol, PollingOptions, submit_and_poll,
};
use aiperf_transport::transport::url::build_url;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};

use super::{HttpDispatchResult, HttpRequest, TransportSink};

impl TransportSink {
    /// Dispatch a materialized request through lifecycle policy selected only by
    /// endpoint metadata and its effective per-turn configuration.
    pub(super) async fn dispatch_endpoint_collect_with_hooks(
        &self,
        req: HttpRequest,
        endpoint: &dyn Endpoint,
        endpoint_config: &EndpointConfig,
        obs: &dyn RequestObserver,
        mut on_first_token: impl FnMut(i64),
    ) -> Result<HttpDispatchResult> {
        let HttpRequest {
            uuid,
            max_output_tokens,
            prompt_text,
            request_body,
            request_body_bytes,
            mut headers,
            parameters,
            endpoint_path,
            streaming,
            x_correlation_id,
            is_final_turn,
            cancel_after_ns,
            url_index,
            ..
        } = req;
        obs.on_admit(uuid, self.ms(self.clock.now_ns()), 0);

        ensure!(
            request_body.is_none() || request_body_bytes.is_none(),
            "an HTTP request cannot supply both JSON and serialized bodies"
        );
        let body = match request_body_bytes {
            Some(body) => body,
            None => {
                let payload = request_body.unwrap_or_else(|| {
                    let prompt = prompt_text.unwrap_or_default();
                    aiperf_core::chat::chat_request_body(
                        &self.model,
                        &[("user", prompt.as_str())],
                        max_output_tokens,
                    )
                });
                Bytes::from(serde_json::to_vec(&payload)?)
            }
        };
        let mut endpoint_metrics = ObservedEndpointMetrics {
            num_images: serde_json::from_slice::<Value>(&body)
                .ok()
                .map(|payload| endpoint.extract_payload_inputs(&payload).image_count as usize)
                .filter(|count| *count > 0),
            ..ObservedEndpointMetrics::default()
        };
        let body = self
            .prepare_endpoint_body(body, endpoint, endpoint_config, &mut headers)
            .await?;

        let selected_url = self.endpoint_url(
            endpoint,
            endpoint_config,
            endpoint_path.as_deref(),
            streaming,
            url_index,
        )?;
        let mut request_config = RequestConfig::new(selected_url);
        request_config.headers = headers;
        request_config.params = parameters;
        request_config.correlation_id = x_correlation_id;
        request_config.is_final_turn = is_final_turn;
        request_config.cancel_after_ns = cancel_after_ns;

        let first_token_released = Cell::new(false);
        let record = if endpoint.metadata().requires_polling {
            let options = PollingOptions {
                timeout_ns: seconds_to_ns(endpoint_config.timeout_seconds, "timeout_seconds")?,
                interval_ns: seconds_to_ns(
                    endpoint_config.polling_interval_seconds,
                    "polling_interval_seconds",
                )?,
                download_content: endpoint_config.download_video_content,
            };
            submit_and_poll(
                &self.transport,
                self.clock.clone(),
                &request_config,
                body,
                options,
                &JsonVideoPollingProtocol,
            )
            .await
            .record
        } else {
            self.transport
                .send_request_bytes_with_first_token_filter(
                    &request_config,
                    body,
                    streaming,
                    |ttft_ns, message| {
                        if !meaningful_token_frame(endpoint, endpoint_config, message) {
                            return false;
                        }
                        if !first_token_released.replace(true) {
                            on_first_token(ttft_ns);
                        }
                        true
                    },
                )
                .await
        };

        let mut parsed_any = false;
        let mut parsed_content = false;
        let mut parse_failed = false;
        let mut response_text = String::new();
        let mut prompt_tokens = None;
        let mut completion_tokens = None;
        for response in &record.responses {
            let Some(server_response) = endpoint_response(response) else {
                continue;
            };
            let parsed = match parse_endpoint_response(endpoint, endpoint_config, &server_response)
            {
                Ok(parsed) => parsed,
                Err(error) => {
                    tracing::warn!(
                        uuid = %uuid,
                        endpoint = ?endpoint.metadata().endpoint_type,
                        error = %error,
                        "endpoint response parsing failed"
                    );
                    parse_failed = true;
                    continue;
                }
            };
            let Some(parsed) = parsed else { continue };
            parsed_any = true;
            absorb_usage(&parsed, &mut prompt_tokens, &mut completion_tokens);
            let Some(data) = parsed.data.as_ref() else {
                continue;
            };
            parsed_content = true;
            absorb_endpoint_metrics(data, &mut endpoint_metrics);
            let text = data.get_text();
            if text.is_empty() {
                continue;
            }
            response_text.push_str(&text);
            if endpoint.metadata().produces_tokens {
                let at_ns = i64::try_from(parsed.perf_ns).unwrap_or(i64::MAX);
                if !first_token_released.replace(true) {
                    on_first_token(at_ns.saturating_sub(record.start_ns));
                }
                obs.on_classified_token(uuid, self.ms(at_ns), token_kind(data));
            }
        }

        let terminal = match record.error.as_ref().map(|error| error.kind) {
            Some(ErrorKind::Cancelled) => ReplayTerminalStatus::Canceled,
            Some(_) => ReplayTerminalStatus::Failed,
            None if record
                .status
                .is_some_and(|status| (200..300).contains(&status))
                && parsed_content
                && !parse_failed =>
            {
                ReplayTerminalStatus::Completed
            }
            None => ReplayTerminalStatus::Failed,
        };
        tracing::debug!(
            uuid = %uuid,
            endpoint = ?endpoint.metadata().endpoint_type,
            url = %request_config.url,
            status = ?record.status,
            responses = record.responses.len(),
            error = ?record.error,
            parsed_any,
            parsed_content,
            parse_failed,
            terminal = ?terminal,
            "classified endpoint dispatch"
        );
        obs.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: prompt_tokens.map(|value| value as usize),
                completion_tokens: completion_tokens.map(|value| value as usize),
            },
        );
        obs.on_endpoint_metrics(uuid, endpoint_metrics);
        obs.on_terminal(uuid, terminal);

        Ok(HttpDispatchResult {
            start_ns: record.start_ns,
            end_ns: record.end_ns.unwrap_or_else(|| self.clock.now_ns()),
            status: record.status,
            terminal,
            response_text,
            prompt_tokens,
            completion_tokens,
            http: http_trace(&record),
        })
    }

    async fn prepare_endpoint_body(
        &self,
        body: Bytes,
        endpoint: &dyn Endpoint,
        endpoint_config: &EndpointConfig,
        headers: &mut BTreeMap<String, String>,
    ) -> Result<Bytes> {
        let metadata = endpoint.metadata();
        let content_type =
            endpoint_config
                .request_content_type
                .unwrap_or(if metadata.requires_form_data {
                    RequestContentType::MultipartFormData
                } else {
                    RequestContentType::ApplicationJson
                });
        ensure!(
            metadata.requires_form_data
                == matches!(content_type, RequestContentType::MultipartFormData),
            "endpoint {:?} requires_form_data={} but request content type is {:?}",
            metadata.endpoint_type,
            metadata.requires_form_data,
            content_type
        );
        if !metadata.requires_inline_media
            && !matches!(content_type, RequestContentType::MultipartFormData)
        {
            return Ok(body);
        }

        let mut payload = serde_json::from_slice::<Value>(&body).with_context(|| {
            format!(
                "decode {:?} request before applying its wire lifecycle",
                metadata.endpoint_type
            )
        })?;
        if metadata.requires_inline_media {
            inline_image_urls(
                &mut payload,
                &HttpMediaFetcher::new(&self.transport),
                &ImageDataUrlEncoder,
            )
            .await
            .map_err(|error| anyhow!(error.message))?;
        }
        let encoded = match content_type {
            RequestContentType::ApplicationJson => JsonBodyEncoder.encode(&payload),
            RequestContentType::MultipartFormData => MultipartBodyEncoder.encode(&payload),
        }
        .map_err(|error| anyhow!(error.message))?;
        headers.retain(|name, _| !name.eq_ignore_ascii_case("content-type"));
        headers.insert("Content-Type".into(), encoded.content_type);
        Ok(encoded.bytes)
    }

    fn endpoint_url(
        &self,
        endpoint: &dyn Endpoint,
        endpoint_config: &EndpointConfig,
        authored_path: Option<&str>,
        streaming: bool,
        url_index: Option<u32>,
    ) -> Result<String> {
        let selected_index = url_index.unwrap_or(0) as usize;
        let base_urls = if endpoint_config.urls.is_empty() {
            &self.base_urls
        } else {
            &endpoint_config.urls
        };
        let base_url = base_urls.get(selected_index).ok_or_else(|| {
            anyhow!(
                "URL index {selected_index} is out of range for {} configured endpoints",
                base_urls.len()
            )
        })?;
        let metadata = endpoint.metadata();
        let target = authored_path
            .or(endpoint_config.path.as_deref())
            .or_else(|| streaming.then_some(metadata.streaming_path).flatten())
            .or(metadata.endpoint_path);
        match target {
            None => Ok(base_url.trim_end_matches('/').to_string()),
            Some(path) if path.starts_with('/') => append_endpoint_path(base_url, path),
            Some(url) if url::Url::parse(url).is_ok() => Ok(url.to_string()),
            Some(value) => Err(anyhow!(
                "dataset endpoint target {value:?} must be an absolute path or URL"
            )),
        }
    }
}

fn append_endpoint_path(base_url: &str, endpoint_path: &str) -> Result<String> {
    build_url(base_url, endpoint_path, &BTreeMap::new()).map_err(|error| {
        anyhow!("cannot append endpoint path {endpoint_path:?} to {base_url:?}: {error}")
    })
}

fn meaningful_token_frame(
    endpoint: &dyn Endpoint,
    endpoint_config: &EndpointConfig,
    message: &SseMessage,
) -> bool {
    if !endpoint.metadata().produces_tokens || message.is_done() {
        return false;
    }
    sse_endpoint_response(message)
        .and_then(|response| {
            parse_endpoint_response(endpoint, endpoint_config, &response)
                .ok()
                .flatten()
        })
        .and_then(|parsed| parsed.data)
        .is_some_and(|data| !data.get_text().is_empty())
}

fn parse_endpoint_response(
    endpoint: &dyn Endpoint,
    endpoint_config: &EndpointConfig,
    response: &ServerResponse,
) -> EndpointResult<Option<ParsedResponse>> {
    let parsed = endpoint.parse_response_with_config(response, endpoint_config)?;
    if parsed.is_some() || endpoint.metadata().endpoint_type != EndpointType::Chat {
        return Ok(parsed);
    }
    let Some(mut object) = response.json.as_ref().and_then(Value::as_object).cloned() else {
        return Ok(None);
    };
    if object.contains_key("object") || !object.contains_key("choices") {
        return Ok(None);
    }

    // Older OpenAI-compatible mocks omitted `object` while retaining a valid
    // `choices` envelope. Keep the endpoint adapter itself source-strict and
    // normalize only this established wire-compatibility shape at dispatch.
    object.insert(
        "object".into(),
        Value::String(
            if endpoint_config.streaming {
                "chat.completion.chunk"
            } else {
                "chat.completion"
            }
            .into(),
        ),
    );
    endpoint.parse_response_with_config(
        &ServerResponse {
            perf_ns: response.perf_ns,
            json: Some(Value::Object(object)),
            raw: response.raw.clone(),
        },
        endpoint_config,
    )
}

fn endpoint_response(response: &Response) -> Option<ServerResponse> {
    match response {
        Response::Sse(message) => sse_endpoint_response(message),
        Response::Text(response) => Some(ServerResponse {
            perf_ns: non_negative_ns(response.perf_ns),
            json: response.json(),
            raw: Some(response.text.clone()),
        }),
    }
}

fn sse_endpoint_response(message: &SseMessage) -> Option<ServerResponse> {
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

fn token_kind(data: &ResponseData) -> ObservedTokenKind {
    match data {
        ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
            ObservedTokenKind::Reasoning
        }
        _ => ObservedTokenKind::Output,
    }
}

fn absorb_endpoint_metrics(data: &ResponseData, metrics: &mut ObservedEndpointMetrics) {
    let ResponseData::Video(video) = data else {
        return;
    };
    metrics.video_inference_seconds = video
        .inference_time_s
        .filter(|value| value.is_finite())
        .or(metrics.video_inference_seconds);
    metrics.video_peak_memory_mb = video
        .peak_memory_mb
        .filter(|value| value.is_finite())
        .or(metrics.video_peak_memory_mb);
}

fn absorb_usage(
    parsed: &ParsedResponse,
    prompt_tokens: &mut Option<u32>,
    completion_tokens: &mut Option<u32>,
) {
    let Some(usage) = parsed.usage.as_ref() else {
        return;
    };
    if let Some(value) = usage_count(usage, &["prompt_tokens", "input_tokens"]) {
        *prompt_tokens = Some(value);
    }
    if let Some(value) = usage_count(usage, &["completion_tokens", "output_tokens"]) {
        *completion_tokens = Some(value);
    }
}

fn usage_count(usage: &Value, names: &[&str]) -> Option<u32> {
    names.iter().find_map(|name| {
        usage
            .get(*name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    })
}

fn seconds_to_ns(seconds: f64, field: &str) -> Result<i64> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0,
        "{field} must be finite and non-negative"
    );
    let nanoseconds = seconds * 1_000_000_000.0;
    ensure!(
        nanoseconds <= i64::MAX as f64,
        "{field} exceeds the nanosecond clock range"
    );
    Ok(nanoseconds.round() as i64)
}

fn http_trace(record: &RequestRecord) -> HttpTrace {
    let mut http = record
        .trace
        .as_ref()
        .map_or_else(HttpTrace::default, |trace| HttpTrace {
            blocked_ns: trace.blocked(),
            dns_lookup_ns: trace.dns_lookup(),
            connecting_ns: trace.connecting(),
            sending_ns: trace.sending(),
            waiting_ns: trace.waiting(),
            receiving_ns: trace.receiving(),
            duration_ns: trace.duration(),
            connection_reused: Some(trace.connection_reused_ns.is_some()),
            data_sent_bytes: Some(trace.request_bytes_total),
            data_received_bytes: Some(trace.response_bytes_total),
            chunks_sent: Some(u64::from(trace.request_chunks_count)),
            chunks_received: Some(u64::from(trace.response_chunks_count)),
            ..HttpTrace::default()
        });
    http.stream_setup_ns = record
        .recv_start_ns
        .map(|receive_start| receive_start.saturating_sub(record.start_ns));
    http
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf_endpoints::{ChatEndpoint, HuggingFaceGenerateEndpoint, ImageGenerationEndpoint};

    #[test]
    fn endpoint_sse_filter_uses_the_selected_dialect() {
        let mut config = EndpointConfig {
            streaming: true,
            ..EndpointConfig::default()
        };
        let tgi = SseMessage::parse(r#"data: {"token":{"text":"hello"}}"#, 10);
        assert!(meaningful_token_frame(
            &HuggingFaceGenerateEndpoint,
            &config,
            &tgi
        ));

        config.endpoint_type = aiperf_endpoints::EndpointType::ImageGeneration;
        let image = SseMessage::parse(r#"data: {"b64_json":"AA=="}"#, 11);
        assert!(!meaningful_token_frame(
            &ImageGenerationEndpoint,
            &config,
            &image
        ));

        config.endpoint_type = EndpointType::Chat;
        let legacy_chat =
            SseMessage::parse(r#"data: {"choices":[{"delta":{"content":"compat"}}]}"#, 12);
        assert!(meaningful_token_frame(&ChatEndpoint, &config, &legacy_chat));
    }

    #[test]
    fn usage_aliases_and_second_conversion_are_bounded() {
        let parsed = ParsedResponse {
            perf_ns: 1,
            data: None,
            usage: Some(serde_json::json!({"input_tokens":3,"output_tokens":5})),
            sources: None,
        };
        let mut prompt = None;
        let mut completion = None;
        absorb_usage(&parsed, &mut prompt, &mut completion);
        assert_eq!((prompt, completion), (Some(3), Some(5)));
        assert_eq!(seconds_to_ns(0.5, "interval").unwrap(), 500_000_000);
        assert!(seconds_to_ns(f64::INFINITY, "timeout").is_err());
    }

    #[test]
    fn endpoint_path_join_collapses_v1_and_full_suffix_overlap() {
        assert_eq!(
            append_endpoint_path("http://host/v1", "/v1/embeddings").unwrap(),
            "http://host/v1/embeddings"
        );
        assert_eq!(
            append_endpoint_path(
                "http://host/v1/images/generations",
                "/v1/images/generations"
            )
            .unwrap(),
            "http://host/v1/images/generations"
        );
    }
}
