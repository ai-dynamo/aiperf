// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-driven submit/poll/optional-download request lifecycle with
//! provider-neutral protocol parsing.

use std::rc::Rc;

use crate::clock::Clock;
use bytes::Bytes;
use serde_json::Value;

use crate::transport::core::{ErrorDetails, ErrorKind, RequestRecord, Response, TraceData};
use crate::transport::http::client::cancellation::{CancelOutcome, race_cancel};
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;

/// Provider-neutral classification of one poll response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollingState {
    /// The job remains queued or in progress.
    Pending,
    /// The job completed successfully.
    Completed,
    /// The job terminated with a provider failure message.
    Failed(String),
}

/// Provider-specific submit and status response parser.
pub trait PollingProtocol {
    /// Extract the stable job identifier from the submit response.
    fn job_id(&self, submission: &RequestRecord) -> Result<String, ErrorDetails>;
    /// Classify one status response.
    fn state(&self, poll: &RequestRecord) -> Result<PollingState, ErrorDetails>;
    /// Return a provider-supplied content URL from a completed status response.
    fn content_url(&self, poll: &RequestRecord) -> Option<String>;
}

/// OpenAI/SGLang video JSON polling protocol.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonVideoPollingProtocol;

impl PollingProtocol for JsonVideoPollingProtocol {
    fn job_id(&self, submission: &RequestRecord) -> Result<String, ErrorDetails> {
        let object = response_json(submission)?;
        match object.get("id") {
            Some(Value::String(id)) if !id.is_empty() => Ok(id.clone()),
            Some(Value::Number(id)) => Ok(id.to_string()),
            _ => Err(ErrorDetails::other(format!(
                "video submission returned no job ID: {}",
                Value::Object(object.clone())
            ))),
        }
    }

    fn state(&self, poll: &RequestRecord) -> Result<PollingState, ErrorDetails> {
        let object = response_json(poll)?;
        Ok(match object.get("status").and_then(Value::as_str) {
            Some("completed") => PollingState::Completed,
            Some("failed") => PollingState::Failed(provider_error(&object)),
            _ => PollingState::Pending,
        })
    }

    fn content_url(&self, poll: &RequestRecord) -> Option<String> {
        response_json(poll).ok().and_then(|object| {
            object
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
    }
}

/// Timeout, cadence, and download controls for one polling lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollingOptions {
    /// Maximum time from the start of polling through completion.
    pub timeout_ns: i64,
    /// Clock delay between pending polls.
    pub interval_ns: i64,
    /// Fetch completed content bytes when true.
    pub download_content: bool,
}

/// Aggregated submit/poll result returned to endpoint response parsing.
#[derive(Debug, Clone)]
pub struct PollingResult {
    /// Aggregate request record containing submit and completed status responses.
    pub record: RequestRecord,
    /// Optional downloaded content bytes; intentionally excluded from parsed responses.
    pub downloaded_content: Option<Bytes>,
}

/// Execute a POST submit followed by Clock-paced GET polls and optional content GET.
pub async fn submit_and_poll(
    transport: &HttpTransport,
    clock: Rc<dyn Clock>,
    submit_config: &RequestConfig,
    body: Bytes,
    options: PollingOptions,
    protocol: &dyn PollingProtocol,
) -> PollingResult {
    let submission = transport
        .send_request_bytes(submit_config, body, false, |_| {})
        .await;
    let mut aggregate = submission;
    if aggregate.error.is_some() {
        return PollingResult {
            record: aggregate,
            downloaded_content: None,
        };
    }
    let cancellation_deadline_ns = submit_config.cancel_after_ns.and_then(|delay| {
        aggregate
            .trace
            .as_ref()
            .and_then(|trace| trace.request_send_end_ns)
            .map(|sent_ns| sent_ns.saturating_add(delay.max(0)))
    });
    let job_id = match protocol.job_id(&aggregate) {
        Ok(job_id) => job_id,
        Err(error) => {
            aggregate.error = Some(error);
            aggregate.end_ns = Some(clock.now_ns());
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        }
    };
    let poll_url = match append_path(&submit_config.url, &job_id) {
        Ok(url) => url,
        Err(error) => {
            aggregate.error = Some(error);
            aggregate.end_ns = Some(clock.now_ns());
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        }
    };
    let mut poll_config = submit_config.clone();
    poll_config.url.clone_from(&poll_url);
    poll_config.cancel_after_ns = None;
    let poll_start = clock.now_ns();
    let completed_poll = loop {
        if cancellation_deadline_ns.is_some_and(|deadline| clock.now_ns() >= deadline) {
            mark_cancelled(
                &mut aggregate,
                clock.now_ns(),
                submit_config.cancel_after_ns.unwrap_or_default(),
            );
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        }
        let elapsed = clock.now_ns().saturating_sub(poll_start);
        if elapsed >= options.timeout_ns {
            aggregate.error = Some(ErrorDetails {
                kind: ErrorKind::Timeout,
                code: Some(504),
                message: format!("video generation timed out after {}ns", options.timeout_ns),
            });
            aggregate.end_ns = Some(clock.now_ns());
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        }
        let Some(poll) = get_before_deadline(
            transport,
            clock.clone(),
            &poll_config,
            cancellation_deadline_ns,
        )
        .await
        else {
            mark_cancelled(
                &mut aggregate,
                clock.now_ns(),
                submit_config.cancel_after_ns.unwrap_or_default(),
            );
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        };
        if let Some(error) = poll.error.clone() {
            merge_attempt(&mut aggregate, &poll, false);
            aggregate.error = Some(error);
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        }
        let state = protocol.state(&poll);
        merge_attempt(
            &mut aggregate,
            &poll,
            matches!(state, Ok(PollingState::Completed)),
        );
        match state {
            Ok(PollingState::Completed) => break poll,
            Ok(PollingState::Failed(message)) => {
                aggregate.error = Some(ErrorDetails::other(format!(
                    "video generation failed: {message}"
                )));
                return PollingResult {
                    record: aggregate,
                    downloaded_content: None,
                };
            }
            Ok(PollingState::Pending) => {
                let remaining = options
                    .timeout_ns
                    .saturating_sub(clock.now_ns().saturating_sub(poll_start));
                let cancellation_remaining = cancellation_deadline_ns
                    .map(|deadline| deadline.saturating_sub(clock.now_ns()).max(0))
                    .unwrap_or(i64::MAX);
                clock
                    .clone()
                    .sleep(
                        options
                            .interval_ns
                            .min(remaining)
                            .min(cancellation_remaining)
                            .max(0),
                    )
                    .await;
            }
            Err(error) => {
                aggregate.error = Some(error);
                return PollingResult {
                    record: aggregate,
                    downloaded_content: None,
                };
            }
        }
    };
    let downloaded_content = if options.download_content {
        let authored = protocol.content_url(&completed_poll);
        let content_url = match authored {
            Some(url) => match resolve_content_url(&poll_url, &url) {
                Ok(url) => url,
                Err(error) => {
                    aggregate.error = Some(error);
                    return PollingResult {
                        record: aggregate,
                        downloaded_content: None,
                    };
                }
            },
            None => match append_path(&poll_url, "content") {
                Ok(url) => url,
                Err(error) => {
                    aggregate.error = Some(error);
                    return PollingResult {
                        record: aggregate,
                        downloaded_content: None,
                    };
                }
            },
        };
        let mut content_config = submit_config.clone();
        content_config.url = content_url;
        content_config.cancel_after_ns = None;
        let Some(download) = get_before_deadline(
            transport,
            clock.clone(),
            &content_config,
            cancellation_deadline_ns,
        )
        .await
        else {
            mark_cancelled(
                &mut aggregate,
                clock.now_ns(),
                submit_config.cancel_after_ns.unwrap_or_default(),
            );
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        };
        merge_attempt(&mut aggregate, &download, false);
        if let Some(error) = download.error {
            aggregate.error = Some(ErrorDetails::other(format!(
                "failed to download video {job_id}: {}",
                error.message
            )));
            return PollingResult {
                record: aggregate,
                downloaded_content: None,
            };
        }
        match download
            .responses
            .into_iter()
            .find_map(|response| match response {
                Response::Text(response) => Some(response.body),
                Response::Sse(_) => None,
            }) {
            Some(bytes) => Some(bytes),
            None => {
                aggregate.error = Some(ErrorDetails::other(format!(
                    "no content returned for video {job_id}"
                )));
                return PollingResult {
                    record: aggregate,
                    downloaded_content: None,
                };
            }
        }
    } else {
        None
    };

    PollingResult {
        record: aggregate,
        downloaded_content,
    }
}

async fn get_before_deadline(
    transport: &HttpTransport,
    clock: Rc<dyn Clock>,
    config: &RequestConfig,
    deadline_ns: Option<i64>,
) -> Option<RequestRecord> {
    let Some(deadline_ns) = deadline_ns else {
        return Some(transport.get(config).await);
    };
    let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
    if remaining_ns <= 0 {
        return None;
    }
    match race_cancel(clock, remaining_ns, transport.get(config)).await {
        CancelOutcome::Completed(record) => Some(record),
        CancelOutcome::Cancelled => None,
    }
}

fn mark_cancelled(record: &mut RequestRecord, now_ns: i64, delay_ns: i64) {
    record.end_ns = Some(now_ns);
    record.status = Some(499);
    record.cancellation_ns = Some(now_ns);
    record.error = Some(ErrorDetails::cancelled(format!(
        "RequestCancellationError: polling lifecycle cancelled {delay_ns}ns after submission was sent"
    )));
    if let Some(trace) = &mut record.trace {
        trace.error_timestamp_ns = Some(now_ns);
    }
}

fn response_json(record: &RequestRecord) -> Result<serde_json::Map<String, Value>, ErrorDetails> {
    record
        .responses
        .iter()
        .rev()
        .find_map(|response| match response {
            Response::Text(response) => response.json(),
            Response::Sse(_) => None,
        })
        .and_then(|value| value.as_object().cloned())
        .ok_or_else(|| ErrorDetails::other("polling response did not contain a JSON object"))
}

fn provider_error(object: &serde_json::Map<String, Value>) -> String {
    object
        .get("error")
        .and_then(|error| {
            error
                .as_object()
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| error.as_str().map(str::to_string))
        })
        .unwrap_or_else(|| "unknown error".into())
}

fn append_path(base: &str, component: &str) -> Result<String, ErrorDetails> {
    let mut url = url::Url::parse(base)
        .map_err(|error| ErrorDetails::other(format!("invalid polling URL {base:?}: {error}")))?;
    let path = format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        component.trim_matches('/')
    );
    url.set_path(&path);
    Ok(url.into())
}

fn resolve_content_url(base: &str, value: &str) -> Result<String, ErrorDetails> {
    if let Ok(url) = url::Url::parse(value) {
        return Ok(url.into());
    }
    url::Url::parse(base)
        .and_then(|base| base.join(value))
        .map(Into::into)
        .map_err(|error| {
            ErrorDetails::other(format!("invalid video content URL {value:?}: {error}"))
        })
}

fn merge_attempt(aggregate: &mut RequestRecord, attempt: &RequestRecord, keep_responses: bool) {
    aggregate.end_ns = attempt.end_ns;
    aggregate.status = attempt.status;
    aggregate
        .response_headers
        .clone_from(&attempt.response_headers);
    aggregate.recv_start_ns = aggregate.recv_start_ns.or(attempt.recv_start_ns);
    if keep_responses {
        aggregate
            .responses
            .extend(attempt.responses.iter().cloned());
    }
    match (&mut aggregate.trace, &attempt.trace) {
        (Some(target), Some(source)) => merge_trace(target, source),
        (None, Some(source)) => aggregate.trace = Some(source.clone()),
        _ => {}
    }
}

fn merge_trace(target: &mut TraceData, source: &TraceData) {
    target.request_chunks_count = target
        .request_chunks_count
        .saturating_add(source.request_chunks_count);
    target.request_bytes_total = target
        .request_bytes_total
        .saturating_add(source.request_bytes_total);
    target.response_chunks_count = target
        .response_chunks_count
        .saturating_add(source.response_chunks_count);
    target.response_bytes_total = target
        .response_bytes_total
        .saturating_add(source.response_bytes_total);
    target.response_receive_end_ns = source
        .response_receive_end_ns
        .or(target.response_receive_end_ns);
    target.response_status_code = source.response_status_code.or(target.response_status_code);
    target.response_reason = source
        .response_reason
        .clone()
        .or_else(|| target.response_reason.clone());
    target.error_timestamp_ns = source.error_timestamp_ns.or(target.error_timestamp_ns);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::core::TextResponse;

    fn record(value: Value) -> RequestRecord {
        let body = Bytes::from(serde_json::to_vec(&value).unwrap());
        RequestRecord {
            responses: vec![Response::Text(TextResponse {
                perf_ns: 2,
                text: String::from_utf8(body.to_vec()).unwrap(),
                body,
                content_type: Some("application/json".into()),
            })],
            ..RequestRecord::started(1)
        }
    }

    #[test]
    fn video_protocol_extracts_id_states_and_content_url() {
        let protocol = JsonVideoPollingProtocol;
        assert_eq!(
            protocol
                .job_id(&record(serde_json::json!({"id":"video-1"})))
                .unwrap(),
            "video-1"
        );
        assert_eq!(
            protocol
                .state(&record(serde_json::json!({"status":"queued"})))
                .unwrap(),
            PollingState::Pending
        );
        assert_eq!(
            protocol
                .state(&record(serde_json::json!({"status":"completed"})))
                .unwrap(),
            PollingState::Completed
        );
        assert_eq!(
            protocol
                .state(&record(
                    serde_json::json!({"status":"failed","error":{"message":"boom"}})
                ))
                .unwrap(),
            PollingState::Failed("boom".into())
        );
        assert_eq!(
            protocol.content_url(&record(serde_json::json!({"url":"/content"}))),
            Some("/content".into())
        );
    }

    #[test]
    fn polling_urls_append_without_corrupting_query() {
        assert_eq!(
            append_path("http://host/v1/videos?api=1", "id").unwrap(),
            "http://host/v1/videos/id?api=1"
        );
        assert_eq!(
            resolve_content_url("http://host/v1/videos/id", "/v1/videos/id/content").unwrap(),
            "http://host/v1/videos/id/content"
        );
    }
}
