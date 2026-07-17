// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared online endpoint readiness over injected Clock and transport seams.
//!
//! Python Config v2 owns authoring. The selected endpoint factory owns the
//! exact readiness request. This module expands those policies once into an
//! immutable plan, then drives every `(profile URL, model)` target before the
//! runner creates its exclusive artifact directory. No fallback inference
//! payload or endpoint-kind match lives in the driver.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use crate::clock::Clock;
use crate::endpoints::{
    EndpointId, EndpointRegistry, PreparedReadinessRequest, RawEndpointConfig, ReadinessMethod,
    ReadinessPolicy, ReadinessSuccess,
};
use crate::transport::core::{ConnectionReuseStrategy, Response};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::Value;
use url::Url;

const GET_REQUEST_TIMEOUT_FLOOR_NS: i64 = 5_000_000_000;
const POST_REQUEST_TIMEOUT_FLOOR_NS: i64 = 30_000_000_000;

/// One normalized endpoint profile consumed by readiness preparation.
#[derive(Clone, Debug)]
pub struct ReadinessEndpointProfile {
    profile_id: String,
    endpoint_id: EndpointId,
    config: RawEndpointConfig,
    connection_reuse: ConnectionReuseStrategy,
    client: ClientConfig,
}

impl ReadinessEndpointProfile {
    /// Retain one coordinator-validated endpoint identity and policy.
    pub fn new(
        profile_id: impl Into<String>,
        endpoint_id: EndpointId,
        config: RawEndpointConfig,
        connection_reuse: ConnectionReuseStrategy,
        client: ClientConfig,
    ) -> Self {
        Self {
            profile_id: profile_id.into(),
            endpoint_id,
            config,
            connection_reuse,
            client,
        }
    }

    /// Run-local endpoint profile identity.
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    /// Canonical open endpoint dialect identity.
    pub fn endpoint_id(&self) -> &EndpointId {
        &self.endpoint_id
    }

    /// Coordinator-normalized endpoint policy.
    pub fn config(&self) -> &RawEndpointConfig {
        &self.config
    }

    /// Authored connection reuse strategy.
    pub const fn connection_reuse(&self) -> ConnectionReuseStrategy {
        self.connection_reuse
    }

    /// Fully validated HTTP client policy for this profile.
    pub fn client(&self) -> &ClientConfig {
        &self.client
    }
}

/// Borrowed inputs used to prepare one run's immutable readiness plan.
pub struct ReadinessPlanInput<'a> {
    /// Frozen endpoint factory registry selected by the runner image.
    pub endpoints: &'a EndpointRegistry,
    /// Normalized endpoint profiles in authored order.
    pub profiles: &'a [ReadinessEndpointProfile],
    /// Effective model names in model-selection order.
    pub models: &'a [String],
}

/// Side-effect-free factory for one prepared online readiness plan.
pub trait OnlineReadinessPlanFactory: fmt::Debug + Send + Sync {
    /// Resolve every endpoint-owned request without opening sockets or files.
    fn prepare(&self, input: ReadinessPlanInput<'_>) -> Result<Box<dyn PreparedOnlineReadiness>>;
}

/// One immutable readiness plan retained from pair preparation to execution.
#[async_trait(?Send)]
pub trait PreparedOnlineReadiness: fmt::Debug {
    /// Whether no endpoint profile enabled readiness.
    fn is_empty(&self) -> bool;

    /// Number of sequential `(URL, model)` targets.
    fn target_count(&self) -> usize;

    /// Wait for every target through the injected Clock and transport.
    async fn wait(
        &self,
        clock: Rc<dyn Clock>,
        transport: Rc<dyn ReadinessTransport>,
    ) -> Result<ReadinessReport>;
}

/// Factory for the control transport paired with a run-owned Clock.
pub trait ReadinessTransportFactory: fmt::Debug + Send + Sync {
    /// Build one transport used for the complete prepared plan.
    fn build(&self, clock: Rc<dyn Clock>) -> Rc<dyn ReadinessTransport>;
}

/// Transport-neutral execution seam for one readiness attempt.
#[async_trait(?Send)]
pub trait ReadinessTransport: fmt::Debug {
    /// Execute one exact prepared request within its Clock deadline.
    async fn execute(&self, request: ReadinessAttemptRequest) -> ReadinessAttemptResponse;
}

/// Exact request and attempt deadline handed to a readiness transport.
#[derive(Clone)]
pub struct ReadinessAttemptRequest {
    method: ReadinessMethod,
    url: String,
    headers: BTreeMap<String, String>,
    body: Option<Value>,
    timeout_ns: i64,
    connection_reuse: ConnectionReuseStrategy,
    client: ClientConfig,
}

impl fmt::Debug for ReadinessAttemptRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadinessAttemptRequest")
            .field("method", &self.method)
            .field("timeout_ns", &self.timeout_ns)
            .field("header_count", &self.headers.len())
            .field("has_body", &self.body.is_some())
            .field("connection_reuse", &self.connection_reuse)
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

impl ReadinessAttemptRequest {
    /// HTTP method selected by the endpoint dialect.
    pub const fn method(&self) -> ReadinessMethod {
        self.method
    }

    /// Absolute request URL after profile/path composition.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Endpoint-owned request headers.
    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    /// Optional decoded JSON request body.
    pub fn body(&self) -> Option<&Value> {
        self.body.as_ref()
    }

    /// Clock deadline for this attempt.
    pub const fn timeout_ns(&self) -> i64 {
        self.timeout_ns
    }

    /// Authored connection reuse policy for this target.
    pub const fn connection_reuse(&self) -> ConnectionReuseStrategy {
        self.connection_reuse
    }

    /// Fully validated HTTP client policy for this target.
    pub fn client(&self) -> &ClientConfig {
        &self.client
    }
}

/// Minimal response facts consumed by an injected readiness classifier.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadinessAttemptResponse {
    /// HTTP status when response headers were received.
    pub status: Option<u16>,
    /// Non-streaming response body, when retained.
    pub body: Option<String>,
    /// Structured transport diagnostic without request secrets.
    pub error: Option<String>,
}

/// Completed readiness totals retained for terminal diagnostics/provenance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadinessReport {
    /// Targets that reached the endpoint-owned success condition.
    pub targets_ready: usize,
    /// Total transport attempts across all targets.
    pub attempts: usize,
}

trait ReadinessResponseClassifier: fmt::Debug + Send + Sync {
    fn is_ready(&self, response: &ReadinessAttemptResponse) -> bool;
}

#[derive(Debug)]
struct SuccessfulHttpStatus;

impl ReadinessResponseClassifier for SuccessfulHttpStatus {
    fn is_ready(&self, response: &ReadinessAttemptResponse) -> bool {
        response
            .status
            .is_some_and(|status| (200..300).contains(&status))
    }
}

#[derive(Debug)]
struct ListedModel {
    model: String,
}

impl ReadinessResponseClassifier for ListedModel {
    fn is_ready(&self, response: &ReadinessAttemptResponse) -> bool {
        if response.status != Some(200) {
            return false;
        }
        let Some(body) = response.body.as_deref() else {
            return false;
        };
        let Ok(Value::Object(payload)) = serde_json::from_str::<Value>(body) else {
            return false;
        };
        payload
            .get("data")
            .and_then(Value::as_array)
            .is_some_and(|models| {
                models.iter().any(|entry| {
                    entry
                        .as_object()
                        .and_then(|model| model.get("id"))
                        .and_then(Value::as_str)
                        == Some(self.model.as_str())
                })
            })
    }
}

#[derive(Debug)]
struct NonServerError;

impl ReadinessResponseClassifier for NonServerError {
    fn is_ready(&self, response: &ReadinessAttemptResponse) -> bool {
        response.status.is_some_and(|status| status < 500)
    }
}

fn readiness_classifier(success: &ReadinessSuccess) -> Arc<dyn ReadinessResponseClassifier> {
    match success {
        ReadinessSuccess::SuccessfulStatus => Arc::new(SuccessfulHttpStatus),
        ReadinessSuccess::ModelListed(model) => Arc::new(ListedModel {
            model: model.clone(),
        }),
        ReadinessSuccess::NonServerError => Arc::new(NonServerError),
    }
}

struct PreparedReadinessTarget {
    profile_id: String,
    model: String,
    /// Authored base URL for this target, used verbatim in progress and
    /// timeout diagnostics so operators can correlate the failing endpoint.
    base_url: String,
    request: ReadinessAttemptRequest,
    /// Endpoint-owned success condition, retained to pick the exact progress
    /// message (model listing vs. inference liveness) and to gate the
    /// models-mode base-URL fallback.
    success: ReadinessSuccess,
    /// Base-URL `GET` issued when a models-listing probe returns 404, allowing a
    /// responsive server without a model-list endpoint to satisfy liveness.
    fallback: Option<ReadinessAttemptRequest>,
    timeout_ns: i64,
    interval_ns: i64,
    request_timeout_floor_ns: i64,
    classifier: Arc<dyn ReadinessResponseClassifier>,
}

impl PreparedReadinessTarget {
    /// Emit the endpoint-appropriate readiness-ready progress line to stderr.
    ///
    /// Message wording is a compatibility contract for log-scraping consumers.
    fn log_ready(&self, attempts: usize, response: &ReadinessAttemptResponse) {
        match &self.success {
            ReadinessSuccess::ModelListed(_) => {
                // Keep `Model '<id>' ready` stable for readiness-log consumers.
                tracing::info!(
                    "Model '{}' ready at {} after {} attempt(s)",
                    self.model,
                    self.base_url,
                    attempts
                );
            }
            ReadinessSuccess::NonServerError => {
                let status = response
                    .status
                    .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
                // Keep the capitalized prefix stable for readiness-log consumers.
                tracing::info!(
                    "Inference probe ready at {} (status={}, attempt {})",
                    self.request.url,
                    status,
                    attempts
                );
            }
            ReadinessSuccess::SuccessfulStatus => {
                tracing::info!(
                    url = %self.request.url,
                    attempts,
                    "readiness: endpoint ready"
                );
            }
        }
    }

    /// Emit the endpoint-appropriate retry progress line to stderr.
    fn log_retry(&self, attempts: usize, response: &ReadinessAttemptResponse, interval_s: f64) {
        let status = response
            .status
            .map_or_else(|| "connection error".to_owned(), |value| value.to_string());
        match &self.success {
            ReadinessSuccess::ModelListed(_) if response.status == Some(200) => {
                tracing::warn!(
                    model = %self.model,
                    url = %self.request.url,
                    attempts,
                    interval_s,
                    "readiness: model not yet listed, retrying"
                );
            }
            ReadinessSuccess::NonServerError => {
                tracing::warn!(
                    url = %self.request.url,
                    status = %status,
                    attempts,
                    interval_s,
                    "readiness: inference probe returned error, retrying"
                );
            }
            _ => {
                tracing::warn!(
                    url = %self.request.url,
                    status = %status,
                    attempts,
                    interval_s,
                    "readiness: probe returned error, retrying"
                );
            }
        }
    }
}

impl fmt::Debug for PreparedReadinessTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedReadinessTarget")
            .field("profile_id", &self.profile_id)
            .field("model", &self.model)
            .field("method", &self.request.method)
            .field("timeout_ns", &self.timeout_ns)
            .field("interval_ns", &self.interval_ns)
            .finish_non_exhaustive()
    }
}

/// Native HTTP adapter from current endpoint-owned readiness policies.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeHttpReadinessPlanFactory;

impl OnlineReadinessPlanFactory for NativeHttpReadinessPlanFactory {
    fn prepare(&self, input: ReadinessPlanInput<'_>) -> Result<Box<dyn PreparedOnlineReadiness>> {
        let mut targets = Vec::new();

        for profile in input.profiles {
            let endpoint = input
                .endpoints
                .prepare(&profile.endpoint_id, profile.config.clone())
                .with_context(|| {
                    format!(
                        "preparing readiness endpoint profile {:?}",
                        profile.profile_id
                    )
                })?;
            let effective = endpoint.config().to_raw();
            if effective.wait_for_model_timeout <= 0.0 {
                continue;
            }
            ensure!(
                !input.models.is_empty(),
                "readiness profile {:?} requires at least one model",
                profile.profile_id
            );
            let timeout_ns =
                positive_seconds_to_ns(effective.wait_for_model_timeout, "wait_for_model_timeout")?;
            let interval_ns = positive_seconds_to_ns(
                effective.wait_for_model_interval,
                "wait_for_model_interval",
            )?;

            for base_url in &effective.urls {
                for model in input.models {
                    let requests = match endpoint.readiness_policy(model).with_context(|| {
                        format!(
                            "preparing readiness policy for profile {:?}, model {model:?}",
                            profile.profile_id
                        )
                    })? {
                        ReadinessPolicy::Request(request) => vec![request],
                        ReadinessPolicy::Requests(requests) => {
                            ensure!(
                                !requests.is_empty(),
                                "endpoint {:?} returned an empty readiness policy for profile {:?}, model {model:?}",
                                profile.endpoint_id,
                                profile.profile_id
                            );
                            requests
                        }
                        ReadinessPolicy::Unsupported { .. } => {
                            bail!(
                                "endpoint {:?} does not provide readiness for profile {:?}, model {model:?}",
                                profile.endpoint_id,
                                profile.profile_id
                            )
                        }
                    };
                    for request in requests {
                        let classifier = readiness_classifier(&request.success);
                        targets.push(prepare_target(
                            profile,
                            base_url,
                            model,
                            request,
                            timeout_ns,
                            interval_ns,
                            classifier,
                        )?);
                    }
                }
            }
        }

        Ok(Box::new(NativePreparedOnlineReadiness { targets }))
    }
}

fn prepare_target(
    profile: &ReadinessEndpointProfile,
    base_url: &str,
    model: &str,
    request: PreparedReadinessRequest,
    timeout_ns: i64,
    interval_ns: i64,
    classifier: Arc<dyn ReadinessResponseClassifier>,
) -> Result<PreparedReadinessTarget> {
    ensure!(
        request.path.starts_with('/'),
        "readiness path for profile {:?} must be absolute",
        profile.profile_id
    );
    let url = readiness_url(base_url, &request.path).with_context(|| {
        format!(
            "composing readiness URL for profile {:?}",
            profile.profile_id
        )
    })?;
    let request_timeout_floor_ns = match request.method {
        ReadinessMethod::Get => {
            ensure!(
                request.body.is_none(),
                "GET readiness request for profile {:?} cannot carry a JSON body",
                profile.profile_id
            );
            GET_REQUEST_TIMEOUT_FLOOR_NS
        }
        ReadinessMethod::Post => POST_REQUEST_TIMEOUT_FLOOR_NS,
    };
    let success = request.success.clone();
    // A models-listing probe accepts a responsive-but-listless server: when
    // GET /v1/models 404s (endpoint disabled or unimplemented), a plain 2xx on
    // the base URL proves the stack is up. Only this success mode declares the
    // fallback; inference/liveness probes never fall back.
    let fallback = if matches!(success, ReadinessSuccess::ModelListed(_)) {
        let fallback_url = readiness_url(base_url, "/").with_context(|| {
            format!(
                "composing readiness base-URL fallback for profile {:?}",
                profile.profile_id
            )
        })?;
        Some(ReadinessAttemptRequest {
            method: ReadinessMethod::Get,
            url: fallback_url,
            headers: request.headers.clone(),
            body: None,
            timeout_ns,
            connection_reuse: profile.connection_reuse,
            client: profile.client.clone(),
        })
    } else {
        None
    };
    Ok(PreparedReadinessTarget {
        profile_id: profile.profile_id.clone(),
        model: model.to_owned(),
        base_url: base_url.to_owned(),
        success,
        fallback,
        request: ReadinessAttemptRequest {
            method: request.method,
            url,
            headers: request.headers,
            body: request.body,
            timeout_ns,
            connection_reuse: profile.connection_reuse,
            client: profile.client.clone(),
        },
        timeout_ns,
        interval_ns,
        request_timeout_floor_ns,
        classifier,
    })
}

fn readiness_url(base_url: &str, path: &str) -> Result<String> {
    let mut url = Url::parse(base_url).with_context(|| format!("parsing base URL {base_url:?}"))?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "native HTTP readiness requires an http:// or https:// URL"
    );
    url.set_path(path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.into())
}

fn positive_seconds_to_ns(value: f64, field: &str) -> Result<i64> {
    ensure!(
        value.is_finite() && value > 0.0,
        "{field} must be finite and positive"
    );
    let nanoseconds = value * 1_000_000_000.0;
    ensure!(
        nanoseconds < i64::MAX as f64,
        "{field} exceeds the native Clock range"
    );
    let nanoseconds = nanoseconds.round() as i64;
    ensure!(
        nanoseconds > 0,
        "{field} must be at least one native Clock nanosecond"
    );
    Ok(nanoseconds)
}

#[derive(Debug)]
struct NativePreparedOnlineReadiness {
    targets: Vec<PreparedReadinessTarget>,
}

#[async_trait(?Send)]
impl PreparedOnlineReadiness for NativePreparedOnlineReadiness {
    fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    fn target_count(&self) -> usize {
        self.targets.len()
    }

    async fn wait(
        &self,
        clock: Rc<dyn Clock>,
        transport: Rc<dyn ReadinessTransport>,
    ) -> Result<ReadinessReport> {
        // Keep this operator-facing message stable for log consumers.
        tracing::info!(
            targets = self.targets.len(),
            "Waiting for endpoint readiness across {} target(s)",
            self.targets.len(),
        );
        let mut report = ReadinessReport::default();
        for target in &self.targets {
            let attempts = wait_for_target(target, clock.clone(), transport.as_ref()).await?;
            report.targets_ready += 1;
            report.attempts += attempts;
        }
        Ok(report)
    }
}

async fn wait_for_target(
    target: &PreparedReadinessTarget,
    clock: Rc<dyn Clock>,
    transport: &dyn ReadinessTransport,
) -> Result<usize> {
    let deadline_ns = clock
        .now_ns()
        .checked_add(target.timeout_ns)
        .ok_or_else(|| anyhow!("readiness deadline exceeds the native Clock range"))?;
    let mut attempts = 0usize;
    let mut last_response = ReadinessAttemptResponse::default();
    let interval_s = target.interval_ns as f64 / 1_000_000_000.0;

    loop {
        let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return Err(readiness_timeout(target, attempts, &last_response));
        }
        attempts += 1;
        let mut request = target.request.clone();
        request.timeout_ns =
            remaining_ns.min(target.interval_ns.max(target.request_timeout_floor_ns));
        let response = transport.execute(request).await;
        if target.classifier.is_ready(&response) {
            target.log_ready(attempts, &response);
            return Ok(attempts);
        }
        // Models-listing probes accept a responsive base URL when the model
        // list is unavailable (404). The fallback is a single extra GET within
        // the same attempt; a 2xx there proves liveness even without a catalog.
        if response.status == Some(404)
            && let Some(fallback) = &target.fallback
        {
            let fallback_remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
            if fallback_remaining_ns > 0 {
                let mut fallback_request = fallback.clone();
                fallback_request.timeout_ns = fallback_remaining_ns
                    .min(target.interval_ns.max(target.request_timeout_floor_ns));
                let fallback_response = transport.execute(fallback_request).await;
                if fallback_response
                    .status
                    .is_some_and(|status| (200..300).contains(&status))
                {
                    tracing::warn!(
                        base_url = %target.base_url,
                        status = fallback_response
                            .status
                            .expect("checked 2xx status is present"),
                        "readiness: /v1/models unavailable; base URL responded, accepting as ready"
                    );
                    return Ok(attempts);
                }
            }
        }
        target.log_retry(attempts, &response, interval_s);
        last_response = response;

        let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return Err(readiness_timeout(target, attempts, &last_response));
        }
        clock
            .clone()
            .sleep(target.interval_ns.min(remaining_ns))
            .await;
    }
}

fn readiness_timeout(
    target: &PreparedReadinessTarget,
    attempts: usize,
    response: &ReadinessAttemptResponse,
) -> anyhow::Error {
    let status = response
        .status
        .map_or_else(|| "connection error".to_owned(), |value| value.to_string());
    let diagnostic = response
        .error
        .as_deref()
        .unwrap_or("no response diagnostic");
    anyhow!(
        "Timed out waiting for endpoint readiness for profile {:?}, model {:?} at {} after {attempts} attempt(s); last status: {status} ({diagnostic})",
        target.profile_id,
        target.model,
        target.base_url
    )
}

/// Native Clock-injected HTTP readiness transport factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeHttpReadinessTransportFactory;

impl ReadinessTransportFactory for NativeHttpReadinessTransportFactory {
    fn build(&self, clock: Rc<dyn Clock>) -> Rc<dyn ReadinessTransport> {
        Rc::new(NativeHttpReadinessTransport {
            clock,
            transports: RefCell::new(Vec::new()),
        })
    }
}

struct NativeHttpReadinessTransport {
    clock: Rc<dyn Clock>,
    transports: RefCell<Vec<(ClientConfig, Rc<HttpTransport>)>>,
}

impl NativeHttpReadinessTransport {
    fn transport_for(&self, timeout_ns: i64, client: &ClientConfig) -> Rc<HttpTransport> {
        let mut effective = client.clone();
        effective.total_timeout_ns = Some(timeout_ns);
        if let Some((_, transport)) = self
            .transports
            .borrow()
            .iter()
            .find(|(candidate, _)| candidate == &effective)
        {
            return transport.clone();
        }
        let transport = Rc::new(HttpTransport::new(self.clock.clone(), effective.clone()));
        self.transports
            .borrow_mut()
            .push((effective, transport.clone()));
        transport
    }
}

impl fmt::Debug for NativeHttpReadinessTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeHttpReadinessTransport")
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl ReadinessTransport for NativeHttpReadinessTransport {
    async fn execute(&self, request: ReadinessAttemptRequest) -> ReadinessAttemptResponse {
        let transport = self.transport_for(request.timeout_ns, &request.client);
        let mut config = RequestConfig::new(request.url).reuse(request.connection_reuse);
        config.headers = request.headers;
        let record = match (request.method, request.body) {
            (ReadinessMethod::Get, _) => transport.get(&config).await,
            (ReadinessMethod::Post, Some(body)) => {
                transport.send_request(&config, body, false, |_| {}).await
            }
            (ReadinessMethod::Post, None) => {
                transport
                    .send_request_bytes(&config, Bytes::new(), false, |_| {})
                    .await
            }
        };
        ReadinessAttemptResponse {
            status: record.status,
            body: record.responses.iter().find_map(|response| match response {
                Response::Text(response) => Some(response.text.clone()),
                Response::Sse(_) => None,
            }),
            error: record.error.map(|error| error.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;

    use crate::transport::http::models::HttpVersion;

    use super::*;

    struct AdvancingClock {
        now_ns: Cell<i64>,
    }

    impl AdvancingClock {
        fn new() -> Self {
            Self {
                now_ns: Cell::new(0),
            }
        }
    }

    impl Clock for AdvancingClock {
        fn now_ns(&self) -> i64 {
            self.now_ns.get()
        }

        fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
            self.now_ns
                .set(self.now_ns.get().saturating_add(duration_ns.max(0)));
            Box::pin(async {})
        }
    }

    #[derive(Debug)]
    struct ScriptedTransport {
        responses: RefCell<VecDeque<ReadinessAttemptResponse>>,
        attempts: RefCell<Vec<ReadinessAttemptRequest>>,
    }

    impl ScriptedTransport {
        fn new(responses: impl IntoIterator<Item = ReadinessAttemptResponse>) -> Self {
            Self {
                responses: RefCell::new(responses.into_iter().collect()),
                attempts: RefCell::new(Vec::new()),
            }
        }
    }

    #[async_trait(?Send)]
    impl ReadinessTransport for ScriptedTransport {
        async fn execute(&self, request: ReadinessAttemptRequest) -> ReadinessAttemptResponse {
            self.attempts.borrow_mut().push(request);
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(ReadinessAttemptResponse {
                    status: Some(503),
                    body: None,
                    error: Some("script exhausted".into()),
                })
        }
    }

    fn profile(
        endpoint_id: &str,
        urls: Vec<String>,
        timeout_seconds: f64,
    ) -> ReadinessEndpointProfile {
        ReadinessEndpointProfile::new(
            "default",
            EndpointId::new(endpoint_id).unwrap(),
            RawEndpointConfig {
                urls,
                wait_for_model_timeout: timeout_seconds,
                wait_for_model_interval: if timeout_seconds > 0.0 { 2.0 } else { 5.0 },
                wait_for_model_mode: if timeout_seconds > 0.0 {
                    "models".into()
                } else {
                    "inference".into()
                },
                wait_for_model_interval_set: true,
                wait_for_model_mode_set: true,
                api_key: Some("readiness-secret".into()),
                ..RawEndpointConfig::default()
            },
            ConnectionReuseStrategy::Pooled,
            ClientConfig::default(),
        )
    }

    fn run_plan(
        plan: &dyn PreparedOnlineReadiness,
        clock: Rc<dyn Clock>,
        transport: Rc<dyn ReadinessTransport>,
    ) -> Result<ReadinessReport> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(plan.wait(clock, transport))
    }

    #[test]
    fn disabled_profiles_prepare_an_empty_side_effect_free_plan() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let profiles = [profile(
            "chat",
            vec!["http://example.test/v1/chat/completions".into()],
            0.0,
        )];
        let models = ["model-a".to_owned()];
        let plan = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap();

        assert!(plan.is_empty());
        assert_eq!(plan.target_count(), 0);
    }

    #[test]
    fn disabled_profiles_still_pass_through_endpoint_config_validation() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let mut invalid = profile(
            "chat",
            vec!["http://example.test/v1/chat/completions".into()],
            0.0,
        );
        invalid.config.wait_for_model_interval = 2.0;
        let profiles = [invalid];
        let models = ["model-a".to_owned()];

        let error = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap_err();
        let diagnostic = format!("{error:#}");

        assert!(diagnostic.contains("has no effect"), "{diagnostic}");
    }

    #[test]
    fn chat_models_readiness_waits_until_the_requested_model_is_listed() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let profiles = [profile(
            "chat",
            vec!["http://example.test/v1/chat/completions".into()],
            10.0,
        )];
        let models = ["model-a".to_owned()];

        let plan = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap();
        let clock = Rc::new(AdvancingClock::new());
        let transport = Rc::new(ScriptedTransport::new([
            ReadinessAttemptResponse {
                status: Some(200),
                body: Some(r#"{"data":[{"id":"another-model"}]}"#.into()),
                error: None,
            },
            ReadinessAttemptResponse {
                status: Some(200),
                body: Some(r#"{"data":[{"id":"model-a"}]}"#.into()),
                error: None,
            },
        ]));

        let report = run_plan(plan.as_ref(), clock, transport.clone()).unwrap();

        assert_eq!(report.targets_ready, 1);
        assert_eq!(report.attempts, 2);
        let attempts = transport.attempts.borrow();
        assert_eq!(attempts.len(), 2);
        for attempt in attempts.iter() {
            assert_eq!(attempt.method(), ReadinessMethod::Get);
            assert_eq!(attempt.url(), "http://example.test/v1/models");
            assert!(attempt.body().is_none());
        }
    }

    #[test]
    fn chat_models_readiness_falls_back_to_base_url_on_404() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let profiles = [profile(
            "chat",
            vec!["http://example.test/v1/chat/completions".into()],
            10.0,
        )];
        let models = ["model-a".to_owned()];

        let plan = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap();
        let clock = Rc::new(AdvancingClock::new());
        // Primary GET /v1/models 404s (endpoint disabled); the base-URL GET
        // then answers 2xx, which the models probe accepts as ready.
        let transport = Rc::new(ScriptedTransport::new([
            ReadinessAttemptResponse {
                status: Some(404),
                body: Some(r#"{"detail":"Not Found"}"#.into()),
                error: None,
            },
            ReadinessAttemptResponse {
                status: Some(200),
                body: Some(r#"{"message":"AIPerf Mock Server"}"#.into()),
                error: None,
            },
        ]));

        let report = run_plan(plan.as_ref(), clock, transport.clone()).unwrap();

        assert_eq!(report.targets_ready, 1);
        // Both the primary probe and the single fallback GET land in one attempt.
        assert_eq!(report.attempts, 1);
        let attempts = transport.attempts.borrow();
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].method(), ReadinessMethod::Get);
        assert_eq!(attempts[0].url(), "http://example.test/v1/models");
        assert_eq!(attempts[1].method(), ReadinessMethod::Get);
        assert_eq!(attempts[1].url(), "http://example.test/");
        assert!(attempts[1].body().is_none());
    }

    #[test]
    fn chat_inference_readiness_accepts_a_non_server_error_response() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let mut readiness_profile = profile(
            "chat",
            vec!["http://example.test/v1/chat/completions".into()],
            10.0,
        );
        readiness_profile.config.wait_for_model_mode = "inference".into();
        let profiles = [readiness_profile];
        let models = ["model-a".to_owned()];
        let plan = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap();
        let transport = Rc::new(ScriptedTransport::new([ReadinessAttemptResponse {
            status: Some(401),
            body: Some(r#"{"error":"unauthorized"}"#.into()),
            error: None,
        }]));

        let report = run_plan(
            plan.as_ref(),
            Rc::new(AdvancingClock::new()),
            transport.clone(),
        )
        .unwrap();

        assert_eq!(report.targets_ready, 1);
        assert_eq!(report.attempts, 1);
        let attempt = transport.attempts.borrow().last().cloned().unwrap();
        assert_eq!(attempt.method(), ReadinessMethod::Post);
        assert_eq!(attempt.url(), "http://example.test/v1/chat/completions");
        assert_eq!(attempt.body().unwrap()["model"], "model-a");
        assert_eq!(attempt.body().unwrap()["max_tokens"], 1);
    }

    #[test]
    fn kserve_policy_retries_on_server_error_with_injected_clock_and_transport() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let profiles = [profile(
            "kserve_chat",
            vec!["http://example.test:8080/old/path?discard=yes".into()],
            10.0,
        )];
        let models = ["model-a".to_owned()];
        let plan = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap();
        let clock = Rc::new(AdvancingClock::new());
        let transport = Rc::new(ScriptedTransport::new([
            ReadinessAttemptResponse {
                status: Some(503),
                body: Some("loading".into()),
                error: Some("server unavailable".into()),
            },
            ReadinessAttemptResponse {
                status: Some(204),
                body: None,
                error: None,
            },
        ]));

        let report = run_plan(plan.as_ref(), clock.clone(), transport.clone()).unwrap();

        assert_eq!(report.targets_ready, 1);
        assert_eq!(report.attempts, 2);
        assert_eq!(clock.now_ns(), 2_000_000_000);
        let attempts = transport.attempts.borrow();
        assert_eq!(attempts.len(), 2);
        for attempt in attempts.iter() {
            assert_eq!(attempt.method(), ReadinessMethod::Get);
            assert_eq!(attempt.url(), "http://example.test:8080/openai/v1/models");
            assert_eq!(attempt.timeout_ns(), GET_REQUEST_TIMEOUT_FLOOR_NS);
            assert_eq!(
                attempt.headers().get("Authorization").map(String::as_str),
                Some("Bearer readiness-secret")
            );
            assert!(attempt.body().is_none());
        }
    }

    #[test]
    fn timeout_is_per_target_and_caps_the_last_attempt_deadline() {
        let endpoints = EndpointRegistry::builtin().unwrap();
        let profiles = [profile(
            "kserve_chat",
            vec!["http://example.test".into()],
            3.0,
        )];
        let models = ["model-a".to_owned()];
        let plan = NativeHttpReadinessPlanFactory
            .prepare(ReadinessPlanInput {
                endpoints: &endpoints,
                profiles: &profiles,
                models: &models,
            })
            .unwrap();
        let clock = Rc::new(AdvancingClock::new());
        let transport = Rc::new(ScriptedTransport::new([]));

        let error = run_plan(plan.as_ref(), clock.clone(), transport.clone())
            .unwrap_err()
            .to_string();

        assert!(error.contains("after 2 attempt(s)"), "{error}");
        assert_eq!(clock.now_ns(), 3_000_000_000);
        let attempts = transport.attempts.borrow();
        assert_eq!(attempts[0].timeout_ns(), 3_000_000_000);
        assert_eq!(attempts[1].timeout_ns(), 1_000_000_000);
    }

    #[test]
    fn native_transport_factory_retains_the_injected_clock() {
        let clock = Rc::new(AdvancingClock::new());
        let transport = NativeHttpReadinessTransportFactory.build(clock);
        assert!(format!("{transport:?}").contains("NativeHttpReadinessTransport"));
    }

    #[test]
    fn native_transport_reuses_one_pool_for_equal_attempt_policy() {
        let transport = NativeHttpReadinessTransport {
            clock: Rc::new(AdvancingClock::new()),
            transports: RefCell::new(Vec::new()),
        };

        let client = ClientConfig {
            ssl_verify: false,
            max_connections_per_origin: 7,
            keepalive_ns: Some(250_000_000),
            http_version: HttpVersion::Http2PriorKnowledge,
            ..ClientConfig::default()
        };
        let first = transport.transport_for(5_000_000_000, &client);
        let second = transport.transport_for(5_000_000_000, &client);
        let capped = transport.transport_for(1_000_000_000, &client);

        assert!(Rc::ptr_eq(&first, &second));
        assert!(!Rc::ptr_eq(&first, &capped));
        assert_eq!(transport.transports.borrow().len(), 2);
    }
}
