// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared endpoint-local control hooks over profile-bound control-plane HTTP.

use std::cell::Cell;
use std::fmt::{self, Debug, Formatter};
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use tracing::debug;
use url::Url;

use crate::clock::Clock;
use crate::endpoints::{EndpointId, RawEndpointConfig, ResetKvCacheConfig, ServerProfilerConfig};
use crate::engine::control_plane_http::{
    ControlPlaneCredentialReference, ControlPlaneHttp, ControlPlaneHttpErrorKind,
    ControlPlaneHttpProvider, ControlPlaneMethod, ControlPlaneRequest, ControlPlaneTlsReference,
    LocalCancellationSignal, ValidatedControlPlaneProfile,
};
use crate::engine::registry::ValidatedEndpointProfileV2;
use crate::graph::replay::ReplayRunIdentity;
use crate::graph::tools::{ContainerRuntime, cleanup_recorded_agent_containers};
use crate::timing::LocalPhaseFuture;
use crate::transport::core::ConnectionReuseStrategy;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::HttpVersion;

const DEFAULT_CONTROL_HOOK_TIMEOUT_NS: i64 = 30_000_000_000;
const DEFAULT_RESET_KV_CACHE_MAX_RETRY_NS: i64 = 60_000_000_000;
const RETRY_BACKOFF_INITIAL_NS: i64 = 1_000_000_000;
const RETRY_BACKOFF_CAP_NS: i64 = 8_000_000_000;
const RETRY_BACKOFF_MULTIPLIER: i64 = 2;
/// HTTP statuses treated as transient for `reset_kv_cache` and retried.
///
/// 409 Conflict / 423 Locked / 429 Too Many Requests / 503 Service Unavailable
/// are the standard "busy with transient state, try again" signals - e.g. a
/// server reporting a profiler-cleanup race explicitly instead of holding the
/// socket open. Any other non-2xx status is a real rejection.
const RESET_KV_CACHE_RETRYABLE_STATUS_CODES: [u16; 4] = [409, 423, 429, 503];
const DEFAULT_RESET_KV_CACHE_PATH: &str = "/reset_prefix_cache";
const DEFAULT_SERVER_PROFILER_START_PATH: &str = "/start_profile";
const DEFAULT_SERVER_PROFILER_STOP_PATH: &str = "/stop_profile";
const CONTROL_RESPONSE_MAX_BYTES: usize = 64 * 1024;

type PreparedControlPlaneHandles = (Vec<Rc<dyn ControlPlaneHttp>>, Vec<String>);

/// Perform restart or signal cleanup for exactly one persisted replay run.
pub async fn cleanup_recorded_agent_docker_on_shutdown(
    runtime: &dyn ContainerRuntime,
    run_identity: &ReplayRunIdentity,
) -> Result<()> {
    cleanup_recorded_agent_containers(runtime, run_identity)
        .await
        .map_err(anyhow::Error::from)
}

/// Aggregate outcome for one logical control-hook invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlHookOutcome {
    /// Number of endpoint-local control requests that completed successfully.
    pub request_count: usize,
}

/// Prepared reset-kv-cache hook bound to one run clock and endpoint handle set.
#[derive(Clone)]
pub struct PreparedResetKvCacheHook {
    pub timeout_ns: i64,
    pub path: String,
    /// Total budget for retrying a retryable POST against one endpoint origin.
    pub max_retry_ns: i64,
    pub handles: Vec<Rc<dyn ControlPlaneHttp>>,
    clock: Rc<dyn Clock>,
    target_urls: Vec<String>,
}

impl PreparedResetKvCacheHook {
    /// Display the normalized control path that will be requested.
    pub fn display_path(&self) -> &str {
        &self.path
    }
}

impl Debug for PreparedResetKvCacheHook {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedResetKvCacheHook")
            .field("timeout_ns", &self.timeout_ns)
            .field("path", &self.path)
            .field("max_retry_ns", &self.max_retry_ns)
            .field("handle_count", &self.handles.len())
            .field("target_urls", &self.target_urls)
            .finish()
    }
}

/// Prepared server-profiler hook bound to one run clock and endpoint handle set.
#[derive(Clone)]
pub struct PreparedServerProfilerHook {
    pub timeout_ns: i64,
    pub start_path: String,
    pub stop_path: String,
    pub handles: Vec<Rc<dyn ControlPlaneHttp>>,
    clock: Rc<dyn Clock>,
    target_urls: Vec<String>,
}

impl PreparedServerProfilerHook {
    /// Display the normalized control path used to start profiling.
    pub fn start_path(&self) -> &str {
        &self.start_path
    }

    /// Display the normalized control path used to stop profiling.
    pub fn stop_path(&self) -> &str {
        &self.stop_path
    }

    /// Build a request-free hook for ownership-state tests.
    #[cfg(test)]
    pub(crate) fn empty_for_test(clock: Rc<dyn Clock>) -> Self {
        Self {
            timeout_ns: DEFAULT_CONTROL_HOOK_TIMEOUT_NS,
            start_path: DEFAULT_SERVER_PROFILER_START_PATH.to_owned(),
            stop_path: DEFAULT_SERVER_PROFILER_STOP_PATH.to_owned(),
            handles: Vec::new(),
            clock,
            target_urls: Vec::new(),
        }
    }
}

impl Debug for PreparedServerProfilerHook {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedServerProfilerHook")
            .field("timeout_ns", &self.timeout_ns)
            .field("start_path", &self.start_path)
            .field("stop_path", &self.stop_path)
            .field("handle_count", &self.handles.len())
            .field("target_urls", &self.target_urls)
            .finish()
    }
}

/// Run-local ownership of one server-profiler session across overlapping phases.
///
/// Phase setup is serialized by the phase orchestrator, so the worker-local
/// ownership count needs no synchronization. The first owner starts the remote
/// profiler and the last owner to drain stops it.
pub(crate) struct ServerProfilerCoordinator {
    hook: PreparedServerProfilerHook,
    owners: Cell<usize>,
}

impl ServerProfilerCoordinator {
    /// Bind one prepared hook to a fresh run-local ownership set.
    pub(crate) fn new(hook: PreparedServerProfilerHook) -> Self {
        Self {
            hook,
            owners: Cell::new(0),
        }
    }

    /// Acquire profiler ownership for one phase after its setup gate opens.
    pub(crate) fn acquire(self: &Rc<Self>) -> LocalPhaseFuture<Result<()>> {
        let coordinator = self.clone();
        Box::pin(async move {
            let owner_count = coordinator.owners.get();
            if owner_count == 0 {
                start_server_profiler(&coordinator.hook).await?;
            }
            let next_count = owner_count
                .checked_add(1)
                .ok_or_else(|| anyhow!("server profiler ownership count overflow"))?;
            coordinator.owners.set(next_count);
            Ok(())
        })
    }

    /// Release one drained phase and stop the profiler after the last owner.
    pub(crate) fn release(self: &Rc<Self>) -> LocalPhaseFuture<Result<()>> {
        let coordinator = self.clone();
        Box::pin(async move {
            let owner_count = coordinator.owners.get();
            ensure!(
                owner_count > 0,
                "server profiler ownership released without an active owner"
            );
            coordinator.owners.set(owner_count - 1);
            if owner_count == 1 {
                stop_server_profiler(&coordinator.hook).await?;
            }
            Ok(())
        })
    }

    /// Stop one still-owned profiler at a terminal run barrier.
    pub(crate) fn force_stop(self: &Rc<Self>) -> LocalPhaseFuture<Result<()>> {
        let coordinator = self.clone();
        Box::pin(async move {
            if coordinator.owners.replace(0) > 0 {
                stop_server_profiler(&coordinator.hook).await?;
            }
            Ok(())
        })
    }

    /// Whether at least one phase currently owns the profiler.
    pub(crate) fn has_owners(&self) -> bool {
        self.owners.get() > 0
    }
}

/// Prepared endpoint-local control hooks shared by reset and profiler lifecycles.
pub struct PreparedEndpointControlHooks {
    pub reset_kv_cache: Option<PreparedResetKvCacheHook>,
    pub server_profiler: Option<PreparedServerProfilerHook>,
}

impl Debug for PreparedEndpointControlHooks {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedEndpointControlHooks")
            .field("reset_kv_cache", &self.reset_kv_cache)
            .field("server_profiler", &self.server_profiler)
            .finish()
    }
}

#[derive(Clone, Deserialize)]
struct ControlHookProfileValue {
    #[serde(rename = "type")]
    endpoint_id: String,
    urls: Vec<String>,
    timeout_seconds: f64,
    ssl_verify: bool,
    #[serde(default)]
    http2: bool,
    connection_limit: usize,
    keepalive_timeout: f64,
    #[serde(default)]
    max_connect_retries: u32,
    #[serde(default)]
    connect_retry_backoff_seconds: f64,
    #[serde(default)]
    reset_kv_cache: Option<ResetKvCacheConfig>,
    #[serde(default)]
    server_profiler: Option<ServerProfilerConfig>,
}

/// Prepare shared endpoint-local control hooks against one profile-local provider.
pub fn prepare_endpoint_control_hooks(
    clock: Rc<dyn Clock>,
    control_plane: &dyn ControlPlaneHttpProvider,
    profile: &ValidatedEndpointProfileV2,
) -> Result<PreparedEndpointControlHooks> {
    if profile.config.reset_kv_cache.is_none() && profile.config.server_profiler.is_none() {
        return Ok(PreparedEndpointControlHooks {
            reset_kv_cache: None,
            server_profiler: None,
        });
    }
    let (handles, target_urls) =
        prepare_handles(control_plane, &profile.config.urls, &profile.client)?;
    let reset_kv_cache = profile
        .config
        .reset_kv_cache
        .as_ref()
        .map(|config| prepare_reset_hook(config, &handles, &target_urls, &clock))
        .transpose()?;
    let server_profiler = profile
        .config
        .server_profiler
        .as_ref()
        .map(|config| prepare_server_profiler_hook(config, &handles, &target_urls, &clock))
        .transpose()?;
    Ok(PreparedEndpointControlHooks {
        reset_kv_cache,
        server_profiler,
    })
}

/// Prepare shared endpoint-local control hooks from one authored endpoint-profile object.
pub fn prepare_endpoint_control_hooks_from_profile_value(
    clock: Rc<dyn Clock>,
    control_plane: &dyn ControlPlaneHttpProvider,
    profile_value: &serde_json::Value,
) -> Result<PreparedEndpointControlHooks> {
    prepare_endpoint_control_hooks(
        clock,
        control_plane,
        &validated_profile_from_value(profile_value)?,
    )
}

/// Execute one prepared reset-kv-cache hook across every endpoint-local handle.
///
/// A transport failure, a deadline expiry, or a transient-busy status is
/// retried per origin with bounded exponential backoff inside
/// [`PreparedResetKvCacheHook::max_retry_ns`]; any other non-2xx fails fast.
pub fn run_reset_kv_cache(
    hook: &PreparedResetKvCacheHook,
) -> LocalPhaseFuture<Result<ControlHookOutcome>> {
    let timeout_ns = hook.timeout_ns;
    let path = hook.path.clone();
    let handles = hook.handles.clone();
    let clock = hook.clock.clone();
    let target_urls = hook.target_urls.clone();
    let max_retry_ns = hook.max_retry_ns;
    Box::pin(async move {
        execute_control_hook(
            "reset_kv_cache",
            timeout_ns,
            path,
            handles,
            target_urls,
            clock,
            max_retry_ns,
        )
        .await
    })
}

/// Execute one prepared profiler-start hook across every endpoint-local handle.
pub fn start_server_profiler(
    hook: &PreparedServerProfilerHook,
) -> LocalPhaseFuture<Result<ControlHookOutcome>> {
    let timeout_ns = hook.timeout_ns;
    let path = hook.start_path.clone();
    let handles = hook.handles.clone();
    let clock = hook.clock.clone();
    let target_urls = hook.target_urls.clone();
    Box::pin(async move {
        execute_control_hook(
            "server_profiler.start",
            timeout_ns,
            path,
            handles,
            target_urls,
            clock,
            0,
        )
        .await
    })
}

/// Execute one prepared profiler-stop hook across every endpoint-local handle.
pub fn stop_server_profiler(
    hook: &PreparedServerProfilerHook,
) -> LocalPhaseFuture<Result<ControlHookOutcome>> {
    let timeout_ns = hook.timeout_ns;
    let path = hook.stop_path.clone();
    let handles = hook.handles.clone();
    let clock = hook.clock.clone();
    let target_urls = hook.target_urls.clone();
    Box::pin(async move {
        execute_control_hook(
            "server_profiler.stop",
            timeout_ns,
            path,
            handles,
            target_urls,
            clock,
            0,
        )
        .await
    })
}

fn prepare_handles(
    control_plane: &dyn ControlPlaneHttpProvider,
    endpoint_urls: &[String],
    endpoint_client: &ClientConfig,
) -> Result<PreparedControlPlaneHandles> {
    let mut handles = Vec::with_capacity(endpoint_urls.len());
    let mut target_urls = Vec::with_capacity(endpoint_urls.len());
    for endpoint_url in endpoint_urls {
        let target_url = control_plane_base_url(endpoint_url)?;
        let handle = control_plane
            .prepare(control_plane_profile(&target_url, endpoint_client)?)
            .with_context(|| {
                format!("preparing endpoint-local control-plane handle for {target_url:?}")
            })?;
        handles.push(handle);
        target_urls.push(target_url);
    }
    Ok((handles, target_urls))
}

fn control_plane_base_url(endpoint_url: &str) -> Result<String> {
    let mut parsed = Url::parse(endpoint_url)
        .with_context(|| format!("parsing endpoint URL {endpoint_url:?}"))?;
    ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "endpoint-local control hooks require http:// or https:// endpoint URLs"
    );
    parsed.set_path("/");
    parsed.set_query(None);
    parsed.set_fragment(None);
    Ok(parsed.into())
}

fn control_plane_profile(
    base_url: &str,
    endpoint_client: &ClientConfig,
) -> Result<ValidatedControlPlaneProfile> {
    let mut client = endpoint_client.clone();
    client.max_connections_per_origin = 1;
    ValidatedControlPlaneProfile::new(
        Url::parse(base_url).with_context(|| format!("parsing control-plane URL {base_url:?}"))?,
        client,
        ControlPlaneCredentialReference::None,
        ControlPlaneTlsReference::default(),
        vec!["application/json".to_owned(), "text/plain".to_owned()],
        vec!["identity".to_owned()],
        CONTROL_RESPONSE_MAX_BYTES,
    )
    .map_err(|error| anyhow!("preparing endpoint-local control-plane profile: {error}"))
}

fn validated_profile_from_value(
    profile_value: &serde_json::Value,
) -> Result<ValidatedEndpointProfileV2> {
    let value: ControlHookProfileValue = serde_json::from_value(profile_value.clone())
        .context("decoding endpoint-local control-hook profile")?;
    ensure!(
        !value.urls.is_empty(),
        "endpoint-local control hooks require at least one endpoint URL"
    );
    ensure!(
        value.connection_limit > 0,
        "endpoint-local control hooks require a positive connection_limit"
    );
    Ok(ValidatedEndpointProfileV2 {
        profile_id: "default".to_owned(),
        endpoint_id: EndpointId::new(&value.endpoint_id)?,
        config: RawEndpointConfig {
            urls: value.urls,
            timeout_seconds: value.timeout_seconds,
            reset_kv_cache: value.reset_kv_cache,
            server_profiler: value.server_profiler,
            ..RawEndpointConfig::default()
        },
        connection_reuse: ConnectionReuseStrategy::Pooled,
        client: ClientConfig {
            total_timeout_ns: Some(seconds_to_ns(value.timeout_seconds)?),
            ssl_verify: value.ssl_verify,
            http_version: if value.http2 {
                HttpVersion::Http2PriorKnowledge
            } else {
                HttpVersion::Auto
            },
            keepalive_ns: Some(seconds_to_ns(value.keepalive_timeout)?),
            max_connections_per_origin: value.connection_limit,
            max_connect_retries: value.max_connect_retries,
            connect_retry_backoff_ns: nonnegative_seconds_to_ns(
                value.connect_retry_backoff_seconds,
                "endpoint.connect_retry_backoff_seconds",
            )?,
            ..ClientConfig::default()
        },
        session_header: None,
    })
}

/// Resolve one control-hook timeout, in nanoseconds.
///
/// An unset `timeout_seconds` deliberately does not inherit `endpoint.timeout`:
/// that value is tuned for inference requests and defaults to six hours, so a
/// stalled control-plane POST would appear to hang for the whole run.
fn resolve_timeout_ns(timeout_seconds: Option<f64>) -> Result<i64> {
    let timeout_ns = match timeout_seconds {
        Some(seconds) => seconds_to_ns(seconds)?,
        None => DEFAULT_CONTROL_HOOK_TIMEOUT_NS,
    };
    ensure!(
        timeout_ns > 0,
        "endpoint-local control hook timeout must be positive"
    );
    Ok(timeout_ns)
}

fn normalize_control_path(path: Option<&str>, default_path: &str, field: &str) -> Result<String> {
    let path = path.unwrap_or(default_path);
    ensure!(
        !path.is_empty() && path.starts_with('/'),
        "{field} must start with a leading slash"
    );
    ensure!(
        !path.chars().any(char::is_whitespace) && !path.contains(['?', '#']),
        "{field} must be a path-only control endpoint"
    );
    Ok(path.to_owned())
}

fn seconds_to_ns(seconds: f64) -> Result<i64> {
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "endpoint-local control hook timeout_seconds must be finite and positive"
    );
    let timeout_ns = (seconds * 1_000_000_000.0).round();
    ensure!(
        timeout_ns < i64::MAX as f64 && timeout_ns >= 1.0,
        "endpoint-local control hook timeout_seconds exceeds the native Clock range"
    );
    Ok(timeout_ns as i64)
}

fn nonnegative_seconds_to_ns(seconds: f64, field: &str) -> Result<i64> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0,
        "{field} must be finite and non-negative"
    );
    let timeout_ns = (seconds * 1_000_000_000.0).round();
    ensure!(
        timeout_ns < i64::MAX as f64 && timeout_ns >= 0.0,
        "{field} exceeds the native Clock range"
    );
    Ok(timeout_ns as i64)
}

fn prepare_reset_hook(
    config: &ResetKvCacheConfig,
    handles: &[Rc<dyn ControlPlaneHttp>],
    target_urls: &[String],
    clock: &Rc<dyn Clock>,
) -> Result<PreparedResetKvCacheHook> {
    Ok(PreparedResetKvCacheHook {
        timeout_ns: resolve_timeout_ns(config.timeout_seconds)?,
        path: normalize_control_path(
            config.path.as_deref(),
            DEFAULT_RESET_KV_CACHE_PATH,
            "endpoint.reset_kv_cache.path",
        )?,
        max_retry_ns: match config.max_retry_seconds {
            Some(seconds) => nonnegative_seconds_to_ns(
                seconds,
                "endpoint.reset_kv_cache.max_retry_seconds",
            )?,
            None => DEFAULT_RESET_KV_CACHE_MAX_RETRY_NS,
        },
        handles: handles.to_vec(),
        clock: clock.clone(),
        target_urls: target_urls.to_vec(),
    })
}

fn prepare_server_profiler_hook(
    config: &ServerProfilerConfig,
    handles: &[Rc<dyn ControlPlaneHttp>],
    target_urls: &[String],
    clock: &Rc<dyn Clock>,
) -> Result<PreparedServerProfilerHook> {
    Ok(PreparedServerProfilerHook {
        timeout_ns: resolve_timeout_ns(config.timeout_seconds)?,
        start_path: normalize_control_path(
            config.start_path.as_deref(),
            DEFAULT_SERVER_PROFILER_START_PATH,
            "endpoint.server_profiler.start_path",
        )?,
        stop_path: normalize_control_path(
            config.stop_path.as_deref(),
            DEFAULT_SERVER_PROFILER_STOP_PATH,
            "endpoint.server_profiler.stop_path",
        )?,
        handles: handles.to_vec(),
        clock: clock.clone(),
        target_urls: target_urls.to_vec(),
    })
}

/// One failed control-hook attempt plus whether waiting could change it.
struct ControlHookAttemptError {
    error: anyhow::Error,
    is_retryable: bool,
}

async fn attempt_control_request(
    handle: &dyn ControlPlaneHttp,
    kind: &'static str,
    index: usize,
    path: &str,
    timeout_ns: i64,
    target_url: &str,
    clock: &Rc<dyn Clock>,
    retryable_status_codes: &[u16],
) -> std::result::Result<(), ControlHookAttemptError> {
    let absolute_deadline_ns = clock.now_ns().saturating_add(timeout_ns);
    let response = handle
        .execute(
            ControlPlaneRequest {
                request_id: format!("control-hook-{kind}-{index}"),
                method: ControlPlaneMethod::Post,
                path: path.to_owned(),
            },
            absolute_deadline_ns,
            LocalCancellationSignal::new(),
        )
        .await
        .map_err(|error| ControlHookAttemptError {
            // A transport failure or an expired per-attempt deadline can clear
            // on its own; an invalid request or oversized reply cannot.
            is_retryable: matches!(
                error.kind,
                ControlPlaneHttpErrorKind::Transport | ControlPlaneHttpErrorKind::Timeout
            ),
            error: anyhow::Error::new(error).context(format!(
                "executing endpoint-local {kind} hook against {target_url:?} at {path:?}"
            )),
        })?;
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    Err(ControlHookAttemptError {
        is_retryable: retryable_status_codes.contains(&response.status),
        error: anyhow!(
            "endpoint-local {kind} hook against {target_url:?} at {path:?} returned HTTP {}",
            response.status
        ),
    })
}

async fn execute_control_hook(
    kind: &'static str,
    timeout_ns: i64,
    path: String,
    handles: Vec<Rc<dyn ControlPlaneHttp>>,
    target_urls: Vec<String>,
    clock: Rc<dyn Clock>,
    max_retry_ns: i64,
) -> Result<ControlHookOutcome> {
    // Only reset_kv_cache opts into transient-busy status retries; a profiler
    // hook passes an empty set so every non-2xx stays fatal.
    let retryable_status_codes: &[u16] = if max_retry_ns > 0 {
        &RESET_KV_CACHE_RETRYABLE_STATUS_CODES
    } else {
        &[]
    };
    let request_count = handles.len();
    for (index, (handle, target_url)) in handles.into_iter().zip(target_urls).enumerate() {
        let retry_deadline_ns = clock.now_ns().saturating_add(max_retry_ns);
        let mut backoff_ns = RETRY_BACKOFF_INITIAL_NS;
        loop {
            let attempt = attempt_control_request(
                handle.as_ref(),
                kind,
                index,
                &path,
                timeout_ns,
                &target_url,
                &clock,
                retryable_status_codes,
            )
            .await;
            let failure = match attempt {
                Ok(()) => break,
                Err(failure) => failure,
            };
            if !failure.is_retryable
                || clock.now_ns().saturating_add(backoff_ns) >= retry_deadline_ns
            {
                return Err(failure.error);
            }
            debug!(
                kind,
                target_url,
                backoff_ns,
                error = %failure.error,
                "retrying endpoint-local control hook"
            );
            clock.clone().sleep(backoff_ns).await;
            backoff_ns = backoff_ns
                .saturating_mul(RETRY_BACKOFF_MULTIPLIER)
                .min(RETRY_BACKOFF_CAP_NS);
        }
    }
    Ok(ControlHookOutcome { request_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use crate::endpoints::{EndpointId, RawEndpointConfig};
    use crate::engine::control_plane_http::{
        ControlPlaneHttpError, ControlPlaneHttpProvider, ControlPlaneResponse,
        ControlPlaneTransportTimings,
    };
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::cell::RefCell;
    use std::collections::{BTreeMap, VecDeque};

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct RecordedRequest {
        request_id: String,
        method: ControlPlaneMethod,
        path: String,
        absolute_deadline_ns: i64,
    }

    #[derive(Clone, Debug, Default)]
    struct RecordingState {
        prepared_urls: Rc<RefCell<Vec<String>>>,
        requests: Rc<RefCell<Vec<RecordedRequest>>>,
        /// Statuses returned in order; the last one repeats once exhausted.
        scripted_statuses: Rc<RefCell<VecDeque<u16>>>,
    }

    #[derive(Debug)]
    struct RecordingProvider {
        state: RecordingState,
    }

    impl RecordingProvider {
        fn new() -> Self {
            Self {
                state: RecordingState::default(),
            }
        }

        /// Build a provider whose handle replays `statuses` in order.
        fn with_statuses(statuses: &[u16]) -> Self {
            let provider = Self::new();
            *provider.state.scripted_statuses.borrow_mut() = statuses.iter().copied().collect();
            provider
        }
    }

    impl ControlPlaneHttpProvider for RecordingProvider {
        fn prepare(
            &self,
            profile: ValidatedControlPlaneProfile,
        ) -> std::result::Result<
            Rc<dyn ControlPlaneHttp>,
            crate::engine::control_plane_http::ControlPlanePrepareError,
        > {
            self.state
                .prepared_urls
                .borrow_mut()
                .push(profile.display_url().to_owned());
            Ok(Rc::new(RecordingHandle {
                state: self.state.clone(),
            }))
        }
    }

    #[derive(Debug)]
    struct RecordingHandle {
        state: RecordingState,
    }

    #[async_trait(?Send)]
    impl ControlPlaneHttp for RecordingHandle {
        async fn execute(
            &self,
            request: ControlPlaneRequest,
            absolute_deadline_ns: i64,
            _cancellation: LocalCancellationSignal,
        ) -> std::result::Result<ControlPlaneResponse, ControlPlaneHttpError> {
            self.state.requests.borrow_mut().push(RecordedRequest {
                request_id: request.request_id,
                method: request.method,
                path: request.path,
                absolute_deadline_ns,
            });
            let mut scripted = self.state.scripted_statuses.borrow_mut();
            let status = if scripted.len() > 1 {
                scripted.pop_front().unwrap_or(204)
            } else {
                scripted.front().copied().unwrap_or(204)
            };
            Ok(ControlPlaneResponse {
                status,
                headers: BTreeMap::new(),
                encoded_body: Bytes::new(),
                timings: ControlPlaneTransportTimings::default(),
            })
        }
    }

    fn client() -> ClientConfig {
        ClientConfig {
            total_timeout_ns: Some(9_000_000_000),
            max_connections_per_origin: 8,
            ..ClientConfig::default()
        }
    }

    fn validated_profile_with_paths(base_url: &str) -> ValidatedEndpointProfileV2 {
        ValidatedEndpointProfileV2 {
            profile_id: "default".to_owned(),
            endpoint_id: EndpointId::new("chat").unwrap(),
            config: RawEndpointConfig {
                urls: vec![base_url.to_owned()],
                reset_kv_cache: Some(ResetKvCacheConfig {
                    timeout_seconds: None,
                    path: Some("/reset_prefix_cache".to_owned()),
                    max_retry_seconds: None,
                }),
                server_profiler: Some(ServerProfilerConfig {
                    timeout_seconds: None,
                    start_path: Some("/start_profile".to_owned()),
                    stop_path: Some("/stop_profile".to_owned()),
                }),
                ..RawEndpointConfig::default()
            },
            client: client(),
            connection_reuse: crate::transport::core::ConnectionReuseStrategy::default(),
            session_header: None,
        }
    }

    #[test]
    fn prepared_control_hooks_join_relative_paths_against_endpoint_origins() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider = RecordingProvider::new();
        let hooks = prepare_endpoint_control_hooks(
            clock,
            &provider,
            &validated_profile_with_paths(
                "http://127.0.0.1:8000/v1/chat/completions?trace=keep-me-out",
            ),
        )
        .expect("hooks prepare");

        assert_eq!(
            provider.state.prepared_urls.borrow().as_slice(),
            &["http://127.0.0.1:8000/"]
        );
        assert_eq!(
            hooks
                .reset_kv_cache
                .as_ref()
                .expect("reset hook")
                .display_path(),
            "/reset_prefix_cache"
        );
        assert_eq!(
            hooks
                .server_profiler
                .as_ref()
                .expect("profiler hook")
                .start_path(),
            "/start_profile"
        );
        assert_eq!(
            hooks
                .server_profiler
                .as_ref()
                .expect("profiler hook")
                .stop_path(),
            "/stop_profile"
        );
    }

    #[tokio::test]
    async fn prepared_control_hook_runners_post_configured_paths() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider = RecordingProvider::new();
        let mut profile = validated_profile_with_paths("http://127.0.0.1:8000");
        profile.config.reset_kv_cache = Some(ResetKvCacheConfig {
            timeout_seconds: Some(5.0),
            path: Some("/reset_prefix_cache".to_owned()),
            max_retry_seconds: None,
        });
        profile.config.server_profiler = Some(ServerProfilerConfig {
            timeout_seconds: Some(6.0),
            start_path: Some("/start_profile".to_owned()),
            stop_path: Some("/stop_profile".to_owned()),
        });
        let hooks =
            prepare_endpoint_control_hooks(clock, &provider, &profile).expect("hooks prepare");

        let reset = hooks.reset_kv_cache.as_ref().expect("reset hook");
        let profiler = hooks.server_profiler.as_ref().expect("profiler hook");
        let reset_outcome = run_reset_kv_cache(reset).await.expect("reset executes");
        let start_outcome = start_server_profiler(profiler)
            .await
            .expect("start executes");
        let stop_outcome = stop_server_profiler(profiler).await.expect("stop executes");
        assert_eq!(reset_outcome.request_count, 1);
        assert_eq!(start_outcome.request_count, 1);
        assert_eq!(stop_outcome.request_count, 1);

        assert_eq!(
            provider.state.requests.borrow().as_slice(),
            &[
                RecordedRequest {
                    request_id: "control-hook-reset_kv_cache-0".to_owned(),
                    method: ControlPlaneMethod::Post,
                    path: "/reset_prefix_cache".to_owned(),
                    absolute_deadline_ns: 5_000_000_000,
                },
                RecordedRequest {
                    request_id: "control-hook-server_profiler.start-0".to_owned(),
                    method: ControlPlaneMethod::Post,
                    path: "/start_profile".to_owned(),
                    absolute_deadline_ns: 6_000_000_000,
                },
                RecordedRequest {
                    request_id: "control-hook-server_profiler.stop-0".to_owned(),
                    method: ControlPlaneMethod::Post,
                    path: "/stop_profile".to_owned(),
                    absolute_deadline_ns: 6_000_000_000,
                },
            ]
        );
    }

    #[tokio::test]
    async fn overlapping_profiler_owners_share_one_control_session() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider = RecordingProvider::new();
        let hooks = prepare_endpoint_control_hooks(
            clock,
            &provider,
            &validated_profile_with_paths("http://127.0.0.1:8000"),
        )
        .expect("hooks prepare");
        let profiler = Rc::new(ServerProfilerCoordinator::new(
            hooks.server_profiler.expect("profiler hook"),
        ));

        profiler
            .acquire()
            .await
            .expect("first phase starts profiler");
        profiler
            .acquire()
            .await
            .expect("overlapping phase shares profiler");
        profiler
            .release()
            .await
            .expect("successor releases ownership");
        profiler
            .release()
            .await
            .expect("predecessor stops profiler after drain");

        assert_eq!(
            provider
                .state
                .requests
                .borrow()
                .iter()
                .map(|request| request.path.as_str())
                .collect::<Vec<_>>(),
            ["/start_profile", "/stop_profile"]
        );
    }

    #[test]
    fn profile_value_helper_prepares_hooks_from_authored_endpoint_shape() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider = RecordingProvider::new();
        let hooks = prepare_endpoint_control_hooks_from_profile_value(
            clock,
            &provider,
            &serde_json::json!({
                "type": "chat",
                "urls": ["http://127.0.0.1:8000/v1/chat/completions"],
                "timeout_seconds": 30.0,
                "ssl_verify": true,
                "http2": false,
                "connection_limit": 4,
                "keepalive_timeout": 15.0,
                "reset_kv_cache": {
                    "path": "/reset_prefix_cache"
                },
                "server_profiler": {
                    "start_path": "/start_profile",
                    "stop_path": "/stop_profile"
                }
            }),
        )
        .expect("hooks prepare from value");

        assert!(hooks.reset_kv_cache.is_some());
        assert!(hooks.server_profiler.is_some());
        assert_eq!(
            provider.state.prepared_urls.borrow().as_slice(),
            &["http://127.0.0.1:8000/"]
        );
    }

    #[test]
    fn unset_control_hook_timeouts_use_the_control_default_not_endpoint_timeout() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider = RecordingProvider::new();
        let mut profile = validated_profile_with_paths("http://127.0.0.1:8000");
        // A six-hour inference timeout must not leak into a control POST.
        profile.client.total_timeout_ns = Some(21_600_000_000_000);
        let hooks =
            prepare_endpoint_control_hooks(clock, &provider, &profile).expect("hooks prepare");

        let reset = hooks.reset_kv_cache.expect("reset hook");
        assert_eq!(reset.timeout_ns, DEFAULT_CONTROL_HOOK_TIMEOUT_NS);
        assert_eq!(reset.max_retry_ns, DEFAULT_RESET_KV_CACHE_MAX_RETRY_NS);
        assert_eq!(
            hooks.server_profiler.expect("profiler hook").timeout_ns,
            DEFAULT_CONTROL_HOOK_TIMEOUT_NS
        );
    }

    #[test]
    fn transient_reset_statuses_retry_within_budget_and_other_statuses_fail_fast() {
        for (statuses, expected_attempts, expects_success) in [
            (vec![503, 503, 204], 3, true),
            (vec![400], 1, false),
        ] {
            let sim_clock = Rc::new(SimClock::new());
            let clock: Rc<dyn Clock> = sim_clock.clone();
            let provider = RecordingProvider::with_statuses(&statuses);
            let hooks = prepare_endpoint_control_hooks(
                clock,
                &provider,
                &validated_profile_with_paths("http://127.0.0.1:8000"),
            )
            .expect("hooks prepare");
            let reset = hooks.reset_kv_cache.expect("reset hook");

            let outcome = Rc::new(RefCell::new(None));
            let outcome_slot = outcome.clone();
            sim_clock.drive(Box::pin(async move {
                *outcome_slot.borrow_mut() = Some(run_reset_kv_cache(&reset).await);
            }));

            let outcome = outcome.borrow_mut().take().expect("reset hook completes");
            assert_eq!(outcome.is_ok(), expects_success, "statuses {statuses:?}");
            assert_eq!(
                provider.state.requests.borrow().len(),
                expected_attempts,
                "statuses {statuses:?}"
            );
        }
    }
}
