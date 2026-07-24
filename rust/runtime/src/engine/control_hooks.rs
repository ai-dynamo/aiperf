// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared endpoint-local control hooks over profile-bound control-plane HTTP.

use std::fmt::{self, Debug, Formatter};
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use url::Url;

use crate::clock::Clock;
use crate::endpoints::{EndpointId, RawEndpointConfig, ResetKvCacheConfig, ServerProfilerConfig};
use crate::engine::control_plane_http::{
    ControlPlaneCredentialReference, ControlPlaneHttp, ControlPlaneHttpProvider,
    ControlPlaneMethod, ControlPlaneRequest, ControlPlaneTlsReference, LocalCancellationSignal,
    ValidatedControlPlaneProfile,
};
use crate::engine::registry::ValidatedEndpointProfileV2;
use crate::timing::LocalPhaseFuture;
use crate::transport::core::ConnectionReuseStrategy;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::HttpVersion;

const DEFAULT_CONTROL_HOOK_TIMEOUT_NS: i64 = 30_000_000_000;
const DEFAULT_RESET_KV_CACHE_PATH: &str = "/reset_prefix_cache";
const DEFAULT_SERVER_PROFILER_START_PATH: &str = "/start_profile";
const DEFAULT_SERVER_PROFILER_STOP_PATH: &str = "/stop_profile";
const CONTROL_RESPONSE_MAX_BYTES: usize = 64 * 1024;

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
        .map(|config| prepare_reset_hook(config, &profile.client, &handles, &target_urls, &clock))
        .transpose()?;
    let server_profiler = profile
        .config
        .server_profiler
        .as_ref()
        .map(|config| {
            prepare_server_profiler_hook(config, &profile.client, &handles, &target_urls, &clock)
        })
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
pub fn run_reset_kv_cache(
    hook: &PreparedResetKvCacheHook,
) -> LocalPhaseFuture<Result<ControlHookOutcome>> {
    let timeout_ns = hook.timeout_ns;
    let path = hook.path.clone();
    let handles = hook.handles.clone();
    let clock = hook.clock.clone();
    let target_urls = hook.target_urls.clone();
    Box::pin(async move {
        execute_control_hook(
            "reset_kv_cache",
            timeout_ns,
            path,
            handles,
            target_urls,
            clock,
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
        )
        .await
    })
}

fn prepare_handles(
    control_plane: &dyn ControlPlaneHttpProvider,
    endpoint_urls: &[String],
    endpoint_client: &ClientConfig,
) -> Result<(Vec<Rc<dyn ControlPlaneHttp>>, Vec<String>)> {
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

fn validated_profile_from_value(profile_value: &serde_json::Value) -> Result<ValidatedEndpointProfileV2> {
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

fn resolve_timeout_ns(endpoint_client: &ClientConfig, timeout_seconds: Option<f64>) -> Result<i64> {
    let timeout_ns = match timeout_seconds {
        Some(seconds) => seconds_to_ns(seconds)?,
        None => endpoint_client
            .total_timeout_ns
            .unwrap_or(DEFAULT_CONTROL_HOOK_TIMEOUT_NS),
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
    endpoint_client: &ClientConfig,
    handles: &[Rc<dyn ControlPlaneHttp>],
    target_urls: &[String],
    clock: &Rc<dyn Clock>,
) -> Result<PreparedResetKvCacheHook> {
    Ok(PreparedResetKvCacheHook {
        timeout_ns: resolve_timeout_ns(endpoint_client, config.timeout_seconds)?,
        path: normalize_control_path(
            config.path.as_deref(),
            DEFAULT_RESET_KV_CACHE_PATH,
            "endpoint.reset_kv_cache.path",
        )?,
        handles: handles.to_vec(),
        clock: clock.clone(),
        target_urls: target_urls.to_vec(),
    })
}

fn prepare_server_profiler_hook(
    config: &ServerProfilerConfig,
    endpoint_client: &ClientConfig,
    handles: &[Rc<dyn ControlPlaneHttp>],
    target_urls: &[String],
    clock: &Rc<dyn Clock>,
) -> Result<PreparedServerProfilerHook> {
    Ok(PreparedServerProfilerHook {
        timeout_ns: resolve_timeout_ns(endpoint_client, config.timeout_seconds)?,
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

async fn execute_control_hook(
    kind: &'static str,
    timeout_ns: i64,
    path: String,
    handles: Vec<Rc<dyn ControlPlaneHttp>>,
    target_urls: Vec<String>,
    clock: Rc<dyn Clock>,
) -> Result<ControlHookOutcome> {
    let request_count = handles.len();
    for (index, (handle, target_url)) in handles.into_iter().zip(target_urls).enumerate() {
        let absolute_deadline_ns = clock.now_ns().saturating_add(timeout_ns);
        let response = handle
            .execute(
                ControlPlaneRequest {
                    request_id: format!("control-hook-{kind}-{index}"),
                    method: ControlPlaneMethod::Post,
                    path: path.clone(),
                },
                absolute_deadline_ns,
                LocalCancellationSignal::new(),
            )
            .await
            .with_context(|| {
                format!("executing endpoint-local {kind} hook against {target_url:?} at {path:?}")
            })?;
        ensure!(
            (200..300).contains(&response.status),
            "endpoint-local {kind} hook against {target_url:?} at {path:?} returned HTTP {}",
            response.status
        );
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
    use std::collections::BTreeMap;

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
            Ok(ControlPlaneResponse {
                status: 204,
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
}
