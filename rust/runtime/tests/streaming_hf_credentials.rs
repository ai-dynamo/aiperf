// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior suite for the redacted Hugging Face credential authority and its
//! loopback-safe HTTP client.

#[allow(dead_code)]
#[path = "support/streaming_checkpoint.rs"]
mod support;

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aiperf_runtime::clock::{Clock, RealClock, SimClock};
use aiperf_runtime::streaming::blocking::StreamingBlockingExecutor;
use aiperf_runtime::streaming::hf_credentials::{
    HfCredentialAuthority, HfCredentialError, HfCredentialMaterialReader, HfCredentialProvider,
    HfCredentialSettings, HfCredentialSourceDescriptor, HfCredentialSourceKind,
    HfHttpClientFactory, HfHttpSettings, HfProxySelection, HfRefreshOutcome,
    ProcessHfCredentialReader,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;

const SECRET: &str = "hf_supersecrettokenvalue0001";
const ROTATED_SECRET: &str = "hf_supersecrettokenvalue0002";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Rotation seam that never touches process environment.
#[derive(Debug)]
struct FakeCredentialReader {
    material: Mutex<Option<String>>,
    reads: AtomicUsize,
}

impl FakeCredentialReader {
    fn new(initial: Option<&str>) -> Arc<Self> {
        Arc::new(Self {
            material: Mutex::new(initial.map(str::to_owned)),
            reads: AtomicUsize::new(0),
        })
    }

    fn set(&self, next: Option<&str>) {
        if let Ok(mut slot) = self.material.lock() {
            *slot = next.map(str::to_owned);
        }
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::Acquire)
    }
}

impl HfCredentialMaterialReader for FakeCredentialReader {
    fn read(
        &self,
        _descriptor: &HfCredentialSourceDescriptor,
    ) -> Result<Option<String>, HfCredentialError> {
        self.reads.fetch_add(1, Ordering::AcqRel);
        Ok(self
            .material
            .lock()
            .map_err(|_| HfCredentialError::SourceUnavailable)?
            .clone())
    }
}

fn executor() -> StreamingBlockingExecutor {
    StreamingBlockingExecutor::for_test(support::run_id(1), 8, 64, 262_144).expect("executor")
}

fn hub_endpoint() -> Url {
    Url::parse("https://huggingface.co").expect("endpoint")
}

/// Settings that pin an environment-variable descriptor without reading it: the
/// injected reader supplies material, so no process variable is consulted.
fn pinned_settings() -> HfCredentialSettings {
    let mut settings = HfCredentialSettings::new(hub_endpoint());
    settings.authored_env_var = Some("AIPERF_TEST_HF_TOKEN".to_owned());
    settings.allow_anonymous = false;
    settings.refresh_backoff_base_ns = 1_000;
    settings.refresh_backoff_cap_ns = 1_000;
    settings
}

fn real_clock() -> Rc<dyn Clock> {
    RealClock::new()
}

async fn prepared(
    settings: HfCredentialSettings,
    reader: Arc<FakeCredentialReader>,
) -> Result<HfCredentialAuthority, HfCredentialError> {
    HfCredentialAuthority::prepare(settings, reader, real_clock(), executor()).await
}

/// Bind one loopback listener that records the first request and answers 200.
async fn recording_listener(host: &str) -> (u16, tokio::sync::oneshot::Receiver<String>) {
    let listener = tokio::net::TcpListener::bind((host, 0))
        .await
        .expect("bind loopback listener");
    let port = listener.local_addr().expect("listener address").port();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        if let Ok((mut socket, _)) = listener.accept().await {
            let mut buffer = vec![0_u8; 4096];
            let read = socket.read(&mut buffer).await.unwrap_or(0);
            let _ = sender.send(String::from_utf8_lossy(&buffer[..read]).into_owned());
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: text/plain\r\n\r\nok")
                .await;
            let _ = socket.flush().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    (port, receiver)
}

// ---------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn authority_debug_and_errors_never_render_the_token() {
    let reader = FakeCredentialReader::new(Some(SECRET));
    let authority = prepared(pinned_settings(), reader)
        .await
        .expect("prepared authority");
    let lease = authority.lease().await.expect("lease");

    let mut http = HfHttpSettings::new(hub_endpoint());
    http.proxy = Some("http://proxyuser:sekritproxypass@proxy.example:3128".to_owned());
    let factory = HfHttpClientFactory::resolve(&http).expect("factory");
    let client = factory.build(real_clock());

    let renderings = [
        format!("{authority:?}"),
        format!("{lease:?}"),
        format!("{factory:?}"),
        format!("{client:?}"),
        format!("{:?}", authority.descriptor()),
    ];
    for rendering in &renderings {
        assert!(
            !rendering.contains(SECRET),
            "rendering leaked the token: {rendering}"
        );
        assert!(
            !rendering.contains("sekritproxypass"),
            "rendering leaked the proxy password: {rendering}"
        );
        assert!(
            !rendering.contains("Basic "),
            "rendering leaked a proxy authorization header: {rendering}"
        );
    }
    assert!(renderings[0].contains("<redacted>"));

    for error in [
        HfCredentialError::SourceUnavailable,
        HfCredentialError::MalformedMaterial,
        HfCredentialError::InvalidSettings,
        HfCredentialError::RefreshExhausted,
        HfCredentialError::RefreshBudgetExhausted,
        HfCredentialError::BlockingUnavailable,
        HfCredentialError::Cancelled,
    ] {
        assert_eq!(error.to_string(), error.code());
        assert!(!format!("{error:?}").contains(SECRET));
    }
}

// ---------------------------------------------------------------------------
// Refresh authority
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn expired_credential_refresh_rotates_without_changing_source_id() {
    let reader = FakeCredentialReader::new(Some(SECRET));
    let authority = prepared(pinned_settings(), Arc::clone(&reader))
        .await
        .expect("prepared authority");
    let before = authority.lease().await.expect("lease");
    assert_eq!(before.generation().get(), 0);
    assert!(!before.is_anonymous());

    reader.set(Some(ROTATED_SECRET));
    let outcome = authority
        .refresh(before.generation())
        .await
        .expect("refresh outcome");

    assert!(matches!(outcome, HfRefreshOutcome::Rotated(_)));
    let after = outcome.lease();
    assert_eq!(after.source_id().as_bytes(), before.source_id().as_bytes());
    assert_eq!(after.generation().get(), 1);
    assert_eq!(authority.refresh_count(), 1);
    assert_eq!(authority.source_kind(), HfCredentialSourceKind::Environment);
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_invalidation_of_one_generation_refreshes_exactly_once() {
    let reader = FakeCredentialReader::new(Some(SECRET));
    let authority = prepared(pinned_settings(), Arc::clone(&reader))
        .await
        .expect("prepared authority");
    let reads_after_prepare = reader.reads();
    reader.set(Some(ROTATED_SECRET));

    let seen = authority.lease().await.expect("lease").generation();
    let (a, b, c, d) = tokio::join!(
        authority.refresh(seen),
        authority.refresh(seen),
        authority.refresh(seen),
        authority.refresh(seen),
    );

    let outcomes = [
        a.expect("first"),
        b.expect("second"),
        c.expect("third"),
        d.expect("fourth"),
    ];
    let rotated = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, HfRefreshOutcome::Rotated(_)))
        .count();
    let superseded = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, HfRefreshOutcome::Superseded(_)))
        .count();
    assert_eq!(rotated, 1, "exactly one caller performs the refresh");
    assert_eq!(superseded, 3, "the rest observe the newer generation");
    assert_eq!(authority.refresh_count(), 1);
    assert_eq!(
        reader.reads() - reads_after_prepare,
        1,
        "the pinned source is read exactly once per refresh"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn refresh_exhaustion_is_typed_and_never_downgrades_to_anonymous() {
    let mut settings = pinned_settings();
    settings.max_refresh_attempts = 2;
    let reader = FakeCredentialReader::new(Some(SECRET));
    let authority = prepared(settings, Arc::clone(&reader))
        .await
        .expect("prepared authority");
    let seen = authority.lease().await.expect("lease").generation();

    for _ in 0..2 {
        let outcome = authority.refresh(seen).await.expect("unchanged refresh");
        assert!(matches!(outcome, HfRefreshOutcome::Unchanged(_)));
    }
    assert_eq!(
        authority.refresh(seen).await.unwrap_err(),
        HfCredentialError::RefreshExhausted
    );

    let vanishing = FakeCredentialReader::new(Some(SECRET));
    let authority = prepared(pinned_settings(), Arc::clone(&vanishing))
        .await
        .expect("prepared authority");
    let seen = authority.lease().await.expect("lease").generation();
    vanishing.set(None);
    assert_eq!(
        authority.refresh(seen).await.unwrap_err(),
        HfCredentialError::SourceUnavailable
    );
    assert!(
        !authority.lease().await.expect("lease").is_anonymous(),
        "a lost credential never silently downgrades to anonymous"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn refresh_backoff_is_exponential_capped_and_clock_driven() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let sim = Rc::new(SimClock::new());
            let clock: Rc<dyn Clock> = Rc::clone(&sim);
            let pump = tokio::task::spawn_local({
                let sim = Rc::clone(&sim);
                async move {
                    loop {
                        if let Some(next) = sim.next_event_time() {
                            sim.advance_to(next);
                        }
                        tokio::time::sleep(Duration::from_micros(200)).await;
                    }
                }
            });

            let mut settings = pinned_settings();
            settings.max_refresh_attempts = 3;
            settings.refresh_backoff_base_ns = 250_000_000;
            settings.refresh_backoff_cap_ns = 600_000_000;
            let reader = FakeCredentialReader::new(Some(SECRET));
            let authority =
                HfCredentialAuthority::prepare(settings, reader, clock, executor())
                    .await
                    .expect("prepared authority");
            let seen = authority.lease().await.expect("lease").generation();
            assert_eq!(sim.now_ns(), 0, "preparation performs no clock wait");

            for expected in [250_000_000_i64, 750_000_000, 1_350_000_000] {
                authority.refresh(seen).await.expect("unchanged refresh");
                assert_eq!(sim.now_ns(), expected);
            }
            pump.abort();
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn note_authorized_resets_the_episode_but_not_the_run_ceiling() {
    let mut settings = pinned_settings();
    settings.max_refresh_attempts = 2;
    settings.max_total_refreshes = 3;
    let reader = FakeCredentialReader::new(Some(SECRET));
    let authority = prepared(settings, reader)
        .await
        .expect("prepared authority");
    let seen = authority.lease().await.expect("lease").generation();

    for _ in 0..2 {
        authority.refresh(seen).await.expect("unchanged refresh");
    }
    assert_eq!(
        authority.refresh(seen).await.unwrap_err(),
        HfCredentialError::RefreshExhausted
    );

    authority.note_authorized(seen);
    authority
        .refresh(seen)
        .await
        .expect("episode reset restores refresh availability");
    assert_eq!(
        authority.refresh(seen).await.unwrap_err(),
        HfCredentialError::RefreshBudgetExhausted,
        "the run ceiling survives an episode reset"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn token_file_rotation_is_read_through_the_blocking_executor() {
    let path: PathBuf = std::env::temp_dir().join(format!(
        "aiperf-hf-token-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::write(&path, format!("{SECRET}\n")).expect("write token file");

    let mut settings = HfCredentialSettings::new(hub_endpoint());
    settings.authored_token_file = Some(path.clone());
    settings.allow_anonymous = false;
    settings.refresh_backoff_base_ns = 1_000;
    settings.refresh_backoff_cap_ns = 1_000;

    let owner = executor();
    let authority = HfCredentialAuthority::prepare(
        settings,
        Arc::new(ProcessHfCredentialReader),
        real_clock(),
        owner.clone(),
    )
    .await
    .expect("prepared authority");
    assert_eq!(authority.source_kind(), HfCredentialSourceKind::TokenFile);
    assert_eq!(
        owner.snapshot().output_bytes,
        0,
        "the blocking output reservation is released with the read"
    );

    let seen = authority.lease().await.expect("lease").generation();
    std::fs::write(&path, format!("{ROTATED_SECRET}\n")).expect("rewrite token file");
    let outcome = authority.refresh(seen).await.expect("refresh outcome");
    assert!(matches!(outcome, HfRefreshOutcome::Rotated(_)));

    owner.cancel_and_join().await.expect("clean shutdown");
    assert_eq!(
        authority
            .refresh(outcome.lease().generation())
            .await
            .unwrap_err(),
        HfCredentialError::Cancelled,
        "a shut-down blocking owner terminates refresh instead of blocking"
    );
    let _ = std::fs::remove_file(&path);
}

#[tokio::test(flavor = "current_thread")]
async fn malformed_material_is_refused_before_any_request() {
    let oversized = "a".repeat(5000);
    for material in ["", "hf_abc\r\ndef", "hf_tokén", oversized.as_str()] {
        let reader = FakeCredentialReader::new(Some(material));
        assert_eq!(
            prepared(pinned_settings(), reader)
                .await
                .expect_err("malformed material must be refused")
                .code(),
            HfCredentialError::MalformedMaterial.code(),
            "material {material:?} was accepted"
        );
    }
}

// ---------------------------------------------------------------------------
// Source identity
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn restart_with_the_same_descriptor_yields_the_same_source_id_after_rotation() {
    let first = prepared(pinned_settings(), FakeCredentialReader::new(Some(SECRET)))
        .await
        .expect("first authority");
    let second = prepared(
        pinned_settings(),
        FakeCredentialReader::new(Some(ROTATED_SECRET)),
    )
    .await
    .expect("second authority");

    assert_eq!(first.source_id().as_bytes(), second.source_id().as_bytes());
    assert_eq!(
        second.lease().await.expect("lease").generation().get(),
        0,
        "a restart restarts the refresh generation"
    );
}

#[test]
fn descriptor_change_produces_a_different_source_id() {
    let host = "huggingface.co";
    let ids = [
        HfCredentialSourceDescriptor::anonymous(host).source_id(),
        HfCredentialSourceDescriptor::environment("HF_TOKEN", host).source_id(),
        HfCredentialSourceDescriptor::environment("OTHER_TOKEN", host).source_id(),
        HfCredentialSourceDescriptor::token_file("/etc/hf/token", host).source_id(),
        HfCredentialSourceDescriptor::token_file("/etc/hf/other", host).source_id(),
        HfCredentialSourceDescriptor::environment("HF_TOKEN", "hub.internal").source_id(),
    ];
    for (index, left) in ids.iter().enumerate() {
        for right in &ids[index + 1..] {
            assert_ne!(left.as_bytes(), right.as_bytes());
        }
    }
    assert_eq!(
        HfCredentialSourceDescriptor::environment("HF_TOKEN", host)
            .source_id()
            .short_hex()
            .len(),
        16
    );
}

#[tokio::test(flavor = "current_thread")]
async fn absent_authored_credential_fails_preparation_before_any_request() {
    let reader = FakeCredentialReader::new(None);
    assert_eq!(
        prepared(pinned_settings(), Arc::clone(&reader))
            .await
            .unwrap_err(),
        HfCredentialError::SourceUnavailable
    );
    assert_eq!(reader.reads(), 1, "preparation reads once and then refuses");
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_settings_are_refused() {
    let mutations: [fn(&mut HfCredentialSettings); 4] = [
        |settings| settings.max_refresh_attempts = 0,
        |settings| settings.max_total_refreshes = 0,
        |settings| settings.refresh_backoff_cap_ns = settings.refresh_backoff_base_ns - 1,
        |settings| settings.refresh_backoff_cap_ns = 60_000_000_001,
    ];
    for mutate in mutations {
        let mut settings = pinned_settings();
        settings.refresh_backoff_base_ns = 1_000;
        settings.refresh_backoff_cap_ns = 1_000;
        mutate(&mut settings);
        assert_eq!(
            prepared(settings, FakeCredentialReader::new(Some(SECRET)))
                .await
                .unwrap_err(),
            HfCredentialError::InvalidSettings
        );
    }

    let hostless = Url::parse("data:text/plain,hub").expect("hostless url");
    assert_eq!(
        HfHttpClientFactory::resolve(&HfHttpSettings::new(hostless)).unwrap_err(),
        HfCredentialError::InvalidSettings
    );
    let mut unbounded = HfHttpSettings::new(hub_endpoint());
    unbounded.max_response_body_bytes = 0;
    assert_eq!(
        HfHttpClientFactory::resolve(&unbounded).unwrap_err(),
        HfCredentialError::InvalidSettings
    );
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn loopback_hf_endpoint_is_never_proxied_even_with_ambient_proxy_env() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (port, received) = recording_listener("127.0.0.1").await;
            let endpoint = Url::parse(&format!("http://127.0.0.1:{port}/")).expect("endpoint");
            let settings = HfHttpSettings::new(endpoint.clone());
            assert!(
                settings.proxy_from_env,
                "downloads opt into the ambient proxy environment by default"
            );

            let factory = HfHttpClientFactory::resolve(&settings).expect("factory");
            assert_eq!(factory.proxy_selection(), HfProxySelection::Disabled);

            let reader = FakeCredentialReader::new(Some(SECRET));
            let authority = prepared(pinned_settings(), reader)
                .await
                .expect("prepared authority");
            let lease = authority.lease().await.expect("lease");
            let client = factory.build(real_clock());
            let response = client
                .authorized_get(&lease, &endpoint, &[])
                .await
                .expect("authorized get");

            assert_eq!(response.record.status, Some(200));
            let request = received.await.expect("listener observed the request");
            assert!(
                request.starts_with("GET / HTTP/1.1"),
                "an absolute-form request line means the request went through a proxy: {request}"
            );
        })
        .await;
}

#[test]
fn explicit_proxy_wins_and_its_authorization_is_never_rendered() {
    let mut settings = HfHttpSettings::new(hub_endpoint());
    settings.proxy = Some("http://proxyuser:sekritproxypass@proxy.example:3128".to_owned());
    settings.proxy_from_env = false;
    let factory = HfHttpClientFactory::resolve(&settings).expect("factory");

    assert_eq!(factory.proxy_selection(), HfProxySelection::Explicit);
    assert_eq!(factory.endpoint_host(), "huggingface.co");
    let rendering = format!("{factory:?}");
    assert!(!rendering.contains("sekritproxypass"), "{rendering}");
    assert!(!rendering.contains("proxyuser"), "{rendering}");
    assert!(rendering.contains("explicit"), "{rendering}");

    let mut invalid = HfHttpSettings::new(hub_endpoint());
    invalid.proxy = Some("not a url".to_owned());
    assert_eq!(
        HfHttpClientFactory::resolve(&invalid).unwrap_err(),
        HfCredentialError::InvalidSettings
    );
}

#[tokio::test(flavor = "current_thread")]
async fn bearer_is_stamped_only_for_the_pinned_endpoint_host() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (pinned_port, pinned_request) = recording_listener("127.0.0.1").await;
            let (other_port, other_request) = recording_listener("127.0.0.2").await;
            let pinned =
                Url::parse(&format!("http://127.0.0.1:{pinned_port}/api")).expect("pinned url");
            let other =
                Url::parse(&format!("http://127.0.0.2:{other_port}/cdn")).expect("cdn url");

            let factory =
                HfHttpClientFactory::resolve(&HfHttpSettings::new(pinned.clone())).expect("factory");
            let client = factory.build(real_clock());
            let reader = FakeCredentialReader::new(Some(SECRET));
            let authority = prepared(pinned_settings(), reader)
                .await
                .expect("prepared authority");
            let lease = authority.lease().await.expect("lease");

            let stamped = client
                .authorized_get(&lease, &pinned, &[("range", "bytes=0-15")])
                .await
                .expect("pinned request");
            assert!(stamped.bearer_stamped);
            let observed = pinned_request.await.expect("pinned listener");
            assert!(
                observed.contains(&format!("Bearer {SECRET}")),
                "pinned host must carry the bearer: {observed}"
            );
            assert!(observed.to_ascii_lowercase().contains("range: bytes=0-15"));

            let unstamped = client
                .authorized_get(&lease, &other, &[])
                .await
                .expect("cdn request");
            assert!(!unstamped.bearer_stamped);
            let observed = other_request.await.expect("cdn listener");
            assert!(
                !observed.contains(SECRET),
                "a non-pinned host must never see the credential: {observed}"
            );
        })
        .await;
}
