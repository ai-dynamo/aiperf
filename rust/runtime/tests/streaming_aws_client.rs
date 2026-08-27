// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Socket-free construction gate for the shared streaming AWS S3 client.
//!
//! Every case here exercises construction, `Debug` rendering, proxy resolution,
//! and refresh identity. None opens a socket or reaches AWS.

#![cfg(feature = "streaming-s3")]

use std::rc::Rc;
use std::sync::Arc;

use aiperf_runtime::clock::{Clock, SimClock};
use aiperf_runtime::streaming::aws::{
    AwsClientSettings, AwsCredentialProviderAuthority, AwsCredentialSourceKind, AwsProxySelection,
    AwsS3ClientFactory, AwsSecret,
};

const SECRET: &str = "authored-secret-value";
const ACCESS_KEY_ID: &str = "AKIAAIPERFTESTONLY";

fn settings(endpoint: &str, proxy: AwsProxySelection) -> AwsClientSettings {
    AwsClientSettings {
        region: Some("us-east-1".to_owned()),
        endpoint_url: Some(endpoint.to_owned()),
        force_path_style: true,
        proxy,
        operation_timeout_ns: 30_000_000_000,
        connect_timeout_ns: 5_000_000_000,
    }
}

fn authority() -> Arc<AwsCredentialProviderAuthority> {
    Arc::new(AwsCredentialProviderAuthority::from_authored(
        ACCESS_KEY_ID,
        &AwsSecret::new(SECRET),
        None,
        Some("us-east-1"),
    ))
}

#[tokio::test(flavor = "current_thread")]
async fn client_factory_honors_endpoint_and_redacts_credentials() {
    let authority = authority();
    let factory = AwsS3ClientFactory::prepare(
        settings("http://127.0.0.1:9000", AwsProxySelection::Disabled),
        authority.clone(),
    )
    .await
    .expect("authored MinIO settings prepare");

    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let (client, _projection) = factory.build_client(clock);
    assert_eq!(
        client.config().endpoint_url(),
        Some("http://127.0.0.1:9000")
    );
    assert_eq!(authority.kind(), AwsCredentialSourceKind::AuthoredStatic);

    for rendered in [format!("{factory:?}"), format!("{authority:?}")] {
        assert!(!rendered.contains(SECRET), "credential leaked: {rendered}");
        assert!(
            !rendered.contains(ACCESS_KEY_ID),
            "access key leaked: {rendered}"
        );
    }
    assert_eq!(format!("{:?}", AwsSecret::new(SECRET)), "<redacted>");
    assert_eq!(format!("{}", AwsSecret::new(SECRET)), "<redacted>");
}

#[tokio::test(flavor = "current_thread")]
async fn credential_refresh_rebuilds_client_without_changing_source_authority() {
    let authority = authority();
    let before = authority.source_id();
    let factory = AwsS3ClientFactory::prepare(
        settings("http://127.0.0.1:9000", AwsProxySelection::Disabled),
        authority.clone(),
    )
    .await
    .expect("settings prepare");

    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let (_first, _first_projection) = factory.build_client(clock.clone());
    authority.invalidate();
    let (_second, _second_projection) = factory.build_client(clock);

    assert_eq!(authority.source_id(), before);
    assert!(Arc::ptr_eq(factory.authority(), &authority));
}

#[tokio::test(flavor = "current_thread")]
async fn ambient_proxy_never_applies_to_a_loopback_endpoint() {
    // `--proxy-from-env` against http://127.0.0.1:9000 must resolve to no proxy
    // even with an ambient HTTP_PROXY set; the SDK's default HTTPS client would
    // have proxied it, which is why this module installs its own connector.
    let probe = url::Url::parse("http://127.0.0.1:9000").expect("probe parses");
    assert!(
        aiperf_runtime::transport::http::client::proxy::ProxyConfig::from_env_for(&probe).is_none()
    );

    // The whole loopback path still prepares, with the ambient environment opted
    // into, and never fails on a proxy it must not use.
    let factory = AwsS3ClientFactory::prepare(
        settings("http://127.0.0.1:9000", AwsProxySelection::FromEnvironment),
        authority(),
    )
    .await
    .expect("loopback endpoint prepares under --proxy-from-env");
    assert!(!format!("{factory:?}").contains(SECRET));
}
