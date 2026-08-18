// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict WebSocket URL, TCP/proxy, rustls, and upgrade establishment.

use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::{Error as TungsteniteError, handshake::client::Request};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use url::Url;

use crate::clock::Clock;
use crate::transport::http::config::{ClientConfig, apply_socket_opts};
use crate::transport::ws::driver::{FallbackReason, classify_upgrade_failure};

/// Concrete socket returned by the native connector.
pub(crate) type WebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Typed connection failure retaining whether a fallback is authorized.
#[derive(Debug)]
pub(crate) struct ConnectFailure {
    message: String,
    fallback_reason: Option<FallbackReason>,
}

impl ConnectFailure {
    pub(crate) fn fallback_reason(&self) -> Option<FallbackReason> {
        self.fallback_reason
    }

    fn closed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fallback_reason: None,
        }
    }

    fn network(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fallback_reason: Some(FallbackReason::NetworkConnect),
        }
    }
}

impl Display for ConnectFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ConnectFailure {}

/// Connect with Clock-driven retry and one absolute operation deadline.
pub(crate) async fn connect(
    url: &Url,
    headers: &std::collections::BTreeMap<String, String>,
    client: &ClientConfig,
    websocket: WebSocketConfig,
    clock: Rc<dyn Clock>,
    deadline_ns: Option<i64>,
) -> Result<WebSocket, ConnectFailure> {
    let request = request(url, headers)?;
    for attempt in 0..=client.max_connect_retries {
        let result = connect_attempt(
            url,
            request.clone(),
            client,
            websocket.clone(),
            clock.clone(),
            deadline_ns,
        )
        .await;
        match result {
            Ok(socket) => return Ok(socket),
            Err(error)
                if attempt < client.max_connect_retries
                    && error.fallback_reason() == Some(FallbackReason::NetworkConnect) =>
            {
                let delay_ns = client
                    .connect_retry_backoff_ns
                    .saturating_mul(i64::from(attempt + 1));
                sleep_before_deadline(clock.clone(), delay_ns, deadline_ns).await?;
            }
            Err(error) => return Err(error),
        }
    }
    Err(ConnectFailure::closed(
        "websocket retry loop ended without a connection result",
    ))
}

fn request(
    url: &Url,
    headers: &std::collections::BTreeMap<String, String>,
) -> Result<Request, ConnectFailure> {
    if !matches!(url.scheme(), "ws" | "wss") {
        return Err(ConnectFailure::closed(format!(
            "websocket URL requires ws:// or wss://, got {:?}",
            url.scheme()
        )));
    }
    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|error| ConnectFailure::closed(format!("building websocket request: {error}")))?;
    for (name, value) in headers {
        let name = http::header::HeaderName::try_from(name.as_str()).map_err(|error| {
            ConnectFailure::closed(format!("invalid websocket header name {name:?}: {error}"))
        })?;
        let value = http::header::HeaderValue::try_from(value.as_str()).map_err(|error| {
            ConnectFailure::closed(format!(
                "invalid websocket header value for {name}: {error}"
            ))
        })?;
        request.headers_mut().insert(name, value);
    }
    Ok(request)
}

async fn connect_attempt(
    url: &Url,
    request: Request,
    client: &ClientConfig,
    websocket: WebSocketConfig,
    clock: Rc<dyn Clock>,
    operation_deadline_ns: Option<i64>,
) -> Result<WebSocket, ConnectFailure> {
    let now_ns = clock.now_ns();
    let connect_deadline_ns = match (operation_deadline_ns, client.connect_timeout_ns) {
        (Some(operation), Some(timeout)) => Some(operation.min(now_ns.saturating_add(timeout))),
        (Some(operation), None) => Some(operation),
        (None, Some(timeout)) => Some(now_ns.saturating_add(timeout)),
        (None, None) => None,
    };
    let future = connect_attempt_inner(url, request, client, websocket);
    match connect_deadline_ns {
        Some(deadline_ns) if now_ns >= deadline_ns => Err(ConnectFailure::network(
            "websocket connection reached its Clock deadline",
        )),
        Some(deadline_ns) => tokio::select! {
            result = future => result,
            () = clock.sleep(deadline_ns.saturating_sub(now_ns)) => {
                Err(ConnectFailure::network("websocket connection reached its Clock deadline"))
            }
        },
        None => future.await,
    }
}

async fn connect_attempt_inner(
    url: &Url,
    request: Request,
    client: &ClientConfig,
    websocket: WebSocketConfig,
) -> Result<WebSocket, ConnectFailure> {
    let host = url
        .host_str()
        .ok_or_else(|| ConnectFailure::closed("websocket URL has no host"))?;
    let port = url.port_or_known_default().ok_or_else(|| {
        ConnectFailure::closed("websocket URL has no explicit or scheme-default port")
    })?;
    let tcp = if let Some(proxy) = &client.proxy {
        crate::transport::http::client::proxy::connect_via_proxy(proxy, host, port)
            .await
            .map_err(|error| {
                ConnectFailure::network(format!("websocket proxy tunnel failed: {}", error.message))
            })?
    } else {
        TcpStream::connect((host, port))
            .await
            .map_err(|error| ConnectFailure::network(format!("websocket TCP connect: {error}")))?
    };
    let _ = apply_socket_opts(&socket2::SockRef::from(&tcp));
    let stream = if url.scheme() == "wss" {
        let server_name =
            rustls::pki_types::ServerName::try_from(host.to_owned()).map_err(|error| {
                ConnectFailure::closed(format!("invalid websocket TLS server name: {error}"))
            })?;
        let connector = TlsConnector::from(
            crate::transport::http::client::connection::websocket_rustls_config(client),
        );
        let tls = connector.connect(server_name, tcp).await.map_err(|error| {
            ConnectFailure::closed(format!("websocket TLS handshake failed: {error}"))
        })?;
        MaybeTlsStream::Rustls(tls)
    } else {
        MaybeTlsStream::Plain(tcp)
    };
    tokio_tungstenite::client_async_with_config(request, stream, Some(websocket))
        .await
        .map(|(socket, _)| socket)
        .map_err(|error| ConnectFailure {
            fallback_reason: classify_upgrade_failure(&error),
            message: upgrade_message(&error),
        })
}

fn upgrade_message(error: &TungsteniteError) -> String {
    match error {
        TungsteniteError::Http(response) => {
            format!("websocket upgrade returned HTTP {}", response.status())
        }
        _ => format!("websocket upgrade failed: {error}"),
    }
}

async fn sleep_before_deadline(
    clock: Rc<dyn Clock>,
    delay_ns: i64,
    deadline_ns: Option<i64>,
) -> Result<(), ConnectFailure> {
    if delay_ns <= 0 {
        return Ok(());
    }
    let now_ns = clock.now_ns();
    if deadline_ns.is_some_and(|deadline_ns| now_ns.saturating_add(delay_ns) >= deadline_ns) {
        return Err(ConnectFailure::network(
            "websocket retry backoff reached the operation deadline",
        ));
    }
    clock.sleep(delay_ns).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use url::Url;

    use super::request;

    #[test]
    fn upgrade_request_preserves_authored_headers() {
        let mut headers = BTreeMap::new();
        headers.insert("authorization".to_owned(), "Bearer test".to_owned());
        let url =
            Url::parse("wss://example.test/v1/realtime?model=test").expect("test URL is valid");

        let request = request(&url, &headers).expect("request is valid");

        assert_eq!(
            request.uri().path_and_query().map(|value| value.as_str()),
            Some("/v1/realtime?model=test")
        );
        assert_eq!(request.headers()["authorization"], "Bearer test");
    }

    #[test]
    fn upgrade_request_rejects_non_websocket_urls_and_invalid_headers() {
        let headers = BTreeMap::new();
        let https = Url::parse("https://example.test/v1/realtime").expect("test URL is valid");
        assert!(request(&https, &headers).is_err());

        let mut invalid_headers = BTreeMap::new();
        invalid_headers.insert("bad header".to_owned(), "value".to_owned());
        let websocket = Url::parse("wss://example.test/v1/realtime").expect("test URL is valid");
        assert!(request(&websocket, &invalid_headers).is_err());
    }
}
