// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! HTTP `CONNECT`-tunnel proxy support for the Clock-injected connector.
//!
//! Only dataset/tokenizer downloads opt into this (by setting
//! [`crate::transport::http::config::ClientConfig::proxy`]); the measured
//! benchmark hot path never sets it, so its connect stays byte-identical and
//! proxy-free. Tunnelling happens before any request bytes, so the injected
//! clock is unaffected. Loopback targets are always excluded, preserving the
//! rule that inference traffic must never traverse an ambient proxy.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use url::Url;

use crate::transport::core::{ErrorDetails, ErrorKind};

/// A resolved forward proxy for one target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyConfig {
    /// Proxy host (IP or name); the proxy performs origin DNS on our behalf.
    pub host: String,
    /// Proxy port.
    pub port: u16,
    /// Full `Proxy-Authorization` header value (`Basic <b64>`), when the proxy
    /// URL carried credentials.
    pub auth_header: Option<String>,
}

impl ProxyConfig {
    /// Resolve the proxy that applies to `target`, from the standard proxy
    /// environment (`HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`, case-insensitive),
    /// honoring `NO_PROXY`. Returns `None` when no proxy applies — including for
    /// any loopback host, which is never proxied.
    pub fn from_env_for(target: &Url) -> Option<Self> {
        let host = target.host_str()?;
        if is_loopback_host(host)
            || host_matches_no_proxy(host, &env_any(&["NO_PROXY", "no_proxy"]))
        {
            return None;
        }
        let raw = match target.scheme() {
            "https" => env_any(&["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]),
            _ => env_any(&["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]),
        }?;
        Self::parse(&raw)
    }

    /// Parse a `scheme://[user:pass@]host:port` proxy URL.
    pub fn parse(raw: &str) -> Option<Self> {
        let url = Url::parse(raw).ok()?;
        let host = url.host_str()?.to_string();
        let port = url.port_or_known_default()?;
        let auth_header = (!url.username().is_empty()).then(|| {
            let user = percent_decode(url.username());
            let pass = url.password().map(percent_decode).unwrap_or_default();
            format!("Basic {}", STANDARD.encode(format!("{user}:{pass}")))
        });
        Some(Self {
            host,
            port,
            auth_header,
        })
    }
}

/// Resolve a benchmark-path proxy. An explicit URL wins and is applied as
/// authored (the user asked for it); otherwise `from_env` opts into the ambient
/// proxy environment for `target` (honoring `NO_PROXY` and loopback). Returns
/// `Err` only when an explicit URL fails to parse.
pub fn resolve(
    explicit: Option<&str>,
    from_env: bool,
    target: Option<&Url>,
) -> Result<Option<ProxyConfig>, String> {
    if let Some(raw) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        return match ProxyConfig::parse(raw) {
            Some(proxy) => Ok(Some(proxy)),
            None => Err(format!("invalid proxy URL {raw:?}")),
        };
    }
    if from_env {
        return Ok(target.and_then(ProxyConfig::from_env_for));
    }
    Ok(None)
}

/// Open a TCP tunnel to `origin_host:origin_port` through `proxy` using HTTP
/// `CONNECT`, returning the tunnelled stream ready for TLS. The proxy resolves
/// the origin, so the caller must not resolve it locally.
pub async fn connect_via_proxy(
    proxy: &ProxyConfig,
    origin_host: &str,
    origin_port: u16,
) -> Result<TcpStream, ErrorDetails> {
    let mut stream = TcpStream::connect((proxy.host.as_str(), proxy.port))
        .await
        .map_err(|e| connect_err(format!("proxy connect {}:{}: {e}", proxy.host, proxy.port)))?;

    let request = connect_request(origin_host, origin_port, proxy.auth_header.as_deref());
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| connect_err(format!("proxy CONNECT write: {e}")))?;

    // Read until the end of the response headers (CRLFCRLF). The response can
    // arrive across reads, and the tunnelled body must not be consumed here, so
    // read one byte past the terminator boundary only.
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0u8; 256];
    loop {
        let n = stream
            .read(&mut chunk)
            .await
            .map_err(|e| connect_err(format!("proxy CONNECT read: {e}")))?;
        if n == 0 {
            return Err(connect_err(
                "proxy closed the connection during CONNECT".to_string(),
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_header_end(&buf) {
            // A well-behaved proxy sends nothing after the 200 headers until we
            // start the TLS ClientHello, so there should be no trailing bytes;
            // guard anyway rather than silently drop tunnelled data.
            if pos != buf.len() {
                return Err(connect_err(
                    "proxy sent unexpected data after CONNECT response".to_string(),
                ));
            }
            break;
        }
        if buf.len() > 8192 {
            return Err(connect_err(
                "proxy CONNECT response headers too large".to_string(),
            ));
        }
    }

    let status = connect_status_code(&buf)
        .ok_or_else(|| connect_err("proxy CONNECT response had no status line".to_string()))?;
    if status != 200 {
        return Err(connect_err(format!(
            "proxy CONNECT to {origin_host}:{origin_port} returned HTTP {status}"
        )));
    }
    Ok(stream)
}

/// Build the raw `CONNECT` request line + headers.
fn connect_request(host: &str, port: u16, auth_header: Option<&str>) -> String {
    let mut request = format!("CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n");
    if let Some(auth) = auth_header {
        request.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
    }
    request.push_str("Proxy-Connection: keep-alive\r\n\r\n");
    request
}

/// Byte offset just past the CRLFCRLF that ends the response headers.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

/// Parse the numeric status from an `HTTP/1.x NNN ...` status line.
fn connect_status_code(buf: &[u8]) -> Option<u16> {
    let line_end = buf.windows(2).position(|w| w == b"\r\n")?;
    let line = std::str::from_utf8(&buf[..line_end]).ok()?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host == "127.0.0.1"
        || host == "::1"
        || host.starts_with("127.")
}

/// Minimal `NO_PROXY` matching: `*` disables proxying entirely; otherwise an
/// entry matches an exact host or any subdomain of it (`example.com` and
/// `.example.com` both match `sub.example.com`). Ports in entries are ignored.
fn host_matches_no_proxy(host: &str, no_proxy: &Option<String>) -> bool {
    let Some(list) = no_proxy else { return false };
    for entry in list.split(',') {
        let entry = entry.trim().trim_end_matches('.');
        if entry.is_empty() {
            continue;
        }
        if entry == "*" {
            return true;
        }
        let entry = entry.split(':').next().unwrap_or(entry);
        let bare = entry.trim_start_matches('.');
        if host.eq_ignore_ascii_case(bare)
            || host
                .to_ascii_lowercase()
                .ends_with(&format!(".{}", bare.to_ascii_lowercase()))
        {
            return true;
        }
    }
    false
}

fn env_any(keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
}

fn percent_decode(s: &str) -> String {
    percent_encoding::percent_decode_str(s)
        .decode_utf8_lossy()
        .into_owned()
}

fn connect_err(message: String) -> ErrorDetails {
    ErrorDetails {
        kind: ErrorKind::Connect,
        code: None,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_request_includes_host_and_auth() {
        let req = connect_request("huggingface.co", 443, Some("Basic Zm9vOmJhcg=="));
        assert!(req.starts_with("CONNECT huggingface.co:443 HTTP/1.1\r\n"));
        assert!(req.contains("Host: huggingface.co:443\r\n"));
        assert!(req.contains("Proxy-Authorization: Basic Zm9vOmJhcg==\r\n"));
        assert!(req.ends_with("\r\n\r\n"));
    }

    #[test]
    fn connect_request_omits_auth_when_absent() {
        let req = connect_request("example.com", 443, None);
        assert!(!req.contains("Proxy-Authorization"));
    }

    #[test]
    fn status_and_header_end_parsing() {
        let ok = b"HTTP/1.1 200 Connection established\r\nX: y\r\n\r\n";
        assert_eq!(find_header_end(ok), Some(ok.len()));
        assert_eq!(connect_status_code(ok), Some(200));
        assert_eq!(
            connect_status_code(b"HTTP/1.1 407 Proxy Auth\r\n\r\n"),
            Some(407)
        );
    }

    #[test]
    fn no_proxy_matches_host_and_subdomains_and_star() {
        let list = Some("example.com, .internal, localhost".to_string());
        assert!(host_matches_no_proxy("example.com", &list));
        assert!(host_matches_no_proxy("api.example.com", &list));
        assert!(host_matches_no_proxy("svc.internal", &list));
        assert!(!host_matches_no_proxy("example.org", &list));
        assert!(host_matches_no_proxy("anything", &Some("*".to_string())));
        assert!(!host_matches_no_proxy("example.com", &None));
    }

    #[test]
    fn loopback_is_never_proxied() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.5.5.5"));
        assert!(!is_loopback_host("huggingface.co"));
    }

    #[test]
    fn parses_proxy_url_with_credentials() {
        let p = ProxyConfig::parse("http://user:pass@10.0.0.1:3128").unwrap();
        assert_eq!(p.host, "10.0.0.1");
        assert_eq!(p.port, 3128);
        assert_eq!(p.auth_header.as_deref(), Some("Basic dXNlcjpwYXNz"));
    }
}
