// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed transport errors.

/// The category of a transport failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Non-2xx HTTP status.
    Http,
    /// An `event: error` SSE message from the server.
    Sse,
    /// Request cancelled after being sent (`cancel_after_ns`). HTTP 499.
    Cancelled,
    /// Connection establishment (DNS/TCP/TLS/handshake) failed.
    Connect,
    /// A peer response violated the expected wire protocol.
    Protocol,
    /// A read/response timeout elapsed.
    Timeout,
    /// Another transport failure.
    Other,
}

/// A structured error detail attached to a [`crate::transport::core::RequestRecord`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorDetails {
    /// The failure category.
    pub kind: ErrorKind,
    /// The associated status/pseudo-status code, if any (e.g. 503, 499).
    pub code: Option<u16>,
    /// A human-readable message.
    pub message: String,
}

impl ErrorDetails {
    /// A non-2xx HTTP error.
    pub fn http(code: u16, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Http,
            code: Some(code),
            message: message.into(),
        }
    }
    /// An SSE `event: error` failure with pseudo-status 502.
    pub fn sse(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Sse,
            code: Some(502),
            message: message.into(),
        }
    }
    /// A post-send cancellation (HTTP 499 Client Closed Request).
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Cancelled,
            code: Some(499),
            message: message.into(),
        }
    }
    /// A generic transport error.
    pub fn other(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Other,
            code: None,
            message: message.into(),
        }
    }
}

impl From<std::io::Error> for ErrorDetails {
    fn from(e: std::io::Error) -> Self {
        Self {
            kind: ErrorKind::Connect,
            code: None,
            message: format!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_error_carries_code_and_kind() {
        let e = ErrorDetails::http(503, "unavailable");
        assert_eq!(e.kind, ErrorKind::Http);
        assert_eq!(e.code, Some(503));
        assert_eq!(e.message, "unavailable");
    }

    #[test]
    fn cancelled_uses_499() {
        let e = ErrorDetails::cancelled("cancelled after send");
        assert_eq!(e.kind, ErrorKind::Cancelled);
        assert_eq!(e.code, Some(499));
    }

    #[test]
    fn io_error_maps_to_connect() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "nope");
        let e = ErrorDetails::from(io);
        assert_eq!(e.kind, ErrorKind::Connect);
        assert!(e.message.contains("nope"));
    }
}
