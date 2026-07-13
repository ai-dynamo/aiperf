// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fine-grained connection/request trace timing. Behavioral port of
//! `AioHttpTraceData` — all timestamps are `Clock::now_ns()` clock-nanoseconds.
//!
//! Per-chunk vectors are opt-in at collection time.

/// Per-request trace timing. All `_ns` fields are clock-nanoseconds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceData {
    // Connection pool
    pub connection_pool_wait_start_ns: Option<i64>,
    pub connection_pool_wait_end_ns: Option<i64>,
    // TCP connect (pure socket connect)
    pub tcp_connect_start_ns: Option<i64>,
    pub tcp_connect_end_ns: Option<i64>,
    // TLS handshake (None for cleartext)
    pub tls_connect_start_ns: Option<i64>,
    pub tls_connect_end_ns: Option<i64>,
    pub connection_reused_ns: Option<i64>,
    // DNS
    pub dns_cache_hit_ns: Option<i64>,
    pub dns_cache_miss_ns: Option<i64>,
    pub dns_lookup_start_ns: Option<i64>,
    pub dns_lookup_end_ns: Option<i64>,
    // Request send
    pub request_send_start_ns: Option<i64>,
    pub request_headers_sent_ns: Option<i64>,
    pub request_send_end_ns: Option<i64>,
    pub request_chunks_count: u32,
    pub request_bytes_total: u64,
    pub request_chunks: Vec<(i64, u64)>,
    // Response receive
    pub response_status_code: Option<u16>,
    pub response_reason: Option<String>,
    pub response_receive_start_ns: Option<i64>,
    pub response_headers_received_ns: Option<i64>,
    pub response_chunks_count: u32,
    pub response_bytes_total: u64,
    pub response_chunks: Vec<(i64, u64)>,
    pub response_receive_end_ns: Option<i64>,
    // Error
    pub error_timestamp_ns: Option<i64>,
    // Socket info
    pub local_ip: Option<String>,
    pub local_port: Option<u16>,
    pub remote_ip: Option<String>,
    pub remote_port: Option<u16>,
}

fn diff(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(y - x),
        _ => None,
    }
}

impl TraceData {
    /// Request send time (k6 http_req_sending).
    pub fn sending(&self) -> Option<i64> {
        diff(self.request_send_start_ns, self.request_send_end_ns)
    }
    /// TTFB / server processing (k6 http_req_waiting): send-complete to first
    /// response body byte (first SSE token for a streaming response).
    pub fn waiting(&self) -> Option<i64> {
        diff(self.request_send_end_ns, self.response_receive_start_ns)
    }
    /// Time to first response header: send-complete to response headers
    /// received. For a streaming LLM this is the server admit + prefill up to
    /// the response head, before the first token arrives (see [`waiting`]).
    ///
    /// [`waiting`]: Self::waiting
    pub fn time_to_first_header(&self) -> Option<i64> {
        diff(self.request_send_end_ns, self.response_headers_received_ns)
    }
    /// Response transfer time (k6 http_req_receiving).
    pub fn receiving(&self) -> Option<i64> {
        match self.response_chunks_count {
            0 => None,
            1 => Some(0),
            _ => diff(self.response_receive_start_ns, self.response_receive_end_ns),
        }
    }
    /// Total request duration (k6 http_req_duration).
    pub fn duration(&self) -> Option<i64> {
        diff(self.request_send_start_ns, self.response_receive_end_ns)
    }
    /// Connection pool wait (k6 http_req_blocked).
    pub fn blocked(&self) -> Option<i64> {
        diff(
            self.connection_pool_wait_start_ns,
            self.connection_pool_wait_end_ns,
        )
    }
    /// DNS lookup time (k6 http_req_looking_up).
    pub fn dns_lookup(&self) -> Option<i64> {
        diff(self.dns_lookup_start_ns, self.dns_lookup_end_ns)
    }
    /// Pure TCP connect time.
    pub fn tcp_connect(&self) -> Option<i64> {
        diff(self.tcp_connect_start_ns, self.tcp_connect_end_ns)
    }
    /// TLS handshake time (`None` for cleartext).
    pub fn tls_handshake(&self) -> Option<i64> {
        diff(self.tls_connect_start_ns, self.tls_connect_end_ns)
    }
    /// Total connect time TCP+TLS (k6 http_req_connecting): from the TCP connect
    /// start to the end of TLS (or of TCP when cleartext).
    pub fn connecting(&self) -> Option<i64> {
        let end = self.tls_connect_end_ns.or(self.tcp_connect_end_ns);
        diff(self.tcp_connect_start_ns, end)
    }

    /// Convert to a wall-clock export using an explicit `(clock_ns, wall_ns)`
    /// reference pair. The crate never reads a wall clock itself.
    pub fn to_export(&self, reference: TraceReference) -> TraceExport {
        let conv = |v: Option<i64>| v.map(|p| reference.wall_ns + (p - reference.clock_ns));
        TraceExport {
            request_send_start_ns: conv(self.request_send_start_ns),
            request_chunks: self
                .request_chunks
                .iter()
                .map(|(timestamp_ns, size)| {
                    (
                        reference.wall_ns + (*timestamp_ns - reference.clock_ns),
                        *size,
                    )
                })
                .collect(),
            response_receive_start_ns: conv(self.response_receive_start_ns),
            response_receive_end_ns: conv(self.response_receive_end_ns),
            response_chunks: self
                .response_chunks
                .iter()
                .map(|(timestamp_ns, size)| {
                    (
                        reference.wall_ns + (*timestamp_ns - reference.clock_ns),
                        *size,
                    )
                })
                .collect(),
            dns_cache_hit_ns: conv(self.dns_cache_hit_ns),
            dns_cache_miss_ns: conv(self.dns_cache_miss_ns),
            sending_ns: self.sending(),
            waiting_ns: self.waiting(),
            receiving_ns: self.receiving(),
            duration_ns: self.duration(),
            blocked_ns: self.blocked(),
            dns_lookup_ns: self.dns_lookup(),
            connecting_ns: self.connecting(),
            response_status_code: self.response_status_code,
            local_port: self.local_port,
            remote_port: self.remote_port,
        }
    }
}

/// A `(clock_ns, wall_ns)` pairing captured by the caller for wall-clock export.
#[derive(Debug, Clone, Copy)]
pub struct TraceReference {
    pub clock_ns: i64,
    pub wall_ns: i64,
}

/// Wall-clock trace export (k6/HAR-compatible durations pre-computed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceExport {
    pub request_send_start_ns: Option<i64>,
    pub request_chunks: Vec<(i64, u64)>,
    pub response_receive_start_ns: Option<i64>,
    pub response_receive_end_ns: Option<i64>,
    pub response_chunks: Vec<(i64, u64)>,
    pub dns_cache_hit_ns: Option<i64>,
    pub dns_cache_miss_ns: Option<i64>,
    pub sending_ns: Option<i64>,
    pub waiting_ns: Option<i64>,
    pub receiving_ns: Option<i64>,
    pub duration_ns: Option<i64>,
    pub blocked_ns: Option<i64>,
    pub dns_lookup_ns: Option<i64>,
    pub connecting_ns: Option<i64>,
    pub response_status_code: Option<u16>,
    pub local_port: Option<u16>,
    pub remote_port: Option<u16>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td() -> TraceData {
        TraceData {
            request_send_start_ns: Some(1_000),
            request_send_end_ns: Some(1_200),
            response_receive_start_ns: Some(1_500),
            response_receive_end_ns: Some(2_000),
            response_chunks_count: 3,
            connection_pool_wait_start_ns: Some(100),
            connection_pool_wait_end_ns: Some(150),
            dns_lookup_start_ns: Some(200),
            dns_lookup_end_ns: Some(260),
            dns_cache_miss_ns: Some(190),
            tcp_connect_start_ns: Some(300),
            tcp_connect_end_ns: Some(500),
            request_chunks: vec![(1_200, 12)],
            response_chunks: vec![(1_500, 4), (2_000, 8)],
            ..TraceData::default()
        }
    }

    #[test]
    fn durations_match_k6_har_math() {
        let t = td();
        assert_eq!(t.sending(), Some(200)); // send_end - send_start
        assert_eq!(t.waiting(), Some(300)); // recv_start - send_end
        assert_eq!(t.receiving(), Some(500)); // recv_end - recv_start (count>1)
        assert_eq!(t.duration(), Some(1_000)); // recv_end - send_start
        assert_eq!(t.blocked(), Some(50));
        assert_eq!(t.dns_lookup(), Some(60));
        assert_eq!(t.connecting(), Some(200));
    }

    #[test]
    fn receiving_is_zero_for_single_chunk() {
        let mut t = td();
        t.response_chunks_count = 1;
        assert_eq!(t.receiving(), Some(0));
    }

    #[test]
    fn receiving_is_none_for_zero_chunks() {
        let mut t = td();
        t.response_chunks_count = 0;
        assert_eq!(t.receiving(), None);
    }

    #[test]
    fn export_converts_perf_to_wall() {
        let t = td();
        // reference: clock 1_000 == wall 10_000
        let exp = t.to_export(TraceReference {
            clock_ns: 1_000,
            wall_ns: 10_000,
        });
        assert_eq!(exp.request_send_start_ns, Some(10_000)); // 10_000 + (1_000-1_000)
        assert_eq!(exp.response_receive_end_ns, Some(11_000)); // 10_000 + (2_000-1_000)
        assert_eq!(exp.duration_ns, Some(1_000));
        assert_eq!(exp.dns_cache_miss_ns, Some(9_190));
        assert_eq!(exp.request_chunks, vec![(10_200, 12)]);
        assert_eq!(exp.response_chunks, vec![(10_500, 4), (11_000, 8)]);
    }
}
