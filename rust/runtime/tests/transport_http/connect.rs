// rust/transport-http/tests/connect.rs
mod common;
use common::{MockServer, run_local};

use std::rc::Rc;

use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::client::connection::establish;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::TraceData;

#[test]
fn establishes_h1_connection_to_mock_and_records_socket_info() {
    run_local(async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let url = url::Url::parse(&mock.base_url).unwrap();
        let mut trace = TraceData::default();
        let cfg = ClientConfig::default();
        let (sender, sock) = establish(&url, &cfg, clock, &mut trace)
            .await
            .expect("connect");
        let _ = sender.is_ready();
        assert_eq!(sock.remote.ip().to_string(), "127.0.0.1");
        assert!(trace.tcp_connect_start_ns.is_some());
        assert!(trace.tcp_connect_end_ns >= trace.tcp_connect_start_ns);
        assert!(trace.local_port.is_some());
        assert!(trace.dns_lookup_start_ns.is_some());
        assert!(trace.dns_lookup_end_ns >= trace.dns_lookup_start_ns);
    });
}
