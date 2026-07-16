// rust/transport-http/tests/mock_smoke.rs
mod common;
use common::{MockServer, run_local};

#[test]
fn mock_server_spawns_or_skips() {
    run_local(async {
        match MockServer::spawn(&[]).await {
            Some(s) => assert!(s.base_url.starts_with("http://127.0.0.1:")),
            None => eprintln!("mock unavailable — smoke skipped"),
        }
    });
}
