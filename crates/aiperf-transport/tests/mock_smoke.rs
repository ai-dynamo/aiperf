// crates/aiperf-transport/tests/mock_smoke.rs
mod common;
use common::MockServer;

#[test]
fn mock_server_spawns_or_skips() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        match MockServer::spawn(&[]).await {
            Some(s) => assert!(s.base_url.starts_with("http://127.0.0.1:")),
            None => eprintln!("mock unavailable — smoke skipped"),
        }
    });
}
