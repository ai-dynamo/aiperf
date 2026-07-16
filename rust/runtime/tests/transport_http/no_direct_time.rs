// rust/transport-http/tests/no_direct_time.rs
//! Enforces the Global Constraint: no direct time access in `src/`.

use std::fs;
use std::path::Path;

const FORBIDDEN: &[&str] = &[
    "Instant::now",
    "SystemTime::now",
    "tokio::time::sleep",
    "tokio::time::timeout",
    "tokio::time::interval",
    "tokio::time::Instant",
];

fn scan(dir: &Path, hits: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            scan(&path, hits);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = fs::read_to_string(&path).unwrap();
            for line in src.lines() {
                // Skip comments so docs may mention the forbidden APIs.
                let code = line.split("//").next().unwrap_or("");
                for pat in FORBIDDEN {
                    if code.contains(pat) {
                        hits.push(format!("{}: {}", path.display(), line.trim()));
                    }
                }
            }
        }
    }
}

#[test]
fn no_direct_time_access_in_src() {
    // After the monocrate collapse this test lives in `rust/aiperf`, whose
    // `src/` legitimately contains non-transport code that uses wall-clock APIs
    // (RealClock fallback, dynosim). The Global Constraint it enforces applies
    // to the HTTP transport source, so scope the scan to that module only.
    let mut hits = Vec::new();
    scan(Path::new("src/transport_http"), &mut hits);
    assert!(
        hits.is_empty(),
        "direct time access found (use Clock):\n{}",
        hits.join("\n")
    );
}
