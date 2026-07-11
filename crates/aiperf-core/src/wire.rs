// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw HTTP request/response wire tracing.
//!
//! A [`WireTraceSink`] receives, per request, the exact JSON body sent and the
//! raw response body received (the concatenated SSE stream). [`JsonlWireSink`]
//! writes one `{uuid, status, request, response}` object per line — useful for
//! eyeballing exactly what went on the wire.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::Serialize;

/// One request/response pair as it went over the wire.
#[derive(Debug, Clone, Serialize)]
pub struct WireEntry {
    /// Request correlation id.
    pub uuid: String,
    /// HTTP status code of the response.
    pub status: u16,
    /// The exact JSON body sent to the server.
    pub request: serde_json::Value,
    /// The raw response body received (concatenated SSE text).
    pub response: String,
}

/// Receives raw request/response pairs as requests complete.
pub trait WireTraceSink: Send + Sync {
    /// Record one wire entry. Called concurrently from request tasks.
    fn record(&self, entry: WireEntry);
}

/// Writes wire entries as JSONL (one `WireEntry` per line).
pub struct JsonlWireSink {
    writer: Mutex<BufWriter<File>>,
}

impl JsonlWireSink {
    /// Create a JSONL wire sink writing to `path`.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("creating wire output {}", path.display()))?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }
}

impl WireTraceSink for JsonlWireSink {
    fn record(&self, entry: WireEntry) {
        let Ok(line) = serde_json::to_string(&entry) else {
            return;
        };
        // No per-record flush: it would defeat the `BufWriter` on the hot path.
        // Durability at shutdown is handled by the `Drop` impl below.
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.write_all(line.as_bytes());
            let _ = writer.write_all(b"\n");
        }
    }
}

impl Drop for JsonlWireSink {
    /// Flush any buffered wire entries so records survive shutdown.
    fn drop(&mut self) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_wire_sink_writes_entry() {
        let path = std::env::temp_dir().join(format!("aiperf_wire_{}.jsonl", std::process::id()));
        {
            let sink = JsonlWireSink::new(&path).unwrap();
            sink.record(WireEntry {
                uuid: "u".into(),
                status: 200,
                request: serde_json::json!({"model": "m"}),
                response: "data: [DONE]\n".into(),
            });
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(parsed["status"], 200);
        assert_eq!(parsed["request"]["model"], "m");
        assert!(parsed["response"].as_str().unwrap().contains("[DONE]"));
        let _ = std::fs::remove_file(&path);
    }
}
