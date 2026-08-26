// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text and JSON rendering for native Kubernetes command output.

use serde_json::Value;

use super::error::KubeError;

/// Output shape selected on the native Kubernetes command line.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutputFormat {
    /// Human-readable single-line-per-resource summary.
    #[default]
    Text,
    /// Pretty-printed JSON re-serialization of the decoded API response.
    Json,
}

impl OutputFormat {
    /// Select the format from repeatable `--output`/`-o` arguments.
    pub fn from_args(args: &[String]) -> Result<Self, KubeError> {
        let mut arguments = args.iter();
        while let Some(argument) = arguments.next() {
            let value = if let Some(value) = argument.strip_prefix("--output=") {
                Some(value.to_string())
            } else if argument == "--output" || argument == "-o" {
                Some(
                    arguments
                        .next()
                        .ok_or_else(|| KubeError::Decode("--output requires a value".to_string()))?
                        .clone(),
                )
            } else {
                None
            };
            if let Some(value) = value {
                return match value.as_str() {
                    "text" => Ok(Self::Text),
                    "json" => Ok(Self::Json),
                    other => Err(KubeError::Decode(format!(
                        "unsupported native Kubernetes output format {other}"
                    ))),
                };
            }
        }
        Ok(Self::Text)
    }
}

/// Render one bounded API response body in the selected format.
pub fn render(format: OutputFormat, body: &[u8]) -> Result<String, KubeError> {
    let document: Value = serde_json::from_slice(body).map_err(|error| {
        KubeError::Decode(format!("Kubernetes API response is not JSON: {error}"))
    })?;
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(&document)
            .map_err(|error| KubeError::Decode(error.to_string())),
        OutputFormat::Text => Ok(render_text(&document)),
    }
}

fn render_text(document: &Value) -> String {
    match document.get("items").and_then(Value::as_array) {
        Some(items) => {
            let mut lines = Vec::with_capacity(items.len());
            for item in items {
                lines.push(summarize(item));
            }
            lines.join("\n")
        }
        None => summarize(document),
    }
}

fn summarize(item: &Value) -> String {
    let name = item
        .pointer("/metadata/name")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>");
    let namespace = item
        .pointer("/metadata/namespace")
        .and_then(Value::as_str)
        .unwrap_or("<none>");
    let phase = item
        .pointer("/status/phase")
        .and_then(Value::as_str)
        .unwrap_or("Unknown");
    format!("{namespace}/{name}\t{phase}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_defaults_to_text_and_rejects_unknown_values() {
        assert_eq!(
            OutputFormat::from_args(&["list".to_string()]).expect("default"),
            OutputFormat::Text
        );
        assert_eq!(
            OutputFormat::from_args(&["list".to_string(), "--output=json".to_string()])
                .expect("json"),
            OutputFormat::Json
        );
        assert!(
            OutputFormat::from_args(&["-o".to_string(), "yaml".to_string()]).is_err(),
            "unsupported formats must fail closed"
        );
    }

    #[test]
    fn text_rendering_summarizes_every_collection_item() {
        let body = br#"{"items":[
            {"metadata":{"name":"job-1","namespace":"bench"},"status":{"phase":"Running"}},
            {"metadata":{"name":"job-2","namespace":"bench"}}
        ]}"#;
        let rendered = render(OutputFormat::Text, body).expect("render");
        assert_eq!(rendered, "bench/job-1\tRunning\nbench/job-2\tUnknown");
    }

    #[test]
    fn text_rendering_summarizes_retained_result_index_items() {
        let body = br#"{"items":[
            {"metadata":{"name":"run-1","namespace":"bench"},"jobId":"job-1","ready":true,"artifactCount":3,"created":1700000000.0},
            {"metadata":{"name":"run-2","namespace":"bench"},"jobId":"job-2","ready":false,"artifactCount":0,"created":1700000001.0}
        ]}"#;
        let rendered = render(OutputFormat::Text, body).expect("render");
        assert_eq!(
            rendered,
            "bench/run-1\tjob-1\tReady\t3 artifact(s)\nbench/run-2\tjob-2\tPending\t0 artifact(s)"
        );
    }

    #[test]
    fn json_rendering_preserves_the_exact_document() {
        let body = br#"{"metadata":{"name":"job-1"}}"#;
        let rendered = render(OutputFormat::Json, body).expect("render");
        let parsed: Value = serde_json::from_str(&rendered).expect("parse");
        assert_eq!(
            parsed.pointer("/metadata/name").and_then(Value::as_str),
            Some("job-1")
        );
    }
}
