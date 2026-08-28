// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writing exporter output through the artifact capability alone.
//!
//! Every function here takes `&dyn ArtifactAccess` and a relative path. None
//! takes, returns, or derives a directory: the capability approves each path
//! itself, so a helper cannot widen what the exporter was granted. An
//! [`ArtifactError`] is mapped to [`ExporterError::Artifact`] with the
//! capability's own message, so the refusal an exporter reports is the refusal
//! the host issued.

use aiperf_core::artifact::{ArtifactAccess, ArtifactError};
use aiperf_plugin_api::ExporterError;
use serde::Serialize;

/// Map one artifact refusal into the exporter error vocabulary.
fn artifact_failed(error: ArtifactError) -> ExporterError {
    ExporterError::Artifact(error.to_string())
}

/// Create or replace one artifact with exactly `contents`.
pub fn write_bytes(
    artifacts: &dyn ArtifactAccess,
    relative_path: &str,
    contents: &[u8],
) -> Result<(), ExporterError> {
    artifacts
        .create(relative_path, contents)
        .map_err(artifact_failed)
}

/// Append `contents` to one artifact, creating it when absent.
pub fn append_bytes(
    artifacts: &dyn ArtifactAccess,
    relative_path: &str,
    contents: &[u8],
) -> Result<(), ExporterError> {
    artifacts
        .append(relative_path, contents)
        .map_err(artifact_failed)
}

/// Create or replace one UTF-8 text artifact.
pub fn write_text(
    artifacts: &dyn ArtifactAccess,
    relative_path: &str,
    contents: &str,
) -> Result<(), ExporterError> {
    write_bytes(artifacts, relative_path, contents.as_bytes())
}

/// Serialize `value` as pretty JSON and write it as one artifact.
///
/// Pretty rather than compact because these artifacts are read by people as
/// often as by tools, and the host's own finalized report is committed the same
/// way.
pub fn write_json(
    artifacts: &dyn ArtifactAccess,
    relative_path: &str,
    value: &impl Serialize,
) -> Result<(), ExporterError> {
    let json = serde_json::to_vec_pretty(value)
        .map_err(|error| ExporterError::Backend(format!("serializing {relative_path}: {error}")))?;
    write_bytes(artifacts, relative_path, &json)
}

/// Write one CRLF-terminated CSV artifact from a header and its rows.
///
/// The whole table is built in memory before a single `create`, so a failing row
/// leaves no truncated artifact behind. Rows are `IntoIterator` over fields so a
/// caller can stream owned or borrowed strings without collecting twice.
pub fn write_csv<Row, Field>(
    artifacts: &dyn ArtifactAccess,
    relative_path: &str,
    header: impl IntoIterator<Item = Field>,
    rows: impl IntoIterator<Item = Row>,
) -> Result<(), ExporterError>
where
    Row: IntoIterator<Item = Field>,
    Field: AsRef<[u8]>,
{
    let mut writer = crate::helpers::crlf_csv_writer(Vec::new());
    let write_failed =
        |error: csv::Error| ExporterError::Backend(format!("writing {relative_path}: {error}"));
    writer.write_record(header).map_err(write_failed)?;
    for row in rows {
        writer.write_record(row).map_err(write_failed)?;
    }
    let table = writer
        .into_inner()
        .map_err(|error| ExporterError::Backend(format!("flushing {relative_path}: {error}")))?;
    write_bytes(artifacts, relative_path, &table)
}

#[cfg(test)]
mod tests {
    use aiperf_core::artifact::DirectoryArtifacts;

    use super::*;

    #[test]
    fn a_csv_artifact_is_crlf_terminated_and_written_once() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let artifacts = DirectoryArtifacts::new(root.path());
        write_csv(
            &artifacts,
            "table.csv",
            ["metric", "avg"],
            [["ttft", "1.5"]],
        )
        .expect("write csv");
        assert_eq!(
            artifacts.read("table.csv").expect("read"),
            b"metric,avg\r\nttft,1.5\r\n"
        );
    }

    #[test]
    fn json_round_trips_through_the_capability() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let artifacts = DirectoryArtifacts::new(root.path());
        write_json(
            &artifacts,
            "nested/out.json",
            &serde_json::json!({ "a": 1 }),
        )
        .expect("write json");
        let written = artifacts.read("nested/out.json").expect("read");
        let parsed: serde_json::Value = serde_json::from_slice(&written).expect("parse");
        assert_eq!(parsed, serde_json::json!({ "a": 1 }));
    }

    #[test]
    fn a_path_the_capability_refuses_becomes_an_artifact_error() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let artifacts = DirectoryArtifacts::new(root.path());
        assert!(matches!(
            write_text(&artifacts, "../escape.txt", "x"),
            Err(ExporterError::Artifact(_))
        ));
    }
}
