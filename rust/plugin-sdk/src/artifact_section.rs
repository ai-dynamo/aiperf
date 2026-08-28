// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read and write plugin build record sections embedded in binary artifacts.
//!
//! The section format is:
//!   - 4-byte magic: `b"APF1"`
//!   - 4-byte little-endian length (of the JSON payload)
//!   - JSON bytes of `PluginArtifactBuildRecordV1`
//!
//! Platform-specific section names:
//!   - Linux ELF:  `.aiperf_build_record`
//!   - macOS Mach-O: `__DATA,__aiperf_rec`
//!   - Windows PE: `.apfbrec`

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

use crate::identity::PluginArtifactBuildRecordV1;

const MAGIC: &[u8; 4] = b"APF1";

/// Encodes a `PluginArtifactBuildRecordV1` into the section payload bytes.
pub fn encode_section(record: &PluginArtifactBuildRecordV1) -> Vec<u8> {
    let json = serde_json::to_vec(record).expect("record serialization cannot fail");
    let len = json.len() as u32;
    let mut out = Vec::with_capacity(8 + json.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&json);
    out
}

/// Decodes the section payload bytes into a `PluginArtifactBuildRecordV1`.
///
/// Returns `None` when the magic bytes do not match (section absent or wrong format).
pub fn decode_section(
    bytes: &[u8],
) -> Result<Option<PluginArtifactBuildRecordV1>, ArtifactSectionError> {
    if bytes.len() < 8 {
        return Ok(None);
    }
    if &bytes[..4] != MAGIC {
        return Ok(None);
    }
    let len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let end = 8 + len;
    if bytes.len() < end {
        return Err(ArtifactSectionError::Truncated {
            declared_len: len,
            available: bytes.len() - 8,
        });
    }
    let record = serde_json::from_slice(&bytes[8..end])
        .map_err(|e| ArtifactSectionError::JsonParse(e.to_string()))?;
    Ok(Some(record))
}

/// Errors from artifact section operations.
#[derive(Debug)]
pub enum ArtifactSectionError {
    Io(io::Error),
    Truncated {
        declared_len: usize,
        available: usize,
    },
    JsonParse(String),
    ToolFailed {
        tool: String,
        stderr: String,
    },
    UnsupportedPlatform,
}

impl std::fmt::Display for ArtifactSectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactSectionError::Io(e) => write!(f, "I/O error: {e}"),
            ArtifactSectionError::Truncated {
                declared_len,
                available,
            } => write!(
                f,
                "section payload truncated: declared {declared_len} bytes, only {available} available"
            ),
            ArtifactSectionError::JsonParse(e) => write!(f, "JSON parse error: {e}"),
            ArtifactSectionError::ToolFailed { tool, stderr } => {
                write!(f, "{tool} failed: {stderr}")
            }
            ArtifactSectionError::UnsupportedPlatform => {
                write!(
                    f,
                    "artifact section embedding not supported on this platform"
                )
            }
        }
    }
}

impl std::error::Error for ArtifactSectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let ArtifactSectionError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<io::Error> for ArtifactSectionError {
    fn from(e: io::Error) -> Self {
        ArtifactSectionError::Io(e)
    }
}

/// Embeds `record` as a named section in the binary at `artifact_path`.
///
/// Uses `objcopy --add-section` (Linux/GNU) or `llvm-objcopy` (macOS/Windows).
/// The section content is the `encode_section` payload written to a temp file.
pub fn embed_record(
    artifact_path: &Path,
    record: &PluginArtifactBuildRecordV1,
) -> Result<(), ArtifactSectionError> {
    let payload = encode_section(record);

    // Write payload to a temp file that objcopy can read.
    let mut tmp = tempfile::NamedTempFile::new()?;
    tmp.write_all(&payload)?;
    let tmp_path = tmp.path().to_owned();
    // Keep the temp file alive until objcopy finishes.
    let _tmp_guard = tmp;

    #[cfg(target_os = "linux")]
    {
        let section_name = ".aiperf_build_record";
        let out = Command::new("objcopy")
            .arg(format!(
                "--add-section={section_name}={}",
                tmp_path.display()
            ))
            .arg(artifact_path)
            .output()?;
        if !out.status.success() {
            return Err(ArtifactSectionError::ToolFailed {
                tool: "objcopy".to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: llvm-objcopy with Mach-O section specifier.
        let out = Command::new("llvm-objcopy")
            .args([
                "--add-section",
                &format!("__DATA,__aiperf_rec={}", tmp_path.display()),
            ])
            .arg(artifact_path)
            .output()?;
        if !out.status.success() {
            return Err(ArtifactSectionError::ToolFailed {
                tool: "llvm-objcopy".to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    {
        let out = Command::new("llvm-objcopy")
            .args(["--add-section", &format!(".apfbrec={}", tmp_path.display())])
            .arg(artifact_path)
            .output()?;
        if !out.status.success() {
            return Err(ArtifactSectionError::ToolFailed {
                tool: "llvm-objcopy".to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(ArtifactSectionError::UnsupportedPlatform)
    }
}

/// Extracts a `PluginArtifactBuildRecordV1` from the named section of a binary.
///
/// Returns `None` when the section is absent or the magic bytes do not match.
pub fn extract_record(
    artifact_path: &Path,
) -> Result<Option<PluginArtifactBuildRecordV1>, ArtifactSectionError> {
    #[cfg(target_os = "linux")]
    let section_name = ".aiperf_build_record";
    #[cfg(target_os = "macos")]
    let section_name = "__DATA,__aiperf_rec";
    #[cfg(target_os = "windows")]
    let section_name = ".apfbrec";
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    return Err(ArtifactSectionError::UnsupportedPlatform);

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    {
        let out = Command::new("objcopy")
            .args(["--dump-section", &format!("{section_name}=/dev/stdout")])
            .arg(artifact_path)
            .output()?;
        if !out.status.success() {
            // Section absent — not an error.
            return Ok(None);
        }
        decode_section(&out.stdout)
    }
}
