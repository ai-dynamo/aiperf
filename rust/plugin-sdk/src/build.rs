// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hermetic plugin build pre-flight and execution.

use std::path::{Path, PathBuf};

/// Configuration for building a plugin.
pub struct BuildConfig {
    /// Directory containing the plugin's `Cargo.toml` and `plugin.toml`.
    pub plugin_dir: PathBuf,
    /// Optional path to an alternate SDK directory.
    pub sdk_dir: Option<PathBuf>,
    /// Build in release mode (required for production; enforces `panic=abort`).
    pub release: bool,
    /// Cross-compilation target triple (defaults to host triple).
    pub target: Option<String>,
}

/// Errors from preflight or build.
#[derive(Debug)]
pub enum BuildError {
    MissingCdylibCrateType,
    MissingPanicAbort,
    MissingManifest,
    InvalidManifest(crate::manifest::ManifestError),
    MissingCargoToml,
    CargoTomlParse(String),
    Io(std::io::Error),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCdylibCrateType => {
                write!(f, "Cargo.toml must declare crate-type = [\"cdylib\"]")
            }
            Self::MissingPanicAbort => {
                write!(
                    f,
                    "Cargo.toml must set panic = \"abort\" in [profile.release]"
                )
            }
            Self::MissingManifest => write!(f, "plugin.toml manifest not found"),
            Self::InvalidManifest(e) => write!(f, "invalid manifest: {e}"),
            Self::MissingCargoToml => write!(f, "Cargo.toml not found in plugin directory"),
            Self::CargoTomlParse(e) => write!(f, "Cargo.toml parse error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::InvalidManifest(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for BuildError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Validate the plugin directory structure without running cargo.
///
/// Checks:
/// 1. `Cargo.toml` is present and declares `crate-type = ["cdylib"]`.
/// 2. `plugin.toml` is present and valid.
/// 3. When `release` is set, `[profile.release] panic = "abort"` is declared.
pub fn preflight_plugin(config: &BuildConfig) -> Result<(), BuildError> {
    let cargo_toml_path = config.plugin_dir.join("Cargo.toml");
    if !cargo_toml_path.exists() {
        return Err(BuildError::MissingCargoToml);
    }
    let cargo_toml_bytes = std::fs::read(&cargo_toml_path).map_err(BuildError::Io)?;
    let cargo_toml_str = String::from_utf8(cargo_toml_bytes)
        .map_err(|e| BuildError::CargoTomlParse(e.to_string()))?;
    let cargo_val: toml::Value =
        toml::from_str(&cargo_toml_str).map_err(|e| BuildError::CargoTomlParse(e.to_string()))?;

    // Verify cdylib crate-type.
    let has_cdylib = cargo_val
        .get("lib")
        .and_then(|lib| lib.get("crate-type"))
        .and_then(|ct| ct.as_array())
        .map(|arr| arr.iter().any(|v| v.as_str() == Some("cdylib")))
        .unwrap_or(false);
    if !has_cdylib {
        return Err(BuildError::MissingCdylibCrateType);
    }

    // Verify panic=abort in release profile when building --release.
    if config.release {
        let has_panic_abort = cargo_val
            .get("profile")
            .and_then(|p| p.get("release"))
            .and_then(|r| r.get("panic"))
            .and_then(|v| v.as_str())
            == Some("abort");
        if !has_panic_abort {
            return Err(BuildError::MissingPanicAbort);
        }
    }

    // Verify plugin.toml is present and valid.
    let manifest_path = config.plugin_dir.join("plugin.toml");
    if !manifest_path.exists() {
        return Err(BuildError::MissingManifest);
    }
    let manifest_bytes = std::fs::read(&manifest_path).map_err(BuildError::Io)?;
    let manifest =
        crate::manifest::parse_manifest(&manifest_bytes).map_err(BuildError::InvalidManifest)?;
    crate::manifest::validate_manifest(&manifest).map_err(BuildError::InvalidManifest)?;

    Ok(())
}

/// Build a plugin cdylib, returning the path to the output artifact.
///
/// Runs preflight validation first. The actual `cargo build` invocation uses
/// `--offline` to enforce hermetic builds (no network downloads).
pub fn build_plugin(config: &BuildConfig) -> Result<PathBuf, BuildError> {
    preflight_plugin(config)?;
    run_cargo_build(config)
}

fn run_cargo_build(config: &BuildConfig) -> Result<PathBuf, BuildError> {
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build");
    cmd.arg("--offline");
    if config.release {
        cmd.arg("--release");
    }
    if let Some(target) = &config.target {
        cmd.args(["--target", target]);
    }
    cmd.current_dir(&config.plugin_dir);

    // Restrict environment to the allowed set.
    let allowed = [
        "PATH",
        "HOME",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CARGO_TARGET_DIR",
        "TMPDIR",
        "TMP",
        "TEMP",
        "OUT_DIR",
    ];
    let filtered: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| allowed.contains(&k.as_str()))
        .collect();
    cmd.env_clear();
    for (k, v) in &filtered {
        cmd.env(k, v);
    }

    let status = cmd.status().map_err(BuildError::Io)?;
    if !status.success() {
        return Err(BuildError::Io(std::io::Error::other("cargo build failed")));
    }

    // Determine artifact path.
    let profile_dir = if config.release { "release" } else { "debug" };
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| config.plugin_dir.join("target"));
    let lib_name = cargo_package_name(&config.plugin_dir)?;
    let artifact = find_cdylib(
        &target_dir,
        profile_dir,
        &lib_name,
        config.target.as_deref(),
    )?;
    Ok(artifact)
}

fn cargo_package_name(plugin_dir: &Path) -> Result<String, BuildError> {
    let bytes = std::fs::read(plugin_dir.join("Cargo.toml")).map_err(BuildError::Io)?;
    let s = String::from_utf8(bytes).map_err(|e| BuildError::CargoTomlParse(e.to_string()))?;
    let val: toml::Value =
        toml::from_str(&s).map_err(|e| BuildError::CargoTomlParse(e.to_string()))?;
    val.get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .map(|n| n.replace('-', "_"))
        .ok_or_else(|| BuildError::CargoTomlParse("missing [package] name".to_string()))
}

fn find_cdylib(
    target_dir: &Path,
    profile_dir: &str,
    lib_name: &str,
    target: Option<&str>,
) -> Result<PathBuf, BuildError> {
    let base = if let Some(t) = target {
        target_dir.join(t).join(profile_dir)
    } else {
        target_dir.join(profile_dir)
    };

    for (prefix, suffix) in [("lib", ".so"), ("lib", ".dylib"), ("", ".dll")] {
        let candidate = base.join(format!("{prefix}{lib_name}{suffix}"));
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(BuildError::Io(std::io::Error::other(format!(
        "no cdylib artifact found in {}",
        base.display()
    ))))
}
