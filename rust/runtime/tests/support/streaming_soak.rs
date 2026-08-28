// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming resource soak support: configuration, process probes, and budget
//! high-water records.

use std::{env, fmt, num::NonZeroU64, path::PathBuf};

use aiperf_runtime::streaming::budget::{BudgetLimits, BudgetSnapshot};

/// One mebibyte, as the byte unit every threshold in this module uses.
pub const MIB: u64 = 1024 * 1024;

/// Default immutable source the soak run reads from.
const DEFAULT_SOURCE_ID: &str = "hf_hub";
/// Default record format the soak run decodes.
const DEFAULT_FORMAT_ID: &str = "baseten_trace";
/// Default number of source objects the soak run keeps in flight.
const DEFAULT_OBJECT_CONCURRENCY: usize = 8;
/// Default fault period, in records, between injected faults.
const DEFAULT_FAULT_PERIOD: u64 = 997;

/// Resolved authored configuration for one streaming resource soak run.
#[derive(Clone, Debug)]
pub struct SoakConfig {
    /// Absolute directory the run may create scratch state under.
    pub scratch_dir: PathBuf,
    /// Logical input volume, in gibibytes.
    pub input_gib: NonZeroU64,
    /// Logical stream duration, in hours.
    pub logical_hours: NonZeroU64,
    /// Registered immutable-source identifier.
    pub source_id: String,
    /// Registered record-format identifier.
    pub format_id: String,
    /// Source objects kept in flight concurrently.
    pub object_concurrency: usize,
    /// Records between injected faults.
    pub fault_period: NonZeroU64,
}

/// Soak configuration, probe, or observation failure.
#[derive(Debug)]
pub enum SoakError {
    /// A required environment variable is absent or empty.
    MissingEnv {
        /// Variable name.
        name: &'static str,
    },
    /// A present environment variable does not parse or is out of range.
    InvalidEnv {
        /// Variable name.
        name: &'static str,
        /// Rejected value.
        value: String,
        /// Why the value was refused.
        reason: &'static str,
    },
    /// The authored scratch directory is not a location the run may own.
    UnsafeScratch {
        /// Rejected path.
        path: PathBuf,
        /// Why the path was refused.
        reason: &'static str,
    },
    /// A soak fixture could not be materialized.
    Fixture {
        /// Fixture path.
        path: PathBuf,
        /// Underlying filesystem failure.
        source: std::io::Error,
    },
    /// A `/proc/self` probe could not be read or parsed.
    Probe {
        /// Probe source path.
        source_path: &'static str,
        /// Why the probe failed.
        reason: &'static str,
    },
    /// The soak run itself failed.
    Run {
        /// Failure detail.
        detail: String,
    },
    /// A recorded observation is inconsistent.
    Observation {
        /// Inconsistency detail.
        detail: String,
    },
}

impl fmt::Display for SoakError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEnv { name } => {
                write!(formatter, "required soak environment variable {name} is unset")
            }
            Self::InvalidEnv {
                name,
                value,
                reason,
            } => write!(
                formatter,
                "soak environment variable {name} value {value:?} is invalid: {reason}"
            ),
            Self::UnsafeScratch { path, reason } => write!(
                formatter,
                "soak scratch directory {} is unusable: {reason}",
                path.display()
            ),
            Self::Fixture { path, source } => write!(
                formatter,
                "soak fixture {} could not be prepared: {source}",
                path.display()
            ),
            Self::Probe {
                source_path,
                reason,
            } => write!(formatter, "process probe {source_path} failed: {reason}"),
            Self::Run { detail } => write!(formatter, "soak run failed: {detail}"),
            Self::Observation { detail } => {
                write!(formatter, "soak observation is inconsistent: {detail}")
            }
        }
    }
}

impl std::error::Error for SoakError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fixture { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl SoakConfig {
    /// Read and validate the soak environment.
    ///
    /// `AIPERF_STREAM_SOAK_DIR`, `AIPERF_STREAM_SOAK_GIB`, and
    /// `AIPERF_STREAM_SOAK_LOGICAL_HOURS` are required. The scratch directory
    /// must be absolute, must be neither the filesystem root nor `$HOME`, and
    /// must sit under an existing parent directory the process may create in.
    /// Optional `AIPERF_STREAM_SOAK_SOURCE`, `_FORMAT`, `_OBJECT_CONCURRENCY`,
    /// and `_FAULT_PERIOD` override the compiled defaults.
    ///
    /// # Errors
    ///
    /// Returns [`SoakError::MissingEnv`], [`SoakError::InvalidEnv`], or
    /// [`SoakError::UnsafeScratch`].
    pub fn from_env() -> Result<Self, SoakError> {
        let scratch_dir = required_env("AIPERF_STREAM_SOAK_DIR")?;
        let scratch_dir = validate_scratch_dir(PathBuf::from(scratch_dir))?;

        let input_gib = required_nonzero_env("AIPERF_STREAM_SOAK_GIB")?;
        let logical_hours = required_nonzero_env("AIPERF_STREAM_SOAK_LOGICAL_HOURS")?;

        let source_id = optional_env("AIPERF_STREAM_SOAK_SOURCE")
            .unwrap_or_else(|| DEFAULT_SOURCE_ID.to_string());
        let format_id = optional_env("AIPERF_STREAM_SOAK_FORMAT")
            .unwrap_or_else(|| DEFAULT_FORMAT_ID.to_string());

        let object_concurrency = match optional_env("AIPERF_STREAM_SOAK_OBJECT_CONCURRENCY") {
            Some(value) => {
                let parsed: usize = value.parse().map_err(|_| SoakError::InvalidEnv {
                    name: "AIPERF_STREAM_SOAK_OBJECT_CONCURRENCY",
                    value: value.clone(),
                    reason: "not a non-negative integer",
                })?;
                if parsed == 0 {
                    return Err(SoakError::InvalidEnv {
                        name: "AIPERF_STREAM_SOAK_OBJECT_CONCURRENCY",
                        value,
                        reason: "must be greater than zero",
                    });
                }
                parsed
            }
            None => DEFAULT_OBJECT_CONCURRENCY,
        };

        let fault_period = match optional_env("AIPERF_STREAM_SOAK_FAULT_PERIOD") {
            Some(value) => parse_nonzero("AIPERF_STREAM_SOAK_FAULT_PERIOD", value)?,
            // The default is a compile-time nonzero constant.
            None => NonZeroU64::new(DEFAULT_FAULT_PERIOD).unwrap_or(NonZeroU64::MIN),
        };

        Ok(Self {
            scratch_dir,
            input_gib,
            logical_hours,
            source_id,
            format_id,
            object_concurrency,
            fault_period,
        })
    }
}

fn optional_env(name: &'static str) -> Option<String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Some(value),
        _ => None,
    }
}

fn required_env(name: &'static str) -> Result<String, SoakError> {
    optional_env(name).ok_or(SoakError::MissingEnv { name })
}

fn parse_nonzero(name: &'static str, value: String) -> Result<NonZeroU64, SoakError> {
    let parsed: u64 = value.parse().map_err(|_| SoakError::InvalidEnv {
        name,
        value: value.clone(),
        reason: "not a non-negative integer",
    })?;
    NonZeroU64::new(parsed).ok_or(SoakError::InvalidEnv {
        name,
        value,
        reason: "must be greater than zero",
    })
}

fn required_nonzero_env(name: &'static str) -> Result<NonZeroU64, SoakError> {
    parse_nonzero(name, required_env(name)?)
}

/// Refuse any scratch path the soak run must not own or create under.
fn validate_scratch_dir(path: PathBuf) -> Result<PathBuf, SoakError> {
    if !path.is_absolute() {
        return Err(SoakError::UnsafeScratch {
            path,
            reason: "must be an absolute path",
        });
    }

    let Some(parent) = path.parent() else {
        return Err(SoakError::UnsafeScratch {
            path,
            reason: "the filesystem root is not a scratch directory",
        });
    };

    if let Some(home) = env::var_os("HOME")
        && !home.is_empty()
        && path == PathBuf::from(home)
    {
        return Err(SoakError::UnsafeScratch {
            path,
            reason: "the home directory is not a scratch directory",
        });
    }

    if !parent.is_dir() {
        return Err(SoakError::UnsafeScratch {
            path: path.clone(),
            reason: "parent directory does not exist",
        });
    }

    Ok(path)
}

/// One process-wide resource sample.
#[derive(Clone, Copy, Debug)]
pub struct ProcessSample {
    /// Resident set size in bytes, from `/proc/self/statm` field 1 times the
    /// page size.
    pub rss_bytes: u64,
    /// Open file descriptors, excluding the probe's own directory handle.
    pub open_fds: usize,
}

/// Sample this process's RSS and open-descriptor count from `/proc/self`.
///
/// Reading `/proc/self/fd` itself holds one descriptor open, so the returned
/// count is decremented by exactly one to exclude the probe's own handle.
///
/// # Errors
///
/// Returns [`SoakError::Probe`] when either procfs entry cannot be read or
/// parsed.
#[cfg(target_os = "linux")]
pub fn sample_process() -> Result<ProcessSample, SoakError> {
    const STATM: &str = "/proc/self/statm";
    const FD_DIR: &str = "/proc/self/fd";

    let statm = std::fs::read_to_string(STATM).map_err(|_| SoakError::Probe {
        source_path: STATM,
        reason: "unreadable",
    })?;
    let resident_pages: u64 = statm
        .split_ascii_whitespace()
        .nth(1)
        .ok_or(SoakError::Probe {
            source_path: STATM,
            reason: "missing resident-page field",
        })?
        .parse()
        .map_err(|_| SoakError::Probe {
            source_path: STATM,
            reason: "resident-page field is not an integer",
        })?;

    // `sysconf(_SC_PAGESIZE)` cannot fail on Linux; it is a fixed positive
    // kernel constant, so a non-positive result is treated as a probe failure
    // rather than unwrapped.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page_size).map_err(|_| SoakError::Probe {
        source_path: STATM,
        reason: "page size is not positive",
    })?;

    let rss_bytes = resident_pages
        .checked_mul(page_size)
        .ok_or(SoakError::Probe {
            source_path: STATM,
            reason: "resident size is not representable",
        })?;

    let mut entries = 0_usize;
    for entry in std::fs::read_dir(FD_DIR).map_err(|_| SoakError::Probe {
        source_path: FD_DIR,
        reason: "unreadable",
    })? {
        entry.map_err(|_| SoakError::Probe {
            source_path: FD_DIR,
            reason: "directory entry unreadable",
        })?;
        entries += 1;
    }
    // The `read_dir` handle is itself one of the counted descriptors.
    let open_fds = entries.saturating_sub(1);

    Ok(ProcessSample {
        rss_bytes,
        open_fds,
    })
}

/// Read the peak resident set size from `VmHWM:` in `/proc/self/status`.
///
/// `VmHWM` is kernel-maintained, so it never under-reports the way a polled
/// maximum of [`sample_process`] would between samples.
///
/// # Errors
///
/// Returns [`SoakError::Probe`] when the entry cannot be read or parsed.
#[cfg(target_os = "linux")]
pub fn peak_rss_bytes() -> Result<u64, SoakError> {
    const STATUS: &str = "/proc/self/status";

    let status = std::fs::read_to_string(STATUS).map_err(|_| SoakError::Probe {
        source_path: STATUS,
        reason: "unreadable",
    })?;
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or(SoakError::Probe {
            source_path: STATUS,
            reason: "missing VmHWM field",
        })?;
    let kibibytes: u64 = line
        .split_ascii_whitespace()
        .next()
        .ok_or(SoakError::Probe {
            source_path: STATUS,
            reason: "VmHWM field has no value",
        })?
        .parse()
        .map_err(|_| SoakError::Probe {
            source_path: STATUS,
            reason: "VmHWM value is not an integer",
        })?;

    kibibytes.checked_mul(1024).ok_or(SoakError::Probe {
        source_path: STATUS,
        reason: "VmHWM value is not representable in bytes",
    })
}

/// A named budget's authored limits and observed peaks.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct StateHighWater {
    /// Authored maximum retained items.
    pub item_limit: usize,
    /// Authored maximum retained bytes.
    pub byte_limit: usize,
    /// Greatest observed item charge.
    pub high_water_items: usize,
    /// Greatest observed byte charge.
    pub high_water_bytes: usize,
    /// Item charge still outstanding at drain.
    pub residual_items: usize,
    /// Byte charge still outstanding at drain.
    pub residual_bytes: usize,
}

impl StateHighWater {
    /// Build a record from an authored limit and the budget's snapshot.
    #[must_use]
    pub fn from_snapshot(limits: BudgetLimits, snapshot: BudgetSnapshot) -> Self {
        Self {
            item_limit: limits.max_items,
            byte_limit: limits.max_bytes,
            high_water_items: snapshot.high_water_items,
            high_water_bytes: snapshot.high_water_bytes,
            residual_items: snapshot.used_items,
            residual_bytes: snapshot.used_bytes,
        }
    }

    /// Whether peaks stayed within authored limits and nothing leaked at drain.
    ///
    /// The residual checks are what turn a bound check into a leak check: a
    /// budget can respect its ceiling while never releasing what it charged.
    #[must_use]
    pub fn is_within_budget(&self) -> bool {
        self.high_water_items <= self.item_limit
            && self.high_water_bytes <= self.byte_limit
            && self.residual_items == 0
            && self.residual_bytes == 0
    }
}
