// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build metadata for the `aiperf._native` wheel extension.
//!
//! Maturin builds this PyO3 module. Wheel repacking injects the native executable
//! into the wheel's `.data/scripts/aiperf` path, which pip installs as the
//! `aiperf` command. This extension does not discover or launch that executable.

use pyo3::prelude::*;

/// Native executable basename reported in build metadata.
#[pyfunction]
fn runner_filename() -> &'static str {
    "aiperf"
}

/// Return the `_bin/aiperf` metadata value.
///
/// Wheel installation and executable discovery do not use this value.
#[pyfunction]
fn runner_relpath() -> &'static str {
    "_bin/aiperf"
}

/// Cargo profile this extension (and, by the same maturin build, the wheel) was
/// compiled with.
#[pyfunction]
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// Version string of the compiled extension crate (workspace version).
#[pyfunction]
fn pyext_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `aiperf._native` build-metadata module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(runner_filename, module)?)?;
    module.add_function(wrap_pyfunction!(runner_relpath, module)?)?;
    module.add_function(wrap_pyfunction!(build_profile, module)?)?;
    module.add_function(wrap_pyfunction!(pyext_version, module)?)?;
    Ok(())
}
