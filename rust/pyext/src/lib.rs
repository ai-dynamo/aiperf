// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf._native` — the compiled target of the single interned `aiperf` wheel.
//!
//! This module exists so maturin has a legal binding target to build alongside
//! the `aiperf` `[project.scripts]` console command (maturin forbids
//! `bindings = "bin"` there; see the crate-level docs in `Cargo.toml`). The
//! full-fat native `aiperf` executable ships in the same wheel as interned
//! package data at `aiperf/_bin/aiperf`, resolved at runtime via
//! `importlib.resources` — that path deliberately does **not** import this
//! module, so a load failure here never blocks binary discovery.
//!
//! The functions below expose build metadata for `aiperf --version` / support
//! diagnostics only.

use pyo3::prelude::*;

/// Basename of the interned native executable inside the wheel.
///
/// The single authority for the filename shared by the maturin `include` glob
/// (`aiperf/_bin/aiperf`) and Python discovery.
#[pyfunction]
fn runner_filename() -> &'static str {
    "aiperf"
}

/// Package-relative path of the interned binary within the installed `aiperf`
/// package (POSIX separators; the wheel is built on Unix build hosts).
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
