// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builds the typed bridge to libmimalloc-sys experimental option constants.

use std::path::PathBuf;

fn main() {
    let include_dir = std::env::var_os("DEP_MIMALLOC_INCLUDE_DIR")
        .map(PathBuf::from)
        .expect("libmimalloc-sys did not expose its compiled header directory");
    println!("cargo:rerun-if-changed=src/mimalloc_options.c");
    cc::Build::new()
        .include(include_dir)
        .file("src/mimalloc_options.c")
        .warnings_into_errors(true)
        .compile("aiperf_mimalloc_options");
}
