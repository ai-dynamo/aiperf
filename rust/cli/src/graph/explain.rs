// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Task-4 explanation presentation boundary.

use super::LoadedGraphInput;

/// Emit a compact success line after the single retained load.
pub(super) fn run(input: LoadedGraphInput) {
    let LoadedGraphInput { source, prepared } = input;
    let _ = prepared;
    println!("graph explain loaded {}", source.display());
}
