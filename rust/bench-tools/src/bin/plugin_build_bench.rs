// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Refusal entry point for superseded standalone build measurements.

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    Err(
        "standalone build measurement is non-authoritative; use the same-process paired build controller"
            .into(),
    )
}
