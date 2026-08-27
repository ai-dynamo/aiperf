// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic command-line entry point for plugin runtime parity reports.

use std::{
    error::Error,
    fs,
    io::{self, Write},
};

use aiperf_bench_tools::plugin_stats::{decode_samples_jsonl, encode_samples_jsonl};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [mode, input] if mode == "canonicalize-jsonl" => {
            let samples = decode_samples_jsonl(&fs::read(input)?)?;
            io::stdout().write_all(&encode_samples_jsonl(&samples)?)?;
        }
        [mode, ..] if mode == "evaluate" => {
            return Err(
                "authoritative plugin parity evaluation requires a same-process controlled measurement capability; standalone JSON evaluation is unavailable"
                    .into(),
            );
        }
        _ => {
            return Err("usage: plugin-runtime-bench canonicalize-jsonl <samples.jsonl>".into());
        }
    }
    Ok(())
}
