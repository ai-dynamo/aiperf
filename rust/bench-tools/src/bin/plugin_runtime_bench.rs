// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic command-line entry point for plugin runtime parity reports.

use std::{
    error::Error,
    fs,
    io::{self, Write},
};

use aiperf_bench_tools::plugin_stats::{
    PairedCase, SimultaneousGatePolicy, decode_samples_jsonl, encode_samples_jsonl,
    evaluate_simultaneous_gate,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [mode, input] if mode == "canonicalize-jsonl" => {
            let samples = decode_samples_jsonl(&fs::read(input)?)?;
            io::stdout().write_all(&encode_samples_jsonl(&samples)?)?;
        }
        [mode, input, seed] if mode == "evaluate" => {
            let cases: Vec<PairedCase> = serde_json::from_slice(&fs::read(input)?)?;
            let bootstrap_seed = seed.parse::<u64>()?;
            let report = evaluate_simultaneous_gate(
                &cases,
                &SimultaneousGatePolicy::normative(),
                bootstrap_seed,
            )?;
            let stdout = io::stdout();
            let mut output = stdout.lock();
            serde_json::to_writer(&mut output, &report)?;
            output.write_all(b"\n")?;
        }
        _ => {
            return Err(
                "usage: plugin-runtime-bench canonicalize-jsonl <samples.jsonl> | evaluate <cases.json> <seed>"
                    .into(),
            );
        }
    }
    Ok(())
}
