// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic command-line entry point for plugin runtime parity reports.

use std::{
    error::Error,
    fs,
    io::{self, Write},
};

use aiperf_bench_tools::plugin_stats::{
    ExperimentObservationReceipt, ObservedExperimentAuthority, SimultaneousGateInput,
    SimultaneousGatePolicy, decode_samples_jsonl, encode_samples_jsonl, evaluate_simultaneous_gate,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [mode, input] if mode == "canonicalize-jsonl" => {
            let samples = decode_samples_jsonl(&fs::read(input)?)?;
            io::stdout().write_all(&encode_samples_jsonl(&samples)?)?;
        }
        [mode, observation_path, input] if mode == "evaluate" => {
            let observation: ExperimentObservationReceipt =
                serde_json::from_slice(&fs::read(observation_path)?)?;
            let authority = ObservedExperimentAuthority::acquire(&observation)?;
            let input: SimultaneousGateInput = serde_json::from_slice(&fs::read(input)?)?;
            let report = evaluate_simultaneous_gate(
                &input,
                &authority,
                &SimultaneousGatePolicy::normative(),
            )?;
            let stdout = io::stdout();
            let mut output = stdout.lock();
            serde_json::to_writer(&mut output, &report)?;
            output.write_all(b"\n")?;
        }
        _ => {
            return Err(
                "usage: plugin-runtime-bench canonicalize-jsonl <samples.jsonl> | evaluate <harness-observation.json> <input.json>"
                    .into(),
            );
        }
    }
    Ok(())
}
