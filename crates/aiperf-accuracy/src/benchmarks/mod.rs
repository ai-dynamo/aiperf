// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native benchmark prompt builders ported from inherited AIPerf accuracy plugins.

mod aime;
mod bigbench;
mod common;
mod gpqa;
mod hellaswag;
mod lcb;
mod mmlu;
mod simple;

pub use aime::{AIME_SYSTEM_PROMPT, AimeBenchmark};
pub use bigbench::{BIGBENCH_TASKS, BigBenchBenchmark};
pub use gpqa::GpqaDiamondBenchmark;
pub use hellaswag::{HELLASWAG_MAX_N_SHOTS, HellaSwagBenchmark};
pub use lcb::LcbCodeGenerationBenchmark;
pub use mmlu::{MMLU_SUBJECTS, MmluBenchmark};
pub use simple::{Aime24Benchmark, Aime25Benchmark, Gsm8kBenchmark, Math500Benchmark};
