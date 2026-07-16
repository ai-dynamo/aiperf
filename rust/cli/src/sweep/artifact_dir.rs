// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Per-cell artifact-directory resolution — a byte-exact port of
//! `orchestrator/orchestrator.py::_resolve_artifact_dir`.
//!
//! | sweep | trials | order       | layout                                          |
//! |-------|--------|-------------|-------------------------------------------------|
//! | no    | 1      | -           | `<base>/`                                       |
//! | no    | >1     | -           | `<base>/profile_runs/run_NNNN/`                 |
//! | yes   | 1      | -           | `<base>/<dir_name>/`                            |
//! | yes   | >1     | REPEATED    | `<base>/profile_runs/trial_NNNN/<dir_name>/`    |
//! | yes   | >1     | INDEPENDENT | `<base>/<dir_name>/profile_runs/trial_NNNN/`    |
//!
//! `trial_index` is zero-based; emitted names are 1-based, zero-padded to 4.
//! Note the asymmetry: `run_NNNN` (no-sweep multi-run) vs `trial_NNNN` (sweep).

use std::path::{Path, PathBuf};

/// Trial iteration order (`SweepMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterationOrder {
    /// Trials outer, variations inner (the default).
    Repeated,
    /// Variations outer, trials inner.
    Independent,
}

/// Resolve the artifact directory for one `(variation, trial)` cell.
pub fn resolve(
    base: &Path,
    is_sweep: bool,
    trials: u32,
    dir_name: &str,
    trial_index: u32,
    order: IterationOrder,
) -> PathBuf {
    let multi_run = trials > 1;
    let trial_1 = format!("trial_{:04}", trial_index + 1);
    let run_1 = format!("run_{:04}", trial_index + 1);
    match (is_sweep, multi_run) {
        (false, false) => base.to_path_buf(),
        (false, true) => base.join("profile_runs").join(run_1),
        (true, false) => base.join(dir_name),
        (true, true) => match order {
            IterationOrder::Repeated => base.join("profile_runs").join(trial_1).join(dir_name),
            IterationOrder::Independent => base.join(dir_name).join("profile_runs").join(trial_1),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_row_table() {
        let b = Path::new("/base");
        // no sweep, single trial
        assert_eq!(resolve(b, false, 1, "", 0, IterationOrder::Repeated), b);
        // no sweep, multi-run -> run_NNNN
        assert_eq!(
            resolve(b, false, 3, "", 1, IterationOrder::Repeated),
            b.join("profile_runs").join("run_0002")
        );
        // sweep, single trial -> <dir_name>
        assert_eq!(
            resolve(b, true, 1, "concurrency_4", 0, IterationOrder::Repeated),
            b.join("concurrency_4")
        );
        // sweep, multi-run, REPEATED -> profile_runs/trial_NNNN/<dir_name>
        assert_eq!(
            resolve(b, true, 2, "concurrency_4", 0, IterationOrder::Repeated),
            b.join("profile_runs")
                .join("trial_0001")
                .join("concurrency_4")
        );
        // sweep, multi-run, INDEPENDENT -> <dir_name>/profile_runs/trial_NNNN
        assert_eq!(
            resolve(b, true, 2, "concurrency_4", 1, IterationOrder::Independent),
            b.join("concurrency_4")
                .join("profile_runs")
                .join("trial_0002")
        );
    }
}
