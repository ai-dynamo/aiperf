// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SLURM allocation -> cellular topology mapping.
//!
//! Under an `srun`/`sbatch` allocation every task runs the same `aiperf slurm run`
//! command; the task's role and identity come entirely from the `SLURM_*`
//! environment the allocation injects, not from an operator or a DNS record. This
//! module is the one place that decision lives, kept as pure functions so it is
//! exhaustively unit-testable without a live SLURM allocation:
//!
//! - task rank 0 (`SLURM_PROCID == 0`) is the cellular **controller**;
//! - ranks `1..ntasks` are **cells**, cell `id = rank - 1`, `cell_count = ntasks - 1`;
//! - the controller coordinate every cell dials is derived from the allocation's
//!   rank-0 node hostname (the first host in `SLURM_JOB_NODELIST`) plus the velo
//!   bootstrap port, so discovery is zero-round-trip and matches on every task.
//!
//! The rank-0 node is taken to be the first node of the expanded nodelist. This is
//! SLURM's default block task distribution (task 0 lands on the first node of the
//! allocation); an explicit `AIPERF_SLURM_CONTROLLER_HOST` override is honored first
//! for exotic distributions or user-pinned placement.

use std::fmt::{self, Display, Formatter};

/// The velo bootstrap port the controller binds and cells dial. Shared with the
/// k8s path's `AIPERF_CONTROLLER_PORT` so there is one port convention, not two.
pub const CONTROLLER_PORT_ENV: &str = "AIPERF_CONTROLLER_PORT";

/// Explicit override for the rank-0 (controller) node hostname. Honored before the
/// `SLURM_JOB_NODELIST` first-host derivation, for allocations whose task
/// distribution does not place `SLURM_PROCID == 0` on the first node.
pub const CONTROLLER_HOST_ENV: &str = "AIPERF_SLURM_CONTROLLER_HOST";

/// Default velo bootstrap port when [`CONTROLLER_PORT_ENV`] is unset (matches the
/// k8s controller default in
/// [`cellular_controller`](crate::engine::cellular_controller)).
pub const DEFAULT_CONTROLLER_PORT: u16 = 9500;

/// A resolved SLURM task's place in the cellular topology.
///
/// Constructed from the `SLURM_*` environment via [`SlurmTopology::from_env`] (or
/// [`SlurmTopology::new`] for tests). All role/identity queries are pure functions
/// of `proc_id`, `ntasks`, and the resolved `controller_host`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlurmTopology {
    /// This task's global rank in the allocation (`SLURM_PROCID`).
    proc_id: u32,
    /// The total number of tasks in the allocation (`SLURM_NTASKS`).
    ntasks: u32,
    /// The resolved rank-0 (controller) node hostname.
    controller_host: String,
}

/// Why a SLURM allocation could not be mapped to a cellular topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlurmTopologyError {
    /// A required `SLURM_*` variable was absent.
    MissingEnv(&'static str),
    /// A `SLURM_*` variable was present but not a valid integer.
    InvalidEnv {
        /// The offending variable name.
        var: &'static str,
        /// Its raw (rejected) value.
        value: String,
    },
    /// The allocation has fewer than two tasks: a cellular run needs one controller
    /// task plus at least one cell task.
    TooFewTasks(u32),
    /// A task's rank is `>= ntasks`, which the allocation should never produce.
    RankOutOfRange {
        /// The offending rank.
        proc_id: u32,
        /// The allocation's task count.
        ntasks: u32,
    },
    /// `SLURM_JOB_NODELIST` was present but expanded to no hosts (and no explicit
    /// [`CONTROLLER_HOST_ENV`] override was set).
    EmptyNodelist(String),
}

impl Display for SlurmTopologyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEnv(var) => write!(
                f,
                "missing SLURM environment variable `{var}` (is this running under srun/sbatch?)"
            ),
            Self::InvalidEnv { var, value } => {
                write!(
                    f,
                    "SLURM environment variable `{var}` is not an integer: {value:?}"
                )
            }
            Self::TooFewTasks(ntasks) => write!(
                f,
                "SLURM allocation has {ntasks} task(s); a cellular run needs at least 2 \
                 (one controller task plus one or more cell tasks)"
            ),
            Self::RankOutOfRange { proc_id, ntasks } => write!(
                f,
                "SLURM_PROCID {proc_id} is out of range for SLURM_NTASKS {ntasks}"
            ),
            Self::EmptyNodelist(raw) => write!(
                f,
                "SLURM_JOB_NODELIST {raw:?} expanded to no hosts and no \
                 AIPERF_SLURM_CONTROLLER_HOST override was set"
            ),
        }
    }
}

impl std::error::Error for SlurmTopologyError {}

impl SlurmTopology {
    /// Construct and validate a topology from explicit values (test/programmatic
    /// entry). Fails when the allocation is too small or the rank is out of range.
    pub fn new(
        proc_id: u32,
        ntasks: u32,
        controller_host: impl Into<String>,
    ) -> Result<Self, SlurmTopologyError> {
        if ntasks < 2 {
            return Err(SlurmTopologyError::TooFewTasks(ntasks));
        }
        if proc_id >= ntasks {
            return Err(SlurmTopologyError::RankOutOfRange { proc_id, ntasks });
        }
        Ok(Self {
            proc_id,
            ntasks,
            controller_host: controller_host.into(),
        })
    }

    /// Resolve this task's topology from the `SLURM_*` environment.
    ///
    /// Reads `SLURM_PROCID` and `SLURM_NTASKS`, and resolves the controller host
    /// from [`CONTROLLER_HOST_ENV`] if set, otherwise the first host of the expanded
    /// `SLURM_JOB_NODELIST`.
    pub fn from_env() -> Result<Self, SlurmTopologyError> {
        let proc_id = parse_env_u32("SLURM_PROCID")?;
        let ntasks = parse_env_u32("SLURM_NTASKS")?;
        let controller_host = resolve_controller_host_from_env()?;
        Self::new(proc_id, ntasks, controller_host)
    }

    /// Whether a `SLURM_*` allocation is present at all (used to auto-detect the
    /// SLURM launcher without forcing the user to pass a flag).
    pub fn is_slurm_allocation() -> bool {
        std::env::var_os("SLURM_JOB_ID").is_some() || std::env::var_os("SLURM_PROCID").is_some()
    }

    /// This task's global rank (`SLURM_PROCID`).
    pub fn proc_id(&self) -> u32 {
        self.proc_id
    }

    /// The allocation's total task count (`SLURM_NTASKS`).
    pub fn ntasks(&self) -> u32 {
        self.ntasks
    }

    /// The resolved rank-0 (controller) node hostname.
    pub fn controller_host(&self) -> &str {
        &self.controller_host
    }

    /// Whether this task is the cellular controller (rank 0).
    pub fn is_controller(&self) -> bool {
        self.proc_id == 0
    }

    /// The number of cells the run is partitioned across: every task except the
    /// controller. Always `>= 1` because [`Self::new`] rejects `ntasks < 2`.
    pub fn cell_count(&self) -> u32 {
        self.ntasks - 1
    }

    /// This task's zero-based `cell_id`, or `None` for the controller task. Cell
    /// ranks `1..ntasks` map to ids `0..cell_count` so the id space is dense and
    /// tiles exactly, matching [`ModuloCellPartition`](crate::cellular::partition::ModuloCellPartition).
    pub fn cell_id(&self) -> Option<u32> {
        (self.proc_id >= 1).then(|| self.proc_id - 1)
    }

    /// The `tcp://HOST:PORT` coordinate every cell dials and the controller
    /// advertises, from the rank-0 host and the given velo bootstrap port.
    pub fn controller_coordinate(&self, port: u16) -> String {
        format!("tcp://{}:{}", self.controller_host, port)
    }
}

/// The velo bootstrap port from [`CONTROLLER_PORT_ENV`], defaulting to
/// [`DEFAULT_CONTROLLER_PORT`].
pub fn controller_port_from_env() -> u16 {
    std::env::var(CONTROLLER_PORT_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_CONTROLLER_PORT)
}

fn parse_env_u32(var: &'static str) -> Result<u32, SlurmTopologyError> {
    let raw = std::env::var(var).map_err(|_| SlurmTopologyError::MissingEnv(var))?;
    raw.trim()
        .parse()
        .map_err(|_| SlurmTopologyError::InvalidEnv { var, value: raw })
}

fn resolve_controller_host_from_env() -> Result<String, SlurmTopologyError> {
    if let Ok(host) = std::env::var(CONTROLLER_HOST_ENV)
        && !host.trim().is_empty()
    {
        return Ok(host.trim().to_owned());
    }
    let raw = std::env::var("SLURM_JOB_NODELIST")
        .map_err(|_| SlurmTopologyError::MissingEnv("SLURM_JOB_NODELIST"))?;
    let first = expand_nodelist(&raw).into_iter().next();
    first.ok_or(SlurmTopologyError::EmptyNodelist(raw))
}

/// Expand a SLURM `SLURM_JOB_NODELIST` hostlist into concrete hostnames.
///
/// Handles the compressed range/list syntax SLURM emits, e.g.
/// `node01` -> `[node01]`, `node[01-04]` -> `node01..node04`,
/// `node[01-02,05]` -> `node01, node02, node05`, and top-level comma lists such as
/// `node01,gpu[1-2]`. Numeric ranges preserve the zero-padded width of their lower
/// bound (`01-04` -> `01,02,03,04`). Bracketless names pass through verbatim.
///
/// Only the FIRST expanded host is load-bearing for topology (the controller node),
/// but the full expansion keeps the parser honest and unit-testable.
pub fn expand_nodelist(nodelist: &str) -> Vec<String> {
    let mut hosts = Vec::new();
    for group in split_top_level_commas(nodelist.trim()) {
        expand_group(group.trim(), &mut hosts);
    }
    hosts
}

/// Split on commas that are NOT inside a `[...]` bracket group, so a bracketed list
/// like `node[01-04,07]` stays one group while a top-level `a,b` splits into two.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start <= s.len() {
        parts.push(&s[start..]);
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Expand one comma-free group (a bare host or a single `prefix[ranges]suffix`)
/// into `out`.
fn expand_group(group: &str, out: &mut Vec<String>) {
    let Some(open) = group.find('[') else {
        if !group.is_empty() {
            out.push(group.to_owned());
        }
        return;
    };
    let Some(close_rel) = group[open..].find(']') else {
        // Malformed (unclosed bracket): pass through verbatim rather than drop it.
        out.push(group.to_owned());
        return;
    };
    let close = open + close_rel;
    let prefix = &group[..open];
    let inner = &group[open + 1..close];
    let suffix = &group[close + 1..];
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                let width = lo.len();
                match (lo.parse::<u64>(), hi.parse::<u64>()) {
                    (Ok(lo), Ok(hi)) if lo <= hi => {
                        for n in lo..=hi {
                            out.push(format!("{prefix}{n:0width$}{suffix}"));
                        }
                    }
                    // Non-numeric or inverted range: emit the literal token so a
                    // malformed nodelist never silently yields zero hosts.
                    _ => out.push(format!("{prefix}{part}{suffix}")),
                }
            }
            None => out.push(format!("{prefix}{part}{suffix}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_single_bare_host() {
        assert_eq!(expand_nodelist("node01"), vec!["node01"]);
    }

    #[test]
    fn expand_padded_range() {
        assert_eq!(
            expand_nodelist("node[01-04]"),
            vec!["node01", "node02", "node03", "node04"]
        );
    }

    #[test]
    fn expand_range_preserves_lower_bound_width() {
        // Width comes from the lower bound; crossing a decade keeps 3 digits.
        assert_eq!(
            expand_nodelist("nid[008-011]"),
            vec!["nid008", "nid009", "nid010", "nid011"]
        );
    }

    #[test]
    fn expand_mixed_list_and_range_in_brackets() {
        assert_eq!(
            expand_nodelist("node[01-02,05]"),
            vec!["node01", "node02", "node05"]
        );
    }

    #[test]
    fn expand_top_level_comma_list_does_not_split_inside_brackets() {
        assert_eq!(
            expand_nodelist("node[01-02],gpu[3-4]"),
            vec!["node01", "node02", "gpu3", "gpu4"]
        );
    }

    #[test]
    fn expand_top_level_bare_comma_list() {
        assert_eq!(expand_nodelist("c1,c2,c3"), vec!["c1", "c2", "c3"]);
    }

    #[test]
    fn expand_bracket_with_suffix() {
        assert_eq!(expand_nodelist("dgx[1-2]-ib"), vec!["dgx1-ib", "dgx2-ib"]);
    }

    #[test]
    fn expand_first_host_is_controller_node() {
        // The load-bearing property: first expanded host is rank-0's node.
        assert_eq!(
            expand_nodelist("compute[07-10],login01")
                .into_iter()
                .next()
                .unwrap(),
            "compute07"
        );
    }

    #[test]
    fn topology_controller_is_rank_zero() {
        let topo = SlurmTopology::new(0, 4, "node01").unwrap();
        assert!(topo.is_controller());
        assert_eq!(topo.cell_count(), 3);
        assert_eq!(topo.cell_id(), None);
        assert_eq!(topo.controller_coordinate(9500), "tcp://node01:9500");
    }

    #[test]
    fn topology_cells_map_rank_to_dense_cell_id() {
        // ranks 1,2,3 -> cell ids 0,1,2 over a 4-task allocation (3 cells).
        for (proc_id, expected_id) in [(1u32, 0u32), (2, 1), (3, 2)] {
            let topo = SlurmTopology::new(proc_id, 4, "node01").unwrap();
            assert!(!topo.is_controller());
            assert_eq!(topo.cell_id(), Some(expected_id));
            assert_eq!(topo.cell_count(), 3);
            // Every cell dials the identical rank-0 coordinate.
            assert_eq!(topo.controller_coordinate(9500), "tcp://node01:9500");
        }
    }

    #[test]
    fn topology_two_task_allocation_is_one_cell() {
        let controller = SlurmTopology::new(0, 2, "n0").unwrap();
        let cell = SlurmTopology::new(1, 2, "n0").unwrap();
        assert_eq!(controller.cell_count(), 1);
        assert_eq!(cell.cell_id(), Some(0));
    }

    #[test]
    fn topology_rejects_single_task_allocation() {
        assert_eq!(
            SlurmTopology::new(0, 1, "n0"),
            Err(SlurmTopologyError::TooFewTasks(1))
        );
    }

    #[test]
    fn topology_rejects_rank_out_of_range() {
        assert_eq!(
            SlurmTopology::new(4, 4, "n0"),
            Err(SlurmTopologyError::RankOutOfRange {
                proc_id: 4,
                ntasks: 4
            })
        );
    }

    #[test]
    fn cell_ids_tile_the_cell_count_exactly() {
        // The union of every cell task's id is exactly 0..cell_count, disjoint and
        // complete — the invariant ModuloCellPartition depends on.
        let ntasks = 6;
        let mut ids: Vec<u32> = (1..ntasks)
            .map(|rank| {
                SlurmTopology::new(rank, ntasks, "n0")
                    .unwrap()
                    .cell_id()
                    .unwrap()
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..ntasks - 1).collect::<Vec<_>>());
    }
}
