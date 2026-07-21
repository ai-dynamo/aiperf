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
//!   rank-0 node hostname (the first host of the highest-precedence non-empty
//!   nodelist — `SLURM_STEP_NODELIST`, then `SLURM_NODELIST`, then
//!   `SLURM_JOB_NODELIST`) plus the velo bootstrap port, so discovery is
//!   zero-round-trip and matches on every task.
//!
//! The rank-0 node is taken to be the first node of the expanded nodelist. This is
//! SLURM's default block task distribution (task 0 lands on the first node of the
//! allocation); an explicit `AIPERF_SLURM_CONTROLLER_HOST` override is honored first
//! for exotic distributions or user-pinned placement. The step-scoped lists take
//! precedence over the job-wide list so a nested `srun` step (e.g. an orchestrator
//! launching aiperf against a node subset) resolves rank 0 within the step's nodes,
//! not the job's first host.

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
    /// A nodelist variable was present but expanded to no hosts (and no explicit
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
                "SLURM nodelist {raw:?} expanded to no hosts and no \
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
    /// highest-precedence non-empty nodelist (`SLURM_STEP_NODELIST`, then
    /// `SLURM_NODELIST`, then `SLURM_JOB_NODELIST`).
    pub fn from_env() -> Result<Self, SlurmTopologyError> {
        let proc_id = parse_env_u32("SLURM_PROCID")?;
        let ntasks = parse_env_u32("SLURM_NTASKS")?;
        let controller_host = resolve_controller_host_from_env()?;
        Self::new(proc_id, ntasks, controller_host)
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

/// Nodelist environment variables consulted, in order, to derive the rank-0
/// (controller) host — first non-empty wins.
///
/// An orchestrator (e.g. srt-slurm) that launches aiperf as a nested `srun` **step**
/// scoped to a subset of the job's nodes exposes that step's nodes in
/// `SLURM_STEP_NODELIST` (and its `SLURM_NODELIST` alias), while `SLURM_JOB_NODELIST`
/// stays the job-wide list whose first host is not necessarily where this step's rank
/// 0 landed. The narrower step lists therefore take precedence over the job-wide list;
/// a plain `srun`/`sbatch` allocation (no nested step) sets all three to the same
/// value, so the order is a no-op there.
const NODELIST_ENV_PRECEDENCE: [&str; 3] = [
    "SLURM_STEP_NODELIST",
    "SLURM_NODELIST",
    "SLURM_JOB_NODELIST",
];

/// The `&'static str` reported by [`SlurmTopologyError::MissingEnv`] when none of the
/// [`NODELIST_ENV_PRECEDENCE`] variables (nor the [`CONTROLLER_HOST_ENV`] override) is
/// present.
const NODELIST_ENV_LABEL: &str = "SLURM_STEP_NODELIST/SLURM_NODELIST/SLURM_JOB_NODELIST";

fn resolve_controller_host_from_env() -> Result<String, SlurmTopologyError> {
    if let Ok(host) = std::env::var(CONTROLLER_HOST_ENV)
        && !host.trim().is_empty()
    {
        return Ok(host.trim().to_owned());
    }
    // First non-empty nodelist in precedence order supplies the controller host (its
    // first expanded entry). A present-but-empty value is skipped rather than treated
    // as authoritative, so a narrower step list only wins when it actually names nodes.
    let mut last_nonempty_raw = None;
    for var in NODELIST_ENV_PRECEDENCE {
        let Ok(raw) = std::env::var(var) else {
            continue;
        };
        if raw.trim().is_empty() {
            continue;
        }
        if let Some(first) = expand_nodelist(&raw).into_iter().next() {
            return Ok(first);
        }
        last_nonempty_raw = Some(raw);
    }
    match last_nonempty_raw {
        // Present but expanded to no hosts (a malformed hostlist that yielded nothing).
        Some(raw) => Err(SlurmTopologyError::EmptyNodelist(raw)),
        // None of the nodelist variables were set at all.
        None => Err(SlurmTopologyError::MissingEnv(NODELIST_ENV_LABEL)),
    }
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
    use std::sync::Mutex;

    /// `std::env` is process-global; serialize the env-reading tests so parallel
    /// runners do not observe each other's `SLURM_*` mutations.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// The full set of variables [`resolve_controller_host_from_env`] consults, so a
    /// test starts from a known-clean slate regardless of the ambient (possibly-real
    /// SLURM) environment.
    const HOST_ENV_VARS: [&str; 4] = [
        CONTROLLER_HOST_ENV,
        "SLURM_STEP_NODELIST",
        "SLURM_NODELIST",
        "SLURM_JOB_NODELIST",
    ];

    /// Clear every controller-host variable, set `vars`, run `f`, then clear again —
    /// all under [`ENV_LOCK`] so the mutation window is not visible to other tests.
    fn with_host_env<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: all env mutation is confined to this locked window; no other thread
        // reads or writes these variables while the guard is held.
        unsafe {
            for var in HOST_ENV_VARS {
                std::env::remove_var(var);
            }
            for (key, value) in vars {
                std::env::set_var(key, value);
            }
        }
        let out = f();
        unsafe {
            for var in HOST_ENV_VARS {
                std::env::remove_var(var);
            }
        }
        out
    }

    #[test]
    fn controller_host_override_beats_every_nodelist() {
        let host = with_host_env(
            &[
                (CONTROLLER_HOST_ENV, "pinned-host"),
                ("SLURM_STEP_NODELIST", "step01"),
                ("SLURM_NODELIST", "node01"),
                ("SLURM_JOB_NODELIST", "job01"),
            ],
            resolve_controller_host_from_env,
        );
        assert_eq!(host.unwrap(), "pinned-host");
    }

    #[test]
    fn step_nodelist_wins_over_job_nodelist() {
        // A nested `srun` step scoped to a node subset: rank 0 is within the step's
        // nodes, not the job-wide first host.
        let host = with_host_env(
            &[
                ("SLURM_STEP_NODELIST", "step[07-09]"),
                ("SLURM_JOB_NODELIST", "job[01-16]"),
            ],
            resolve_controller_host_from_env,
        );
        assert_eq!(host.unwrap(), "step07");
    }

    #[test]
    fn node_nodelist_used_when_no_step_nodelist() {
        // `SLURM_NODELIST` (the step alias) outranks the job-wide list.
        let host = with_host_env(
            &[
                ("SLURM_NODELIST", "alias05"),
                ("SLURM_JOB_NODELIST", "job01"),
            ],
            resolve_controller_host_from_env,
        );
        assert_eq!(host.unwrap(), "alias05");
    }

    #[test]
    fn job_nodelist_is_the_final_fallback() {
        let host = with_host_env(
            &[("SLURM_JOB_NODELIST", "compute[03-06]")],
            resolve_controller_host_from_env,
        );
        assert_eq!(host.unwrap(), "compute03");
    }

    #[test]
    fn empty_higher_precedence_nodelist_falls_through() {
        // A present-but-empty step list must not shadow a populated job list.
        let host = with_host_env(
            &[
                ("SLURM_STEP_NODELIST", ""),
                ("SLURM_NODELIST", "  "),
                ("SLURM_JOB_NODELIST", "job42"),
            ],
            resolve_controller_host_from_env,
        );
        assert_eq!(host.unwrap(), "job42");
    }

    #[test]
    fn missing_every_nodelist_is_a_missing_env_error() {
        let err = with_host_env(&[], resolve_controller_host_from_env).unwrap_err();
        assert!(matches!(err, SlurmTopologyError::MissingEnv(_)));
    }

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
        // A 2-task allocation is a valid single-cell cellular run: rank 0 controls,
        // rank 1 is the sole cell (dense id 0), both dialing the same coordinate.
        let controller = SlurmTopology::new(0, 2, "n0").unwrap();
        let cell = SlurmTopology::new(1, 2, "n0").unwrap();
        assert!(controller.is_controller());
        assert_eq!(controller.cell_count(), 1);
        assert_eq!(controller.cell_id(), None);
        assert!(!cell.is_controller());
        assert_eq!(cell.cell_count(), 1);
        assert_eq!(cell.cell_id(), Some(0));
        assert_eq!(
            controller.controller_coordinate(9500),
            cell.controller_coordinate(9500)
        );
        assert_eq!(cell.controller_coordinate(9500), "tcp://n0:9500");
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
