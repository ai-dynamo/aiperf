// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic `(cell_id, cell_count)` work partition.
//!
//! A [`CellPartition`] is the seam that lets `cell_count` cells produce the same
//! trace *set* as a single cell with different *ownership*: every cell selects the
//! trace instances it owns from the one seed space, so identical
//! `(workload_seed, cell_count, partition)` inputs yield byte-stable artifacts.

use std::fmt::{self, Display, Formatter};

use crate::rng::RngRoot;

/// A cell's static slice of the deterministic trace-instance space.
///
/// Implementors MUST partition the `u64` instance-index space into `cell_count`
/// disjoint classes whose union is complete: for every `i`, exactly one
/// `cell_id in 0..cell_count` has `owns(i) == true`. Violating this silently drops
/// or double-issues trace instances, so the concrete impls validate their
/// parameters at construction.
pub trait CellPartition {
    /// This cell's zero-based identifier, in `0..cell_count`.
    fn cell_id(&self) -> u32;

    /// Total number of cells the workload budget is partitioned across (`>= 1`).
    fn cell_count(&self) -> u32;

    /// Whether this cell owns — and therefore issues — the trace instance at
    /// `instance_index` (the instance's canonical index in the sampled space).
    fn owns(&self, instance_index: u64) -> bool;

    /// Derive this cell's private child root for the named per-cell stream.
    ///
    /// Ownership-independent by construction: the derived seed depends only on
    /// `(base, identifier, cell_id)`, never on execution order or thread, so a
    /// cell's selection/derivation stream is reproducible regardless of how many
    /// other cells exist or which thread runs it. Content identity is untouched —
    /// only per-cell selection streams key off the cell.
    fn derive_cell_root(&self, base: RngRoot, identifier: &str) -> RngRoot;
}

/// Round-robin partition: cell `k` owns every instance index `i` with
/// `i % cell_count == cell_id`.
///
/// The identity case `(0, 1)` ([`direct`](Self::direct)) owns the whole instance
/// space. Round-robin keeps each cell's owned indices ascending, so
/// a cell's `n`-th issued instance is exactly `n * cell_count + cell_id` — the
/// mapping the issuance authority uses to reconstruct a dense global dispatch
/// ordinal from a cell-local counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuloCellPartition {
    cell_id: u32,
    cell_count: u32,
}

/// Env var carrying this process's zero-based cell id (set by the controller).
pub const CELL_ID_ENV: &str = "AIPERF_CELL_ID";
/// Env var carrying the total cell count (set by the controller).
pub const CELL_COUNT_ENV: &str = "AIPERF_CELL_COUNT";

impl ModuloCellPartition {
    /// The identity partition: one cell `(0, 1)` owning the entire instance space.
    ///
    /// `const` so it can seed the default single-process runtime with no fallible
    /// construction on the hot bootstrap path.
    pub const fn direct() -> Self {
        Self {
            cell_id: 0,
            cell_count: 1,
        }
    }

    /// Reads this process's cell partition from [`CELL_ID_ENV`] / [`CELL_COUNT_ENV`],
    /// or `None` when the process is not a cell. This is how the ordinary execute
    /// path selects the autonomous issuer and the per-cell sampler without a new
    /// wire field; absent the vars the single-process path is byte-unchanged.
    pub fn from_env() -> Option<Self> {
        let cell_id = std::env::var(CELL_ID_ENV).ok()?.parse().ok()?;
        let cell_count = std::env::var(CELL_COUNT_ENV).ok()?.parse().ok()?;
        Self::new(cell_id, cell_count).ok()
    }

    /// Construct a validated round-robin partition.
    ///
    /// Fails when `cell_count == 0` (no owner for any index) or
    /// `cell_id >= cell_count` (this cell would own nothing while some index is
    /// double-counted), so a live partition can never silently drop or double-own
    /// trace instances.
    pub fn new(cell_id: u32, cell_count: u32) -> Result<Self, CellPartitionError> {
        if cell_count == 0 {
            return Err(CellPartitionError::ZeroCells);
        }
        if cell_id >= cell_count {
            return Err(CellPartitionError::IdOutOfRange {
                cell_id,
                cell_count,
            });
        }
        Ok(Self {
            cell_id,
            cell_count,
        })
    }
}

impl Default for ModuloCellPartition {
    /// The identity `(0, 1)` partition — the single-process cell of one.
    fn default() -> Self {
        Self::direct()
    }
}

impl CellPartition for ModuloCellPartition {
    fn cell_id(&self) -> u32 {
        self.cell_id
    }

    fn cell_count(&self) -> u32 {
        self.cell_count
    }

    fn owns(&self, instance_index: u64) -> bool {
        // `cell_count` is validated `>= 1` at construction, so the modulo is safe.
        instance_index % self.cell_count as u64 == self.cell_id as u64
    }

    fn derive_cell_root(&self, base: RngRoot, identifier: &str) -> RngRoot {
        // `derive_indexed_root` already frames indexed splits as "adding another
        // worker cannot perturb any existing worker's sequence" — exactly the
        // per-cell primitive. Seedless roots stay seedless.
        base.derive_indexed_root(identifier, self.cell_id as u64)
    }
}

/// Error constructing a [`ModuloCellPartition`].
///
/// A plain enum with a hand-written [`Display`] per the crate's error convention
/// (no `thiserror` in library crates).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellPartitionError {
    /// `cell_count` was zero — no cell could own any instance.
    ZeroCells,
    /// `cell_id` was not strictly less than `cell_count`.
    IdOutOfRange {
        /// The out-of-range cell identifier.
        cell_id: u32,
        /// The cell count it was checked against.
        cell_count: u32,
    },
}

impl Display for CellPartitionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCells => write!(f, "cell_count must be at least 1, got 0"),
            Self::IdOutOfRange {
                cell_id,
                cell_count,
            } => write!(
                f,
                "cell_id {cell_id} is out of range for cell_count {cell_count} (must be < cell_count)"
            ),
        }
    }
}

impl std::error::Error for CellPartitionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_partition_owns_every_instance() {
        let partition = ModuloCellPartition::direct();
        assert_eq!(partition.cell_id(), 0);
        assert_eq!(partition.cell_count(), 1);
        for i in 0..1000 {
            assert!(
                partition.owns(i),
                "identity partition must own instance {i}"
            );
        }
    }

    #[test]
    fn new_rejects_zero_cells_and_out_of_range_id() {
        assert_eq!(
            ModuloCellPartition::new(0, 0),
            Err(CellPartitionError::ZeroCells)
        );
        assert_eq!(
            ModuloCellPartition::new(3, 3),
            Err(CellPartitionError::IdOutOfRange {
                cell_id: 3,
                cell_count: 3,
            })
        );
        assert!(ModuloCellPartition::new(2, 3).is_ok());
    }

    #[test]
    fn ownership_is_disjoint_and_complete_across_cells() {
        // For any cell_count, every instance index is owned by exactly one cell.
        for cell_count in 1..=8u32 {
            let cells: Vec<_> = (0..cell_count)
                .map(|id| ModuloCellPartition::new(id, cell_count).unwrap())
                .collect();
            for instance in 0..2000u64 {
                let owners = cells.iter().filter(|c| c.owns(instance)).count();
                assert_eq!(
                    owners, 1,
                    "instance {instance} must be owned by exactly one of {cell_count} cells"
                );
            }
        }
    }

    #[test]
    fn round_robin_owned_indices_reconstruct_instance_index() {
        // A cell's n-th owned instance (ascending) is n*cell_count + cell_id — the
        // mapping the issuance authority relies on for the dense global ordinal.
        let cell_count = 4u32;
        for cell_id in 0..cell_count {
            let cell = ModuloCellPartition::new(cell_id, cell_count).unwrap();
            let owned: Vec<u64> = (0..10_000u64).filter(|&i| cell.owns(i)).collect();
            for (n, &instance) in owned.iter().enumerate() {
                assert_eq!(instance, n as u64 * cell_count as u64 + cell_id as u64);
            }
        }
    }

    #[test]
    fn derive_cell_root_is_deterministic_distinct_and_ownership_independent() {
        let base = RngRoot::new(Some(42));
        let cell_count = 4u32;
        let roots: Vec<RngRoot> = (0..cell_count)
            .map(|id| {
                ModuloCellPartition::new(id, cell_count)
                    .unwrap()
                    .derive_cell_root(base, "runner.arrival")
            })
            .collect();

        // Deterministic: same inputs → same derived root.
        let again = ModuloCellPartition::new(1, cell_count)
            .unwrap()
            .derive_cell_root(base, "runner.arrival");
        assert_eq!(roots[1], again);

        // Distinct per cell: no two cells share a selection stream.
        for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                assert_ne!(roots[i], roots[j], "cells {i} and {j} share a root");
            }
        }

        // Seedless roots stay seedless (entropy semantics preserved).
        assert_eq!(
            ModuloCellPartition::new(2, cell_count)
                .unwrap()
                .derive_cell_root(RngRoot::new(None), "runner.arrival"),
            RngRoot::new(None)
        );
    }
}
