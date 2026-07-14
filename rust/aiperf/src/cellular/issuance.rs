// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S1 — the issuance authority: the single central dispatch-ordinal assignment.
//!
//! Every issued turn receives a **dense global dispatch ordinal** — the absolute
//! record slot the records-first re-ingest orders by, and the basis of the
//! codebase's worker-count-independent byte parity (roadmap
//! `specs/2026-07-12-cellular-ready-seams-and-roadmap.md`, S1). An
//! [`IssuanceAuthority`] maps a cell's monotonic local dispatch index (`0, 1, 2,
//! …` in issue order) to that global ordinal.
//!
//! This is a **single central assignment**, never a shared atomic: a shared-atomic
//! self-issue interleaves nondeterministically and breaks run-to-run float
//! reproducibility. Today the [`DirectIssuanceAuthority`] (identity) ships for the
//! single-process cell of one. The [`CellularAutonomousIssuer`] — the deferred
//! per-cell issuer with zero coordinator hop — is defined and tested here so the
//! seam's Phase-2 path is proven; the cellular controller injects it per cell.

use crate::cellular::partition::{CellPartition, ModuloCellPartition};

/// Assigns the dense global dispatch ordinal for each issued turn.
///
/// Object-safe so the runner holds `Rc<dyn IssuanceAuthority>`; the autonomous
/// issuer drops in behind this trait without touching the dispatch path. The
/// contract: over every cell, `global_ordinal` applied to each cell's local issue
/// sequence must produce the dense `0..total` ordinal space with no gap or
/// collision, so the merged report re-ingests cleanly in ordinal order.
pub trait IssuanceAuthority {
    /// Map a cell-local dispatch index (`0, 1, 2, …` in issue order) to the dense
    /// global dispatch ordinal used as the record's absolute slot.
    fn global_ordinal(&self, local_dispatch_index: usize) -> usize;

    /// The cell partition this issuer serves (identity for the cell of one).
    fn partition(&self) -> &dyn CellPartition;
}

/// Tier-0 "Direct" issuer: identity ordinal for the single-process cell of one.
///
/// `global_ordinal(local) == local`, reproducing today's sequential dispatch
/// ordinal exactly — the shipping default, and the reason wiring it through the
/// dispatch path changes no output.
#[derive(Debug, Clone, Default)]
pub struct DirectIssuanceAuthority {
    partition: ModuloCellPartition,
}

impl DirectIssuanceAuthority {
    /// Builds the identity `(0, 1)` issuer.
    pub fn new() -> Self {
        Self {
            partition: ModuloCellPartition::direct(),
        }
    }
}

impl IssuanceAuthority for DirectIssuanceAuthority {
    fn global_ordinal(&self, local_dispatch_index: usize) -> usize {
        local_dispatch_index
    }

    fn partition(&self) -> &dyn CellPartition {
        &self.partition
    }
}

/// Tier-2 "Cellular Autonomous" issuer: a cell assigns global ordinals from its
/// round-robin partition with zero coordinator hop.
///
/// A cell's `j`-th issued turn is global ordinal `j * cell_count + cell_id`. For a
/// [`ModuloCellPartition`] — where cell `k` issues its owned instances in ascending
/// order — this is exactly the trace instance index, so the union across cells is
/// the dense `0..total` ordinal space. Under deterministic (sequential) sampling,
/// where dispatch order equals instance order, a merged multi-cell report is
/// byte-identical to the same run executed as one cell.
#[derive(Debug, Clone)]
pub struct CellularAutonomousIssuer {
    partition: ModuloCellPartition,
}

impl CellularAutonomousIssuer {
    /// Builds an autonomous issuer for a cell's round-robin partition.
    pub fn new(partition: ModuloCellPartition) -> Self {
        Self { partition }
    }
}

impl IssuanceAuthority for CellularAutonomousIssuer {
    fn global_ordinal(&self, local_dispatch_index: usize) -> usize {
        local_dispatch_index * self.partition.cell_count() as usize
            + self.partition.cell_id() as usize
    }

    fn partition(&self) -> &dyn CellPartition {
        &self.partition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_issuer_is_identity_over_the_cell_of_one() {
        let issuer = DirectIssuanceAuthority::new();
        assert_eq!(issuer.partition().cell_id(), 0);
        assert_eq!(issuer.partition().cell_count(), 1);
        for local in 0..1000 {
            assert_eq!(issuer.global_ordinal(local), local);
        }
    }

    #[test]
    fn direct_matches_a_cellular_issuer_over_one_cell() {
        let direct = DirectIssuanceAuthority::new();
        let cellular = CellularAutonomousIssuer::new(ModuloCellPartition::direct());
        for local in 0..500 {
            assert_eq!(direct.global_ordinal(local), cellular.global_ordinal(local));
        }
    }

    #[test]
    fn cellular_issuers_tile_the_dense_global_ordinal_space() {
        // For any cell_count, the union over cells of each cell's local dispatch
        // sequence mapped through global_ordinal is exactly 0..total, with no gap
        // and no collision — the invariant the records-first merge relies on.
        for cell_count in 1..=8u32 {
            let issuers: Vec<_> = (0..cell_count)
                .map(|id| {
                    CellularAutonomousIssuer::new(ModuloCellPartition::new(id, cell_count).unwrap())
                })
                .collect();
            let locals_per_cell = 250usize;
            let mut global: Vec<usize> = Vec::new();
            for issuer in &issuers {
                for local in 0..locals_per_cell {
                    global.push(issuer.global_ordinal(local));
                }
            }
            global.sort_unstable();
            let total = cell_count as usize * locals_per_cell;
            assert_eq!(global.len(), total);
            for (expected, actual) in global.iter().copied().enumerate() {
                assert_eq!(
                    actual, expected,
                    "cell_count {cell_count} left a gap/collision"
                );
            }
        }
    }

    #[test]
    fn cellular_global_ordinal_is_the_instance_index_for_round_robin() {
        // Cell k's j-th owned instance (ascending) is j*count + k; the issuer's
        // global ordinal for local index j must equal that instance index so a
        // merged report lands each record at its single-cell slot.
        let cell_count = 4u32;
        for cell_id in 0..cell_count {
            let partition = ModuloCellPartition::new(cell_id, cell_count).unwrap();
            let issuer = CellularAutonomousIssuer::new(partition);
            let owned: Vec<u64> = (0..10_000u64).filter(|&i| partition.owns(i)).collect();
            for (local, &instance) in owned.iter().enumerate() {
                assert_eq!(issuer.global_ordinal(local) as u64, instance);
            }
        }
    }
}
