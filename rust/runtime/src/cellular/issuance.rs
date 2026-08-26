// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dense global dispatch-ordinal assignment.
//!
//! Every issued turn receives a **dense dispatch ordinal** — the record slot the
//! records-first re-ingest orders by, and the basis of worker-count-independent
//! byte parity. An
//! [`IssuanceAuthority`] maps a turn's dispatch indices to that ordinal.
//!
//! The runner rebuilds the per-cell sampler fresh at each phase boundary (the
//! dataset RNG is re-seeded per phase), so a cell draws its owned instances of
//! *each phase* from position 0. The cellular ordinal is therefore its phase's
//! global base plus its phase-local slot — cell `k`'s `m`-th turn of a phase whose
//! prior phases dispatched `base` turns is `base + m*count + k` — which equals the
//! absolute slot a single-cell run assigns that instance, so the merged report is
//! byte-identical. The single-process identity issuer keeps the cumulative flat slot.
//!
//! This is a single central assignment, never a shared atomic: a shared-atomic
//! self-issue interleaves nondeterministically and breaks run-to-run float
//! reproducibility.

use crate::cellular::partition::{CellPartition, ModuloCellPartition};

/// Assigns the dense global dispatch ordinal for each issued turn.
///
/// Object-safe so the runner holds `Rc<dyn IssuanceAuthority>`; the autonomous
/// issuer drops in behind this trait without touching the dispatch path. The
/// contract: over every cell, `global_ordinal` must produce the dense `0..total`
/// absolute-slot space with no gap or collision, so the merged report re-ingests
/// cleanly in ordinal order.
pub trait IssuanceAuthority {
    /// Map a turn's dispatch indices to its record's dense absolute dispatch slot.
    ///
    /// `flat_local` is the cell's cumulative dispatch index across all phases (`0, 1,
    /// 2, …` in issue order); `phase_ordinal_base` is the number of turns the run's
    /// prior phases dispatched globally (0 for the first phase); `within_phase_local`
    /// is the index within the current phase, reset at each phase boundary. The
    /// identity issuer uses `flat_local`; the cellular issuer uses
    /// `phase_ordinal_base + within_phase_local` because its sampler restarts each
    /// phase (see the module docs).
    fn global_ordinal(
        &self,
        flat_local: usize,
        phase_ordinal_base: usize,
        within_phase_local: usize,
    ) -> usize;

    /// The cell partition this issuer serves (identity for the cell of one).
    fn partition(&self) -> &dyn CellPartition;
}

/// Identity issuer for a single-process cell of one.
///
/// `global_ordinal == flat_local` preserves cumulative sequential dispatch order.
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
    fn global_ordinal(
        &self,
        flat_local: usize,
        _phase_ordinal_base: usize,
        _within_phase_local: usize,
    ) -> usize {
        flat_local
    }

    fn partition(&self) -> &dyn CellPartition {
        &self.partition
    }
}

/// Assigns global ordinals from a cell's round-robin partition without a
/// coordinator hop.
///
/// A cell's `m`-th turn *of a phase* is the absolute slot `phase_base + m *
/// cell_count + cell_id`, where `phase_base` is the turns dispatched by the run's
/// prior phases. For a [`ModuloCellPartition`] — where cell `k` draws its owned
/// instances of the phase in ascending order (the sampler restarts each phase) —
/// `m*cell_count + cell_id` is exactly the phase-local instance index, so within each
/// phase the union across cells is the dense `[phase_base, phase_base+phase_total)`
/// slot range and the whole run tiles `0..total`. A `cell_count == 1` partition
/// instead keeps the cumulative flat index, because the strided form assumes each
/// prior phase emitted exactly its reserved span. Under deterministic (sequential)
/// sampling, where per-phase dispatch order equals per-phase instance order, a merged
/// multi-cell report is byte-identical to the same run executed as one cell.
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
    fn global_ordinal(
        &self,
        flat_local: usize,
        phase_ordinal_base: usize,
        within_phase_local: usize,
    ) -> usize {
        // Single cell: the run's own dispatch is already globally dense, so the flat
        // cumulative index IS the absolute slot. The strided `phase_base +
        // within_phase_local` form below assumes each prior phase emitted EXACTLY its
        // reserved `phase_ordinal_base` span — true for count/duration phases, but the
        // accelerated cache-warmup phase emits a runtime-determined number of pressure
        // records far exceeding its static prime reservation, which would overflow the
        // next phase's base and collide. The flat index is byte-identical to the strided
        // form whenever a prior phase emits exactly its reservation, so this only
        // changes (and fixes) the over-emitting case. Multi-cell tiling still needs the
        // strided form (each cell owns a disjoint round-robin residue class).
        if self.partition.cell_count() == 1 {
            return flat_local;
        }
        phase_ordinal_base
            + within_phase_local * self.partition.cell_count() as usize
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
    fn direct_issuer_is_the_cumulative_slot_over_the_cell_of_one() {
        let issuer = DirectIssuanceAuthority::new();
        assert_eq!(issuer.partition().cell_id(), 0);
        assert_eq!(issuer.partition().cell_count(), 1);
        // Direct uses the cumulative flat index and ignores the base + phase-local one.
        for flat in 0..1000 {
            assert_eq!(issuer.global_ordinal(flat, 7, 999 - flat), flat);
        }
    }

    #[test]
    fn direct_matches_a_cellular_issuer_over_one_cell_single_phase() {
        // Over one cell and a single (base-0) phase, flat == within, so the identity
        // issuer and the cellular issuer agree — the property that keeps the cell of
        // one byte-unchanged.
        let direct = DirectIssuanceAuthority::new();
        let cellular = CellularAutonomousIssuer::new(ModuloCellPartition::direct());
        for i in 0..500 {
            assert_eq!(
                direct.global_ordinal(i, 0, i),
                cellular.global_ordinal(i, 0, i)
            );
        }
    }

    #[test]
    fn cellular_issuers_tile_the_dense_ordinal_space_across_phase_bases() {
        // Two phases stacked by their global bases (warmup [0, W), profiling [W,
        // W+P)): within each phase the union over cells of the phase-local turns tiles
        // [base, base+phase_total), so the whole run tiles 0..total with no gap or
        // collision — the invariant the cumulative merge relies on. (flat is ignored.)
        for cell_count in 1..=8u32 {
            let issuers: Vec<_> = (0..cell_count)
                .map(|id| {
                    CellularAutonomousIssuer::new(ModuloCellPartition::new(id, cell_count).unwrap())
                })
                .collect();
            let warmup_per_cell = 7usize;
            let profiling_per_cell = 250usize;
            let warmup_total = cell_count as usize * warmup_per_cell;
            let mut ordinals: Vec<usize> = Vec::new();
            // `flat` is the run's cumulative dispatch index; the multi-cell strided form
            // ignores it, the single-cell form uses it. Feed the true running index so
            // the single-cell (cell_count == 1) case tiles densely off the flat ordinal.
            let mut flat = 0usize;
            for issuer in &issuers {
                for within in 0..warmup_per_cell {
                    ordinals.push(issuer.global_ordinal(flat, 0, within));
                    flat += 1;
                }
                for within in 0..profiling_per_cell {
                    ordinals.push(issuer.global_ordinal(flat, warmup_total, within));
                    flat += 1;
                }
            }
            ordinals.sort_unstable();
            let total = warmup_total + cell_count as usize * profiling_per_cell;
            assert_eq!(ordinals.len(), total);
            for (expected, actual) in ordinals.iter().copied().enumerate() {
                assert_eq!(
                    actual, expected,
                    "cell_count {cell_count} left a gap/collision"
                );
            }
        }
    }

    #[test]
    fn single_cell_uses_flat_ordinal_so_an_over_emitting_warmup_never_collides() {
        // Regression for the accelerated cache-warmup path: the WARMUP phase reserves a
        // small static prime span (here `warmup_reservation = 8`) but actually emits far
        // more pressure records (`warmup_actual = 722`). The strided `phase_base +
        // within` form would place profiling records at `8 + within`, colliding with the
        // warmup records already at `8..722`. The single-cell issuer instead uses the
        // flat cumulative dispatch index, so every record lands at a unique dense slot.
        let issuer = CellularAutonomousIssuer::new(ModuloCellPartition::direct());
        let warmup_reservation = 8usize;
        let warmup_actual = 722usize;
        let profiling = 214usize;
        let mut slots = Vec::new();
        let mut flat = 0usize;
        for within in 0..warmup_actual {
            slots.push(issuer.global_ordinal(flat, 0, within));
            flat += 1;
        }
        for within in 0..profiling {
            slots.push(issuer.global_ordinal(flat, warmup_reservation, within));
            flat += 1;
        }
        // Dense, unique, no collision: exactly `0..warmup_actual + profiling`.
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..warmup_actual + profiling).collect::<Vec<_>>());
    }

    #[test]
    fn cellular_ordinal_is_the_phase_base_plus_instance_index_for_round_robin() {
        // Cell k's j-th owned instance of a phase (ascending) is j*count + k; the
        // issuer's ordinal must equal base + that index so the merged report lands
        // each record at its single-cell absolute slot.
        let cell_count = 4u32;
        let base = 41usize;
        for cell_id in 0..cell_count {
            let partition = ModuloCellPartition::new(cell_id, cell_count).unwrap();
            let issuer = CellularAutonomousIssuer::new(partition);
            let owned: Vec<u64> = (0..10_000u64).filter(|&i| partition.owns(i)).collect();
            for (within, &instance) in owned.iter().enumerate() {
                assert_eq!(
                    issuer.global_ordinal(0, base, within) as u64,
                    base as u64 + instance
                );
            }
        }
    }
}
