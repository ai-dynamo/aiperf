// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded fair arbitration for evaluator host operations.
//!
//! A provider can recursively create model/tool work while another unit is
//! waiting. This queue preserves FIFO order within a unit and round-robin order
//! across units, while enforcing both global and per-unit memory bounds before
//! route-specific admission. The generic key/value seam also supports future
//! remote evaluator workers without changing scheduling policy.

use std::collections::{BTreeMap, VecDeque};

use anyhow::{Result, anyhow, ensure};

/// Hard queue limits enforced before an evaluator operation is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FairQueueLimits {
    /// Maximum operations buffered across every unit.
    pub global: usize,
    /// Maximum operations buffered for one unit.
    pub per_unit: usize,
}

impl FairQueueLimits {
    /// Validate positive, internally consistent queue limits.
    pub fn new(global: usize, per_unit: usize) -> Result<Self> {
        ensure!(
            global > 0,
            "global evaluator operation queue limit must be positive"
        );
        ensure!(
            per_unit > 0,
            "per-unit evaluator operation queue limit must be positive"
        );
        ensure!(
            per_unit <= global,
            "per-unit evaluator operation queue limit {per_unit} exceeds global limit {global}"
        );
        Ok(Self { global, per_unit })
    }
}

/// Why a ready operation could not be admitted to the bounded fair queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FairQueueRejection {
    /// The global queue bound is already occupied.
    GlobalLimit,
    /// The originating unit already occupies its complete fair share.
    PerUnitLimit,
}

/// Per-unit FIFO, cross-unit round-robin operation arbiter.
///
/// Keys appear at most once in `ready_units`; a successful pop from a
/// non-empty unit rotates that key to the tail. This makes the fairness
/// property independent from lexical key order and provider event batch shape.
#[derive(Debug)]
pub struct FairOperationArbiter<K, T> {
    limits: FairQueueLimits,
    queues: BTreeMap<K, VecDeque<T>>,
    ready_units: VecDeque<K>,
    len: usize,
}

impl<K, T> FairOperationArbiter<K, T>
where
    K: Clone + Ord,
{
    /// Construct an empty arbiter with validated limits.
    pub fn new(limits: FairQueueLimits) -> Self {
        Self {
            limits,
            queues: BTreeMap::new(),
            ready_units: VecDeque::new(),
            len: 0,
        }
    }

    /// Number of queued operations across all units.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no operation is waiting for route admission.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of units with at least one ready operation.
    pub fn active_units(&self) -> usize {
        self.queues.len()
    }

    /// Queue one operation or return a precise bound rejection.
    pub fn push(&mut self, unit: K, operation: T) -> Result<(), FairQueueRejection> {
        self.check_push(&unit)?;
        let queue = self.queues.entry(unit.clone()).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(operation);
        self.len += 1;
        if was_empty {
            self.ready_units.push_back(unit);
        }
        Ok(())
    }

    /// Check queue credits before another subsystem commits correlated state.
    pub fn check_push(&self, unit: &K) -> Result<(), FairQueueRejection> {
        if self.len >= self.limits.global {
            return Err(FairQueueRejection::GlobalLimit);
        }
        if self
            .queues
            .get(unit)
            .is_some_and(|queue| queue.len() >= self.limits.per_unit)
        {
            return Err(FairQueueRejection::PerUnitLimit);
        }
        Ok(())
    }

    /// Pop the next operation under cross-unit round-robin policy.
    pub fn pop(&mut self) -> Option<(K, T)> {
        let unit = self.ready_units.pop_front()?;
        let queue = self
            .queues
            .get_mut(&unit)
            .expect("ready unit must own a non-empty operation queue");
        let operation = queue
            .pop_front()
            .expect("ready unit must own a non-empty operation queue");
        self.len -= 1;
        if queue.is_empty() {
            self.queues.remove(&unit);
        } else {
            self.ready_units.push_back(unit.clone());
        }
        Some((unit, operation))
    }

    /// Cancel one unit and return its queued operations in FIFO order.
    pub fn drain_unit(&mut self, unit: &K) -> Vec<T> {
        let Some(queue) = self.queues.remove(unit) else {
            return Vec::new();
        };
        self.ready_units.retain(|candidate| candidate != unit);
        self.len = self
            .len
            .checked_sub(queue.len())
            .expect("arbiter length must cover every per-unit queue");
        queue.into_iter().collect()
    }

    /// Remove one queued operation from a specific unit.
    ///
    /// This is the provider-requested cancellation path for work that has not
    /// reached a route executor. The remaining FIFO order and cross-unit ready
    /// ring are unchanged.
    pub fn remove_where(&mut self, unit: &K, mut predicate: impl FnMut(&T) -> bool) -> Option<T> {
        let queue = self.queues.get_mut(unit)?;
        let position = queue.iter().position(&mut predicate)?;
        let operation = queue.remove(position)?;
        self.len = self
            .len
            .checked_sub(1)
            .expect("removed evaluator operation must be counted");
        if queue.is_empty() {
            self.queues.remove(unit);
            self.ready_units.retain(|candidate| candidate != unit);
        }
        Some(operation)
    }

    /// Drain every queued operation in the same fair order as admission.
    pub fn drain(&mut self) -> Vec<(K, T)> {
        let mut operations = Vec::with_capacity(self.len);
        while let Some(operation) = self.pop() {
            operations.push(operation);
        }
        operations
    }

    /// Verify internal count and ready-ring invariants.
    ///
    /// The runtime calls this at evaluator lifecycle boundaries; tests call it
    /// after adversarial mutation sequences.
    pub fn validate(&self) -> Result<()> {
        let queued = self.queues.values().map(VecDeque::len).sum::<usize>();
        ensure!(
            queued == self.len,
            "fair evaluator queue counted {queued} operations but stored {}",
            self.len
        );
        ensure!(
            self.len <= self.limits.global,
            "fair evaluator queue exceeded its global bound"
        );
        for (unit, queue) in &self.queues {
            ensure!(
                !queue.is_empty(),
                "fair evaluator queue retained an empty unit"
            );
            ensure!(
                queue.len() <= self.limits.per_unit,
                "fair evaluator queue exceeded one unit's bound"
            );
            let ring_count = self
                .ready_units
                .iter()
                .filter(|candidate| *candidate == unit)
                .count();
            ensure!(
                ring_count == 1,
                "fair evaluator queue unit appears {ring_count} times in ready ring"
            );
        }
        ensure!(
            self.ready_units.len() == self.queues.len(),
            "fair evaluator ready ring and queue map diverged"
        );
        if self.len == 0 && (!self.queues.is_empty() || !self.ready_units.is_empty()) {
            return Err(anyhow!(
                "empty fair evaluator queue retained unit bookkeeping"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_unit_cannot_monopolize_other_ready_units() {
        let limits = FairQueueLimits::new(64, 32).unwrap();
        let mut queue = FairOperationArbiter::new(limits);
        for operation in 0..16 {
            queue.push("recursive", operation).unwrap();
        }
        queue.push("waiting-a", 100).unwrap();
        queue.push("waiting-b", 200).unwrap();

        assert_eq!(queue.pop(), Some(("recursive", 0)));
        assert_eq!(queue.pop(), Some(("waiting-a", 100)));
        assert_eq!(queue.pop(), Some(("waiting-b", 200)));
        assert_eq!(queue.pop(), Some(("recursive", 1)));
        queue.validate().unwrap();
    }

    #[test]
    fn global_and_per_unit_bounds_fail_before_allocation_growth() {
        let mut queue = FairOperationArbiter::new(FairQueueLimits::new(3, 2).unwrap());
        queue.push("a", 1).unwrap();
        queue.push("a", 2).unwrap();
        assert_eq!(
            queue.check_push(&"a"),
            Err(FairQueueRejection::PerUnitLimit)
        );
        assert_eq!(queue.push("a", 3), Err(FairQueueRejection::PerUnitLimit));
        queue.push("b", 4).unwrap();
        assert_eq!(queue.check_push(&"c"), Err(FairQueueRejection::GlobalLimit));
        assert_eq!(queue.push("c", 5), Err(FairQueueRejection::GlobalLimit));
        assert_eq!(queue.len(), 3);
        queue.validate().unwrap();
    }

    #[test]
    fn unit_cancellation_returns_fifo_and_preserves_other_fairness() {
        let mut queue = FairOperationArbiter::new(FairQueueLimits::new(16, 8).unwrap());
        queue.push("a", 1).unwrap();
        queue.push("b", 10).unwrap();
        queue.push("a", 2).unwrap();
        queue.push("c", 20).unwrap();
        assert_eq!(queue.drain_unit(&"a"), vec![1, 2]);
        assert_eq!(queue.pop(), Some(("b", 10)));
        assert_eq!(queue.pop(), Some(("c", 20)));
        assert!(queue.is_empty());
        queue.validate().unwrap();
    }

    #[test]
    fn full_drain_uses_round_robin_order() {
        let mut queue = FairOperationArbiter::new(FairQueueLimits::new(16, 8).unwrap());
        for value in [1, 2, 3] {
            queue.push("a", value).unwrap();
        }
        for value in [10, 11] {
            queue.push("b", value).unwrap();
        }
        assert_eq!(
            queue.drain(),
            vec![("a", 1), ("b", 10), ("a", 2), ("b", 11), ("a", 3)]
        );
        queue.validate().unwrap();
    }

    #[test]
    fn targeted_queued_cancellation_preserves_fifo_and_ready_ring() {
        let mut queue = FairOperationArbiter::new(FairQueueLimits::new(16, 8).unwrap());
        queue.push("a", ("a-1", 1)).unwrap();
        queue.push("b", ("b-1", 10)).unwrap();
        queue.push("a", ("a-2", 2)).unwrap();
        assert_eq!(
            queue.remove_where(&"a", |(id, _)| *id == "a-1"),
            Some(("a-1", 1))
        );
        assert_eq!(queue.pop(), Some(("a", ("a-2", 2))));
        assert_eq!(queue.pop(), Some(("b", ("b-1", 10))));
        queue.validate().unwrap();
    }
}
