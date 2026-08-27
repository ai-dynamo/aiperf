// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AIPerf load-generation runtime.
//!
//! The crate composes endpoint, transport, scheduling, workload, reporting,
//! accuracy, and extension subsystems for the `aiperf` binary.
//! With the `dynosim` Cargo feature, the `dynosim` module composes the
//! same workloads and observers with `SimClock` plus Dynamo's passive mock
//! engine for deterministic, socket-free co-simulation.

#[cfg(test)]
use std::alloc::{GlobalAlloc, Layout};

#[cfg(test)]
struct CountingMiMalloc;

#[cfg(test)]
unsafe impl GlobalAlloc for CountingMiMalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { mimalloc::MiMalloc.alloc(layout) };
        if !pointer.is_null() {
            allocation_probe::record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocation_probe::record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { mimalloc::MiMalloc.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { mimalloc::MiMalloc.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            allocation_probe::record_allocation(new_size);
        }
        replacement
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: CountingMiMalloc = CountingMiMalloc;

pub mod accuracy;
pub mod adaptive;
/// AgentX agentic-replay timing mode (scheduled-runtime Workload).
pub mod agentic_replay;
/// Subagent tree-spec side channel (`TreeSpec`) for the `agentic_replay` timing mode.
pub mod agentic_tree;
/// Byte-exact AgentX port (WEKA replay, scenario locks, trajectory timing).
pub mod agentx;
#[cfg(feature = "dynamo-aic-forward-pass")]
pub mod aic_runtime;
pub mod ancillary;
#[cfg(feature = "dynosim")]
pub mod dynosim;
pub mod export;
pub mod failure;
pub mod fixed_schedule;
pub mod metrics;
pub mod multiturn;
pub mod phase_runtime;
pub mod realtime;
pub mod report;
pub mod request_rate;
pub mod run;
pub mod scheduled;
pub mod scheduler;
pub mod user_centric;
pub mod workload;

#[cfg(feature = "streaming")]
pub mod streaming;

#[cfg(feature = "engine")]
pub mod engine;

pub mod accuracy_core;
pub mod adaptive_core;
pub mod body_plan;
pub mod cellular;
pub mod clock;
pub mod config;
pub mod content_server;
pub mod dataset;
pub mod dispatch;
pub mod endpoints;
pub mod eval;
pub mod extensions;
pub mod gpu_telemetry;
pub mod graph;
#[cfg(all(feature = "cellular", feature = "engine"))]
pub mod hub;
pub mod metrics_core;
pub mod network_latency;

/// Unified definition facade: `Definition`, lookups, and the per-tag metric
/// lookup resolve through one path (`aiperf_runtime::definitions`).
pub use crate::metrics_core::definition as definitions;

pub mod rng;
pub mod server_metrics;
pub mod timing;
pub mod transport;

#[cfg(test)]
mod test_util;

/// Thread-local allocation accounting for deterministic unit-test probes.
#[cfg(test)]
pub(crate) mod allocation_probe {
    use std::cell::Cell;

    thread_local! {
        static IS_MEASURING: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<u64> = const { Cell::new(0) };
        static ALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
    }

    /// One completed allocation interval.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct AllocationSample {
        pub(crate) allocation_count: u64,
        pub(crate) allocated_bytes: u64,
    }

    /// Failure-safe allocation interval on the current test thread.
    pub(crate) struct AllocationProbe {
        is_active: bool,
    }

    impl AllocationProbe {
        /// Begin an allocation interval on the current test thread.
        pub(crate) fn start() -> Self {
            IS_MEASURING.with(|flag| {
                assert!(!flag.replace(true), "allocation probes may not nest");
            });
            ALLOCATION_COUNT.set(0);
            ALLOCATED_BYTES.set(0);
            Self { is_active: true }
        }

        /// Stop accounting and return the interval snapshot.
        pub(crate) fn finish(mut self) -> AllocationSample {
            self.disable();
            AllocationSample {
                allocation_count: ALLOCATION_COUNT.get(),
                allocated_bytes: ALLOCATED_BYTES.get(),
            }
        }

        fn disable(&mut self) {
            if self.is_active {
                IS_MEASURING.with(|flag| {
                    assert!(flag.replace(false), "allocation probe was not active");
                });
                self.is_active = false;
            }
        }
    }

    impl Drop for AllocationProbe {
        fn drop(&mut self) {
            self.disable();
        }
    }

    pub(super) fn record_allocation(bytes: usize) {
        IS_MEASURING.with(|flag| {
            if flag.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get().saturating_add(1)));
                ALLOCATED_BYTES.with(|total| {
                    total.set(total.get().saturating_add(bytes as u64));
                });
            }
        });
    }

    #[test]
    fn dropping_probe_disables_the_thread_local_interval() {
        {
            let _probe = AllocationProbe::start();
            record_allocation(7);
        }

        let probe = AllocationProbe::start();
        record_allocation(11);
        assert_eq!(
            probe.finish(),
            AllocationSample {
                allocation_count: 1,
                allocated_bytes: 11,
            }
        );
    }
}
