// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(
    not(feature = "graph-transport-bench"),
    doc = r#"
The direct raw-HTTP graph microbenchmark is intentionally absent from the
default library surface:

```compile_fail
use aiperf_runtime::graph::transport_bench::run_transport_bench;
```
"#
)]

//! Graph-IR async-dataflow workload driver.
//!
//! Runs a DAG of chat requests with fan-out/fan-in dependencies and firing-gate
//! timing: nodes fire when their input channels are ready, dispatch through the
//! extensible [`sink::GraphSink`] (for example, HTTP or an in-process engine), and
//! measurement flows to `crate::dispatch`'s shared `TraceCollector`. Prompts are
//! materialized from a content-addressed [`segment::SegmentStore`] plus dynamic
//! predecessor replies.
//!
//! This crate stays independent of application backends. Its
//! [`runtime::drive_sim_with_source`] pump merges the [`crate::clock::SimClock`]
//! sleeper heap with any injected [`runtime::SimEventSource`]. The optional
//! Dynamo adapter lives in the `aiperf` application crate, so neither this
//! graph engine nor its clock/measurement leaves acquire a mocker dependency.

pub mod agent;
pub mod bench;
pub mod channel_store;
pub mod channels;
pub mod conditional;
pub mod context;
pub mod dag_source;
pub mod driver;
pub mod errors;
pub mod execution;
pub mod executor;
pub mod flat;
pub mod input;
pub mod inspect;
mod lowering;
pub mod materialize;
pub mod model;
pub mod placement;
pub mod policy;
pub mod recorded;
pub mod reducers;
pub mod replay;
pub mod report;
pub mod run;
pub mod runtime;
pub mod scheduler;
pub mod segment;
pub mod sink;
pub mod snapshot;
mod static_readiness;
pub mod supplement;
mod syslimits;
mod timing;
pub mod tools;
#[cfg(feature = "graph-transport-bench")]
pub mod transport_bench;
pub mod transport_sink;
pub mod tstar;
pub mod validate;
pub mod warmup_handoff;
pub mod wire;
pub mod workload;

#[cfg(test)]
mod report_tests {
    use super::report::GraphRpsReport;

    fn report(wall_secs: f64) -> GraphRpsReport {
        GraphRpsReport {
            completed: 12,
            errors: 1,
            output_tokens: 36,
            wall_secs,
            ttft_p50_ms: 10.0,
            ttft_p90_ms: 20.0,
            ttft_p99_ms: 30.0,
            ttft_mean_ms: 15.0,
            native_metrics: crate::metrics_core::AccumulatorSummary::new(),
        }
    }

    #[test]
    fn neutral_graph_report_derives_rates_and_rejects_zero_duration_division() {
        let metered = report(2.0);
        assert_eq!(metered.rps(), 6.0);
        assert_eq!(metered.output_tps(), 18.0);

        let no_duration = report(0.0);
        assert_eq!(no_duration.rps(), 0.0);
        assert_eq!(no_duration.output_tps(), 0.0);
    }
}
