---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Rust Internals Diagram Atlas
---

# Rust Internals Diagram Atlas

This page collects the Mermaid source diagrams that document the native Rust
architecture at both the entry-point and subsystem-internals levels. Each link
opens the source `.mmd` file under `docs/diagrams/rust/`.

## Entry-Point Diagrams

- [Rust Overview](rust/overview.mmd) - crate relationships and ownership.
- [Execution Path](rust/execution-path.mmd) - `aiperf profile` through
  `aiperf --execute`.
- [Runtime Seams](rust/runtime-seams.mmd) - `Clock`, workload, transport, and
  metrics composition seams.

## Cellular Internals

- [Cellular Execution and Merge](rust/cellular-execution-and-merge.mmd) -
  controller and cell partitioning, shipping paths, and merge modes.
- [Cellular Velo Hub Planes](rust/cellular-velo-hub-planes.mmd) - hub plugins
  across the HTTP diagnostics and Velo control-data surfaces.
- [SLURM Native Cellular Topology](rust/slurm-native-cellular-topology.mmd) -
  rank topology, controller addressing, and flat-star collection.

## Graph, Dataset, and Materialization

- [Graph-IR Compilation](rust/graph-ir-compilation.mmd) - `dag_jsonl`,
  `weka_trace`, and `dynamo_trace` lowering into graph plans plus one frozen
  segment store.
- [Graph Trace Dataflow Execution](rust/graph-trace-dataflow-execution.mmd) -
  deterministic async execution through the scheduler, channel store, and
  `GraphSink`.
- [Dataset Linear Build Pipeline](rust/dataset-linear-build-pipeline.mmd) -
  source loading, format detection, composition, context rebasing, and frozen
  dataset assembly.
- [Segment Store to Wire Materialization](rust/segment-store-to-wire-materialization.mmd) -
  segment interning, endpoint payload planning, JSON splicing, and separate
  gRPC codec materialization.

## Execution, Registry, and Metrics

- [Scheduling Admission and Dispatch](rust/scheduling-admission-dispatch.mmd) -
  `sharded`, `global`, and `global-hop` execution flow.
- [Extension Registry Bootstrap](rust/extension-registry-bootstrap.mmd) -
  transactional extension registration into one frozen `AIPerfRegistry`.
- [Metrics, Export, and Telemetry Flow](rust/metrics-export-telemetry-flow.mmd) -
  observer ingest, exact-sketch reduction, telemetry sidecars, and exporter
  fan-out.
