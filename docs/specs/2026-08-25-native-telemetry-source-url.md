<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native GPU telemetry source URL artifact migration

## Context

Origin/main commit `93b6223373` replaces the GPU telemetry record field
`dcgm_url` with the collector-neutral `telemetry_source_url`. The upstream
Python model, message, accumulator hierarchy, dashboard, JSONL writer,
fixtures, and tests all adopt the new spelling without emitting an alias.

The native runtime already represents source identity neutrally as
`GpuTelemetryRecord.endpoint_url`. Its public JSONL adapter is the remaining
gap: `TelemetryRow` exposes that value as `dcgm_url`.

## Contract

Every native `gpu_telemetry_export.jsonl` record must contain exactly one
source-identity field named `telemetry_source_url`. Its value is the existing
`GpuTelemetryRecord.endpoint_url` verbatim, including HTTP DCGM URLs and local
collector identifiers such as `pynvml://localhost` and
`amdsmi://localhost`.

The deprecated `dcgm_url` member must not be serialized. This is an intentional
schema replacement matching origin/main, not a dual-write transition. Field
ordering, optional metadata omission, platform identity, timestamps, and the
normalized `telemetry_data` map remain unchanged.

## Implementation boundary

Rename only the source member of the private `TelemetryRow` serialization DTO
in `rust/runtime/src/engine/gpu_telemetry.rs`, retaining the mapping from the
internal `record.endpoint_url`. Do not rename DCGM-specific configuration,
source implementations, mock-server helpers, metric tag variables, or the
internal record and series-key fields.

Update the native telemetry design record so it no longer promises the old
compatibility spelling. Upstream's merged dataflow references already use
`telemetry_source_url`.

## Tests

The focused runtime serializer test must assert the exact new key/value and use
object membership to prove `dcgm_url` is absent. The real sidecar JSONL test
must make the same assertion for AMD SMI records while preserving its existing
platform and metric checks.

The two native product tests that inspect emitted telemetry JSONL must require a
non-empty string at `telemetry_source_url` and assert that `dcgm_url` is absent.
No new mock layer is needed: these tests already exercise the native serializer
through a real runtime and mock DCGM endpoints.

Acceptance requires focused runtime unit tests and the GPU telemetry E2E test
target using the configured `sccache` wrapper and a target directory below
`/mnt/4tb`.

When `profile_export_prefix` is configured, native resolution must apply the
same normalized export stem to the GPU telemetry sidecar and produce
`<stem>_gpu_telemetry.jsonl`. With no prefix, the existing
`gpu_telemetry_export.jsonl` default remains unchanged. This is required so the
custom-prefix product test reads and validates the migrated public schema
instead of silently skipping the artifact.

## Non-goals

- Backward-compatible parsing of historical JSONL artifacts.
- Dual-writing the old and new keys.
- Renaming DCGM collector configuration or mock-server APIs.
- Changing telemetry collection, accumulation, summaries, cadence, or error
  handling.
