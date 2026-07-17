<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Content server

## Purpose

`aiperf_runtime::content_server` is a run-owned HTTP delivery sidecar that serves
generated media (images, video) so an inference server can fetch it by URL. It is
an online delivery sidecar, not a transport or execution mode.

## Built

The front end retains the public `AIPERF_CONTENT_SERVER_*` environment surface and
projects it into the strict protocol-v2 request; the runner validates that
projection without filesystem or socket effects. At execution time one run-owned
resource binds the HTTP listener, serves a confined directory, tracks complete
transfers, and shuts down after benchmark drain.

- Disabled by default; externalization activates only when both `ENABLED` and a
  non-empty content directory are present. Default host `0.0.0.0`, port `8090`.
  The advertised URL derives from configured host and port.
- Ready-before-use startup; `/healthz` health endpoint and `/content/*` file
  routes. MIME selection by extension; path/symlink traversal confinement with 404
  behavior.
- A request tracker keeps bounded recent records plus non-evicting lifetime
  totals, with lowercase/duplicate header handling, two-clock timing (monotonic
  intervals), and terminal body/chunk/byte accounting.
- Graceful run-lifecycle cleanup.

Synthetic media generation stays in `aiperf_runtime::dataset`; an injected
publication trait selects the inline representation or persisted image/video URLs
(`images/img_NNNNNN.ext`, `video/vid_NNNNNN.ext`; audio remains inline). Scheduled
and graph online HTTP execution run the server; native gRPC and offline pairs
reject the sidecar rather than accepting an inert configuration, and continue
synthetic media generation through the same publisher seam with the inline
implementation.

## Future requirements

### Request-correlated media-fetch metrics

Today the tracker measures each transfer in full (TTFB, time-to-first-body-byte,
transfer duration, total latency, bytes, chunks) and then discards it: the only
caller of `request_snapshot()` is a unit test. The records also carry no benchmark
identity — `path` is just `images/img_NNNNNN.ext`, which dataset recycling reuses
across many requests — so a transfer cannot be tied back to the request that
carried it. This section specifies closing both gaps: correlate every content
fetch to the originating request (and the specific media slot within it) and
surface the timings as first-class metrics.

**Correlation key — `(rid, mi)` plus dispatch time, embedded in the media URL.**
The URL is the only datum that provenance-flows across all three hops (AIPerf →
inference server → content-server GET); a real inference server does not forward
AIPerf's `X-Request-ID` header onto its own fetch. At dispatch, each `http(s)`
`image_url`/`video_url` value whose base matches this run's content server is
tagged `?rid=<x_request_id>&mi=<media_ordinal>&td=<dispatch_wall_ns>`, where `mi`
is the zero-based ordinal of the media part within that turn's payload (assigned
by walk order). `mi` is required, not optional: a single turn may carry many
media, and `rid` alone collapses them into one ambiguous bucket. Only
content-server URLs are tagged; user-supplied external image URLs are left
untouched. The content server already records the raw `query_string`, so capture
needs no server change; `rid`/`mi`/`td` parse out at drain time. The join key is
`(rid, mi)`.

**Self-describing records (the linchpin for `time_to_media_fetch`).** Benchmark
dispatch timestamps are monotonic ns off `RealClockAnchor` (no wall component);
the tracker records arrival as wall-clock Unix-epoch `timestamp_ns`. Rather than
reconcile the two clocks across a shared map, the dispatch wall time (`td`,
`SystemTime::now()` at tag time) travels inside the URL and is recorded verbatim
in the content server's `query_string`. `time_to_media_fetch = timestamp_ns − td`
is then computed from the single record with no cross-thread dispatch map and no
run-start clock pairing — both wall-clock, same epoch.

**Streaming drain (correct at 1M-scale).** A snapshot-at-teardown join is bounded
by `max_tracked_records` (FIFO eviction) and by retain-all memory, so it silently
drops requests under load. Instead the tracker forwards each completed record over
an optional channel to an online aggregator that folds the derived values into
`DistributionStats` streaming — no retain-all, no bounded-buffer loss. Because
each record is self-describing (`rid`/`mi`/`td` in the query string), the
aggregator needs no request-side state. Any record it cannot parse (missing/late
tag) is counted and logged, never silently discarded.

**Metrics.** Per `(rid, mi)`, then rolled up per request:

| Metric | Meaning | Record source |
|---|---|---|
| `time_to_media_fetch` | dispatch → content-server arrival of the GET (server fetch lag; parallels TTFT) | `timestamp_ns` (bridged) − dispatch |
| `media_serving_latency` | arrival → last byte sent | `latency_ns` |
| `media_time_to_first_byte` | arrival → response start | `time_to_first_byte_ns` |
| `media_transfer_duration` | first → last body chunk | `transfer_duration_ns` |
| `media_bytes_served` | body bytes (summed per request) | `body_bytes` |
| `media_fetch_count` | fetches observed for the request | count of records per `rid` |

Only `time_to_media_fetch` is new math; the other five are already measured and
merely unsurfaced. Comparing arrival timestamps across one `rid` reveals serial
vs concurrent fetching; a request whose carried-media count exceeds its
`media_fetch_count` reveals a partial/skipped fetch by the inference server.

**Surfaces.** (1) A dedicated **Media** metrics section rendered with the standard
avg/p50/p90/p99 shape, produced by the streaming aggregator's own
`DistributionStats` (the domain-neutral `SidecarMetric` channel, as
`server_metrics` uses) rather than late per-record injection into the main
accumulator — the scalable fit for the streaming path. (2) A per-record
`media_records` artifact (JSONL/parquet) keyed by `(rid, mi)` with every timing,
status, and byte field for offline joining against the request records.
Considered alternative: registering `Record`-kind catalog metrics so the values
land inline in the main percentile table and existing per-record artifacts; this
needs late per-record injection (retain records or a late-fold entry point) that
does not compose with the streaming, exact-fold hot path at scale.

**Mock-server prerequisite (built).** Exercising this end to end requires the
inference server to actually fetch the tagged URLs. The Rust mock server does so
under `--fetch-content-urls` (`MOCK_SERVER_FETCH_CONTENT_URLS`), forwarding the
tagged URL verbatim so `rid`/`mi` reach the content server; see
[`mock-server.md`](mock-server.md).

## Source anchors

- `rust/runtime/src/content_server/` (`server.rs`, `tracker.rs`, `publisher.rs`,
  `model.rs`, `error.rs`).
- `rust/runtime/src/dataset/generator/` (media generation and publication seam).
- `rust/cli/tests/online_v2_stdio.rs`.
- `docs/tutorials/content-server.md`.
- Planned touch points for media-fetch metrics: URL tagging at
  `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs` (dispatch site with
  `uuid` in scope; graph path unifies at
  `rust/runtime/src/transport/http/transport/endpoint_binding.rs`); clock pairing
  at `rust/runtime/src/engine/execute.rs` run start; drain/aggregate/surface in
  `rust/runtime/src/engine/execute.rs` (mirroring the gpu_telemetry /
  network_latency `records_path` and `SidecarMetric` seams) and
  `rust/runtime/src/metrics_core/` (`sidecar.rs`, `catalog.rs`).
