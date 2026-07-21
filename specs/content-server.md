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

Every content-server-hosted media fetch is correlated back to the request and
media slot that carried it, and surfaced as first-class metrics.

**Correlation key — `(rid, mi)` plus dispatch time, embedded in the media URL.**
The URL is the only datum that provenance-flows across all three hops (AIPerf →
inference server → content-server GET); a real inference server does not forward
AIPerf's `X-Request-ID` header onto its own fetch. At dispatch,
`tag_content_urls` (`rust/runtime/src/transport/http/sink.rs`) walks the outgoing
JSON payload and rewrites every string value that starts with this run's
content-server base URL, appending
`?rid=<x_request_id>&mi=<media_ordinal>&td=<dispatch_wall_ns>` (`&` instead of `?`
if the URL already carries a query string). `tag_media_urls`/`parse_media_tag`
(`rust/runtime/src/content_server/media_tag.rs`) implement the walk and the
inverse parse; `mi` is assigned by document walk order and is agnostic to the
endpoint dialect's media-part shape (Chat `image_url:{url}`, Responses
`image_url:"<url>"`, Messages `source:{url}`). Only strings whose prefix matches
the content-server base are rewritten, so user-supplied external URLs are left
untouched. The content server already records the raw `query_string` verbatim, so
no server-side change was needed to capture it; `rid`/`mi`/`td` parse back out at
drain time. The join key is `(rid, mi)`.

**Self-describing records (the linchpin for `time_to_media_fetch`).** The dispatch
wall time (`td`, captured at tag time) travels inside the URL and is recorded
verbatim in the content server's `query_string`. `time_to_media_fetch` is computed
per record as `timestamp_ns − td` (`MediaFetchAggregator::ingest`,
`rust/runtime/src/content_server/media_metrics.rs`) — both wall-clock, same
epoch, no cross-thread dispatch map and no run-start clock pairing. A negative
result (clock skew) is clamped to `0` and counted under `negative_ttmf`.

**Streaming drain (correct at 1M-scale).** Each completed content-server transfer
is folded online into `MediaFetchAggregator`, which accumulates per-fetch samples
(`time_to_media_fetch`, serving latency, time-to-first-byte, transfer duration)
and a per-request rollup (bytes, fetch count) without retaining the records
themselves, so memory scales with distinct requests in flight, not total fetch
volume. `finish()` folds the accumulated samples into six
`SidecarMetric`/`DistributionStats` gauges (`linear_distribution`, ddof=1). A
record whose query string carries no parseable `(rid, mi, td)` tag is excluded,
counted under `unmatched`, and logged via `tracing::warn!` — never silently
dropped.

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

**Surfaces.** (1) A dedicated **Media** metrics section (`RunOutcome::media_metrics`
in `rust/runtime/src/metrics_core/report.rs`, populated from
`SidecarExecutionState::finalize_media_metrics` in
`rust/runtime/src/engine/execute.rs`), rendered with the standard avg/p50/p90/p99
shape from the streaming aggregator's own `SidecarMetric`/`DistributionStats`
channel — the same domain-neutral channel `server_metrics` uses — rather than
late per-record injection into the main accumulator. Empty (absent from the
report) unless a content server served tagged media. (2) A per-record
`media_records.jsonl` artifact (one `MediaRecord` per line, keyed by `(rid, mi)`,
written as fetches arrive via `MediaRecordWriter`) with every timing, status, and
byte field for offline joining against the request records.

**Mock-server prerequisite (built).** Exercising this end to end requires the
inference server to actually fetch the tagged URLs. The Rust mock server does so
under `--fetch-content-urls` (`MOCK_SERVER_FETCH_CONTENT_URLS`), forwarding the
tagged URL verbatim so `rid`/`mi` reach the content server; see
[`mock-server.md`](mock-server.md).

## Source anchors

- `rust/runtime/src/content_server/` (`server.rs`, `tracker.rs`, `publisher.rs`,
  `model.rs`, `error.rs`) — sidecar lifecycle, routes, and the publication seam.
- `rust/runtime/src/content_server/media_tag.rs` — `tag_media_urls`/
  `parse_media_tag`, the `(rid, mi, td)` URL-tagging correlation scheme.
- `rust/runtime/src/content_server/media_metrics.rs` — `MediaFetchAggregator`,
  `MediaRecord`, `MediaRecordWriter`; the streaming join, the six
  `SidecarMetric` distributions, and the `media_records.jsonl` artifact writer.
- `rust/runtime/src/transport/http/sink.rs` (`tag_content_urls`) — dispatch-time
  tagging of outgoing payloads.
- `rust/runtime/src/engine/execute.rs` (`finalize_media_metrics`,
  `MEDIA_RECORDS_FILENAME`) — aggregator/writer lifecycle wiring and drain into
  `RunOutcome::media_metrics`.
- `rust/runtime/src/metrics_core/report.rs` (`RunOutcome::media_metrics`) —
  report-surface field.
- `rust/runtime/src/dataset/generator/` (media generation and publication seam).
- `rust/cli/tests/online_v2_stdio.rs`.
- `docs/tutorials/content-server.md`.
