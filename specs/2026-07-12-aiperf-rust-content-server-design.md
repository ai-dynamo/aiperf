<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust-native generated-content server

**Status:** built for the canonical Python Config-v2 → `aiperf-cli` online
HTTP path.

## Decision

Port the behavior added by `ajc/content-server` into the canonical native
architecture without recreating its Python service, multiprocessing hooks, or
message-bus status broadcast.

Python retains the public `AIPERF_CONTENT_SERVER_*` environment surface and
projects it into the strict authored protocol-v2 request. The runner validates
that projection without filesystem or socket effects. At execution time one
run-owned Rust resource binds the HTTP listener, serves a confined directory,
tracks complete transfers, and shuts down after benchmark drain. Synthetic
media generation remains in `aiperf_runtime::dataset`; an injected publication trait
selects the existing inline representation or persisted image/video URLs.

This is an online delivery sidecar, not a new transport or execution mode.
Native gRPC and offline pairs reject the sidecar rather than accepting an inert
configuration. Their synthetic media generation continues through the same
publisher seam with the inline implementation.

## Source audit and compatibility contract

The implementation was derived after reading the complete feature diff and its
meaningful callers, models, and tests:

| Python source on `ajc/content-server` | Native contract retained |
|---|---|
| `src/aiperf/common/environment.py:53-96` | Disabled-by-default settings, host `0.0.0.0`, port `8090`, empty directory, and bounded record limits. |
| `src/aiperf/content_server/models.py:10-124` | Serializable request record, status, and tracker snapshot fields and defaults. |
| `src/aiperf/content_server/request_tracker.py:25-206` | Bounded recent records, non-evicting lifetime totals, lowercase/duplicate header handling, two-clock timing, and terminal body accounting. |
| `src/aiperf/content_server/server.py:30-199` | Existing-directory validation, temporary-root behavior, ready-before-use startup, `/healthz`, `/content/*`, MIME selection, traversal confinement, and run cleanup. |
| `src/aiperf/dataset/generator/base.py:12-67` | Optional file-output policy and stable URL construction. |
| `src/aiperf/dataset/generator/image.py:18-180` | Persist the final PNG/JPEG encoded bytes and use `images/img_NNNNNN.ext`. |
| `src/aiperf/dataset/generator/audio.py:21-193` | Audio remains inline. |
| `src/aiperf/dataset/generator/video.py:20-507` | Persist the final MP4/WebM bytes and use `video/vid_NNNNNN.ext`. |
| `src/aiperf/dataset/composer/{base,synthetic,synthetic_rankings,custom}.py` | Publication is selected outside codec/generation logic and shared by applicable generated media. |
| `src/aiperf/dataset/dataset_manager.py:261-289` | Externalization activates only when both `ENABLED` and non-empty `CONTENT_DIR` are present; the advertised URL derives from configured host and port. |
| `src/aiperf/endpoints/openai_chat.py` | Endpoint-ready image/video URL values flow through the ordinary OpenAI-compatible request builder; audio keeps its required encoded shape. |

The original unit tests under `tests/unit/content_server/` and
`tests/unit/dataset/test_content_server_integration.py` define the observable
edge cases: independent counters, exact names and URLs, disabled/empty-directory
fallback, health and file responses, MIME types, 404/traversal behavior, body
chunk and byte totals, monotonic intervals, duplicate headers, bounded eviction,
startup readiness, and cleanup.

## Public configuration and strict wire

`src/aiperf/common/environment.py` owns the public settings:

| Setting | Default | Validation |
|---|---:|---|
| `AIPERF_CONTENT_SERVER_ENABLED` | `false` | Boolean. |
| `AIPERF_CONTENT_SERVER_HOST` | `0.0.0.0` | Non-empty. |
| `AIPERF_CONTENT_SERVER_PORT` | `8090` | `1..=65535`. |
| `AIPERF_CONTENT_SERVER_CONTENT_DIR` | empty | Expanded to an absolute path only when non-empty. |
| `AIPERF_CONTENT_SERVER_MAX_TRACKED_RECORDS` | `10000` | `100..=1_000_000`. |

`src/aiperf/orchestrator/rust_wire.py` omits the sidecar when disabled. When
enabled it authors `resources.sidecars.content_server` with host, port, and
record capacity, plus `content_dir` only when the environment value is
non-empty. This projection does not create, canonicalize, or read the directory.

`runner/src/sidecar_input.rs` owns the single strict decode into
`ContentServerSpec`. Unknown fields, whitespace-padded or invalid hosts,
non-origin URL shapes, invalid bounds, empty authored paths, and relative paths
fail during validation. A missing absolute directory remains valid at this
stage so `validate` is side-effect-free; execution reports the missing path.

The sidecar is an authored resource, not a runner capability pair. Pair
adapters decide whether it is executable:

| Pair | Content-server policy |
|---|---|
| `online_http + scheduled` | Built; generated image/video values can be externalized. |
| `online_http + graph` | Server resource is accepted; graph-provided URLs can fetch from the serving root. |
| `online_http + static_accuracy` | Resource lifecycle is accepted, though evaluator-authored inputs are not rewritten. |
| `online_grpc + scheduled` | Rejected with the other unsupported sidecars. |
| `dynosim + scheduled/graph` | Rejected as an online sidecar. |

## Dataset publication seam

`aiperf_runtime::dataset::SyntheticMediaPublisher` is the extension boundary between a
codec/generator and its endpoint-ready representation:

```text
native image/audio/video generator
              │ encoded bytes + SyntheticMediaFormat
              ▼
    SyntheticMediaPublisher
       ├── InlineSyntheticMediaPublisher
       │      └── data URI or audio `format,base64`
       └── ContentServerMediaPublisher
              ├── image/video: atomic name → file → HTTP URL
              └── audio: delegate to inline publisher
```

`SyntheticMediaGeneratorFactory` continues to select generator
implementations. `NativeSyntheticMediaGeneratorFactory` now accepts an
`Arc<dyn SyntheticMediaPublisher>`, so alternate publication does not leak
filesystem or HTTP policy into image, audio, video, composer, segment-store, or
endpoint code. The default constructor uses the inline publisher and therefore
preserves every existing mode and wire value.

The content publisher validates one canonical existing root and one plain
HTTP(S) origin. Image and video counters are independent atomics starting at
one. It creates only the `images/` and `video/` children, writes complete final
encoded objects, and returns:

- `{base}/content/images/img_{counter:06}.{png|jpeg}`
- `{base}/content/video/vid_{counter:06}.{mp4|webm}`

The publisher canonicalizes the modality directory beneath the serving root,
rejects subdirectory symlink escapes, writes through a same-directory temporary
file, and atomically replaces the final name. Existing final symlinks are
replaced rather than followed, so publication cannot overwrite their targets.
Failures are dataset construction failures; the runner never falls back to
base64 after the user explicitly selected externalization.

## Native server module

The `aiperf_runtime::content_server` module (formerly the `aiperf-content-server` crate,
now inlined as a module of `aiperf`) is a run-resource leaf that depends on the
sibling `aiperf_runtime::dataset` module for the publication trait. Its replaceable
boundaries are:

- `ContentServerClock`: wall time for correlation plus monotonic time for
  intervals.
- `ContentServerFactory`: listener/resource construction.
- `ContentServerRuntime`: status, bound address, tracker snapshot, and graceful
  shutdown.
- `SyntheticMediaPublisher` (owned by `aiperf_runtime::dataset`): final media delivery
  representation.

The built factory uses Axum and `tower-http::ServeDir`. It binds the listener
before returning, which is the readiness barrier. An absent directory creates a
run-scoped `TempDir`; an authored directory must already exist and is
canonicalized once. Drop aborts the task as a safety net, while the runner's
normal path performs graceful shutdown.

The HTTP surface is deliberately small:

- `GET /healthz` returns `200` and `ok`.
- `/content/{path}` streams files, infers MIME type, supports HTTP byte ranges,
  and returns normal 404 responses.

Before `ServeDir`, the server percent-decodes and validates path components,
canonicalizes existing targets, and requires the result to remain beneath the
canonical root. This rejects textual parent traversal, encoded traversal, and
symlink escapes. It is stronger than lexical prefix checking and preserves the
original feature's confinement intent.

## Full-response tracking

Tracking wraps the actual response body, not only the route future. For every
HTTP request it captures:

- wall-clock arrival timestamp;
- method, path, raw query, version, client address, and lowercase request
  headers;
- response status, MIME type, and lowercase response headers;
- actual non-empty body bytes and chunks yielded toward Hyper;
- monotonic arrival→headers, arrival→first-body, first-body→last-body, and
  arrival→terminal intervals;
- a terminal body error or early body drop.

Duplicate header values are joined with `, `, matching the Python helper, and
raw header bytes use the HTTP/ASGI-compatible byte-to-codepoint mapping rather
than lossy UTF-8 replacement. The tracker retains at most the configured recent
records while lifetime request and byte counters use saturating addition and
survive eviction. A locked snapshot is internally consistent.

The runtime exposes the snapshot for future native metrics/export consumers.
As on the source branch, the content-server feature itself does not create a
new report artifact or metric identity.

## Runner lifecycle and failures

The runner owns the resource in `PreparedNativeSidecarResources`:

1. Python authors one strict Config-v2 request.
2. Rust validates every authored input without effects.
3. Pair preparation generates the dataset. With an authored directory, final
   image/video bytes are written and URLs enter the shared segment store.
4. Execution preparation validates/canonicalizes the root and binds the content
   listener before endpoint readiness.
5. The same server remains alive through phases, cancellation, and drain.
6. Other run resources shut down, then the content server gracefully drains and
   releases its port; a temporary root is deleted with the runtime.

Bind failures, missing/non-directory roots, unsafe paths, and media write
failures are terminal infrastructure errors with context. Disabled settings are
represented by absence, not by a dormant runtime object. An enabled server with
an empty directory is intentionally live but does not switch generated values
away from inline form.

## Verification gates

The implementation is complete only when these remain green:

- `cargo test -p aiperf --lib`: publisher injection sees raw final
  bytes, the returned endpoint value is interned, and inline defaults retain
  their existing wire form.
- `cargo test -p aiperf`: naming, file bytes, audio fallback,
  model/header parity, bounded tracking, health, MIME, range, 404, traversal,
  serving and publication symlink confinement, atomic final replacement,
  startup failure, temporary-root ownership, transfer telemetry, and shutdown.
- runner sidecar tests: strict decode, bounds, default HTTP port, unknown-field
  rejection, and missing-path validation without creation.
- `runner/tests/online_v2_stdio.rs`: a real runner child generates a PNG
  URL, a mock inference server fetches it from the child content server and
  verifies the PNG signature, the run succeeds, the file persists in the
  authored directory, and the listener port is released at child exit.
- Python environment/wire tests: defaults, environment parsing and bounds,
  disabled omission, non-empty path expansion without filesystem access, and
  enabled/empty-directory fallback.

## Rejected shapes

- **A second Python service:** reproduces process lifecycle and message-bus
  machinery deliberately removed from the native path.
- **Embedding file writes in each generator:** couples codecs to one delivery
  method and cannot preserve the inline implementation as an injected policy.
- **Starting the server during validation:** violates side-effect-free authored
  protocol-v2 validation.
- **Serving the artifact directory:** mixes user-visible result ownership with
  externally fetched input objects and widens traversal impact.
- **Silently accepting the sidecar for offline or gRPC:** advertises behavior
  those pairs do not execute.
- **Trusting `ServeDir` alone for confinement:** does not explicitly enforce the
  product contract against symlinks outside the serving root.

## Implementation map

- Public settings: `src/aiperf/common/environment.py`
- Config-v2 projection: `src/aiperf/orchestrator/rust_wire.py`
- Strict sidecar adapter: `rust/runtime/src/runner_protocol/sidecar_input.rs`
- Wire field: `rust/runtime/src/protocol_v2.rs`
- Dataset adapter injection: `rust/runtime/src/runner_protocol/dataset_input.rs`
- Pair preparation: `rust/runtime/src/runner_protocol/online_execution.rs`
- Lifecycle ownership: `rust/runtime/src/runner_protocol/execute.rs`
- Publication seam/generators: `rust/runtime/src/dataset/generator/`
- Server/runtime/tracking: `rust/runtime/src/content_server/`
- Product subprocess proof: `rust/cli/tests/online_v2_stdio.rs`
