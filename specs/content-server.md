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

## Source anchors

- `rust/runtime/src/content_server/` (`server.rs`, `tracker.rs`, `publisher.rs`,
  `model.rs`, `error.rs`).
- `rust/runtime/src/dataset/generator/` (media generation and publication seam).
- `rust/e2e/tests/test_content_server*` and content-server tutorial coverage.
