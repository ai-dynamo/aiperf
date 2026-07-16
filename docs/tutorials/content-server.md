<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Serve Synthetic Multimodal Content over HTTP

By default, AIPerf embeds generated images and videos directly in inference
requests as base64 data URIs. The optional content server writes those encoded
objects to disk and replaces them with short HTTP URLs. The inference server
then fetches each object through its ordinary `image_url` or `video_url` path.

The content server is owned by the same `aiperf --execute` child as the benchmark:
it binds before endpoint readiness and request execution, remains available
while in-flight requests drain, and shuts down before the child exits. It also
retains a bounded record of complete HTTP transfers, including response status,
headers, bytes, chunks, time to headers, time to first body byte, transfer time,
and total latency.

## Quick start

Create a vision configuration and an existing directory for generated media:

```bash
aiperf config init --template multimodal_vision --output vision.yaml
mkdir -p /tmp/aiperf-content

AIPERF_CONTENT_SERVER_ENABLED=true \
AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
aiperf profile --config vision.yaml
```

Generated values now look like
`http://0.0.0.0:8090/content/images/img_000001.png`, and the corresponding file
is stored under `/tmp/aiperf-content/images/`. Images use independent,
zero-padded names (`img_000001.png` or `.jpeg`); videos use
`video/vid_000001.mp4` or `.webm`.

Audio remains inline. The OpenAI `input_audio` shape requires encoded audio and
does not accept a URL in place of its base64 data.

## Configuration

| Environment variable | Default | Description |
|---|---:|---|
| `AIPERF_CONTENT_SERVER_ENABLED` | `false` | Add the run-owned native HTTP sidecar. |
| `AIPERF_CONTENT_SERVER_HOST` | `0.0.0.0` | Interface to bind and host embedded in generated URLs. |
| `AIPERF_CONTENT_SERVER_PORT` | `8090` | Listener port (`1`–`65535`). |
| `AIPERF_CONTENT_SERVER_CONTENT_DIR` | empty | Existing media directory. A non-empty value activates image/video file publication. |
| `AIPERF_CONTENT_SERVER_MAX_TRACKED_RECORDS` | `10000` | Recent completed transfers retained in memory (`100`–`1000000`); lifetime counters do not evict. |

File publication activates only when both `ENABLED=true` and `CONTENT_DIR` is
non-empty. If the server is enabled without a directory, AIPerf creates a
run-scoped temporary serving root but deliberately preserves the normal inline
image/video representation. This matches the original feature contract.

The directory is resolved to an absolute path by the Python Config-v2 frontend.
Validation does not create or inspect it; it must exist and be a directory when
execution begins.

## HTTP surface

| Endpoint | Behavior |
|---|---|
| `GET /healthz` | Returns `200 OK` with `ok`. |
| `GET /content/{path}` | Streams a file with an inferred MIME type and supports byte ranges. |

The serving root is canonicalized. Parent traversal, encoded traversal, and
symlink escapes outside that root are rejected. Generated media is published
through same-directory temporary files; modality-directory escapes are refused
and an existing final symlink is replaced rather than followed.

## Network placement

The inference server—not the AIPerf client transport—fetches these URLs. Set
`AIPERF_CONTENT_SERVER_HOST` to an address the inference server can reach:

- On one host, `127.0.0.1` is the clearest advertised address.
- With host-networked containers, use the host address visible in the container.
- Across machines or Kubernetes pods, use a routable pod IP, host IP, or DNS
  name and expose the selected port.

`0.0.0.0` is useful as a bind interface but is not a portable remote
destination. Because the configured host is also embedded in media URLs, choose
an explicit reachable address when the inference server is not local.

The native sidecar is registered for scheduled, graph, and static-accuracy
online HTTP execution. Agentic, evaluation, native gRPC, and offline pairs
reject it instead of silently ignoring the configuration.

## Troubleshooting

- **Generated values are still base64:** set both `ENABLED=true` and a non-empty
  `CONTENT_DIR`.
- **The run fails before sending requests:** create the configured directory and
  ensure it is readable and writable by AIPerf.
- **The model server cannot fetch files:** use a reachable advertised host, open
  the port, and verify `curl http://HOST:PORT/healthz` from the model server's
  network namespace.
- **Port already in use:** select another `AIPERF_CONTENT_SERVER_PORT`.
