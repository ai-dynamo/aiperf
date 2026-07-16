---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Native gRPC Transport
---

# Native gRPC Transport

AIPerf’s native gRPC path uses Tonic and KServe’s Open Inference Protocol
(OIP). The native `aiperf` entry point owns Config v2, orchestration, and
presentation; a fresh `aiperf --execute` child owns the run. There is no Python
gRPC plugin, `plugins.yaml`, or protocol-v1 fallback on this path.

## Supported bindings

The built-in gRPC registry supports exactly these prepared endpoint IDs:

- `kserve_v2_infer`
- `kserve_v2_embeddings`
- `kserve_v2_rankings`
- `kserve_v2_vlm`
- `kserve_v2_images`

Use `transport.type: grpc` with `grpc://` for plaintext HTTP/2 or
`grpcs://` for TLS with WebPKI roots. See [Profile KServe Endpoints](./kserve.md)
for a complete Config v2 example.

## Request lifecycle

For every request, the worker-local endpoint formats canonical JSON. Its
prepared gRPC binding converts that JSON to `ModelInferRequest` protobuf,
selects `ModelInfer` or `ModelStreamInfer`, and decodes responses back to the
same canonical shape consumed by HTTP endpoint parsing.

The native codec covers OIP BYTES, BOOL, signed and unsigned integer,
FP16/FP32/FP64 tensors, request and tensor parameters, typed response contents,
and Triton-style raw response buffers. The checked-in proto is byte-identical
to the KServe definition used by the source implementation.

All benchmark-visible timestamps, request deadlines, and cancellation delays
come from AIPerf’s injected `Clock`. Tonic does not expose Hyper’s individual
DNS/TCP/TLS events, so those HTTP-only fields remain absent rather than being
estimated.

## Unary and streaming calls

`streaming: false` uses unary `ModelInfer`. `streaming: true` uses
`ModelStreamInfer` only for endpoint bindings that advertise it. In-band stream
errors terminate the request while retaining response messages already
received.

The first meaningful parsed token releases prefill credit and anchors TTFT.
Reasoning and visible tokens retain their normal classifications before both
paths feed the shared native metrics accumulator.

## Channels and multiple targets

Set `endpoint.connectionReuse` to one of:

| Value | Native behavior |
|---|---|
| `pooled` | One multiplexed HTTP/2 channel per target |
| `never` | A fresh channel for every request |
| `sticky-user-sessions` | One channel lease per correlation ID, released on the final turn or failure |

Multiple URLs are selected by the normal URL strategy. Each worker owns its
own endpoint table, binding table, and channels. With multiple workers, AIPerf
runs one current-thread Tokio runtime and `LocalSet` on each OS thread; channels
and hot-path observers are never shared across workers.

## Metadata and authentication

Endpoint headers become lowercase ASCII gRPC metadata. AIPerf adds request and
correlation IDs and can forward the correlation ID under `sessionHeader`.
Binary `-bin` metadata is rejected because Config v2 header values are strings.

```yaml
endpoint:
  headers:
    authorization: ${INFERENCE_AUTHORIZATION}
    x-routing-tenant: benchmark
  sessionHeader: x-session-id
```

## Message sizes, keepalive, and TLS

The native defaults match the source transport:

- maximum send and receive message size: 256 MiB;
- HTTP/2 keepalive interval: 30 seconds;
- keepalive timeout: 10 seconds; and
- keepalive while idle: enabled.

`grpcs://` validates certificates using enabled WebPKI roots. A custom CA or
client-certificate surface is not currently registered; deployments requiring
one must terminate TLS at a trusted proxy or add a typed transport policy.

## Timeouts and cancellation

The endpoint timeout bounds the whole request. Channel establishment also has
a 30-second cap. Authored request cancellation is armed only after the RPC
future has been submitted, so connection establishment does not consume the
cancellation delay. At equal timestamps the deadline wins deterministically.

Cancelled calls map to HTTP-equivalent status 499 in shared metrics. Native
gRPC status and message facts remain available in the compatibility record.

## Status mapping

| gRPC | Shared status |
|---|---:|
| `OK` | 200 |
| `CANCELLED` | 499 |
| `INVALID_ARGUMENT` | 400 |
| `DEADLINE_EXCEEDED` | 504 |
| `NOT_FOUND` | 404 |
| `PERMISSION_DENIED` | 403 |
| `RESOURCE_EXHAUSTED` | 429 |
| `UNIMPLEMENTED` | 501 |
| `INTERNAL` | 500 |
| `UNAVAILABLE` | 503 |
| `UNAUTHENTICATED` | 401 |

The runner maps all 17 canonical gRPC codes; the table shows the most common
ones.

## Troubleshooting

- Triton commonly exposes HTTP on port 8000 and gRPC on port 8001.
- A scheme/backend mismatch fails Config v2 validation before the runner is
  launched.
- `kserve_v1_predict` and the KServe OpenAI-compatible endpoints are HTTP-only.
- If tensor output is empty, confirm the endpoint’s `v2_*_output_name` selector
  matches the deployed model configuration.
- Positive gRPC readiness retries and runner sidecars currently fail closed;
  use an external readiness gate and leave `waitForModelTimeout: 0`.

The successful run writes the normal `native-v2.json` report. Until the shared
placement DTOs are renamed, some internal/raw compatibility fields retain
HTTP-oriented names, but wire I/O is native protobuf over Tonic.
