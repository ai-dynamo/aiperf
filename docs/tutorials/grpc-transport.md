<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gRPC Transport Guide

AIPerf supports gRPC as a transport for the KServe V2 Open Inference Protocol, enabling benchmarking of Triton Inference Server and TRT-LLM deployments over gRPC with HTTP/2 multiplexing, server-side streaming, and TLS.

## Overview

The KServe V2 Open Inference Protocol defines both HTTP/REST and gRPC variants. AIPerf's gRPC transport works with the same `kserve_v2_infer` endpoint as HTTP -- the transport is auto-detected from the URL scheme:

| URL Scheme | Transport | Wire Format | Streaming |
|---|---|---|---|
| `http://` / `https://` | HTTP/1.1 (aiohttp) | V2 JSON REST | No (V2 REST has no streaming format) |
| `grpc://` / `grpcs://` | gRPC (HTTP/2) | V2 Protobuf | Yes (`ModelStreamInfer`) |

**Key characteristics:**

- Same `--endpoint-type kserve_v2_infer` for both transports
- Automatic transport selection based on URL scheme
- HTTP/2 multiplexing (single connection for all concurrent requests)
- Server-side streaming via `ModelStreamInfer` RPC
- gRPC status codes mapped to HTTP equivalents in metrics

## Quick Start

### Unary (Non-Streaming) Request

Send a single request and wait for the complete response:

```bash
aiperf profile \
    --model my-trtllm-model \
    --url grpc://triton:8001 \
    --endpoint-type kserve_v2_infer \
    --request-count 50 \
    --concurrency 4
```

### Streaming Request

Use `--streaming` to call `ModelStreamInfer`, which sends responses token-by-token. This enables Time to First Token (TTFT) and Inter Token Latency (ITL) metrics:

```bash
aiperf profile \
    --model my-trtllm-model \
    --url grpc://triton:8001 \
    --endpoint-type kserve_v2_infer \
    --streaming \
    --request-count 50 \
    --concurrency 4
```

### TLS-Encrypted Connection

Use the `grpcs://` scheme for TLS-encrypted channels:

```bash
aiperf profile \
    --model my-model \
    --url grpcs://secure-triton:8001 \
    --endpoint-type kserve_v2_infer \
    --request-count 50
```

## Architecture

The gRPC transport is protocol-agnostic. It delegates all proto knowledge to a pluggable **serializer** class, loaded from endpoint metadata in `plugins.yaml`:

```
Endpoint (kserve_v2_infer)          -- format_payload() returns dict
    | dict payload
GrpcTransport (BaseTransport)       -- uses serializer to convert dict <-> bytes
    | raw bytes
GenericGrpcClient                   -- proto-free, sends/receives raw bytes via gRPC
    | gRPC wire protocol (HTTP/2)
Triton / TRT-LLM Server
```

The endpoint never knows it's running over gRPC. The serializer (e.g., `KServeV2GrpcSerializer`) converts the endpoint's dict payload to protobuf bytes on the way out, and converts response bytes back to a JSON-serialized `TextResponse` on the way in. This means all existing `--extra-inputs` options (like `v2_input_name`, `v2_output_name`) work identically over gRPC.

The serializer class and gRPC method paths are declared in `plugins.yaml` endpoint metadata, so adding support for a new gRPC protocol requires only a new serializer — no transport changes.

### Request Flow

**Unary (ModelInfer):**

1. Endpoint's `format_payload()` produces a V2 JSON dict
2. `KServeV2GrpcSerializer.serialize_request()` converts dict to protobuf bytes
3. `GenericGrpcClient.unary()` sends the raw bytes via gRPC
4. `KServeV2GrpcSerializer.deserialize_response()` converts response bytes back to a dict
5. JSON is wrapped in a `TextResponse` and stored in `RequestRecord`

**Streaming (ModelStreamInfer):**

1. Same payload preparation as unary
2. `GenericGrpcClient.server_stream()` returns an async iterator of raw bytes
3. Each chunk is deserialized via `KServeV2GrpcSerializer.deserialize_stream_response()` to a `StreamChunk`
4. `first_token_callback` fires on the first non-error chunk (enabling TTFT)
5. All responses are collected in `RequestRecord.responses`

## Custom Tensor Names

If your model uses non-default tensor names, configure them via `--extra-inputs`:

```bash
aiperf profile \
    --model my-model \
    --url grpc://triton:8001 \
    --endpoint-type kserve_v2_infer \
    --extra-inputs v2_input_name:INPUT_TEXT \
    --extra-inputs v2_output_name:OUTPUT_TEXT \
    --request-count 50
```

To discover your model's tensor names, query the V2 model metadata:

```bash
# Over HTTP
curl http://triton:8000/v2/models/my-model

# Over gRPC (requires grpcurl)
grpcurl -plaintext triton:8001 inference.GRPCInferenceService/ModelMetadata
```

## Multi-URL Load Balancing

Distribute requests across multiple gRPC targets:

```bash
aiperf profile \
    --model my-model \
    --url grpc://triton-1:8001 \
    --url grpc://triton-2:8001 \
    --url grpc://triton-3:8001 \
    --endpoint-type kserve_v2_infer \
    --request-count 300 \
    --concurrency 12
```

Each URL gets its own gRPC channel. Requests are distributed across channels using the configured strategy. See the [Multi-URL Load Balancing](./multi-url-load-balancing.md) tutorial for details.

## Connection Behavior

### HTTP/2 Multiplexing

gRPC uses HTTP/2, which multiplexes all requests over a single TCP connection per target. This is fundamentally different from HTTP/1.1 connection pooling:

| Behavior | HTTP/1.1 (aiohttp) | gRPC (HTTP/2) |
|---|---|---|
| Concurrent requests per connection | 1 | Unlimited (multiplexed) |
| Connection pool needed | Yes | No (single channel) |
| Connection reuse strategy | Configurable | Always multiplexed |
| Head-of-line blocking | Per-connection | Per-stream only |

Because of this, the `--connection-reuse-strategy` option has no effect on gRPC connections. If set to `never` or `sticky_user_sessions`, AIPerf logs a warning and uses multiplexed behavior.

### Channel Options

The gRPC transport uses optimized defaults for benchmarking:

| Option | Default | Purpose |
|---|---|---|
| `grpc.max_receive_message_length` | 256 MB | Large model responses |
| `grpc.max_send_message_length` | 256 MB | Large input payloads |
| `grpc.keepalive_time_ms` | 30,000 | Keepalive ping interval |
| `grpc.keepalive_timeout_ms` | 10,000 | Keepalive ping timeout |
| `grpc.keepalive_permit_without_calls` | true | Keep channel alive between requests |
| `grpc.http2.max_pings_without_data` | 0 | Unlimited pings |

## Trace Data

The gRPC transport captures timing information in `GrpcTraceData`, which extends the base trace data model with gRPC-specific fields.

### gRPC-Specific Fields

| Field | Description |
|---|---|
| `grpc_status_code` | gRPC status code (0 = OK, 14 = UNAVAILABLE, etc.) |
| `grpc_status_message` | gRPC status message from the server |

### Inherited Base Fields (Export Format)

All base trace data timing fields are included in the export. Field names below are the exported JSON names (internally, the `BaseTraceData` model uses `_perf_ns` suffix, e.g., `request_send_start_perf_ns`):

| Field | Description |
|---|---|
| `request_send_start_ns` | When the request was sent |
| `response_receive_start_ns` | When the first response chunk arrived |
| `response_receive_end_ns` | When the last response chunk arrived |
| `request_chunks` | Array of `[timestamp_ns, size_bytes]` for each request |
| `response_chunks` | Array of `[timestamp_ns, size_bytes]` for each response chunk |
| `error_timestamp_ns` | When an error occurred (if any) |

### Enabling Trace Export

To include trace data in the export file:

```bash
aiperf profile ... --export-http-trace --export-level records
```

Example trace data in `profile_export.jsonl`:

```json
{
  "trace_data": {
    "trace_type": "grpc",
    "grpc_status_code": 0,
    "grpc_status_message": null,
    "request_send_start_ns": 1768309400341882300,
    "response_receive_start_ns": 1768309404815807889,
    "response_receive_end_ns": 1768309437763141384,
    "request_chunks": [[1768309400341882300, 1024]],
    "response_chunks": [
      [1768309404815807889, 256],
      [1768309405191294711, 128]
    ]
  }
}
```

## gRPC Status Code Mapping

gRPC uses its own status code system. AIPerf maps these to HTTP equivalents for consistent metrics reporting:

| gRPC Code | gRPC Name | HTTP Status | Meaning |
|---|---|---|---|
| 0 | OK | 200 | Success |
| 1 | CANCELLED | 499 | Client cancelled |
| 3 | INVALID_ARGUMENT | 400 | Bad request |
| 4 | DEADLINE_EXCEEDED | 504 | Timeout |
| 5 | NOT_FOUND | 404 | Model not found |
| 7 | PERMISSION_DENIED | 403 | Forbidden |
| 8 | RESOURCE_EXHAUSTED | 429 | Too many requests / out of memory |
| 12 | UNIMPLEMENTED | 501 | RPC not supported |
| 13 | INTERNAL | 500 | Server error |
| 14 | UNAVAILABLE | 503 | Server not ready |
| 16 | UNAUTHENTICATED | 401 | Missing/invalid credentials |

Unknown gRPC status codes default to HTTP 500.

## Request Cancellation

gRPC supports request cancellation with the same `--request-cancellation-*` options as HTTP:

```bash
aiperf profile \
    --model my-model \
    --url grpc://triton:8001 \
    --endpoint-type kserve_v2_infer \
    --streaming \
    --request-cancellation-rate 0.1 \
    --request-cancellation-delay 0.5 \
    --request-count 100
```

Cancelled requests use `asyncio.wait_for()` with the configured delay. When cancelled, the record's `error.code` is set to 499 (Client Cancelled). See [Request Cancellation Testing](./request-cancellation.md) for details.

## HTTP vs gRPC Comparison

When choosing between HTTP and gRPC for V2 inference:

| Feature | HTTP (`http://`) | gRPC (`grpc://`) |
|---|---|---|
| Streaming | No (V2 REST has no streaming format) | Yes (`ModelStreamInfer`) |
| TTFT / ITL metrics | Not available | Yes (with `--streaming`) |
| Connection efficiency | Connection pool | HTTP/2 multiplexing |
| Payload format | JSON | Protobuf (more compact) |
| TLS | `https://` | `grpcs://` |
| Default port (Triton) | 8000 | 8001 |
| Authentication | HTTP headers | gRPC metadata (from headers) |

**When to use gRPC:**

- You need streaming metrics (TTFT, ITL) for V2 inference
- Your model is deployed behind Triton's gRPC port
- You want HTTP/2 multiplexing efficiency
- Your payloads include large tensors (protobuf is more compact)

**When to use HTTP:**

- Your infrastructure only exposes HTTP
- You're behind a reverse proxy that doesn't support gRPC
- You want compatibility with standard HTTP debugging tools

## Troubleshooting

### Connection Refused

- Verify the gRPC port (Triton defaults to **8001**, not 8000)
- Check that the server is running and accepting gRPC connections
- Test connectivity: `grpcurl -plaintext triton:8001 inference.GRPCInferenceService/ServerReady`

### Model Not Found

- Ensure `--model` matches the exact model name loaded in the server
- Check model readiness: `grpcurl -plaintext triton:8001 inference.GRPCInferenceService/ModelReady`

### TLS Errors

- Verify the server has TLS enabled and the certificate is trusted
- For self-signed certificates, the gRPC channel uses the system's default CA bundle
- Test with: `grpcurl triton:8001 inference.GRPCInferenceService/ServerReady` (without `-plaintext`)

### Streaming Not Working

- Verify `--streaming` is passed on the command line
- Check that the server supports `ModelStreamInfer` (Triton Inference Server does)
- Without `--streaming`, gRPC uses the unary `ModelInfer` RPC (no TTFT/ITL metrics)

### Large Payload Errors

- Default max message size is 256 MB
- For models with extremely large tensor inputs/outputs, check server-side gRPC max message settings

## Related Documentation

- [KServe InferenceService Tutorial](./kserve.md) - Full KServe endpoint reference
- [HTTP Trace Metrics Guide](./http-trace-metrics.md) - HTTP trace timing reference (aiohttp transport)
- [Multi-URL Load Balancing](./multi-url-load-balancing.md) - Distributing requests across targets
- [Request Cancellation Testing](./request-cancellation.md) - Cancellation testing guide
- [Adding gRPC Endpoints](../dev/adding-grpc-endpoints.md) - Developer guide for adding new gRPC protocols
- [Source: grpc_transport.py](../../src/aiperf/transports/grpc/grpc_transport.py) - Generic transport implementation
- [Source: grpc_client.py](../../src/aiperf/transports/grpc/grpc_client.py) - Proto-free gRPC client
- [Source: kserve_v2_serializers.py](../../src/aiperf/transports/grpc/kserve_v2_serializers.py) - KServe V2 serializer
- [Source: payload_converter.py](../../src/aiperf/transports/grpc/payload_converter.py) - Dict/protobuf conversion
- [Source: status_mapping.py](../../src/aiperf/transports/grpc/status_mapping.py) - gRPC to HTTP status mapping
