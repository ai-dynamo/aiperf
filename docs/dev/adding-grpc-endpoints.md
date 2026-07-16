---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Add Native gRPC Endpoints
---

# Add Native gRPC Endpoints

Native gRPC extensions are compile-time Rust composition. Do not add a Python
transport plugin, `plugins.yaml` entry, closed `EndpointType` variant, global
registry, or dynamic-library ABI.

## Extension boundaries

One dialect crosses two independent open registries:

1. `aiperf_runtime::endpoints::EndpointFactory` prepares transport-neutral canonical
   request/response behavior and a co-located `EndpointDescriptor`.
2. `aiperf_runtime::transport_grpc::GrpcEndpointBindingFactory` prepares worker-local
   protobuf encoding, method paths, readiness encoding, and response decoding.

`GrpcTransport` owns channels, Clock deadlines, cancellation, metadata, status
mapping, and traces. It has no endpoint-specific match statement. The runner
resolves both dense tables during preparation; request loops dispatch by key.

## Reuse an existing OIP binding

If a new endpoint uses the existing KServe OIP messages and RPC paths, implement
only an endpoint factory and register its ID with an OIP binding factory. The
endpoint should produce the canonical V2 JSON tensor shape:

```json
{
  "inputs": [
    {"name": "INPUT", "shape": [1], "datatype": "BYTES", "data": ["hello"]}
  ],
  "parameters": {"temperature": 0.7}
}
```

Keep selectors and compiled templates in the prepared endpoint object. Never
branch on endpoint ID in the scheduler or transport.

## Add a different protobuf protocol

For a genuinely different service:

1. Check in the exact licensed `.proto` input and either checked-in Prost DTOs
   or a reproducible build that does not depend on an ambient `protoc`.
2. Implement a `GrpcEndpointBinding` for canonical JSON to protobuf and back.
3. Implement `GrpcEndpointBindingFactory::endpoint_ids` and `prepare`.
4. Register the factory transactionally with `GrpcBindingRegistryBuilder`.
5. Inject the frozen registry through
   `NativeGrpcExecutionBackendFactory::new` and
   `RunnerExecutionFactories::with_grpc` in the owning distribution.

The base AIPerf build registers only KServe V2 OIP. A distribution must not
advertise an endpoint/backend pair until both its endpoint and executable gRPC
binding factories are present.

## Required tests

- Pin endpoint payload and parsing parity against the complete source behavior.
- Decode native request bytes with the real protobuf type; use a byte-exact
  canary when deterministic encoding matters.
- Cover typed and raw tensor representations, malformed inputs, and in-band
  streaming errors.
- Start a real loopback Tonic service and exercise unary/streaming calls,
  metadata, status mapping, TLS policy where applicable, channel reuse,
  deadlines, and post-submit cancellation.
- Prove side-effect-free runner-v2 validation and a strict subprocess execute.
- Add a user-facing `aiperf profile --config ...` proof when the binding becomes
  product-reachable.

Every source file needs the project SPDX header, module docs, public-item docs,
and Clock-only application timing. Worker state remains local to a
current-thread runtime plus `LocalSet`; do not introduce work-stealing or a
shared hot-path lock.
