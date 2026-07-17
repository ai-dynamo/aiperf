<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoint body construction

## Purpose

Define how an endpoint declares its request shape and how that shape becomes wire
bytes. The endpoint formatter is split from the byte materializer: `format_payload`
returns a declarative `BodyPlan` whose values may be segment handles, consumed by
two shared materializers chosen by wire type. This record owns "handles → wire
bytes"; segment storage and lowering belong to [dataset.md](dataset.md).

## Built

### `format_payload → BodyPlan`

`format_payload` returns a `BodyPlan` (`rust/runtime/src/body_plan.rs`) at
lowering — once per turn, with the run's endpoint known at config — never per
dispatch. Dispatch only materializes the plan (splice static and live segments,
fold in param overrides). A per-request `format_payload → Value → serialize` is
prohibited on the hot path.

```rust
pub enum BodyPlan {
    Raw(Handle),                          // degenerate whole-body case
    Fields(SmallVec<[(FieldName, FieldValue); 8]>),
}

pub enum FieldValue {
    Literal(Value),                       // endpoint-generated scalars/structs
    Segment(Handle),                      // one pre-serialized content segment
    Segments(SmallVec<[Handle; 1]>),      // an ordered array of interned handles
    Wires(SmallVec<[Bytes; 1]>),          // pre-serialized array not in the frozen store
}
```

The endpoint declares its shape with segment slots; it never touches
commas/brackets and never re-serializes content.

### The two materializers

The split is fundamental and made explicit rather than unified: protobuf
endpoints cannot splice pre-serialized JSON.

- **JSON (`transport::http`)** — the shared `JsonBodyMaterializer` walks the plan
  and concatenates literal bytes plus segment bytes from the frozen store into the
  single `Full<Bytes>`, splicing arbitrary named fields with zero content
  re-serialization. The single-`Full` buffer plus `SendCompletion` constraint is
  honored — no scatter-gather.
- **Protobuf (`transport::grpc`)** — packs token/tensor segments straight into the
  OIP `raw_input_contents` from structure, with no per-request `Value` and no
  per-element walk. Segments are storage the codec reads, not bytes it splices.

This is exactly the `transport::http` vs `transport::grpc` boundary. The endpoint
picks neither materializer. Live continuation content that is not in the frozen
store arrives as `Wires` and materializes the same way.

## Source anchors

- `rust/runtime/src/body_plan.rs` (`BodyPlan`, `FieldValue`,
  `JsonBodyMaterializer`).
- `rust/runtime/src/endpoints/registry.rs` (`format_payload` contract).
- `rust/runtime/src/dataset/materialize.rs` (splice primitives).
- `rust/runtime/src/transport/grpc/codec.rs` (protobuf encode from structure).
