<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming dataset and shadow replay design

This directory is the normative design dossier for native Rust architectural
support for continuously produced datasets and time-shifted shadow replay.

The design is intentionally broader than any first integration:

- S3/NVCF is one streaming source adapter, not the streaming abstraction.
- `dynamo.request.trace.v1` is one decoder, not the streaming record model.
- Shadow replay is one event-time consumer, not the only streaming workload.
- Existing finite datasets and Graph-IR remain valid consumers of sealed input.

Documents:

- [scope-and-principles.md](scope-and-principles.md) defines the problem and
  non-negotiable architectural constraints.
- `current-rust-seams.md` inventories the native traits and ownership boundaries
  that the design should reuse or generalize.
- `architecture-options.md` compares candidate decompositions.
- `streaming-dataset-shadow-replay-design.md` is the normative design,
  including cross-chunk sessions and checkpointed result publication.
- `spec-review-record.md` records the immutable reviewed commit and approval
  gates.
- `implementation-plan-review-record.md` records the exact approved executable
  plan-set commit and its three independent approval gates.

Implementation and review records remain in this directory while the normative
design is implemented through the linked durable plan.
