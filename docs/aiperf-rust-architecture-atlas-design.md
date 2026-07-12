<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Rust architecture atlas design

## Goal

Create three complementary canvas products that explain the implemented AIPerf
Rust port, its integration with the Python system, component communication, and
remaining parity gaps:

1. a six-canvas progressive-disclosure suite;
2. one unified architecture atlas;
3. a crate-by-crate maintainer reference.

The progressive-disclosure suite is delivered first.

## Audience modes

Every canvas uses one persistent audience selector. The selected audience changes
terminology, diagram labels, annotations, and evidence rather than merely showing
more text.

### Executive / cross-org

Presentation-ready visuals and basic language. Emphasize product value, ownership,
migration status, supported modes, and major risks. Omit internal trait names and
file paths.

### Developer

Explain subsystem contracts, lifecycle, communication, extension points, and
failure behavior. Provide enough detail to integrate with or review the system
without exposing every implementation scar.

### Core maintainer

Show exact crates, modules, types, ownership/threading models, protocol schemas,
feature gates, byte-parity logic, source anchors, compatibility scars, and
unresolved structural gaps. Target AIPerf and AI-Dynamo maintainers.

## Progressive-disclosure suite

1. **System ownership** — Python frontend, Rust runner, external systems, and
   legacy-only surfaces.
2. **One-run lifecycle** — Config v2 through capability preflight, protocol v2,
   execution, native reporting, and Python presentation.
3. **Execution modes** — clocks, scheduling, phases, adaptive controls, HTTP,
   gRPC, online mock, and feature-gated Dynamo offline simulation.
4. **Data and request shaping** — loaders, composition, segment storage,
   materialization, endpoint preparation, Graph-IR, exact token IDs, and content
   serving.
5. **Measurement and evaluation** — native metrics, sweeplines, telemetry,
   static accuracy, agentic evaluation, and provider-neutral evaluation.
6. **Parity ledger** — filterable built, conditional, compatibility-only,
   legacy-parallel, and unbuilt surfaces.

## Visual system

Use a restrained systems-instrument aesthetic based entirely on host theme
tokens. The signature element is a persistent handoff rail:

`Python authoring → runner boundary → execution seams → native report → Python presentation`

Diagrams are primary. Tables are reserved for the parity ledger and compact mode
comparisons. Avoid gradients, shadows, decorative color, giant text, emojis, and
walls of identical cards.

## Interaction model

- Persist the audience mode across all canvases.
- Filter by execution mode: `online_http`, `online_grpc`, `dynamo_offline`.
- Filter by implementation status: built, conditional, compatibility-only,
  legacy-parallel, or gap.
- Select components to reveal audience-appropriate details.
- Collapse secondary implementation evidence.
- Open exact repository source files from maintainer-level evidence.
- Preserve spatial context when changing audience modes.

## Accuracy rules

- Ground implementation claims in current source.
- Distinguish built, feature-gated, runtime-conditional, compatibility-only,
  legacy-parallel, and unbuilt.
- Treat specs as intent, never proof of implementation.
- Keep Python and Rust ownership explicit.
- Call out unsupported pairings and representation limits directly.
- Never imply semantic accuracy support for offline timing simulation.

## Responsive and accessibility requirements

Wide flow diagrams become vertically ordered stages on narrow viewports.
Interactive controls remain keyboard-accessible, labels remain visible without
color, and diagrams retain readable text at all supported widths.

## Verification

Each canvas must pass the Canvas TypeScript check. The finished suite must also
be reviewed for hierarchy, composition variety, forbidden visual patterns,
source-grounded claims, and audience-mode consistency.
