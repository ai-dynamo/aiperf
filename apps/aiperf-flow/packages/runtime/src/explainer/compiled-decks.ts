// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AUTO-GENERATED: Compiled from .flow explainer source files.
 * Run: node scripts/compile-explainer-flows.mjs
 *
 * This file contains TypeScript representations of all .flow explainer decks,
 * compiled at build time to ensure byte-exact rendering from .flow source.
 */

import type { ExplainerDefinition } from './registry.js';

export const RUST_ARCHITECTURE_DECK: ExplainerDefinition = {
  id: 'rust-architecture',
  topic: 'system-architecture',
  slides: [
  {
    "id": "product-shell",
    "eyebrow": "Product shell",
    "title": "Product shell",
    "lede": "AIPerf ships as a single native `aiperf` executable from crate `aiperf-cli`. There is no separate runner process—the same binary parses public commands and re-executes itself for benchmark runs.",
    "narration": "AIPerf ships as one native aiperf binary. That same executable is both the public command line and the hidden execution engine.",
    "points": [
      "`rust/cli` builds the `aiperf` binary.",
      "Global allocator is mimalloc on the hot path.",
      "Internal modes are intercepted before normal CLI parsing."
    ],
    "caption": "One executable, two hats: operator CLI and execution child.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"product-shell-box\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n          children: [\n            {\n              id: \"binary-label\"\n              capability: \"core.text\"\n              text: \"aiperf binary\"\n              layout: { x: 60, y: 60, width: 380, height: 40 }\n              style: { fontSize: 20, fontWeight: \"bold\", fill: @theme.ink.primary }\n            },\n            {\n              id: \"cli-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 110, width: 160, height: 60 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"cli-text\"\n                  capability: \"core.text\"\n                  text: \"CLI Shell\"\n                  layout: { x: 70, y: 125, width: 140, height: 30 }\n                  style: { fontSize: 14, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"engine-box\"\n              capability: \"core.rect\"\n              layout: { x: 230, y: 110, width: 160, height: 60 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"engine-text\"\n                  capability: \"core.text\"\n                  text: \"Execution Engine\"\n                  layout: { x: 240, y: 125, width: 140, height: 30 }\n                  style: { fontSize: 14, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "workspace-map",
    "eyebrow": "Workspace map",
    "title": "Workspace map",
    "lede": "The Rust workspace is intentionally small. Capability flows cli → runtime → loadgen-core. Mock server and e2e are separate targets; pyext is packaging-only.",
    "narration": "The Rust workspace stays small. Capability flows from aiperf-cli into aiperf-runtime and then into loadgen-core.",
    "points": [
      "Members: loadgen-core, runtime, cli, mock-server, e2e, pyext.",
      "aiperf-cli depends on aiperf-runtime with the engine feature.",
      "aiperf-mock-server is a standalone benchmark target."
    ],
    "caption": "cli → runtime → loadgen-core",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"workspace-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"loadgen-core-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 190, width: 140, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"loadgen-text\"\n                  capability: \"core.text\"\n                  text: \"loadgen-core\"\n                  layout: { x: 70, y: 205, width: 120, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"runtime-box\"\n              capability: \"core.rect\"\n              layout: { x: 210, y: 130, width: 140, height: 50 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"runtime-text\"\n                  capability: \"core.text\"\n                  text: \"aiperf-runtime\"\n                  layout: { x: 220, y: 145, width: 120, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"cli-box-wm\"\n              capability: \"core.rect\"\n              layout: { x: 360, y: 70, width: 140, height: 50 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"cli-text-wm\"\n                  capability: \"core.text\"\n                  text: \"aiperf-cli\"\n                  layout: { x: 370, y: 85, width: 120, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"arrow1\"\n              capability: \"core.line\"\n              from: { x: 280, y: 165 }\n              to: { x: 350, y: 95 }\n              style: { stroke: @theme.ink.secondary, strokeWidth: 2 }\n            },\n            {\n              id: \"arrow2\"\n              capability: \"core.line\"\n              from: { x: 210, y: 160 }\n              to: { x: 140, y: 195 }\n              style: { stroke: @theme.ink.secondary, strokeWidth: 2 }\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "startup-order",
    "eyebrow": "Startup order",
    "title": "Startup order",
    "lede": "Every process starts the same way: initialize logging, check for hidden execution flags like `--execute`, `--cell`, or `--aggregator`, and only then route public subcommands.",
    "narration": "Startup always checks hidden execution modes first. Only after that does the process route public subcommands.",
    "points": [
      "`main.rs` calls `execute_mode::is_execution_mode`.",
      "Internal handlers never return to public dispatch.",
      "Everything else goes through `dispatch::run`."
    ],
    "caption": "Hidden modes short-circuit; public commands fall through.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"startup-timeline\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"init-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 60, width: 380, height: 40 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"init-text\"\n                  capability: \"core.text\"\n                  text: \"1. Initialize logging\"\n                  layout: { x: 70, y: 70, width: 360, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"check-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 110, width: 380, height: 40 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"check-text\"\n                  capability: \"core.text\"\n                  text: \"2. Check hidden modes (--execute, --cell, --aggregator)\"\n                  layout: { x: 70, y: 120, width: 360, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"dispatch-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 160, width: 380, height: 40 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"dispatch-text\"\n                  capability: \"core.text\"\n                  text: \"3. Dispatch public commands\"\n                  layout: { x: 70, y: 170, width: 360, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "command-surface",
    "eyebrow": "Command surface",
    "title": "Command surface",
    "lede": "`profile`, `config`, cellular roles, `slurm run`, and several utility commands are native. Most operational tooling still delegates to the Python package unless built with pyo3-embed.",
    "narration": "Core benchmark commands stay native in Rust. Most operational tooling still delegates to Python unless the build embeds it.",
    "points": [
      "`dispatch.rs` matches the first argv token.",
      "Unknown commands delegate to Python.",
      "Feature gates select gRPC, cellular, dynosim, parquet, and embed modes."
    ],
    "caption": "Benchmark hot path native; extended surface delegated.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"command-surface\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"native-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 60, width: 160, height: 180 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"native-title\"\n                  capability: \"core.text\"\n                  text: \"Native (Rust)\"\n                  layout: { x: 70, y: 70, width: 140, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"native-commands\"\n                  capability: \"core.text\"\n                  text: \"profile\\nconfig\\ncellular\\nslurm\"\n                  layout: { x: 70, y: 95, width: 140, height: 130 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"delegated-box\"\n              capability: \"core.rect\"\n              layout: { x: 230, y: 60, width: 160, height: 180 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"delegated-title\"\n                  capability: \"core.text\"\n                  text: \"Delegated (Python)\"\n                  layout: { x: 240, y: 70, width: 140, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"delegated-commands\"\n                  capability: \"core.text\"\n                  text: \"Operational\\ntooling\"\n                  layout: { x: 240, y: 110, width: 140, height: 110 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "configuration",
    "eyebrow": "Configuration",
    "title": "Configuration",
    "lede": "`aiperf profile --config` reads YAML, expands env vars and Jinja, applies CLI overrides, and materializes a strict `BenchmarkRun` object—the wire payload for execution.",
    "narration": "Profile reads Config v2, expands it, and resolves a strict BenchmarkRun object that describes the whole benchmark.",
    "points": [
      "`yaml.rs` handles substitution and alias normalization.",
      "`load.rs` and `model/` build the typed run object.",
      "`config init|validate|expand` are native helpers."
    ],
    "caption": "YAML in → validated BenchmarkRun out.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"config-pipeline\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"yaml-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 80, height: 50 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"yaml-text\"\n                  capability: \"core.text\"\n                  text: \"YAML\"\n                  layout: { x: 65, y: 85, width: 70, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"expand-box\"\n              capability: \"core.rect\"\n              layout: { x: 160, y: 70, width: 80, height: 50 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"expand-text\"\n                  capability: \"core.text\"\n                  text: \"Expand\"\n                  layout: { x: 165, y: 85, width: 70, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"validate-box\"\n              capability: \"core.rect\"\n              layout: { x: 260, y: 70, width: 80, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"validate-text\"\n                  capability: \"core.text\"\n                  text: \"Validate\"\n                  layout: { x: 265, y: 85, width: 70, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"result-box\"\n              capability: \"core.rect\"\n              layout: { x: 360, y: 70, width: 80, height: 50 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"result-text\"\n                  capability: \"core.text\"\n                  text: \"BenchmarkRun\"\n                  layout: { x: 365, y: 80, width: 70, height: 30 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "self-execution",
    "eyebrow": "Self execution",
    "title": "Self execution",
    "lede": "Each run is a fresh child of the same binary. Parent writes JSON to stdin; child returns one terminal JSONL envelope on stdout while stderr carries diagnostics.",
    "narration": "Each profile run spawns a fresh child of the same binary with aiperf execute over stdio.",
    "points": [
      "`execute::run_once` spawns `current_exe()` with hidden flags.",
      "Stdout stays reserved for protocol traffic.",
      "Panics in the child become typed failure envelopes."
    ],
    "caption": "Same binary, new process, protocol on stdio.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"stdio-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"parent-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 60, width: 160, height: 80 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"parent-text\"\n                  capability: \"core.text\"\n                  text: \"Parent Process\"\n                  layout: { x: 70, y: 80, width: 140, height: 20 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"arrow-down\"\n              capability: \"core.line\"\n              from: { x: 250, y: 100 }\n              to: { x: 250, y: 120 }\n              style: { stroke: @theme.ink.secondary, strokeWidth: 2 }\n            },\n            {\n              id: \"child-box\"\n              capability: \"core.rect\"\n              layout: { x: 170, y: 130, width: 160, height: 80 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"child-text\"\n                  capability: \"core.text\"\n                  text: \"Child Process\\n(--execute)\"\n                  layout: { x: 180, y: 155, width: 140, height: 30 }\n                  style: { fontSize: 12, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "wire-contract",
    "eyebrow": "Wire contract",
    "title": "Wire contract",
    "lede": "The child accepts protocol version 2 only. It composes a stock `Application` once, then `handle_v2` validates, prepares, executes, and persists results.",
    "narration": "Protocol version two wraps the run, composes Application once, and returns one terminal envelope.",
    "points": [
      "`protocol_v2.rs` defines strict authored types.",
      "Unknown fields fail closed at serde boundaries.",
      "Factory-owned config stays opaque until registry decode."
    ],
    "caption": "Envelope v2 in → terminal envelope out.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"protocol-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"input-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 100, height: 60 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"input-text\"\n                  capability: \"core.text\"\n                  text: \"Protocol v2 In\"\n                  layout: { x: 65, y: 90, width: 90, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"validate-box-p\"\n              capability: \"core.rect\"\n              layout: { x: 180, y: 70, width: 70, height: 60 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"validate-text-p\"\n                  capability: \"core.text\"\n                  text: \"Validate\"\n                  layout: { x: 185, y: 95, width: 60, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"execute-box-p\"\n              capability: \"core.rect\"\n              layout: { x: 270, y: 70, width: 70, height: 60 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"execute-text-p\"\n                  capability: \"core.text\"\n                  text: \"Execute\"\n                  layout: { x: 275, y: 95, width: 60, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"output-box\"\n              capability: \"core.rect\"\n              layout: { x: 360, y: 70, width: 90, height: 60 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"output-text\"\n                  capability: \"core.text\"\n                  text: \"Terminal Out\"\n                  layout: { x: 365, y: 90, width: 80, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "bootstrap",
    "eyebrow": "Bootstrap",
    "title": "Bootstrap",
    "lede": "At startup one registry is built transactionally from explicit extensions: loaders, samplers, endpoints, transports, workloads, exporters, and actuators. There is no runtime plugin scanning.",
    "narration": "At bootstrap, AIPerfRegistry freezes loaders, samplers, endpoints, transports, workloads, and exporters.",
    "points": [
      "Built-ins register HTTP, gRPC, dry_run, dynosim when features are on.",
      "Duplicate identifiers are rejected.",
      "Unknown transport or workload IDs fail closed."
    ],
    "caption": "Capabilities are frozen before the first request.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"registry-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"registry-box\"\n              capability: \"core.rect\"\n              layout: { x: 100, y: 60, width: 300, height: 200 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"registry-title\"\n                  capability: \"core.text\"\n                  text: \"AIPerfRegistry\"\n                  layout: { x: 110, y: 70, width: 280, height: 20 }\n                  style: { fontSize: 14, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"registry-contents\"\n                  capability: \"core.text\"\n                  text: \"Loaders • Samplers\\nEndpoints • Transports\\nWorkloads • Exporters\"\n                  layout: { x: 110, y: 100, width: 280, height: 140 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "composition-root",
    "eyebrow": "Composition root",
    "title": "Composition root",
    "lede": "`Coordinator::handle` is the orchestration spine: validate envelope and endpoint profiles, prepare dataset and sidecars, execute the selected program shape, write `native-v2.json`, then run exporters best-effort.",
    "narration": "Coordinator is the composition root: validate, prepare, execute, persist native-v2.json, then run exporters.",
    "points": [
      "Sidecar collectors start only on the primary cell when needed.",
      "Exporter failures are logged, not run-fatal.",
      "Cellular and graph runs branch inside the same coordinator."
    ],
    "caption": "One coordinator owns the whole run lifecycle.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"coordinator-flow\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"coord-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 60, width: 380, height: 170 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"coord-title\"\n                  capability: \"core.text\"\n                  text: \"Coordinator::handle\"\n                  layout: { x: 70, y: 70, width: 360, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"coord-steps\"\n                  capability: \"core.text\"\n                  text: \"1. Validate\\n2. Prepare\\n3. Execute\\n4. Persist native-v2.json\\n5. Run exporters\"\n                  layout: { x: 70, y: 100, width: 360, height: 120 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "time-seam",
    "eyebrow": "Time seam",
    "title": "Time seam",
    "lede": "Hot paths never call wall-clock APIs directly. `RealClock` drives online HTTP and gRPC; `SimClock` drives dynosim and graph co-simulation with deterministic ordering.",
    "narration": "All timing goes through Clock. RealClock drives online traffic; SimClock drives deterministic simulation.",
    "points": [
      "Measurement must not use `Instant::now` on hot paths.",
      "Virtual clocks expose integer-nanosecond time.",
      "Transport retry backoff is clock-driven too."
    ],
    "caption": "Real time online, virtual time in simulation.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"clock-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"realclock-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 160, height: 120 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"realclock-title\"\n                  capability: \"core.text\"\n                  text: \"RealClock\"\n                  layout: { x: 70, y: 80, width: 140, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"realclock-text\"\n                  capability: \"core.text\"\n                  text: \"Online HTTP/gRPC\"\n                  layout: { x: 70, y: 110, width: 140, height: 70 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"simclock-box\"\n              capability: \"core.rect\"\n              layout: { x: 230, y: 70, width: 160, height: 120 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"simclock-title\"\n                  capability: \"core.text\"\n                  text: \"SimClock\"\n                  layout: { x: 240, y: 80, width: 140, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"simclock-text\"\n                  capability: \"core.text\"\n                  text: \"Deterministic simulation\"\n                  layout: { x: 240, y: 110, width: 140, height: 70 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "inputs",
    "eyebrow": "Inputs",
    "title": "Inputs",
    "lede": "Loaders ingest files and traces, conversations intern into a BLAKE3-addressed segment pool, samplers choose turns, and materializers build wire payloads per endpoint dialect.",
    "narration": "Datasets load, sample, and materialize into endpoint-ready requests through one shared substrate.",
    "points": [
      "LoaderRegistry and SamplerRegistry live in the frozen registry.",
      "Synthetic, JSONL, graph, and trace formats share one substrate.",
      "Prompt materialization stays endpoint-aware."
    ],
    "caption": "Inputs become endpoint-ready requests once, then reuse handles.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"input-pipeline\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"load-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 85, height: 50 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"load-text\"\n                  capability: \"core.text\"\n                  text: \"Load\"\n                  layout: { x: 65, y: 85, width: 75, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"pool-box\"\n              capability: \"core.rect\"\n              layout: { x: 160, y: 70, width: 85, height: 50 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"pool-text\"\n                  capability: \"core.text\"\n                  text: \"Segment Pool\"\n                  layout: { x: 165, y: 85, width: 75, height: 20 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"sample-box\"\n              capability: \"core.rect\"\n              layout: { x: 260, y: 70, width: 85, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"sample-text\"\n                  capability: \"core.text\"\n                  text: \"Sample\"\n                  layout: { x: 265, y: 85, width: 75, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"material-box\"\n              capability: \"core.rect\"\n              layout: { x: 360, y: 70, width: 85, height: 50 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"material-text\"\n                  capability: \"core.text\"\n                  text: \"Materialize\"\n                  layout: { x: 365, y: 80, width: 75, height: 30 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "work-generation",
    "eyebrow": "Work generation",
    "title": "Work generation",
    "lede": "Scheduled workloads—request rate, concurrency, user-centric, fixed schedule—emit turn schedules. Phase runtime applies warmup, profiling, ramp, drain, and cancellation policy before transport ever sends.",
    "narration": "Workloads and phase runtime decide when turns fire. Transports only send and observe the wire path.",
    "points": [
      "Graph programs use the separate DAG executor.",
      "`scheduled.rs` bridges Workload to TurnDispatcher.",
      "Cancellation and grace propagate through shared phase code."
    ],
    "caption": "Scheduling decides when; transport decides how to send.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"workload-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"workload-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 140, height: 110 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"workload-title\"\n                  capability: \"core.text\"\n                  text: \"Workload\"\n                  layout: { x: 70, y: 80, width: 120, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"workload-types\"\n                  capability: \"core.text\"\n                  text: \"Request-rate\\nConcurrency\\nUser-centric\"\n                  layout: { x: 70, y: 110, width: 120, height: 60 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"phase-box\"\n              capability: \"core.rect\"\n              layout: { x: 220, y: 70, width: 140, height: 110 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"phase-title\"\n                  capability: \"core.text\"\n                  text: \"Phase Runtime\"\n                  layout: { x: 230, y: 80, width: 120, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"phase-types\"\n                  capability: \"core.text\"\n                  text: \"Warmup\\nRamp\\nDrain\"\n                  layout: { x: 230, y: 110, width: 120, height: 60 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "observation-seam",
    "eyebrow": "Observation seam",
    "title": "Observation seam",
    "lede": "Transports implement `RequestSink`; observers receive arrival, admission, token, usage, and terminal events. TTFT is the first token observation—there is no separate first-token event.",
    "narration": "loadgen-core defines the neutral observer seam. Transports implement RequestSink; measurement stays shared.",
    "points": [
      "HTTP and gRPC share reduction in `transport::reduce`.",
      "Measurement accumulates in `transport::measure`.",
      "Endpoint usage is captured verbatim when present."
    ],
    "caption": "Transport sends; observer measures; sink owns the lifecycle.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"observation-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 200 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"sink-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 60, width: 110, height: 150 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"sink-text\"\n                  capability: \"core.text\"\n                  text: \"RequestSink\"\n                  layout: { x: 65, y: 75, width: 100, height: 120 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"observer-box\"\n              capability: \"core.rect\"\n              layout: { x: 180, y: 60, width: 110, height: 150 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"observer-text\"\n                  capability: \"core.text\"\n                  text: \"RequestObserver\"\n                  layout: { x: 185, y: 75, width: 100, height: 120 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"measure-box\"\n              capability: \"core.rect\"\n              layout: { x: 300, y: 60, width: 110, height: 150 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"measure-text\"\n                  capability: \"core.text\"\n                  text: \"Measurement\"\n                  layout: { x: 305, y: 75, width: 100, height: 120 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "parallelism",
    "eyebrow": "Parallelism",
    "title": "Parallelism",
    "lede": "When `workers == 1`, scheduler and sink co-locate on a current-thread runtime. When `workers > 1`, each OS thread runs a self-contained sub-cell with its own LocalSet and transport sink.",
    "narration": "Parallelism is thread-per-core. Each worker owns local scheduling, transport, and capture without hot-path mutexes.",
    "points": [
      "Read-only Arc data crosses thread boundaries.",
      "Per-worker state stays local and merges at boundaries.",
      "Hyper and SSE paths rely on LocalSet because clients are not Send."
    ],
    "caption": "Scale out by adding worker threads, not shared mutable hot state.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"parallelism-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"worker1-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 100, height: 160 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"worker1-title\"\n                  capability: \"core.text\"\n                  text: \"Worker 1\"\n                  layout: { x: 65, y: 80, width: 90, height: 20 }\n                  style: { fontSize: 11, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"worker1-content\"\n                  capability: \"core.text\"\n                  text: \"LocalSet\\nTransport\\nMeasure\"\n                  layout: { x: 65, y: 110, width: 90, height: 110 }\n                  style: { fontSize: 9, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"worker2-box\"\n              capability: \"core.rect\"\n              layout: { x: 170, y: 70, width: 100, height: 160 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"worker2-title\"\n                  capability: \"core.text\"\n                  text: \"Worker 2\"\n                  layout: { x: 175, y: 80, width: 90, height: 20 }\n                  style: { fontSize: 11, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"worker2-content\"\n                  capability: \"core.text\"\n                  text: \"LocalSet\\nTransport\\nMeasure\"\n                  layout: { x: 175, y: 110, width: 90, height: 110 }\n                  style: { fontSize: 9, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"worker3-box\"\n              capability: \"core.rect\"\n              layout: { x: 280, y: 70, width: 100, height: 160 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"worker3-title\"\n                  capability: \"core.text\"\n                  text: \"Worker N\"\n                  layout: { x: 285, y: 80, width: 90, height: 20 }\n                  style: { fontSize: 11, fontWeight: \"bold\", fill: @theme.ink.primary }\n                },\n                {\n                  id: \"worker3-content\"\n                  capability: \"core.text\"\n                  text: \"LocalSet\\nTransport\\nMeasure\"\n                  layout: { x: 285, y: 110, width: 90, height: 110 }\n                  style: { fontSize: 9, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "scale-out",
    "eyebrow": "Scale-out",
    "title": "Scale-out",
    "lede": "`--cells N` promotes the run to a controller that partitions work, launches `aiperf --cell` children, merges records or folded stores over Velo, and writes one authoritative report. SLURM and Kubernetes reuse the same model.",
    "narration": "Cellular mode adds a controller, remote cells, Velo control traffic, and a single merged report.",
    "points": [
      "Rank 0 or the controller owns merge and artifact shipping.",
      "Velo carries control only; HTTP and gRPC carry measured load.",
      "Bulk artifacts can use a separate HTTP/1 + zstd path."
    ],
    "caption": "Same engine, more processes, partitioned ownership.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"cellular-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"controller-box\"\n              capability: \"core.rect\"\n              layout: { x: 150, y: 60, width: 120, height: 70 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"controller-text\"\n                  capability: \"core.text\"\n                  text: \"Controller\"\n                  layout: { x: 160, y: 85, width: 100, height: 20 }\n                  style: { fontSize: 12, fontWeight: \"bold\", fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"cell1-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 160, width: 80, height: 70 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"cell1-text\"\n                  capability: \"core.text\"\n                  text: \"Cell 0\"\n                  layout: { x: 65, y: 185, width: 70, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"cell2-box\"\n              capability: \"core.rect\"\n              layout: { x: 160, y: 160, width: 80, height: 70 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"cell2-text\"\n                  capability: \"core.text\"\n                  text: \"Cell 1\"\n                  layout: { x: 165, y: 185, width: 70, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"celln-box\"\n              capability: \"core.rect\"\n              layout: { x: 260, y: 160, width: 80, height: 70 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"celln-text\"\n                  capability: \"core.text\"\n                  text: \"Cell N\"\n                  layout: { x: 265, y: 185, width: 70, height: 20 }\n                  style: { fontSize: 11, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  },
  {
    "id": "outputs--gates",
    "eyebrow": "Outputs & gates",
    "title": "Outputs & gates",
    "lede": "Workers accumulate metrics locally; the report plane writes `native-v2.json` as authoritative truth, then exporters emit JSON, CSV, Parquet, OTLP, and more. Cargo features gate gRPC, cellular, parquet, dynosim, and pyo3-embed.",
    "narration": "Metrics merge into native-v2.json, exporters fan out artifacts, and Cargo features opt into gRPC, cellular, dynosim, and more.",
    "points": [
      "Sketch mode uses mergeable t-digests for approximate percentiles.",
      "`--steady-state` derives a measurement window from concurrency crossings.",
      "Mock server stays outside profile supervision for deterministic tests."
    ],
    "caption": "Measure locally, merge once, export many formats.",
    "sceneBlock": "{\n      roots: [\n        {\n          id: \"output-diagram\"\n          capability: \"core.rect\"\n          layout: { x: 50, y: 50, width: 400, height: 250 }\n          style: { fill: @theme.surface.primary }\n          children: [\n            {\n              id: \"metrics-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 70, width: 120, height: 60 }\n              style: { fill: @theme.accent.primary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"metrics-text\"\n                  capability: \"core.text\"\n                  text: \"Metrics\"\n                  layout: { x: 70, y: 90, width: 100, height: 20 }\n                  style: { fontSize: 11, fontWeight: \"bold\", fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"report-box\"\n              capability: \"core.rect\"\n              layout: { x: 200, y: 70, width: 140, height: 60 }\n              style: { fill: @theme.accent.secondary, stroke: @theme.ink.secondary, strokeWidth: 2 }\n              children: [\n                {\n                  id: \"report-text\"\n                  capability: \"core.text\"\n                  text: \"native-v2.json\"\n                  layout: { x: 210, y: 90, width: 120, height: 20 }\n                  style: { fontSize: 11, fontWeight: \"bold\", fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"export1-box\"\n              capability: \"core.rect\"\n              layout: { x: 60, y: 160, width: 80, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"export1-text\"\n                  capability: \"core.text\"\n                  text: \"JSON\"\n                  layout: { x: 65, y: 175, width: 70, height: 20 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"export2-box\"\n              capability: \"core.rect\"\n              layout: { x: 160, y: 160, width: 80, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"export2-text\"\n                  capability: \"core.text\"\n                  text: \"CSV\"\n                  layout: { x: 165, y: 175, width: 70, height: 20 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"export3-box\"\n              capability: \"core.rect\"\n              layout: { x: 260, y: 160, width: 80, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"export3-text\"\n                  capability: \"core.text\"\n                  text: \"Parquet\"\n                  layout: { x: 265, y: 175, width: 70, height: 20 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            },\n            {\n              id: \"export4-box\"\n              capability: \"core.rect\"\n              layout: { x: 360, y: 160, width: 80, height: 50 }\n              style: { fill: @theme.accent.tertiary, stroke: @theme.ink.secondary, strokeWidth: 1 }\n              children: [\n                {\n                  id: \"export4-text\"\n                  capability: \"core.text\"\n                  text: \"OTLP\"\n                  layout: { x: 365, y: 175, width: 70, height: 20 }\n                  style: { fontSize: 10, fill: @theme.ink.primary }\n                }\n              ]\n            }\n          ]\n        }\n      ]\n    }"
  }
],
  glossary: [],
};

export const SLURM_VELO_DECK: ExplainerDefinition = {
  id: 'slurm-velo',
  topic: 'distributed-execution',
  slides: [
  {
    "id": "you-want-to-load-test-a-big-ai-server",
    "eyebrow": "The problem",
    "title": "You want to load-test a big AI server",
    "lede": "AIPerf sends many requests to an inference server and measures how fast it answers. To push a really large server hard, one computer sending traffic is not enough — you need many computers sending at once.",
    "narration": "Large AI servers need many load generators acting together as one benchmark.",
    "points": [
      "One laptop can only send so many requests per second.",
      "To stress a large server you need a fleet of machines generating load together.",
      "That fleet has to act like one coordinated test, not many disconnected ones."
    ],
    "caption": "Goal: many machines, one benchmark, one result."
  },
  {
    "id": "slurm-hands-you-a-cluster-of-machines",
    "eyebrow": "The tool",
    "title": "SLURM hands you a cluster of machines",
    "lede": "SLURM is the software that shares a big cluster among many people. You ask it for machines; it finds free ones, reserves them for you, and runs your command on every one of them at the same time.",
    "narration": "SLURM reserves cluster machines and launches your program across all of them.",
    "points": [
      "`sbatch` submits a batch job; `srun` launches tasks right now.",
      "An allocation is your reserved set of machines for this job.",
      "SLURM runs the exact same command on every task in the allocation."
    ],
    "caption": "SLURM = the landlord that lends you machines for a while."
  },
  {
    "id": "every-machine-runs-the-identical-command",
    "eyebrow": "The key trick",
    "title": "Every machine runs the identical command",
    "lede": "This is the part that surprises newcomers. SLURM does not run a different program on each machine. It launches the very same line — `aiperf slurm run` — on all of them at once.",
    "narration": "Every task runs the same AIPerf command, then its rank determines its role.",
    "points": [
      "So how does each copy know what to do differently?",
      "SLURM gives each task a numbered identity called its rank.",
      "The program reads that number and decides its own job."
    ],
    "caption": "Same command everywhere — the rank number breaks the tie."
  },
  {
    "id": "rank-0-leads-everyone-else-does-the-work",
    "eyebrow": "Splitting the roles",
    "title": "Rank 0 leads; everyone else does the work",
    "lede": "AIPerf reads the rank SLURM assigned. Rank 0 becomes the controller — the coordinator. Every other rank becomes a cell — a worker that actually sends load to the server.",
    "narration": "Rank zero coordinates the benchmark. Every other rank becomes a load-generating cell.",
    "points": [
      "Rank 0 → the single controller (it coordinates, it does not send benchmark load).",
      "Ranks 1, 2, 3, … → cells, numbered cell_id = rank − 1.",
      "So a 4-task job = 1 controller + 3 cells."
    ],
    "caption": "controller = rank 0 · cell_id = rank − 1 · cell_count = tasks − 1."
  },
  {
    "id": "cells-dial-the-controller-with-one-shared-fact",
    "eyebrow": "Finding each other",
    "title": "Cells dial the controller with one shared fact",
    "lede": "The cells need to talk to the controller, but nobody set up a directory service. Instead, every task computes the same address from the SLURM environment: the first machine in the allocation, on a known port.",
    "narration": "Each cell derives the rank-zero controller address from the shared SLURM allocation.",
    "points": [
      "SLURM tells every task the list of machines and its own rank.",
      "All tasks agree rank 0 lives on the first machine in that list.",
      "Default connection port is 9500, so every cell knows exactly where to call."
    ],
    "caption": "One fact, computed the same everywhere — no discovery service needed."
  },
  {
    "id": "velo-is-the-walkie-talkie-between-controller-and-cells",
    "eyebrow": "Meet Velo",
    "title": "Velo is the walkie-talkie between controller and cells",
    "lede": "Once cells know the controller address, they still need a messaging system. That system is Velo: a small control-plane library AIPerf uses so the controller and cells can exchange short messages across machines.",
    "narration": "Velo carries coordination messages between the controller and its remote cells.",
    "points": [
      "Velo is how AIPerf processes talk to each other across hosts.",
      "It carries small control messages, not the benchmark requests themselves.",
      "Without cellular mode (`--cells`), Velo is not constructed at all."
    ],
    "caption": "SLURM launches the processes. Velo lets those processes talk."
  },
  {
    "id": "a-cell-connects-once-then-velo-learns-the-peer",
    "eyebrow": "Velo bootstrap",
    "title": "A cell connects once, then Velo learns the peer",
    "lede": "Each cell calls Velo with the one known address: tcp://HOST:9500. Velo's hello handshake discovers the controller's real peer identity, then the two sides can send named messages to each other.",
    "narration": "Each cell connects once, allowing Velo to establish the peer relationship.",
    "points": [
      "The cell only needs AIPERF_CELL_CONTROLLER_ADDR.",
      "Velo connects, shakes hands, and registers both peers.",
      "After that, named handlers like register / heartbeat / partition work."
    ],
    "caption": "Address in → peer connection out. No service discovery backend."
  },
  {
    "id": "register-and-start-travel-over-velo",
    "eyebrow": "Getting ready together",
    "title": "Register and START travel over Velo",
    "lede": "Before any traffic flies, each cell registers over Velo. The controller replies with that cell's work slice and a START event handle. When every expected cell has registered, the controller triggers START and they begin together.",
    "narration": "Cells register through Velo, then wait until the controller broadcasts START.",
    "points": [
      "aiperf.cell.register — cell joins and receives its sliced envelope.",
      "The controller waits until cell_count cells have registered.",
      "One START trigger releases every waiting cell at once."
    ],
    "caption": "Line everyone up over Velo, then start the race together."
  },
  {
    "id": "benchmark-requests-do-not-use-velo",
    "eyebrow": "Doing the work",
    "title": "Benchmark requests do NOT use Velo",
    "lede": "After START, cells send their share of requests straight to the inference server over HTTP or gRPC. Velo stays out of that path — it only carries lightweight heartbeats back to the controller.",
    "narration": "Benchmark requests travel directly from cells to the inference server, never through Velo.",
    "points": [
      "Each cell sends only its assigned slice — no overlap, no gaps.",
      "Request traffic goes cell → inference server directly.",
      "The controller is not a bottleneck: it never sits in the request path."
    ],
    "caption": "Velo coordinates. HTTP/gRPC generates the measured load."
  },
  {
    "id": "three-completely-different-kinds-of-traffic",
    "eyebrow": "Three planes",
    "title": "Three completely different kinds of traffic",
    "lede": "It helps to keep three roads separate in your head. Velo is only the control road. The load road hits the AI server. Large result files take a third bulk-upload road.",
    "narration": "Control, benchmark traffic, and bulk artifacts use three deliberately separate paths.",
    "points": [
      "Velo — register, START, heartbeats, result partitions / stores.",
      "HTTP / gRPC — the real benchmark requests to the inference server.",
      "HTTP/1 + zstd — large per-record artifact files, not carried on Velo."
    ],
    "caption": "Mixing these up is the main source of confusion."
  },
  {
    "id": "result-partitions-return-to-rank-0-over-velo",
    "eyebrow": "One answer",
    "title": "Result partitions return to rank 0 over Velo",
    "lede": "When the run ends, each cell ships its measurements back to the original rank-0 controller over Velo. The controller merges every cell's numbers into one report.",
    "narration": "When work finishes, each cell returns its result partition to rank zero over Velo.",
    "points": [
      "aiperf.cell.partition or store_partition — one terminal ship per cell.",
      "Merge happens inside the original rank-0 controller process.",
      "The report looks like one benchmark, not N separate jobs."
    ],
    "caption": "Cells measure. Rank 0 merges. One authoritative report."
  },
  {
    "id": "huge-per-record-files-take-a-different-road",
    "eyebrow": "Bulk files",
    "title": "Huge per-record files take a different road",
    "lede": "If the run keeps large per-record artifact files, those bytes do not ride Velo. They upload over a separate HTTP/1 path with zstd compression. The controller concatenates them after every cell finishes.",
    "narration": "Large per-request artifacts use compressed HTTP instead of crowding the control plane.",
    "points": [
      "Velo stays small: control messages and metric summaries.",
      "Bulk files use HTTP/1 + zstd so they do not clog the control plane.",
      "Synthetic / summary-only runs may never need this path."
    ],
    "caption": "Small control on Velo. Big files on HTTP."
  },
  {
    "id": "why-spend-a-whole-rank-on-a-non-loading-process",
    "eyebrow": "Controller cost",
    "title": "Why spend a whole rank on a non-loading process?",
    "lede": "A dedicated controller rank keeps coordination, START sync, merging, and artifact handling away from measured load. That does not always mean a dedicated node — only a dedicated role.",
    "narration": "A dedicated controller rank keeps coordination responsive while cells generate maximum load.",
    "points": [
      "Dedicated role: yes — keeps measurement clean and merge simple.",
      "Dedicated node: optional — the script defaults to one task per node.",
      "Co-locate controller + cell when machine count is scarce."
    ],
    "caption": "Pay for coordination. Do not always pay for a whole idle node."
  },
  {
    "id": "rank-0-fans-distinct-work-slices-out-to-the-cells",
    "eyebrow": "Fan-out",
    "title": "Rank 0 fans distinct work slices out to the cells",
    "lede": "The controller begins with one global benchmark plan. As each cell registers over Velo, rank 0 replies with that cell's sliced envelope so the cells divide the work without overlap or omissions.",
    "narration": "The controller partitions one global plan into distinct slices and fans them out together.",
    "points": [
      "The controller owns the one resolved Config v2 benchmark plan.",
      "Cell 0 gets slice 0, cell 1 gets slice 1, and so on.",
      "Together the slices tile the global request or conversation budget exactly."
    ],
    "caption": "One benchmark plan fans out into disjoint cell-owned slices."
  },
  {
    "id": "cells-fan-their-finished-results-back-into-rank-0",
    "eyebrow": "Fan-in",
    "title": "Cells fan their finished results back into rank 0",
    "lede": "At the end, direction reverses. Every cell sends exactly one terminal partition or folded store over Velo. Rank 0 collects all expected children and merges them into the global result.",
    "narration": "Cells return their completed slices in parallel, and rank zero merges one final report.",
    "points": [
      "Each cell ships one result partition or one folded metric store.",
      "Rank 0 waits for every expected cell before completing the merge.",
      "The merged output becomes one authoritative AIPerf report."
    ],
    "caption": "Many cell results fan in to the original rank-0 controller."
  },
  {
    "id": "the-two-commands-you-actually-type",
    "eyebrow": "Try it",
    "title": "The two commands you actually type",
    "lede": "You rarely wire this up by hand. AIPerf generates a ready-to-submit SLURM script for you, and a single command runs the whole cellular benchmark inside the allocation.",
    "narration": "Generate the batch script, submit it, and AIPerf handles ranks, Velo, load, and results.",
    "points": [
      "Generate a submission script from your benchmark config.",
      "Submit it with SLURM; every task launches the same run command.",
      "Rank assignment, Velo wiring, and merging all happen automatically."
    ],
    "caption": "You describe the benchmark; AIPerf handles the cluster choreography."
  }
],
  glossary: [],
};

export const DYNOSIM_DECK: ExplainerDefinition = {
  id: 'dynosim',
  topic: 'simulation',
  slides: [],
  glossary: [],
};

export const AIPERF_FLOW_SYSTEM_DECK: ExplainerDefinition = {
  id: 'aiperf-flow-system',
  topic: 'flow-system',
  slides: [
  {
    "id": "understanding-aiperf-flow",
    "eyebrow": "Module 1",
    "title": "Understanding AIPerf Flow",
    "lede": "A clock-aware load generator with integrated visualization and measurement",
    "narration": "AIPerf Flow is a native Rust load generator and measurement system for inference servers. It combines protocol-aware request dispatch, clock-driven scheduling, and real-time measurement into a single integrated runtime. The Flow visualization system maps this execution model into interactive diagrams that teach the request lifecycle and inference topology.",
    "points": [
      "Native Rust CLI for load generation and profiling",
      "Clock-aware scheduling with deterministic simulation support",
      "Protocol-neutral transport abstraction over HTTP and gRPC",
      "Integrated visualization of request journeys and system topology"
    ],
    "caption": "AIPerf unifies load generation, measurement, and visualization of inference systems"
  },
  {
    "id": "the-request-lifecycle",
    "eyebrow": "Module 2",
    "title": "The Request Lifecycle",
    "lede": "A single request crosses admission, transport, model, stream, and observation boundaries",
    "narration": "Every request passes through seven distinct lifecycle boundaries: arrival at the scheduler, admission through a clock-aware queue, dispatch to the transport layer, service beginning in the model, first token emission, terminal stream event, and final observer record. Each boundary represents a causal and temporal milestone that the measurement system captures with stable evidence identifiers.",
    "points": [
      "Arrival: scheduler receives the request at wall or virtual time",
      "Admission: clock-aware queue decides to begin service",
      "Dispatch: transport serializes and sends the request",
      "First token: response stream emits its first output token",
      "Terminal: stream closes without error or signals completion",
      "Record: observer finalizes measurement and persists the evidence"
    ],
    "caption": "Request lifecycle evidence flows from arrival through observer finalization"
  },
  {
    "id": "clock-aware-admission",
    "eyebrow": "Module 3",
    "title": "Clock-Aware Admission",
    "lede": "A clock-driven admission boundary that enforces request-rate and concurrency policies",
    "narration": "The admission queue is the first clock-aware component in the execution pipeline. It receives requests from the client scheduler, applies admission policy based on the configured workload, and dispatches admitted requests to the transport layer at the correct clock time. The queue is policy-agnostic: it can enforce fixed request rates, target concurrency levels, or multi-turn user-centric conversations. Arrival and admission timestamps are captured as stable evidence.",
    "points": [
      "Receives requests from the client scheduler",
      "Enforces request-rate, concurrency, or user-centric policies",
      "Dispatches admitted requests to the transport at scheduled time",
      "Preserves arrival and admission timestamps as evidence",
      "Supports both real-wall-time and deterministic simulation modes"
    ],
    "caption": "Clock-aware admission enforces scheduling policy while preserving causality"
  },
  {
    "id": "transport-and-protocol-binding",
    "eyebrow": "Module 4",
    "title": "Transport and Protocol Binding",
    "lede": "HTTP and gRPC endpoints bound through pluggable transport abstractions",
    "narration": "The transport layer dispatches admitted requests to inference endpoints through protocol-specific implementations. HTTP transport supports HTTP/1, h2c, UDS, TLS, and SSE streaming. gRPC transport supports KServe OIP and NVIDIA Riva endpoint families. Connection establishment includes clock-driven retry logic with configurable backoff. The transport is transport-neutral to the dispatcher: both HTTP and gRPC implement a common sink interface that receives requests and reports response events.",
    "points": [
      "HTTP: HTTP/1, h2c, UDS, TLS, Server-Sent Events",
      "gRPC: KServe OIP, NVIDIA Riva ASR, TTS, NLP families",
      "Connection pooling with clock-driven linear backoff retry",
      "Streaming response handling with token-by-token observation",
      "Request-local and worker-local state isolation"
    ],
    "caption": "Transport abstractions decouple protocol details from request lifecycle"
  },
  {
    "id": "worker-local-request-execution",
    "eyebrow": "Module 5",
    "title": "Worker-Local Request Execution",
    "lede": "Thread-per-core execution model with request-local state isolation",
    "narration": "The execution model is thread-per-core: a single worker owns its own scheduling, admission, transport, capture, and measurement state. This design eliminates per-request allocation overhead and contention in the critical measurement path. One-worker deployments use Tokio's LocalSet on the coordinator's current-thread runtime. Multi-worker runs spawn each worker on its own OS thread, each with a self-contained runtime. A request sink is the worker-local handler that executes one request: it serializes the request, sends it to the endpoint, observes the response stream, and participates in measurement.",
    "points": [
      "Thread-per-core execution with local state ownership",
      "Single-worker LocalSet on current-thread Tokio runtime",
      "Multi-worker OS threads with independent scheduling",
      "Worker-local sink owns transport, capture, and measurement",
      "No Arc<Mutex<_>> contention on request or token paths"
    ],
    "caption": "Worker-local architecture eliminates measurement overhead from allocation and contention"
  },
  {
    "id": "stream-observation-and-evidence-capture",
    "eyebrow": "Module 6",
    "title": "Stream Observation and Evidence Capture",
    "lede": "Streaming response observation with first-token and terminal event separation",
    "narration": "The observer receives events from the response stream and records them as stable evidence. First-token emission is distinct from stream terminal: a response that produces zero tokens has a terminal event but no first-token event. Token arrivals are observed but not recorded individually; instead, the observer accumulates generated-token count and measures inter-token latency from first to terminal. When the stream closes, the observer completes measurement, finalizes usage and accuracy data, and emits the final record to persistent storage. Measurement data includes input and output token counts, first-token latency, generated-token inter-token latency, and endpoint-specific usage observations.",
    "points": [
      "First-token and terminal events are observed as separate boundaries",
      "Token stream is preserved as bytes until complete lines available",
      "Generated-token latency measured from first to terminal event",
      "Token counts captured from endpoint usage or local tokenizer",
      "Accuracy and adaptive control observe inference output",
      "Final record emitted after terminal event and observer finalization"
    ],
    "caption": "Token-by-token observation preserves first-token, streaming, and terminal semantics"
  },
  {
    "id": "measurement-and-metrics-plane",
    "eyebrow": "Module 7",
    "title": "Measurement and Metrics Plane",
    "lede": "Record, aggregate, and phase-window metrics derived from stable evidence",
    "narration": "The measurement system operates at four levels: per-record measurement captures every request with its evidence timestamps and inference outputs; aggregation computes summary statistics over a batch of records; phase metrics apply window-based constraints from the workload lifecycle like warmup, ramp, steady-state, and drain; sweep metrics compute derived histograms, percentiles, and rates over configured dimensions. Exact mode retains all records; sketch mode uses mergeable t-digests for throughput and percentile estimates. Steady-state mode derives a measurement window from the in-flight concurrency curve. Every metric is grounded in the stable evidence timestamps, making them reproducible across replays and deterministic simulation.",
    "points": [
      "Per-record measurement with evidence-backed timestamps",
      "Exact and sketch-mode aggregation",
      "Phase-window metrics: warmup, ramp, steady-state, drain",
      "Sweep metrics over configured dimensions",
      "Steady-state window derived from concurrency curve",
      "Reproducible across real and simulated runs"
    ],
    "caption": "Evidence-backed measurement ensures reproducibility and correctness"
  },
  {
    "id": "flow-visualization-pipeline",
    "eyebrow": "Module 8",
    "title": "Flow Visualization Pipeline",
    "lede": "Real-time visualization of request topology and lifecycle causality",
    "narration": "The Flow visualization system renders the request lifecycle and system topology as interactive diagrams. A Flow document defines scenes that arrange stable entities like queues, transports, models, and streams spatially, then connects them with relations showing how requests flow. A scene narrates the spatial arrangement, then a timeline orchestrates camera movements, element reveals, and connection traces that teach the lifecycle progression. Interactive selections let viewers inspect entity details, evidence metadata, and causal relations. Responsive layouts adapt to viewport size. Fallback modes preserve narration, reading order, and semantic meaning for SVG and HTML-only contexts.",
    "points": [
      "Scenes define spatial topology with stable entities and relations",
      "Timelines choreograph camera, reveals, traces, and narration",
      "Interactive inspection of entities and evidence",
      "Reading order, keyboard navigation, accessibility roles preserved",
      "Reduced-motion support and semantic fallback for non-interactive contexts",
      "Evidence-backed narration tied to lifecycle events"
    ],
    "caption": "Flow visualization teaches request lifecycle through interactive spatial diagrams"
  },
  {
    "id": "integrating-aiperf-and-flow",
    "eyebrow": "Module 9",
    "title": "Integrating AIPerf and Flow",
    "lede": "AIPerf execution model instrumented through Flow visualization",
    "narration": "AIPerf Flow integrates load generation, measurement, and visualization into a single system. The AIPerf runtime captures evidence-backed measurement from the request lifecycle. The Flow compiler transforms .flow source into interactive scenes. At runtime, the explainer deck integrates the two: each slide corresponds to a phase or concept in the request lifecycle, with optional embedded scenes that visualize the current topic. Narration is coupled to timeline progression, allowing viewers to see request flow animations while hearing description of the causality and measurement semantics. The integration supports both browser-driven interactive learning and deterministic offline simulation and replay through Dynamo.",
    "points": [
      "AIPerf captures stable evidence from request lifecycle",
      "Flow compiler produces interactive visual scenes",
      "Explainer deck couples narration to scene progression",
      "Optional embedded scenes on relevant slides",
      "Support for real and deterministic simulation modes",
      "Keyboard and voice navigation for accessibility"
    ],
    "caption": "AIPerf Flow unifies execution measurement, visualization, and interactive learning"
  }
],
  glossary: [
  {
    "word": "Evidence",
    "meaning": "A stable, unique identifier for a lifecycle boundary with timestamp and causal context"
  },
  {
    "word": "Clock",
    "meaning": "Abstraction providing wall time (RealClock) or deterministic virtual time (SimClock) to every scheduled action"
  },
  {
    "word": "Sink",
    "meaning": "A transport-specific handler that receives one request and drives it to terminal, observing lifecycle events"
  },
  {
    "word": "Worker",
    "meaning": "An OS thread or LocalSet that owns a request sink and independent scheduling state"
  },
  {
    "word": "TTFT",
    "meaning": "Time-To-First-Token: latency from request dispatch to first output token received"
  },
  {
    "word": "Steady-state",
    "meaning": "A measurement window derived from the concurrency ramp curve, excluding warmup and drain"
  },
  {
    "word": "Scene",
    "meaning": "A spatial arrangement of entities and relations with timeline choreography and interaction handlers"
  },
  {
    "word": "Explainer Deck",
    "meaning": "A slideshow combining narration, concepts, and optional embedded scenes to teach system architecture"
  }
],
};

export const COMPILED_EXPLAINER_DECKS = [
  RUST_ARCHITECTURE_DECK,
  SLURM_VELO_DECK,
  DYNOSIM_DECK,
  AIPERF_FLOW_SYSTEM_DECK,
] as const;
