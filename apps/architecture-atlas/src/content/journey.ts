// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { LifecycleStage } from "../domain/architecture";
import { copy, evidence } from "./helpers";

export const oneRunLifecycle: LifecycleStage[] = [
  {
    id: "lifecycle.author",
    kind: "lifecycle",
    order: 0,
    title: copy("Choose the run", "Author Config v2", "Project authored-v2 input"),
    summary: copy(
      "The user makes product choices once in the supported front door.",
      "Python validates configuration and expands only outer run loops.",
      "Python emits side-effect-free authored protocol-v2 state without resolved endpoint or dataset objects.",
    ),
    componentIds: ["component.python-frontend"],
    evidence: [evidence("src/aiperf/cli_runner/_single_run.py")],
  },
  {
    id: "lifecycle.preflight",
    kind: "lifecycle",
    order: 1,
    title: copy("Confirm availability", "Preflight exact capabilities", "Bind exact-image capability hash"),
    summary: copy(
      "Unsupported combinations fail before expensive work begins.",
      "Python asks the selected runner whether the requested backend and workload pair is executable.",
      "Capabilities are derived from the same frozen RunnerApplication used for validation and execution.",
    ),
    componentIds: ["component.python-frontend", "component.rust-runner"],
    evidence: [evidence("src/aiperf/cli_runner/_preflight.py"), evidence("crates/runner/src/registry.rs")],
  },
  {
    id: "lifecycle.validate",
    kind: "lifecycle",
    order: 2,
    title: copy("Validate the contract", "Validate strict request", "Deserialize deny-unknown-fields DTOs"),
    summary: copy(
      "Configuration errors are reported without starting a run.",
      "The runner validates authored input against the selected pair and linked registries.",
      "protocol_v2 uses strict serde DTOs and typed validate responses; authored validation remains side-effect free.",
    ),
    componentIds: ["component.rust-runner"],
    evidence: [evidence("crates/runner/src/protocol_v2.rs")],
  },
  {
    id: "lifecycle.execute",
    kind: "lifecycle",
    order: 3,
    title: copy("Run and measure", "Execute one prepared operation", "Dispatch on current_thread plus LocalSet"),
    summary: copy(
      "One isolated run produces detailed performance evidence.",
      "The pair adapter prepares inputs, executes the workload, drains lifecycle state, and accumulates native metrics.",
      "Online and offline factories enter pair-specific execute modules while retaining Clock and observer contracts.",
    ),
    componentIds: ["component.rust-runner", "component.rust-runtime", "component.inference-target"],
    evidence: [evidence("crates/runner/src/execute.rs")],
  },
  {
    id: "lifecycle.present",
    kind: "lifecycle",
    order: 4,
    title: copy("Return decision evidence", "Persist then present native-v2", "Serialize finite native-v2 report"),
    summary: copy(
      "Results return to the product layer for comparison and communication.",
      "Rust writes deterministic native results and Python owns user-facing presentation.",
      "The IO-free Reporter feeds runner persistence; no native human-facing CLI or table path is canonical.",
    ),
    componentIds: ["component.rust-runtime", "component.python-frontend"],
    evidence: [evidence("crates/aiperf/src/report.rs"), evidence("crates/metrics/src/report.rs")],
  },
];
