// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it } from "vitest";

import { createAppRouter } from "./router";

async function renderAndAudit(path: string) {
  const router = createAppRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  const rendered = render(<RouterProvider router={router} />);
  await screen.findByRole("heading", { level: 1 });
  const results = await axe(rendered.container);
  expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual(
    [],
  );
}

describe("automated accessibility", () => {
  it.each([
    ["/?audience=developer", "ownership"],
    ["/scenes/runner-protocol-registries?audience=developer", "runner registries"],
    ["/scenes/scheduling-phase-lifecycle?audience=developer", "scheduling"],
    ["/scenes/dataset-segment-pipeline?audience=developer", "dataset"],
    ["/scenes/endpoint-bindings-transports?audience=developer", "transports"],
    ["/scenes/graph-ir-execution?audience=developer", "graph ir"],
    ["/scenes/metrics-telemetry?audience=developer", "metrics"],
    ["/scenes/accuracy-evaluator-hosting?audience=developer", "accuracy"],
    ["/scenes/crate-dependency-topology?audience=developer", "crate topology"],
    ["/?audience=executive", "executive lens"],
    ["/scenes/endpoint-bindings-transports?audience=maintainer", "maintainer lens"],
  ])("has no detectable violations on the %s %s route", async (path) => {
    await renderAndAudit(path);
  });

  it("audits graph-first command controls and status notice", async () => {
    await renderAndAudit("/");
    expect(
      screen.getByRole("searchbox", { name: "Graph search" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Share graph state" }),
    ).toBeInTheDocument();
  });

  it.each([
    "/crates/aiperf-clock?audience=maintainer",
    "/crates/not-real?audience=developer",
  ])("has no detectable violations on crate route %s", async (path) => {
    await renderAndAudit(path);
  });
});
