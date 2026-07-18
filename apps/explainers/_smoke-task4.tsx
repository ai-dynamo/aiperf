/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
/** Temporary Task 4 smoke — delete after run. */
import { renderToStaticMarkup } from "react-dom/server";
import { SceneRenderer, type SceneIrLike } from "./src/core/diagram/SceneRenderer";
import { ThemeProvider } from "./src/core/ui";

const scene: SceneIrLike = {
  id: "smoke",
  roots: [
    {
      id: "panel-a",
      kind: "group",
      capabilityId: "core.panel",
      geometry: { x: 40, y: 40, width: 160, height: 90 },
      style: { fill: "@theme.surface.elevated", stroke: "@theme.stroke.secondary" },
      children: [
        {
          id: "panel-a-title",
          kind: "text",
          capabilityId: "core.text",
          geometry: { x: 8, y: 10, width: 144, height: 18 },
          style: { textAnchor: "middle", fontSize: 13 },
          text: "Panel A",
        },
      ],
    },
    {
      id: "stack",
      kind: "group",
      capabilityId: "layout.stack",
      geometry: { x: 240, y: 40, width: 0, height: 0 },
      style: { direction: "column", gap: 12 },
      children: [
        {
          id: "box-1",
          kind: "rect",
          capabilityId: "core.rect",
          geometry: { x: 0, y: 0, width: 100, height: 40 },
          style: {},
        },
        {
          id: "box-2",
          kind: "rect",
          capabilityId: "core.rect",
          geometry: { x: 0, y: 0, width: 100, height: 40 },
          style: {},
        },
      ],
    },
    {
      id: "left",
      kind: "rect",
      capabilityId: "core.rect",
      geometry: { x: 40, y: 200, width: 80, height: 50 },
      style: {},
    },
    {
      id: "right",
      kind: "rect",
      capabilityId: "core.rect",
      geometry: { x: 280, y: 280, width: 80, height: 50 },
      style: {},
    },
    {
      id: "elbow",
      kind: "connector",
      capabilityId: "core.elbow",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      style: { route: "elbow" },
      from: { nodeId: "left", anchor: "e" },
      to: { nodeId: "right", anchor: "w" },
      axis: "x",
    },
    {
      id: "oval",
      kind: "rect",
      capabilityId: "core.ellipse",
      geometry: { x: 400, y: 40, width: 100, height: 60 },
      style: { rx: 50, ry: 30 },
    },
  ],
  timeline: [{ id: "t0", at: 0, duration: 1, action: "enter", target: "panel-a" }],
};

const html = renderToStaticMarkup(
  <ThemeProvider>
    <SceneRenderer scene={scene} playing={false} restartKey={0} reducedMotion />
  </ThemeProvider>,
);

const checks: string[] = [];
if (!html.includes('data-flow-local-layout="true"')) {
  checks.push("missing local-layout on panel/stack");
}
if (!html.includes("Panel A")) {
  checks.push("missing panel title text");
}
if (!html.includes('data-flow-panel-chrome="true"')) {
  checks.push("missing panel chrome");
}
if (!html.includes('data-flow-elbow="true"')) {
  checks.push("missing elbow marker");
}
if (!/H[\d.]+ V[\d.]+ H[\d.]+/.test(html) && !/H[\d.]+V[\d.]+H[\d.]+/.test(html)) {
  // path may have spaces: H120 V305 H280
  if (!html.includes(" H") || !html.includes(" V")) {
    checks.push(`elbow path not orthogonal: ${html.match(/d="[^"]+"/)?.[0] ?? "no d"}`);
  }
}
if (!html.includes('data-flow-ellipse="true"')) {
  checks.push("missing ellipse");
}
// stack second box should be offset by 40+12=52 locally → look for y="52"
if (!html.includes('y="52"')) {
  checks.push("stack second box not at y=52");
}
// panel title local y=10 inside translate(40,40)
if (!html.includes('data-flow-layout-offset="40,40"')) {
  checks.push("panel missing translate(40,40)");
}

console.log(checks.length === 0 ? "SMOKE OK" : `SMOKE FAIL: ${checks.join("; ")}`);
if (checks.length > 0) {
  const elbow = html.match(/data-flow-node-id="elbow"[\s\S]*?<\/g>/)?.[0]?.slice(0, 400);
  console.log("elbow snippet:", elbow);
  const stack = html.match(/data-flow-node-id="stack"[\s\S]*?data-flow-node-id="box-2"[\s\S]*?<\/rect>/)?.[0]?.slice(0, 500);
  console.log("stack snippet:", stack);
}
process.exit(checks.length === 0 ? 0 : 1);
