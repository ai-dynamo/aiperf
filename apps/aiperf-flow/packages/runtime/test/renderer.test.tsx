// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import {
  createFoundationRegistry,
  SceneRenderer,
} from "../src/renderer.js";
import { FlowApp } from "../src/app.js";
import { loadPackedFlow } from "../src/site.js";
import {
  createInitialSceneState,
  sceneReducer,
} from "../src/store.js";

const sourceMap = {
  source: "request-flow.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});

function scene(): SceneIr {
  return {
    id: "execution",
    title: "Execution boundary",
    summary: "The CLI starts a runtime.",
    roots: [
      {
        kind: "rect",
        id: "cli",
        geometry: { x: 40, y: 100, width: 160, height: 72 },
        style: { fill: "#315b8a" },
        accessibility: {
          label: "CLI",
          description: "Command-line process",
        },
        fallback: "CLI",
        sourceMap,
      },
      {
        kind: "rect",
        id: "runtime",
        geometry: { x: 300, y: 100, width: 180, height: 72 },
        style: { fill: "#244a35" },
        accessibility: {
          label: "Runtime",
          description: "Execution runtime",
        },
        fallback: "Runtime",
        sourceMap,
      },
      {
        kind: "text",
        id: "label",
        geometry: { x: 60, y: 130, width: 120, height: 24 },
        style: { fill: "#ffffff" },
        text: "CLI",
        accessibility: {
          label: "CLI label",
          description: "Label inside the CLI node",
        },
        fallback: "CLI",
        sourceMap,
      },
      {
        kind: "connector",
        id: "spawn",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#7aa2f7" },
        from: { nodeId: "cli" },
        to: { nodeId: "runtime" },
        accessibility: {
          label: "Spawn connection",
          description: "CLI starts the runtime",
        },
        fallback: "CLI starts Runtime",
        sourceMap,
      },
    ],
    camera: [],
    timeline: [
      {
        id: "reveal-cli",
        at: 0,
        duration: 400,
        action: "reveal",
        target: "cli",
        sourceMap,
      },
      {
        id: "reveal-runtime",
        at: 2200,
        duration: 400,
        action: "reveal",
        target: "runtime",
        sourceMap,
      },
    ],
    narration: "The CLI starts a fresh runtime and dispatches work.",
    interactions: [
      {
        id: "inspect-runtime",
        event: "select",
        target: "runtime",
        action: "inspect",
        sourceMap,
      },
    ],
    responsive: [],
    accessibility: {
      label: "Execution boundary diagram",
      readingOrder: ["cli", "runtime", "spawn"],
    },
    fallback: "CLI starts Runtime.",
    sourceMap,
  };
}

describe("sceneReducer", () => {
  test("updates serializable scene state without mutating the prior state", () => {
    const initial = createInitialSceneState("execution");
    const selected = sceneReducer(initial, {
      type: "select-node",
      nodeId: "runtime",
    });

    expect(selected).not.toBe(initial);
    expect(initial.selectedNodeId).toBeNull();
    expect(selected.selectedNodeId).toBe("runtime");
  });
});

describe("SceneRenderer", () => {
  test("renders rect, text, and connector nodes with SVG semantics", () => {
    const { container } = render(<SceneRenderer scene={scene()} />);

    expect(container.querySelectorAll("rect")).toHaveLength(2);
    expect(container.querySelector("text")?.textContent).toBe("CLI");
    expect(container.querySelector("line")).not.toBeNull();
    expect(screen.getByLabelText("Execution boundary diagram")).not.toBeNull();
  });

  test("exposes authored accessible names and descriptions", () => {
    render(<SceneRenderer scene={scene()} />);

    const runtime = screen.getByLabelText("Runtime");
    expect(runtime.getAttribute("aria-describedby")).toBe(
      "flow-node-runtime-description",
    );
    expect(
      document.getElementById("flow-node-runtime-description")?.textContent,
    ).toBe("Execution runtime");
  });

  test("renders equivalent scene chunks with the same semantic output", () => {
    const normal = render(<SceneRenderer scene={scene()} />);
    const normalStage = normal.container.querySelector("svg")?.outerHTML;
    normal.unmount();

    const packedChunk = JSON.parse(JSON.stringify(scene())) as SceneIr;
    const packed = render(<SceneRenderer scene={packedChunk} />);
    expect(packed.container.querySelector("svg")?.outerHTML).toBe(normalStage);
  });

  test("executes inspect interactions and focuses the inspector", () => {
    render(<SceneRenderer scene={scene()} />);

    fireEvent.click(screen.getByLabelText("Runtime"));

    const inspector = screen.getByRole("region", { name: "Node inspector" });
    expect(inspector.getAttribute("tabindex")).toBe("-1");
    expect(document.activeElement).toBe(inspector);
    expect(inspector.textContent).toContain("Execution runtime");
  });

  test("uses the authored fallback when a capability is unavailable", () => {
    const registry = createFoundationRegistry();
    const incompleteScene = {
      ...scene(),
      roots: [
        {
          ...(scene().roots[0] as object),
          kind: "unknown",
          fallback: "This diagram is unavailable.",
        },
      ],
    } as unknown as SceneIr;

    render(<SceneRenderer registry={registry} scene={incompleteScene} />);
    expect(screen.getByText("This diagram is unavailable.")).not.toBeNull();
  });

  test("applies final timeline state in reduced-motion mode", () => {
    const { container } = render(
      <SceneRenderer reducedMotion scene={scene()} />,
    );

    expect(
      container
        .querySelector('[data-flow-node-id="runtime"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");
  });
});

describe("FlowApp", () => {
  test("keeps fallback, transcript, and navigation available for an invalid scene", () => {
    const invalid = {
      ...scene(),
      roots: null,
      summary: "Execution scene summary",
      fallback: "Execution scene text fallback",
    };
    const next = {
      ...scene(),
      id: "results",
      title: "Results",
    };
    const flow = {
      irVersion: 2,
      id: "request-flow",
      title: "Request flow",
      capabilities: [],
      tokens: {},
      themes: [],
      scenes: [invalid, next],
      sourceMap,
    } as unknown as FlowIr;

    render(<FlowApp flow={flow} />);

    expect(screen.getByText("Execution scene summary")).not.toBeNull();
    expect(screen.getByText("Execution scene text fallback")).not.toBeNull();
    expect(screen.getAllByText(/fresh runtime/)).toHaveLength(1);
    expect(screen.getByRole("link", { name: "Skip to transcript" })).not.toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Next scene" }));
    expect(screen.getByRole("heading", { name: "Results" })).not.toBeNull();
  });

  test("provides playback controls and scene progress", () => {
    const flow = {
      irVersion: 2,
      id: "request-flow",
      title: "Request flow",
      capabilities: [],
      tokens: {},
      themes: [],
      scenes: [scene()],
      sourceMap,
    } as unknown as FlowIr;

    render(<FlowApp flow={flow} />);

    expect(screen.getByText(/Scene 1 of 1/u)).not.toBeNull();
    expect(screen.getByRole("button", { name: "Play" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Open commands" })).not.toBeNull();
  });
});

describe("packed site loader", () => {
  test("validates capabilities and lazily fetches only the first scene chunk", async () => {
    const first = scene();
    const responses = new Map<string, unknown>([
      [
        "./flow.manifest.json",
        {
          formatVersion: 1,
          id: "request-flow",
          title: "Request flow",
          capabilities: [{ id: "core.rect", range: "^1.0.0" }],
          scenes: [
            {
              id: "execution",
              title: "Execution boundary",
              summary: "Execution summary",
              fallback: "Execution fallback",
              transcript: "Execution transcript",
              chunkPath: "chunks/scene-execution.json",
            },
            {
              id: "results",
              title: "Results",
              summary: "Results summary",
              fallback: "Results fallback",
              transcript: "Results transcript",
              chunkPath: "chunks/scene-results.json",
            },
          ],
          source: "request-flow.flow",
        },
      ],
      ["./chunks/scene-execution.json", first],
    ]);
    const requested: string[] = [];
    const fetcher = async (input: RequestInfo | URL): Promise<Response> => {
      const path = String(input);
      requested.push(path);
      const payload = responses.get(path);
      return new Response(JSON.stringify(payload), {
        status: payload === undefined ? 404 : 200,
      });
    };

    const flow = await loadPackedFlow(fetcher);

    expect(recordFlow(flow).scenes).toHaveLength(2);
    expect(requested).toEqual([
      "./flow.manifest.json",
      "./chunks/scene-execution.json",
    ]);
  });
});

function recordFlow(flow: FlowIr): {
  readonly scenes: readonly SceneIr[];
} {
  return flow as unknown as { readonly scenes: readonly SceneIr[] };
}
