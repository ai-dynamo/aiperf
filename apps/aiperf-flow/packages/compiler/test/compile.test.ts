/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { FOUNDATION_CAPABILITIES } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { compileSource, type CompileRequest } from "../src/index.js";
import { FOUNDATION_SOURCE } from "./fixture.js";

function request(overrides: Partial<CompileRequest> = {}): CompileRequest {
  return {
    source: FOUNDATION_SOURCE,
    sourceName: "request-flow.flow",
    capabilities: FOUNDATION_CAPABILITIES,
    strict: false,
    ...overrides,
  };
}

describe("compileSource", () => {
  test("compiles the foundation grammar to validated, deterministic Flow IR", () => {
    const result = compileSource(request());

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const ir = result.value;
    expect(ir.irVersion).toBe(2);
    expect(ir.themes).toEqual([]);
    expect(ir.id).toBe("request-flow");
    expect(ir.title).toBe("Request flow");
    expect(ir.capabilities.map((capability) => capability.id)).toEqual([
      "core.connector",
      "core.rect",
      "core.text",
    ]);
    expect(ir.tokens.accent).toBe("#7aa2f7");

    const scene = ir.scenes[0];
    expect(scene?.id).toBe("execution");
    expect(scene?.accessibility.readingOrder).toEqual([
      "cli",
      "runtime",
      "spawn",
    ]);

    const cli = scene?.roots.find((node) => node.id === "cli");
    expect(cli?.kind).toBe("rect");
    expect(cli?.style.fill).toBe("#7aa2f7");
    expect(cli?.accessibility).toEqual({
      label: "CLI",
      description: "Command-line process",
    });

    const spawn = scene?.roots.find((node) => node.id === "spawn");
    expect(spawn?.kind).toBe("connector");
    expect(spawn?.style.stroke).toBe("#7aa2f7");
    if (spawn?.kind === "connector") {
      expect(spawn.from).toEqual({ nodeId: "cli" });
      expect(spawn.to).toEqual({ nodeId: "runtime" });
    }

    expect(scene?.camera).toEqual([
      expect.objectContaining({ id: "main-0", at: 0, x: 120, y: 136, zoom: 1 }),
      expect.objectContaining({
        id: "main-1",
        at: 2000,
        x: 390,
        y: 136,
        zoom: 1.4,
      }),
    ]);

    expect(scene?.timeline).toEqual([
      expect.objectContaining({
        id: "primary-0",
        at: 0,
        duration: 400,
        target: "cli",
        action: "reveal",
      }),
      expect.objectContaining({
        id: "primary-1",
        at: 800,
        duration: 1200,
        target: "spawn",
        action: "trace",
      }),
      expect.objectContaining({
        id: "primary-2",
        at: 2200,
        duration: 400,
        target: "runtime",
        action: "reveal",
      }),
    ]);

    expect(scene?.interactions).toEqual([
      {
        id: "inspect-runtime",
        event: "select",
        target: "runtime",
        action: "inspect",
        sourceMap: expect.anything(),
      },
    ]);

    expect(scene?.responsive).toHaveLength(1);
    const compact = scene?.responsive[0];
    expect(compact?.id).toBe("compact");
    expect(compact?.condition).toBe("width < 720");
    const overriddenRuntime = compact?.roots.find(
      (node) => node.id === "runtime",
    );
    expect(overriddenRuntime?.geometry).toEqual({
      x: 40,
      y: 240,
      width: 180,
      height: 72,
    });
    const untouchedCli = compact?.roots.find((node) => node.id === "cli");
    expect(untouchedCli?.geometry).toEqual({
      x: 40,
      y: 100,
      width: 160,
      height: 72,
    });
  });

  test("compiling the same source twice produces identical IR", () => {
    const first = compileSource(request());
    const second = compileSource(request());

    expect(first).toEqual(second);
  });

  test("reports LINK_UNKNOWN_REFERENCE for a connector target that does not exist", () => {
    const source = FOUNDATION_SOURCE.replace("to runtime", "to missing-node");

    const result = compileSource(request({ source }));

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "LINK_UNKNOWN_REFERENCE",
          severity: "error",
        }),
      ]),
    );
  });

  test("reports LINK_DUPLICATE_ID when two rects share an id", () => {
    const source = FOUNDATION_SOURCE.replace("rect runtime {", "rect cli {");

    const result = compileSource(request({ source }));

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "LINK_DUPLICATE_ID",
          severity: "error",
        }),
      ]),
    );
  });

  test("reports CAPABILITY_MISSING for an unregistered requirement", () => {
    const source = FOUNDATION_SOURCE.replace(
      'require core.connector "^1.0.0"',
      'require core.connector "^1.0.0"\n  require unknown.capability "^1.0.0"',
    );

    const result = compileSource(request({ source }));

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "CAPABILITY_MISSING",
          severity: "error",
        }),
      ]),
    );
  });

  test("reports ACCESSIBILITY_REQUIRED when a scene omits reading-order", () => {
    const source = `flow "Minimal" as minimal {
  language 1
  require core.rect "^1.0.0"

  scene "Solo" as solo {
    summary "A single labeled rectangle used to test accessibility validation."

    rect box {
      x 0
      y 0
      width 10
      height 10
      fill "#000000"
      label "Box"
      description "A single box"
      fallback "Box"
    }

    narrate "This scene intentionally omits a reading order for testing."
    fallback "Box scene."
  }
}
`;

    const result = compileSource(request({ source }));

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "ACCESSIBILITY_REQUIRED",
          severity: "error",
        }),
      ]),
    );
  });

  test("reports NARRATION_SHORT as an error in strict mode", () => {
    const source = FOUNDATION_SOURCE.replace(
      'narrate "The CLI starts a fresh runtime and dispatches work."',
      'narrate "hi"',
    );

    const strictResult = compileSource(request({ source, strict: true }));
    expect(strictResult.ok).toBe(false);
    expect(strictResult.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "NARRATION_SHORT", severity: "error" }),
      ]),
    );

    const lenientResult = compileSource(request({ source, strict: false }));
    expect(lenientResult.ok).toBe(true);
    expect(lenientResult.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "NARRATION_SHORT",
          severity: "warning",
        }),
      ]),
    );
  });
});
