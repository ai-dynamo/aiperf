/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { parseDocument } from "./parser.js";

describe("parseDocument", () => {
  it("does not throw when a quoted string contains a raw tab", () => {
    const source = `flow "hello\tworld" as demo {
  language 1
}
`;

    expect(() => parseDocument(source, "tab.flow")).not.toThrow();

    const result = parseDocument(source, "tab.flow");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.title).toBe("hello\tworld");
    }
  });

  it("still unescapes standard JSON string escapes", () => {
    const source = `flow "line\\nnext \\"quoted\\"" as demo {
  language 1
}
`;

    const result = parseDocument(source, "escapes.flow");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.title).toBe('line\nnext "quoted"');
    }
  });

  it("parses single-segment require capability names", () => {
    const source = `flow "Demo" as demo {
  language 1
  require foo "1.0"
}
`;

    const result = parseDocument(source, "require.flow");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.requirements).toEqual([
        expect.objectContaining({
          kind: "requirement",
          capability: "foo",
          versionRange: "1.0",
        }),
      ]);
    }
  });

  it("parses multi-segment require capability names", () => {
    const source = `flow "Demo" as demo {
  language 1
  require foo.bar "1.0"
}
`;

    const result = parseDocument(source, "require-dot.flow");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.requirements[0]?.capability).toBe("foo.bar");
    }
  });

  it("parses single-segment theme role references", () => {
    const source = `flow "Demo" as demo {
  language 1
  scene "S" as s {
    rect box {
      x 0
      y 0
      width 10
      height 10
      fill theme(bg)
    }
  }
}
`;

    const result = parseDocument(source, "theme.flow");
    expect(result.ok).toBe(true);
    if (result.ok) {
      const rect = result.value.scenes[0]?.renderDeclarations.find(
        (statement) => statement.kind === "rect",
      );
      expect(rect).toMatchObject({
        kind: "rect",
        fill: {
          kind: "theme-role-reference",
          role: "bg",
        },
      });
    }
  });
});
