/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { validateProps, type ComponentPropsSchema } from "../src/components.js";

function range(): SourceRange {
  return {
    source: "<test>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

const queueSchema: ComponentPropsSchema = {
  id: "viz.queue",
  props: {
    capacity: { kind: "number", required: true },
    label: { kind: "string", required: false },
  },
};

describe("validateProps", () => {
  test("accepts props that match the declared schema", () => {
    const result = validateProps(
      { capacity: 8, label: "Ready queue" },
      queueSchema,
      range(),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toEqual({ capacity: 8, label: "Ready queue" });
    }
  });

  test("accepts a schema with only its required props supplied", () => {
    const result = validateProps({ capacity: 8 }, queueSchema, range());

    expect(result.ok).toBe(true);
  });

  test("reports STRICT_UNKNOWN_PROP for a prop not declared by the schema", () => {
    const result = validateProps({ capacity: 8, wrongProp: true }, queueSchema, range());

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "STRICT_UNKNOWN_PROP",
          severity: "error",
          message: expect.stringContaining("wrongProp"),
        }),
      ]),
    );
  });

  test("reports PROP_MISSING_REQUIRED when a required prop is absent", () => {
    const result = validateProps({}, queueSchema, range());

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "PROP_MISSING_REQUIRED", severity: "error" }),
      ]),
    );
  });

  test("reports PROP_TYPE_MISMATCH when a prop value has the wrong runtime type", () => {
    const result = validateProps({ capacity: "eight" }, queueSchema, range());

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "PROP_TYPE_MISMATCH", severity: "error" }),
      ]),
    );
  });

  test("accumulates one diagnostic per violation across multiple bad props", () => {
    const result = validateProps({ wrongProp: 1 }, queueSchema, range());

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    const codes = result.diagnostics.map((diagnostic) => diagnostic.code);
    expect(codes).toEqual(
      expect.arrayContaining(["STRICT_UNKNOWN_PROP", "PROP_MISSING_REQUIRED"]),
    );
  });

  test("accepts any JSON value for a prop declared with kind \"json\"", () => {
    const schema: ComponentPropsSchema = {
      id: "core.structured-payload",
      props: { data: { kind: "json", required: true } },
    };

    const result = validateProps({ data: { nested: [1, 2, 3] } }, schema, range());

    expect(result.ok).toBe(true);
  });
});
