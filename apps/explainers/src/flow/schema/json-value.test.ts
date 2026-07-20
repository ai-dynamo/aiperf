/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, expectTypeOf, it } from "vitest";

import { jsonValueSchema, type JsonValue } from "./json-value.js";

type IsAny<T> = 0 extends 1 & T ? true : false;

describe("JsonValue", () => {
  it("is a proper recursive JSON type, not any", () => {
    // Compile-time: assigning true fails while JsonValue is any.
    const notAny: IsAny<JsonValue> extends false ? true : never = true;
    expect(notAny).toBe(true);
    expectTypeOf<JsonValue>().not.toBeAny();
  });

  it("accepts recursive JSON shapes (type-level)", () => {
    const scalar: JsonValue = "ok";
    const nested: JsonValue = { a: [1, true, null, { b: "c" }] };
    expectTypeOf(scalar).toMatchTypeOf<JsonValue>();
    expectTypeOf(nested).toMatchTypeOf<JsonValue>();
  });

  it("rejects non-JSON values (type-level)", () => {
    // @ts-expect-error functions are not JSON
    const bad: JsonValue = () => undefined;
    void bad;
  });

  it("parses nested JSON at runtime", () => {
    const value = {
      label: "panel",
      count: 3,
      enabled: true,
      missing: null,
      kids: [{ id: "a" }, "leaf"],
    };
    expect(jsonValueSchema.parse(value)).toEqual(value);
  });

  it("rejects non-finite numbers and functions at runtime", () => {
    expect(() => jsonValueSchema.parse(Number.NaN)).toThrow();
    expect(() => jsonValueSchema.parse(() => 1)).toThrow();
  });
});
