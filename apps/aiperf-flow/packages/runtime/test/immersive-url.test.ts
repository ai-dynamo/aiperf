// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  parseImmersiveUrl,
  serializeImmersiveUrl,
  type ImmersiveUrlState,
} from "../src/immersive-url.js";

describe("parseImmersiveUrl", () => {
  test("reads scene, beat, and entity selections from the query string", () => {
    expect(parseImmersiveUrl("?scene=arrival&beat=ttft&entity=request-a")).toEqual(
      {
        sceneId: "arrival",
        beatId: "ttft",
        entityId: "request-a",
      },
    );
  });

  test("reports absent selections as null rather than empty strings", () => {
    expect(parseImmersiveUrl("?scene=arrival")).toEqual({
      sceneId: "arrival",
      beatId: null,
      entityId: null,
    });
  });

  test("returns all-null state for an empty search string", () => {
    expect(parseImmersiveUrl("")).toEqual({
      sceneId: null,
      beatId: null,
      entityId: null,
    });
  });

  test("decodes percent-encoded selection values", () => {
    expect(parseImmersiveUrl("?entity=request%2Fa").entityId).toBe("request/a");
  });
});

describe("serializeImmersiveUrl", () => {
  test("emits scene, beat, and entity keys in stable authored order", () => {
    const state: ImmersiveUrlState = {
      sceneId: "arrival",
      beatId: "ttft",
      entityId: "request-a",
    };

    expect(serializeImmersiveUrl(state)).toBe(
      "?scene=arrival&beat=ttft&entity=request-a",
    );
  });

  test("omits null selections instead of writing empty parameters", () => {
    const state: ImmersiveUrlState = {
      sceneId: "arrival",
      beatId: null,
      entityId: "request-a",
    };

    expect(serializeImmersiveUrl(state)).toBe("?scene=arrival&entity=request-a");
  });

  test("returns an empty string when no selection is present", () => {
    const state: ImmersiveUrlState = {
      sceneId: null,
      beatId: null,
      entityId: null,
    };

    expect(serializeImmersiveUrl(state)).toBe("");
  });

  test("round-trips a fully populated selection through parse and serialize", () => {
    const state: ImmersiveUrlState = {
      sceneId: "arrival",
      beatId: "ttft",
      entityId: "request/a",
    };

    expect(parseImmersiveUrl(serializeImmersiveUrl(state))).toEqual(state);
  });
});
