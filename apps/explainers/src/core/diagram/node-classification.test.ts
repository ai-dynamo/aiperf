/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  capabilityOf,
  isMotionSignalNode,
} from "./node-classification.js";
import type { SceneNodeLike } from "./scene-types.js";

describe("node-classification", () => {
  it("falls back to core.${kind} for capability resolution", () => {
    const node: SceneNodeLike = { id: "text-1", kind: "text" };
    expect(capabilityOf(node)).toBe("core.text");
  });

  it("treats core.motion foundation connectors as motion signals", () => {
    const node: SceneNodeLike = {
      id: "guide",
      kind: "connector",
      capabilityId: "core.motion",
    };
    expect(isMotionSignalNode(node)).toBe(true);
  });

  it("treats core.motion-signal as motion signals", () => {
    const node: SceneNodeLike = {
      id: "guide",
      kind: "connector",
      capabilityId: "core.motion-signal",
    };
    expect(isMotionSignalNode(node)).toBe(true);
  });

  it("treats style.role motion as motion signals", () => {
    const node: SceneNodeLike = {
      id: "edge",
      kind: "connector",
      style: { role: "motion" },
    };
    expect(isMotionSignalNode(node)).toBe(true);
  });

  it("does not classify core.dot nodes as motion signals", () => {
    const node: SceneNodeLike = {
      id: "dot",
      kind: "dot",
      capabilityId: "core.dot",
    };
    expect(isMotionSignalNode(node)).toBe(false);
  });
});
