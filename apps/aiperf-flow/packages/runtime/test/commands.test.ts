// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import { searchCommands, type FlowCommand } from "../src/commands.js";

function command(overrides: Partial<FlowCommand> = {}): FlowCommand {
  return {
    id: overrides.id ?? "focus-request",
    label: overrides.label ?? "Focus request",
    category: overrides.category ?? "entity",
    keywords: overrides.keywords ?? [],
    ...(overrides.shortcut === undefined ? {} : { shortcut: overrides.shortcut }),
    ...(overrides.disabledReason === undefined
      ? {}
      : { disabledReason: overrides.disabledReason }),
    execute: overrides.execute ?? vi.fn(),
  };
}

const catalog: readonly FlowCommand[] = [
  command({ id: "play", label: "Play scene", category: "scene" }),
  command({ id: "pause", label: "Pause playback", category: "scene" }),
  command({
    id: "twin",
    label: "Open semantic twin",
    category: "accessibility",
    keywords: ["screen reader", "table"],
  }),
  command({ id: "focus", label: "Focus request A", category: "entity" }),
];

describe("searchCommands", () => {
  test("returns a frozen copy of every command for a blank query", () => {
    const results = searchCommands(catalog, "   ");

    expect(results.map(({ id }) => id)).toEqual([
      "play",
      "pause",
      "twin",
      "focus",
    ]);
    expect(Object.isFrozen(results)).toBe(true);
    expect(results).not.toBe(catalog);
  });

  test("ranks whole-label prefixes ahead of interior token prefixes", () => {
    const results = searchCommands(catalog, "p");

    // "Play scene" and "Pause playback" start with the query (tier 0) and keep
    // authored order; "Focus request A" only matches on an interior token
    // ("request") and falls to the token tier.
    expect(results.map(({ id }) => id)).toEqual(["play", "pause"]);
  });

  test("matches an interior label token when no label starts with the query", () => {
    const results = searchCommands(catalog, "request");

    expect(results.map(({ id }) => id)).toEqual(["focus"]);
  });

  test("matches keyword prefixes below label matches", () => {
    const results = searchCommands(catalog, "screen");

    expect(results.map(({ id }) => id)).toEqual(["twin"]);
  });

  test("orders label-prefix, token, and keyword tiers deterministically", () => {
    const commands: readonly FlowCommand[] = [
      command({ id: "keyword-only", label: "Reset camera", keywords: ["seek"] }),
      command({ id: "token", label: "Jump to seek marker" }),
      command({ id: "label", label: "Seek to next beat" }),
    ];

    const results = searchCommands(commands, "seek");

    expect(results.map(({ id }) => id)).toEqual([
      "label",
      "token",
      "keyword-only",
    ]);
  });

  test("normalizes case and collapses interior whitespace before matching", () => {
    const results = searchCommands(catalog, "  PLAY   SCENE ");

    expect(results.map(({ id }) => id)).toEqual(["play"]);
  });

  test("returns an empty frozen list when nothing matches", () => {
    const results = searchCommands(catalog, "nonexistent");

    expect(results).toEqual([]);
    expect(Object.isFrozen(results)).toBe(true);
  });
});
