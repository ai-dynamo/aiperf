// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable command contracts and deterministic catalog search.

/** User-facing grouping for one runtime command. */
export type FlowCommandCategory =
  | "scene"
  | "beat"
  | "entity"
  | "evidence"
  | "action"
  | "accessibility";

/** One immutable command catalog entry backed by an existing runtime action. */
export type FlowCommand = Readonly<{
  id: string;
  label: string;
  category: FlowCommandCategory;
  keywords: readonly string[];
  shortcut?: string;
  disabledReason?: string;
  execute(): void;
}>;

type RankedCommand = Readonly<{
  command: FlowCommand;
  authoredIndex: number;
  rank: number;
}>;

function normalize(value: string): string {
  return value.trim().replace(/\s+/gu, " ").toLowerCase();
}

function labelTokens(label: string): readonly string[] {
  return normalize(label).match(/[\p{L}\p{N}]+/gu) ?? [];
}

function commandRank(command: FlowCommand, query: string): number | null {
  const label = normalize(command.label);
  if (label.startsWith(query)) {
    return 0;
  }
  if (labelTokens(label).some((token) => token.startsWith(query))) {
    return 1;
  }
  if (
    command.keywords.some((keyword) => normalize(keyword).startsWith(query))
  ) {
    return 2;
  }
  return null;
}

/**
 * Searches a command catalog by case-insensitive prefix while preserving
 * authored order within each deterministic match tier.
 */
export function searchCommands(
  commands: readonly FlowCommand[],
  query: string,
): readonly FlowCommand[] {
  const normalizedQuery = normalize(query);
  if (normalizedQuery === "") {
    return Object.freeze([...commands]);
  }

  const matches: RankedCommand[] = [];
  commands.forEach((command, authoredIndex) => {
    const rank = commandRank(command, normalizedQuery);
    if (rank !== null) {
      matches.push({ command, authoredIndex, rank });
    }
  });

  matches.sort(
    (left, right) =>
      left.rank - right.rank || left.authoredIndex - right.authoredIndex,
  );
  return Object.freeze(matches.map(({ command }) => command));
}
