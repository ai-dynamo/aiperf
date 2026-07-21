/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The cellular ability map: what works per dimension, its boundary, and its status. Ported from
//! `ABILITIES` / `AbilityMap` in the source canvas. `capability ≠ fidelity` — a dimension can be
//! Built, Partial, Planned, Rejected, or an Approximation.

import { useState } from "react";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Toggle } from "../../prose/Toggle.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { inkClassName } from "../../theme/tokens.js";
import { ABILITIES, type Ability } from "./data.js";

const COLUMNS: TableColumn[] = [
  { key: "dimension", label: "Dimension" },
  { key: "built", label: "What works" },
  { key: "boundary", label: "Boundary" },
  { key: "status", label: "Status" },
];

const STATUS_TEXT: Record<Ability["status"], string> = {
  Built: "text-category-green",
  Rejected: "text-category-red",
  Approximation: "text-category-yellow",
  Partial: "text-category-yellow",
  Planned: "text-ink-tertiary",
};

const STATUS_TONE: Record<Ability["status"], TableRow["tone"]> = {
  Built: "success",
  Rejected: "danger",
  Approximation: "warning",
  Partial: "warning",
  Planned: "neutral",
};

/** Cellular ability matrix. Self-contained; the roadmap toggle hides Planned/Partial rows. */
export function AbilitiesPage(): React.JSX.Element {
  const [roadmap, setRoadmap] = useState(true);

  const rows: TableRow[] = ABILITIES.filter(
    (ability) => roadmap || (ability.status !== "Planned" && ability.status !== "Partial"),
  ).map((ability) => ({
    dimension: <span className="font-semibold">{ability.dimension}</span>,
    built: ability.built,
    boundary: ability.boundary,
    status: <span className={`font-semibold ${STATUS_TEXT[ability.status]}`}>{ability.status}</span>,
    tone: STATUS_TONE[ability.status],
  }));

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Ability map</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          What the cellular runtime can do today, dimension by dimension, with the exact boundary of
          each capability. Capability is not the same as fidelity: some dimensions are Built, others
          Partial, Planned, Rejected, or an Approximation.
        </p>
      </div>

      <Row gap={10} align="center">
        <Toggle checked={roadmap} onChange={setRoadmap} label="Show roadmap (Planned / Partial)" />
        <span className={`text-xs ${inkClassName("tertiary")}`}>capability ≠ fidelity</span>
      </Row>

      <Table columns={COLUMNS} rows={rows} />
    </Stack>
  );
}
