/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Interactive step-through simulator for `SegmentPool` content-addressed interning, ported
//! from `docs/canvases/segment-pools-and-body-plans.canvas.tsx` (page 2, `PagePool`). Walks the
//! user through interning dataset rows one at a time, growing an "arena" table and highlighting
//! deduplication when a row's content+parent identity has already been interned.

import { useMemo } from "react";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { Callout } from "../../prose/Callout.js";
import { Stat } from "../../prose/Stat.js";
import { Button } from "../../prose/Button.js";
import { surfaceClassName, inkClassName } from "../../theme/tokens.js";

/** One dataset row to intern, mirroring the canvas source's `Step` shape. */
type DatasetRow = {
  id: string;
  conversation: 1 | 2;
  role: "system" | "user" | "assistant";
  content: string;
  parent?: string;
};

/**
 * Two conversations sharing a system prompt and first user turn, so the walkthrough can
 * demonstrate cross-conversation dedup exactly like the source canvas.
 */
const DATASET_ROWS: DatasetRow[] = [
  { id: "c1s", conversation: 1, role: "system", content: "You are a helpful assistant." },
  { id: "c1u", conversation: 1, role: "user", content: "What is 2+2?", parent: "c1s" },
  { id: "c1a", conversation: 1, role: "assistant", content: "4", parent: "c1u" },
  { id: "c2s", conversation: 2, role: "system", content: "You are a helpful assistant." },
  { id: "c2u", conversation: 2, role: "user", content: "What is 2+2?", parent: "c2s" },
  { id: "c2a", conversation: 2, role: "assistant", content: "It equals four.", parent: "c2u" },
];

// FNV-1a — an *illustrative* stand-in for blake3 so handles show a stable, content+parent
// derived id in the UI. Not the real hash used by the runtime's SegmentPool.
function fnv(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0").slice(0, 6);
}

type Resolved = { handle: number; hash: string; deduped: boolean };

type ArenaEntry = {
  handle: number;
  hash: string;
  role: string;
  content: string;
  parent: number | null;
};

export type PoolSimulation = {
  resolved: Record<string, Resolved>;
  arena: ArenaEntry[];
  dedup: number;
};

/**
 * Pure recomputation of interning state as of `upTo` rows processed (0..DATASET_ROWS.length).
 * Mirrors the canvas source's `simulate(upTo)`: a child's identity key folds in the *parent's*
 * resolved content hash (not its index), so identical prefixes collapse to the same handle even
 * across conversations and regardless of load order.
 */
export function simulatePoolInterning(rows: readonly DatasetRow[], upTo: number): PoolSimulation {
  const resolved: Record<string, Resolved> = {};
  const arena: ArenaEntry[] = [];
  const ids = new Map<string, number>();
  let dedup = 0;

  for (let i = 0; i < upTo && i < rows.length; i++) {
    const row = rows[i];
    const parentRes = row.parent ? resolved[row.parent] : undefined;
    const parentHash = parentRes ? parentRes.hash : "root";
    const key = `message|${parentHash}|${row.role}|${row.content}`;
    const hash = fnv(key);

    const existingHandle = ids.get(key);
    if (existingHandle !== undefined) {
      resolved[row.id] = { handle: existingHandle, hash, deduped: true };
      dedup++;
    } else {
      const handle = arena.length;
      ids.set(key, handle);
      arena.push({
        handle,
        hash,
        role: row.role,
        content: row.content,
        parent: parentRes ? parentRes.handle : null,
      });
      resolved[row.id] = { handle, hash, deduped: false };
    }
  }

  return { resolved, arena, dedup };
}

const ARENA_COLUMNS: TableColumn[] = [
  { key: "handle", label: "Handle", align: "end" },
  { key: "hash", label: "Hash" },
  { key: "role", label: "Role" },
  { key: "content", label: "Content" },
  { key: "parent", label: "Parent", align: "end" },
];

const CALLS_COLUMNS: TableColumn[] = [
  { key: "id", label: "Row" },
  { key: "role", label: "Role" },
  { key: "content", label: "Content" },
  { key: "outcome", label: "Outcome" },
];

const BYTES_SAVED_PER_DEDUP = 42;

// `useStepSimulator` steps over "how many rows have been interned so far" (0..DATASET_ROWS.length)
// rather than over the rows themselves, so index 0 cleanly means "empty arena" and each `next()`
// advances the count by exactly one intern call — matching the canvas source's `setN`/`simulate(n)`
// step counter rather than a cursor that always points at a row.
const STEP_COUNTS: number[] = Array.from({ length: DATASET_ROWS.length + 1 }, (_, i) => i);

/**
 * `SegmentPool` interning walkthrough: steps through interned-row counts via
 * {@link useStepSimulator}, recomputing the arena with {@link simulatePoolInterning} on every
 * step. Self-contained; takes no required props.
 */
export function PoolPage(): React.JSX.Element {
  const sim = useStepSimulator(STEP_COUNTS, { autoPlayMs: 900 });
  const upTo = sim.current ?? 0;
  const result = useMemo(() => simulatePoolInterning(DATASET_ROWS, upTo), [upTo]);

  const bytesSaved = result.dedup * BYTES_SAVED_PER_DEDUP;
  const lastRow = upTo > 0 ? DATASET_ROWS[upTo - 1] : undefined;
  const lastResolved = lastRow ? result.resolved[lastRow.id] : undefined;

  const callRows: TableRow[] = DATASET_ROWS.slice(0, upTo).map((row) => {
    const res = result.resolved[row.id];
    return {
      id: row.id,
      role: row.role,
      content: row.content,
      outcome: res?.deduped ? "deduped → reused handle" : "interned → new handle",
      tone: res?.deduped ? "success" : "neutral",
    };
  });

  const arenaRows: TableRow[] = result.arena.map((entry) => ({
    handle: entry.handle,
    hash: entry.hash,
    role: entry.role,
    content: entry.content,
    parent: entry.parent === null ? "—" : entry.parent,
  }));

  return (
    <Stack gap={16} className={surfaceClassName("page")}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
          SegmentPool — content-addressed interning
        </h2>
        <p className={`mt-1 text-sm ${inkClassName("secondary")}`}>
          Step through interning two conversations that share a system prompt and first user
          turn. Because a child&apos;s id folds in its parent&apos;s content hash, identical
          prefixes collapse to the same handle — even across conversations, even in a different
          load order.
        </p>
      </div>

      <Row gap={12} align="center" wrap>
        <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>
          Intern next
        </Button>
        <Button
          variant="secondary"
          onClick={() => {
            // `sim.next()` schedules a state update rather than mutating `sim` in place, so a
            // `while (!sim.isLast)` loop here would spin forever on the stale closed-over
            // `isLast`. Call `next()` a fixed, bounded number of times instead; each call is a
            // no-op once the simulator is already at its last step.
            for (let i = 0; i < STEP_COUNTS.length; i++) {
              sim.next();
            }
          }}
        >
          Run all
        </Button>
        <Button variant="ghost" onClick={sim.reset}>
          Reset
        </Button>
        <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
          {upTo}/{DATASET_ROWS.length} steps
        </span>
      </Row>

      <Grid columns={4} gap={12}>
        <Stat value={result.arena.length} label="arena size (handles)" />
        <Stat value={result.dedup} label="dedup hits" tone={result.dedup > 0 ? "positive" : "neutral"} />
        <Stat value={`~${bytesSaved}B`} label="content not re-stored" tone="positive" />
        <Stat value={upTo} label="intern calls" />
      </Grid>

      <Grid columns="1fr 1fr" gap={14}>
        <div>
          <h3 className={`mb-2 text-sm font-semibold ${inkClassName("secondary")}`}>Intern calls</h3>
          <Table columns={CALLS_COLUMNS} rows={callRows} />
        </div>
        <div>
          <h3 className={`mb-2 text-sm font-semibold ${inkClassName("secondary")}`}>Arena</h3>
          <Table columns={ARENA_COLUMNS} rows={arenaRows} />
        </div>
      </Grid>

      {lastResolved?.deduped && lastRow && (
        <Callout tone="success" title="Dedup hit">
          Row <strong>{lastRow.id}</strong> (&quot;{lastRow.content}&quot;) resolved to the same
          handle as a previously interned row — its bytes were never re-stored.
        </Callout>
      )}

      <Callout tone="info" title="Illustrative hash">
        Hashes shown here use a small FNV-1a stand-in for readability, not the runtime&apos;s real
        blake3 <code>SegmentId</code>. The dedup behavior it demonstrates — content+parent identity
        collapsing to one arena handle — mirrors the real <code>SegmentPool</code>.
      </Callout>
    </Stack>
  );
}
