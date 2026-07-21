/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Workbook mode of the cellular-algorithm workbook: a searchable, chapter-grouped index of
//! source-grounded algorithms on the left, and a stepped "algorithm sheet" on the right. The sheet
//! walks the algorithm's trace frames (via `useStepSimulator`), highlighting the synchronized
//! pseudocode line and re-drawing the actor execution tape as a real React Flow node/edge graph.
//! Ported from `docs/canvases/cellular-algorithm-workbook.canvas.tsx` (WorkbookMode / AlgorithmSheet
//! / StateTrace / PseudocodePanel / ContractRail).

import { useMemo, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Button } from "../../prose/Button.js";
import { Code } from "../../prose/Code.js";
import { Select } from "../../prose/Select.js";
import { CollapsibleSection } from "../../prose/CollapsibleSection.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import {
  ALGORITHMS,
  CHAPTERS,
  ACTORS,
  ACTOR_LABELS,
  filterAlgorithms,
  type AlgorithmDefinition,
  type Actor,
  type TraceFrame,
  type PseudocodeLine,
  type ChapterFilter,
} from "./data.js";
import { Pill, StatusLabel, Eyebrow, Framed } from "./ui.js";

function actorLinkAliases(from: Actor, to: Actor): readonly string[] {
  return [`${from}->${to}`, `${from}:${to}`, `${from}-${to}`, `${from}/${to}`];
}

/** The actor execution tape for one frame, drawn as a real React Flow graph. */
function StateTrace({
  algorithm,
  frame,
}: {
  algorithm: AlgorithmDefinition;
  frame: TraceFrame;
}): React.JSX.Element {
  const usedActors = useMemo(
    () => new Set(algorithm.frames.flatMap((item) => item.activeActors)),
    [algorithm],
  );
  const activeActors = new Set(frame.activeActors);
  const activeLinks = new Set(frame.activeLinks);

  const nodes: Node[] = ACTORS.map((actor, index) => ({
    id: actor,
    type: "panel",
    position: { x: index * 180, y: 0 },
    draggable: false,
    data: {
      title: ACTOR_LABELS[actor],
      detail: activeActors.has(actor)
        ? "active"
        : usedActors.has(actor)
          ? "in path"
          : "context",
      strokeRole: activeActors.has(actor) ? "primary" : "tertiary",
    },
  }));

  const edges: Edge[] = ACTORS.slice(0, -1).map((actor, index) => {
    const to = ACTORS[index + 1];
    const active = actorLinkAliases(actor, to).some((alias) => activeLinks.has(alias));
    return {
      id: `${actor}-${to}`,
      source: actor,
      target: to,
      type: active ? "flow" : undefined,
      data: active ? { speed: "normal" as const } : undefined,
    };
  });

  return (
    <div
      className={`rounded-none border ${strokeClassName("tertiary")}`}
      style={{ height: 200 }}
      aria-label={`${algorithm.title}: ${frame.label}`}
    >
      <ReactFlow
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        nodesDraggable={false}
        nodesConnectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-tertiary)" />
      </ReactFlow>
    </div>
  );
}

function PseudocodePanel({
  lines,
  activeLineId,
}: {
  lines: readonly PseudocodeLine[];
  activeLineId: string;
}): React.JSX.Element {
  return (
    <div className={`rounded-none border ${strokeClassName("tertiary")} ${surfaceClassName("elevated")}`}>
      {lines.map((line) => {
        const active = line.id === activeLineId;
        return (
          <div
            key={line.id}
            className={`grid gap-3 px-3 py-2 ${active ? surfaceClassName("panel") : ""}`}
            style={{
              gridTemplateColumns: "58px minmax(0, 1fr)",
              borderLeft: active
                ? "2px solid var(--color-accent-primary)"
                : "2px solid transparent",
            }}
          >
            <span className={`text-xs ${active ? inkClassName("primary") : inkClassName("tertiary")}`}>
              {line.id}
            </span>
            <Code>{line.text}</Code>
          </div>
        );
      })}
    </div>
  );
}

function ContractList({ label, values }: { label: string; values: readonly string[] }) {
  if (values.length === 0) return null;
  return (
    <Stack gap={4}>
      <Eyebrow>{label}</Eyebrow>
      {values.map((value) => (
        <p key={value} className={`text-sm ${inkClassName("secondary")}`}>
          {value}
        </p>
      ))}
    </Stack>
  );
}

function ContractRail({ algorithm }: { algorithm: AlgorithmDefinition }): React.JSX.Element {
  return (
    <Stack gap={14}>
      <Stack gap={5}>
        <Eyebrow>Source contract</Eyebrow>
        <Code>{algorithm.source.symbol}</Code>
        <p className={`text-sm ${inkClassName("secondary")}`}>
          {algorithm.source.path}:{algorithm.source.startLine}-{algorithm.source.endLine}
        </p>
      </Stack>
      <Divider />
      <ContractList label="Inputs" values={algorithm.inputs} />
      <ContractList label="Outputs" values={algorithm.outputs} />
      <ContractList label="State" values={algorithm.state} />
      <ContractList label="Gates" values={algorithm.gates} />
      <ContractList label="Invariants" values={algorithm.invariants} />
      <ContractList label="Failures" values={algorithm.failures} />
      <Stack gap={4}>
        <Eyebrow>Complexity</Eyebrow>
        <p className={`text-sm ${inkClassName("secondary")}`}>
          <Code inline>time</Code> {algorithm.complexity.time}
        </p>
        <p className={`text-sm ${inkClassName("secondary")}`}>
          <Code inline>memory</Code> {algorithm.complexity.memory}
        </p>
      </Stack>
      {algorithm.routeTags.length > 0 && (
        <Row gap={6} wrap>
          {algorithm.routeTags.map((tag) => (
            <Pill key={tag}>{tag}</Pill>
          ))}
        </Row>
      )}
      <Divider />
      <Stack gap={6}>
        <Eyebrow>Proof boundaries</Eyebrow>
        {algorithm.evidence.map((evidence) => (
          <Stack key={`${evidence.path}/${evidence.symbol}`} gap={3}>
            <Row gap={6} align="center" wrap>
              <Pill>{evidence.kind}</Pill>
              <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{evidence.symbol}</span>
            </Row>
            <span className={`text-xs ${inkClassName("tertiary")}`}>{evidence.path}</span>
          </Stack>
        ))}
      </Stack>
    </Stack>
  );
}

/** The stepped detail sheet for one algorithm. Remounted (via `key`) when the selection changes. */
function AlgorithmSheet({
  algorithm,
  onSelect,
}: {
  algorithm: AlgorithmDefinition;
  onSelect: (id: string) => void;
}): React.JSX.Element {
  const sim = useStepSimulator(algorithm.frames as TraceFrame[], { autoPlayMs: 1100 });
  const frameIndex = sim.index;
  const frame = sim.current ?? algorithm.frames[0];
  const chapterLabel = CHAPTERS.find((c) => c.id === algorithm.chapter)?.label;

  return (
    <Stack gap={16}>
      <Row gap={12} align="start" wrap justify="space-between">
        <Stack gap={5} className="min-w-0 flex-1">
          <Row gap={7} align="center" wrap>
            <StatusLabel status={algorithm.status} />
            <span className={`text-sm ${inkClassName("tertiary")}`}>{chapterLabel}</span>
          </Row>
          <h2 className={`text-xl font-semibold ${inkClassName("primary")}`}>{algorithm.title}</h2>
          <p className={`text-sm ${inkClassName("secondary")}`}>{algorithm.summary}</p>
        </Stack>
        <Code>{algorithm.id}</Code>
      </Row>

      <div
        className="workbook-live-region sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {algorithm.title}. Frame {frameIndex + 1} of {algorithm.frames.length}: {frame.label}.
      </div>

      <Grid columns="minmax(0, 1fr) minmax(260px, 0.42fr)" gap={20}>
        <Stack gap={14}>
          <Row gap={8} align="center" wrap>
            <Eyebrow>
              Frame {frameIndex + 1}/{algorithm.frames.length}
            </Eyebrow>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{frame.label}</span>
            <span className="flex-1" />
            <Button variant="ghost" disabled={sim.isFirst} onClick={sim.back}>
              Back
            </Button>
            <Button variant="secondary" disabled={sim.isLast} onClick={sim.next}>
              Step
            </Button>
            <Button variant="primary" disabled={sim.isLast} onClick={sim.togglePlay}>
              {sim.isPlaying ? "Pause" : "Play"}
            </Button>
            <Button variant="ghost" disabled={sim.isFirst} onClick={sim.reset}>
              Reset
            </Button>
          </Row>

          <StateTrace algorithm={algorithm} frame={frame} />

          <Framed>
            <Grid columns="1fr auto 1fr" gap={10} align="center">
              <Stack gap={4}>
                <Eyebrow>Before</Eyebrow>
                {frame.before.map((item) => (
                  <span key={item} className={`text-sm ${inkClassName("secondary")}`}>
                    {item}
                  </span>
                ))}
              </Stack>
              <Stack gap={2} className="text-center">
                <span className={`text-xs ${inkClassName("tertiary")}`}>emits</span>
                <Code>{frame.emitted ?? "state"}</Code>
              </Stack>
              <Stack gap={4}>
                <Eyebrow>After</Eyebrow>
                {frame.after.map((item) => (
                  <span key={item} className={`text-sm ${inkClassName("primary")}`}>
                    {item}
                  </span>
                ))}
              </Stack>
            </Grid>
            <div className="mt-2" style={{ borderLeft: "2px solid var(--color-accent-primary)", paddingLeft: 10 }}>
              {frame.invariantChecks.map((check) => (
                <p key={check} className={`text-sm ${inkClassName("secondary")}`}>
                  <Code inline>check</Code> {check}
                </p>
              ))}
            </div>
          </Framed>

          <Stack gap={7}>
            <Row gap={8} align="center">
              <Eyebrow>Synchronized pseudocode</Eyebrow>
              <span className="flex-1" />
              <Pill active>{frame.activeLineId}</Pill>
            </Row>
            <PseudocodePanel lines={algorithm.pseudocode} activeLineId={frame.activeLineId} />
          </Stack>
        </Stack>

        <div className={`border-l pl-4 ${strokeClassName("secondary")}`}>
          <CollapsibleSection title="Contracts and proof" defaultOpen>
            <ContractRail algorithm={algorithm} />
          </CollapsibleSection>
        </div>
      </Grid>

      <Divider />
      <Row gap={8} align="center" wrap>
        <Eyebrow>Traverse</Eyebrow>
        {algorithm.predecessors.map((id) => (
          <Pill key={`pred-${id}`} onClick={() => onSelect(id)}>
            Back to {ALGORITHMS.find((a) => a.id === id)?.title ?? id}
          </Pill>
        ))}
        {algorithm.successors.map((id) => (
          <Pill key={`succ-${id}`} active onClick={() => onSelect(id)}>
            Next: {ALGORITHMS.find((a) => a.id === id)?.title ?? id}
          </Pill>
        ))}
        {algorithm.predecessors.length === 0 && (
          <span className={`text-sm ${inkClassName("tertiary")}`}>Entry algorithm</span>
        )}
        {algorithm.successors.length === 0 && (
          <span className={`text-sm ${inkClassName("tertiary")}`}>Terminal algorithm</span>
        )}
      </Row>
    </Stack>
  );
}

function WorkbookIndex({
  selectedId,
  search,
  chapter,
  onSearchChange,
  onChapterChange,
  onSelect,
}: {
  selectedId: string;
  search: string;
  chapter: ChapterFilter;
  onSearchChange: (value: string) => void;
  onChapterChange: (value: ChapterFilter) => void;
  onSelect: (id: string) => void;
}): React.JSX.Element {
  const algorithms = filterAlgorithms(
    ALGORITHMS,
    search,
    chapter === "all" ? undefined : chapter,
  );

  return (
    <Stack gap={10}>
      <Eyebrow>Workbook index</Eyebrow>
      <Stack gap={4}>
        <label className={`text-sm ${inkClassName("secondary")}`}>Search algorithms</label>
        <input
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Source, invariant, failure…"
          className={`w-full rounded-none border px-2 py-1 text-sm ${strokeClassName("secondary")} ${surfaceClassName("page")} ${inkClassName("primary")}`}
        />
      </Stack>
      <Select
        label="Chapter"
        value={chapter}
        onChange={(value) => onChapterChange(value as ChapterFilter)}
        options={[
          { value: "all", label: "All chapters" },
          ...CHAPTERS.map((c) => ({ value: c.id, label: c.label })),
        ]}
      />
      <Row gap={8} align="center">
        <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{algorithms.length} indexed</span>
        <span className={`text-sm ${inkClassName("tertiary")}`}>of {ALGORITHMS.length}</span>
      </Row>
      <div className={`border-t ${strokeClassName("tertiary")}`} style={{ maxHeight: 620, overflowY: "auto" }}>
        {CHAPTERS.map((chapterDef) => {
          const chapterAlgorithms = algorithms.filter((a) => a.chapter === chapterDef.id);
          if (chapterAlgorithms.length === 0) return null;
          return (
            <div key={chapterDef.id}>
              <div className={`sticky top-0 z-10 px-2 py-2 ${surfaceClassName("page")} border-b ${strokeClassName("tertiary")}`}>
                <Eyebrow>
                  {chapterDef.label} · {chapterAlgorithms.length}
                </Eyebrow>
              </div>
              {chapterAlgorithms.map((algorithm) => {
                const active = algorithm.id === selectedId;
                return (
                  <button
                    type="button"
                    key={algorithm.id}
                    onClick={() => onSelect(algorithm.id)}
                    aria-pressed={active}
                    className={`block w-full cursor-pointer border-b px-2 py-2 text-left ${strokeClassName("tertiary")} ${active ? surfaceClassName("panel") : ""}`}
                    style={{ borderLeft: active ? "2px solid var(--color-accent-primary)" : "2px solid transparent" }}
                  >
                    <span className={`block text-sm ${active ? "font-semibold" : ""} ${inkClassName("primary")}`}>
                      {algorithm.title}
                    </span>
                    <span className={`block truncate text-xs ${inkClassName("tertiary")}`}>{algorithm.id}</span>
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </Stack>
  );
}

/** Workbook mode: index + stepped algorithm sheet. */
export function WorkbookPage(): React.JSX.Element {
  const [selectedId, setSelectedId] = useState<string>(ALGORITHMS[0].id);
  const [search, setSearch] = useState("");
  const [chapter, setChapter] = useState<ChapterFilter>("all");
  const selected = ALGORITHMS.find((a) => a.id === selectedId) ?? ALGORITHMS[0];

  return (
    <Grid columns="minmax(230px, 0.28fr) minmax(0, 1fr)" gap={28} align="start">
      <WorkbookIndex
        selectedId={selected.id}
        search={search}
        chapter={chapter}
        onSearchChange={setSearch}
        onChapterChange={setChapter}
        onSelect={setSelectedId}
      />
      <AlgorithmSheet key={selected.id} algorithm={selected} onSelect={setSelectedId} />
    </Grid>
  );
}
