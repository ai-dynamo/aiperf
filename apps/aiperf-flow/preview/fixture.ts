// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  parseFlowIr,
  parseSceneIr,
  type CapabilityRequirement,
  type FlowIr,
  type GeometryIr,
  type RenderNodeIr,
  type SceneIr,
} from "../packages/schema/src/ir.js";
import type { SourceRange } from "../packages/schema/src/source.js";

const REQUEST_SOURCE = "request-flow.flow";
const ARCHITECTURE_SOURCE = "architecture.flow";
const ENDPOINT_SOURCE = "endpoint-lifecycle.flow";

function sourceMapFor(source: string): SourceRange {
  return {
    source,
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 1, line: 1, column: 2 },
  };
}

const requestSourceMap = sourceMapFor(REQUEST_SOURCE);
const architectureSourceMap = sourceMapFor(ARCHITECTURE_SOURCE);
const endpointSourceMap = sourceMapFor(ENDPOINT_SOURCE);

const SYSTEMS_CHALK = {
  canvas: "#181b1d",
  panel: "#24282b",
  ink: "#f2eee3",
  muted: "#9da6aa",
  guide: "#596266",
  cyan: "#65d9de",
  blue: "#74a9ff",
  violet: "#9b72ff",
  amber: "#f6bd60",
  coral: "#f58b76",
  green: "#72d6a2",
} as const;

const FOUNDATION_CAPABILITY_REQUIREMENTS: readonly CapabilityRequirement[] = [
  { id: "core.connector", range: "^1.0.0" },
  { id: "core.rect", range: "^1.0.0" },
  { id: "core.text", range: "^1.0.0" },
];

/** Pack-manifest scene entry shaped for browser navigation and lazy chunks. */
export type PreviewManifestScene = Readonly<{
  id: string;
  title: string;
  chunkPath: string;
  hash: string;
  summary: string;
}>;

/**
 * Manifest-like descriptor for one packed Flow document.
 * Mirrors {@link PackManifest} plus a scene summary for shell chrome.
 */
export type PreviewManifest = Readonly<{
  formatVersion: 1;
  id: string;
  title: string;
  sourceName: string;
  capabilities: readonly CapabilityRequirement[];
  scenes: readonly PreviewManifestScene[];
  transcriptPath: string;
  contentHash: string;
}>;

/** One selectable scene row in the hierarchical Flow browser. */
export type PreviewBrowserScene = Readonly<{
  id: string;
  title: string;
}>;

/** Chapter grouping scenes within a Flow file. */
export type PreviewBrowserChapter = Readonly<{
  id: string;
  name: string;
  scenes: readonly PreviewBrowserScene[];
}>;

/** Top-level Flow file node in the workspace browser tree. */
export type PreviewBrowserFile = Readonly<{
  id: string;
  sourceName: string;
  title: string;
  chapters: readonly PreviewBrowserChapter[];
}>;

/** Active selection within the preview workspace navigation graph. */
export type PreviewNavigationSelection = Readonly<{
  flowId: string;
  chapterId: string;
  sceneId: string;
}>;

/** Manifest-driven hierarchical navigation for the preview shell. */
export type PreviewNavigation = Readonly<{
  files: readonly PreviewBrowserFile[];
  active: PreviewNavigationSelection;
}>;

/** Production-shaped preview host payload: Flow IR plus pack-like navigation. */
export type PreviewWorkspace = Readonly<{
  flow: FlowIr;
  flows: Readonly<Record<string, FlowIr>>;
  manifests: readonly PreviewManifest[];
  navigation: PreviewNavigation;
}>;

function fixtureHash(path: string): string {
  return `preview:${path}`;
}

function sceneChunkPath(sceneId: string): string {
  return `chunks/scene-${sceneId}.json`;
}

function stubScene(input: {
  id: string;
  title: string;
  summary: string;
  narration: string;
  fallback: string;
  sourceMap: SourceRange;
  label: string;
}): SceneIr {
  const nodeId = `${input.id}-node`;
  return parseSceneIr({
    id: input.id,
    title: input.title,
    summary: input.summary,
    roots: [
      {
        kind: "rect",
        id: nodeId,
        geometry: { x: 120, y: 120, width: 220, height: 88 },
        style: { fill: "#17171d", stroke: "#2e2e38" },
        accessibility: {
          label: input.label,
          description: input.summary,
        },
        fallback: input.label,
        sourceMap: input.sourceMap,
      },
      {
        kind: "text",
        id: `${input.id}-label`,
        geometry: { x: 140, y: 152, width: 180, height: 24 },
        style: { fill: "#f6f6f8", fontSize: 16, fontWeight: 650 },
        text: input.label,
        accessibility: {
          label: `${input.label} label`,
          description: input.label,
        },
        fallback: input.label,
        sourceMap: input.sourceMap,
      },
    ],
    camera: [],
    timeline: [
      {
        id: `reveal-${input.id}`,
        at: 0,
        duration: 400,
        action: "reveal",
        target: nodeId,
        sourceMap: input.sourceMap,
      },
    ],
    narration: input.narration,
    interactions: [],
    responsive: [],
    accessibility: {
      label: `${input.title} diagram`,
      readingOrder: [nodeId],
    },
    fallback: input.fallback,
    sourceMap: input.sourceMap,
  });
}

type PreviewNodeInput = Readonly<{
  id: string;
  geometry: GeometryIr;
  label: string;
  description?: string;
  fallback?: string;
}>;

function previewRect(
  input: PreviewNodeInput,
  fill: string,
  stroke: string,
  strokeWidth = 1,
  radius = 0,
): RenderNodeIr {
  return {
    kind: "rect",
    id: input.id,
    geometry: input.geometry,
    style: {
      fill,
      stroke,
      strokeWidth,
      ...(radius > 0 ? { radius } : {}),
    },
    accessibility: {
      label: input.label,
      ...(input.description === undefined
        ? {}
        : { description: input.description }),
    },
    fallback: input.fallback ?? input.label,
    sourceMap: requestSourceMap,
  };
}

function previewText(
  input: PreviewNodeInput & Readonly<{ text: string }>,
  fill: string,
  fontSize: number,
  fontWeight = 500,
  fontFamily = "Inter, ui-sans-serif, system-ui, sans-serif",
): RenderNodeIr {
  return {
    kind: "text",
    id: input.id,
    geometry: input.geometry,
    text: input.text,
    style: { fill, fontFamily, fontSize, fontWeight },
    accessibility: {
      label: input.label,
      ...(input.description === undefined
        ? {}
        : { description: input.description }),
    },
    fallback: input.fallback ?? input.text,
    sourceMap: requestSourceMap,
  };
}

/** One glyph in a card's inline "signature" diagram row. */
type CardGlyph =
  | Readonly<{ kind: "pill"; idSuffix: string; label: string; accented?: boolean }>
  | Readonly<{ kind: "queue"; idSuffix: string; labels: readonly string[]; activeIndex?: number }>
  | Readonly<{ kind: "bars"; idSuffix: string; heights: readonly number[] }>;

type EvidenceCard = Readonly<{
  number: number;
  id: string;
  title: string;
  detail: string;
  left: CardGlyph;
  right: CardGlyph;
  accent: string;
  x: number;
  y: number;
}>;

const CARD_WIDTH = 252;
const CARD_HEIGHT = 138;
const CARD_PAD = 18;
const ROW_Y = 68;
const GLYPH_HEIGHT = 26;
const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** Approximate rendered width of a monospace label at a given size. */
function monoWidth(label: string, fontSize: number): number {
  return label.length * fontSize * 0.62;
}

/** Centered monospace caption inside a token rectangle. */
function tokenLabel(
  id: string,
  bounds: GeometryIr,
  text: string,
  fill: string,
  fontSize: number,
): RenderNodeIr {
  const x = bounds.x + (bounds.width - monoWidth(text, fontSize)) / 2;
  const y = bounds.y + bounds.height / 2 + fontSize * 0.36;
  return previewText(
    {
      id,
      geometry: { x, y, width: bounds.width, height: fontSize + 2 },
      label: text,
      text,
    },
    fill,
    fontSize,
    700,
    MONO,
  );
}

/** Intrinsic width of a glyph so a row can be right-aligned. */
function glyphWidth(glyph: CardGlyph): number {
  switch (glyph.kind) {
    case "pill":
      return Math.max(46, monoWidth(glyph.label, 10) + 22);
    case "queue":
      return glyph.labels.length * 26 - 4;
    case "bars":
      return glyph.heights.length * 9 - 3;
  }
}

/** Builds one glyph's nodes and its connector anchor id at a row origin. */
function glyphNodes(
  card: EvidenceCard,
  glyph: CardGlyph,
  originX: number,
  rowTop: number,
): Readonly<{ nodes: readonly RenderNodeIr[]; anchorId: string }> {
  const id = `${card.id}-${glyph.idSuffix}`;
  switch (glyph.kind) {
    case "pill": {
      const width = glyphWidth(glyph);
      const bounds = { x: originX, y: rowTop, width, height: GLYPH_HEIGHT };
      const stroke = glyph.accented ? card.accent : "rgba(255,255,255,0.18)";
      const ink = glyph.accented ? card.accent : SYSTEMS_CHALK.ink;
      return {
        anchorId: id,
        nodes: [
          previewRect(
            { id, geometry: bounds, label: glyph.label },
            "#22282b",
            stroke,
            1,
            8,
          ),
          tokenLabel(`${id}-label`, bounds, glyph.label, ink, 10),
        ],
      };
    }
    case "queue": {
      const size = 22;
      const nodes: RenderNodeIr[] = [];
      glyph.labels.forEach((value, index) => {
        const chipId = `${card.id}-queue-${value}`;
        const bounds = {
          x: originX + index * 26,
          y: rowTop + (GLYPH_HEIGHT - size) / 2,
          width: size,
          height: size,
        };
        const active = index === (glyph.activeIndex ?? -1);
        nodes.push(
          previewRect(
            { id: chipId, geometry: bounds, label: `Queue slot ${value}` },
            "#22282b",
            active ? card.accent : "rgba(255,255,255,0.16)",
            active ? 1.5 : 1,
            size / 2,
          ),
          tokenLabel(
            `${chipId}-label`,
            bounds,
            value,
            active ? card.accent : SYSTEMS_CHALK.muted,
            9,
          ),
        );
      });
      const last = glyph.labels.at(-1) ?? "";
      return { nodes, anchorId: `${card.id}-queue-${last}` };
    }
    case "bars": {
      const nodes: RenderNodeIr[] = [];
      const barWidth = 6;
      const baseline = rowTop + GLYPH_HEIGHT;
      glyph.heights.forEach((fraction, index) => {
        const barId = `${card.id}-bar-${index + 1}`;
        const height = Math.max(4, Math.round(GLYPH_HEIGHT * fraction));
        nodes.push(
          previewRect(
            {
              id: barId,
              geometry: {
                x: originX + index * 9,
                y: baseline - height,
                width: barWidth,
                height,
              },
              label: `Bar ${index + 1}`,
            },
            card.accent,
            card.accent,
            0,
            2,
          ),
        );
      });
      return { nodes, anchorId: `${card.id}-bar-1` };
    }
  }
}

function evidenceCard(card: EvidenceCard): RenderNodeIr {
  const geometry = { x: card.x, y: card.y, width: CARD_WIDTH, height: CARD_HEIGHT };
  const label = `${card.number}. ${card.title}`;
  const rowTop = card.y + ROW_Y;

  const left = glyphNodes(card, card.left, card.x + CARD_PAD, rowTop);
  const rightWidth = glyphWidth(card.right);
  const right = glyphNodes(
    card,
    card.right,
    card.x + CARD_WIDTH - CARD_PAD - rightWidth,
    rowTop,
  );
  const link: RenderNodeIr = {
    kind: "connector",
    id: `${card.id}-diagram-link`,
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: { stroke: "rgba(255,255,255,0.22)", strokeWidth: 1.5 },
    from: { nodeId: left.anchorId },
    to: { nodeId: right.anchorId },
    accessibility: {
      label: `${card.title} signature`,
      description: `${card.title} data path`,
    },
    fallback: card.title,
    sourceMap: requestSourceMap,
  };

  const badgeSize = 24;
  const badge = {
    x: card.x + CARD_PAD,
    y: card.y + 20,
    width: badgeSize,
    height: badgeSize,
  };

  return {
    kind: "group",
    id: card.id,
    geometry,
    style: {},
    accessibility: { label, description: card.detail },
    fallback: `${label}. ${card.detail}`,
    sourceMap: requestSourceMap,
    children: [
      previewRect(
        {
          id: `${card.id}-panel`,
          geometry,
          label: `${card.title} card`,
        },
        "#1c2023",
        "rgba(255,255,255,0.1)",
        1,
        16,
      ),
      // Connector paints before the tokens so the tokens overdraw its ends,
      // leaving only the visible span between them.
      link,
      ...left.nodes,
      ...right.nodes,
      previewRect(
        {
          id: `${card.id}-number-badge`,
          geometry: badge,
          label: `Step ${card.number} badge`,
        },
        "#22282b",
        card.accent,
        1.5,
        badgeSize / 2,
      ),
      tokenLabel(
        `${card.id}-number`,
        badge,
        String(card.number),
        card.accent,
        12,
      ),
      previewText(
        {
          id: `${card.id}-title`,
          geometry: { x: card.x + CARD_PAD + 34, y: card.y + 32, width: 176, height: 18 },
          label: `${card.title} title`,
          text: card.title,
        },
        SYSTEMS_CHALK.ink,
        13,
        650,
      ),
      previewText(
        {
          id: `${card.id}-detail`,
          geometry: { x: card.x + CARD_PAD, y: card.y + 116, width: 220, height: 14 },
          label: `${card.title} annotation`,
          text: card.detail,
        },
        SYSTEMS_CHALK.muted,
        9,
        500,
      ),
    ],
  };
}

function translateNode(node: RenderNodeIr, dx: number, dy: number): RenderNodeIr {
  const geometry = {
    ...node.geometry,
    x: node.geometry.x + dx,
    y: node.geometry.y + dy,
  };
  if (node.kind === "group" || node.kind === "component") {
    return {
      ...node,
      geometry,
      children: node.children.map((child) => translateNode(child, dx, dy)),
    };
  }
  return { ...node, geometry };
}

/** Systems Chalk request investigation expressed in foundation Flow IR. */
function requestInvestigationScene(): SceneIr {
  const cards: readonly EvidenceCard[] = [
    {
      number: 1,
      id: "gateway",
      title: "Prompt enters the gateway",
      detail: "Route and admission decisions become the first authored beat.",
      left: { kind: "pill", idSuffix: "client", label: "CLIENT" },
      right: { kind: "pill", idSuffix: "edge", label: "EDGE", accented: true },
      accent: SYSTEMS_CHALK.amber,
      x: 23,
      y: 41,
    },
    {
      number: 2,
      id: "admission",
      title: "Admission queues the work",
      detail: "Queue depth and wait time remain attached to the request.",
      left: {
        kind: "queue",
        idSuffix: "queue",
        labels: ["8", "9", "10"],
        activeIndex: 1,
      },
      right: { kind: "pill", idSuffix: "gpu", label: "GPU", accented: true },
      accent: SYSTEMS_CHALK.blue,
      x: 454,
      y: 7,
    },
    {
      number: 3,
      id: "prefix-cache",
      title: "Prefix cache is consulted",
      detail: "A cache hit shortens prefill; a miss explains the extra work.",
      left: { kind: "pill", idSuffix: "hash", label: "HASH" },
      right: { kind: "pill", idSuffix: "kv", label: "KV", accented: true },
      accent: SYSTEMS_CHALK.violet,
      x: 885,
      y: 41,
    },
    {
      number: 4,
      id: "prefill",
      title: "Prefill claims compute",
      detail: "Batch pressure and memory occupancy explain time to first token.",
      left: { kind: "pill", idSuffix: "prefill", label: "PREFILL" },
      right: { kind: "bars", idSuffix: "bars", heights: [0.4, 0.7, 1, 0.6] },
      accent: SYSTEMS_CHALK.green,
      x: 0,
      y: 352,
    },
    {
      number: 5,
      id: "decode",
      title: "Decode streams tokens",
      detail: "Inter-token gaps reveal contention without losing narrative rhythm.",
      left: { kind: "pill", idSuffix: "gpu", label: "GPU" },
      right: { kind: "pill", idSuffix: "tokens", label: "t·t·t", accented: true },
      accent: SYSTEMS_CHALK.coral,
      x: 908,
      y: 352,
    },
    {
      number: 6,
      id: "telemetry",
      title: "Telemetry supplies evidence",
      detail: "Metrics are supporting evidence, not a competing dashboard.",
      left: { kind: "bars", idSuffix: "bars", heights: [0.5, 0.8, 0.4, 1, 0.7] },
      right: { kind: "pill", idSuffix: "trace", label: "TRACE", accented: true },
      accent: SYSTEMS_CHALK.cyan,
      x: 209,
      y: 560,
    },
    {
      number: 7,
      id: "resolution",
      title: "The causal path resolves",
      detail: "Every spoke collapses back into one explainable request story.",
      left: { kind: "pill", idSuffix: "arrive", label: "ARRIVE" },
      right: { kind: "pill", idSuffix: "done", label: "DONE", accented: true },
      accent: SYSTEMS_CHALK.amber,
      x: 699,
      y: 560,
    },
  ];
  const hubGeometry = { x: 455, y: 271, width: 250, height: 148 };
  const hub: RenderNodeIr = {
    kind: "group",
    id: "request-hub",
    geometry: hubGeometry,
    style: {},
    accessibility: {
      label: "What made this slow?",
      description:
        "Request R-017. Follow one causal path across every layer of inference.",
    },
    fallback: "Request R-017: What made this slow?",
    sourceMap: requestSourceMap,
    children: [
      previewRect(
        {
          id: "request-hub-panel",
          geometry: hubGeometry,
          label: "Selected request R-017",
        },
        "#241d38",
        SYSTEMS_CHALK.violet,
        1.5,
        18,
      ),
      previewText(
        {
          id: "request-hub-kicker",
          geometry: { x: 512, y: 317, width: 136, height: 14 },
          label: "Request identifier",
          text: "REQUEST · R-017",
        },
        SYSTEMS_CHALK.violet,
        10,
        700,
        "ui-monospace, SFMono-Regular, Menlo, monospace",
      ),
      previewText(
        {
          id: "request-hub-question",
          geometry: { x: 483, y: 353, width: 194, height: 28 },
          label: "Investigation question",
          text: "What made this slow?",
        },
        SYSTEMS_CHALK.ink,
        25,
        610,
      ),
      previewText(
        {
          id: "request-hub-detail",
          geometry: { x: 480, y: 382, width: 200, height: 16 },
          label: "Investigation instruction",
          text: "Follow one causal path across every layer of inference.",
        },
        SYSTEMS_CHALK.muted,
        10,
      ),
    ],
  };
  const connectors = cards.map((card, index): RenderNodeIr => ({
    kind: "connector",
    id: `request-to-${card.id}`,
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: {
      stroke: index === 0 ? SYSTEMS_CHALK.cyan : SYSTEMS_CHALK.guide,
      strokeWidth: index === 0 ? 2 : 1.5,
    },
    from: { nodeId: "request-hub" },
    to: { nodeId: card.id },
    accessibility: {
      label:
        index === 0
          ? "Active cause path to gateway"
          : `Structural evidence path to ${card.title}`,
      description: `Request R-017 connects to step ${card.number}, ${card.title}.`,
    },
    fallback: `Request connects to ${card.title}`,
    sourceMap: requestSourceMap,
  }));
  const narrowCards = cards.map((card) => ({
    ...card,
    x: 34,
    y: 190 + (card.number - 1) * 145,
  }));
  const narrowConnectors = connectors.map((connector) => ({
    ...connector,
    style: { ...connector.style, strokeWidth: 0 },
  }));

  return parseSceneIr({
    id: "request-investigation",
    title: "What made this slow?",
    summary: "One request connects to seven layers of causal evidence.",
    roots: [
      previewRect(
        {
          id: "systems-chalk-field",
          geometry: { x: 0, y: 0, width: 1160, height: 690 },
          label: "Systems Chalk field",
        },
        SYSTEMS_CHALK.canvas,
        SYSTEMS_CHALK.canvas,
        0,
      ),
      ...connectors,
      hub,
      ...cards.map(evidenceCard),
    ],
    camera: [],
    timeline: [
      {
        id: "reveal-request-hub",
        at: 0,
        duration: 700,
        action: "reveal",
        target: "request-hub",
        sourceMap: requestSourceMap,
      },
      ...cards.map((card, index) => ({
        id: `reveal-${card.id}`,
        at: 200 + index * 120,
        duration: 550,
        action: "reveal",
        target: card.id,
        sourceMap: requestSourceMap,
      })),
      {
        id: "trace-active-cause",
        at: 700,
        duration: 1200,
        action: "trace",
        target: "request-to-gateway",
        sourceMap: requestSourceMap,
      },
    ],
    narration:
      "Request R-017 asks what made this slow, tracing one causal path from gateway admission through cache, compute, decode, telemetry, and resolution.",
    narrativeTrack: {
      language: "en-US",
      voice: "narrator",
      cues: [
        {
          id: "question",
          startMs: 0,
          endMs: 2400,
          spokenText: "Request R-017 asks what made this slow,",
          subtitleText: "What made request R-017 slow?",
        },
        {
          id: "evidence",
          startMs: 2400,
          endMs: 6200,
          spokenText:
            "tracing one causal path from gateway admission through cache, compute, decode, telemetry, and resolution.",
          subtitleText: "Follow the causal path across every inference layer",
        },
      ],
    },
    interactions: [
      {
        id: "inspect-request",
        event: "select",
        target: "request-hub",
        action: "inspect",
        sourceMap: requestSourceMap,
      },
    ],
    responsive: [
      {
        id: "request-investigation-narrow",
        condition: "(max-width: 860px)",
        roots: [
          previewRect(
            {
              id: "systems-chalk-field",
              geometry: { x: 0, y: 0, width: 320, height: 1_230 },
              label: "Systems Chalk field",
            },
            SYSTEMS_CHALK.canvas,
            SYSTEMS_CHALK.canvas,
            0,
          ),
          ...narrowConnectors,
          translateNode(hub, 34 - hub.geometry.x, 20 - hub.geometry.y),
          ...narrowCards.map(evidenceCard),
        ],
        sourceMap: requestSourceMap,
      },
    ],
    accessibility: {
      label: "Hub and spoke request lifecycle diagram",
      readingOrder: [
        "request-hub",
        ...cards.flatMap((card) => [
          card.id,
          `request-to-${card.id}`,
        ]),
      ],
    },
    fallback:
      "Request R-017 connects to seven evidence steps from gateway through causal resolution.",
    sourceMap: requestSourceMap,
  });
}

/** Fully authored foundation execution scene used by the live cinematic stage. */
function executionScene(): SceneIr {
  return parseSceneIr({
    id: "execution",
    title: "Execution boundary",
    summary:
      "One runtime fans work into scheduling, transport, and observation seams.",
    roots: [
      {
        kind: "rect",
        id: "runtime",
        geometry: { x: 76, y: 126, width: 178, height: 98 },
        style: { fill: "#17171d", stroke: "#2e2e38" },
        accessibility: {
          label: "Runtime",
          description: "Execution runtime composition root",
        },
        fallback: "Runtime",
        sourceMap: requestSourceMap,
      },
      {
        kind: "rect",
        id: "scheduler",
        geometry: { x: 390, y: 28, width: 190, height: 82 },
        style: { fill: "#17171d", stroke: "#2e2e38" },
        accessibility: {
          label: "Scheduler",
          description: "Clock-aware workload scheduler",
        },
        fallback: "Scheduler",
        sourceMap: requestSourceMap,
      },
      {
        kind: "rect",
        id: "worker",
        geometry: { x: 390, y: 139, width: 190, height: 82 },
        style: { fill: "#17171d", stroke: "#2e2e38" },
        accessibility: {
          label: "Worker sink",
          description: "Worker-local request transport",
        },
        fallback: "Worker sink",
        sourceMap: requestSourceMap,
      },
      {
        kind: "rect",
        id: "observer",
        geometry: { x: 390, y: 250, width: 190, height: 82 },
        style: { fill: "#17171d", stroke: "#2e2e38" },
        accessibility: {
          label: "Observer",
          description: "Transport-neutral measurement observer",
        },
        fallback: "Observer",
        sourceMap: requestSourceMap,
      },
      {
        kind: "connector",
        id: "schedule",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#26c6da", strokeWidth: 1.75 },
        from: { nodeId: "runtime" },
        to: { nodeId: "scheduler" },
        accessibility: {
          label: "Schedule connection",
          description: "Runtime schedules workload turns",
        },
        fallback: "Runtime schedules workload turns",
        sourceMap: requestSourceMap,
      },
      {
        kind: "connector",
        id: "dispatch",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#26c6da", strokeWidth: 1.75 },
        from: { nodeId: "runtime" },
        to: { nodeId: "worker" },
        accessibility: {
          label: "Dispatch connection",
          description: "Runtime dispatches work to the worker sink",
        },
        fallback: "Runtime dispatches work",
        sourceMap: requestSourceMap,
      },
      {
        kind: "connector",
        id: "measure",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#26c6da", strokeWidth: 1.75 },
        from: { nodeId: "runtime" },
        to: { nodeId: "observer" },
        accessibility: {
          label: "Observation connection",
          description: "Runtime emits observations",
        },
        fallback: "Runtime emits observations",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "runtime-kind",
        geometry: { x: 96, y: 148, width: 110, height: 16 },
        style: { fill: "#94d340", fontSize: 10, fontWeight: 500 },
        text: "COMPOSITION ROOT",
        accessibility: {
          label: "Runtime kind",
          description: "Runtime is the composition root",
        },
        fallback: "Composition root",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "runtime-label",
        geometry: { x: 96, y: 174, width: 120, height: 24 },
        style: { fill: "#f6f6f8", fontSize: 20, fontWeight: 700 },
        text: "Runtime",
        accessibility: {
          label: "Runtime label",
          description: "Runtime",
        },
        fallback: "Runtime",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "scheduler-label",
        geometry: { x: 407, y: 44, width: 142, height: 20 },
        style: { fill: "#f6f6f8", fontSize: 15, fontWeight: 650 },
        text: "Scheduler",
        accessibility: {
          label: "Scheduler label",
          description: "Scheduler",
        },
        fallback: "Scheduler",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "scheduler-detail",
        geometry: { x: 407, y: 67, width: 150, height: 14 },
        style: { fill: "#7f7f8a", fontSize: 9, fontWeight: 500 },
        text: "CLOCK-AWARE · LOCAL",
        accessibility: {
          label: "Scheduler metadata",
          description: "Clock-aware and worker-local",
        },
        fallback: "Clock-aware",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "scheduler-signature",
        geometry: { x: 407, y: 88, width: 156, height: 13 },
        style: { fill: "#5fd6e6", fontSize: 9, fontWeight: 500 },
        text: "schedule → dispatch",
        accessibility: {
          label: "Scheduler signature",
          description: "Schedule to dispatch",
        },
        fallback: "schedule to dispatch",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "worker-label",
        geometry: { x: 407, y: 155, width: 142, height: 20 },
        style: { fill: "#f6f6f8", fontSize: 15, fontWeight: 650 },
        text: "Worker sink",
        accessibility: {
          label: "Worker sink label",
          description: "Worker sink",
        },
        fallback: "Worker sink",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "worker-detail",
        geometry: { x: 407, y: 178, width: 150, height: 14 },
        style: { fill: "#7f7f8a", fontSize: 9, fontWeight: 500 },
        text: "TRANSPORT · REQUEST",
        accessibility: {
          label: "Worker metadata",
          description: "Transport request path",
        },
        fallback: "Transport",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "worker-signature",
        geometry: { x: 407, y: 199, width: 156, height: 13 },
        style: { fill: "#5fd6e6", fontSize: 9, fontWeight: 500 },
        text: "dispatch(request)",
        accessibility: {
          label: "Worker sink signature",
          description: "Dispatch request",
        },
        fallback: "dispatch request",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "observer-label",
        geometry: { x: 407, y: 266, width: 142, height: 20 },
        style: { fill: "#f6f6f8", fontSize: 15, fontWeight: 650 },
        text: "Observer",
        accessibility: {
          label: "Observer label",
          description: "Observer",
        },
        fallback: "Observer",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "observer-detail",
        geometry: { x: 407, y: 289, width: 150, height: 14 },
        style: { fill: "#7f7f8a", fontSize: 9, fontWeight: 500 },
        text: "TOKENS · TERMINAL",
        accessibility: {
          label: "Observer metadata",
          description: "Token and terminal events",
        },
        fallback: "Token events",
        sourceMap: requestSourceMap,
      },
      {
        kind: "text",
        id: "observer-signature",
        geometry: { x: 407, y: 310, width: 160, height: 13 },
        style: { fill: "#5fd6e6", fontSize: 9, fontWeight: 500 },
        text: "on_token · on_terminal",
        accessibility: {
          label: "Observer signature",
          description: "Token and terminal callbacks",
        },
        fallback: "token and terminal callbacks",
        sourceMap: requestSourceMap,
      },
    ],
    camera: [],
    // Visual beats are paced to spoken narration (~150 wpm), not animation-only
    // flash timings. Subtitles and SpeechSynthesis stay on this same clock.
    timeline: [
      {
        id: "reveal-runtime",
        at: 0,
        duration: 1800,
        action: "reveal",
        target: "runtime",
        sourceMap: requestSourceMap,
      },
      {
        id: "trace-schedule",
        at: 2000,
        duration: 1800,
        action: "trace",
        target: "schedule",
        sourceMap: requestSourceMap,
      },
      {
        id: "trace-dispatch",
        at: 4000,
        duration: 1800,
        action: "trace",
        target: "dispatch",
        sourceMap: requestSourceMap,
      },
      {
        id: "trace-measure",
        at: 6000,
        duration: 1800,
        action: "trace",
        target: "measure",
        sourceMap: requestSourceMap,
      },
      {
        id: "reveal-observer",
        at: 8000,
        duration: 1600,
        action: "reveal",
        target: "observer",
        sourceMap: requestSourceMap,
      },
    ],
    narration:
      "The runtime coordinates scheduling, dispatches worker-local transport, and observes every request through one stable seam.",
    narrativeTrack: {
      language: "en-US",
      voice: "narrator",
      cues: [
        {
          id: "schedule",
          startMs: 0,
          endMs: 2000,
          spokenText: "The runtime coordinates scheduling,",
          subtitleText: "Runtime coordinates scheduling",
        },
        {
          id: "dispatch",
          startMs: 2000,
          endMs: 4000,
          spokenText: "dispatches worker-local transport,",
          subtitleText: "Dispatches worker-local transport",
        },
        {
          id: "observe",
          startMs: 4000,
          endMs: 9600,
          spokenText:
            "and observes every request through one stable seam.",
          subtitleText: "Observes every request through one stable seam",
        },
      ],
    },
    interactions: [
      {
        id: "inspect-runtime",
        event: "select",
        target: "runtime",
        action: "inspect",
        sourceMap: requestSourceMap,
      },
    ],
    responsive: [],
    accessibility: {
      label: "Execution boundary diagram",
      readingOrder: [
        "runtime",
        "scheduler",
        "worker",
        "observer",
        "schedule",
        "dispatch",
        "measure",
      ],
    },
    fallback: "Runtime coordinates scheduling, dispatch, and observation.",
    sourceMap: requestSourceMap,
  });
}

function requestFlowScenes(): readonly SceneIr[] {
  // Narrative order matches the Runtime path chapter in previewNavigation().
  return [
    requestInvestigationScene(),
    stubScene({
      id: "author-run",
      title: "Author the run",
      summary: "Config v2 expands into a protocol-v2 benchmark request.",
      narration:
        "The profile command resolves Config v2 and projects a deterministic execute request.",
      fallback: "Author the benchmark run from Config v2.",
      sourceMap: requestSourceMap,
      label: "Author",
    }),
    stubScene({
      id: "launch-runtime",
      title: "Launch runtime",
      summary: "The CLI re-enters the same binary in internal execute mode.",
      narration:
        "aiperf re-executes itself over stdio and boots the runtime composition root.",
      fallback: "Launch the runtime through self-execution.",
      sourceMap: requestSourceMap,
      label: "Launch",
    }),
    stubScene({
      id: "freeze-registry",
      title: "Freeze registry",
      summary: "Registered capabilities freeze closed at Application bootstrap.",
      narration:
        "Unknown endpoint, transport, workload, and exporter identifiers fail closed.",
      fallback: "Freeze the capability registry.",
      sourceMap: requestSourceMap,
      label: "Registry",
    }),
    stubScene({
      id: "choose-clock",
      title: "Choose clock",
      summary: "RealClock or SimClock owns every measurement and firing gate.",
      narration:
        "Scheduling and measurement never call ambient Instant::now on the hot path.",
      fallback: "Choose the execution clock.",
      sourceMap: requestSourceMap,
      label: "Clock",
    }),
    stubScene({
      id: "choose-workload",
      title: "Choose workload",
      summary: "Request-rate, concurrency, user-centric, or fixed-schedule work.",
      narration:
        "The selected workload partitions budgets or conversations across workers.",
      fallback: "Choose the workload policy.",
      sourceMap: requestSourceMap,
      label: "Workload",
    }),
    stubScene({
      id: "bind-endpoint",
      title: "Bind endpoint",
      summary: "Worker-local endpoint tables bind the transport dialect.",
      narration:
        "Prepared endpoint profiles stay worker-local; transports do not own scheduling.",
      fallback: "Bind the endpoint profile.",
      sourceMap: requestSourceMap,
      label: "Endpoint",
    }),
    executionScene(),
    stubScene({
      id: "observe-result",
      title: "Observe result",
      summary: "Token and terminal events reduce into record and aggregate metrics.",
      narration:
        "RequestObserver receives arrival, token, usage, and terminal events through one seam.",
      fallback: "Observe the request result.",
      sourceMap: requestSourceMap,
      label: "Observe",
    }),
  ];
}

function architectureFlowScenes(): readonly SceneIr[] {
  return [
    stubScene({
      id: "control-plane",
      title: "Control plane",
      summary: "Controller partitions cellular work and merges folded stores.",
      narration:
        "Cellular execution selects controller and cell roles before clap parsing.",
      fallback: "Control plane overview.",
      sourceMap: architectureSourceMap,
      label: "Control",
    }),
    stubScene({
      id: "worker-topology",
      title: "Worker topology",
      summary: "Thread-per-core workers own scheduling and transport sinks.",
      narration:
        "workers == 1 co-locates a sink; workers > 1 runs self-contained sub-cells.",
      fallback: "Worker topology overview.",
      sourceMap: architectureSourceMap,
      label: "Workers",
    }),
    stubScene({
      id: "metrics-plane",
      title: "Metrics plane",
      summary: "Record, aggregate, derived, and sweep metrics fold at boundaries.",
      narration:
        "Exact mode retains records; sketch mode merges streaming t-digests.",
      fallback: "Metrics plane overview.",
      sourceMap: architectureSourceMap,
      label: "Metrics",
    }),
  ];
}

function endpointFlowScenes(): readonly SceneIr[] {
  return [
    stubScene({
      id: "resolve-endpoint",
      title: "Resolve endpoint",
      summary: "Endpoint binding selects dialect factories and prepared tables.",
      narration:
        "Registered endpoint identifiers resolve before the first dispatch.",
      fallback: "Resolve the endpoint.",
      sourceMap: endpointSourceMap,
      label: "Resolve",
    }),
    stubScene({
      id: "open-transport",
      title: "Open transport",
      summary: "HTTP or gRPC opens with clock-injected clients and connect retry.",
      narration:
        "Pre-send connect failures retry with linear backoff gated by max_connect_retries.",
      fallback: "Open the transport.",
      sourceMap: endpointSourceMap,
      label: "Transport",
    }),
    stubScene({
      id: "reduce-response",
      title: "Reduce response",
      summary: "Shared reduce and measure layers capture tokens and terminals.",
      narration:
        "HTTP and gRPC share response reduction; TTFT is the first token observation.",
      fallback: "Reduce the response.",
      sourceMap: endpointSourceMap,
      label: "Reduce",
    }),
  ];
}

function manifestFor(
  flow: FlowIr,
  sourceName: string,
): PreviewManifest {
  const scenes = flow.scenes.map((scene) => {
    const chunkPath = sceneChunkPath(scene.id);
    return {
      id: scene.id,
      title: scene.title,
      chunkPath,
      hash: fixtureHash(chunkPath),
      summary: scene.summary,
    };
  });
  return {
    formatVersion: 1,
    id: flow.id,
    title: flow.title,
    sourceName,
    capabilities: flow.capabilities,
    scenes,
    transcriptPath: "transcript.txt",
    contentHash: fixtureHash(
      scenes.map((scene) => `${scene.chunkPath}:${scene.hash}`).join("|"),
    ),
  };
}

function browserScenes(
  manifest: PreviewManifest,
  sceneIds: readonly string[],
): readonly PreviewBrowserScene[] {
  const byId = new Map(manifest.scenes.map((scene) => [scene.id, scene]));
  return sceneIds.flatMap((sceneId) => {
    const scene = byId.get(sceneId);
    return scene === undefined
      ? []
      : [{ id: scene.id, title: scene.title }];
  });
}

function previewBrowserNavigation(
  manifests: readonly PreviewManifest[],
  active: PreviewNavigationSelection,
): PreviewNavigation {
  const byId = new Map(manifests.map((manifest) => [manifest.id, manifest]));
  const request = byId.get("request-flow");
  const architecture = byId.get("architecture");
  const endpoint = byId.get("endpoint-lifecycle");
  if (
    request === undefined ||
    architecture === undefined ||
    endpoint === undefined
  ) {
    throw new Error("Preview workspace is missing a required flow manifest.");
  }

  return {
    files: [
      {
        id: request.id,
        sourceName: request.sourceName,
        title: request.title,
        chapters: [
          {
            id: "runtime-path",
            name: "Runtime path",
            scenes: browserScenes(request, [
              "request-investigation",
              "author-run",
              "launch-runtime",
              "freeze-registry",
              "choose-clock",
              "choose-workload",
              "bind-endpoint",
              "execution",
              "observe-result",
            ]),
          },
        ],
      },
      {
        id: architecture.id,
        sourceName: architecture.sourceName,
        title: architecture.title,
        chapters: [
          {
            id: "system-map",
            name: "System map",
            scenes: browserScenes(architecture, [
              "control-plane",
              "worker-topology",
              "metrics-plane",
            ]),
          },
        ],
      },
      {
        id: endpoint.id,
        sourceName: endpoint.sourceName,
        title: endpoint.title,
        chapters: [
          {
            id: "request-lifecycle",
            name: "Request lifecycle",
            scenes: browserScenes(endpoint, [
              "resolve-endpoint",
              "open-transport",
              "reduce-response",
            ]),
          },
        ],
      },
    ],
    active,
  };
}

function buildRequestFlow(): FlowIr {
  return parseFlowIr({
    irVersion: 2,
    id: "request-flow",
    title: "Request flow",
    capabilities: FOUNDATION_CAPABILITY_REQUIREMENTS,
    tokens: { accent: "#26c6da" },
    themes: [],
    scenes: requestFlowScenes(),
    sourceMap: requestSourceMap,
  });
}

function buildArchitectureFlow(): FlowIr {
  return parseFlowIr({
    irVersion: 2,
    id: "architecture",
    title: "Architecture",
    capabilities: FOUNDATION_CAPABILITY_REQUIREMENTS,
    tokens: {},
    themes: [],
    scenes: architectureFlowScenes(),
    sourceMap: architectureSourceMap,
  });
}

function buildEndpointFlow(): FlowIr {
  return parseFlowIr({
    irVersion: 2,
    id: "endpoint-lifecycle",
    title: "Endpoint lifecycle",
    capabilities: FOUNDATION_CAPABILITY_REQUIREMENTS,
    tokens: {},
    themes: [],
    scenes: endpointFlowScenes(),
    sourceMap: endpointSourceMap,
  });
}

/**
 * Production-shaped preview workspace: validated Flow IR documents plus
 * pack-manifest navigation suitable for FlowApp / shell migration.
 */
export function previewWorkspace(): PreviewWorkspace {
  const requestFlow = buildRequestFlow();
  const architectureFlow = buildArchitectureFlow();
  const endpointFlow = buildEndpointFlow();
  const flows = {
    [requestFlow.id]: requestFlow,
    [architectureFlow.id]: architectureFlow,
    [endpointFlow.id]: endpointFlow,
  } as const;
  const manifests = [
    manifestFor(requestFlow, REQUEST_SOURCE),
    manifestFor(architectureFlow, ARCHITECTURE_SOURCE),
    manifestFor(endpointFlow, ENDPOINT_SOURCE),
  ] as const;
  const navigation = previewBrowserNavigation(manifests, {
    flowId: requestFlow.id,
    chapterId: "runtime-path",
    sceneId: "request-investigation",
  });
  return {
    flow: requestFlow,
    flows,
    manifests,
    navigation,
  };
}

/** Active Flow IR document for the migrated cinematic preview host. */
export function previewFlow(): FlowIr {
  return previewWorkspace().flow;
}

/** Pack-manifest-like descriptor for the active preview Flow. */
export function previewManifest(): PreviewManifest {
  const workspace = previewWorkspace();
  const active = workspace.manifests.find(
    (manifest) => manifest.id === workspace.navigation.active.flowId,
  );
  if (active === undefined) {
    throw new Error("Preview workspace is missing the active flow manifest.");
  }
  return active;
}

/** Hierarchical file → chapter → scene navigation for the Flow browser. */
export function previewNavigation(): PreviewNavigation {
  return previewWorkspace().navigation;
}

/**
 * Live development scene fixture for the active execution boundary.
 * Kept for compatibility with hosts that still mount a single SceneIr.
 */
export function previewScene(): SceneIr {
  const workspace = previewWorkspace();
  const { flowId, sceneId } = workspace.navigation.active;
  const flow = workspace.flows[flowId] ?? workspace.flow;
  const scene = flow.scenes.find((entry) => entry.id === sceneId);
  if (scene === undefined) {
    throw new Error(`Preview workspace is missing active scene "${sceneId}".`);
  }
  return scene;
}

/** Derives playback duration from timeline and narrative track end bounds. */
export function previewDurationMs(scene: SceneIr): number {
  const timelineEnd = scene.timeline.reduce(
    (max, cue) => Math.max(max, cue.at + cue.duration),
    0,
  );
  const narrativeEnd =
    scene.narrativeTrack?.cues.reduce(
      (max, cue) => Math.max(max, cue.endMs),
      0,
    ) ?? 0;
  return Math.max(timelineEnd, narrativeEnd);
}
