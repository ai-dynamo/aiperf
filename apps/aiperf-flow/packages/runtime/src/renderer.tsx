// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FOUNDATION_CAPABILITIES,
  resolveCapabilityId,
  type RenderNodeIr,
  type SceneIr,
} from "@aiperf/flow-schema";
import {
  type CSSProperties,
  type ReactNode,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from "react";

import {
  CapabilityRegistry,
  type RenderContext,
  type RuntimeCapability,
} from "./registry.js";
import {
  createInitialSceneState,
  sceneReducer,
  type SceneState,
} from "./store.js";

/** User Timing measure names collected by Playwright runtime metrics. */
export const RUNTIME_PERFORMANCE_ENTRY_NAMES = Object.freeze({
  evaluation: "aiperf-flow:evaluation",
  draw: "aiperf-flow:draw",
  total: "aiperf-flow:total",
});

let runtimeMeasureSequence = 0;

function canRecordRuntimeMeasures(): boolean {
  return (
    typeof performance !== "undefined" &&
    typeof performance.mark === "function" &&
    typeof performance.measure === "function" &&
    typeof performance.clearMarks === "function"
  );
}

/**
 * Records one real wall-clock phase as a User Timing measure when available.
 * Falls through unchanged when the Performance API is absent.
 */
export function measureRuntimePhase<T>(
  phase: keyof typeof RUNTIME_PERFORMANCE_ENTRY_NAMES,
  work: () => T,
): T {
  if (!canRecordRuntimeMeasures()) {
    return work();
  }
  const name = RUNTIME_PERFORMANCE_ENTRY_NAMES[phase];
  const token = `${name}:${runtimeMeasureSequence++}`;
  const startMark = `${token}:start`;
  const endMark = `${token}:end`;
  performance.mark(startMark);
  try {
    return work();
  } finally {
    performance.mark(endMark);
    try {
      performance.measure(name, startMark, endMark);
    } finally {
      performance.clearMarks(startMark);
      performance.clearMarks(endMark);
    }
  }
}

type UnknownRecord = Readonly<Record<string, unknown>>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function string(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function number(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nodeId(node: RenderNodeIr): string {
  return string(record(node).id);
}

function nodeGeometry(node: RenderNodeIr): UnknownRecord {
  return record(record(node).geometry);
}

function nodeStyle(node: RenderNodeIr): CSSProperties {
  return record(record(node).style) as CSSProperties;
}

function nodeAccessibility(node: RenderNodeIr): UnknownRecord {
  return record(record(node).accessibility);
}

function foundationDescriptor(id: string) {
  const descriptor = FOUNDATION_CAPABILITIES.capabilities.find(
    (candidate) => candidate.id === id,
  );
  if (descriptor === undefined) {
    throw new Error(`Foundation capability "${id}" is not declared.`);
  }
  return descriptor;
}

function accessibleNode(
  node: RenderNodeIr,
  context: RenderContext,
  child: ReactNode,
): ReactNode {
  const id = nodeId(node);
  const accessibility = nodeAccessibility(node);
  const descriptionId = `flow-node-${id}-description`;
  const timelineState = timelineStateForNode(
    node,
    context.state,
    context.timeline,
  );

  return (
    <g
      aria-describedby={descriptionId}
      aria-label={string(accessibility.label, id)}
      data-flow-node-id={id}
      data-timeline-state={timelineState}
      onClick={() => context.activateNode(id)}
      role={accessibility.decorative === true ? "presentation" : "img"}
      style={timelineState === "hidden" ? { opacity: 0 } : undefined}
      tabIndex={0}
    >
      <desc id={descriptionId}>{string(accessibility.description)}</desc>
      {child}
    </g>
  );
}

function timelineStateForNode(
  node: RenderNodeIr,
  state: SceneState,
  timeline: SceneIr["timeline"],
): "hidden" | "revealed" | "unchanged" {
  const cue = timeline
    .map(record)
    .filter(
      (candidate) =>
        string(candidate.target) === nodeId(node) &&
        string(candidate.action) === "reveal",
    )
    .at(-1);
  if (cue === undefined) {
    return "unchanged";
  }
  const endMs = number(cue.at) + number(cue.duration);
  return state.playbackTimeMs >= endMs ? "revealed" : "hidden";
}

function rectCapability(): RuntimeCapability {
  return {
    descriptor: foundationDescriptor("core.rect"),
    render(node, context) {
      const geometry = nodeGeometry(node);
      return accessibleNode(
        node,
        context,
        <rect
          height={number(geometry.height)}
          rx={number(record(node).radius)}
          style={nodeStyle(node)}
          width={number(geometry.width)}
          x={number(geometry.x)}
          y={number(geometry.y)}
        />,
      );
    },
  };
}

function textCapability(): RuntimeCapability {
  return {
    descriptor: foundationDescriptor("core.text"),
    render(node, context) {
      const geometry = nodeGeometry(node);
      return accessibleNode(
        node,
        context,
        <text
          dominantBaseline="hanging"
          style={nodeStyle(node)}
          x={number(geometry.x)}
          y={number(geometry.y)}
        >
          {string(record(node).text)}
        </text>,
      );
    },
  };
}

function connectorCapability(): RuntimeCapability {
  return {
    descriptor: foundationDescriptor("core.connector"),
    render(node, context) {
      const properties = record(node);
      const from = context.nodeById(string(record(properties.from).nodeId));
      const to = context.nodeById(string(record(properties.to).nodeId));
      const fromGeometry = from === undefined ? {} : nodeGeometry(from);
      const toGeometry = to === undefined ? {} : nodeGeometry(to);
      const x1 = number(fromGeometry.x) + number(fromGeometry.width) / 2;
      const y1 = number(fromGeometry.y) + number(fromGeometry.height) / 2;
      const x2 = number(toGeometry.x) + number(toGeometry.width) / 2;
      const y2 = number(toGeometry.y) + number(toGeometry.height) / 2;

      return accessibleNode(
        node,
        context,
        <line style={nodeStyle(node)} x1={x1} x2={x2} y1={y1} y2={y2} />,
      );
    },
  };
}

function groupCapability(): RuntimeCapability {
  return {
    descriptor: foundationDescriptor("core.group"),
    render(node, context) {
      const children = record(node).children;
      return accessibleNode(
        node,
        context,
        Array.isArray(children)
          ? children.map((child) => context.renderNode(child as RenderNodeIr))
          : null,
      );
    },
  };
}

function inertCapability(
  id: "core.camera" | "core.timeline" | "core.inspect",
): RuntimeCapability {
  return {
    descriptor: foundationDescriptor(id),
    render: () => null,
  };
}

export function createFoundationRegistry(): CapabilityRegistry {
  const registry = new CapabilityRegistry();
  [
    groupCapability(),
    rectCapability(),
    textCapability(),
    connectorCapability(),
    inertCapability("core.camera"),
    inertCapability("core.timeline"),
    inertCapability("core.inspect"),
  ].forEach((capability) => registry.register(capability));
  return registry;
}

export type SceneRendererProps = Readonly<{
  scene: SceneIr;
  registry?: CapabilityRegistry;
  reducedMotion?: boolean;
  playbackTimeMs?: number;
}>;

export function SceneRenderer({
  scene,
  registry: suppliedRegistry,
  reducedMotion = false,
  playbackTimeMs = 0,
}: SceneRendererProps): ReactNode {
  const registry = useMemo(
    () => suppliedRegistry ?? createFoundationRegistry(),
    [suppliedRegistry],
  );
  const sceneProperties = record(scene);
  const renderTree = Array.isArray(sceneProperties.roots)
    ? (sceneProperties.roots as readonly RenderNodeIr[])
    : [];
  const nodes = useMemo(
    () => new Map(renderTree.map((node) => [nodeId(node), node])),
    [renderTree],
  );
  const [state, dispatch] = useReducer(
    sceneReducer,
    string(sceneProperties.id),
    createInitialSceneState,
  );
  const inspectorRef = useRef<HTMLElement>(null);
  const timeline = Array.isArray(sceneProperties.timeline)
    ? sceneProperties.timeline
    : [];
  const durationMs = timeline.reduce((maximum, value) => {
    const cue = record(value);
    return Math.max(
      maximum,
      number(cue.at) + number(cue.duration),
    );
  }, 0);
  const effectiveTimeMs = reducedMotion ? durationMs : playbackTimeMs;
  const renderState = {
    ...state,
    playbackTimeMs: effectiveTimeMs,
  };

  useEffect(() => {
    if (state.inspector.open) {
      inspectorRef.current?.focus();
    }
  }, [state.inspector.open]);

  function activateNode(id: string): void {
    dispatch({ type: "select-node", nodeId: id });
    const interactions = Array.isArray(sceneProperties.interactions)
      ? sceneProperties.interactions
      : [];
    const interaction = interactions
      .map(record)
      .find(
        (candidate) =>
          string(candidate.event) === "select" &&
          string(candidate.target) === id &&
          string(candidate.action) === "inspect",
      );
    if (interaction !== undefined) {
      registry.require("core.inspect");
      dispatch({ type: "open-inspector", nodeId: id });
    }
  }

  const context = {} as RenderContext;
  const renderNode = (node: RenderNodeIr): ReactNode => {
    const fallback = string(record(node).fallback, "Scene content unavailable.");
    try {
      return registry.require(resolveCapabilityId(node)).render(node, context);
    } catch {
      return (
        <foreignObject height="100%" key={nodeId(node)} width="100%">
          <div className="aiperf-flow__node-fallback" role="note">
            {fallback}
          </div>
        </foreignObject>
      );
    }
  };
  Object.assign(context, {
    state: renderState,
    timeline,
    dispatch,
    activateNode,
    renderNode,
    nodeById: (id: string) => nodes.get(id),
  });

  const accessibility = record(sceneProperties.accessibility);
  const inspectedNode =
    state.inspector.nodeId === null
      ? undefined
      : nodes.get(state.inspector.nodeId);

  const stage = measureRuntimePhase("total", () => {
    const roots = measureRuntimePhase("evaluation", () =>
      renderTree.map(renderNode),
    );
    return measureRuntimePhase("draw", () => (
      <div className="aiperf-flow__scene">
        <svg
          aria-label={string(
            accessibility.label,
            string(sceneProperties.title, "Flow scene"),
          )}
          className="aiperf-flow__stage"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          viewBox="0 0 640 360"
        >
          <desc>{string(sceneProperties.summary)}</desc>
          {roots}
        </svg>
        {state.inspector.open && inspectedNode !== undefined ? (
          <aside
            aria-label="Node inspector"
            className="aiperf-flow__inspector"
            ref={inspectorRef}
            role="region"
            tabIndex={-1}
          >
            <strong>
              {string(
                nodeAccessibility(inspectedNode).label,
                nodeId(inspectedNode),
              )}
            </strong>
            <p>{string(nodeAccessibility(inspectedNode).description)}</p>
            <button
              onClick={() => dispatch({ type: "close-inspector" })}
              type="button"
            >
              Close inspector
            </button>
          </aside>
        ) : null}
      </div>
    ));
  });

  return stage;
}
