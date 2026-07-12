// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  BaseEdge,
  Background,
  Controls,
  EdgeLabelRenderer,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
  type ReactFlowInstance,
} from "@xyflow/react";
import { Link } from "@tanstack/react-router";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useRef, useState } from "react";

import { architectureCatalog } from "../../content";
import type { Audience } from "../../domain/audience";
import {
  dependencyNeighborhood,
  deriveAtlasGraph,
} from "../../domain/atlas-graph";
import type {
  ArchitectureComponent,
  ArchitectureEdge,
  ArchitectureStatus,
  ExecutionMode,
  Ownership,
} from "../../domain/architecture";
import {
  EvidenceCitation,
  StatusBadge,
  modeLabels,
  statusLabels,
} from "../guided/primitives";
import { layoutAtlas, type LayoutPerspective } from "./layout";

interface AtlasRouteState {
  layout: LayoutPerspective;
  modes: readonly ExecutionMode[];
  owners: readonly Ownership[];
  query: string;
  selected?: string;
  statuses: readonly ArchitectureStatus[];
}

interface AtlasViewProps {
  audience: Audience;
  state: AtlasRouteState;
  onStateChange(change: Partial<AtlasRouteState>): void;
}

interface ComponentNodeData extends Record<string, unknown> {
  audience: Audience;
  component: ArchitectureComponent;
  dimmed: boolean;
  selected: boolean;
  select(id: string): void;
}

const modeOrder = Object.keys(modeLabels) as ExecutionMode[];
const statusOrder = Object.keys(statusLabels) as ArchitectureStatus[];
const ownerOrder: Ownership[] = ["python", "rust", "external", "legacy"];
const ownerLabels: Record<Ownership, string> = {
  python: "Python product",
  rust: "Rust execution",
  external: "External peer",
  legacy: "Legacy semantics",
};

function ComponentNode({ data }: NodeProps<Node<ComponentNodeData>>) {
  return (
    <article
      className="atlas-node"
      data-dimmed={data.dimmed}
      data-owner={data.component.owner}
      data-selected={data.selected}
    >
      <Handle position={Position.Left} type="target" />
      <button
        aria-pressed={data.selected}
        onClick={() => data.select(data.component.id)}
        type="button"
      >
        <span className="atlas-node-owner">{ownerLabels[data.component.owner]}</span>
        <strong>{data.component.title[data.audience]}</strong>
        <span>{data.component.summary[data.audience]}</span>
        <StatusBadge status={data.component.status} />
      </button>
      <Handle position={Position.Right} type="source" />
    </article>
  );
}

const nodeTypes = { component: ComponentNode };

interface SemanticEdgeData extends Record<string, unknown> {
  edge: ArchitectureEdge;
}

function SemanticEdge({
  data,
  id,
  markerEnd,
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
}: EdgeProps<Edge<SemanticEdgeData>>) {
  const [path, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  return (
    <>
      <BaseEdge
        id={id}
        markerEnd={markerEnd}
        path={path}
      />
      {data ? (
        <EdgeLabelRenderer>
          <span
            className="atlas-edge-label"
            data-kind={data.edge.kind}
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {data.edge.label}
          </span>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

const edgeTypes = { semantic: SemanticEdge };

function FilterGroup<T extends string>({
  label,
  options,
  selected,
  labels,
  onChange,
}: {
  label: string;
  options: readonly T[];
  selected: readonly T[];
  labels: Record<T, string>;
  onChange(values: T[]): void;
}) {
  return (
    <fieldset>
      <legend>{label}</legend>
      <div className="filter-options">
        {options.map((option) => (
          <label key={option}>
            <input
              checked={selected.includes(option)}
              onChange={() =>
                onChange(
                  selected.includes(option)
                    ? selected.filter((value) => value !== option)
                    : [...selected, option],
                )
              }
              type="checkbox"
            />
            <span>{labels[option]}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function ComponentDrawer({
  audience,
  component,
  close,
  upstream,
  downstream,
}: {
  audience: Audience;
  component: ArchitectureComponent;
  close(): void;
  upstream: readonly string[];
  downstream: readonly string[];
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
  }, [component.id]);
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [close]);

  const inbound = architectureCatalog.edges.filter(
    ({ to }) => to === component.id,
  );
  const outbound = architectureCatalog.edges.filter(
    ({ from }) => from === component.id,
  );
  const crates = architectureCatalog.crates.filter(({ id }) =>
    component.crateIds.includes(id),
  );
  const risks = architectureCatalog.risks.filter(({ componentIds }) =>
    componentIds.includes(component.id),
  );

  return (
    <aside
      aria-label={component.title[audience]}
      aria-modal="false"
      className="atlas-drawer"
      role="dialog"
    >
      <header>
        <p className="route-eyebrow">{ownerLabels[component.owner]}</p>
        <button
          aria-label="Clear selected component"
          onClick={close}
          ref={closeRef}
          type="button"
        >
          Close
        </button>
        <h2>{component.title[audience]}</h2>
        <p>{component.summary[audience]}</p>
      </header>
      <dl className="detail-facts">
        <div>
          <dt>Status</dt>
          <dd>{statusLabels[component.status]}</dd>
        </div>
        <div>
          <dt>Modes</dt>
          <dd>{component.modes.map((mode) => modeLabels[mode]).join(", ") || "None"}</dd>
        </div>
        <div>
          <dt>Trace</dt>
          <dd>{upstream.length} upstream · {downstream.length} downstream</dd>
        </div>
      </dl>
      <section>
        <h3>Contracts</h3>
        <ul className="reference-list">
          {component.contracts.map((contract) => (
            <li key={contract}>{contract}</li>
          ))}
        </ul>
      </section>
      <section>
        <h3>Related crates</h3>
        <ul className="reference-list">
          {crates.map((crate) => (
            <li key={crate.id}>
              <Link
                params={{ crateId: crate.packageName }}
                search={{ audience }}
                to="/crates/$crateId"
              >
                {crate.packageName}
              </Link>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h3>Inbound messages</h3>
        <ul className="reference-list">
          {inbound.map((edge) => (
            <li key={edge.id}>
              {edge.label} —{" "}
              {"protocol" in edge
                ? edge.protocol
                : "contract" in edge
                  ? edge.contract
                  : edge.control}
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h3>Outbound messages</h3>
        <ul className="reference-list">
          {outbound.map((edge) => (
            <li key={edge.id}>
              {edge.label} —{" "}
              {"protocol" in edge
                ? edge.protocol
                : "contract" in edge
                  ? edge.contract
                  : edge.control}
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h3>Parity risks</h3>
        {risks.length > 0 ? (
          <ul className="reference-list">
            {risks.map((risk) => (
              <li key={risk.id}>{risk.title[audience]} — {risk.summary[audience]}</li>
            ))}
          </ul>
        ) : (
          <p>No linked parity risk.</p>
        )}
      </section>
      <EvidenceCitation evidence={component.evidence} />
    </aside>
  );
}

export function AtlasView({
  audience,
  state,
  onStateChange,
}: AtlasViewProps) {
  const graph = useMemo(
    () =>
      deriveAtlasGraph(architectureCatalog, {
        modes: state.modes,
        owners: state.owners,
        query: state.query,
        statuses: state.statuses,
      }),
    [state.modes, state.owners, state.query, state.statuses],
  );
  const selected = graph.components.find(({ id }) => id === state.selected);
  const neighborhood = useMemo(
    () =>
      selected
        ? dependencyNeighborhood(selected.id, graph.edges)
        : { upstream: [], downstream: [], related: new Set<string>() },
    [graph.edges, selected],
  );
  const [positions, setPositions] = useState(new Map<string, { x: number; y: number }>());
  const [instance, setInstance] = useState<ReactFlowInstance | null>(null);

  useEffect(() => {
    if (state.selected && !selected) {
      onStateChange({ selected: undefined });
    }
  }, [onStateChange, selected, state.selected]);

  useEffect(() => {
    let active = true;
    void layoutAtlas(
      graph.components,
      graph.edges,
      state.layout,
      architectureCatalog.lifecycleStages,
    ).then((layout) => {
      if (active) {
        setPositions(new Map(layout.map(({ id, x, y }) => [id, { x, y }])));
      }
    });
    return () => {
      active = false;
    };
  }, [graph.components, graph.edges, state.layout]);

  const nodes: Node<ComponentNodeData>[] = graph.components.map((component) => ({
    id: component.id,
    type: "component",
    position: positions.get(component.id) ?? { x: 0, y: 0 },
    data: {
      audience,
      component,
      dimmed: Boolean(selected && !neighborhood.related.has(component.id)),
      selected: selected?.id === component.id,
      select: (id) => onStateChange({ selected: id }),
    },
  }));
  const edges = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.from,
    target: edge.to,
    type: "semantic",
    data: { edge },
    className:
      selected &&
      (!neighborhood.related.has(edge.from) ||
        !neighborhood.related.has(edge.to))
        ? "atlas-edge-dimmed"
        : undefined,
  }));

  const reset = () => {
    onStateChange({
      layout: "ownership",
      modes: [],
      owners: [],
      query: "",
      selected: undefined,
      statuses: [],
    });
    window.setTimeout(() => void instance?.fitView({ padding: 0.14 }), 0);
  };

  return (
    <section className={`atlas-route audience-${audience}`}>
      <header className="guided-header">
        <p className="route-eyebrow">Unified view / System map</p>
        <h1>{architectureCatalog.views.find(({ route }) => route === "/atlas")?.title[audience]}</h1>
        <p className="route-summary">
          {architectureCatalog.views.find(({ route }) => route === "/atlas")?.summary[audience]}
        </p>
      </header>
      <div aria-label="Atlas controls" className="atlas-controls">
        <label className="atlas-search">
          <span>Search atlas</span>
          <input
            aria-label="Search atlas"
            onChange={(event) => onStateChange({ query: event.target.value })}
            placeholder="Label, summary, crate, contract"
            type="search"
            value={state.query}
          />
        </label>
        <label>
          <span>Layout perspective</span>
          <select
            aria-label="Layout perspective"
            onChange={(event) =>
              onStateChange({
                layout: event.target.value as LayoutPerspective,
              })
            }
            value={state.layout}
          >
            <option value="ownership">Ownership</option>
            <option value="lifecycle">Lifecycle</option>
          </select>
        </label>
        <button onClick={reset} type="button">Reset</button>
        <button
          onClick={() => void instance?.fitView({ padding: 0.14 })}
          type="button"
        >
          Fit view
        </button>
        <FilterGroup
          label="Modes"
          labels={modeLabels}
          onChange={(modes) => onStateChange({ modes })}
          options={modeOrder}
          selected={state.modes}
        />
        <FilterGroup
          label="Statuses"
          labels={statusLabels}
          onChange={(statuses) => onStateChange({ statuses })}
          options={statusOrder}
          selected={state.statuses}
        />
        <FilterGroup
          label="Ownership"
          labels={ownerLabels}
          onChange={(owners) => onStateChange({ owners })}
          options={ownerOrder}
          selected={state.owners}
        />
      </div>
      <p
        aria-label="Atlas graph summary"
        className="atlas-summary"
        role="status"
      >
        {graph.components.length} components, {graph.edges.length} connections
        {selected
          ? `; ${neighborhood.upstream.length} upstream and ${neighborhood.downstream.length} downstream of ${selected.title[audience]}`
          : ""}
      </p>
      <div className="atlas-workspace">
        <div aria-label="Architecture graph" className="atlas-canvas">
          <ReactFlow
            colorMode="dark"
            edges={edges}
            edgeTypes={edgeTypes}
            fitView
            minZoom={0.2}
            nodeTypes={nodeTypes}
            nodes={nodes}
            onInit={setInstance}
            onNodeClick={(_, node) => onStateChange({ selected: node.id })}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={28} size={1} />
            <Controls showInteractive={false} />
            <MiniMap
              ariaLabel="Architecture overview"
              maskColor="rgba(16, 18, 20, 0.76)"
              nodeColor="#6f7882"
            />
          </ReactFlow>
        </div>
        {selected ? (
          <ComponentDrawer
            audience={audience}
            close={() => onStateChange({ selected: undefined })}
            component={selected}
            downstream={neighborhood.downstream}
            upstream={neighborhood.upstream}
          />
        ) : null}
      </div>
      <details className="atlas-inventory">
        <summary>Text inventory</summary>
        <ul aria-label="Visible architecture components">
          {graph.components.map((component) => (
            <li key={component.id}>
              <button
                onClick={() => onStateChange({ selected: component.id })}
                type="button"
              >
                {component.title[audience]} — {ownerLabels[component.owner]} —{" "}
                {statusLabels[component.status]}
              </button>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
