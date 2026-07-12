// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { architectureCatalog } from "../../content";
import type { Audience } from "../../domain/audience";
import type {
  ArchitectureComponent,
  ArchitectureStatus,
  ExecutionMode,
} from "../../domain/architecture";
import type { GuidedRoute } from "../../domain/routes";
import {
  FlowLane,
  GuidedFilters,
  GuidedSection,
  ModeComparison,
  OwnershipBands,
  ParityLedger,
  SeamDiagram,
  StageRail,
  StatusLegend,
  parityRecords,
} from "./primitives";

const dataPlaneFlowIds = new Set([
  "component.dataset-pipeline",
  "component.segment-store",
  "component.endpoint-registry",
]);
const dataPlaneBoundaryIds = new Set([
  "component.graph-ir",
  "component.content-server",
  "component.exact-token-ids",
]);
const observabilityFlowIds = new Set([
  "component.native-metrics",
  "component.live-telemetry",
  "component.telemetry-archive",
]);
const observabilityBoundaryIds = new Set([
  "component.static-accuracy",
  "component.agentic-evaluation",
  "component.provider-evaluation",
]);

interface GuidedViewProps {
  audience: Audience;
  modes: readonly ExecutionMode[];
  route: GuidedRoute;
  statuses: readonly ArchitectureStatus[];
  onModesChange(values: ExecutionMode[]): void;
  onStatusesChange(values: ArchitectureStatus[]): void;
}

function routeComponents(route: GuidedRoute): ArchitectureComponent[] {
  const view = architectureCatalog.views.find((candidate) => candidate.route === route);
  if (!view) {
    return [];
  }
  return view.componentIds.flatMap((id) => {
    const component = architectureCatalog.components.find(
      (candidate) => candidate.id === id,
    );
    return component ? [component] : [];
  });
}

function matchesFilters(
  component: ArchitectureComponent,
  modes: readonly ExecutionMode[],
  statuses: readonly ArchitectureStatus[],
): boolean {
  const matchesMode =
    modes.length === 0 ||
    modes.some((mode) => component.modes.includes(mode));
  const matchesStatus =
    statuses.length === 0 || statuses.includes(component.status);
  return matchesMode && matchesStatus;
}

function selectComponents(
  components: readonly ArchitectureComponent[],
  ids: ReadonlySet<string>,
): ArchitectureComponent[] {
  return components.filter((component) => ids.has(component.id));
}

function ViewHeader({
  audience,
  route,
}: Pick<GuidedViewProps, "audience" | "route">) {
  const view = architectureCatalog.views.find((candidate) => candidate.route === route);
  if (!view) {
    return null;
  }
  const routePosition =
    architectureCatalog.views
      .filter(({ route: candidate }) => candidate !== "/atlas")
      .findIndex(({ route: candidate }) => candidate === route) + 1;
  return (
    <header className="guided-header">
      <p className="route-eyebrow">
        Guided view {String(routePosition).padStart(2, "0")}
      </p>
      <h1 id="route-title">{view.title[audience]}</h1>
      <p className="route-summary">{view.summary[audience]}</p>
    </header>
  );
}

function OwnershipView({
  audience,
  components,
  modes,
  statuses,
  onModesChange,
  onStatusesChange,
}: Omit<GuidedViewProps, "route"> & {
  components: readonly ArchitectureComponent[];
}) {
  return (
    <>
      <GuidedFilters
        modes={modes}
        onModesChange={onModesChange}
        onStatusesChange={onStatusesChange}
        resultSummary={`${components.length} components`}
        statuses={statuses}
      />
      <StageRail
        audience={audience}
        stages={architectureCatalog.lifecycleStages}
      />
      <GuidedSection title="Accountability boundaries">
        <OwnershipBands audience={audience} components={components} />
      </GuidedSection>
    </>
  );
}

function JourneyView({ audience }: { audience: Audience }) {
  return (
    <>
      <StageRail
        audience={audience}
        stages={architectureCatalog.lifecycleStages}
      />
      <GuidedSection title="Handoffs that fail closed">
        <OwnershipBands
          audience={audience}
          components={routeComponents("/journey")}
        />
      </GuidedSection>
    </>
  );
}

function ExecutionView({
  audience,
  components,
  modes,
  statuses,
  onModesChange,
  onStatusesChange,
}: Omit<GuidedViewProps, "route"> & {
  components: readonly ArchitectureComponent[];
}) {
  const pairs = architectureCatalog.pairSupport.filter(
    (pair) =>
      (modes.length === 0 || modes.includes(pair.mode)) &&
      (statuses.length === 0 || statuses.includes(pair.status)),
  );
  const visibleComponentIds = new Set(components.map(({ id }) => id));
  const edges = architectureCatalog.edges
    .filter(
      (edge) =>
        edge.id.startsWith("edge.scheduling") ||
        edge.id.startsWith("edge.clock") ||
        edge.id.startsWith("edge.controls"),
    )
    .filter(
      (edge) =>
        visibleComponentIds.has(edge.from) &&
        visibleComponentIds.has(edge.to) &&
        (statuses.length === 0 || statuses.includes(edge.status)),
    );
  return (
    <>
      <GuidedFilters
        modes={modes}
        onModesChange={onModesChange}
        onStatusesChange={onStatusesChange}
        resultSummary={`${components.length} components, ${edges.length} connections, ${pairs.length} pairs`}
        statuses={statuses}
      />
      <SeamDiagram
        audience={audience}
        components={components}
        edges={edges}
      />
      <GuidedSection title="Executable pair matrix">
        <ModeComparison audience={audience} pairs={pairs} />
      </GuidedSection>
    </>
  );
}

function DataPlaneView({
  audience,
  components,
  modes,
  statuses,
  onModesChange,
  onStatusesChange,
}: Omit<GuidedViewProps, "route"> & {
  components: readonly ArchitectureComponent[];
}) {
  const flow = selectComponents(components, dataPlaneFlowIds);
  const boundaries = selectComponents(components, dataPlaneBoundaryIds);
  return (
    <>
      <GuidedFilters
        modes={modes}
        onModesChange={onModesChange}
        onStatusesChange={onStatusesChange}
        resultSummary={`${components.length} components`}
        statuses={statuses}
      />
      <FlowLane
        audience={audience}
        components={flow}
        label="Request shaping flow"
      />
      <GuidedSection title="Branch and representation boundaries">
        <OwnershipBands audience={audience} components={boundaries} />
      </GuidedSection>
    </>
  );
}

function ObservabilityView({
  audience,
  components,
  modes,
  statuses,
  onModesChange,
  onStatusesChange,
}: Omit<GuidedViewProps, "route"> & {
  components: readonly ArchitectureComponent[];
}) {
  const flow = selectComponents(components, observabilityFlowIds);
  const boundaries = selectComponents(components, observabilityBoundaryIds);
  return (
    <>
      <GuidedFilters
        modes={modes}
        onModesChange={onModesChange}
        onStatusesChange={onStatusesChange}
        resultSummary={`${components.length} components`}
        statuses={statuses}
      />
      <FlowLane
        audience={audience}
        components={flow}
        label="Measurement and evaluation flow"
      />
      <GuidedSection title="Native evidence and evaluator ownership">
        <OwnershipBands audience={audience} components={boundaries} />
      </GuidedSection>
    </>
  );
}

function ParityView({
  audience,
  modes,
  statuses,
  onModesChange,
  onStatusesChange,
}: Omit<GuidedViewProps, "route">) {
  const records = parityRecords(
    audience,
    architectureCatalog.risks,
    architectureCatalog.pairSupport,
    architectureCatalog.components,
  ).filter(
    (record) =>
      (modes.length === 0 ||
        modes.some((mode) => record.modes.includes(mode))) &&
      (statuses.length === 0 || statuses.includes(record.status)),
  );
  return (
    <>
      <StatusLegend />
      <GuidedFilters
        modes={modes}
        onModesChange={onModesChange}
        onStatusesChange={onStatusesChange}
        resultSummary={`${records.length} results`}
        statuses={statuses}
      />
      <ParityLedger audience={audience} records={records} />
    </>
  );
}

export function GuidedView({
  audience,
  modes,
  route,
  statuses,
  onModesChange,
  onStatusesChange,
}: GuidedViewProps) {
  const components = routeComponents(route).filter((component) =>
    matchesFilters(component, modes, statuses),
  );
  const view = architectureCatalog.views.find((candidate) => candidate.route === route);
  if (!view) {
    return null;
  }
  return (
    <section
      aria-label={view.title[audience]}
      className={`route-stage audience-${audience}`}
    >
      <ViewHeader audience={audience} route={route} />
      {route === "/" ? (
        <OwnershipView
          audience={audience}
          components={components}
          modes={modes}
          onModesChange={onModesChange}
          onStatusesChange={onStatusesChange}
          statuses={statuses}
        />
      ) : null}
      {route === "/journey" ? <JourneyView audience={audience} /> : null}
      {route === "/execution" ? (
        <ExecutionView
          audience={audience}
          components={components}
          modes={modes}
          onModesChange={onModesChange}
          onStatusesChange={onStatusesChange}
          statuses={statuses}
        />
      ) : null}
      {route === "/data-plane" ? (
        <DataPlaneView
          audience={audience}
          components={components}
          modes={modes}
          onModesChange={onModesChange}
          onStatusesChange={onStatusesChange}
          statuses={statuses}
        />
      ) : null}
      {route === "/observability" ? (
        <ObservabilityView
          audience={audience}
          components={components}
          modes={modes}
          onModesChange={onModesChange}
          onStatusesChange={onStatusesChange}
          statuses={statuses}
        />
      ) : null}
      {route === "/parity" ? (
        <ParityView
          audience={audience}
          modes={modes}
          onModesChange={onModesChange}
          onStatusesChange={onStatusesChange}
          statuses={statuses}
        />
      ) : null}
    </section>
  );
}
