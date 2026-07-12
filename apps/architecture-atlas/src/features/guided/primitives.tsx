// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";

import type { Audience } from "../../domain/audience";
import type {
  ArchitectureComponent,
  ArchitectureEdge,
  ArchitectureRisk,
  ArchitectureStatus,
  EvidenceReference,
  ExecutionMode,
  LifecycleStage,
  Ownership,
  PairSupport,
  Workload,
} from "../../domain/architecture";
import {
  REPOSITORY_SOURCE_BASE_URL,
  repositorySource,
} from "./source-url";

export const modeLabels: Record<ExecutionMode, string> = {
  online_http: "Native HTTP",
  online_grpc: "Native gRPC",
  online_mock: "Online mock",
  dynamo_offline: "Dynamo offline",
};

export const statusLabels: Record<ArchitectureStatus, string> = {
  built: "Built",
  "feature-gated": "Feature-gated",
  "runtime-conditional": "Runtime-conditional",
  "compatibility-only": "Compatibility-only",
  "legacy-parallel": "Legacy-parallel",
  unbuilt: "Unbuilt",
};

const workloadLabels: Record<Workload, string> = {
  scheduled: "Scheduled",
  graph: "Graph",
  static_accuracy: "Static accuracy",
  agentic: "Agentic",
  evaluation: "Evaluation",
  telemetry_watch: "Telemetry watch",
};

const ownershipLabels: Record<Ownership, string> = {
  python: "Product control",
  rust: "Run execution",
  external: "External peers",
  legacy: "Retained semantics",
};

const ownerOrder: Ownership[] = ["python", "rust", "external", "legacy"];
const statusOrder = Object.keys(statusLabels) as ArchitectureStatus[];
const modeOrder = Object.keys(modeLabels) as ExecutionMode[];

interface EvidenceCitationProps {
  evidence: readonly EvidenceReference[];
}

export function EvidenceCitation({ evidence }: EvidenceCitationProps) {
  return (
    <footer className="evidence-citations" aria-label="Source evidence">
      {evidence.map((reference) => {
        const source = repositorySource(
          reference,
          REPOSITORY_SOURCE_BASE_URL,
        );
        return source.href ? (
          <a href={source.href} key={source.label}>
            {source.label}
          </a>
        ) : (
          <span key={source.label}>{source.label}</span>
        );
      })}
    </footer>
  );
}

interface EntityBodyProps {
  audience: Audience;
  entity: ArchitectureComponent | ArchitectureRisk | LifecycleStage;
  headingLevel: 2 | 3 | 4;
}

function EntityBody({
  audience,
  entity,
  headingLevel,
}: EntityBodyProps) {
  const contracts = "contracts" in entity ? entity.contracts : [];
  const Heading = `h${headingLevel}` as "h2" | "h3" | "h4";
  return (
    <>
      <Heading>{entity.title[audience]}</Heading>
      <p>{entity.summary[audience]}</p>
      {audience === "developer" && contracts.length > 0 ? (
        <ul className="contract-list" aria-label="Integration contracts">
          {contracts.map((contract) => (
            <li key={contract}>{contract}</li>
          ))}
        </ul>
      ) : null}
      {audience === "maintainer" ? (
        <>
          {contracts.length > 0 ? (
            <p className="contract-line">{contracts.join(" · ")}</p>
          ) : null}
          <EvidenceCitation evidence={entity.evidence} />
        </>
      ) : null}
    </>
  );
}

interface StatusBadgeProps {
  status: ArchitectureStatus;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span className="status-badge" data-status={status}>
      {statusLabels[status]}
    </span>
  );
}

export function StatusLegend() {
  return (
    <ul className="status-legend" aria-label="Architecture status legend">
      {statusOrder.map((status) => (
        <li key={status}>
          <StatusBadge status={status} />
        </li>
      ))}
    </ul>
  );
}

interface StageRailProps {
  audience: Audience;
  stages: readonly LifecycleStage[];
}

export function StageRail({ audience, stages }: StageRailProps) {
  return (
    <ol className="stage-rail" aria-label="Product handoff sequence">
      {[...stages]
        .sort((left, right) => left.order - right.order)
        .map((stage) => (
          <li key={stage.id}>
            <span className="stage-index" aria-hidden="true">
              {String(stage.order + 1).padStart(2, "0")}
            </span>
            <EntityBody
              audience={audience}
              entity={stage}
              headingLevel={2}
            />
          </li>
        ))}
    </ol>
  );
}

interface OwnershipBandsProps {
  audience: Audience;
  components: readonly ArchitectureComponent[];
}

export function OwnershipBands({
  audience,
  components,
}: OwnershipBandsProps) {
  return (
    <div className="ownership-bands">
      {ownerOrder.map((owner) => {
        const owned = components.filter((component) => component.owner === owner);
        if (owned.length === 0) {
          return null;
        }
        return (
          <section
            className="ownership-band"
            data-owner={owner}
            key={owner}
            aria-labelledby={`owner-${owner}`}
          >
            <header>
              <span>{owned.length} systems</span>
            </header>
            <h3 id={`owner-${owner}`}>{ownershipLabels[owner]}</h3>
            <div className="band-entities">
              {owned.map((component) => (
                <article key={component.id}>
                  <StatusBadge status={component.status} />
                  <EntityBody
                    audience={audience}
                    entity={component}
                    headingLevel={4}
                  />
                </article>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

interface SeamDiagramProps {
  audience: Audience;
  components: readonly ArchitectureComponent[];
  edges: readonly ArchitectureEdge[];
}

export function SeamDiagram({
  audience,
  components,
  edges,
}: SeamDiagramProps) {
  return (
    <div className="seam-diagram">
      <div className="seam-nodes">
        {components.map((component) => (
          <article key={component.id} data-status={component.status}>
            <StatusBadge status={component.status} />
            <EntityBody
              audience={audience}
              entity={component}
              headingLevel={2}
            />
          </article>
        ))}
      </div>
      <ul className="seam-contracts" aria-label="Execution seam contracts">
        {edges.map((edge) => (
          <li key={edge.id}>
            <span>{edge.label}</span>
            {audience === "maintainer" ? (
              <code>
                {"protocol" in edge
                  ? edge.protocol
                  : "contract" in edge
                    ? edge.contract
                    : edge.control}
              </code>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

interface FlowLaneProps {
  audience: Audience;
  components: readonly ArchitectureComponent[];
  label: string;
}

export function FlowLane({
  audience,
  components,
  label,
}: FlowLaneProps) {
  return (
    <ol className="flow-lane" aria-label={label}>
      {components.map((component, index) => (
        <li key={component.id}>
          <span className="flow-index" aria-hidden="true">
            {index + 1}
          </span>
          <article>
            <StatusBadge status={component.status} />
            <EntityBody
              audience={audience}
              entity={component}
              headingLevel={2}
            />
          </article>
        </li>
      ))}
    </ol>
  );
}

interface ModeComparisonProps {
  audience: Audience;
  pairs: readonly PairSupport[];
}

export function ModeComparison({ audience, pairs }: ModeComparisonProps) {
  return (
    <div className="mode-comparison">
      {modeOrder.map((mode) => {
        const modePairs = pairs.filter((pair) => pair.mode === mode);
        return (
          <section key={mode} aria-labelledby={`mode-${mode}`}>
            <header>
              <h3 id={`mode-${mode}`}>{modeLabels[mode]}</h3>
              <span>{modePairs.length}</span>
            </header>
            <ul>
              {modePairs.map((pair) => (
                <li key={pair.id}>
                  <span>{workloadLabels[pair.workload]}</span>
                  <StatusBadge status={pair.status} />
                  <p>{pair.notes[audience]}</p>
                  {audience === "maintainer" ? (
                    <EvidenceCitation evidence={pair.evidence} />
                  ) : null}
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

interface FilterSetProps<T extends string> {
  legend: string;
  labels: Record<T, string>;
  options: readonly T[];
  selected: readonly T[];
  onChange(values: T[]): void;
}

function FilterSet<T extends string>({
  legend,
  labels,
  options,
  selected,
  onChange,
}: FilterSetProps<T>) {
  const toggle = (option: T) => {
    onChange(
      selected.includes(option)
        ? selected.filter((value) => value !== option)
        : [...selected, option],
    );
  };
  return (
    <fieldset>
      <legend>{legend}</legend>
      <div className="filter-options">
        {options.map((option) => (
          <label key={option}>
            <input
              checked={selected.includes(option)}
              onChange={() => toggle(option)}
              type="checkbox"
            />
            <span>{labels[option]}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

interface GuidedFiltersProps {
  modes: readonly ExecutionMode[];
  statuses: readonly ArchitectureStatus[];
  resultSummary: string;
  onModesChange(values: ExecutionMode[]): void;
  onStatusesChange(values: ArchitectureStatus[]): void;
}

export function GuidedFilters({
  modes,
  statuses,
  resultSummary,
  onModesChange,
  onStatusesChange,
}: GuidedFiltersProps) {
  return (
    <section className="guided-filters" aria-label="Architecture filters">
      <FilterSet
        labels={modeLabels}
        legend="Execution mode"
        onChange={onModesChange}
        options={modeOrder}
        selected={modes}
      />
      <FilterSet
        labels={statusLabels}
        legend="Architecture status"
        onChange={onStatusesChange}
        options={statusOrder}
        selected={statuses}
      />
      <p
        aria-label="Filtered result count"
        aria-live="polite"
        className="result-count"
        role="status"
      >
        {resultSummary}
      </p>
    </section>
  );
}

export interface ParityRecord {
  id: string;
  kind: "pair" | "gap";
  modes: ExecutionMode[];
  status: ArchitectureStatus;
  title: string;
  summary: string;
  evidence: EvidenceReference[];
}

export function parityRecords(
  audience: Audience,
  risks: readonly ArchitectureRisk[],
  pairs: readonly PairSupport[],
  components: readonly ArchitectureComponent[],
): ParityRecord[] {
  const riskRecords = risks.map((risk): ParityRecord => {
    const modes = [
      ...new Set(
        risk.componentIds.flatMap(
          (id) =>
            components.find((component) => component.id === id)?.modes ?? [],
        ),
      ),
    ];
    return {
      id: risk.id,
      kind: "gap",
      modes,
      status: risk.status,
      title: risk.title[audience],
      summary: risk.summary[audience],
      evidence: risk.evidence,
    };
  });
  const pairRecords = pairs.map(
    (pair): ParityRecord => ({
      id: pair.id,
      kind: "pair",
      modes: [pair.mode],
      status: pair.status,
      title: `${modeLabels[pair.mode]} · ${workloadLabels[pair.workload]}`,
      summary: pair.notes[audience],
      evidence: pair.evidence,
    }),
  );
  return [...riskRecords, ...pairRecords];
}

interface ParityLedgerProps {
  audience: Audience;
  records: readonly ParityRecord[];
}

export function ParityLedger({ audience, records }: ParityLedgerProps) {
  return (
    <ol className="parity-ledger" aria-label="Parity and migration entries">
      {records.map((record) => (
        <li key={record.id}>
          <div className="ledger-class">
            <span>{record.kind === "pair" ? "Executable pair" : "Boundary"}</span>
            <StatusBadge status={record.status} />
          </div>
          <article>
            <h2>{record.title}</h2>
            <p>{record.summary}</p>
            <div className="mode-tags">
              {record.modes.map((mode) => (
                <span key={mode}>{modeLabels[mode]}</span>
              ))}
            </div>
            {audience === "maintainer" ? (
              <EvidenceCitation evidence={record.evidence} />
            ) : null}
          </article>
        </li>
      ))}
    </ol>
  );
}

interface GuidedSectionProps {
  title: string;
  children: ReactNode;
}

export function GuidedSection({ title, children }: GuidedSectionProps) {
  const id = `section-${title.toLocaleLowerCase().replaceAll(/[^a-z0-9]+/gu, "-")}`;
  return (
    <section className="guided-section" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      {children}
    </section>
  );
}
