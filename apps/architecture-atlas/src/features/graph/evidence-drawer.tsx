// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, type RefObject } from "react";

import type { AudienceLevel, GraphEdge, GraphNode } from "../../domain/architecture";
import {
  REPOSITORY_SOURCE_BASE_URL,
  repositorySource,
} from "../guided/source-url";

export interface NodeEvidenceDrawerEntity {
  kind: "node";
  node: GraphNode;
  relatedEdges?: readonly GraphEdge[];
}

export interface EdgeEvidenceDrawerEntity {
  kind: "edge";
  edge: GraphEdge;
  sourceNode?: GraphNode;
  targetNode?: GraphNode;
}

export type EvidenceDrawerEntity =
  | NodeEvidenceDrawerEntity
  | EdgeEvidenceDrawerEntity;

export interface EvidenceDrawerProps {
  audience: AudienceLevel;
  entity: EvidenceDrawerEntity | null;
  fallbackFocusRef: RefObject<HTMLElement | null>;
  onClose(): void;
  getTriggerElement?(entityId: string): HTMLElement | null;
  sourceBaseUrl?: string;
}

function evidenceRoleLabel(role: "source" | "design" | undefined): string {
  return role === "design" ? "Design" : "Source";
}

function hasVisibleTrigger(element: HTMLElement | null | undefined): element is HTMLElement {
  if (!element || !element.isConnected) {
    return false;
  }
  if (element.closest("details:not([open])")) {
    return false;
  }
  return true;
}

export function EvidenceDrawer({
  audience,
  entity,
  fallbackFocusRef,
  onClose,
  getTriggerElement,
  sourceBaseUrl = REPOSITORY_SOURCE_BASE_URL,
}: EvidenceDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const entityId = entity?.kind === "node" ? entity.node.id : entity?.edge.id;

  const restoreFocus = () => {
    const focusTarget = () => {
      const trigger = entityId ? getTriggerElement?.(entityId) : undefined;
      const visibleTrigger = hasVisibleTrigger(trigger) ? trigger : undefined;
      (visibleTrigger ?? fallbackFocusRef.current)?.focus();
    };
    window.requestAnimationFrame(() => {
      focusTarget();
      window.setTimeout(() => {
        const activeElement = document.activeElement;
        if (
          !activeElement ||
          activeElement === document.body ||
          !activeElement.isConnected
        ) {
          focusTarget();
        }
      }, 100);
    });
  };

  const closeAndRestoreFocus = () => {
    onClose();
    restoreFocus();
  };

  useEffect(() => {
    if (!entity) {
      return undefined;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      event.preventDefault();
      closeAndRestoreFocus();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  useEffect(() => {
    if (!entity) {
      return;
    }
    closeRef.current?.focus();
  }, [entity]);

  const contracts = useMemo(() => {
    if (!entity) {
      return [];
    }
    if (entity.kind === "edge") {
      return [entity.edge.protocol];
    }
    const contractsFromEdges = (entity.relatedEdges ?? []).map(
      ({ protocol }) => protocol,
    );
    return [...new Set(contractsFromEdges)];
  }, [entity]);

  if (!entity) {
    return null;
  }

  const title =
    entity.kind === "node" ? entity.node.title[audience] : entity.edge.id;
  const summary =
    entity.kind === "node"
      ? entity.node.summary[audience]
      : `${entity.sourceNode?.title[audience] ?? entity.edge.source.nodeId} -> ${entity.targetNode?.title[audience] ?? entity.edge.target.nodeId}`;
  const status =
    entity.kind === "node" ? entity.node.status : entity.edge.status;
  const flavors =
    entity.kind === "node" ? entity.node.flavors : entity.edge.flavors;
  const evidence =
    entity.kind === "node" ? entity.node.evidence : entity.edge.evidence;
  const footnotes =
    entity.kind === "node" ? entity.node.footnotes : entity.edge.footnotes;

  const sourcePortName =
    entity.kind === "edge"
      ? entity.sourceNode?.seamPorts.find(
          ({ id }) => id === entity.edge.source.portId,
        )?.name ?? entity.edge.source.portId
      : undefined;
  const targetPortName =
    entity.kind === "edge"
      ? entity.targetNode?.seamPorts.find(
          ({ id }) => id === entity.edge.target.portId,
        )?.name ?? entity.edge.target.portId
      : undefined;

  return (
    <aside
      aria-label={`${title} evidence`}
      aria-modal="false"
      className="graph-evidence-drawer"
      role="dialog"
    >
      <header>
        <button
          aria-label="Close evidence panel"
          onClick={closeAndRestoreFocus}
          ref={closeRef}
          type="button"
        >
          Close
        </button>
        <h2>{title}</h2>
        <p>{summary}</p>
      </header>

      <dl>
        <div>
          <dt>Status</dt>
          <dd>{status.state} / {status.delivery}</dd>
        </div>
        <div>
          <dt>Flavors</dt>
          <dd>{flavors.join(", ")}</dd>
        </div>
        {entity.kind === "edge" ? (
          <>
            <div>
              <dt>Direction</dt>
              <dd>
                {entity.sourceNode?.title[audience] ?? entity.edge.source.nodeId}
                {" -> "}
                {entity.targetNode?.title[audience] ?? entity.edge.target.nodeId}
              </dd>
            </div>
            <div>
              <dt>Ports</dt>
              <dd>
                {sourcePortName} {"->"} {targetPortName}
              </dd>
            </div>
            <div>
              <dt>Flow channel</dt>
              <dd>{entity.edge.channel}</dd>
            </div>
          </>
        ) : null}
      </dl>

      {entity.kind === "node" ? (
        <section>
          <h3>Ports</h3>
          <ul>
            {entity.node.seamPorts.map((port) => (
              <li key={port.id}>
                {port.name} ({port.channel})
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {contracts.length > 0 ? (
        <section>
          <h3>Contracts</h3>
          <ul>
            {contracts.map((contract) => (
              <li key={contract}>{contract}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <section>
        <h3>Evidence</h3>
        <ul>
          {evidence.map((item, index) => {
            const link = repositorySource(item, sourceBaseUrl);
            const key = `${item.path}:${index}`;
            return (
              <li key={key}>
                {link.href ? (
                  <a href={link.href} target="_blank" rel="noreferrer">
                    {link.label}
                  </a>
                ) : (
                  <span>{link.label}</span>
                )}
                {" "}
                <span>({evidenceRoleLabel(item.role)})</span>
              </li>
            );
          })}
        </ul>
      </section>

      {footnotes.length > 0 ? (
        <section>
          <h3>Footnotes</h3>
          <ul>
            {footnotes.map((footnote, index) => (
              <li key={`${title}-footnote-${String(index)}`}>
                {footnote[audience]}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </aside>
  );
}
