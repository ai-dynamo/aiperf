// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Link } from "@tanstack/react-router";
import { useState } from "react";

import { architectureCatalog } from "../../content";
import type { Audience } from "../../domain/audience";
import {
  deriveCrateDependents,
  searchCrates,
} from "../../domain/atlas-graph";
import type {
  ArchitectureComponent,
  CargoDependencyKind,
  CrateReference,
} from "../../domain/architecture";
import {
  canonicalGraphState,
  encodeGraphStateForUrl,
} from "../../domain/graph-state";
import {
  routeCapabilities,
  type AtlasRoutePath,
} from "../../domain/routes";
import {
  EvidenceCitation,
  StatusBadge,
  modeLabels,
} from "../guided/primitives";

interface CrateDirectoryProps {
  audience: Audience;
  current?: string;
}

function CrateDirectory({ audience, current }: CrateDirectoryProps) {
  const [query, setQuery] = useState("");
  const crates = searchCrates(architectureCatalog.crates, query);
  return (
    <nav aria-label="Crate directory" className="crate-directory">
      <label>
        <span>Find a crate</span>
        <input
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Package, contract, responsibility"
          type="search"
          value={query}
        />
      </label>
      <p aria-live="polite">{crates.length} crates</p>
      <ul>
        {crates.map((crate) => (
          <li key={crate.id}>
            <Link
              aria-current={crate.packageName === current ? "page" : undefined}
              params={{ crateId: crate.packageName }}
              search={{ audience }}
              to="/crates/$crateId"
            >
              <code>{crate.packageName}</code>
              <span>{crate.summary[audience]}</span>
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

const dependencyKindLabels: Record<CargoDependencyKind, string> = {
  normal: "Normal dependencies",
  build: "Build dependencies",
  dev: "Development dependencies",
};
const dependentKindLabels: Record<CargoDependencyKind, string> = {
  normal: "Normal dependents",
  build: "Build dependents",
  dev: "Development dependents",
};

function graphTargetFor(
  component: ArchitectureComponent,
  audience: Audience,
): {
  path: AtlasRoutePath;
  search: {
    audience: Audience;
    primary: "native_http" | "native_grpc" | "online_mock" | "dynamo_offline" | "dynamo_online";
    q: string;
    s: string;
  };
} {
  const evidencePaths = new Set(component.evidence.map(({ path }) => path));
  const graphNode = architectureCatalog.graphNodes.find((node) =>
    node.evidence.some(({ path }) => evidencePaths.has(path)),
  );
  const route = graphNode
    ? routeCapabilities.find(({ sceneId }) =>
        architectureCatalog.graphScenes
          .find((scene) => scene.id === sceneId)
          ?.nodeIds.includes(graphNode.id),
      )
    : undefined;
  const primary = graphNode?.flavors[0] ?? "native_http";
  const sceneId = route?.sceneId ?? "scene.runtime-composition";
  const q = graphNode?.title[audience] ?? component.title[audience];
  const state = canonicalGraphState({
    audience,
    focusedEntityId: graphNode?.id,
    primaryFlavor: primary,
    sceneId,
  });

  return {
    path: route?.path ?? "/",
    search: {
      audience,
      primary,
      q,
      s: encodeGraphStateForUrl(state),
    },
  };
}

function DependencySections({
  audience,
  crate,
}: {
  audience: Audience;
  crate: CrateReference;
}) {
  return (Object.keys(dependencyKindLabels) as CargoDependencyKind[]).map(
    (kind) => {
      const dependencies = crate.dependencies
        .filter((dependency) => dependency.kind === kind)
        .flatMap((dependency) => {
          const related = architectureCatalog.crates.find(
            ({ id }) => id === dependency.crateId,
          );
          return related ? [related] : [];
        });
      const label = dependencyKindLabels[kind];
      return (
        <section aria-label={label} key={kind}>
          <h2>{label}</h2>
          {dependencies.length > 0 ? (
            <ul className="reference-list">
              {dependencies.map((dependency) => (
                <li key={dependency.id}>
                  <Link
                    params={{ crateId: dependency.packageName }}
                    search={{ audience }}
                    to="/crates/$crateId"
                  >
                    {dependency.packageName}
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p>None.</p>
          )}
        </section>
      );
    },
  );
}

function DependentSections({
  audience,
  crate,
}: {
  audience: Audience;
  crate: CrateReference;
}) {
  const relationships = deriveCrateDependents(
    architectureCatalog.crates,
    crate.id,
  );
  return (Object.keys(dependentKindLabels) as CargoDependencyKind[]).map(
    (kind) => {
      const dependents = relationships.filter(
        (relationship) => relationship.kind === kind,
      );
      const label = dependentKindLabels[kind];
      return (
        <section aria-label={label} key={kind}>
          <h2>{label}</h2>
          {dependents.length > 0 ? (
            <ul className="reference-list">
              {dependents.map(({ crate: dependent }) => (
                <li key={dependent.id}>
                  <Link
                    params={{ crateId: dependent.packageName }}
                    search={{ audience }}
                    to="/crates/$crateId"
                  >
                    {dependent.packageName}
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p>None.</p>
          )}
        </section>
      );
    },
  );
}

export function CrateReferenceView({
  audience,
  crateId,
}: {
  audience: Audience;
  crateId: string;
}) {
  const crate = architectureCatalog.crates.find(
    ({ packageName }) => packageName === crateId,
  );
  if (!crate) {
    const first = architectureCatalog.crates[0];
    return (
      <section className="reference-route">
        <header className="guided-header">
          <p className="route-eyebrow">Crate reference</p>
          <h1>Crate not found</h1>
          <p className="route-summary">
            No typed crate record matches <code>{crateId}</code>.
          </p>
        </header>
        <div className="not-found-actions">
          {first ? (
            <Link
              params={{ crateId: first.packageName }}
              search={{ audience }}
              to="/crates/$crateId"
            >
              Browse crate directory
            </Link>
          ) : null}
          <Link search={{ audience }} to="/atlas">
            Open unified atlas
          </Link>
        </div>
        <CrateDirectory audience={audience} />
      </section>
    );
  }

  const components = architectureCatalog.components.filter((component) =>
    component.crateIds.includes(crate.id),
  );

  return (
    <section className={`reference-route audience-${audience}`}>
      <header className="guided-header">
        <p className="route-eyebrow">Crate reference / {crate.status}</p>
        <h1>{crate.title[audience]}</h1>
        <p className="route-summary">{crate.responsibility[audience]}</p>
      </header>
      <div className="reference-layout">
        <CrateDirectory
          audience={audience}
          current={crate.packageName}
        />
        <article className="crate-dossier">
          <div className="dossier-lead">
            <StatusBadge status={crate.status} />
            <p>{crate.summary[audience]}</p>
          </div>
          <section>
            <h2>Contracts and seams</h2>
            <ul className="reference-list">
              {crate.contracts.map((contract) => (
                <li key={contract}>{contract}</li>
              ))}
            </ul>
          </section>
          <section>
            <h2>Supported modes</h2>
            <ul className="reference-list">
              {crate.modes.map((mode) => (
                <li key={mode}>{modeLabels[mode]}</li>
              ))}
            </ul>
          </section>
          <DependencySections audience={audience} crate={crate} />
          <DependentSections audience={audience} crate={crate} />
          <section>
            <h2>Related components</h2>
            {components.length > 0 ? (
              <ul className="reference-list">
                {components.map((component) => (
                  <li key={component.id}>
                    {(() => {
                      const target = graphTargetFor(component, audience);
                      return (
                        <Link search={target.search} to={target.path}>
                          {component.title[audience]}
                        </Link>
                      );
                    })()}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No component record links directly to this crate.</p>
            )}
          </section>
          <section>
            <h2>Key source paths</h2>
            <EvidenceCitation
              evidence={crate.keySourcePaths.map((path) => ({ path }))}
            />
          </section>
          <section>
            <h2>Parity scars</h2>
            {crate.parityScars.length > 0 ? (
              <ul className="reference-list">
                {crate.parityScars.map((scar) => (
                  <li key={scar}>{scar}</li>
                ))}
              </ul>
            ) : (
              <p>No crate-specific scar is recorded.</p>
            )}
          </section>
          <EvidenceCitation evidence={crate.evidence} />
        </article>
      </div>
    </section>
  );
}
