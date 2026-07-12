// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Link } from "@tanstack/react-router";
import { useState } from "react";

import { architectureCatalog } from "../../content";
import type { Audience } from "../../domain/audience";
import { searchCrates } from "../../domain/atlas-graph";
import type { CrateReference } from "../../domain/architecture";
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

function RelationList({
  audience,
  crates,
  label,
}: {
  audience: Audience;
  crates: readonly CrateReference[];
  label: string;
}) {
  return (
    <section>
      <h2>{label}</h2>
      {crates.length > 0 ? (
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
      ) : (
        <p>None in the workspace catalog.</p>
      )}
    </section>
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

  const dependencies = crate.dependencyCrateIds.flatMap((id) => {
    const dependency = architectureCatalog.crates.find(
      (candidate) => candidate.id === id,
    );
    return dependency ? [dependency] : [];
  });
  const dependents = architectureCatalog.crates.filter((candidate) =>
    candidate.dependencyCrateIds.includes(crate.id),
  );
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
          <RelationList
            audience={audience}
            crates={dependencies}
            label="Dependencies"
          />
          <RelationList
            audience={audience}
            crates={dependents}
            label="Dependents"
          />
          <section>
            <h2>Related components</h2>
            {components.length > 0 ? (
              <ul className="reference-list">
                {components.map((component) => (
                  <li key={component.id}>
                    <Link
                      search={{
                        audience,
                        selected: component.id,
                      }}
                      to="/atlas"
                    >
                      {component.title[audience]}
                    </Link>
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
