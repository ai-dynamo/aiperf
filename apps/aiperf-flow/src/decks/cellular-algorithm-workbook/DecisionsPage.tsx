/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decision laboratory: neighbouring selector states compared side by side. Shared lifecycle bands
//! (common prefix / suffix of the two derived routes) collapse so the branch, storage tradeoff,
//! artifact boundary, or first rejecting gate stays visible. Ported from
//! `docs/canvases/cellular-algorithm-workbook.canvas.tsx` (DecisionLaboratory + its collapseSharedRoute
//! / decisionFacet helpers).

import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Code } from "../../prose/Code.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";
import {
  ALGORITHMS,
  DECISIONS,
  GATE_STAGE_LABELS,
  cachedRoute,
  type RouteResult,
  type SelectorState,
} from "./data.js";
import { AdmissionLabel, Eyebrow, Framed } from "./ui.js";

type RouteBands = {
  prefix: readonly string[];
  leftDelta: readonly string[];
  rightDelta: readonly string[];
  suffix: readonly string[];
};

function collapseSharedRoute(left: readonly string[], right: readonly string[]): RouteBands {
  let prefixLength = 0;
  while (
    prefixLength < left.length &&
    prefixLength < right.length &&
    left[prefixLength] === right[prefixLength]
  ) {
    prefixLength += 1;
  }
  let suffixLength = 0;
  while (
    suffixLength < left.length - prefixLength &&
    suffixLength < right.length - prefixLength &&
    left[left.length - suffixLength - 1] === right[right.length - suffixLength - 1]
  ) {
    suffixLength += 1;
  }
  const leftRemainder = left.slice(prefixLength, left.length - suffixLength);
  const rightRemainder = right.slice(prefixLength, right.length - suffixLength);
  const leftRemainderIds = new Set(leftRemainder);
  const rightRemainderIds = new Set(rightRemainder);
  return {
    prefix: left.slice(0, prefixLength),
    leftDelta: leftRemainder.filter((id) => !rightRemainderIds.has(id)),
    rightDelta: rightRemainder.filter((id) => !leftRemainderIds.has(id)),
    suffix: suffixLength === 0 ? [] : left.slice(left.length - suffixLength),
  };
}

type DecisionFacet = "effective" | "memory" | "fidelity" | "artifacts" | "limitations" | "rejection";

const DECISION_FACET_LABELS: Readonly<Record<DecisionFacet, string>> = {
  effective: "Effective settings",
  memory: "Memory",
  fidelity: "Fidelity",
  artifacts: "Artifacts",
  limitations: "Limitations",
  rejection: "Admission",
};

function decisionFacet(route: RouteResult, facet: DecisionFacet): string {
  if (facet === "effective") {
    return `workload=${route.effective.workload}; topology=${route.effective.topology}; storage=${route.effective.storage}; artifacts=${route.effective.artifacts}`;
  }
  if (!route.valid) {
    if (facet === "rejection") {
      return `${GATE_STAGE_LABELS[route.gateStage]} · ${route.rejectedBy}: ${route.reason}`;
    }
    if (facet === "limitations") return route.limitations.join(" ") || "None";
    return "Not reached";
  }
  if (facet === "limitations") return route.limitations.join(" ") || "None";
  if (facet === "rejection") return "Admitted";
  return route[facet];
}

function algorithmTitle(id: string): string {
  return ALGORITHMS.find((a) => a.id === id)?.title ?? id;
}

function SharedRouteBand({ label, algorithmIds }: { label: string; algorithmIds: readonly string[] }) {
  if (algorithmIds.length === 0) return null;
  return (
    <Framed>
      <span className={`text-sm ${inkClassName("secondary")}`}>
        {label} · {algorithmIds.length} shared stops
      </span>
    </Framed>
  );
}

function DecisionSide({
  label,
  route,
  delta,
  differingFacets,
}: {
  label: string;
  selection: SelectorState;
  route: RouteResult;
  delta: readonly string[];
  differingFacets: readonly DecisionFacet[];
}): React.JSX.Element {
  return (
    <section
      aria-label={label}
      className={`border-t pt-3 pl-3 ${strokeClassName("tertiary")}`}
      style={{ borderLeft: route.valid ? "2px solid var(--color-stroke-secondary)" : "2px solid var(--color-category-red)" }}
    >
      <Stack gap={12}>
        <Row gap={8} align="center" wrap justify="space-between">
          <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>{label}</h3>
          <AdmissionLabel valid={route.valid} />
        </Row>
        <Stack gap={6}>
          <Eyebrow>Distinct algorithm stops</Eyebrow>
          {delta.length > 0 ? (
            delta.map((id) => (
              <span key={id} className={`text-sm ${inkClassName("secondary")}`}>
                <Code inline>{id}</Code> · {algorithmTitle(id)}
              </span>
            ))
          ) : (
            <span className={`text-sm ${inkClassName("tertiary")}`}>All stops are shared.</span>
          )}
        </Stack>
        {differingFacets.map((facet) => (
          <Stack key={facet} gap={3}>
            <Eyebrow>{DECISION_FACET_LABELS[facet]}</Eyebrow>
            <span className={`text-sm ${inkClassName("secondary")}`}>{decisionFacet(route, facet)}</span>
          </Stack>
        ))}
      </Stack>
    </section>
  );
}

/** Decision laboratory page. */
export function DecisionsPage(): React.JSX.Element {
  return (
    <Stack gap={14}>
      <Stack gap={5} className="max-w-2xl">
        <h2 className={`text-xl font-semibold ${inkClassName("primary")}`}>Decision laboratory</h2>
        <p className={`text-sm ${inkClassName("secondary")}`}>
          Compare neighboring selector states. Shared lifecycle bands collapse so the branch,
          storage tradeoff, artifact boundary, or first rejecting gate stays visible.
        </p>
      </Stack>
      {DECISIONS.map((decision, index) => {
        const leftRoute = cachedRoute(decision.left);
        const rightRoute = cachedRoute(decision.right);
        const bands = collapseSharedRoute(leftRoute.algorithmIds, rightRoute.algorithmIds);
        const differingFacets = (
          ["memory", "fidelity", "artifacts", "limitations", "rejection"] as const
        ).filter((facet) => decisionFacet(leftRoute, facet) !== decisionFacet(rightRoute, facet));
        return (
          <section key={decision.id} aria-labelledby={`decision-${decision.id}`} className={`border-t pt-4 ${strokeClassName("secondary")}`}>
            <Stack gap={12}>
              <Row gap={14} align="start" wrap>
                <Eyebrow>{String(index + 1).padStart(2, "0")}</Eyebrow>
                <Stack gap={4} className="min-w-0 flex-1">
                  <h2 id={`decision-${decision.id}`} className={`text-lg font-semibold ${inkClassName("primary")}`}>
                    {decision.title}
                  </h2>
                  <span className={`text-sm ${inkClassName("secondary")}`}>{decision.invariant}</span>
                </Stack>
              </Row>
              <SharedRouteBand label="Shared prefix" algorithmIds={bands.prefix} />
              <Grid columns={2} gap={14}>
                <DecisionSide
                  label={decision.leftLabel}
                  selection={decision.left}
                  route={leftRoute}
                  delta={bands.leftDelta}
                  differingFacets={differingFacets}
                />
                <DecisionSide
                  label={decision.rightLabel}
                  selection={decision.right}
                  route={rightRoute}
                  delta={bands.rightDelta}
                  differingFacets={differingFacets}
                />
              </Grid>
              <SharedRouteBand label="Shared suffix" algorithmIds={bands.suffix} />
            </Stack>
          </section>
        );
      })}
    </Stack>
  );
}
