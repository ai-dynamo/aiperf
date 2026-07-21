/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compose mode: pick a run shape from thirteen selectors and derive the exact ordered admission →
//! ownership → … → artifact algorithm route it implies, plus its effective runtime settings,
//! verification-boundary limitations, and first rejecting gate. Ported from
//! `docs/canvases/cellular-algorithm-workbook.canvas.tsx` (ComposeMode / RecipeRail).

import { useState } from "react";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Button } from "../../prose/Button.js";
import { Code } from "../../prose/Code.js";
import { Select } from "../../prose/Select.js";
import { Callout } from "../../prose/Callout.js";
import { CollapsibleSection } from "../../prose/CollapsibleSection.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";
import {
  ALGORITHMS,
  CHAPTERS,
  ROUTE_ORDER,
  SELECTOR_OPTIONS,
  SELECTOR_LABELS,
  GATE_STAGE_LABELS,
  STORAGE_INVARIANTS,
  STORAGE_MODE_DETAILS,
  ROUTE_RECIPES,
  DEFAULT_SELECTION,
  cachedRoute,
  optionLabel,
  type SelectorState,
  type SelectorKey,
  type AlgorithmDefinition,
} from "./data.js";
import { StatusLabel, Eyebrow, Framed } from "./ui.js";

function RecipeRail({ onSelect }: { onSelect: (s: SelectorState) => void }): React.JSX.Element {
  const render = (kind: "canonical" | "rejected") => (
    <Row gap={10} wrap aria-label={`${kind} route recipes`}>
      {ROUTE_RECIPES.filter((r) => r.kind === kind).map((recipe) => {
        const route = cachedRoute(recipe.selection);
        const outcome = route.valid
          ? `${route.algorithmIds.length} stops`
          : `${GATE_STAGE_LABELS[route.gateStage]} · ${route.rejectedBy}`;
        return (
          <Stack key={recipe.id} gap={4} className="w-60">
            <Button variant="secondary" onClick={() => onSelect(recipe.selection)}>
              {recipe.title}
            </Button>
            <span className={`text-xs ${inkClassName("tertiary")}`}>{outcome}</span>
          </Stack>
        );
      })}
    </Row>
  );

  return (
    <Stack gap={6}>
      <CollapsibleSection
        title={`Canonical recipes · ${ROUTE_RECIPES.filter((r) => r.kind === "canonical").length}`}
      >
        {render("canonical")}
      </CollapsibleSection>
      <CollapsibleSection
        title={`First-rejection recipes · ${ROUTE_RECIPES.filter((r) => r.kind === "rejected").length}`}
      >
        {render("rejected")}
      </CollapsibleSection>
    </Stack>
  );
}

/** Compose mode page. */
export function ComposePage(): React.JSX.Element {
  const [selection, setSelection] = useState<SelectorState>(DEFAULT_SELECTION);
  const route = cachedRoute(selection);
  const update = (key: SelectorKey, value: string) =>
    setSelection({ ...selection, [key]: value } as SelectorState);

  const routeSections = route.valid
    ? ROUTE_ORDER.map((chapter) => ({
        key: chapter as string,
        chapter,
        algorithms: route.algorithmIds
          .map((id) => ALGORITHMS.find((a) => a.id === id))
          .filter(
            (a): a is AlgorithmDefinition => a !== undefined && a.chapter === chapter,
          ),
      })).filter((item) => item.algorithms.length > 0)
    : route.algorithmIds.flatMap((id, index) => {
        const algorithm = ALGORITHMS.find((a) => a.id === id);
        return algorithm
          ? [{ key: `${index}-${id}`, chapter: algorithm.chapter, algorithms: [algorithm] }]
          : [];
      });

  return (
    <Stack gap={22}>
      <Row align="start" wrap justify="space-between" gap={12}>
        <Stack gap={5} className="max-w-2xl">
          <h2 className={`text-xl font-semibold ${inkClassName("primary")}`}>Compose an execution route</h2>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Select a run shape to derive the exact admission, ownership, transport, execution,
            capture, merge, and artifact algorithms in runtime order.
          </p>
        </Stack>
        <Button variant="secondary" onClick={() => setSelection(DEFAULT_SELECTION)}>
          Reset selectors
        </Button>
      </Row>

      <RecipeRail onSelect={setSelection} />

      <Grid columns="minmax(240px, 0.34fr) minmax(0, 1fr)" gap={20} align="start">
        <Framed surfaceRole="elevated">
          <Stack gap={12}>
            <Eyebrow>Route selectors</Eyebrow>
            {(Object.keys(SELECTOR_OPTIONS) as SelectorKey[]).map((key) => (
              <Select
                key={key}
                label={SELECTOR_LABELS[key]}
                value={selection[key]}
                onChange={(value) => update(key, value)}
                options={SELECTOR_OPTIONS[key].map((value) => ({ value, label: optionLabel(value) }))}
              />
            ))}
          </Stack>
        </Framed>

        <Stack gap={16}>
          <Callout tone="info" title="Requested → effective runtime settings">
            <Row gap={12} wrap>
              <span className="text-sm">
                Workload: <Code inline>{selection.workload}</Code> → <Code inline>{route.effective.workload}</Code>
              </span>
              <span className="text-sm">
                Topology: <Code inline>{selection.topology}</Code> → <Code inline>{route.effective.topology}</Code>
              </span>
              <span className="text-sm">
                Artifacts: <Code inline>{selection.artifacts}</Code> → <Code inline>{route.effective.artifacts}</Code>
              </span>
              <span className="text-sm">
                Storage: <Code inline>{selection.storage}</Code> → <Code inline>{route.effective.storage}</Code>
              </span>
            </Row>
          </Callout>

          {route.valid ? (
            <Callout tone="success" title={`${route.algorithmIds.length} ordered algorithm stops`}>
              This selection passes every current cellular gate.
            </Callout>
          ) : (
            <Callout tone="danger" title={`${GATE_STAGE_LABELS[route.gateStage]} · ${route.rejectedBy}`}>
              <Stack gap={4}>
                <span className="text-sm">{route.reason}</span>
                <span className={`text-sm ${inkClassName("secondary")}`}>
                  The ordered route stops at the actual runtime enforcement stage.
                </span>
              </Stack>
            </Callout>
          )}

          {route.limitations.length > 0 && (
            <Callout tone="warning" title="Verification boundary">
              <Stack gap={4}>
                {route.limitations.map((limitation) => (
                  <span key={limitation} className="text-sm">
                    {limitation}
                  </span>
                ))}
              </Stack>
            </Callout>
          )}

          <Stack gap={8}>
            <Eyebrow>Stage-aware composed lifecycle</Eyebrow>
            <div aria-label="Composed cellular algorithm route">
              {routeSections.map(({ key, chapter, algorithms }) => (
                <Stack key={key} gap={7} className="mb-3">
                  <Eyebrow>{CHAPTERS.find((c) => c.id === chapter)?.label}</Eyebrow>
                  {algorithms.map((algorithm) => (
                    <details key={algorithm.id}>
                      <summary className={`cursor-pointer border p-2 text-sm font-semibold ${strokeClassName("tertiary")}`}>
                        {route.algorithmIds.indexOf(algorithm.id) + 1}. {algorithm.title}
                      </summary>
                      <Stack gap={8} className="p-2">
                        <p className={`text-sm ${inkClassName("secondary")}`}>{algorithm.summary}</p>
                        <Row gap={6} wrap align="center">
                          <StatusLabel status={algorithm.status} />
                          <Code inline>{algorithm.id}</Code>
                        </Row>
                      </Stack>
                    </details>
                  ))}
                </Stack>
              ))}
            </div>
          </Stack>

          {route.valid && (
            <Grid columns={2} gap={12}>
              <Stack gap={4}>
                <Eyebrow>Effective memory</Eyebrow>
                <span className={`text-sm ${inkClassName("secondary")}`}>{route.memory}</span>
              </Stack>
              <Stack gap={4}>
                <Eyebrow>Fidelity</Eyebrow>
                <span className={`text-sm ${inkClassName("secondary")}`}>{route.fidelity}</span>
              </Stack>
              <Stack gap={4}>
                <Eyebrow>Artifacts</Eyebrow>
                <span className={`text-sm ${inkClassName("secondary")}`}>{route.artifacts}</span>
              </Stack>
              <Stack gap={4}>
                <Eyebrow>Compile features</Eyebrow>
                {route.compileFeatures.map((v) => (
                  <Code key={v} inline>{v}</Code>
                ))}
              </Stack>
              <Stack gap={4}>
                <Eyebrow>Environment variables</Eyebrow>
                {route.environment.map((v) => (
                  <Code key={v} inline>{v}</Code>
                ))}
              </Stack>
            </Grid>
          )}
        </Stack>
      </Grid>

      <Framed>
        <Stack gap={10}>
          <Eyebrow>Storage invariant matrix</Eyebrow>
          <Grid columns={3} gap={12}>
            {(
              [
                ["Retain", STORAGE_INVARIANTS.retain, STORAGE_MODE_DETAILS.retain],
                ["Exact fold", STORAGE_INVARIANTS.exactFold, STORAGE_MODE_DETAILS.exactFold],
                ["Sketch", STORAGE_INVARIANTS.sketch, STORAGE_MODE_DETAILS.sketch],
              ] as const
            ).map(([label, assertion, details]) => (
              <Stack key={label} gap={5}>
                <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>{label}</h3>
                <span className="text-sm"><Code inline>assertion</Code> {assertion}</span>
                <span className="text-sm"><Code inline>rows</Code> {details.rows}</span>
                <span className="text-sm"><Code inline>memory</Code> {details.memory}</span>
                <span className="text-sm"><Code inline>fidelity</Code> {details.fidelity}</span>
                <span className="text-sm"><Code inline>count</Code> {details.count}</span>
              </Stack>
            ))}
          </Grid>
        </Stack>
      </Framed>
    </Stack>
  );
}
