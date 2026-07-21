/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! "SLURM + Velo from scratch" — a 16-step interactive walkthrough of how AIPerf launches a
//! cellular benchmark under SLURM and coordinates the controller (rank 0) and its load cells over
//! Velo. Ported from the Cursor canvas `slurm-explained-step-by-step.canvas.tsx`. Unlike the
//! tabbed segment-pools deck, this is a single step-through scene sequence, so it is driven by
//! `useStepSimulator` over the 16 steps (mirroring `PoolPage`), with a bounded-loop "Play all".

import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Button } from "../../prose/Button.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName, categoryClassName, surfaceClassName } from "../../theme/tokens.js";
import { STEPS, NARRATION, SCENE_LABELS, SCENE_NOTES } from "./steps-data.js";
import { SlurmDiagram } from "./SlurmDiagram.js";

/**
 * Top-level component for the SLURM + Velo explainer deck. Self-contained (takes no props): steps
 * through {@link STEPS} via {@link useStepSimulator}, rendering the active step's prose alongside
 * its {@link SlurmDiagram} React Flow scene.
 */
export function SlurmExplainedStepByStepDeck(): React.JSX.Element {
  const sim = useStepSimulator(STEPS, { autoPlayMs: 4000 });
  const index = sim.index;
  const step = STEPS[index];
  const isLast = sim.isLast;

  return (
    <div className={`min-h-screen ${surfaceClassName("page")} px-8 py-8`}>
      <Stack gap={16}>
        <div>
          <div className={`text-xs font-semibold tracking-wide ${inkClassName("secondary")}`}>
            AIPERF · SLURM + VELO FROM SCRATCH · STEP {index + 1} OF {STEPS.length}
          </div>
          <div className={`mt-1 text-sm font-semibold ${categoryClassName("green")}`}>{step.eyebrow}</div>
          <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>{step.title}</h1>
          <p className={`mt-2 max-w-3xl text-sm ${inkClassName("secondary")}`}>{step.lede}</p>
        </div>

        {/* Step rail — jump to any step. */}
        <Row gap={6} wrap>
          {STEPS.map((entry, i) => (
            <Button
              key={entry.title}
              variant={i === index ? "primary" : "ghost"}
              title={entry.title}
              onClick={() => {
                // Move the simulator to step `i` via bounded next()/back() calls. Each is a no-op
                // at the clamp, so a fixed loop count is safe (never an unbounded while loop).
                const delta = i - index;
                for (let n = 0; n < Math.abs(delta); n++) {
                  if (delta > 0) sim.next();
                  else sim.back();
                }
              }}
              className="px-3 py-1"
            >
              {String(i + 1)}
            </Button>
          ))}
        </Row>

        <Row gap={10} align="center" wrap>
          <Button variant="primary" onClick={sim.togglePlay}>
            {sim.isPlaying ? "Pause" : isLast ? "Replay from start" : "Play all"}
          </Button>
          <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>
            Back
          </Button>
          <Button variant="secondary" onClick={sim.next} disabled={isLast}>
            Next
          </Button>
          <Button variant="ghost" onClick={sim.reset}>
            Reset
          </Button>
          <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
            {index + 1} / {STEPS.length}
          </span>
        </Row>

        <Divider />

        {/* Scene header (label + note) above the diagram. */}
        <div className="flex items-baseline justify-between">
          <div className={`text-sm font-bold tracking-wide ${inkClassName("primary")}`}>
            {SCENE_LABELS[index]}
          </div>
          <div className={`text-xs ${inkClassName("secondary")}`}>{SCENE_NOTES[index]}</div>
        </div>

        <SlurmDiagram key={index} stepIndex={index} />

        <div className={`text-sm font-medium ${inkClassName("secondary")}`}>{step.caption}</div>

        <Grid columns="minmax(240px, 0.8fr) minmax(320px, 1.2fr)" gap={14} align="start">
          {step.term ? (
            <Callout tone="success" title={step.term.word}>
              {step.term.meaning}
            </Callout>
          ) : (
            <div />
          )}
          <div>
            <div className={`mb-3 text-sm font-bold ${inkClassName("primary")}`}>What happens here</div>
            <Stack gap={8}>
              {step.points.map((point) => (
                <Row key={point} gap={8} align="start">
                  <span className={`font-bold ${categoryClassName("green")}`}>·</span>
                  <span className={`text-sm ${inkClassName("primary")}`}>{point}</span>
                </Row>
              ))}
            </Stack>
          </div>
        </Grid>

        <Callout tone="info" title="Narration">
          {NARRATION[index]}
        </Callout>

        {isLast && (
          <div className="border border-stroke-secondary p-4">
            <div className={`mb-3 text-sm font-bold ${inkClassName("primary")}`}>The commands</div>
            <Stack gap={12}>
              <Stack gap={4}>
                <div className={`text-xs font-bold ${inkClassName("secondary")}`}>
                  1 · GENERATE A SUBMISSION SCRIPT
                </div>
                <Code>aiperf slurm generate --config benchmark.yaml --cells 4 --output job.sbatch</Code>
              </Stack>
              <Stack gap={4}>
                <div className={`text-xs font-bold ${inkClassName("secondary")}`}>2 · SUBMIT IT TO SLURM</div>
                <Code>sbatch job.sbatch</Code>
                <div className={`text-xs ${inkClassName("secondary")}`}>
                  Every task launches <Code inline>aiperf slurm run</Code>. Rank picks controller vs. cell; Velo
                  wires the control plane automatically.
                </div>
              </Stack>
            </Stack>
          </div>
        )}
      </Stack>
    </div>
  );
}
