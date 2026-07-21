/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Button } from "../../prose/Button.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `LoopPage`: the step-through
//! centerpiece walking the Poll -> Compare -> Advance -> Step -> Route cycle that drives the
//! whole simulated run, synced against the virtual-time bar.

type Frame = { stage: number; vt: number; cap: string };

const FRAMES: Frame[] = [
  {
    stage: 0,
    vt: 0,
    cap: "Poll the workload to quiescence — two turns are admitted and submitted to the engine; their futures park (Pending).",
  },
  {
    stage: 1,
    vt: 0,
    cap: "Compare the next clock sleeper vs the next engine event. No sleeper is parked and the engine is ready now, so the engine wins.",
  },
  {
    stage: 3,
    vt: 0,
    cap: "Step the engine: it forms the first batch. step_until is bounded by the next sleeper, so it can't overshoot.",
  },
  {
    stage: 4,
    vt: 1,
    cap: "Route the emitted events: waiters wake and fire on_admit then the first on_token — that first token is TTFT.",
  },
  { stage: 1, vt: 1, cap: "Back to Compare: the next engine event is still the earliest, so the engine wins again." },
  { stage: 3, vt: 2, cap: "Step: a decode tick produces the next tokens." },
  { stage: 4, vt: 3, cap: "Route the terminals: on_usage then on_terminal fire; both request futures resolve." },
  {
    stage: 0,
    vt: 3,
    cap: "Poll once more: StopChecker's bound is met, the workload future is Ready, and the pump exits.",
  },
];

const STAGE_NAMES = ["Poll", "Compare", "Advance", "Step", "Route"];
const TICKS = [
  { x: 0, ms: "0" },
  { x: 1, ms: "1.8" },
  { x: 2, ms: "2.0" },
  { x: 3, ms: "22 ms" },
];
const STAGE_MAINT_SUB = ["LocalSet", "clock ≤ src", "advance_to", "step_until", "wake waiters"];

/**
 * The simulation loop page — a live step-through of the Poll/Compare/Advance/Step/Route cycle
 * that drives the whole dynosim offline run, using {@link useStepSimulator} over the source
 * canvas's fixed `FRAMES` sequence.
 */
export function LoopPage({ level }: { level: Level }): React.JSX.Element {
  const sim = useStepSimulator(FRAMES, { autoPlayMs: 1400 });
  const frame = sim.current ?? FRAMES[0]!;
  const maint = atLeast(level, "maintainer");
  const dev = atLeast(level, "developer");

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>The simulation loop</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          The whole run is one loop. Step through it — the highlighted stage is what&apos;s
          executing, the clock below shows virtual time jumping only when the loop advances it.
        </p>
      </div>

      <Row gap={8} align="center" wrap>
        <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>
          Back
        </Button>
        <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>
          Step
        </Button>
        <Button variant="ghost" onClick={sim.reset} disabled={sim.isFirst}>
          Reset
        </Button>
        <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
          {sim.index + 1} / {sim.total}
        </span>
      </Row>

      <Grid columns={5} gap={10}>
        {STAGE_NAMES.map((name, idx) => {
          const active = idx === frame.stage;
          return (
            <div
              key={name}
              className={clsx(
                "rounded-lg border px-3 py-4 text-center shadow-sm",
                active ? "border-accent-primary bg-accent-primary text-white" : clsx(strokeClassName("primary"), surfaceClassName("elevated")),
              )}
            >
              <div className="text-sm font-semibold">{name}</div>
              {maint && (
                <div className={clsx("mt-1 text-[10px]", active ? "text-white/80" : inkClassName("tertiary"))}>
                  {STAGE_MAINT_SUB[idx]}
                </div>
              )}
            </div>
          );
        })}
      </Grid>
      {dev && (
        <p className={`text-center text-xs ${inkClassName("tertiary")}`}>clock wins ties · overshoot rejected</p>
      )}

      <Callout tone={frame.stage === 4 ? "success" : "info"} title={STAGE_NAMES[frame.stage]}>
        {frame.cap}
      </Callout>

      <div
        className={clsx("rounded-lg border p-4 shadow-sm", strokeClassName("secondary"), surfaceClassName("elevated"))}
      >
        <div className="mb-2 flex items-center justify-between">
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Virtual time</span>
          <span className={`rounded-md border px-2 py-0.5 text-[11px] ${strokeClassName("secondary")} ${inkClassName("tertiary")}`}>
            SimClock
          </span>
        </div>
        <Row gap={0} align="center">
          {TICKS.map((tick) => {
            const on = tick.x === frame.vt;
            return (
              <div key={tick.ms} className="flex flex-1 flex-col items-center gap-1">
                <div className={clsx("h-2 w-0.5", on ? "bg-accent-primary" : "bg-stroke-secondary")} />
                <span className={clsx("text-[11px]", on ? inkClassName("primary") : inkClassName("tertiary"))}>
                  {tick.ms}
                </span>
              </div>
            );
          })}
        </Row>
      </div>
    </Stack>
  );
}
