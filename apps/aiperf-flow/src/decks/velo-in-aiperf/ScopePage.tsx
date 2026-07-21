/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! H / Heartbeat scope — fire-and-forget snapshots expose counters, sketches, observation time,
//! lag, and a missing cell. Ported from the canvas `Scope`: "Emit heartbeat" advances the tick
//! counter (issued = ticks·12, completed = ticks·11); "Fail cell 2" drops liveness to 2/3 and
//! marks channel 2's pulse missing.

import { useState } from "react";
import clsx from "clsx";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Button } from "../../prose/Button.js";
import { Stat } from "../../prose/Stat.js";
import { inkClassName, categoryClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import { MechHeader } from "./parts.js";

export function ScopePage(): React.JSX.Element {
  const [ticks, setTicks] = useState(4);
  const [failed, setFailed] = useState(false);
  const safeTicks = Math.min(32, Math.max(1, ticks));

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="H / heartbeat scope"
        title="Read the live pulse"
        sentence="Fire-and-forget snapshots expose counters, sketches, observation time, lag, and a missing cell."
      />

      <Row gap={12} align="center" wrap>
        <Button variant="primary" onClick={() => setTicks((t) => Math.min(32, t + 1))}>
          Emit heartbeat
        </Button>
        <Button variant="secondary" aria-pressed={failed} onClick={() => setFailed((v) => !v)}>
          {failed ? "Restore cell 2" : "Fail cell 2"}
        </Button>
      </Row>

      <Grid columns={3} gap={12}>
        <Stat label="issued" value={safeTicks * 12} />
        <Stat label="completed" value={safeTicks * 11} />
        <Stat label="liveness" value={failed ? "2 / 3" : "3 / 3"} tone={failed ? "negative" : "positive"} />
      </Grid>

      <Grid columns={3} gap={12}>
        {[0, 1, 2].map((cell) => {
          const dead = failed && cell === 2;
          return (
            <div
              key={cell}
              className={clsx("rounded-none border p-3", strokeClassName("primary"), surfaceClassName("elevated"))}
            >
              <div className="flex items-center justify-between">
                <span className={clsx("text-xs font-bold uppercase", dead ? inkClassName("tertiary") : inkClassName("primary"))}>
                  CH {cell} / cell {cell}
                </span>
                <span className={clsx("text-[11px] font-mono", dead ? categoryClassName("cyan") : inkClassName("tertiary"))}>
                  {dead ? "lag ↑ · pulse missing" : `observed_at +${safeTicks}s`}
                </span>
              </div>
              <svg viewBox="0 0 800 95" preserveAspectRatio="none" className="mt-2 h-16 w-full">
                {Array.from({ length: Math.min(8, safeTicks) }, (_, i) => i).map((i) =>
                  dead && i > 3 ? null : (
                    <path
                      key={i}
                      d={`M${i * 100} 52h23l8-30 13 58 13-42 12 14h31`}
                      fill="none"
                      stroke="var(--color-category-cyan)"
                      strokeWidth={2}
                      opacity={0.35 + (i / Math.min(8, safeTicks)) * 0.65}
                    />
                  ),
                )}
              </svg>
            </div>
          );
        })}
      </Grid>
    </div>
  );
}
