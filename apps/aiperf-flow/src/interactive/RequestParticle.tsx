/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Animated play-head marker for a `useFlowPlayer`. Shows which node is active and the current
//! step's caption, with a `motion` pulse dot (instant when the user prefers reduced motion). Fully
//! generic over `FlowStep` — no domain specifics — so any deck can render "the request is here now".

import { motion, useReducedMotion } from "motion/react";
import type { FlowStep } from "./types.js";
import { Row } from "../layout/Row.js";
import { Eyebrow } from "../prose/Eyebrow.js";
import {
  categoryBgClassName,
  inkClassName,
  strokeClassName,
  surfaceClassName,
} from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export interface RequestParticleProps {
  /** The active step (its `nodeId` is the highlight, its `caption` is the readout). */
  step: FlowStep | undefined;
  /** 1-based position for the progress readout (e.g. `player.index + 1`). */
  position?: number;
  /** Total number of steps, shown alongside `position`. */
  total?: number;
  /** Resolved human label for the active node; falls back to `step.nodeId`. */
  nodeLabel?: string;
  /** Marker/dot color. Defaults to `"cyan"`. */
  tone?: CategoryRole;
  className?: string;
}

/**
 * A one-line "now playing" bar: a pulsing marker dot, the active node's label, and the step
 * caption, plus an optional `position/total` readout. Presentational — it reflects whatever
 * `step` a {@link useFlowPlayer} makes current.
 */
export function RequestParticle({
  step,
  position,
  total,
  nodeLabel,
  tone = "cyan",
  className,
}: RequestParticleProps): React.JSX.Element {
  const prefersReduced = useReducedMotion() ?? false;
  const label = nodeLabel ?? step?.nodeId;

  return (
    <div
      className={`rounded-lg border px-4 py-3 shadow-sm ${surfaceClassName("elevated")} ${strokeClassName("secondary")} ${className ?? ""}`}
    >
      <Row gap={12} align="center">
        <motion.span
          aria-hidden="true"
          className={`inline-block h-3 w-3 shrink-0 rounded-full ${categoryBgClassName(tone)}`}
          animate={prefersReduced ? undefined : { scale: [1, 1.55, 1], opacity: [1, 0.55, 1] }}
          transition={
            prefersReduced ? undefined : { duration: 1.1, repeat: Infinity, ease: "easeInOut" }
          }
        />
        <div className="min-w-0">
          <Row gap={8} align="center" wrap>
            <Eyebrow>Now</Eyebrow>
            {label !== undefined && (
              <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{label}</span>
            )}
            {position !== undefined && total !== undefined && total > 0 && (
              <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
                step {position}/{total}
              </span>
            )}
          </Row>
          <p className={`mt-0.5 text-sm ${inkClassName("secondary")}`}>
            {step?.caption ?? "Press Play to send a request through the pipeline."}
          </p>
        </div>
      </Row>
    </div>
  );
}
