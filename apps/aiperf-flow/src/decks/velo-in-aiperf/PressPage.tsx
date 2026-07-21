/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! P / MessagePack press — `rmp-serde` converts a typed `CellMessage::Heartbeat` to raw bytes
//! and back, preserving NaN and infinity. Ported from the canvas `Press`: a `useStepSimulator`
//! drives the four stages Load → Apply pressure → Inspect raw bytes → Reconstruct.

import { useMemo } from "react";
import clsx from "clsx";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { Callout } from "../../prose/Callout.js";
import { inkClassName, categoryClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import { MechHeader } from "./parts.js";

const STAGES = ["Load typed value", "Apply pressure", "Inspect raw bytes", "Reconstruct"] as const;
const BYTES = ["83", "a7", "63", "6f", "75", "6e", "74", "2a", "cb", "7f", "f8", "00", "00", "00", "00", "00"];

export function PressPage(): React.JSX.Element {
  const sim = useStepSimulator(STAGES, { autoPlayMs: 1100 });
  const stage = sim.index;

  const chamber = useMemo(() => {
    if (stage === 0) {
      return (
        <div className={clsx("font-mono text-xs", inkClassName("primary"))}>
          <b>CellMessage::Heartbeat</b>
          <div>count: 42</div>
          <div>ttft: NaN</div>
          <div>max: +∞</div>
        </div>
      );
    }
    if (stage === 1) {
      return <div className={clsx("font-mono text-xs", categoryClassName("cyan"))}>rmp_serde::to_vec — applying pressure…</div>;
    }
    if (stage === 2) {
      return (
        <div className="grid grid-cols-8 gap-1">
          {BYTES.map((b, i) => (
            <span
              key={`${b}-${i}`}
              className={clsx(
                "flex items-center justify-center border py-1 font-mono text-[10px]",
                i > 7 ? clsx("border-category-cyan", categoryClassName("cyan")) : clsx(strokeClassName("secondary"), inkClassName("secondary")),
              )}
            >
              {b}
            </span>
          ))}
        </div>
      );
    }
    return (
      <div className={clsx("font-mono text-xs", categoryClassName("cyan"))}>
        <b>Decoded Heartbeat</b>
        <div>count: 42</div>
        <div>ttft: NaN</div>
        <div>max: +∞</div>
      </div>
    );
  }, [stage]);

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="P / MessagePack press"
        title="Typed state becomes raw bytes"
        sentence="rmp-serde preserves NaN and infinity in a Velo raw payload, then reconstructs the cellular value at the handler."
      />

      <Row gap={8} align="center" wrap>
        {STAGES.map((label, i) => (
          <Button
            key={label}
            variant={stage === i ? "primary" : "secondary"}
            aria-pressed={stage === i}
            onClick={() => {
              // `useStepSimulator` has no absolute `goto`; each `next`/`back` is a functional
              // state update, so stepping `|i - stage|` times lands exactly on the clicked stage.
              const delta = i - stage;
              for (let n = 0; n < Math.abs(delta); n++) delta > 0 ? sim.next() : sim.back();
            }}
          >
            {i + 1} / {label}
          </Button>
        ))}
        <Button variant="ghost" onClick={sim.reset}>Reset</Button>
      </Row>

      <div className={clsx("rounded-lg border p-4 shadow-sm", strokeClassName("primary"), surfaceClassName("elevated"))}>
        <div className={clsx("mb-2 text-center text-xs font-bold uppercase", stage === 1 ? categoryClassName("cyan") : inkClassName("tertiary"))}>
          rmp_serde::to_vec
        </div>
        <div className="min-h-[120px] py-4">{chamber}</div>
        <div className={clsx("mt-2 text-center text-xs font-bold uppercase", stage === 3 ? categoryClassName("cyan") : inkClassName("tertiary"))}>
          rmp_serde::from_slice
        </div>
      </div>

      <Callout tone="info" title="Preserves NaN and infinity">
        The typed <code>CellMessage::Heartbeat {"{ count: 42, ttft: NaN, max: +∞ }"}</code> round-trips through the
        16 MessagePack bytes above and reconstructs exactly, including the non-finite floats.
      </Callout>
    </div>
  );
}
