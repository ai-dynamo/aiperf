/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! D / Dataset floodgate — MessagePack + zstd carries the published prefix as replay at
//! attachment and every later chunk as live; each cell retains only `request_id % 3 == cell_id`.
//! Ported from the canvas `Dataset`: publish chunks from the reservoir, attach each of three
//! cell channels, and watch each of six slots classify as H/R/L and own/pass by modulo.

import { useState } from "react";
import clsx from "clsx";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { inkClassName, categoryClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import { MechHeader } from "./parts.js";

export function DatasetPage(): React.JSX.Element {
  const [published, setPublished] = useState(2);
  const [attach, setAttach] = useState<Array<number | null>>([null, null, null]);
  const safePublished = Math.min(6, Math.max(0, published));

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="D / dataset floodgate"
        title="Broadcast once. Retain by modulo."
        sentence="MessagePack plus zstd carries the published prefix as replay at attachment and every later chunk as live; each cell retains only request_id % 3 == cell_id."
      />

      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        H = history before attach · R = reply replay · L = live push · own/pass = modulo decision
      </p>

      <Row gap={8} align="center" wrap>
        {Array.from({ length: 6 }, (_, i) => i).map((i) => (
          <Button
            key={i}
            variant={i < safePublished ? "primary" : "secondary"}
            aria-pressed={i < safePublished}
            disabled={i > safePublished}
            onClick={() => setPublished((p) => Math.min(6, Math.max(p, i + 1)))}
          >
            zpack chunk {i}
          </Button>
        ))}
        <Button variant="secondary" onClick={() => setPublished((p) => Math.min(6, p + 1))} disabled={safePublished === 6}>
          Open floodgate
        </Button>
        <Button variant="ghost" onClick={() => setAttach([null, null, null])}>
          Reset subscriptions
        </Button>
        <span className={clsx("text-xs font-medium", categoryClassName("cyan"))}>published {safePublished}/6</span>
      </Row>

      <div className={clsx("rounded-none border", strokeClassName("primary"))}>
        {[0, 1, 2].map((cell) => {
          const boundary = attach[cell];
          return (
            <div key={cell} className={clsx("grid grid-cols-[140px_repeat(6,1fr)] items-stretch border-b", strokeClassName("secondary"))}>
              <div className={clsx("border-r p-2", strokeClassName("secondary"))}>
                <button
                  type="button"
                  disabled={boundary !== null}
                  onClick={() => setAttach((a) => a.map((v, n) => (n === cell ? safePublished : v)))}
                  className={clsx(
                    "w-full rounded-none border px-2 py-1 text-left text-xs font-semibold",
                    boundary !== null ? clsx("border-category-cyan", categoryClassName("cyan")) : clsx(strokeClassName("secondary"), inkClassName("primary")),
                  )}
                >
                  cell {cell}
                  <span className="block font-mono text-[10px]">
                    {boundary === null ? `attach now @ ${safePublished}` : `attached @ ${boundary}`}
                  </span>
                </button>
              </div>
              {Array.from({ length: 6 }, (_, id) => id).map((id) => {
                const publishedNow = id < safePublished;
                const kind = boundary === null ? "H" : id < boundary ? "R" : "L";
                const owns = id % 3 === cell;
                return (
                  <div
                    key={id}
                    className={clsx(
                      "flex flex-col items-center justify-center p-2 text-center font-mono text-[10px]",
                      surfaceClassName("elevated"),
                      publishedNow ? (owns ? categoryClassName("cyan") : inkClassName("tertiary")) : inkClassName("quaternary"),
                      owns && "font-bold",
                    )}
                  >
                    {publishedNow ? (
                      <>
                        <span>{kind} · {owns ? "own" : "pass"}</span>
                        <span>{id}</span>
                      </>
                    ) : (
                      "·"
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
