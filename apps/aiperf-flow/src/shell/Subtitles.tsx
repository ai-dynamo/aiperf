/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { inkClassName, strokeClassName } from "../theme/tokens.js";

/**
 * Karaoke-style narration transcript: the spoken word is highlighted, earlier
 * words dim, later words stay muted.
 *
 * `aria-live="polite"` announces each step's narration to screen readers without
 * interrupting whatever they are currently reading.
 */
export function Subtitles({
  text,
  activeWordIndex,
  visible,
}: {
  text: string;
  /** Index into the whitespace-split words, or -1 when nothing is being spoken. */
  activeWordIndex: number;
  visible: boolean;
}): React.JSX.Element | null {
  if (!visible) return null;
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return null;

  return (
    <div
      aria-live="polite"
      className={clsx("border-t px-4 py-2", strokeClassName("secondary"))}
    >
      <div className={clsx("text-[10px] font-semibold tracking-widest", inkClassName("tertiary"))}>
        SUBTITLES
      </div>
      <div className="flex flex-wrap gap-x-1.5 text-sm leading-relaxed">
        {words.map((word, index) => (
          <span
            key={`${index}-${word}`}
            className={clsx(
              "transition-colors duration-150",
              index === activeWordIndex
                ? "font-semibold text-accent-primary"
                : activeWordIndex >= 0 && index < activeWordIndex
                  ? inkClassName("secondary")
                  : inkClassName("quaternary"),
            )}
          >
            {word}
          </span>
        ))}
      </div>
    </div>
  );
}
