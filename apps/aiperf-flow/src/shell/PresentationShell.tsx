/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, type ReactNode } from "react";
import { inkClassName, strokeClassName, surfaceClassName } from "../theme/tokens.js";
import type { SlideDefinition } from "../deck/types.js";

export function PresentationShell({
  slides,
  slideIndex,
  onSlideIndexChange,
  children,
}: {
  slides: readonly SlideDefinition[];
  slideIndex: number;
  onSlideIndexChange: (next: number) => void;
  children: ReactNode;
}): React.JSX.Element {
  const [showNotes, setShowNotes] = useState(false);
  const slide = slides[slideIndex];
  if (slide === undefined) {
    throw new Error(`slideIndex ${slideIndex} out of range for ${slides.length} slides.`);
  }

  return (
    <div className="flex h-screen flex-col">
      <div className={`flex gap-1 border-b px-4 py-2 ${strokeClassName("secondary")}`}>
        {slides.map((s, index) => (
          <button
            key={s.id}
            type="button"
            aria-label={`Go to slide ${index + 1}`}
            onClick={() => onSlideIndexChange(index)}
            className={`h-1 flex-1 rounded-none ${index === slideIndex ? surfaceClassName("elevated") + " bg-accent-primary" : "bg-neutral-200"}`}
          />
        ))}
      </div>

      <div className="min-h-0 flex-1">{children}</div>

      <div className={`border-t px-4 py-2 ${strokeClassName("secondary")}`}>
        <div className={`text-sm ${inkClassName("secondary")}`}>{slide.narration}</div>
      </div>

      {showNotes && (
        <div className={`border-t px-4 py-2 ${strokeClassName("secondary")}`}>
          <div className={`text-xs font-semibold ${inkClassName("tertiary")}`}>
            Speaker notes
          </div>
          <div className={`text-sm ${inkClassName("secondary")}`}>{slide.caption}</div>
        </div>
      )}

      <div
        className={`flex items-center justify-between border-t px-4 py-2 ${strokeClassName("secondary")}`}
      >
        <button
          type="button"
          disabled={slideIndex === 0}
          onClick={() => onSlideIndexChange(slideIndex - 1)}
          className={`px-3 py-1 text-sm disabled:opacity-40 ${inkClassName("primary")}`}
        >
          ← Back
        </button>
        <span className={`text-xs ${inkClassName("tertiary")}`}>
          {slideIndex + 1} / {slides.length}
        </span>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setShowNotes((value) => !value)}
            className={`px-3 py-1 text-xs ${inkClassName("secondary")}`}
          >
            Speaker notes
          </button>
          <button
            type="button"
            disabled={slideIndex === slides.length - 1}
            onClick={() => onSlideIndexChange(slideIndex + 1)}
            className={`px-3 py-1 text-sm disabled:opacity-40 ${inkClassName("primary")}`}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
