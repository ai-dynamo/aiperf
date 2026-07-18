// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Subtitle overlay bound to a slideshow controller's live narration state.
//!
//! Subscribes to the controller's word-synchronized subtitle stream and renders
//! the shared SubtitleOverlay, so captions advance in lock-step with narration
//! for every slide without the caller re-implementing timing.

import React, { useEffect, useState } from 'react';

import type { SlideshowController } from '../controller.js';
import {
  SubtitleOverlay,
  type SubtitleState,
} from '../../narrative/subtitle-overlay.js';

const HIDDEN: SubtitleState = { enabled: false, activeCue: null };

export interface SubtitleRendererProps {
  /** Controller whose narration drives the captions; null before mount. */
  controller: SlideshowController | null;
  /** High-contrast captions for accessibility. */
  contrast?: 'standard' | 'high';
  /** Honor reduced-motion for caption transitions. */
  reducedMotion?: boolean;
}

/**
 * Renders the live subtitle overlay for the current narration.
 *
 * The overlay updates in real time as the controller reports word-level
 * progress. Toggling captions is forwarded to the controller so audio timing is
 * unaffected.
 */
export function SubtitleRenderer({
  controller,
  contrast = 'standard',
  reducedMotion = false,
}: SubtitleRendererProps): React.ReactElement {
  const [state, setState] = useState<SubtitleState>(
    () => controller?.subtitle ?? HIDDEN,
  );

  useEffect(() => {
    if (controller === null) {
      setState(HIDDEN);
      return;
    }
    return controller.subscribeSubtitles(setState);
  }, [controller]);

  return (
    <SubtitleOverlay
      contrast={contrast}
      onEnabledChange={(enabled) => controller?.setSubtitlesEnabled(enabled)}
      reducedMotion={reducedMotion}
      state={state}
    />
  );
}

export default SubtitleRenderer;
