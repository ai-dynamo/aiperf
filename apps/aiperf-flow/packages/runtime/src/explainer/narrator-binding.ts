// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridges narration lifecycle events to slideshow navigation.

import type { SlideshowController } from './controller.js';
import type { NarratorBackend } from '../narrative/narrator.js';

export class NarratorBinding {
  private readonly controller: SlideshowController;
  private readonly narrator: NarratorBackend;

  constructor(controller: SlideshowController, narrator: NarratorBackend) {
    this.controller = controller;
    this.narrator = narrator;
  }

  onNarrationComplete(): void {
    // Auto-advance to next slide once narration finishes.
    void this.controller.nextSlide();
  }

  pauseNarration(): void {
    this.controller.pauseNarration();
  }

  resumeNarration(): void {
    this.controller.resumeNarration();
  }

  skipNarration(): void {
    // Cancel the current narration through the runtime interface (there is no
    // `stop` on NarratorBackend), then advance.
    this.narrator.cancel();
    void this.controller.nextSlide();
  }
}
