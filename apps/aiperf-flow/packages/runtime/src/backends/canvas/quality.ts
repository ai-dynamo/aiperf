// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Canvas quality tier. */
export type QualityTier = "reference" | "degraded";

/** Quality modes exposed by the Canvas renderer. */
export type CanvasQualityMode = "reference" | "interactive";

/** Decorative effects that a quality tier may suppress. */
export type DecorativeQuality = Readonly<{
  blur: boolean;
  glow: boolean;
  particles: boolean;
}>;

/**
 * Canvas quality settings.
 *
 * Profiles contain only decorative controls, so semantic content cannot be
 * removed by changing quality tiers.
 */
export type QualityProfile = Readonly<{
  tier: QualityTier;
  decorative: DecorativeQuality;
}>;

const PROFILES: Readonly<Record<QualityTier, QualityProfile>> = Object.freeze({
  reference: Object.freeze({
    tier: "reference",
    decorative: Object.freeze({
      blur: true,
      glow: true,
      particles: true,
    }),
  }),
  degraded: Object.freeze({
    tier: "degraded",
    decorative: Object.freeze({
      blur: false,
      glow: false,
      particles: false,
    }),
  }),
});

/** Returns the immutable settings for a Canvas quality tier. */
export function qualityProfile(tier: QualityTier): QualityProfile {
  return PROFILES[tier];
}

/** Resolves a renderer mode without changing the display list. */
export function canvasQualityProfile(mode: CanvasQualityMode): QualityProfile {
  return qualityProfile(mode === "reference" ? "reference" : "degraded");
}
