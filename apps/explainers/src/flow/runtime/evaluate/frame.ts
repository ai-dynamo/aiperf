// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure composition of one fully evaluated runtime frame.

import type { SceneIr } from "../../schema/index.js";

import type { Bounds, DisplayList } from "../display-list.js";
import { computeDamageBetween } from "./damage-tracker.js";
import {
  createHitRegionIndex,
  type HitRegionIndex,
} from "./hit-region-index.js";
import {
  applyQualityPolicy,
  qualityPolicyProfile,
  type DegradationReport,
  type DisplayContract,
  type QualityDisplayList,
  type QualityPolicyProfile,
} from "./quality-policy.js";
import {
  evaluateScene,
  type EvaluateSceneOptions,
} from "./scene-evaluator.js";
import type { EvaluatedScene } from "./types.js";

/** Optional scene, quality, and prior-frame inputs for frame evaluation. */
export type EvaluateFrameOptions = Readonly<{
  scene?: EvaluateSceneOptions;
  quality?: QualityPolicyProfile;
  displayContract?: DisplayContract;
  previousDisplayList?: DisplayList;
}>;

/** Immutable visual, semantic, interaction, and damage products for one frame. */
export type EvaluatedFrame = Readonly<{
  scene: EvaluatedScene;
  displayList: QualityDisplayList;
  report: DegradationReport;
  hitIndex: HitRegionIndex;
  damageRegions: readonly Bounds[];
}>;

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) {
      deepFreeze(nested);
    }
    Object.freeze(value);
  }
  return value;
}

/** Evaluates and composes one deterministic frame at an exact virtual time. */
export function evaluateFrame(
  sceneIr: SceneIr,
  timeMs: number,
  options: EvaluateFrameOptions = {},
): EvaluatedFrame {
  if (!Number.isSafeInteger(timeMs) || timeMs < 0) {
    throw new RangeError(
      "Frame evaluation time must be a non-negative safe integer.",
    );
  }

  const evaluatedScene = evaluateScene(sceneIr, timeMs, options.scene);
  const quality = applyQualityPolicy(
    evaluatedScene.displayList,
    options.quality ?? qualityPolicyProfile("reference"),
    options.displayContract,
  );
  const displayList = quality.list;

  return deepFreeze({
    scene: { ...evaluatedScene, displayList },
    displayList,
    report: quality.report,
    hitIndex: createHitRegionIndex(displayList),
    damageRegions:
      options.previousDisplayList === undefined
        ? [displayList.damageBounds]
        : computeDamageBetween(options.previousDisplayList, displayList),
  });
}
