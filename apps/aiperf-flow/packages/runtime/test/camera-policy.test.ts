// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  authoredCameraAt,
  beginCameraTakeover,
  fitCameraTakeover,
  panCameraTakeover,
  resumeAuthoredCamera,
  zoomCameraTakeover,
} from "../src/camera-policy.js";

const sourceMap = {
  source: "camera-policy.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

const camera = [
  { id: "wide", at: 0, x: 0, y: 0, zoom: 1, sourceMap },
  { id: "detail", at: 1_000, x: 100, y: 50, zoom: 2, sourceMap },
] satisfies SceneIr["camera"];

describe("authored camera policy", () => {
  test("samples the authored camera deterministically at an integer beat", () => {
    expect(authoredCameraAt(camera, 500.9)).toEqual({
      x: 50,
      y: 25,
      zoom: 1.5,
    });
  });

  test("clamps authored camera sampling outside the keyframe range", () => {
    expect(authoredCameraAt(camera, -10)).toEqual({
      x: 0,
      y: 0,
      zoom: 1,
    });
    expect(authoredCameraAt(camera, 4_000)).toEqual({
      x: 100,
      y: 50,
      zoom: 2,
    });
  });

  test("temporary pan and zoom preserve the frozen authored beat", () => {
    const takeover = beginCameraTakeover(camera, 500.9);
    const panned = panCameraTakeover(takeover, { x: 12, y: -8 });
    const zoomed = zoomCameraTakeover(panned, 3);

    expect(zoomed).toMatchObject({
      pausedAtMs: 500,
      authored: { x: 50, y: 25, zoom: 1.5 },
      temporary: { x: 62, y: 17, zoom: 3 },
      takeover: true,
    });
  });

  test("fit produces a deterministic temporary camera", () => {
    const takeover = beginCameraTakeover(camera, 250);
    const fitted = fitCameraTakeover(
      takeover,
      { x: 100, y: 50, width: 400, height: 200 },
      { width: 1_000, height: 600 },
      50,
    );

    expect(fitted.temporary).toEqual({ x: 300, y: 150, zoom: 2.25 });
    expect(fitted.pausedAtMs).toBe(250);
  });

  test("fit rejects non-finite geometry instead of producing non-JSON values", () => {
    const takeover = beginCameraTakeover(camera, 250);

    expect(() =>
      fitCameraTakeover(
        takeover,
        { x: 0, y: 0, width: Number.NaN, height: 100 },
        { width: 1_000, height: 600 },
      ),
    ).toThrow(RangeError);
  });

  test("resume restores the exact authored camera and paused beat", () => {
    const takeover = fitCameraTakeover(
      beginCameraTakeover(camera, 750),
      { x: 500, y: 500, width: 10, height: 10 },
      { width: 100, height: 100 },
    );

    expect(resumeAuthoredCamera(takeover)).toEqual({
      resumedAtMs: 750,
      target: { x: 75, y: 37.5, zoom: 1.75 },
      mode: "smooth",
      takeover: false,
    });
    expect(resumeAuthoredCamera(takeover, { reducedMotion: true })).toEqual({
      resumedAtMs: 750,
      target: { x: 75, y: 37.5, zoom: 1.75 },
      mode: "cut",
      takeover: false,
    });
  });

  test("takeover state and resume plans survive JSON round trips", () => {
    const takeover = panCameraTakeover(beginCameraTakeover(camera, 400), {
      x: 4,
      y: 2,
    });
    const restored = JSON.parse(JSON.stringify(takeover)) as typeof takeover;

    expect(restored).toEqual(takeover);
    expect(JSON.parse(JSON.stringify(resumeAuthoredCamera(restored)))).toEqual(
      resumeAuthoredCamera(takeover),
    );
  });
});
