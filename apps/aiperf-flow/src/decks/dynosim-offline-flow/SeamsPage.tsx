/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { AutoLayoutFlow } from "../../layout/graph/index.js";
import { Stack } from "../../layout/Stack.js";
import { Callout } from "../../prose/Callout.js";
import { inkClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast, SegControl } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `SeamsPage`: everything above the
//! `RequestSink` seam is shared by every mode; below it the path forks by transport. The frozen
//! `AIPerfRegistry` registers independent transports (http / dynosim_offline / dynosim_online)
//! rather than a transport x workload pair map — the fork is chosen at composition, not by
//! reading `clock.is_virtual()`.

type ModeId = "http" | "offline" | "online";

const ARCH: Record<
  ModeId,
  { label: string; lane: "http" | "dyn"; clock: string; clockCite: string; driver: string; driverCite: string; tag: string }
> = {
  http: {
    label: "HTTP online",
    lane: "http",
    clock: "RealClock",
    clockCite: "run.rs",
    driver: "tokio LocalSet reactor",
    driverCite: "execute.rs",
    tag: "real HTTP to a real server, wall clock",
  },
  offline: {
    label: "dynosim offline",
    lane: "dyn",
    clock: "SimClock",
    clockCite: "dynosim.rs:2419",
    driver: "drive_sim_with_source",
    driverCite: "graph/runtime.rs:213",
    tag: "virtual clock, in-process engine, deterministic",
  },
  online: {
    label: "dynosim online",
    lane: "dyn",
    clock: "RealClock",
    clockCite: "dynosim.rs:2489",
    driver: "drive_real_with_source",
    driverCite: "graph/runtime.rs:419",
    tag: "wall clock, in-process engine",
  },
};

const MODE_OPTIONS = [
  { id: "http" as const, label: "HTTP online" },
  { id: "offline" as const, label: "dynosim offline" },
  { id: "online" as const, label: "dynosim online" },
];

function nodes(mode: ModeId, maint: boolean): Node[] {
  const a = ARCH[mode];
  const httpOn = a.lane === "http";
  const dynOn = a.lane === "dyn";
  return [
    {
      id: "profile",
      type: "card",
      position: { x: 220, y: 0 },
      data: { title: "aiperf profile", subtitle: maint ? "load.rs / profile.rs" : "native entry point" },
    },
    {
      id: "execute",
      type: "card",
      position: { x: 220, y: 90 },
      data: { title: "aiperf --execute", subtitle: maint ? "RunnerApplication::handle_v2" : "re-exec child" },
    },
    {
      id: "workload",
      type: "card",
      position: { x: 220, y: 180 },
      data: { title: "Workload", subtitle: maint ? "RequestRate · UserCentric · Graph" : "arrival pattern" },
    },
    {
      id: "scheduled-runtime",
      type: "card",
      position: { x: 220, y: 270 },
      data: { title: "ScheduledRuntime", subtitle: "SlotPool · StopChecker", detail: maint ? "+ ObserverTee" : undefined },
    },
    {
      id: "clock-seam",
      type: "panel",
      position: { x: 0, y: 270 },
      data: { title: `Clock → ${a.clock}`, detail: mode === "offline" ? "virtual ns" : "wall ns" },
    },
    {
      id: "driver",
      type: "chip",
      position: { x: 0, y: 350 },
      data: { label: a.driver },
    },
    {
      id: "sink-seam",
      type: "panel",
      position: { x: 220, y: 360 },
      data: { title: "RequestSink<HttpRequest>", detail: "trait seam" },
    },

    {
      id: "transport-sink",
      type: "card",
      position: { x: 40, y: 460 },
      data: {
        title: "TransportSink",
        subtitle: maint ? "impl RequestSink" : "serialize → bytes",
        className: httpOn ? undefined : "opacity-40",
      },
    },
    {
      id: "http-transport",
      type: "card",
      position: { x: 40, y: 550 },
      data: {
        title: "HttpTransport",
        subtitle: "Hyper 1.x + SSE",
        detail: maint ? "http.rs:524" : undefined,
        className: httpOn ? undefined : "opacity-40",
      },
    },
    {
      id: "real-server",
      type: "card",
      position: { x: 40, y: 640 },
      data: { title: "real server", subtitle: "socket / URL", className: httpOn ? undefined : "opacity-40" },
    },

    {
      id: "dynosim-sink",
      type: "card",
      position: { x: 400, y: 460 },
      data: {
        title: "DynosimSink",
        subtitle: maint ? "impl RequestSink" : "→ token array",
        className: dynOn ? undefined : "opacity-40",
      },
    },
    {
      id: "engine-host",
      type: "card",
      position: { x: 400, y: 550 },
      data: {
        title: "EngineHost",
        subtitle: "bounds step_until",
        detail: maint ? "SimEventSource" : undefined,
        className: dynOn ? undefined : "opacity-40",
      },
    },
    {
      id: "steppable-replay",
      type: "card",
      position: { x: 400, y: 640 },
      data: {
        title: "SteppableReplay",
        subtitle: "no sockets · in-process",
        className: dynOn ? undefined : "opacity-40",
      },
    },
  ];
}

function edges(mode: ModeId): Edge[] {
  const a = ARCH[mode];
  const httpOn = a.lane === "http";
  const dynOn = a.lane === "dyn";
  return [
    { id: "e-profile-execute", source: "profile", target: "execute", type: "flow" },
    { id: "e-execute-workload", source: "execute", target: "workload", type: "flow" },
    { id: "e-workload-runtime", source: "workload", target: "scheduled-runtime", type: "flow" },
    { id: "e-clock-runtime", source: "clock-seam", target: "scheduled-runtime", type: "flow" },
    { id: "e-runtime-sink", source: "scheduled-runtime", target: "sink-seam", type: "flow" },
    {
      id: "e-sink-http-sink",
      source: "sink-seam",
      target: "transport-sink",
      type: "flow",
      data: { color: httpOn ? undefined : "var(--color-stroke-secondary)" },
    },
    {
      id: "e-sink-dyn-sink",
      source: "sink-seam",
      target: "dynosim-sink",
      type: "flow",
      data: { color: dynOn ? undefined : "var(--color-stroke-secondary)" },
    },
    { id: "e-http-sink-transport", source: "transport-sink", target: "http-transport", type: "flow" },
    { id: "e-http-transport-server", source: "http-transport", target: "real-server", type: "flow" },
    { id: "e-dyn-sink-host", source: "dynosim-sink", target: "engine-host", type: "flow" },
    { id: "e-dyn-host-replay", source: "engine-host", target: "steppable-replay", type: "flow" },
  ];
}

/**
 * System architecture page — the two seams. Switching `mode` (HTTP online / dynosim offline /
 * dynosim online) re-renders the diagram with the active lane at full opacity and the inactive
 * lane dimmed, mirroring the source canvas's `SeamsPage`.
 */
export function SeamsPage({ level }: { level: Level }): React.JSX.Element {
  const [mode, setMode] = useState<ModeId>("offline");
  const a = ARCH[mode];
  const maint = atLeast(level, "maintainer");
  const dev = atLeast(level, "developer");

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>System architecture — the two seams</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          The real architecture: everything above the <strong>RequestSink</strong> seam is shared
          by every mode. Below it the path forks. Pick a mode — the active branch lights up.
        </p>
      </div>

      <SegControl value={mode} onChange={setMode} options={MODE_OPTIONS} />

      <AutoLayoutFlow key={`${mode}-${maint}`} nodes={nodes(mode, maint)} edges={edges(mode)} layout={{ direction: "DOWN" }} height={780} />

      <Callout tone={mode === "http" ? "info" : "success"} title={a.label}>
        {a.tag}. Above the seam nothing changes; the driver is <strong>{a.driver}</strong> and the
        clock is <strong>{a.clock}</strong>
        {maint ? ` (${a.driverCite}, ${a.clockCite})` : ""}.
      </Callout>

      {dev && (
        <Callout tone="warning" title="The fork is chosen at composition — not is_virtual()">
          The frozen <strong>AIPerfRegistry</strong> registers independent transports (
          <strong>http</strong> / <strong>dynosim_offline</strong> / <strong>dynosim_online</strong>
          ) and workloads; there is no transport×workload pair map. Within dynosim,{" "}
          <strong>run_paced_offline</strong> vs <strong>run_paced_online</strong> pick the
          clock+driver from the transport ID. <strong>clock.is_virtual()</strong> is only read for
          measurement, never to branch the mode.
        </Callout>
      )}
    </Stack>
  );
}
