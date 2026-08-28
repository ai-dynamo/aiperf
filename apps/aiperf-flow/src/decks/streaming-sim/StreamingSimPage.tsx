/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Standalone live simulator for the Streaming · Dynamo Shadow Replay pipeline.
//! Canvas-based animated diagram, step-through simulation, and Web Speech API read-aloud.

import { useCallback, useEffect, useRef, useState } from "react";

// ── layout constants ────────────────────────────────────────────────────────────
const W = 1100;
const H = 618;

// ── node definitions ─────────────────────────────────────────────────────────────
interface SimNode {
  id: string; label: string; sub: string;
  cx: number; cy: number; w: number; h: number;
  color: string; layer: number;
}

const NODES: SimNode[] = [
  // External
  { id:"s3",     label:"S3 / MinIO",               sub:"Dynamo *.jsonl.gz",                     cx:112, cy:42,  w:168, h:46, color:"#1e40af", layer:-1 },
  // Source
  { id:"locsrc", label:"LocalFileSource",           sub:"reads .jsonl.gz from disk",             cx:112, cy:118, w:188, h:46, color:"#1d4ed8", layer:0  },
  { id:"s3src",  label:"S3Source",                  sub:"paginates bucket, versioned snapshot",  cx:330, cy:118, w:188, h:46, color:"#1d4ed8", layer:0  },
  { id:"fmt",    label:"DynamoFormat",              sub:"gzip → JSONL → request units",          cx:594, cy:118, w:192, h:46, color:"#2563eb", layer:0  },
  { id:"chkpt",  label:"LocalCheckpointBackend",    sub:"persists cursor, crash-safe",           cx:862, cy:118, w:200, h:46, color:"#4338ca", layer:0  },
  // Session
  { id:"conv",   label:"ConversationCoordinator",   sub:"joins by session key across chunks",    cx:185, cy:207, w:228, h:46, color:"#15803d", layer:1  },
  { id:"clo",    label:"SessionClosurePolicy",      sub:"finite seal closes sessions, no gaps",  cx:510, cy:207, w:215, h:46, color:"#15803d", layer:1  },
  { id:"host",   label:"ActionHost",                sub:"emits Request actions per closed turn", cx:800, cy:207, w:200, h:46, color:"#166534", layer:1  },
  // Pipeline row 1
  { id:"tci",    label:"TurnClosureIntake",         sub:"Rc<RefCell<VecDeque>>, worker-local",   cx:185, cy:296, w:220, h:46, color:"#7e22ce", layer:2  },
  { id:"pl",     label:"StreamingPipeline",         sub:"select_biased! settlement → actions → stop → barrier → admit", cx:620, cy:296, w:340, h:46, color:"#7e22ce", layer:2 },
  // Pipeline row 2
  { id:"place",  label:"LocalStreamingPlacement",   sub:"routes action to worker thread",        cx:250, cy:373, w:225, h:46, color:"#6d28d9", layer:2  },
  { id:"xport",  label:"CellularStreamingTransport",sub:"bounded multiplexed cross-cell",        cx:620, cy:373, w:250, h:46, color:"#5b21b6", layer:2  },
  // Shadow replay
  { id:"sched",  label:"ScheduledRequestSink",      sub:"issues real HTTP/gRPC request (P4)",    cx:210, cy:462, w:238, h:46, color:"#c2410c", layer:3  },
  { id:"inv",    label:"ActionInventory",           sub:"dense gap-closure ledger",              cx:565, cy:462, w:215, h:46, color:"#c2410c", layer:3  },
  { id:"exec",   label:"shadow_replay workload",    sub:"registered first-class workload type",  cx:878, cy:462, w:225, h:46, color:"#ea580c", layer:3  },
  // Results
  { id:"epoch",  label:"EpochResultCoordinator",    sub:"rotates epochs, fills holes",           cx:168, cy:550, w:215, h:46, color:"#9d174d", layer:4  },
  { id:"compact",label:"StreamingResultCompactor",  sub:"finalizes completed epochs",            cx:468, cy:550, w:225, h:46, color:"#9d174d", layer:4  },
  { id:"deliver",label:"DeliveryRestart",           sub:"target-idempotent export",              cx:760, cy:550, w:210, h:46, color:"#be185d", layer:4  },
  { id:"out",    label:"NativeReport",              sub:"JSON · Parquet · TTFT · throughput",    cx:1010,cy:550, w:180, h:46, color:"#78350f", layer:4  },
];

const NODE_MAP = Object.fromEntries(NODES.map(n => [n.id, n]));

// ── edge definitions ──────────────────────────────────────────────────────────────
interface SimEdge { from: string; to: string; color: string; dashed?: boolean; }

const EDGES: SimEdge[] = [
  { from:"s3",      to:"s3src",   color:"#3b82f6" },
  { from:"locsrc",  to:"fmt",     color:"#3b82f6" },
  { from:"s3src",   to:"fmt",     color:"#3b82f6" },
  { from:"fmt",     to:"conv",    color:"#22c55e" },
  { from:"conv",    to:"clo",     color:"#22c55e" },
  { from:"conv",    to:"host",    color:"#22c55e" },
  { from:"host",    to:"tci",     color:"#a855f7" },
  { from:"tci",     to:"pl",      color:"#a855f7" },
  { from:"pl",      to:"place",   color:"#a855f7" },
  { from:"pl",      to:"xport",   color:"#a855f7" },
  { from:"place",   to:"sched",   color:"#f97316" },
  { from:"xport",   to:"sched",   color:"#f97316" },
  { from:"sched",   to:"inv",     color:"#f97316" },
  { from:"inv",     to:"epoch",   color:"#ec4899" },
  { from:"epoch",   to:"compact", color:"#ec4899" },
  { from:"compact", to:"deliver", color:"#ec4899" },
  { from:"deliver", to:"out",     color:"#fb923c" },
  { from:"chkpt",   to:"epoch",   color:"#818cf8", dashed:true },
];

const EDGE_MAP = Object.fromEntries(EDGES.map(e => [`${e.from}→${e.to}`, e]));

// ── simulation steps ────────────────────────────────────────────────────────────
interface SimStep { nodes: string[]; edges: string[]; title: string; text: string; }

const STEPS: SimStep[] = [
  {
    nodes:["s3","s3src"], edges:["s3→s3src"],
    title:"1 · Source reads partition",
    text:"The S3 source paginates the bucket and downloads Dynamo trace files. Each .jsonl.gz file is one immutable partition, identified by BLAKE3 of the key, generation, and size. VersionedPrefixSnapshot seals the listing after one pass — perfect for a finite batch upload.",
  },
  {
    nodes:["s3src","fmt"], edges:["s3src→fmt"],
    title:"2 · DynamoFormat decodes JSONL",
    text:"The format decoder decompresses gzip and validates each line against schema dynamo.request.trace.v1. Valid records become streaming request units carrying request_id, output_tokens, timing, and input_sequence_hashes.",
  },
  {
    nodes:["fmt","conv"], edges:["fmt→conv"],
    title:"3 · ConversationCoordinator joins session",
    text:"Decoded request units flow to the coordinator. It joins fragments from potentially interleaved sessions by stable session key, building in-memory transcripts. Endpoint replies are folded into the same durable transcript as authored turns.",
  },
  {
    nodes:["conv","clo"], edges:["conv→clo"],
    title:"4 · Closure Policy decides",
    text:"After each fragment append, the closure policy runs. FiniteSeal with no causal gap returns Close immediately. A gap with finite_seal_requires_complete returns Fail — the session is quarantined into the durable tombstone map.",
  },
  {
    nodes:["conv","host"], edges:["conv→host"],
    title:"5 · ActionHost emits Request action",
    text:"The ActionHost materialises a Request action from the closed turn. Its stable action id is derived only from causal inputs — safe to re-emit after a crash without duplicating results. The turn closure receipt lands in the TurnClosureIntake.",
  },
  {
    nodes:["host","tci"], edges:["host→tci"],
    title:"6 · TurnClosureIntake queues receipt",
    text:"The closed turn receipt is appended to a worker-local VecDeque behind an Rc RefCell — zero-copy, no synchronization needed on the hot path. The fused pipeline loop drains this on the Actions arm each iteration.",
  },
  {
    nodes:["tci","pl"], edges:["tci→pl"],
    title:"7 · StreamingPipeline admits action",
    text:"The select_biased! loop processes five event types in priority order: settlement, actions, stop, barrier, then admit. The bounded capacity window gates admission — if full, the source stalls. Settlement always wins to prevent starvation.",
  },
  {
    nodes:["pl","place"], edges:["pl→place"],
    title:"8 · LocalStreamingPlacement routes to worker",
    text:"Five placement traits: sticky, round-robin, hash, global, and local. Sticky keeps a session's actions on the same worker thread throughout its lifetime — critical for correct per-session accumulation. Cellular mode routes through CellularStreamingTransport instead.",
  },
  {
    nodes:["place","sched"], edges:["place→sched"],
    title:"9 · ScheduledRequestSink issues request",
    text:"Task P4: fires one real HTTP or gRPC request per action against the configured endpoint — chat, completions, or responses. endpoint_retry_safety is Unproven — a non-zero retry limit is refused because a retry duplicates measured load.",
  },
  {
    nodes:["sched","inv"], edges:["sched→inv"],
    title:"10 · ActionInventory tracks result",
    text:"The dense gap-closure ledger records each action position as in-flight, completed, or permanently failed. The delivery frontier only advances when no holes remain. A failed action is a durable hole — not a retry trigger.",
  },
  {
    nodes:["inv","epoch"], edges:["inv→epoch"],
    title:"11 · EpochResultCoordinator rotates epoch",
    text:"Task 6B: completed actions flow to the epoch coordinator. It rotates result epochs for long-running streams, holds provisional in-flight holes, and publishes barrier advancements back to the pipeline — allowing the checkpoint cursor to advance.",
  },
  {
    nodes:["epoch","chkpt"], edges:["chkpt→epoch"],
    title:"12 · Checkpoint cursor advanced",
    text:"The CheckpointCoordinator sequences the local backend write. The cursor is atomically renamed and fsync'd. On restart, the source reads the current file, skips fully-processed partitions, and resumes at the next unprocessed position.",
  },
  {
    nodes:["epoch","compact"], edges:["epoch→compact"],
    title:"13 · ResultCompactor finalizes epoch",
    text:"Task 6C1: deterministically finalizes each completed epoch — merges per-action records, fills gaps with zero-scores for permanent holes, and produces per-session scored receipts. Strictly sequential within an epoch.",
  },
  {
    nodes:["compact","deliver","out"], edges:["compact→deliver","deliver→out"],
    title:"14 · DeliveryRestart exports results",
    text:"Task 6C2: idempotent result delivery. Compacted epochs can be re-delivered on restart without duplicating output. Results flow to NativeReport sinks: JSON, Parquet, console, OTLP, MLflow, Weights and Biases, and timeslice exporters.",
  },
];

// ── geometry helpers ────────────────────────────────────────────────────────────
function edgePts(e: SimEdge) {
  const a = NODE_MAP[e.from], b = NODE_MAP[e.to];
  if (!a || !b) return null;
  const dy = b.cy - a.cy, dx = b.cx - a.cx;
  if (Math.abs(dy) < 22) {
    if (dx > 0) return { x1: a.cx + a.w/2, y1: a.cy, x2: b.cx - b.w/2, y2: b.cy };
    return { x1: a.cx - a.w/2, y1: a.cy, x2: b.cx + b.w/2, y2: b.cy };
  }
  if (dy > 0) return { x1: a.cx, y1: a.cy + a.h/2, x2: b.cx, y2: b.cy - b.h/2 };
  return { x1: a.cx, y1: a.cy - a.h/2, x2: b.cx, y2: b.cy + b.h/2 };
}

function cubicCP(x1: number, y1: number, x2: number, y2: number) {
  const dy = y2 - y1, dx = x2 - x1;
  if (Math.abs(dy) < 22) {
    return { c1x: x1 + dx*.35, c1y: y1 - 16, c2x: x2 - dx*.35, c2y: y2 - 16 };
  }
  return { c1x: x1, c1y: y1 + dy*.42, c2x: x2, c2y: y2 - dy*.42 };
}

function cubicAt(t: number, p0: number, p1: number, p2: number, p3: number) {
  const u = 1 - t;
  return u*u*u*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t*t*t*p3;
}

function particleXY(e: SimEdge, t: number) {
  const p = edgePts(e);
  if (!p) return { x: 0, y: 0 };
  const c = cubicCP(p.x1, p.y1, p.x2, p.y2);
  return {
    x: cubicAt(t, p.x1, c.c1x, c.c2x, p.x2),
    y: cubicAt(t, p.y1, c.c1y, c.c2y, p.y2),
  };
}

// ── layer bands ─────────────────────────────────────────────────────────────────
const BANDS = [
  { y1:10,  h:64,  bg:"rgba(30,58,138,0.12)",  label:"External",            lc:"#60a5fa" },
  { y1:86,  h:64,  bg:"rgba(29,78,216,0.12)",  label:"Source Layer",         lc:"#60a5fa" },
  { y1:162, h:72,  bg:"rgba(21,128,61,0.12)",  label:"Session Layer",        lc:"#4ade80" },
  { y1:246, h:148, bg:"rgba(126,34,206,0.12)", label:"Pipeline / Placement", lc:"#c084fc" },
  { y1:436, h:74,  bg:"rgba(194,65,12,0.12)",  label:"Shadow Replay",        lc:"#fb923c" },
  { y1:522, h:80,  bg:"rgba(157,23,77,0.12)",  label:"Result Plane",         lc:"#f472b6" },
];

// ── mutable animation state (all refs to avoid re-render on every frame) ────────
interface Particle { eid: string; t: number; speed: number; }

function makeParticles(): Particle[] {
  return EDGES.flatMap(e => {
    const k = `${e.from}→${e.to}`;
    return [0, 1, 2].map(i => ({
      eid: k,
      t: (i / 3 + Math.random() * 0.12) % 1,
      speed: 0.1 + Math.random() * 0.07,
    }));
  });
}

// ── draw function ─────────────────────────────────────────────────────────────────
function drawFrame(
  ctx: CanvasRenderingContext2D,
  dt: number,
  particles: Particle[],
  activeNodes: Set<string>,
  activeEdges: Set<string>,
) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

  // bands
  BANDS.forEach(b => {
    ctx.save();
    ctx.fillStyle = b.bg;
    ctx.beginPath(); ctx.roundRect(8, b.y1, W - 16, b.h, 7); ctx.fill();
    ctx.fillStyle = b.lc; ctx.font = "600 9px system-ui"; ctx.textAlign = "right";
    ctx.globalAlpha = 0.65;
    ctx.fillText(b.label.toUpperCase(), W - 14, b.y1 + 13);
    ctx.restore();
  });

  // edges
  EDGES.forEach(e => {
    const p = edgePts(e); if (!p) return;
    const { x1, y1, x2, y2 } = p;
    const c = cubicCP(x1, y1, x2, y2);
    const ek = `${e.from}→${e.to}`;
    const active = activeEdges.has(ek);
    ctx.save();
    ctx.strokeStyle = active ? e.color : "#252525";
    ctx.lineWidth = active ? 2 : 1.2;
    ctx.globalAlpha = active ? 0.85 : 0.45;
    if (e.dashed) ctx.setLineDash([5, 4]);
    if (active) { ctx.shadowColor = e.color; ctx.shadowBlur = 6; }
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(c.c1x, c.c1y, c.c2x, c.c2y, x2, y2);
    ctx.stroke();
    ctx.setLineDash([]); ctx.shadowBlur = 0;
    // arrowhead
    const ang = Math.atan2(y2 - c.c2y, x2 - c.c2x);
    const L = active ? 9 : 7;
    ctx.fillStyle = active ? e.color : "#252525";
    ctx.globalAlpha = active ? 0.9 : 0.45;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - L * Math.cos(ang - 0.4), y2 - L * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - L * Math.cos(ang + 0.4), y2 - L * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  });

  // particles
  particles.forEach(p => {
    p.t = (p.t + p.speed * dt) % 1;
    const e = EDGE_MAP[p.eid]; if (!e) return;
    const active = activeEdges.has(p.eid);
    const pos = particleXY(e, p.t);
    const col = active ? e.color : "#2a2a2a";
    const r = active ? 3.5 : 2;
    ctx.save();
    ctx.globalAlpha = active ? 0.9 : 0.22;
    if (active) { ctx.shadowColor = col; ctx.shadowBlur = 10; }
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  });

  // nodes
  NODES.forEach(n => {
    const active = activeNodes.has(n.id);
    const { cx, cy, w, h, color, label, sub } = n;
    const x = cx - w/2, y = cy - h/2;
    ctx.save();
    if (active) { ctx.shadowColor = color; ctx.shadowBlur = 22; }
    ctx.fillStyle = active ? color + "cc" : color + "2a";
    ctx.strokeStyle = active ? color : color + "60";
    ctx.lineWidth = active ? 2 : 1;
    ctx.globalAlpha = active ? 1 : 0.8;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 7); ctx.fill(); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = active ? "#fff" : "#c9d1d9";
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillText(label, cx, cy - 8, w - 14);
    ctx.fillStyle = active ? "#ffffffaa" : "#6e7681";
    ctx.font = "9.5px system-ui, sans-serif";
    ctx.fillText(sub, cx, cy + 9, w - 14);
    ctx.restore();
  });
}

// ── TTS helper ────────────────────────────────────────────────────────────────────
function speak(text: string) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}

// ── component ─────────────────────────────────────────────────────────────────────
export function StreamingSimPage(): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particles = useRef<Particle[]>(makeParticles());
  const activeNodes = useRef<Set<string>>(new Set());
  const activeEdges = useRef<Set<string>>(new Set());
  const stepRef = useRef(-1);
  const lastTRef = useRef(0);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const rafRef = useRef<number>(0);

  const [caption, setCaption] = useState({ title: "Streaming · Dynamo Shadow Replay", text: "A Dynamo trace file is one streaming source. Each .jsonl.gz file is an immutable partition, identified by BLAKE3 of the key, generation, and size. Click Simulate to watch a request travel through the full pipeline.", progress: "Ready", stepDone: false });
  const [readAloud, setReadAloud] = useState(false);
  const readAloudRef = useRef(false);

  // keep ref in sync
  useEffect(() => { readAloudRef.current = readAloud; }, [readAloud]);

  // animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function frame(now: number) {
      const dt = Math.min((now - lastTRef.current) / 1000, 0.05);
      lastTRef.current = now;
      drawFrame(ctx!, dt, particles.current, activeNodes.current, activeEdges.current);
      rafRef.current = requestAnimationFrame(frame);
    }

    lastTRef.current = performance.now();
    rafRef.current = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(rafRef.current);
      if (autoRef.current) clearInterval(autoRef.current);
      window.speechSynthesis?.cancel();
    };
  }, []);

  const applyStep = useCallback((s: SimStep | null, idx: number) => {
    activeNodes.current = new Set(s?.nodes ?? []);
    activeEdges.current = new Set(s?.edges ?? []);
    const n = STEPS.length;
    setCaption({
      title: s?.title ?? "Streaming · Dynamo Shadow Replay",
      text: s?.text ?? "Click Simulate to watch a request travel through the full pipeline.",
      progress: s ? `Step ${idx + 1} / ${n}` : "Ready",
      stepDone: idx >= n - 1,
    });
    if (s && readAloudRef.current) speak(`${s.title}. ${s.text}`);
  }, []);

  const doStep = useCallback(() => {
    const next = Math.min(stepRef.current + 1, STEPS.length - 1);
    stepRef.current = next;
    applyStep(STEPS[next], next);
  }, [applyStep]);

  const stopAuto = useCallback(() => {
    if (autoRef.current) { clearInterval(autoRef.current); autoRef.current = null; }
  }, []);

  const startSim = useCallback(() => {
    stopAuto();
    if (stepRef.current >= STEPS.length - 1) {
      stepRef.current = -1;
      activeNodes.current.clear();
      activeEdges.current.clear();
    }
    doStep();
    autoRef.current = setInterval(() => {
      if (stepRef.current >= STEPS.length - 1) { stopAuto(); return; }
      doStep();
    }, 3000);
  }, [doStep, stopAuto]);

  const reset = useCallback(() => {
    stopAuto();
    window.speechSynthesis?.cancel();
    stepRef.current = -1;
    activeNodes.current.clear();
    activeEdges.current.clear();
    particles.current = makeParticles();
    applyStep(null, -1);
  }, [applyStep, stopAuto]);

  const toggleReadAloud = useCallback(() => {
    setReadAloud(v => {
      if (v) window.speechSynthesis?.cancel();
      return !v;
    });
  }, []);

  return (
    <div className="flex h-screen flex-col" style={{ background: "#0d1117", color: "#e6edf3" }}>
      {/* header */}
      <div style={{ padding: "13px 22px 10px", borderBottom: "1px solid #21262d", display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
        <span style={{ fontSize: 17, fontWeight: 700, letterSpacing: -0.3 }}>Streaming · Dynamo Shadow Replay</span>
        <span style={{ fontSize: 12, color: "#8b949e" }}>live pipeline simulator — trace file → session → shadow replay → results</span>
      </div>

      {/* legend */}
      <div style={{ display: "flex", gap: 16, padding: "7px 22px", borderBottom: "1px solid #21262d", flexWrap: "wrap" }}>
        {[
          { color: "#1d4ed8", label: "Source" },
          { color: "#15803d", label: "Session" },
          { color: "#7e22ce", label: "Pipeline / Placement" },
          { color: "#c2410c", label: "Shadow Replay" },
          { color: "#9d174d", label: "Result Plane" },
          { color: "#818cf8", label: "Checkpoint (dashed)" },
        ].map(l => (
          <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#8b949e" }}>
            <div style={{ width: 9, height: 9, borderRadius: 3, background: l.color, flexShrink: 0 }} />
            {l.label}
          </div>
        ))}
      </div>

      {/* canvas */}
      <div style={{ overflowX: "auto", flex: "0 0 auto" }}>
        <canvas ref={canvasRef} width={W} height={H} style={{ display: "block", margin: "0 auto", width: W, height: H }} />
      </div>

      {/* controls */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "10px 22px", borderTop: "1px solid #21262d", flexWrap: "wrap" }}>
        <button onClick={startSim} style={btnStyle("#1f6feb", "#388bfd")}>▶ Simulate</button>
        <button onClick={doStep} disabled={caption.stepDone} style={btnStyle()}>Step →</button>
        <button onClick={reset} style={btnStyle()}>↺ Reset</button>
        <button
          onClick={toggleReadAloud}
          style={btnStyle(readAloud ? "#15803d" : undefined, readAloud ? "#16a34a" : undefined)}
          title="Read each step aloud using your browser's text-to-speech"
        >
          {readAloud ? "🔊 Reading aloud" : "🔇 Read aloud"}
        </button>
        <span style={{ fontSize: 12, color: "#8b949e", marginLeft: 4 }}>{caption.progress}</span>
      </div>

      {/* caption */}
      <div style={{ padding: "12px 22px 16px", background: "#161b22", borderTop: "1px solid #21262d", flex: "1 1 auto", minHeight: 80 }}>
        <span style={{ fontWeight: 600, fontSize: 14, display: "block", marginBottom: 3, color: "#e6edf3" }}>{caption.title}</span>
        <span style={{ fontSize: 13, color: "#8b949e", lineHeight: 1.6 }}>{caption.text}</span>
      </div>
    </div>
  );
}

function btnStyle(bg?: string, border?: string): React.CSSProperties {
  return {
    padding: "6px 14px", borderRadius: 6,
    border: `1px solid ${border ?? "#30363d"}`,
    background: bg ?? "#21262d",
    color: "#e6edf3", fontSize: 13, cursor: "pointer",
  };
}
