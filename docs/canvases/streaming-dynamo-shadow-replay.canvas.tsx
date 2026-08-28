import {
  H1,
  H3,
  Text,
  Card,
  CardHeader,
  CardBody,
  Callout,
  Pill,
  Button,
  Row,
  Stack,
  Grid,
  Spacer,
  Divider,
  useCanvasState,
  useHostTheme,
  useCanvasAction,
} from "cursor/canvas";

// ═══════════════════════════════════════════════════════════════════════════
// Streaming Dynamo Shadow Replay — multi-page canvas explainer.
// Grounded in tasks P1-P4, 5D-5F, 6B-6D, C1-C3, A-REG, V1 on
// ajc/native-rust-runtime-plugins. Three detail levels.
// ═══════════════════════════════════════════════════════════════════════════

type Level = "executive" | "developer" | "maintainer";
type Theme = ReturnType<typeof useHostTheme>;
type PageId = "overview" | "source" | "session" | "pipeline" | "shadow" | "checkpoint";

const CSS = `
@keyframes sr-flow  { to { stroke-dashoffset: -36; } }
@keyframes sr-soft  { 0%,100% { opacity: .35 } 50% { opacity: .9 } }
@keyframes sr-pulse { 0%,100% { opacity: .5; r: 20px } 50% { opacity: .9; r: 24px } }
@keyframes sr-pop   { 0% { opacity: 0; transform: scale(.76) } 62% { opacity: 1; transform: scale(1.06) } 100% { transform: scale(1) } }
@keyframes sr-scan  { 0% { transform: translateY(0) } 100% { transform: translateY(var(--sh)) } }
`;

const POP = { transformBox: "fill-box", transformOrigin: "center", animation: "sr-pop .4s cubic-bezier(.2,.8,.3,1.2) both" } as const;
const SVG = { width: "100%", height: "auto", display: "block" } as const;

function useLevel()  { return useCanvasState<Level>("sr.level", "developer"); }
function usePage()   { return useCanvasState<PageId>("sr.page", "overview"); }
function atLeast(level: Level, min: Level): boolean {
  const r = { executive: 0, developer: 1, maintainer: 2 } as const;
  return r[level] >= r[min];
}

// ── shared SVG atoms ────────────────────────────────────────────────────────

function Arrow({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill={color} />
    </marker>
  );
}

function Node({
  x, y, w, h, t,
  fill, stroke, accent,
  title, sub, micro, dim, titleFill,
}: {
  x: number; y: number; w: number; h: number; t: Theme;
  fill: string; stroke: string; accent?: string;
  title: string; sub?: string; micro?: string; dim?: boolean; titleFill?: string;
}) {
  const cx = x + w / 2;
  const lines = (sub ? 1 : 0) + (micro ? 1 : 0);
  const ty = lines === 0 ? y + h / 2 : lines === 1 ? y + h / 2 - 4 : y + h / 2 - 9;
  return (
    <g opacity={dim ? 0.38 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={fill} stroke={accent ?? stroke} strokeWidth={accent ? 1.7 : 1} />
      <text x={cx} y={ty} textAnchor="middle" dominantBaseline="middle" fontSize={12} fontWeight={600} fill={titleFill ?? t.text.primary}>{title}</text>
      {sub  && <text x={cx} y={ty + 14} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>{sub}</text>}
      {micro && <text x={cx} y={ty + 26} textAnchor="middle" fontSize={8}   fill={t.text.quaternary}>{micro}</text>}
    </g>
  );
}

function Edge({ d, color, id, moving = true, dur = 2.4 }: { d: string; color: string; id: string; moving?: boolean; dur?: number }) {
  return (
    <path d={d} fill="none" stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`}
      strokeDasharray="4 5"
      style={moving ? { animation: `sr-flow ${dur}s linear infinite` } : undefined} />
  );
}

function Particles({ d, color, n, dur, shape = "dot", size = 4 }: { d: string; color: string; n: number; dur: number; shape?: "dot" | "sq"; size?: number }) {
  const fade = (begin: string) => (
    <animate attributeName="opacity" dur={`${dur}s`} begin={begin} repeatCount="indefinite"
      values="0;1;1;0" keyTimes="0;0.12;0.85;1" calcMode="linear" />
  );
  return (
    <>
      {Array.from({ length: n }).map((_, i) => {
        const begin = `${(dur / n) * i}s`;
        return shape === "dot" ? (
          <circle key={i} r={size} fill={color} opacity={0}>
            <animateMotion dur={`${dur}s`} begin={begin} repeatCount="indefinite" path={d} />
            {fade(begin)}
          </circle>
        ) : (
          <rect key={i} x={-size / 2} y={-size / 2} width={size} height={size} rx={1.5} fill={color} opacity={0}>
            <animateMotion dur={`${dur}s`} begin={begin} repeatCount="indefinite" path={d} />
            {fade(begin)}
          </rect>
        );
      })}
    </>
  );
}

function SeamBox({ x, y, w, h, t, tag, main, sub }: { x: number; y: number; w: number; h: number; t: Theme; tag: string; main: string; sub?: string }) {
  const cx = x + w / 2;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={t.accent.control} stroke={t.accent.primary} strokeWidth={1.6} strokeDasharray="5 4" />
      <text x={x + 8} y={y + 13} fontSize={7.5} fontWeight={700} fill={t.accent.primary} style={{ letterSpacing: 0.6 }}>{tag}</text>
      <text x={cx} y={y + h / 2 + (sub ? 2 : 5)} textAnchor="middle" fontSize={12} fontWeight={700} fill={t.text.onAccent}>{main}</text>
      {sub && <text x={cx} y={y + h / 2 + 15} textAnchor="middle" fontSize={8.5} fill={t.text.onAccent} opacity={0.8}>{sub}</text>}
    </g>
  );
}

// ═══ PAGE 1 · Overview (full pipeline) ══════════════════════════════════════

function OverviewPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const l = t.stroke.secondary, band = t.fill.quaternary, n = t.fill.tertiary;
  const a = t.accent.primary, g = t.category.green, p = t.category.purple, blue = t.category.blue, or = t.category.orange;
  const maint = level === "maintainer";

  const bands = [
    { y: 10,  h: 62, name: "Source",        tone: blue,  file: "streaming/sources/" },
    { y: 82,  h: 62, name: "Session",        tone: g,     file: "streaming/session/" },
    { y: 154, h: 62, name: "Pipeline",       tone: a,     file: "streaming/pipeline.rs" },
    { y: 226, h: 62, name: "Shadow Replay",  tone: or,    file: "engine/streaming_execution.rs" },
    { y: 298, h: 62, name: "Results",        tone: p,     file: "streaming/results/" },
  ];

  const spineX = 340;
  const flow = ["M340,72 L340,82", "M340,144 L340,154", "M340,216 L340,226", "M340,288 L340,298"];

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        A Dynamo trace file is one streaming source. The source reads partitions; the decoder emits request units;
        the session layer joins them into conversations and fires <Text as="span" weight="semibold">Request</Text>{" "}
        actions; the pipeline places each action on a worker and drives it through the sink; the sink issues a real
        endpoint request; results compact and export. A checkpoint backend keeps the cursor so any stage can restart.
      </Text>
      <svg viewBox="0 0 680 378" style={SVG}>
        <defs><Arrow id="ov" color={l} /><Arrow id="ovg" color={a} /></defs>

        {/* layer bands */}
        {bands.map((b) => (
          <g key={b.name}>
            <rect x={12} y={b.y} width={656} height={b.h} rx={11} fill={band} stroke={l} />
            <rect x={12} y={b.y} width={5} height={b.h} rx={2} fill={b.tone} />
            <text x={30} y={b.y + 18} fontSize={9.5} fontWeight={700} fill={b.tone} style={{ letterSpacing: 0.5 }}>{b.name.toUpperCase()}</text>
            {maint && <text x={30} y={b.y + 32} fontSize={8} fill={t.text.quaternary}>{b.file}</text>}
          </g>
        ))}

        {/* inter-layer spine arrows */}
        {flow.map((d, i) => (
          <g key={`sp${i}`}>
            <Edge d={d} color={l} id="ov" dur={2.2} />
            <Particles d={d} color={a} n={1} dur={1.4} />
          </g>
        ))}

        {/* SOURCE row */}
        <Node t={t} x={60} y={28} w={140} h={36} fill={n} stroke={l} accent={blue} title="S3 / Local Source" sub={maint ? "sources/s3.rs · local.rs" : "files = partitions"} />
        <Node t={t} x={218} y={28} w={150} h={36} fill={n} stroke={l} accent={blue} title="DynamoFormat" sub={maint ? "formats/streaming_dynamo.rs" : "gzip + JSONL decode"} />
        <Node t={t} x={386} y={28} w={150} h={36} fill={n} stroke={l} accent={blue} title="Checkpoint Backend" sub={maint ? "checkpoints/local.rs" : "cursor persistence"} />
        <path d="M358,46 L386,46" fill="none" stroke={l} strokeWidth={1.2} strokeDasharray="4 4" markerEnd="url(#ov)" />

        {/* SESSION row */}
        <Node t={t} x={60} y={100} w={180} h={36} fill={n} stroke={l} accent={g} title="ConversationCoordinator" sub={maint ? "session/conversation.rs" : "join by session key"} />
        <Node t={t} x={258} y={100} w={160} h={36} fill={n} stroke={l} accent={g} title="SessionClosurePolicy" sub={maint ? "session/closure.rs · P1B" : "finite seal closes sessions"} />
        <Node t={t} x={436} y={100} w={140} h={36} fill={n} stroke={l} accent={g} title="ActionHost" sub={maint ? "session/host.rs · plan-P2" : "emits Request actions"} />
        <path d="M438,118 L436,118" fill="none" stroke={l} strokeWidth={1.2} strokeDasharray="4 4" markerEnd="url(#ov)" />

        {/* PIPELINE row */}
        <Node t={t} x={60} y={172} w={180} h={36} fill={n} stroke={l} accent={a} title="StreamingPipeline" sub={maint ? "pipeline.rs · select_biased!" : "fused event loop"} />
        <Node t={t} x={258} y={172} w={160} h={36} fill={n} stroke={l} accent={a} title="LocalPlacement" sub={maint ? "placement.rs · 5 traits · P3" : "route to worker thread"} />
        <Node t={t} x={436} y={172} w={140} h={36} fill={n} stroke={l} accent={or} title="CellularTransport" sub={maint ? "C2 / C3" : "multi-cell delivery"} />

        {/* SHADOW REPLAY row */}
        <Node t={t} x={60} y={244} w={160} h={36} fill={n} stroke={l} accent={or} title="ScheduledRequestSink" sub={maint ? "action/scheduled_request.rs · P4" : "issue real endpoint request"} />
        <Node t={t} x={238} y={244} w={150} h={36} fill={n} stroke={l} accent={or} title="ActionInventory" sub={maint ? "action/host/inventory.rs" : "dense gap-closure"} />
        <Node t={t} x={406} y={244} w={160} h={36} fill={n} stroke={l} accent={or} title="shadow_replay workload" sub={maint ? "engine/streaming_execution.rs" : "registered in engine"} />

        {/* RESULTS row */}
        <Node t={t} x={60} y={316} w={140} h={36} fill={n} stroke={l} accent={p} title="EpochCoordinator" sub={maint ? "results/epoch.rs · 6B" : "rotate + hole-fill"} />
        <Node t={t} x={218} y={316} w={140} h={36} fill={n} stroke={l} accent={p} title="ResultCompactor" sub={maint ? "results/compactor.rs · 6C1" : "deterministic finalize"} />
        <Node t={t} x={376} y={316} w={130} h={36} fill={n} stroke={l} accent={p} title="DeliveryRestart" sub={maint ? "results/delivery.rs · 6C2" : "idempotent export"} />

        {/* report exits right */}
        <path d="M516,334 L548,334 L548,44 L536,44" fill="none" stroke={g} strokeWidth={1.5} markerEnd="url(#ovg)" strokeDasharray="4 4" />
        <Particles d="M516,334 L548,334 L548,44 L536,44" color={g} n={2} dur={3.2} />
        <text x={560} y={192} textAnchor="middle" fontSize={8.5} fill={t.text.quaternary} style={{ writingMode: "vertical-rl" }}>JSON · Parquet · Metrics</text>
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="One file = one partition">Each .jsonl.gz becomes a uniquely identified immutable partition. BLAKE3 of bucket+key+gen+size is computed at listing time.</Callout>
        <Callout tone="info" title="Stable action identity">Turn action IDs are derived from causal inputs only — not wall time — so the pipeline can re-emit them after a restart without duplicating results.</Callout>
        <Callout tone="success" title="Resumable from checkpoint">The local backend commits an atomic fsync'd cursor after each partition. A restarted run skips fully-processed partitions and resumes mid-partition.</Callout>
      </Grid>
    </Stack>
  );
}

// ═══ PAGE 2 · Source & format ══════════════════════════════════════════════

type SourceMode = "s3" | "local";

function SourcePage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [mode, setMode] = useCanvasState<SourceMode>("sr.srcMode", "s3");
  const n = t.fill.tertiary, l = t.stroke.secondary, s = t.stroke.primary;
  const a = t.accent.primary, blue = t.category.blue, g = t.category.green;
  const maint = level === "maintainer";
  const s3 = mode === "s3";

  const srcLabel = s3 ? "S3Source" : "LocalSource";
  const srcSub = s3 ? (maint ? "sources/s3.rs · VersionedPrefixSnapshot" : "paginate bucket") : (maint ? "sources/local.rs" : "walk directory");

  // pipeline: source → decode → session
  const nodes = [
    { x: 20, title: srcLabel, sub: srcSub, accent: blue, w: 170 },
    { x: 218, title: "DynamoFormat", sub: maint ? "streaming_dynamo.rs" : "gzip → JSONL → units", accent: g, w: 158 },
    { x: 404, title: "ConversationCoordinator", sub: maint ? "session/conversation.rs" : "join by session key", accent: a, w: 188 },
  ];
  const edges = ["M190,51 L218,51", "M376,51 L404,51"];
  const w = 620, h = 100;

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        Each trace file is one immutable partition. The source factory assigns it a dense position and hands the bytes
        to the Dynamo format decoder, which decompresses gzip and validates JSONL records against
        schema <Text as="span" weight="semibold">dynamo.request.trace.v1</Text> before emitting request units.
        The conversation coordinator receives those units and joins them by session key.
      </Text>

      <Row gap={8} align="center" wrap>
        <SegControl<SourceMode> value={mode} set={setMode} opts={[
          { id: "s3", label: "S3 source" },
          { id: "local", label: "Local source" },
        ]} />
      </Row>

      <svg viewBox={`0 0 ${w} ${h}`} style={SVG}>
        <defs><Arrow id="sf" color={l} /></defs>
        {nodes.map((nd, i) => (
          <g key={nd.title}>
            <Node t={t} x={nd.x} y={20} w={nd.w} h={62} fill={n} stroke={s} accent={nd.accent} title={nd.title} sub={nd.sub} />
            {i < nodes.length - 1 && (
              <>
                <Edge d={edges[i]} color={l} id="sf" dur={2.2} />
                <Particles d={edges[i]} color={i === 0 ? blue : g} n={2} dur={2.2} />
              </>
            )}
          </g>
        ))}
      </svg>

      {s3 ? (
        <Stack gap={10}>
          <H3>S3 inventory policies</H3>
          <Text tone="secondary" size="small">Choose the right policy for how your trace bucket grows.</Text>
          <svg viewBox="0 0 660 110" style={SVG}>
            <defs><Arrow id="sp" color={l} /></defs>
            {[
              { x: 16, label: "VersionedPrefixSnapshot", sub: maint ? "one pass, sealed immediately" : "finite batch upload", tone: g },
              { x: 230, label: "IntervalFollow", sub: maint ? "monotonic keys · live stream" : "continuous follow", tone: a },
              { x: 444, label: "Manifest", sub: maint ? "sealed manifest object" : "pre-indexed", tone: blue },
            ].map((pol) => (
              <g key={pol.label} style={POP}>
                <rect x={pol.x} y={20} width={200} height={72} rx={10} fill={t.fill.secondary} stroke={pol.tone} strokeWidth={1.5} />
                <text x={pol.x + 100} y={52} textAnchor="middle" fontSize={11} fontWeight={700} fill={t.text.primary}>{pol.label}</text>
                <text x={pol.x + 100} y={70} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>{pol.sub}</text>
              </g>
            ))}
          </svg>
          <Callout tone="info" title="VersionedPrefixSnapshot for dynamo traces">
            One listing pass over the bucket, sealed immediately — perfect for a finite uploaded trace batch.
            Use <Text as="span" weight="semibold">IntervalFollow</Text> when the producer keeps appending keys.
          </Callout>
          {atLeast(level, "developer") && (
            <Callout tone="neutral" title="force_path_style required for MinIO">
              Set <Text as="span" weight="semibold">force_path_style: true</Text> and <Text as="span" weight="semibold">endpoint_url</Text>{" "}
              in the S3 source config when pointing at a local MinIO server. Credentials flow through the AWS credential chain (env vars, shared-config profile).
            </Callout>
          )}
        </Stack>
      ) : (
        <Callout tone="neutral" title="Local source">
          Walks a directory tree; every <Text as="span" weight="semibold">.jsonl.gz</Text>{" "}
          file is one partition. Useful for development, dry-run replay, and CI — no S3 credential required.
          {maint && " Each file path becomes the partition key; identity is BLAKE3(path, size, mtime)."}
        </Callout>
      )}
    </Stack>
  );
}

// ═══ PAGE 3 · Session layer (step-through) ══════════════════════════════════

interface SessionStep { stage: number; label: string; cap: string; }
const SESSION_STEPS: SessionStep[] = [
  { stage: 0, label: "Fragment arrives", cap: "A decoded request unit lands from the format decoder. The coordinator looks up the session by key — creating it if new." },
  { stage: 1, label: "Join transcript", cap: "The unit is appended to the session's in-memory transcript. Endpoint replies are folded into the same transcript via the action host." },
  { stage: 2, label: "Closure decision", cap: "After each append, SessionClosurePolicy::decide() checks the evidence. A finite source seal with no causal gap immediately returns Close." },
  { stage: 3, label: "Emit action", cap: "The action host materializes a Request action from the completed turn. Its stable action id is derived from causal inputs — safe to re-emit after restart." },
  { stage: 4, label: "Retire session", cap: "The coordinator removes the session from its live map. The turn-closure intake VecDeque holds the receipt until the pipeline drains it." },
];
const SESSION_NODES = ["Receive unit", "Join transcript", "Closure decision", "Emit Request action", "Retire session"];

function SessionPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [i, setI] = useCanvasState<number>("sr.sesFrame", 0);
  const f = SESSION_STEPS[Math.min(i, SESSION_STEPS.length - 1)];
  const n = t.fill.tertiary, l = t.stroke.secondary, s = t.stroke.primary;
  const a = t.accent.primary, g = t.category.green, red = t.category.red;
  const maint = level === "maintainer";

  const W = 140, H = 42, gap = 14;
  const xs = SESSION_NODES.map((_, idx) => 16 + idx * (W + gap));
  const y = 24;

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        Sessions are the unit of work. A single trace file may interleave records from many concurrent sessions;
        the coordinator holds each open until it is complete, then the closure policy retires it and the action host
        fires one <Text as="span" weight="semibold">Request</Text> action per turn.
      </Text>

      <Row gap={8} align="center" wrap>
        <Button variant="secondary" disabled={i === 0} onClick={() => setI((v) => Math.max(0, v - 1))}>Back</Button>
        <Button variant="primary" disabled={i >= SESSION_STEPS.length - 1} onClick={() => setI((v) => Math.min(SESSION_STEPS.length - 1, v + 1))}>Step</Button>
        <Button variant="ghost" disabled={i === 0} onClick={() => setI(0)}>Reset</Button>
        <Spacer />
        <Pill size="sm">{`${i + 1} / ${SESSION_STEPS.length}`}</Pill>
      </Row>

      <svg viewBox={`0 0 ${16 * 2 + SESSION_NODES.length * (W + gap)} 110`} style={SVG}>
        <defs><Arrow id="se" color={l} /><Arrow id="sea" color={a} /></defs>
        {SESSION_NODES.map((label, idx) => {
          const active = idx === f.stage;
          const done = idx < f.stage;
          const x = xs[idx];
          const edgePath = idx < SESSION_NODES.length - 1
            ? `M${x + W},${y + H / 2} L${x + W + gap},${y + H / 2}`
            : "";
          const subFor = maint
            ? ["unit received", "append turn", "decide()", "stable_action_id()", "retire"][idx]
            : undefined;
          return (
            <g key={label}>
              {active && (
                <circle cx={x + W / 2} cy={y + H / 2} r={20} fill={a}
                  style={{ animation: "sr-pulse 1.6s ease-in-out infinite" }} />
              )}
              <Node t={t} x={x} y={y} w={W} h={H}
                fill={active ? a : done ? t.fill.secondary : n}
                stroke={active ? a : done ? g : s}
                accent={active ? a : done ? g : undefined}
                title={label} sub={subFor}
                titleFill={active ? t.text.onAccent : undefined}
                dim={!active && !done} />
              {edgePath && (
                <>
                  <Edge d={edgePath} color={done ? g : idx === f.stage ? a : l} id={done ? "sea" : "se"} moving={active} dur={1.4} />
                  {active && <Particles d={edgePath} color={a} n={1} dur={1.4} />}
                </>
              )}
            </g>
          );
        })}
        {/* quarantine branch off closure decision */}
        {atLeast(level, "developer") && (() => {
          const cx = xs[2] + W / 2;
          const d = `M${cx},${y + H} L${cx},${y + H + 36}`;
          return (
            <g>
              <Edge d={d} color={red} id="se" moving={false} />
              <Node t={t} x={cx - 70} y={y + H + 36} w={140} h={32} fill={t.fill.secondary} stroke={red}
                title="quarantine" sub={maint ? "causal gap · repeat error" : "unresolvable failure"} />
              {f.stage === 2 && <Particles d={d} color={red} n={1} dur={3} />}
            </g>
          );
        })()}
      </svg>

      <Callout tone={f.stage === 4 ? "success" : f.stage === 2 ? "info" : "neutral"} title={f.label}>{f.cap}</Callout>

      {maint && (
        <Callout tone="info" title="P1B closure semantics">
          <Text as="span" weight="semibold">FiniteSeal &#123; has_causal_gap: false &#125;</Text> always returns{" "}
          <Text as="span" weight="semibold">Close</Text>. A gap with <Text as="span" weight="semibold">finite_seal_requires_complete</Text>{" "}
          set returns <Text as="span" weight="semibold">Fail</Text>, not a silent partial retire.
        </Callout>
      )}
    </Stack>
  );
}

// ═══ PAGE 4 · Pipeline fused event loop ════════════════════════════════════

interface PipelineStep { stage: number; cap: string; }
const PIPELINE_STEPS: PipelineStep[] = [
  { stage: 0, cap: "Settlement: the checkpoint coordinator publishes a barrier advancement, allowing the pipeline to commit the current epoch boundary." },
  { stage: 1, cap: "Actions: the turn-closure intake VecDeque drains closed turn receipts into placed actions. The placement policy routes each to a worker." },
  { stage: 2, cap: "Stop signal: the stop receiver fires — the pipeline begins a graceful drain, waiting for all in-flight actions to reach a terminal receipt." },
  { stage: 3, cap: "Barrier: a checkpoint barrier is received. The pipeline suspends admission until the coordinator acknowledges the commit." },
  { stage: 4, cap: "Admit: the bounded capacity window has room. A new session's first action is admitted and handed to the sink for dispatch." },
];
const PIPELINE_STAGES = ["Settlement", "Actions", "Stop", "Barrier", "Admit"];

function PipelinePage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [i, setI] = useCanvasState<number>("sr.plFrame", 0);
  const f = PIPELINE_STEPS[Math.min(i, PIPELINE_STEPS.length - 1)];
  const n = t.fill.tertiary, l = t.stroke.secondary, s = t.stroke.primary;
  const a = t.accent.primary, or = t.category.orange, p = t.category.purple;
  const maint = level === "maintainer";

  // five stages arranged in a pentagon-ish cycle
  const positions = [
    { x: 194, y: 12 },  // Settlement (top)
    { x: 340, y: 80 },  // Actions (right-top)
    { x: 290, y: 190 }, // Stop (right-bottom)
    { x: 88,  y: 190 }, // Barrier (left-bottom)
    { x: 36,  y: 80 },  // Admit (left-top)
  ];
  const W = 136, H = 44;
  const centerEdges = [
    `M${330},${34} L${340},${80}`,
    `M${395},${124} L${360},${190}`,
    `M${278},${212} L${224},${212}`,
    `M${88},${190} L${52},${124}`,
    `M${36},${80} L${194},${28}`,
  ];

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The pipeline is a <Text as="span" weight="semibold">select_biased!</Text> loop that handles exactly five
        event types in priority order. Admission is bounded by capacity — the source is stalled when the window
        fills. In cellular mode, the cellular transport slot also feeds the Actions arm.
      </Text>

      <Row gap={8} align="center" wrap>
        <Button variant="secondary" disabled={i === 0} onClick={() => setI((v) => Math.max(0, v - 1))}>Back</Button>
        <Button variant="primary" disabled={i >= PIPELINE_STEPS.length - 1} onClick={() => setI((v) => Math.min(PIPELINE_STEPS.length - 1, v + 1))}>Step</Button>
        <Button variant="ghost" disabled={i === 0} onClick={() => setI(0)}>Reset</Button>
        <Spacer />
        <Pill size="sm">{`${i + 1} / ${PIPELINE_STEPS.length}`}</Pill>
      </Row>

      <svg viewBox="0 0 540 260" style={SVG}>
        <defs><Arrow id="pl" color={l} /><Arrow id="pla" color={a} /></defs>
        {centerEdges.map((d, idx) => {
          const incoming = idx === f.stage;
          return (
            <g key={idx}>
              <Edge d={d} color={incoming ? a : l} id={incoming ? "pla" : "pl"} moving={incoming} dur={1.5} />
              {incoming && <Particles d={d} color={a} n={1} dur={1.5} />}
            </g>
          );
        })}
        {positions.map((pos, idx) => {
          const active = idx === f.stage;
          const subFor = maint
            ? ["CheckpointBarrier", "TurnClosureIntake drain", "StopReceiver", "CheckpointBarrier suspend", "capacity window"][idx]
            : undefined;
          return (
            <g key={PIPELINE_STAGES[idx]}>
              {active && (
                <circle cx={pos.x + W / 2} cy={pos.y + H / 2} r={22} fill={a}
                  style={{ animation: "sr-pulse 1.6s ease-in-out infinite" }} />
              )}
              <Node t={t} x={pos.x} y={pos.y} w={W} h={H}
                fill={active ? a : n} stroke={active ? a : s}
                accent={active ? a : undefined}
                title={PIPELINE_STAGES[idx]} sub={subFor}
                titleFill={active ? t.text.onAccent : undefined}
                dim={!active} />
            </g>
          );
        })}
        {/* placement dispatch to workers */}
        {atLeast(level, "developer") && (
          <>
            <text x={410} y={80} fontSize={9} fontWeight={600} fill={or}>placement</text>
            {[0, 1, 2].map((wi) => {
              const wx = 404, wy = 100 + wi * 44;
              const d = `M${476},${80 + wi * 44 + 5} L${wx},${wy + 18}`;
              return (
                <g key={`w${wi}`}>
                  <Edge d={d} color={or} id="pl" dur={2.4} />
                  <Node t={t} x={wx} y={wy} w={56} h={34} fill={n} stroke={s} accent={or}
                    title={`worker ${wi}`} />
                </g>
              );
            })}
            <path d="M476,122 L404,104" fill="none" stroke={or} strokeWidth={1.2} strokeDasharray="4 4" markerEnd="url(#pl)" />
            <path d="M476,130 L460,148" fill="none" stroke={p} strokeWidth={1.2} strokeDasharray="4 4" markerEnd="url(#pl)" />
            <text x={456} y={164} fontSize={8.5} fill={p}>cellular</text>
          </>
        )}
      </svg>

      <Callout tone={f.stage === 2 ? "warning" : f.stage === 3 ? "info" : "neutral"}
        title={PIPELINE_STAGES[f.stage]}>{f.cap}</Callout>
    </Stack>
  );
}

// ═══ PAGE 5 · Shadow replay workload ════════════════════════════════════════

function ShadowPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const n = t.fill.tertiary, l = t.stroke.secondary, s = t.stroke.primary;
  const a = t.accent.primary, or = t.category.orange, g = t.category.green, red = t.category.red;
  const maint = level === "maintainer";

  // main flow: action → sink → endpoint → inventory → epoch
  const flow = [
    { x: 20, title: "Request action", sub: maint ? "DatasetActionV1::Request" : "from pipeline", accent: or, w: 140 },
    { x: 184, title: "ScheduledRequestSink", sub: maint ? "scheduled_request.rs · P4" : "issue real request", accent: or, w: 170 },
    { x: 378, title: "HTTP / gRPC endpoint", sub: maint ? "chat · completions · responses" : "live server", accent: a, w: 164 },
  ];
  const invX = 184, invY = 140;
  const epochX = 378, epochY = 140;

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The <Text as="span" weight="semibold">shadow_replay</Text> workload type is registered in the engine alongside{" "}
        <Text as="span" weight="semibold">scheduled</Text> and <Text as="span" weight="semibold">graph</Text>. Its
        Config-v2 <Text as="span" weight="semibold">shadow_replay:</Text> section maps action kinds to sink implementations,
        selects timing mode, and configures overload behavior. The scheduled-request sink issues the recorded request
        against a real endpoint; the action inventory ledger closes gaps for ordered result delivery.
      </Text>
      <svg viewBox="0 0 560 230" style={SVG}>
        <defs><Arrow id="sh" color={l} /><Arrow id="sha" color={a} /><Arrow id="shor" color={or} /><Arrow id="shg" color={g} /><Arrow id="shr" color={red} /></defs>

        {/* top row: action → sink → endpoint */}
        {flow.map((nd, i) => {
          const x = nd.x, w = nd.w, edgePath = i < flow.length - 1 ? `M${x + w},${51} L${flow[i + 1].x},${51}` : "";
          return (
            <g key={nd.title}>
              <Node t={t} x={x} y={20} w={w} h={62} fill={n} stroke={s} accent={nd.accent} title={nd.title} sub={nd.sub} />
              {edgePath && (
                <>
                  <Edge d={edgePath} color={l} id="sh" dur={2.2} />
                  <Particles d={edgePath} color={nd.accent} n={2} dur={2.2} />
                </>
              )}
            </g>
          );
        })}

        {/* sink → inventory (down) */}
        <Edge d={`M${invX + 85},82 L${invX + 85},${invY}`} color={or} id="shor" dur={1.8} />
        <Particles d={`M${invX + 85},82 L${invX + 85},${invY}`} color={or} n={1} dur={1.8} />
        <Node t={t} x={invX} y={invY} w={170} h={56} fill={n} stroke={s} accent={or}
          title="ActionInventory" sub={maint ? "action/host/inventory.rs" : "dense gap-closure"}
          micro={maint ? "in-flight · done · failed" : undefined} />

        {/* endpoint → inventory (result) */}
        <path d={`M${epochX + 80},82 C${epochX + 80},120 ${invX + 170},${invY + 28} ${invX + 170},${invY + 28}`}
          fill="none" stroke={g} strokeWidth={1.4} strokeDasharray="4 5" markerEnd="url(#shg)"
          style={{ animation: "sr-flow 2.2s linear infinite" }} />
        <Particles d={`M${epochX + 80},82 C${epochX + 80},120 ${invX + 170},${invY + 28} ${invX + 170},${invY + 28}`} color={g} n={1} dur={2.2} />

        {/* inventory → epoch */}
        <Edge d={`M${invX + 170},${invY + 28} L${epochX},${invY + 28}`} color={g} id="shg" dur={2} />
        <Particles d={`M${invX + 170},${invY + 28} L${epochX},${invY + 28}`} color={g} n={1} dur={2} />
        <Node t={t} x={epochX} y={invY} w={164} h={56} fill={n} stroke={s} accent={g}
          title="EpochCoordinator" sub={maint ? "results/epoch.rs · 6B" : "rotate + compact + export"} />

        {/* timing labels */}
        {atLeast(level, "developer") && (() => {
          const lx = 20;
          return (
            <g>
              <rect x={lx} y={155} width={148} height={30} rx={7} fill={t.fill.secondary} stroke={l} />
              <text x={lx + 74} y={165} textAnchor="middle" fontSize={8} fontWeight={700} fill={a}>TIMING MODE</text>
              <text x={lx + 74} y={178} textAnchor="middle" fontSize={9} fill={t.text.secondary}>relative · absolute</text>
            </g>
          );
        })()}
        {atLeast(level, "developer") && (() => {
          const lx = 20;
          return (
            <g>
              <rect x={lx} y={194} width={148} height={30} rx={7} fill={t.fill.secondary} stroke={red} />
              <text x={lx + 74} y={204} textAnchor="middle" fontSize={8} fontWeight={700} fill={red}>OVERLOAD POLICY</text>
              <text x={lx + 74} y={218} textAnchor="middle" fontSize={9} fill={t.text.secondary}>backpressure · shed</text>
            </g>
          );
        })()}
      </svg>

      <Grid columns={2} gap={12}>
        <Callout tone="info" title="endpoint_retry_safety = Unproven">
          The sink refuses a non-zero authored retry limit — retrying a real endpoint duplicates measured load and cannot be proven duplicate-safe without application-level idempotency tokens.
        </Callout>
        {atLeast(level, "developer") && (
          <Callout tone="neutral" title="Config-v2 preflight (V1)">
            The engine validates <Text as="span" weight="semibold">dataset_streams:</Text> ids, resolves the
            shadow_replay stream reference, and round-trips StreamingReliabilityPolicy through strict serde{" "}
            before any S3 listing or endpoint socket is opened.
          </Callout>
        )}
      </Grid>
    </Stack>
  );
}

// ═══ PAGE 6 · Checkpoint backends ══════════════════════════════════════════

type BackendId = "local" | "none" | "cas";

function CheckpointPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [backend, setBackend] = useCanvasState<BackendId>("sr.backend", "local");
  const n = t.fill.tertiary, l = t.stroke.secondary, s = t.stroke.primary;
  const a = t.accent.primary, g = t.category.green, gr = t.category.gray, blue = t.category.blue;
  const maint = level === "maintainer";

  const info: Record<BackendId, { title: string; color: string; rows: [string, string][] }> = {
    local: {
      title: "LocalCheckpointBackend",
      color: g,
      rows: [
        ["commit", maint ? "atomic rename → fsync'd CURRENT" : "atomic file write"],
        ["resume", maint ? "re-read CURRENT · skip fully-read partitions" : "resume mid-partition"],
        ["crash-safe", "no partial commit visible after restart"],
        ["concurrency", "one writer · one reader · no lock needed"],
        ...(maint ? [["path", ".aiperf-checkpoint/<run-id>/CURRENT"] as [string, string]] : []),
      ],
    },
    none: {
      title: "NoneCheckpointBackend (5F1)",
      color: gr,
      rows: [
        ["persistence", "none — every run starts from the beginning"],
        ["use case", "development, dry-run, quick one-shot replay"],
        ["cost", "zero disk I/O"],
        ["caution", "re-reads all partitions on restart"],
      ],
    },
    cas: {
      title: "ObjectStoreCasBackend (5F2)",
      color: blue,
      rows: [
        ["commit", maint ? "S3 conditional PUT (if-none-match ETag)" : "compare-and-swap write"],
        ["use case", "distributed replay across multiple controller instances"],
        ["coordination", "whoever wins the CAS owns the partition cursor"],
        ["fencing", "stale writers lose the CAS and backoff"],
        ...(maint ? [["key", "s3://bucket/prefix/<run-id>/checkpoint"] as [string, string]] : []),
      ],
    },
  };

  const cur = info[backend];
  const rh = 32;
  const svgH = 28 + cur.rows.length * rh + 8;

  // checkpoint flow: coordinator → backend → disk/S3
  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The checkpoint backend persists a read cursor so the source can resume after a restart without re-reading
        fully processed partitions. Three backends ship: <Text as="span" weight="semibold">local</Text> (the default),{" "}
        <Text as="span" weight="semibold">none</Text> (no persistence), and{" "}
        <Text as="span" weight="semibold">cas</Text> (object-store compare-and-swap for distributed replay).
      </Text>

      <SegControl<BackendId> value={backend} set={setBackend} opts={[
        { id: "local", label: "local" },
        { id: "none", label: "none" },
        { id: "cas", label: "cas (S3)" },
      ]} />

      <svg viewBox={`0 0 560 ${svgH}`} style={{ ...SVG }}>
        <defs><Arrow id="cp" color={l} /></defs>
        <g key={backend} style={POP}>
          <rect x={12} y={12} width={536} height={svgH - 20} rx={11} fill={t.fill.secondary} stroke={cur.color} strokeWidth={1.6} />
          <text x={30} y={30} fontSize={10} fontWeight={700} fill={cur.color} style={{ letterSpacing: 0.4 }}>{cur.title.toUpperCase()}</text>
          {cur.rows.map(([k, v], ri) => (
            <g key={k}>
              <text x={30} y={28 + ri * rh + rh} fontSize={9.5} fontWeight={600} fill={t.text.secondary}>{k}</text>
              <text x={140} y={28 + ri * rh + rh} fontSize={10} fill={t.text.primary}>{v}</text>
            </g>
          ))}
        </g>
      </svg>

      {atLeast(level, "developer") && (
        <Callout tone="info" title="shadow_replay.checkpoint.mode">
          Set <Text as="span" weight="semibold">mode: periodic</Text> with an{" "}
          <Text as="span" weight="semibold">interval_seconds</Text> and a{" "}
          <Text as="span" weight="semibold">backend:</Text> entry to enable checkpointing.
          Set <Text as="span" weight="semibold">mode: none</Text> (default) for stateless replay.
        </Callout>
      )}
      {maint && (
        <Callout tone="neutral" title="CheckpointCoordinator (5E)">
          A publication sequencer orders local backend writes and sequences them with the pipeline's barrier events.
          The coordinator owns the only write path to the backend — no two pipeline steps can commit a checkpoint concurrently.
        </Callout>
      )}
    </Stack>
  );
}

// ── shared segmented control ─────────────────────────────────────────────────

function SegControl<T extends string>({
  value, set, opts, size = "md",
}: { value: T; set: (v: T) => void; opts: { id: T; label: string }[]; size?: "sm" | "md" }) {
  const t = useHostTheme();
  const pad = size === "sm" ? "5px 10px" : "7px 13px";
  const fs = size === "sm" ? 11.5 : 12.5;
  return (
    <Row gap={5} wrap>
      {opts.map((o) => {
        const on = value === o.id;
        return (
          <div key={o.id} style={{ display: "contents" }}>
            <button type="button" onClick={() => set(o.id)}
              style={{
                padding: pad, borderRadius: 8,
                border: `1px solid ${on ? t.accent.primary : t.stroke.secondary}`,
                background: on ? t.accent.control : t.fill.tertiary,
                color: on ? t.text.onAccent : t.text.primary,
                fontSize: fs, fontWeight: on ? 600 : 400, cursor: "pointer",
              }}>
              {o.label}
            </button>
          </div>
        );
      })}
    </Row>
  );
}

// ── pages registry ────────────────────────────────────────────────────────────

type El = ReturnType<typeof H1>;
const PAGES: { id: PageId; label: string; title: string; render: (level: Level) => El }[] = [
  { id: "overview",    label: "Overview",    title: "End-to-end streaming pipeline",              render: (lv) => <OverviewPage level={lv} /> },
  { id: "source",      label: "Source",      title: "Acquire and decode trace files",             render: (lv) => <SourcePage level={lv} /> },
  { id: "session",     label: "Session",     title: "Join fragments into conversations",          render: (lv) => <SessionPage level={lv} /> },
  { id: "pipeline",    label: "Pipeline",    title: "Fused event loop — bounded admission",       render: (lv) => <PipelinePage level={lv} /> },
  { id: "shadow",      label: "Shadow",      title: "Re-execute recorded requests",               render: (lv) => <ShadowPage level={lv} /> },
  { id: "checkpoint",  label: "Checkpoint",  title: "Checkpoint backends — resumable from crash", render: (lv) => <CheckpointPage level={lv} /> },
];

const SRC: Record<PageId, { l: string; p: string }[]> = {
  overview:   [{ l: "streaming_execution.rs", p: "rust/runtime/src/engine/streaming_execution.rs" }, { l: "pipeline.rs", p: "rust/runtime/src/streaming/pipeline.rs" }],
  source:     [{ l: "s3.rs", p: "rust/runtime/src/streaming/sources/s3.rs" }, { l: "local.rs", p: "rust/runtime/src/streaming/sources/local.rs" }, { l: "streaming_dynamo.rs", p: "rust/runtime/src/streaming/formats/streaming_dynamo.rs" }],
  session:    [{ l: "conversation.rs", p: "rust/runtime/src/streaming/session/conversation.rs" }, { l: "closure.rs", p: "rust/runtime/src/streaming/session/closure.rs" }, { l: "host.rs", p: "rust/runtime/src/streaming/session/host.rs" }],
  pipeline:   [{ l: "pipeline.rs", p: "rust/runtime/src/streaming/pipeline.rs" }, { l: "placement.rs", p: "rust/runtime/src/streaming/placement.rs" }],
  shadow:     [{ l: "scheduled_request.rs", p: "rust/runtime/src/streaming/action/scheduled_request.rs" }, { l: "inventory.rs", p: "rust/runtime/src/streaming/action/host/inventory.rs" }, { l: "workload_kind.rs", p: "rust/runtime/src/config/model/workload_kind.rs" }],
  checkpoint: [{ l: "checkpoint_coordinator.rs", p: "rust/runtime/src/streaming/checkpoint_coordinator.rs" }, { l: "local.rs", p: "rust/runtime/src/streaming/checkpoints/local.rs" }, { l: "none.rs", p: "rust/runtime/src/streaming/checkpoints/none.rs" }],
};

// ── detail toggle ─────────────────────────────────────────────────────────────

function DetailToggle() {
  const t = useHostTheme();
  const [level, setLevel] = useLevel();
  const opts: Level[] = ["executive", "developer", "maintainer"];
  return (
    <Row gap={4}>
      {opts.map((o) => {
        const on = level === o;
        return (
          <div key={o} style={{ display: "contents" }}>
            <button type="button" onClick={() => setLevel(o)}
              style={{
                padding: "5px 10px", borderRadius: 7, textTransform: "capitalize",
                border: `1px solid ${on ? t.accent.primary : t.stroke.secondary}`,
                background: on ? t.accent.control : t.fill.tertiary,
                color: on ? t.text.onAccent : t.text.secondary,
                fontSize: 11.5, fontWeight: on ? 600 : 400, cursor: "pointer",
              }}>
              {o}
            </button>
          </div>
        );
      })}
    </Row>
  );
}

// ── root canvas ───────────────────────────────────────────────────────────────

export default function StreamingDynamoShadowReplayCanvas() {
  const t = useHostTheme();
  const [page, setPage] = usePage();
  const [level] = useLevel();
  const dispatch = useCanvasAction();
  const current = PAGES.find((p) => p.id === page) ?? PAGES[0];
  const idx = PAGES.findIndex((p) => p.id === current.id);

  return (
    <div style={{ padding: 22, background: t.bg.editor, minHeight: "100%" }}>
      <style>{CSS}</style>
      <Stack gap={16}>
        <Row gap={12} align="center" wrap>
          <H1>Streaming · Dynamo Shadow Replay</H1>
          <Spacer />
          <DetailToggle />
        </Row>

        {/* page tabs */}
        <Row gap={6} wrap>
          {PAGES.map((p, i) => {
            const on = p.id === page;
            return (
              <div key={p.id} style={{ display: "contents" }}>
                <button type="button" onClick={() => setPage(p.id)}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 7,
                    padding: "6px 12px", borderRadius: 8,
                    border: `1px solid ${on ? t.accent.primary : t.stroke.secondary}`,
                    background: on ? t.accent.control : t.fill.tertiary,
                    color: on ? t.text.onAccent : t.text.primary,
                    fontSize: 12.5, fontWeight: on ? 600 : 400, cursor: "pointer",
                  }}>
                  <span style={{
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    width: 17, height: 17, borderRadius: 9999, fontSize: 10, fontWeight: 600,
                    background: on ? t.text.onAccent : t.fill.primary,
                    color: on ? t.accent.primary : t.text.tertiary,
                  }}>{i + 1}</span>
                  {p.label}
                </button>
              </div>
            );
          })}
        </Row>

        <Card>
          <CardHeader trailing={<Pill size="sm">{`${idx + 1} / ${PAGES.length}`}</Pill>}>{current.title}</CardHeader>
          <CardBody>
            <Stack gap={16}>
              {current.render(level)}

              {level === "maintainer" && (
                <>
                  <Divider />
                  <Row gap={6} align="center" wrap>
                    <Text size="small" tone="tertiary">src</Text>
                    {SRC[current.id].map((sr) => (
                      <div key={sr.p} style={{ display: "contents" }}>
                        <Button variant="ghost" onClick={() => dispatch({ type: "openFile", path: sr.p })}>{sr.l}</Button>
                      </div>
                    ))}
                  </Row>
                </>
              )}
            </Stack>
          </CardBody>
        </Card>

        <Row gap={8} align="center">
          <Button variant="secondary" disabled={idx === 0} onClick={() => setPage(PAGES[Math.max(0, idx - 1)].id)}>← Prev</Button>
          <Spacer />
          <Text size="small" tone="tertiary">{current.label}</Text>
          <Spacer />
          <Button variant="primary" disabled={idx === PAGES.length - 1} onClick={() => setPage(PAGES[Math.min(PAGES.length - 1, idx + 1)].id)}>Next →</Button>
        </Row>
      </Stack>
    </div>
  );
}
