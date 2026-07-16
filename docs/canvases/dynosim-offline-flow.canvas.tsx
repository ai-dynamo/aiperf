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
// Dynosim OFFLINE — a small paged explainer. One focused diagram per page,
// calm purposeful motion. Grounded in aiperf-cli profile → load/yaml →
// aiperf --execute → RunnerApplication → offline_execution → dynosim.rs →
// graph/runtime.rs → parity. Detail: Executive | Developer | Maintainer.
// ═══════════════════════════════════════════════════════════════════════════

type Level = "executive" | "developer" | "maintainer";
type Theme = ReturnType<typeof useHostTheme>;
type PageId = "overview" | "launch" | "seams" | "loop" | "dispatch" | "parity" | "engine";

const CSS = `
@keyframes ds-flow { to { stroke-dashoffset: -40; } }
@keyframes ds-breathe { 0%,100% { opacity: .25; r: 26px } 50% { opacity: .7; r: 30px } }
@keyframes ds-soft { 0%,100% { opacity: .4 } 50% { opacity: 1 } }
@keyframes ds-scan { 0% { transform: translateY(0) } 100% { transform: translateY(var(--sh)) } }
@keyframes ds-pop { 0% { opacity: 0; transform: scale(.78) } 62% { opacity: 1; transform: scale(1.05) } 100% { transform: scale(1) } }
@keyframes ds-glow { 0%,100% { opacity: .5 } 50% { opacity: 1 } }
`;

const POP = { transformBox: "fill-box", transformOrigin: "center", animation: "ds-pop .42s cubic-bezier(.2,.8,.3,1.2) both" } as const;

const SVG = { width: "100%", height: "auto", display: "block" } as const;

function useLevel() {
  return useCanvasState<Level>("dyn.level", "developer");
}
function usePage() {
  return useCanvasState<PageId>("dyn.page", "overview");
}
function atLeast(level: Level, min: Level): boolean {
  const r = { executive: 0, developer: 1, maintainer: 2 } as const;
  return r[level] >= r[min];
}

// ── shared SVG atoms ─────────────────────────────────────────────────────────

function Arrow({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill={color} />
    </marker>
  );
}

function Node({
  x, y, w, h, t, fill, stroke, accent, title, sub, micro, dim, titleFill,
}: {
  x: number; y: number; w: number; h: number; t: Theme;
  fill: string; stroke: string; accent?: string;
  title: string; sub?: string; micro?: string; dim?: boolean; titleFill?: string;
}) {
  const cx = x + w / 2;
  const lines = (sub ? 1 : 0) + (micro ? 1 : 0);
  const ty = lines === 0 ? y + h / 2 : lines === 1 ? y + h / 2 - 4 : y + h / 2 - 9;
  return (
    <g opacity={dim ? 0.45 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={fill} stroke={accent ?? stroke} strokeWidth={accent ? 1.6 : 1} />
      <text x={cx} y={ty} textAnchor="middle" dominantBaseline="middle" fontSize={12.5} fontWeight={600} fill={titleFill ?? t.text.primary}>{title}</text>
      {sub && <text x={cx} y={ty + 14} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>{sub}</text>}
      {micro && <text x={cx} y={ty + 25} textAnchor="middle" fontSize={8.5} fill={t.text.quaternary}>{micro}</text>}
    </g>
  );
}

function Edge({ d, color, id, moving = true, dur = 2.4 }: { d: string; color: string; id: string; moving?: boolean; dur?: number }) {
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={1.4}
      markerEnd={`url(#${id})`}
      strokeDasharray="4 5"
      style={moving ? { animation: `ds-flow ${dur}s linear infinite` } : undefined}
    />
  );
}

function Particles({ d, color, n, dur, shape = "dot", size = 4 }: { d: string; color: string; n: number; dur: number; shape?: "dot" | "sq"; size?: number }) {
  // Each particle fades in at the start of its trip and out before the end, so
  // the `animateMotion` loop restart (an instant teleport to the path origin)
  // is never visible — no jarring reset.
  const fade = (begin: string) => (
    <animate
      attributeName="opacity"
      dur={`${dur}s`}
      begin={begin}
      repeatCount="indefinite"
      values="0;1;1;0"
      keyTimes="0;0.12;0.85;1"
      calcMode="linear"
    />
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

// ═══ PAGE 1 · Overview (layered architecture) ════════════════════════════════

function Chip({ x, y, w, t, label, sub, accent }: { x: number; y: number; w: number; t: Theme; label: string; sub?: string; accent?: string }) {
  const cx = x + w / 2;
  return (
    <g>
      <rect x={x} y={y} width={w} height={40} rx={7} fill={t.fill.tertiary} stroke={accent ?? t.stroke.primary} strokeWidth={accent ? 1.5 : 1} />
      <text x={cx} y={sub ? y + 18 : y + 24} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>{label}</text>
      {sub && <text x={cx} y={y + 31} textAnchor="middle" fontSize={8.5} fill={t.text.tertiary}>{sub}</text>}
    </g>
  );
}

function OverviewPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const l = t.stroke.secondary, band = t.fill.quaternary;
  const a = t.accent.primary, g = t.category.green, p = t.category.purple, blue = t.category.blue;
  const maint = level === "maintainer";

  // four stacked layers; flow runs down the left rail, report exits right
  const bands = [
    { y: 10, name: "Rust CLI entry point", tone: blue },
    { y: 90, name: "Execution engine", tone: a },
    { y: 170, name: "aiperf library", tone: g },
    { y: 288, name: "engine / wire", tone: p },
  ];
  const railX = 150;

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The native path is a four-layer stack: the Rust CLI authors Config v2 and re-execs itself in <Text as="span" weight="semibold">--execute</Text> mode, the frozen <Text as="span" weight="semibold">RunnerApplication</Text> validates and dispatches, the library runs the shared benchmark loop, and the bottom layer is either a real server or Dynamo's in-process engine. Flow runs top→down; the report exits the side.
      </Text>
      <svg viewBox="0 0 700 372" style={SVG}>
        <defs><Arrow id="ov" color={l} /><Arrow id="ovg" color={g} /></defs>

        {/* layer bands */}
        {bands.map((b, i) => {
          const h = i === 2 ? 100 : 66;
          return (
            <g key={b.name}>
              <rect x={12} y={b.y} width={676} height={h} rx={11} fill={band} stroke={l} />
              <rect x={12} y={b.y} width={5} height={h} rx={2} fill={b.tone} />
              <text x={30} y={b.y + 20} fontSize={10} fontWeight={700} fill={b.tone} style={{ letterSpacing: 0.4 }}>{b.name.toUpperCase()}</text>
            </g>
          );
        })}

        {/* inter-layer flow — particles ride only the gaps between bands, never
            over the chips inside them */}
        {["M350,76 L350,90", "M350,156 L350,170", "M350,270 L350,288"].map((d, i) => (
          <g key={`f${i}`}>
            <Edge d={d} color={l} id="ov" dur={2.4} />
            <Particles d={d} color={a} n={1} dur={1.3} />
          </g>
        ))}

        {/* CLI entry point */}
        <Chip t={t} x={150} y={30} w={160} label="aiperf profile" sub={maint ? "load.rs / yaml.rs" : "native Config v2"} accent={blue} />
        <Chip t={t} x={330} y={30} w={200} label="BenchmarkRun" sub={maint ? "RunnerRequest execute" : "v2 wire envelope"} accent={blue} />

        {/* Execution engine (--execute re-exec) */}
        <Chip t={t} x={150} y={110} w={150} label="aiperf --execute" sub={maint ? "execute_mode.rs" : "same binary"} accent={a} />
        <Chip t={t} x={320} y={110} w={150} label="RunnerApplication" sub={maint ? "handle_v2" : "frozen registry"} accent={a} />
        <Chip t={t} x={490} y={110} w={160} label="transport + workload" sub={maint ? "AIPerfRegistry" : "independent registries"} accent={a} />

        {/* Library layer (tall — the shared loop + the two seams) */}
        <Chip t={t} x={150} y={192} w={150} label="Workload" sub={maint ? "RequestRate · Graph" : "arrival pattern"} accent={g} />
        <Chip t={t} x={320} y={192} w={180} label="ScheduledRuntime" sub={maint ? "SlotPool · StopChecker" : "schedule + admit"} accent={g} />
        <Chip t={t} x={520} y={192} w={130} label="ObserverTee" sub={maint ? "Collector+Native" : "metrics"} accent={g} />
        {/* the two seams, dashed */}
        <g>
          <rect x={150} y={240} width={220} height={34} rx={7} fill={t.accent.control} stroke={a} strokeWidth={1.4} strokeDasharray="5 4" />
          <text x={158} y={252} fontSize={7} fontWeight={700} fill={a}>SEAM</text>
          <text x={260} y={261} textAnchor="middle" fontSize={10.5} fontWeight={700} fill={t.text.onAccent}>Clock</text>
          <rect x={390} y={240} width={260} height={34} rx={7} fill={t.accent.control} stroke={a} strokeWidth={1.4} strokeDasharray="5 4" />
          <text x={398} y={252} fontSize={7} fontWeight={700} fill={a}>SEAM</text>
          <text x={520} y={261} textAnchor="middle" fontSize={10.5} fontWeight={700} fill={t.text.onAccent}>RequestSink&lt;HttpRequest&gt;</text>
        </g>

        {/* Engine / wire layer — the fork */}
        <Chip t={t} x={150} y={308} w={230} label="HttpTransport → real server" sub="Hyper + SSE" accent={blue} />
        <Chip t={t} x={410} y={308} w={240} label="EngineHost → SteppableReplay" sub="in-process · no sockets" accent={p} />
        <text x={350} y={352} textAnchor="middle" fontSize={8.5} fill={t.text.quaternary}>forks by mode → see Architecture page</text>

        {/* report exits the library layer to the right */}
        <path d="M660,225 L682,225 L682,52 L534,52" fill="none" stroke={g} strokeWidth={1.5} markerEnd="url(#ovg)" strokeDasharray="4 4" />
        <Particles d="M660,225 L682,225 L682,52 L534,52" color={g} n={2} dur={3} />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Shared above the seam">Layers 1–3 are identical for every mode; only the bottom transport fork changes.</Callout>
        <Callout tone="info" title="No sockets (dynosim)">The right fork feeds token arrays straight to the engine.</Callout>
        <Callout tone="success" title="Verified">AIPerf's summary must byte-match Dynamo's, or the run fails.</Callout>
      </Grid>
    </Stack>
  );
}

// ═══ PAGE 2 · Launch & preflight ═════════════════════════════════════════════

function LaunchPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const n = t.fill.tertiary, s = t.stroke.primary, l = t.stroke.secondary;
  const a = t.accent.primary, ok = t.category.green, bad = t.category.red;
  const maint = level === "maintainer";

  const steps = [
    { title: "project", sub: maint ? "load.rs → BenchmarkRun" : "native Config v2" },
    { title: "registry", sub: maint ? "dynosim_offline registered" : "transport exists?" },
    { title: "re-exec", sub: maint ? "aiperf --execute stdin" : "same binary child" },
    { title: "handle_v2", sub: maint ? "offline_execution" : "run + report" },
  ];
  const w = 158, gap = 46, x0 = 26, y = 48, h = 60;
  const rejGap = 34, rejH = 38;
  const contentW = x0 * 2 + steps.length * w + (steps.length - 1) * gap;
  const contentH = y + h + rejGap + rejH + 16;
  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The native CLI projects one protocol-v2 execute envelope and re-execs itself in <Text as="span" weight="semibold">--execute</Text> mode. The frozen <Text as="span" weight="semibold">RunnerApplication</Text> registry must already include <Text as="span" weight="semibold">dynosim_offline</Text> (feature-gated at compile time). Fail-closed, no fallback.
      </Text>
      <svg viewBox={`0 0 ${contentW} ${contentH}`} style={SVG}>
        <defs><Arrow id="lp" color={l} /><Arrow id="lg" color={ok} /><Arrow id="lb" color={bad} /></defs>
        {steps.map((st, i) => {
          const x = x0 + i * (w + gap);
          const path = `M${x + w},${y + h / 2} L${x + w + gap},${y + h / 2}`;
          const last = i === steps.length - 1;
          return (
            <g key={st.title}>
              {!last && <Edge d={path} color={i === steps.length - 2 ? ok : l} id={i === steps.length - 2 ? "lg" : "lp"} dur={2.2} />}
              <Node t={t} x={x} y={y} w={w} h={h} fill={n} stroke={s}
                accent={last ? ok : i === 1 ? a : undefined} title={st.title} sub={st.sub} />
              {!last && <Particles d={path} color={i >= steps.length - 2 ? ok : a} n={1} dur={2.2} />}
            </g>
          );
        })}
        {/* single reject branch off the registry gate (step 1) */}
        {(() => {
          const gx = x0 + 1 * (w + gap) + w / 2;
          const path = `M${gx},${y + h} L${gx},${y + h + rejGap}`;
          return (
            <g>
              <Edge d={path} color={bad} id="lb" moving={false} />
              <Node t={t} x={gx - 74} y={y + h + rejGap} w={148} h={rejH} fill={t.fill.secondary} stroke={bad}
                title="reject" sub={maint ? "transport unregistered" : "unsupported"} />
              <Particles d={path} color={bad} n={1} dur={3} />
            </g>
          );
        })()}
      </svg>
      {atLeast(level, "developer") && (
        <Callout tone="neutral" title="What fails closed">
          Base build without the <Text as="span" weight="semibold">dynosim</Text> Cargo feature (transport absent from registry) · non-v2 envelope · authored <Text as="span" weight="semibold">required_features</Text> the linked image wasn't compiled with · unregistered transport or workload id.
        </Callout>
      )}
    </Stack>
  );
}

// ═══ PAGE 3 · The simulation loop (step-through centerpiece) ═════════════════

interface Frame {
  stage: number; // 0 Poll 1 Compare 2 Advance 3 Step 4 Route
  vt: number;    // tick index 0..3
  cap: string;
}
const FRAMES: Frame[] = [
  { stage: 0, vt: 0, cap: "Poll the workload to quiescence — two turns are admitted and submitted to the engine; their futures park (Pending)." },
  { stage: 1, vt: 0, cap: "Compare the next clock sleeper vs the next engine event. No sleeper is parked and the engine is ready now, so the engine wins." },
  { stage: 3, vt: 0, cap: "Step the engine: it forms the first batch. step_until is bounded by the next sleeper, so it can't overshoot." },
  { stage: 4, vt: 1, cap: "Route the emitted events: waiters wake and fire on_admit then the first on_token — that first token is TTFT." },
  { stage: 1, vt: 1, cap: "Back to Compare: the next engine event is still the earliest, so the engine wins again." },
  { stage: 3, vt: 2, cap: "Step: a decode tick produces the next tokens." },
  { stage: 4, vt: 3, cap: "Route the terminals: on_usage then on_terminal fire; both request futures resolve." },
  { stage: 0, vt: 3, cap: "Poll once more: StopChecker's bound is met, the workload future is Ready, and the pump exits." },
];
const STAGE_NAMES = ["Poll", "Compare", "Advance", "Step", "Route"];
const TICKS = [
  { x: 70, ms: "0" },
  { x: 210, ms: "1.8" },
  { x: 300, ms: "2.0" },
  { x: 520, ms: "22" },
];

function LoopPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [i, setI] = useCanvasState<number>("dyn.loopFrame", 0);
  const f = FRAMES[Math.min(i, FRAMES.length - 1)];
  const n = t.fill.tertiary, s = t.stroke.primary, l = t.stroke.secondary, a = t.accent.primary;
  const maint = level === "maintainer";

  // stage node layout — top row L→R (Poll, Compare, Advance), bottom row R→L (Step, Route)
  const nodes = [
    { x: 24, y: 24 },   // Poll
    { x: 190, y: 24 },  // Compare
    { x: 356, y: 24 },  // Advance
    { x: 300, y: 156 }, // Step
    { x: 70, y: 156 },  // Route
  ];
  const W = 130, H = 46;
  const subFor = (idx: number) =>
    !maint ? undefined : ["LocalSet", "clock ≤ src", "advance_to", "step_until", "wake waiters"][idx];

  // directed cycle edges Poll→Compare→Advance→Step→Route→Poll
  const edges = [
    "M154,47 L190,47",
    "M320,47 L356,47",
    "M421,70 C455,110 445,150 430,156",
    "M300,179 L200,179",
    "M70,179 C34,150 40,90 78,52",
  ];

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The whole run is one loop. Step through it — the highlighted stage is what's executing, the clock below shows virtual time jumping only when the loop advances it.
      </Text>

      <Row gap={8} align="center" wrap>
        <Button variant="secondary" disabled={i === 0} onClick={() => setI((v) => Math.max(0, v - 1))}>Back</Button>
        <Button variant="primary" disabled={i >= FRAMES.length - 1} onClick={() => setI((v) => Math.min(FRAMES.length - 1, v + 1))}>Step</Button>
        <Button variant="ghost" disabled={i === 0} onClick={() => setI(0)}>Reset</Button>
        <Spacer />
        <Pill size="sm">{`${i + 1} / ${FRAMES.length}`}</Pill>
      </Row>

      <svg viewBox="0 0 500 220" style={SVG}>
        <defs><Arrow id="lo" color={l} /></defs>
        {edges.map((d, idx) => {
          const incoming = (f.stage + 4) % 5 === idx; // edge whose target is active
          return (
            <g key={idx}>
              <Edge d={d} color={incoming ? a : l} id="lo" moving={incoming} dur={1.4} />
              {incoming && <Particles d={d} color={a} n={1} dur={1.4} />}
            </g>
          );
        })}
        {nodes.map((pos, idx) => {
          const active = idx === f.stage;
          return (
            <g key={idx}>
              {active && (
                <circle cx={pos.x + W / 2} cy={pos.y + H / 2} fill={a}
                  style={{ animation: "ds-breathe 1.6s ease-in-out infinite" }} />
              )}
              <Node t={t} x={pos.x} y={pos.y} w={W} h={H}
                fill={active ? a : n} stroke={active ? a : s} accent={active ? a : undefined}
                title={STAGE_NAMES[idx]} sub={subFor(idx)} dim={!active && idx !== f.stage}
                titleFill={active ? t.text.onAccent : undefined} />
            </g>
          );
        })}
        {atLeast(level, "developer") && (
          <text x={250} y={116} textAnchor="middle" fontSize={9} fill={t.text.secondary}>clock wins ties · overshoot rejected</text>
        )}
      </svg>

      <Callout tone={f.stage === 4 ? "success" : "info"} title={`${STAGE_NAMES[f.stage]}${maint ? "" : ""}`}>{f.cap}</Callout>

      {/* synced virtual-time bar */}
      <Card>
        <CardHeader trailing={<Pill size="sm">SimClock</Pill>}>Virtual time</CardHeader>
        <CardBody>
          <svg viewBox="0 0 560 64" style={SVG}>
            <line x1={40} y1={34} x2={520} y2={34} stroke={l} strokeWidth={2} />
            {TICKS.map((tk, idx) => {
              const on = idx === f.vt;
              return (
                <g key={tk.ms}>
                  <line x1={tk.x} y1={27} x2={tk.x} y2={41} stroke={on ? a : l} strokeWidth={on ? 2 : 1.4} />
                  <text x={tk.x} y={56} textAnchor="middle" fontSize={9} fill={on ? t.text.primary : t.text.tertiary}>{tk.ms}{idx === TICKS.length - 1 ? " ms" : ""}</text>
                </g>
              );
            })}
            <circle cx={TICKS[f.vt].x} cy={34} r={6} fill={a} />
          </svg>
        </CardBody>
      </Card>
    </Stack>
  );
}

// ═══ PAGE 4 · Request → tokens → engine ══════════════════════════════════════

function DispatchPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const n = t.fill.tertiary, s = t.stroke.primary, l = t.stroke.secondary;
  const a = t.accent.primary;
  const maint = level === "maintainer";

  const branches = [
    { y: 20, label: "raw_token_ids", sub: maint ? "resolve()" : "exact tokens", color: t.category.green, d: "M170,40 C210,40 196,98 226,98" },
    { y: 84, label: "trace_hash_ids", sub: maint ? "synthesize_tokens" : "trace blocks", color: t.category.purple, d: "M170,104 L226,104" },
    { y: 148, label: "text turn", sub: maint ? "tiktoken encode" : "encode text", color: t.category.blue, d: "M170,168 C210,168 196,110 226,110" },
  ];
  const out = "M382,104 L440,104";

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        A turn becomes an engine token array through a three-way priority: exact token ids win, then recorded trace hashes, then plain text. All three converge on <Text as="span" weight="semibold">dispatch_tokens</Text> — no HTTP body is ever built.
      </Text>
      <svg viewBox="0 0 520 200" style={SVG}>
        <defs><Arrow id="dp" color={l} /></defs>
        {branches.map((b, idx) => (
          <g key={b.label}>
            <Edge d={b.d} color={l} id="dp" dur={2.6} />
            <Node t={t} x={12} y={b.y} w={158} h={40} fill={n} stroke={s} accent={b.color} title={b.label} sub={b.sub} />
            <Particles d={b.d} color={b.color} n={1} dur={2.6} />
            {maint && <text x={20} y={b.y - 3} textAnchor="start" fontSize={8} fill={t.text.quaternary}>{idx + 1}</text>}
          </g>
        ))}
        <Edge d={out} color={l} id="dp" dur={2.2} />
        <Node t={t} x={226} y={80} w={156} h={48} fill={t.fill.secondary} stroke={s} accent={a}
          title="dispatch_tokens" sub="no HTTP body" micro={maint ? "DirectRequest → submit" : undefined} />
        <Node t={t} x={440} y={80} w={68} h={48} fill={n} stroke={s} accent={t.category.purple} title="engine" />
        <Particles d={out} color={a} n={3} dur={2.2} shape="sq" size={8} />
      </svg>

      <Divider />
      <H3>What the run observes</H3>
      <Text tone="secondary" size="small">
        As the engine replies, each request emits the same callbacks a real HTTP request would — feeding the metrics accumulator.
      </Text>
      <ObserverStrip level={level} />
    </Stack>
  );
}

function ObserverStrip({ level }: { level: Level }) {
  const t = useHostTheme();
  const n = t.fill.tertiary, s = t.stroke.primary, l = t.stroke.secondary;
  const maint = level === "maintainer";
  const steps = maint
    ? ["on_arrival", "on_admit", "on_token", "on_usage", "on_terminal"]
    : ["arrival", "admit", "token", "usage", "done"];
  const w = 92, gap = 14, x0 = 16, y = 20;
  return (
    <svg viewBox={`0 0 ${x0 * 2 + steps.length * (w + gap)} 90`} style={SVG}>
      <defs><Arrow id="ob" color={l} /></defs>
      {steps.map((label, idx) => {
        const x = x0 + idx * (w + gap);
        const path = `M${x + w},${y + 24} L${x + w + gap},${y + 24}`;
        const first = idx === 0, last = idx === steps.length - 1;
        return (
          <g key={label}>
            {idx < steps.length - 1 && <Edge d={path} color={l} id="ob" dur={2} />}
            <Node t={t} x={x} y={y} w={w} h={46} fill={n} stroke={s}
              accent={first ? t.category.blue : last ? t.accent.primary : undefined} title={label} />
            {idx < steps.length - 1 && <Particles d={path} color={t.accent.primary} n={1} dur={2} />}
          </g>
        );
      })}
      {maint && (
        <text x={x0 + (steps.length * (w + gap)) / 2} y={82} textAnchor="middle" fontSize={8} fill={t.text.quaternary}>
          on_arrival from ScheduledRuntime · the rest from DynosimSink
        </text>
      )}
    </svg>
  );
}

// ═══ PAGE 5 · Parity gate ════════════════════════════════════════════════════

function ParityPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const n = t.fill.tertiary, s = t.stroke.primary, l = t.stroke.secondary;
  const ok = t.category.green, ai = t.accent.primary, dy = t.category.purple, bad = t.category.red;
  const maint = level === "maintainer";

  const rows = maint ? 8 : 6;
  const rh = 11;
  const topA = 40, topB = 150;
  const scanH = topB + rows * rh - topA + 6;
  const labels = ["ttft", "itl", "e2e", "throughput", "sessions", "tokens", "prefill_s", "gpu_hours"];

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        AIPerf accumulates its own summary from the observer stream; Dynamo produces its own from the engine. The run is rejected unless the two serialize to identical bytes.
      </Text>
      <svg viewBox="0 0 540 230" style={SVG}>
        <Node t={t} x={12} y={topA - 26} w={96} h={34} fill={n} stroke={s} accent={ai}
          title="AIPerf" sub={maint ? "Collector" : "observers"} />
        <Node t={t} x={12} y={topB - 26} w={96} h={34} fill={n} stroke={s} accent={dy}
          title="Dynamo" sub={maint ? "take_report_at" : "engine"} />

        {Array.from({ length: rows }).map((_, idx) => {
          const yA = topA + idx * rh, yB = topB + idx * rh;
          const w = 60 + (idx % 4) * 16;
          const delay = `${(idx * 0.14).toFixed(2)}s`;
          return (
            <g key={idx}>
              <rect x={120} y={yA} width={w} height={rh - 2} rx={2} fill={ai} opacity={0.55}
                style={{ animation: `ds-soft 2.4s ease-in-out ${delay} infinite` }} />
              <rect x={120} y={yB} width={w} height={rh - 2} rx={2} fill={dy} opacity={0.55}
                style={{ animation: `ds-soft 2.4s ease-in-out ${delay} infinite` }} />
              {maint && idx < labels.length && (
                <text x={114} y={yA + rh / 2 + 3} textAnchor="end" fontSize={7.5} fill={t.text.quaternary}>{labels[idx]}</text>
              )}
            </g>
          );
        })}

        {/* comparator column with a single scanning line */}
        <rect x={300} y={topA - 6} width={20} height={scanH} rx={3} fill={t.fill.secondary} stroke={l} />
        <line x1={310} y1={topA - 4} x2={310} y2={topA + 12} stroke={ok} strokeWidth={2.5}
          style={{ ["--sh" as string]: `${scanH - 20}px`, animation: "ds-scan 2.6s ease-in-out infinite alternate" }} />

        {/* outcome */}
        <rect x={344} y={topA + 30} width={maint ? 184 : 150} height={maint ? 74 : 52} rx={10}
          fill={t.fill.secondary} stroke={ok} strokeWidth={1.6} style={{ animation: "ds-soft 2.4s ease-in-out infinite" }} />
        <text x={maint ? 436 : 419} y={topA + 52} textAnchor="middle" fontSize={11} fontWeight={700} fill={t.text.primary}>
          {maint ? "canonical_shared_metric_bytes" : "byte-equal"}
        </text>
        <text x={maint ? 436 : 419} y={topA + 68} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>
          {maint ? "finish_shared_metrics → verify_parity" : "74 fields (+3 goodput)"}
        </text>
        {maint && (
          <>
            <rect x={356} y={topA + 76} width={70} height={18} rx={3} fill={n} stroke={s} />
            <text x={391} y={topA + 88} textAnchor="middle" fontSize={8} fill={t.text.secondary}>69 own</text>
            <rect x={434} y={topA + 76} width={80} height={18} rx={3} fill={n} stroke={s} />
            <text x={474} y={topA + 88} textAnchor="middle" fontSize={8} fill={t.text.secondary}>5 from engine</text>
          </>
        )}
      </svg>
      {atLeast(level, "developer") && (
        <Callout tone="warning" title="Mismatch → the run bails">
          Any differing field fails <Text as="span" weight="semibold">finish_shared_metrics</Text> in the library and again at <Text as="span" weight="semibold">verify_parity</Text> in <Text as="span" weight="semibold">offline_execution</Text>. Per-request rows are excluded from the compare; goodput adds 3 fields when an SLA is set.
        </Callout>
      )}
    </Stack>
  );
}

// ═══ reusable segmented control ══════════════════════════════════════════════

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

// ═══ PAGE · System architecture (interactive layered flowchart) ══════════════

type ModeId = "http" | "offline" | "online";

// per-mode active-path facts (concrete impls + file:line for maintainer)
const ARCH: Record<ModeId, {
  label: string; lane: "http" | "dyn"; clock: string; clockCite: string;
  driver: string; driverCite: string; tag: string;
}> = {
  http: { label: "HTTP online", lane: "http", clock: "RealClock", clockCite: "run.rs", driver: "tokio LocalSet reactor", driverCite: "execute.rs", tag: "real HTTP to a real server, wall clock" },
  offline: { label: "dynosim offline", lane: "dyn", clock: "SimClock", clockCite: "dynosim.rs:2419", driver: "drive_sim_with_source", driverCite: "graph/runtime.rs:213", tag: "virtual clock, in-process engine, deterministic" },
  online: { label: "dynosim online", lane: "dyn", clock: "RealClock", clockCite: "dynosim.rs:2489", driver: "drive_real_with_source", driverCite: "graph/runtime.rs:419", tag: "wall clock, in-process engine" },
};

// a dashed, tagged trait-seam box
function SeamBox({
  x, y, w, h, t, tag, main, sub, k,
}: { x: number; y: number; w: number; h: number; t: Theme; tag: string; main: string; sub?: string; k?: string }) {
  const cx = x + w / 2;
  return (
    <g key={k}>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={t.accent.control} stroke={t.accent.primary} strokeWidth={1.6} strokeDasharray="5 4" />
      <text x={x + 8} y={y + 13} fontSize={7.5} fontWeight={700} fill={t.accent.primary} style={{ letterSpacing: 0.6 }}>{tag}</text>
      <text x={cx} y={y + h / 2 + (sub ? 2 : 5)} textAnchor="middle" fontSize={12} fontWeight={700} fill={t.text.onAccent}>{main}</text>
      {sub && <text x={cx} y={y + h / 2 + 15} textAnchor="middle" fontSize={8.5} fill={t.text.onAccent} opacity={0.85}>{sub}</text>}
    </g>
  );
}

function SeamsPage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [mode, setMode] = useCanvasState<ModeId>("dyn.mode", "offline");
  const a = ARCH[mode];
  const maint = level === "maintainer";
  const dev = atLeast(level, "developer");
  const l = t.stroke.secondary, s = t.stroke.primary, n = t.fill.tertiary;
  const acc = t.accent.primary, g = t.category.green, blue = t.category.blue, pur = t.category.purple;
  const httpOn = a.lane === "http", dynOn = a.lane === "dyn";

  return (
    <Stack gap={12}>
      <Text tone="secondary">
        The real architecture: everything above the <Text as="span" weight="semibold">RequestSink</Text> seam is shared by every mode. Below it the path forks. Pick a mode — the active branch lights up and flows.
      </Text>
      <SegControl<ModeId> value={mode} set={setMode} opts={[
        { id: "http", label: "HTTP online" },
        { id: "offline", label: "dynosim offline" },
        { id: "online", label: "dynosim online" },
      ]} />

      <svg viewBox="0 0 640 500" style={SVG}>
        <defs>
          <Arrow id="aA" color={acc} /><Arrow id="aL" color={l} />
          <Arrow id="aB" color={blue} /><Arrow id="aP" color={pur} />
        </defs>

        {/* ── shared spine (always active) — particles ride only the gaps
            between the stacked nodes, never across a node face ── */}
        {["M320,46 L320,62", "M320,100 L320,116", "M320,154 L320,170", "M320,216 L320,232"].map((d, i) => (
          <g key={`sp${i}`}>
            <Edge d={d} color={acc} id="aA" dur={2.4} />
            <Particles d={d} color={acc} n={1} dur={1.3} />
          </g>
        ))}

        <Node t={t} x={220} y={8} w={200} h={38} fill={n} stroke={s} title="aiperf profile"
          sub={maint ? "load.rs / profile.rs" : "native entry point"} />
        <Node t={t} x={220} y={62} w={200} h={38} fill={n} stroke={s} title="aiperf --execute"
          sub={maint ? "RunnerApplication::handle_v2" : "re-exec child"} />
        <Node t={t} x={220} y={116} w={200} h={38} fill={n} stroke={s} title="Workload"
          sub={maint ? "RequestRate · UserCentric · Graph" : "arrival pattern"} />
        <Node t={t} x={220} y={170} w={200} h={46} fill={n} stroke={s} accent={g} title="ScheduledRuntime"
          sub="SlotPool · StopChecker" micro={maint ? "+ ObserverTee (Collector+Native)" : "+ observers"} />

        {/* the RequestSink trait seam */}
        <SeamBox t={t} x={235} y={232} w={170} h={40} tag="TRAIT SEAM" main="RequestSink<HttpRequest>" />

        {/* ── Clock trait seam (left rail, injected) ── */}
        <SeamBox t={t} x={18} y={170} w={166} h={46} tag="TRAIT SEAM" main={`Clock → ${a.clock}`}
          sub={mode === "offline" ? "virtual ns" : "wall ns"} k={`clk-${mode}`} />
        <path d="M184,193 L220,193" fill="none" stroke={acc} strokeWidth={1.5} markerEnd="url(#aA)" strokeDasharray="4 4"
          style={{ animation: "ds-flow 2s linear infinite" }} />
        <text x={101} y={230} textAnchor="middle" fontSize={8} fill={t.text.tertiary}>injected into runtime, sink & driver</text>
        {/* driver chip (selected alongside the clock) */}
        <g key={`drv-${mode}`} style={POP}>
          <rect x={18} y={240} width={166} height={30} rx={7} fill={n} stroke={s} />
          <text x={101} y={253} textAnchor="middle" fontSize={7.5} fontWeight={600} fill={t.text.tertiary}>DRIVER</text>
          <text x={101} y={264} textAnchor="middle" fontSize={9} fill={t.text.primary}>{a.driver}</text>
        </g>

        {/* ── fork below the sink seam ── */}
        <Edge d="M300,272 C210,296 175,296 152,300" color={httpOn ? blue : l} id={httpOn ? "aB" : "aL"} moving={httpOn} dur={2} />
        <Edge d="M340,272 C440,296 478,296 500,300" color={dynOn ? pur : l} id={dynOn ? "aP" : "aL"} moving={dynOn} dur={2} />
        {httpOn && <Particles d="M300,272 C210,296 175,296 152,300" color={blue} n={2} dur={2} />}
        {dynOn && <Particles d="M340,272 C440,296 478,296 500,300" color={pur} n={2} dur={2} />}

        {/* left lane · HTTP */}
        <g opacity={httpOn ? 1 : 0.4}>
          <Node t={t} x={62} y={300} w={180} h={42} fill={n} stroke={s} accent={httpOn ? blue : undefined}
            title="TransportSink" sub={maint ? "impl RequestSink" : "serialize → bytes"} />
          <Edge d="M152,342 L152,358" color={httpOn ? blue : l} id={httpOn ? "aB" : "aL"} moving={httpOn} dur={1.6} />
          <Node t={t} x={62} y={358} w={180} h={44} fill={n} stroke={s} accent={httpOn ? blue : undefined}
            title="HttpTransport" sub="Hyper 1.x + SSE" micro={maint ? "http.rs:524" : undefined} />
          <Edge d="M152,402 L152,420" color={httpOn ? blue : l} id={httpOn ? "aB" : "aL"} moving={httpOn} dur={1.6} />
          <Node t={t} x={62} y={420} w={180} h={34} fill={t.fill.secondary} stroke={s} accent={httpOn ? blue : undefined}
            title="real server" sub="socket / URL" />
        </g>

        {/* right lane · dynosim */}
        <g opacity={dynOn ? 1 : 0.4}>
          <Node t={t} x={410} y={300} w={180} h={42} fill={n} stroke={s} accent={dynOn ? pur : undefined}
            title="DynosimSink" sub={maint ? "impl RequestSink" : "→ token array"} />
          <Edge d="M500,342 L500,358" color={dynOn ? pur : l} id={dynOn ? "aP" : "aL"} moving={dynOn} dur={1.6} />
          <Node t={t} x={410} y={358} w={180} h={44} fill={n} stroke={s} accent={dynOn ? pur : undefined}
            title="EngineHost" sub="bounds step_until" micro={maint ? "SimEventSource" : undefined} />
          <Edge d="M500,402 L500,420" color={dynOn ? pur : l} id={dynOn ? "aP" : "aL"} moving={dynOn} dur={1.6} />
          <Node t={t} x={410} y={420} w={180} h={34} fill={t.fill.secondary} stroke={s} accent={dynOn ? pur : undefined}
            title="SteppableReplay" sub="no sockets · in-process" />
        </g>

        {/* lane captions */}
        <text x={152} y={472} textAnchor="middle" fontSize={9} fontWeight={600} fill={httpOn ? blue : t.text.quaternary}>HTTP lane</text>
        <text x={500} y={472} textAnchor="middle" fontSize={9} fontWeight={600} fill={dynOn ? pur : t.text.quaternary}>dynosim lane</text>
        <text x={320} y={290} textAnchor="middle" fontSize={8} fill={t.text.quaternary}>fork by mode</text>
      </svg>

      <Callout tone={mode === "http" ? "neutral" : "info"} title={a.label}>
        {a.tag}. Above the seam nothing changes; the driver is <Text as="span" weight="semibold">{a.driver}</Text> and the clock is <Text as="span" weight="semibold">{a.clock}</Text>.
      </Callout>
      {dev && (
        <Callout tone="warning" title="The fork is chosen at composition — not is_virtual()">
          The frozen <Text as="span" weight="semibold">AIPerfRegistry</Text> registers independent transports (<Text as="span" weight="semibold">http</Text> / <Text as="span" weight="semibold">dynosim_offline</Text> / <Text as="span" weight="semibold">dynosim_online</Text>) and workloads; there is no transport×workload pair map. Within dynosim, <Text as="span" weight="semibold">run_paced_offline</Text> vs <Text as="span" weight="semibold">run_paced_online</Text> pick the clock+driver from the transport ID. <Text as="span" weight="semibold">clock.is_virtual()</Text> is only read for measurement, never to branch the mode.
        </Callout>
      )}
    </Stack>
  );
}

// ═══ PAGE · Engine internals — topology builder (interactive) ════════════════

type TopoId = "single" | "aggregated" | "disaggregated";
type RouterId = "round_robin" | "kv";

function Worker({ x, y, t, label, color, k, delay }: { x: number; y: number; t: Theme; label: string; color: string; k: string; delay: number }) {
  return (
    <g key={k} style={{ ...POP, animationDelay: `${delay}s` }}>
      <rect x={x} y={y} width={40} height={34} rx={6} fill={t.fill.tertiary} stroke={color} strokeWidth={1.5} />
      <rect x={x + 6} y={y + 24} width={28} height={4} rx={2} fill={color} opacity={0.6} />
      <text x={x + 20} y={y + 15} textAnchor="middle" fontSize={9} fontWeight={600} fill={t.text.primary}>{label}</text>
    </g>
  );
}

function EnginePage({ level }: { level: Level }) {
  const t = useHostTheme();
  const [topo, setTopo] = useCanvasState<TopoId>("dyn.topo", "aggregated");
  const [router, setRouter] = useCanvasState<RouterId>("dyn.router", "kv");
  const maint = level === "maintainer";
  const s = t.stroke.primary, l = t.stroke.secondary;
  const pf = t.category.purple, dc = t.accent.primary, rt = t.category.orange;
  const kv = router === "kv";

  const engineName: Record<TopoId, string> = { single: "SteppableEngine", aggregated: "SteppableAgg", disaggregated: "SteppableDisagg" };
  const engineSub: Record<TopoId, string> = { single: "1 worker · no router", aggregated: "N workers", disaggregated: "prefill + decode pools" };

  return (
    <Stack gap={14}>
      <Text tone="secondary">
        The offline engine (Dynamo's <Text as="span" weight="semibold">SteppableReplay</Text>) is built in one of three shapes. Change the shape and the router and watch workers and routing rewire.
      </Text>
      <Row gap={16} align="center" wrap>
        <Stack gap={4}>
          <Text size="small" tone="tertiary">topology</Text>
          <SegControl<TopoId> value={topo} set={setTopo} size="sm" opts={[
            { id: "single", label: "single" }, { id: "aggregated", label: "aggregated" }, { id: "disaggregated", label: "disaggregated" },
          ]} />
        </Stack>
        <Stack gap={4}>
          <Text size="small" tone="tertiary">router_mode</Text>
          <SegControl<RouterId> value={router} set={setRouter} size="sm" opts={[
            { id: "round_robin", label: "round robin" }, { id: "kv", label: "kv" },
          ]} />
        </Stack>
      </Row>

      <svg viewBox="0 0 560 250" style={SVG}>
        <defs><Arrow id="en" color={l} /><Arrow id="er" color={rt} /></defs>

        {/* engine header */}
        <g key={`hdr-${topo}`} style={POP}>
          <rect x={190} y={12} width={180} height={46} rx={9} fill={t.fill.secondary} stroke={dc} strokeWidth={1.6} />
          <text x={280} y={31} textAnchor="middle" fontSize={12.5} fontWeight={700} fill={t.text.primary}>{engineName[topo]}</text>
          <text x={280} y={46} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>{engineSub[topo]}{maint ? " · build_native()" : ""}</text>
        </g>

        {/* SINGLE */}
        {topo === "single" && (
          <>
            <path d="M280,58 L280,96" fill="none" stroke={l} strokeWidth={1.4} markerEnd="url(#en)" />
            <Worker x={260} y={98} t={t} label="w0" color={dc} k="s0" delay={0} />
            {atLeast(level, "developer") && (
              <text x={280} y={164} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>one ReplayWorkerCore · router_mode ignored</text>
            )}
            {maint && (
              <text x={280} y={182} textAnchor="middle" fontSize={8} fill={t.text.quaternary}>falls back to SteppableAgg(1, RoundRobin) when clock events force step_until</text>
            )}
          </>
        )}

        {/* AGGREGATED */}
        {topo === "aggregated" && (
          <>
            {kv && (
              <g key="agg-router" style={POP}>
                <rect x={225} y={72} width={110} height={30} rx={7} fill={t.fill.tertiary} stroke={rt} strokeWidth={1.5} />
                <text x={280} y={91} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={t.text.primary}>{maint ? "OfflineReplayRouter" : "KV router"}</text>
              </g>
            )}
            {!kv && <text x={280} y={92} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>round-robin index · router: None</text>}
            {[0, 1, 2].map((i) => {
              const x = 176 + i * 76;
              return (
                <g key={`aw-wrap-${i}`}>
                  <path d={`M280,${kv ? 102 : 96} L${x + 20},128`} fill="none" stroke={kv ? rt : l} strokeWidth={1.3} markerEnd={`url(#${kv ? "er" : "en"})`} />
                  <Worker x={x} y={130} t={t} label={`w${i}`} color={dc} k={`aw-${router}-${i}`} delay={i * 0.07} />
                </g>
              );
            })}
            <text x={280} y={196} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>
              {maint ? "workers × OfflineWorkerState in one EngineComponent(Aggregated)" : "N workers, one aggregated component"}
            </text>
          </>
        )}

        {/* DISAGGREGATED */}
        {topo === "disaggregated" && (
          <>
            {/* prefill pool */}
            <text x={150} y={78} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={pf}>prefill{maint ? " · Hidden" : ""}</text>
            {kv && (
              <g key="pf-router" style={POP}>
                <rect x={104} y={86} width={92} height={26} rx={6} fill={t.fill.tertiary} stroke={rt} strokeWidth={1.4} />
                <text x={150} y={103} textAnchor="middle" fontSize={8.5} fontWeight={600} fill={t.text.primary}>prefill_router</text>
              </g>
            )}
            {[0, 1].map((i) => (
              <g key={`pf-wrap-${i}`}>
                <path d={`M150,${kv ? 112 : 84} L${118 + i * 64 + 20},130`} fill="none" stroke={kv ? rt : l} strokeWidth={1.3} markerEnd={`url(#${kv ? "er" : "en"})`} />
                <Worker x={118 + i * 64} y={132} t={t} label={`p${i}`} color={pf} k={`pf-${router}-${i}`} delay={i * 0.07} />
              </g>
            ))}

            {/* handoff */}
            <path d="M254,149 L306,149" fill="none" stroke={t.text.tertiary} strokeWidth={1.6} markerEnd="url(#en)" strokeDasharray="3 3" />
            <text x={280} y={143} textAnchor="middle" fontSize={7.5} fill={t.text.quaternary}>handoff</text>

            {/* decode pool */}
            <text x={410} y={78} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={dc}>decode{maint ? " · Visible" : ""}</text>
            {kv && (
              <g key="dc-router" style={POP}>
                <rect x={364} y={86} width={92} height={26} rx={6} fill={t.fill.tertiary} stroke={rt} strokeWidth={1.4} />
                <text x={410} y={103} textAnchor="middle" fontSize={8.5} fontWeight={600} fill={t.text.primary}>decode_router</text>
              </g>
            )}
            {[0, 1].map((i) => (
              <g key={`dc-wrap-${i}`}>
                <path d={`M410,${kv ? 112 : 84} L${378 + i * 64 + 20},130`} fill="none" stroke={kv ? rt : l} strokeWidth={1.3} markerEnd={`url(#${kv ? "er" : "en"})`} />
                <Worker x={378 + i * 64} y={132} t={t} label={`d${i}`} color={dc} k={`dc-${router}-${i}`} delay={i * 0.07} />
              </g>
            ))}
            <text x={280} y={196} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>
              {maint ? "separate MockEngineArgs (WorkerType::Prefill / Decode) + per-pool routers" : "two independent pools with handoff"}
            </text>
          </>
        )}
      </svg>

      {atLeast(level, "developer") && (
        <Grid columns={2} gap={12}>
          <Callout tone="info" title={kv ? "KV routing" : "Round-robin"}>
            {kv
              ? "Each pool gets an OfflineReplayRouter that places requests by KV-cache affinity."
              : "router is None — a plain round-robin index picks the next worker."}
          </Callout>
          <Callout tone="neutral" title="Stepping seam">
            {maint
              ? "EngineHost::step calls step_until(next_event) so the engine can't overshoot the virtual clock."
              : "Whatever the shape, EngineHost steps it one bounded slice at a time."}
          </Callout>
        </Grid>
      )}
    </Stack>
  );
}

// ── source jump (maintainer) ─────────────────────────────────────────────────

const SRC: Record<PageId, { l: string; p: string }[]> = {
  overview: [
    { l: "profile.rs", p: "rust/cli/src/profile.rs" },
    { l: "dynosim.rs", p: "rust/aiperf/src/dynosim.rs" },
  ],
  launch: [
    { l: "application.rs", p: "rust/aiperf/src/runner_protocol/application.rs" },
    { l: "load.rs", p: "rust/cli/src/load.rs" },
    { l: "execute.rs", p: "rust/cli/src/execute.rs" },
    { l: "execute_mode.rs", p: "rust/cli/src/execute_mode.rs" },
  ],
  seams: [
    { l: "run.rs", p: "rust/aiperf/src/run.rs" },
    { l: "dynosim.rs", p: "rust/aiperf/src/dynosim.rs" },
    { l: "runtime.rs", p: "rust/aiperf/src/graph/runtime.rs" },
    { l: "registry.rs", p: "rust/aiperf/src/runner_protocol/registry.rs" },
  ],
  loop: [
    { l: "runtime.rs", p: "rust/aiperf/src/graph/runtime.rs" },
    { l: "sim_clock.rs", p: "rust/aiperf/src/clock/sim_clock.rs" },
  ],
  dispatch: [{ l: "dynosim.rs", p: "rust/aiperf/src/dynosim.rs" }],
  parity: [
    { l: "dynosim.rs", p: "rust/aiperf/src/dynosim.rs" },
    { l: "offline_execution.rs", p: "rust/aiperf/src/runner_protocol/offline_execution.rs" },
  ],
  engine: [{ l: "offline_execution.rs", p: "rust/aiperf/src/runner_protocol/offline_execution.rs" }],
};

// ── pages registry ───────────────────────────────────────────────────────────

type El = ReturnType<typeof H1>;
const PAGES: { id: PageId; label: string; title: string; render: (level: Level) => El }[] = [
  { id: "overview", label: "Overview", title: "How it fits together", render: (lv) => <OverviewPage level={lv} /> },
  { id: "launch", label: "Launch", title: "Launch & preflight", render: (lv) => <LaunchPage level={lv} /> },
  { id: "seams", label: "Architecture", title: "System architecture — the two seams", render: (lv) => <SeamsPage level={lv} /> },
  { id: "loop", label: "Loop", title: "The simulation loop", render: (lv) => <LoopPage level={lv} /> },
  { id: "dispatch", label: "Dispatch", title: "Request → tokens → engine", render: (lv) => <DispatchPage level={lv} /> },
  { id: "parity", label: "Parity", title: "The verification gate", render: (lv) => <ParityPage level={lv} /> },
  { id: "engine", label: "Engine", title: "Engine internals — topology builder", render: (lv) => <EnginePage level={lv} /> },
];

// ── detail toggle ────────────────────────────────────────────────────────────

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

// ── app ──────────────────────────────────────────────────────────────────────

export default function DynosimOfflineCanvas() {
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
          <H1>Dynosim Offline</H1>
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
          <CardHeader trailing={<Pill size="sm">{`step ${idx + 1} of ${PAGES.length}`}</Pill>}>{current.title}</CardHeader>
          <CardBody>
            <Stack gap={16}>
              {current.render(level)}

              {level === "maintainer" && (
                <>
                  <Divider />
                  <Row gap={6} align="center" wrap>
                    <Text size="small" tone="tertiary">src</Text>
                    {SRC[current.id].map((s) => (
                      <div key={s.p} style={{ display: "contents" }}>
                        <Button variant="ghost" onClick={() => dispatch({ type: "openFile", path: s.p })}>{s.l}</Button>
                      </div>
                    ))}
                  </Row>
                </>
              )}
            </Stack>
          </CardBody>
        </Card>

        {/* prev / next */}
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
