import {
  H1,
  H2,
  H3,
  Text,
  Code,
  Card,
  CardHeader,
  CardBody,
  Callout,
  Pill,
  Button,
  Row,
  Stack,
  Grid,
  Stat,
  Spacer,
  Divider,
  useCanvasState,
  useHostTheme,
  useCanvasAction,
} from "cursor/canvas";

// ═══════════════════════════════════════════════════════════════════════════
// Segment Pools & Body Plans — an interactive explainer for the AIPerf
// dataset → segment store → body plan → wire pipeline.
//
// Grounded in rust/aiperf/src/dataset/{segment.rs, model.rs, dataset.rs,
// request.rs, loader/mod.rs, compose.rs}, rust/aiperf/src/body_plan.rs, and
// rust/aiperf/src/graph/recorded/trie/*. Six pages, two live simulators.
// ═══════════════════════════════════════════════════════════════════════════

type Theme = ReturnType<typeof useHostTheme>;
type PageId = "overview" | "pool" | "payloads" | "bodyplan" | "prefix" | "dispatch";

const SVG = { width: "100%", height: "auto", display: "block" } as const;

const CSS = `
@keyframes sp-flow { to { stroke-dashoffset: -36; } }
@keyframes sp-pop  { 0% { opacity: 0; transform: scale(.8) } 60% { opacity: 1; transform: scale(1.06) } 100% { transform: scale(1) } }
@keyframes sp-glow { 0%,100% { opacity: .35 } 50% { opacity: .9 } }
@keyframes sp-flash { 0% { opacity: .0 } 30% { opacity: .55 } 100% { opacity: 0 } }
`;

// FNV-1a — an *illustrative* stand-in for blake3 so handles show a stable,
// content+parent-derived id. Not the real hash; labelled as such in the UI.
function fnv(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0").slice(0, 6);
}

// ── shared SVG atoms ─────────────────────────────────────────────────────────

function Arrow({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill={color} />
    </marker>
  );
}

function Edge({
  d,
  color,
  id,
  moving = true,
  dur = 2.4,
  width = 1.5,
}: {
  d: string;
  color: string;
  id: string;
  moving?: boolean;
  dur?: number;
  width?: number;
}) {
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={width}
      markerEnd={`url(#${id})`}
      strokeDasharray="4 5"
      style={moving ? { animation: `sp-flow ${dur}s linear infinite` } : undefined}
    />
  );
}

function Dots({ d, color, n, dur, size = 3.4 }: { d: string; color: string; n: number; dur: number; size?: number }) {
  return (
    <>
      {Array.from({ length: n }).map((_, i) => (
        <circle key={i} r={size} fill={color}>
          <animateMotion dur={`${dur}s`} begin={`${(dur / n) * i}s`} repeatCount="indefinite" path={d} />
        </circle>
      ))}
    </>
  );
}

function Box({
  x,
  y,
  w,
  h,
  t,
  title,
  sub,
  micro,
  accent,
  dim,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  t: Theme;
  title: string;
  sub?: string;
  micro?: string;
  accent?: string;
  dim?: boolean;
}) {
  const cx = x + w / 2;
  const lines = (sub ? 1 : 0) + (micro ? 1 : 0);
  const ty = lines === 0 ? y + h / 2 : lines === 1 ? y + h / 2 - 5 : y + h / 2 - 10;
  return (
    <g opacity={dim ? 0.4 : 1}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill={t.bg.elevated}
        stroke={accent ?? t.stroke.primary}
        strokeWidth={accent ? 1.7 : 1}
      />
      {accent && <rect x={x} y={y} width={4} height={h} rx={2} fill={accent} />}
      <text x={cx} y={ty} textAnchor="middle" dominantBaseline="middle" fontSize={12.5} fontWeight={600} fill={t.text.primary}>
        {title}
      </text>
      {sub && (
        <text x={cx} y={ty + 15} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>
          {sub}
        </text>
      )}
      {micro && (
        <text x={cx} y={ty + 27} textAnchor="middle" fontSize={8.5} fill={t.text.quaternary}>
          {micro}
        </text>
      )}
    </g>
  );
}

// ═══ PAGE 1 · Overview — the whole pipeline ═══════════════════════════════════

function PageOverview({ t }: { t: Theme }) {
  const A = t.category.blue;
  const G = t.category.green;
  const P = t.category.purple;
  const Y = t.category.yellow;
  return (
    <Stack gap={16}>
      <div>
        <H2>The pipeline: rows in → wire bytes out</H2>
        <Text tone="secondary">
          Every request body AIPerf sends starts as a dataset row and ends as pre-spliced bytes. Two data structures own
          the middle: the <Text weight="semibold">SegmentPool</Text> (content-addressed, deduplicated storage) and the{" "}
          <Text weight="semibold">BodyPlan</Text> (a shape that says which handles fill which JSON fields). The design
          invariant is <Text weight="semibold">serialize content once, splice bytes forever</Text>.
        </Text>
      </div>

      <Card>
        <CardHeader trailing={<Pill size="sm">rust/aiperf/src</Pill>}>compile-time → freeze → dispatch</CardHeader>
        <CardBody style={{ padding: 12 }}>
          <svg viewBox="0 0 940 430" style={SVG}>
            <defs>
              <Arrow id="ov-a" color={A} />
              <Arrow id="ov-g" color={G} />
              <Arrow id="ov-p" color={P} />
              <Arrow id="ov-y" color={Y} />
            </defs>

            {/* band labels */}
            <text x={20} y={24} fontSize={10.5} fontWeight={700} fill={t.text.tertiary} letterSpacing="0.08em">
              BUILD (mutable)
            </text>
            <text x={470} y={24} fontSize={10.5} fontWeight={700} fill={t.text.tertiary} letterSpacing="0.08em">
              FREEZE
            </text>
            <text x={700} y={24} fontSize={10.5} fontWeight={700} fill={t.text.tertiary} letterSpacing="0.08em">
              DISPATCH (hot path)
            </text>
            <line x1={455} y1={34} x2={455} y2={410} stroke={t.stroke.tertiary} strokeDasharray="3 6" />
            <line x1={685} y1={34} x2={685} y2={410} stroke={t.stroke.tertiary} strokeDasharray="3 6" />

            <Box x={20} y={54} w={150} h={54} t={t} title="Dataset source" sub="JSON / CSV / HF / trace" micro="loader/mod.rs" accent={A} />
            <Box x={20} y={150} w={150} h={58} t={t} title="Composer.compose" sub="intern rows → pool" micro="loader/simple.rs" accent={A} />
            <Box x={20} y={252} w={150} h={58} t={t} title="apply_common_contexts" sub="system / user_context" micro="compose.rs rebase" accent={A} />

            <Box x={250} y={150} w={172} h={92} t={t} title="SegmentPool" sub="arena: Vec<Segment>" micro="ids: HashMap<Id,Handle>" accent={P} />

            <Box x={480} y={160} w={172} h={72} t={t} title="InMemorySegmentStore" sub="Box<[Segment]> (frozen)" micro="ids map dropped" accent={P} />

            <Box x={710} y={62} w={200} h={58} t={t} title="Dataset" sub="Arc<dyn SegmentStore>" micro="+ body_plans cache" accent={G} />
            <Box x={710} y={150} w={200} h={62} t={t} title="precompute_body_plans" sub="BodyPlan per static turn" micro="dataset.rs" accent={G} />
            <Box x={710} y={242} w={200} h={62} t={t} title="JsonBodyMaterializer" sub="splice handles → Bytes" micro="body_plan.rs" accent={Y} />
            <Box x={710} y={334} w={200} h={54} t={t} title="Transport" sub="HTTP / gRPC dispatch" micro="MaterializedRequest.body" accent={Y} />

            {/* build edges */}
            <Edge id="ov-a" color={A} d="M95,108 L95,150" />
            <Edge id="ov-a" color={A} d="M95,208 L95,252" />
            <Edge id="ov-a" color={A} d="M170,180 C205,185 215,193 250,196" />
            <Edge id="ov-a" color={A} d="M170,278 C210,270 220,235 250,222" />

            {/* freeze */}
            <Edge id="ov-p" color={P} d="M422,196 L480,196" dur={1.8} width={1.8} />
            <text x={451} y={186} textAnchor="middle" fontSize={9} fill={P}>
              .freeze()
            </text>

            {/* dispatch */}
            <Edge id="ov-g" color={G} d="M652,190 C675,170 685,120 710,104" />
            <Edge id="ov-g" color={G} d="M810,120 L810,150" />
            <Edge id="ov-y" color={Y} d="M810,212 L810,242" />
            <Edge id="ov-y" color={Y} d="M810,304 L810,334" />

            {/* store feeds materializer */}
            <Edge id="ov-p" color={P} d="M566,232 C600,300 640,290 706,278" dur={3} />
            <text x={628} y={318} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>
              store.get(handle) → wire bytes
            </text>

            <Dots d="M95,108 L95,150" color={A} n={2} dur={2.2} />
            <Dots d="M422,196 L480,196" color={P} n={3} dur={1.6} />
            <Dots d="M810,304 L810,334" color={Y} n={2} dur={1.6} />
          </svg>
        </CardBody>
      </Card>

      <Grid columns={4} gap={12}>
        <Stat value="6" label="payload domains" />
        <Stat value="u32" label="Handle = dense index" />
        <Stat value="[u8;32]" label="SegmentId = blake3" />
        <Stat value="0" label="content re-serializes on hot path" tone="success" />
      </Grid>

      <Callout tone="info" title="The two representations">
        A <Code>Handle(u32)</Code> is the public address — a dense arena index. A <Code>SegmentId([u8;32])</Code> is the
        blake3 content hash used only to deduplicate while the pool is mutable. When the pool freezes into an{" "}
        <Code>InMemorySegmentStore</Code>, the hash→handle map is thrown away; handles stay valid.
      </Callout>
    </Stack>
  );
}

// ═══ PAGE 2 · SegmentPool — interactive interning simulator ═══════════════════

type Step = {
  id: string;
  convo: 1 | 2;
  role: "system" | "user" | "assistant";
  content: string;
  parent?: string; // step id
};

const STEPS: Step[] = [
  { id: "c1s", convo: 1, role: "system", content: "You are a helpful assistant." },
  { id: "c1u", convo: 1, role: "user", content: "What is 2+2?", parent: "c1s" },
  { id: "c1a", convo: 1, role: "assistant", content: "4", parent: "c1u" },
  { id: "c2s", convo: 2, role: "system", content: "You are a helpful assistant." },
  { id: "c2u", convo: 2, role: "user", content: "What is 2+2?", parent: "c2s" },
  { id: "c2a", convo: 2, role: "assistant", content: "It equals four.", parent: "c2u" },
];

type Resolved = { handle: number; hash: string; deduped: boolean };

function simulate(upTo: number): {
  resolved: Record<string, Resolved>;
  arena: { handle: number; hash: string; role: string; content: string; parent: number | null }[];
  dedup: number;
} {
  const resolved: Record<string, Resolved> = {};
  const arena: { handle: number; hash: string; role: string; content: string; parent: number | null }[] = [];
  const ids = new Map<string, number>();
  let dedup = 0;
  for (let i = 0; i < upTo; i++) {
    const s = STEPS[i];
    const parentRes = s.parent ? resolved[s.parent] : undefined;
    const parentHash = parentRes ? parentRes.hash : "root";
    // key mirrors payload_id: child identity folds in the PARENT'S id, not its index
    const key = `message|${parentHash}|${s.role}|${s.content}`;
    const hash = fnv(key);
    if (ids.has(key)) {
      const handle = ids.get(key)!;
      resolved[s.id] = { handle, hash, deduped: true };
      dedup++;
    } else {
      const handle = arena.length;
      ids.set(key, handle);
      arena.push({ handle, hash, role: s.role, content: s.content, parent: parentRes ? parentRes.handle : null });
      resolved[s.id] = { handle, hash, deduped: false };
    }
  }
  return { resolved, arena, dedup };
}

function PagePool({ t }: { t: Theme }) {
  const [n, setN] = useCanvasState<number>("sp.pool.n", 0);
  const sim = simulate(n);
  const P = t.category.purple;
  const G = t.category.green;
  const roleColor = (r: string) => (r === "system" ? t.category.cyan : r === "user" ? t.category.blue : t.category.green);
  const bytesSaved = Object.values(sim.resolved).filter((r) => r.deduped).length * 42;

  return (
    <Stack gap={16}>
      <div>
        <H2>SegmentPool — content-addressed interning</H2>
        <Text tone="secondary">
          Step through interning two conversations that share a system prompt and first user turn. Because a child's id
          folds in its <Text weight="semibold">parent's content hash</Text>, identical prefixes collapse to the{" "}
          <Text weight="semibold">same handle</Text> — even across conversations, even in a different load order.
        </Text>
      </div>

      <Row gap={8} align="center" wrap>
        <Button variant="primary" onClick={() => setN((v) => Math.min(v + 1, STEPS.length))}>
          Intern next
        </Button>
        <Button variant="secondary" onClick={() => setN(STEPS.length)}>
          Run all
        </Button>
        <Button variant="ghost" onClick={() => setN(0)}>
          Reset
        </Button>
        <Spacer />
        <Pill size="sm">{n}/{STEPS.length} steps</Pill>
      </Row>

      <Grid columns={4} gap={12}>
        <Stat value={sim.arena.length} label="arena size (handles)" />
        <Stat value={sim.dedup} label="dedup hits" tone={sim.dedup > 0 ? "success" : undefined} />
        <Stat value={`~${bytesSaved}B`} label="content not re-stored" tone="success" />
        <Stat value={n} label="intern calls" />
      </Grid>

      <Grid columns="1fr 1fr" gap={14}>
        <Card>
          <CardHeader trailing={<Pill size="sm">compose()</Pill>}>Intern calls</CardHeader>
          <CardBody style={{ padding: 0 }}>
            <Stack gap={0}>
              {STEPS.map((s, i) => {
                const done = i < n;
                const r = sim.resolved[s.id];
                const active = i === n - 1;
                return (
                  <div
                    key={s.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: "9px 12px",
                      borderTop: i === 0 ? "none" : `1px solid ${t.stroke.tertiary}`,
                      opacity: done ? 1 : 0.4,
                      background: active ? t.fill.tertiary : "transparent",
                    }}
                  >
                    <span
                      style={{
                        width: 6,
                        height: 6,
                        borderRadius: 9999,
                        background: roleColor(s.role),
                        flex: "0 0 auto",
                      }}
                    />
                    <Text size="small" tone="tertiary" style={{ width: 26 }}>
                      C{s.convo}
                    </Text>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <Text size="small" truncate>
                        <Text as="span" weight="semibold">
                          {s.role}
                        </Text>
                        {"  "}
                        {s.content}
                      </Text>
                    </div>
                    {done && r && (
                      <Pill size="sm" active={!r.deduped}>
                        {r.deduped ? `dedup → H${r.handle}` : `H${r.handle}`}
                      </Pill>
                    )}
                  </div>
                );
              })}
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">InMemorySegmentStore on freeze</Pill>}>Arena (dense)</CardHeader>
          <CardBody style={{ padding: 12 }}>
            {sim.arena.length === 0 ? (
              <Text tone="tertiary" size="small">
                Press <Text as="span" weight="semibold">Intern next</Text> to allocate the first handle.
              </Text>
            ) : (
              <Stack gap={8}>
                {sim.arena.map((seg) => (
                  <div
                    key={seg.handle}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: "8px 10px",
                      border: `1px solid ${t.stroke.secondary}`,
                      borderLeft: `3px solid ${P}`,
                      borderRadius: 6,
                      background: t.bg.elevated,
                      animation: "sp-pop .3s ease both",
                    }}
                  >
                    <Text size="small" weight="bold" style={{ color: P, width: 30 }}>
                      H{seg.handle}
                    </Text>
                    <Code>{seg.hash}</Code>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <Text size="small" tone="secondary" truncate>
                        {seg.role}: {seg.content}
                      </Text>
                    </div>
                    <Text size="small" tone="quaternary">
                      parent {seg.parent === null ? "∅" : `H${seg.parent}`}
                    </Text>
                  </div>
                ))}
              </Stack>
            )}
          </CardBody>
        </Card>
      </Grid>

      {sim.dedup > 0 && (
        <Callout tone="success" title="Prefix-dependent dedup in action">
          C2's system prompt and first user turn resolved to <Code>H0</Code> and <Code>H1</Code> — the exact handles C1
          created — because their parent chains produced identical ids. Only C2's differing assistant reply allocated a
          new handle. This is why shared prefixes cost nothing to store.
        </Callout>
      )}

      <Card collapsible defaultOpen={false}>
        <CardHeader>The types (dataset/segment.rs)</CardHeader>
        <CardBody>
          <pre style={{ margin: 0, fontSize: 12, lineHeight: "18px", color: t.text.secondary, overflowX: "auto" }}>
{`pub struct Handle(u32);            // dense arena index — public address
pub struct SegmentId([u8; 32]);   // blake3 content hash — dedup key only

pub struct Segment {
    pub id: SegmentId,
    pub parent: Option<Handle>,
    pub payload: Payload,
}

pub struct SegmentPool {           // mutable interner
    arena: Vec<Segment>,
    ids: HashMap<SegmentId, Handle>,
}
pub struct InMemorySegmentStore {  // frozen — ids map discarded
    arena: Box<[Segment]>,
}`}
          </pre>
        </CardBody>
      </Card>
    </Stack>
  );
}

// ═══ PAGE 3 · Payload domains + blake3 recipe explorer ════════════════════════

type Domain = {
  key: string;
  name: string;
  color: keyof Theme["category"];
  fields: string;
  use: string;
  prefix: string;
  recipe: string[];
};

const DOMAINS: Domain[] = [
  {
    key: "message",
    name: "Message",
    color: "blue",
    fields: "role, wire: Bytes, tokens: Box<[u32]>",
    use: "Pre-serialized endpoint message; wire spliced into messages[]",
    prefix: `"message\\0"`,
    recipe: ["role.as_str()", `"\\0"`, "each token (u32 LE)", `"\\0"`, "full wire bytes"],
  },
  {
    key: "text",
    name: "Text",
    color: "cyan",
    fields: "role, bytes: Bytes, token_count: u32",
    use: "Text-only field; token ids folded into id, not stored",
    prefix: `"text-only\\0"`,
    recipe: ["role.as_str()", `"\\0"`, "token ids (hashed at intern time)"],
  },
  {
    key: "raw",
    name: "Raw",
    color: "purple",
    fields: "wire: Bytes",
    use: "Complete JSON body / tools / headers → BodyPlan::Raw or a field",
    prefix: `"raw\\0"`,
    recipe: ["wire bytes"],
  },
  {
    key: "tokenids",
    name: "TokenIds",
    color: "green",
    fields: "token_ids: Box<[u32]>",
    use: "Token-native path; gRPC / validation, not spliced into JSON",
    prefix: `"token-ids\\0"`,
    recipe: ["each token id (u32 LE)"],
  },
  {
    key: "media",
    name: "Media",
    color: "pink",
    fields: "kind: MediaKind, bytes: Bytes",
    use: "Multimodal bytes resolved via Turn.content → endpoint",
    prefix: `"media\\0"`,
    recipe: ["kind string", "media bytes"],
  },
  {
    key: "tracehash",
    name: "TraceHashIds",
    color: "orange",
    fields: "hash_ids: Box<[i64]>, block_size: usize",
    use: "DynoSim / simulator adapters — cache identity only",
    prefix: `"trace-hash-ids\\0"`,
    recipe: ["block_size", "each hash id (i64 sequence)"],
  },
];

function PagePayloads({ t }: { t: Theme }) {
  const [sel, setSel] = useCanvasState<string>("sp.payload.sel", "message");
  const d = DOMAINS.find((x) => x.key === sel) ?? DOMAINS[0];
  const c = t.category[d.color];

  return (
    <Stack gap={16}>
      <div>
        <H2>Payload — six disjoint hash domains</H2>
        <Text tone="secondary">
          A segment's <Code>Payload</Code> is one of six variants. Each hashes under its own blake3 domain prefix, so the
          same bytes in two domains never collide. Every recipe is also framed by the parent's id and the version
          constant <Code>b"aiperf-dataset-segment-v1\0"</Code>. Select a domain to see its recipe.
        </Text>
      </div>

      <Row gap={8} wrap>
        {DOMAINS.map((dm) => (
          <span key={dm.key} style={{ display: "contents" }}>
            <Pill active={dm.key === sel} onClick={() => setSel(dm.key)} leadingContent={
              <span style={{ width: 8, height: 8, borderRadius: 2, background: t.category[dm.color], display: "inline-block" }} />
            }>
              {dm.name}
            </Pill>
          </span>
        ))}
      </Row>

      <Grid columns="1fr 1fr" gap={14}>
        <Card>
          <CardHeader trailing={<Pill size="sm">Payload::{d.name}</Pill>}>Variant</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <div>
                <Text size="small" tone="tertiary">Fields</Text>
                <div style={{ marginTop: 4 }}>
                  <Code>{d.fields}</Code>
                </div>
              </div>
              <Divider />
              <div>
                <Text size="small" tone="tertiary">Role in the pipeline</Text>
                <Text size="small" style={{ marginTop: 4 }}>{d.use}</Text>
              </div>
              <Divider />
              <div>
                <Text size="small" tone="tertiary">SegmentDomain discriminant</Text>
                <div style={{ marginTop: 4 }}>
                  <Code>SegmentDomain::{d.name}</Code>{"  "}
                  <Text as="span" size="small" tone="quaternary">drives dispatch, not field precedence</Text>
                </div>
              </div>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">payload_id() · segment.rs:528</Pill>}>blake3 recipe</CardHeader>
          <CardBody style={{ padding: 12 }}>
            <svg viewBox="0 0 420 320" style={SVG}>
              <defs>
                <Arrow id="pl-a" color={c} />
              </defs>
              {(() => {
                const rows = [
                  { label: `HASH_VERSION`, sub: `b"aiperf-dataset-segment-v1\\0"`, faint: true },
                  { label: `domain prefix`, sub: d.prefix, faint: false },
                  { label: `parent id`, sub: `parent SegmentId + "\\0"`, faint: false },
                  ...d.recipe.map((r) => ({ label: "├", sub: r, faint: false })),
                ];
                const rh = 34;
                const top = 12;
                return (
                  <>
                    {rows.map((r, i) => {
                      const y = top + i * rh;
                      return (
                        <g key={i} style={{ animation: `sp-pop .28s ease ${i * 0.03}s both` }}>
                          <rect
                            x={14}
                            y={y}
                            width={392}
                            height={rh - 8}
                            rx={5}
                            fill={r.faint ? t.fill.quaternary : t.bg.elevated}
                            stroke={r.faint ? t.stroke.tertiary : c}
                            strokeWidth={r.faint ? 1 : 1.3}
                            opacity={r.faint ? 0.7 : 1}
                          />
                          <text x={26} y={y + (rh - 8) / 2} dominantBaseline="middle" fontSize={11} fontWeight={700} fill={t.text.tertiary}>
                            {r.label}
                          </text>
                          <text x={392} y={y + (rh - 8) / 2} textAnchor="end" dominantBaseline="middle" fontSize={11} fill={t.text.primary} fontFamily="monospace">
                            {r.sub}
                          </text>
                        </g>
                      );
                    })}
                    <text x={210} y={top + rows.length * rh + 12} textAnchor="middle" fontSize={11} fontWeight={700} fill={c}>
                      ↓ blake3.finalize()
                    </text>
                    <text x={210} y={top + rows.length * rh + 30} textAnchor="middle" fontSize={11} fill={t.text.secondary} fontFamily="monospace">
                      SegmentId([u8; 32])
                    </text>
                  </>
                );
              })()}
            </svg>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="neutral" title="Why parent-by-id, not parent-by-index">
        A child hash includes its parent's <Text weight="semibold">content hash</Text> rather than its insertion index.
        Loading unrelated rows in a different order can't shuffle handles, so ids stay deterministic — and identical text
        under different prefixes stays distinct.
      </Callout>
    </Stack>
  );
}

// ═══ PAGE 4 · BodyPlan — interactive byte-splice materializer ══════════════════

function PageBodyPlan({ t }: { t: Theme }) {
  const [rawMode, setRawMode] = useCanvasState<boolean>("sp.bp.raw", false);
  const [stream, setStream] = useCanvasState<boolean>("sp.bp.stream", true);
  const [tools, setTools] = useCanvasState<boolean>("sp.bp.tools", false);

  const G = t.category.green; // spliced wire (clone bytes)
  const A = t.accent.primary; // literal (serde_json)
  const Y = t.category.yellow; // override tail

  const msgWire1 = `{"role":"user","content":"What is 2+2?"}`;
  const msgWire2 = `{"role":"assistant","content":"4"}`;
  const toolsWire = `[{"type":"function","function":{"name":"calc"}}]`;
  const rawWire = `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":false}`;

  type Tok = { text: string; kind: "lit" | "seg" | "tail" | "punc" };
  const toks: Tok[] = [];
  if (rawMode) {
    toks.push({ text: rawWire, kind: "seg" });
    toks.push({ text: `,"stream":${stream}`, kind: "tail" });
  } else {
    toks.push({ text: `{`, kind: "punc" });
    toks.push({ text: `"messages":[`, kind: "punc" });
    toks.push({ text: msgWire1, kind: "seg" });
    toks.push({ text: `,`, kind: "punc" });
    toks.push({ text: msgWire2, kind: "seg" });
    toks.push({ text: `]`, kind: "punc" });
    toks.push({ text: `,"model":"gpt-4"`, kind: "lit" });
    if (tools) toks.push({ text: `,"tools":${toolsWire}`, kind: "seg" });
    toks.push({ text: `,"stream":${stream}`, kind: "tail" });
    toks.push({ text: `,"max_tokens":128`, kind: "tail" });
    toks.push({ text: `}`, kind: "punc" });
  }

  const kindColor = (k: Tok["kind"]) =>
    k === "seg" ? G : k === "lit" ? A : k === "tail" ? Y : t.text.tertiary;

  const serdeOps = rawMode ? 0 : toks.filter((x) => x.kind === "lit").length;
  const splices = toks.filter((x) => x.kind === "seg").length;

  return (
    <Stack gap={16}>
      <div>
        <H2>BodyPlan — shape now, bytes later</H2>
        <Text tone="secondary">
          A <Code>BodyPlan</Code> declares which fields exist and which slots are filled by segment handles vs literals.
          The <Code>JsonBodyMaterializer</Code> walks it once and produces <Code>Bytes</Code>: literals are the{" "}
          <Text weight="semibold">only</Text> thing serialized on the hot path — segment fields are pre-serialized wires
          cloned straight through.
        </Text>
      </div>

      <Row gap={8} wrap align="center">
        <Pill active={!rawMode} onClick={() => setRawMode(false)}>BodyPlan::Fields</Pill>
        <Pill active={rawMode} onClick={() => setRawMode(true)}>BodyPlan::Raw</Pill>
        <Spacer />
        {!rawMode && (
          <Pill active={tools} onClick={() => setTools((v) => !v)}>tools {tools ? "on" : "off"}</Pill>
        )}
        <Pill active={stream} onClick={() => setStream((v) => !v)}>stream {stream ? "on" : "off"}</Pill>
      </Row>

      <Grid columns="1fr 1fr" gap={14}>
        <Card>
          <CardHeader trailing={<Pill size="sm">{rawMode ? "Raw(Handle)" : "Fields[..]"}</Pill>}>Plan</CardHeader>
          <CardBody style={{ padding: 12 }}>
            {rawMode ? (
              <Stack gap={8}>
                <FieldRow t={t} color={G} name="Raw(H7)" kind="Segment wire" note="complete body — endpoint bypassed" />
                <FieldRow t={t} color={Y} name="+ overrides" kind="tail splice" note="stream / model patched in" />
              </Stack>
            ) : (
              <Stack gap={8}>
                <FieldRow t={t} color={G} name="messages" kind="FieldValue::Segments" note="[H2, H3] → wire clones" />
                <FieldRow t={t} color={A} name="model" kind="FieldValue::Literal" note="serde_json::to_writer" />
                {tools && <FieldRow t={t} color={G} name="tools" kind="FieldValue::Segment" note="H5 → raw wire clone" />}
                <FieldRow t={t} color={Y} name="stream" kind="override tail" note="merge_overrides" />
                <FieldRow t={t} color={Y} name="max_tokens" kind="override tail" note="merge_overrides" />
              </Stack>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">MaterializedRequest.body</Pill>}>Materialized bytes</CardHeader>
          <CardBody style={{ padding: 12 }}>
            <div
              style={{
                fontFamily: "monospace",
                fontSize: 11.5,
                lineHeight: "19px",
                wordBreak: "break-all",
                padding: 10,
                borderRadius: 6,
                background: t.bg.editor,
                border: `1px solid ${t.stroke.secondary}`,
              }}
            >
              {toks.map((tok, i) => (
                <span
                  key={i}
                  style={{
                    color: kindColor(tok.kind),
                    background: tok.kind === "seg" ? t.fill.tertiary : tok.kind === "tail" ? t.fill.quaternary : "transparent",
                    borderRadius: 3,
                    padding: tok.kind === "punc" ? "0" : "1px 2px",
                  }}
                >
                  {tok.text}
                </span>
              ))}
            </div>
            <Row gap={14} style={{ marginTop: 12 }} wrap>
              <Legend t={t} color={G} label="segment wire — cloned" />
              <Legend t={t} color={A} label="literal — serialized" />
              <Legend t={t} color={Y} label="override tail" />
            </Row>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns={3} gap={12}>
        <Stat value={serdeOps} label="serde_json ops (hot path)" tone={serdeOps === 0 ? "success" : undefined} />
        <Stat value={splices} label="byte splices (zero re-serialize)" tone="success" />
        <Stat value={rawMode ? "bypass" : "compose"} label={rawMode ? "endpoint bypassed" : "endpoint.format_payload"} />
      </Grid>

      <Card collapsible defaultOpen={false}>
        <CardHeader>The types (body_plan.rs)</CardHeader>
        <CardBody>
          <pre style={{ margin: 0, fontSize: 12, lineHeight: "18px", color: t.text.secondary, overflowX: "auto" }}>
{`pub enum FieldValue {
    Literal(Value),                    // serialized on the hot path
    Segment(Handle),                   // one wire cloned from the store
    Segments(SmallVec<[Handle; 1]>),   // [ wire, wire, ... ] joined
    Wires(SmallVec<[Bytes; 1]>),       // dynamic content, no store lookup
}

pub enum BodyPlan {
    Raw(Handle),                                   // whole body passthrough
    Fields(SmallVec<[(FieldName, FieldValue); 8]>),
}`}
          </pre>
        </CardBody>
      </Card>
    </Stack>
  );
}

function FieldRow({ t, color, name, kind, note }: { t: Theme; color: string; name: string; kind: string; note: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 10px",
        border: `1px solid ${t.stroke.secondary}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 6,
        background: t.bg.elevated,
      }}
    >
      <Code>{name}</Code>
      <div style={{ minWidth: 0, flex: 1 }}>
        <Text size="small" tone="secondary" truncate>
          {note}
        </Text>
      </div>
      <Text size="small" tone="quaternary">{kind}</Text>
    </div>
  );
}

function Legend({ t, color, label }: { t: Theme; color: string; label: string }) {
  return (
    <Row gap={6} align="center">
      <span style={{ width: 10, height: 10, borderRadius: 2, background: color, display: "inline-block" }} />
      <Text size="small" tone="tertiary">{label}</Text>
    </Row>
  );
}

// ═══ PAGE 5 · Prefix trie / content addressing ════════════════════════════════

function PagePrefix({ t }: { t: Theme }) {
  const A = t.category.blue;
  const G = t.category.green;
  const P = t.category.purple;
  const shared = t.category.cyan;
  return (
    <Stack gap={16}>
      <div>
        <H2>Prefix chains &amp; LCP-trie lowering</H2>
        <Text tone="secondary">
          Composers keep a running <Code>parent: Option&lt;Handle&gt;</Code> per conversation, so each turn extends a
          chain. Recorded traces (WEKA / Dynamo) go further: an LCP trie over block hashes finds the longest shared
          prefix, and the <Code>PromptMessageCache</Code> keyed on <Code>(parent, role, block_hashes)</Code> reuses the
          decoded, interned handle for any prefix two nodes share.
        </Text>
      </div>

      <Card>
        <CardHeader trailing={<Pill size="sm">recorded/trie · compose.rs</Pill>}>Two conversations, one shared prefix</CardHeader>
        <CardBody style={{ padding: 12 }}>
          <svg viewBox="0 0 900 340" style={SVG}>
            <defs>
              <Arrow id="pf-s" color={shared} />
              <Arrow id="pf-b" color={A} />
              <Arrow id="pf-g" color={G} />
            </defs>

            {/* shared root chain */}
            <Box x={40} y={140} w={150} h={54} t={t} title="H0 · system" sub="You are helpful." accent={shared} />
            <Box x={230} y={140} w={150} h={54} t={t} title="H1 · user" sub="What is 2+2?" accent={shared} />

            <text x={115} y={128} textAnchor="middle" fontSize={10} fontWeight={700} fill={shared}>
              SHARED PREFIX (interned once)
            </text>

            {/* branch to C1 */}
            <Box x={470} y={54} w={160} h={54} t={t} title="H2 · assistant" sub={`"4"`} accent={A} />
            <Box x={680} y={54} w={170} h={54} t={t} title="C1 request" sub="H0 · H1 · H2" accent={A} />

            {/* branch to C2 */}
            <Box x={470} y={226} w={160} h={54} t={t} title="H3 · assistant" sub={`"It equals four."`} accent={G} />
            <Box x={680} y={226} w={170} h={54} t={t} title="C2 request" sub="H0 · H1 · H3" accent={G} />

            <Edge id="pf-s" color={shared} d="M190,167 L230,167" dur={1.8} width={1.8} />
            <Edge id="pf-b" color={A} d="M380,155 C420,150 435,100 470,88" />
            <Edge id="pf-g" color={G} d="M380,180 C420,190 435,245 470,252" />
            <Edge id="pf-b" color={A} d="M630,81 L680,81" />
            <Edge id="pf-g" color={G} d="M630,253 L680,253" />

            <Dots d="M190,167 L230,167" color={shared} n={2} dur={1.4} />

            <text x={430} y={330} textAnchor="middle" fontSize={10} fill={t.text.quaternary}>
              H0 and H1 are stored a single time; both requests reference them by handle.
            </text>
          </svg>
        </CardBody>
      </Card>

      <Grid columns="1fr 1fr" gap={14}>
        <Card>
          <CardHeader>resolve_content_parents</CardHeader>
          <CardBody>
            <Text size="small" tone="secondary">
              Walks each node's <Code>hash_ids</Code> along trie edges; the longest match yields the{" "}
              <Code>content_parent</Code>. Prefers the latest full-prefix terminal, else the earliest partial passer.
            </Text>
            <div style={{ marginTop: 8 }}>
              <Text size="small" tone="quaternary">graph/recorded/trie/parents.rs:18</Text>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>rebase on context injection</CardHeader>
          <CardBody>
            <Text size="small" tone="secondary">
              When a system / user_context is injected after compose, <Code>rebase_conversation_handles</Code>{" "}
              re-interns every handle under the new root so blake3 ids reflect the shared prefix — content unchanged,
              identity refreshed.
            </Text>
            <div style={{ marginTop: 8 }}>
              <Text size="small" tone="quaternary">dataset/compose.rs:338</Text>
            </div>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

// ═══ PAGE 6 · Dispatch precedence (Turn.body) ═════════════════════════════════

function PageDispatch({ t }: { t: Theme }) {
  const P = t.category.purple;
  const G = t.category.green;
  const A = t.category.blue;
  const Y = t.category.yellow;
  return (
    <Stack gap={16}>
      <div>
        <H2>Dispatch — one precedence vector, domain-driven</H2>
        <Text tone="secondary">
          A <Code>Turn</Code> stores large data only as handles. <Code>Turn.body</Code> is the single dispatch
          precedence vector; the <Text weight="semibold">domain</Text> of its first handle decides how the request body
          is built — replacing the old five-field precedence.
        </Text>
      </div>

      <Card>
        <CardHeader trailing={<Pill size="sm">request.rs:288 · dataset/model.rs:321</Pill>}>Turn.dispatch_body → materialize</CardHeader>
        <CardBody style={{ padding: 12 }}>
          <svg viewBox="0 0 900 300" style={SVG}>
            <defs>
              <Arrow id="dp-p" color={P} />
              <Arrow id="dp-g" color={G} />
              <Arrow id="dp-a" color={A} />
              <Arrow id="dp-y" color={Y} />
            </defs>

            <Box x={40} y={118} w={170} h={62} t={t} title="Turn.body" sub="SmallVec<[Handle]>" micro="dispatch precedence" accent={t.accent.primary} />

            <Box x={300} y={30} w={200} h={56} t={t} title="Raw handle first?" sub="→ complete body" micro="endpoint bypass" accent={P} />
            <Box x={300} y={122} w={200} h={56} t={t} title="TokenIds handle?" sub="→ token-native" micro="gRPC / validation" accent={G} />
            <Box x={300} y={214} w={200} h={56} t={t} title="Message handles" sub="→ format as array" micro="most common" accent={A} />

            <Box x={590} y={122} w={180} h={62} t={t} title="BodyPlan" sub="raw · cached · format" micro="merge_overrides" accent={A} />
            <Box x={800} y={122} w={80} h={62} t={t} title="Bytes" sub="→ wire" accent={Y} />

            <Edge id="dp-p" color={P} d="M210,140 C255,120 260,70 300,60" />
            <Edge id="dp-g" color={G} d="M210,149 L300,150" />
            <Edge id="dp-a" color={A} d="M210,158 C255,180 260,235 300,242" />

            <Edge id="dp-p" color={P} d="M500,58 C545,70 555,120 590,140" />
            <Edge id="dp-a" color={A} d="M500,150 L590,150" />
            <Edge id="dp-a" color={A} d="M500,242 C545,230 555,180 590,162" />

            <Edge id="dp-y" color={Y} d="M770,153 L800,153" dur={1.6} />
            <Dots d="M770,153 L800,153" color={Y} n={2} dur={1.4} />
          </svg>
        </CardBody>
      </Card>

      <Grid columns="1fr 1fr" gap={14}>
        <Card>
          <CardHeader>dispatch_body precedence</CardHeader>
          <CardBody>
            <pre style={{ margin: 0, fontSize: 11.5, lineHeight: "17px", color: t.text.secondary, overflowX: "auto" }}>
{`pub fn dispatch_body(
    raw_payload: Option<Handle>,
    raw_token_ids: Option<Handle>,
    messages: &[Handle],
) -> SmallVec<[Handle; 1]> {
    let mut body = SmallVec::new();
    if let Some(raw) = raw_payload { body.push(raw); }
    if let Some(tok) = raw_token_ids { body.push(tok); }
    if raw_payload.is_none()
        && raw_token_ids.is_none() {
        body.extend_from_slice(messages);
    }
    body
}`}
            </pre>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>The two seams this feeds</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Callout tone="neutral" title="{ transport }">
                <Text size="small">
                  <Code>RequestSink&lt;R&gt;::dispatch</Code> drives materialized bytes to terminal, emitting arrival /
                  token / usage through a <Code>RequestObserver</Code>.
                </Text>
              </Callout>
              <Callout tone="neutral" title="Graph fast path">
                <Text size="small">
                  Graph HTTP dispatch skips <Code>BodyPlan</Code> entirely — it splices message wires directly via{" "}
                  <Code>build_message_body_from_wire_parts</Code>. Same store, different splice.
                </Text>
              </Callout>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

// ═══ Shell ════════════════════════════════════════════════════════════════════

const PAGES: { id: PageId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "pool", label: "SegmentPool" },
  { id: "payloads", label: "Payloads" },
  { id: "bodyplan", label: "BodyPlan" },
  { id: "prefix", label: "Prefix trie" },
  { id: "dispatch", label: "Dispatch" },
];

export default function SegmentPoolsAndBodyPlans() {
  const t = useHostTheme();
  const [page, setPage] = useCanvasState<PageId>("sp.page", "overview");
  const dispatch = useCanvasAction();

  return (
    <div style={{ background: t.bg.editor, minHeight: "100%", padding: 24 }}>
      <style>{CSS}</style>
      <Stack gap={18}>
        <Row align="center" gap={12} wrap>
          <div style={{ minWidth: 0 }}>
            <H1>Segment Pools &amp; Body Plans</H1>
            <Text tone="tertiary" size="small">
              How AIPerf turns dataset rows into deduplicated, content-addressed segments and splices them into wire
              bytes — <Code>rust/aiperf/src/dataset</Code> · <Code>body_plan.rs</Code>
            </Text>
          </div>
          <Spacer />
          <Button
            variant="ghost"
            onClick={() =>
              dispatch({ type: "openFile", path: "rust/aiperf/src/dataset/segment.rs" })
            }
          >
            Open segment.rs
          </Button>
          <Button
            variant="ghost"
            onClick={() => dispatch({ type: "openFile", path: "rust/aiperf/src/body_plan.rs" })}
          >
            Open body_plan.rs
          </Button>
        </Row>

        <Row gap={8} wrap>
          {PAGES.map((p) => (
            <span key={p.id} style={{ display: "contents" }}>
              <Pill active={p.id === page} onClick={() => setPage(p.id)}>
                {p.label}
              </Pill>
            </span>
          ))}
        </Row>

        <Divider />

        {page === "overview" && <PageOverview t={t} />}
        {page === "pool" && <PagePool t={t} />}
        {page === "payloads" && <PagePayloads t={t} />}
        {page === "bodyplan" && <PageBodyPlan t={t} />}
        {page === "prefix" && <PagePrefix t={t} />}
        {page === "dispatch" && <PageDispatch t={t} />}
      </Stack>
    </div>
  );
}
