import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type View =
  | "system"
  | "processes"
  | "runtime"
  | "protocol"
  | "scheduled"
  | "graph"
  | "endpoints"
  | "metrics"
  | "cellular"
  | "features"
  | "seams";
type Theme = ReturnType<typeof useHostTheme>;

const VIEWS: Array<{ id: View; label: string; hint: string }> = [
  { id: "system", label: "1 · System", hint: "product landscape" },
  { id: "processes", label: "2 · Processes", hint: "crates and boundaries" },
  { id: "runtime", label: "3 · Runtime", hint: "one request end-to-end" },
  { id: "protocol", label: "4 · Protocol", hint: "one child lifecycle" },
  { id: "scheduled", label: "5 · Scheduled", hint: "paced workload path" },
  { id: "graph", label: "6 · Graph", hint: "trace replay path" },
  { id: "endpoints", label: "7 · Endpoints", hint: "dialect preparation" },
  { id: "metrics", label: "8 · Metrics", hint: "measurement and exports" },
  { id: "cellular", label: "9 · Cellular", hint: "multi-process scale" },
  { id: "features", label: "10 · Builds", hint: "feature composition" },
  { id: "seams", label: "11 · Seams", hint: "extension internals" },
];

const SVG_STYLE = { width: "100%", height: "auto", display: "block" } as const;

function Arrow({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill={color} />
    </marker>
  );
}

function Edge({
  d,
  color,
  marker,
  dashed = false,
  width = 1.4,
}: {
  d: string;
  color: string;
  marker: string;
  dashed?: boolean;
  width?: number;
}) {
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dashed ? "5 4" : undefined}
      markerEnd={`url(#${marker})`}
    />
  );
}

function Box({
  t,
  x,
  y,
  w,
  h,
  title,
  sub,
  accent,
  muted = false,
}: {
  t: Theme;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  accent?: string;
  muted?: boolean;
}) {
  return (
    <g opacity={muted ? 0.58 : 1}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill={t.fill.tertiary}
        stroke={accent ?? t.stroke.primary}
        strokeWidth={accent ? 1.5 : 1}
      />
      <text
        x={x + w / 2}
        y={sub ? y + h / 2 - 5 : y + h / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={12}
        fontWeight={650}
        fill={t.text.primary}
      >
        {title}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 11}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={9}
          fill={t.text.tertiary}
        >
          {sub}
        </text>
      )}
    </g>
  );
}

function Band({
  t,
  x,
  y,
  w,
  h,
  label,
  accent,
}: {
  t: Theme;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  accent: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={11} fill={t.fill.quaternary} stroke={t.stroke.secondary} />
      <rect x={x} y={y} width={5} height={h} rx={2} fill={accent} />
      <text x={x + 17} y={y + 19} fontSize={9} fontWeight={700} fill={accent} style={{ letterSpacing: 0.6 }}>
        {label.toUpperCase()}
      </text>
    </g>
  );
}

function EvidenceButtons({ paths }: { paths: Array<{ label: string; path: string }> }) {
  const dispatch = useCanvasAction();
  return (
    <Row gap={8} wrap>
      {paths.map((item) => (
        <Button variant="ghost" onClick={() => dispatch({ type: "openFile", path: item.path })}>
          {item.label}
        </Button>
      ))}
    </Row>
  );
}

function SystemView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        AIPerf’s Rust product is one native binary with two roles: the entry point re-execs itself as
        <Code>aiperf --execute</Code> for each benchmark. The execution child owns load dispatch; the mock server is
        only an independently launched test target.
      </Text>
      <svg viewBox="0 0 900 430" style={SVG_STYLE}>
        <defs>
          <Arrow id="sys" color={line} />
          <Arrow id="sys-blue" color={blue} />
          <Arrow id="sys-green" color={green} />
        </defs>

        <Band t={t} x={12} y={12} w={876} h={78} label="Author and launch" accent={blue} />
        <Band t={t} x={12} y={108} w={876} h={110} label="Execute one run" accent={green} />
        <Band t={t} x={12} y={236} w={876} h={112} label="Dispatch target" accent={purple} />
        <Band t={t} x={12} y={366} w={876} h={52} label="Artifacts and integrations" accent={orange} />

        <Box t={t} x={70} y={32} w={170} h={42} title="User / automation" sub="Config v2 + CLI flags" />
        <Box t={t} x={308} y={28} w={220} h={50} title="aiperf" sub="native aiperf-cli entry point" accent={blue} />
        <Box t={t} x={608} y={32} w={220} h={42} title="Peripheral commands" sub="native or delegated to Python" muted />
        <Edge d="M240,53 L308,53" color={line} marker="sys" />
        <Edge d="M528,53 L608,53" color={line} marker="sys" dashed />

        <Box t={t} x={278} y={132} w={280} h={60} title="aiperf --execute" sub="same binary re-exec · strict protocol v2" accent={green} />
        <Box t={t} x={630} y={132} w={200} h={60} title="aiperf --cell" sub="optional cells > 1 over velo" accent={orange} />
        <Edge d="M418,78 L418,132" color={blue} marker="sys-blue" width={1.8} />
        <Edge d="M558,162 L630,162" color={orange} marker="sys" dashed />

        <Box t={t} x={62} y={264} w={220} h={58} title="Real inference server" sub="OpenAI · Anthropic · KServe · Riva" accent={blue} />
        <Box t={t} x={340} y={264} w={220} h={58} title="aiperf-mock-server" sub="standalone online test target" accent={purple} />
        <Box t={t} x={618} y={264} w={220} h={58} title="Dynamo SteppableReplay" sub="in-process dynosim feature" accent={green} />
        <Edge d="M366,192 C315,224 236,226 172,264" color={line} marker="sys" />
        <Edge d="M418,192 L450,264" color={line} marker="sys" />
        <Edge d="M500,192 C585,220 665,230 728,264" color={line} marker="sys" dashed />

        <Box t={t} x={95} y={373} w={185} h={36} title="native-v2 report" accent={orange} />
        <Box t={t} x={350} y={373} w={200} h={36} title="JSON / CSV / Parquet" accent={orange} />
        <Box t={t} x={620} y={373} w={190} h={36} title="OTLP · MLflow · W&B" accent={orange} />
        <Edge d="M418,192 C420,300 236,340 188,373" color={green} marker="sys-green" />
        <Edge d="M280,391 L350,391" color={line} marker="sys" />
        <Edge d="M550,391 L620,391" color={line} marker="sys" dashed />
      </svg>

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Product boundary">
          The entry-point process authors and launches. The same <Code>aiperf</Code> binary, re-exec’d in internal
          execute mode, is the only process that dispatches benchmark load.
        </Callout>
        <Callout tone="neutral" title="Same online path">
          Real and mock online runs use the same HTTP/gRPC clients; only the target address changes.
        </Callout>
        <Callout tone="warning" title="Feature gate">
          DynoSim is compiled through the execution binary’s <Code>dynosim</Code> feature; it is not a separate command.
        </Callout>
      </Grid>
      <EvidenceButtons
        paths={[
          { label: "CLI routing", path: "rust/cli/src/dispatch.rs" },
          { label: "Execution mode", path: "rust/cli/src/execute_mode.rs" },
          { label: "Workspace crates", path: "Cargo.toml" },
        ]}
      />
    </Stack>
  );
}

function ProcessesView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Solid arrows are compile-time dependencies or self re-exec. Dashed arrows are runtime network or optional
        feature paths. The large <Code>aiperf</Code> library absorbs the former multi-crate runtime modules.
      </Text>
      <svg viewBox="0 0 900 510" style={SVG_STYLE}>
        <defs>
          <Arrow id="proc" color={line} />
          <Arrow id="proc-green" color={green} />
        </defs>

        <text x={28} y={30} fontSize={10} fontWeight={700} fill={t.text.tertiary}>EXECUTABLE PROCESS ROLES</text>
        <Box t={t} x={28} y={46} w={210} h={64} title="aiperf entry point" sub="profile · config · chat · validate…" accent={blue} />
        <Box t={t} x={345} y={46} w={210} h={64} title="aiperf --execute" sub="same binary · stdio v2 · isolated child" accent={green} />
        <Box t={t} x={662} y={46} w={210} h={64} title="aiperf-mock-server" sub="HTTP/SSE · gRPC · TLS · UDS" accent={purple} />
        <Edge d="M238,78 L345,78" color={line} marker="proc" />

        <text x={28} y={159} fontSize={10} fontWeight={700} fill={t.text.tertiary}>LIBRARIES</text>
        <Box t={t} x={235} y={178} w={430} h={82} title="aiperf" sub="runtime composition + runner_protocol + 16 absorbed modules" accent={green} />
        <Box t={t} x={335} y={318} w={230} h={62} title="loadgen-core" sub="Dispatchable · RequestSink · RequestObserver" accent={orange} />
        <Box t={t} x={652} y={318} w={220} h={62} title="e2e harness" sub="product-level integration tests" muted />

        <Edge d="M450,110 L450,178" color={line} marker="proc" />
        <Edge d="M164,110 C174,150 270,168 318,178" color={line} marker="proc" />
        <Edge d="M767,110 C748,150 650,168 582,178" color={line} marker="proc" />
        <Edge d="M450,260 L450,318" color={line} marker="proc" />
        <Edge d="M762,318 C724,286 660,264 615,252" color={line} marker="proc" dashed />

        <rect x={142} y={417} width={616} height={72} rx={10} fill={t.fill.quaternary} stroke={t.stroke.secondary} />
        <text x={160} y={438} fontSize={9} fontWeight={700} fill={t.text.tertiary}>EXTERNAL RUNTIME BOUNDARIES</text>
        <Box t={t} x={168} y={448} w={170} h={30} title="HTTP / gRPC servers" accent={blue} />
        <Box t={t} x={365} y={448} w={170} h={30} title="Dynamo mocker" accent={green} />
        <Box t={t} x={562} y={448} w={170} h={30} title="Python evaluators" accent={purple} />
        <Edge d="M450,260 C382,340 270,391 253,448" color={line} marker="proc" dashed />
        <Edge d="M500,260 C500,340 469,401 450,448" color={line} marker="proc" dashed />
        <Edge d="M550,260 C612,337 645,397 647,448" color={line} marker="proc" dashed />
      </svg>

      <Grid columns="1.2fr 1fr" gap={16}>
        <Stack gap={8}>
          <H2>Dependency direction</H2>
          <Text><Code>aiperf-cli</Code> → <Code>aiperf</Code> → <Code>loadgen-core</Code></Text>
          <Text>The entry point re-execs the current <Code>aiperf</Code> binary with <Code>--execute</Code></Text>
          <Text><Code>aiperf-mock-server</Code> → <Code>aiperf</Code>; execute mode and mock do not depend on each other</Text>
        </Stack>
        <Callout tone="neutral" title="Packaging">
          The workspace still contains and packages the older <Code>aiperf-runner</Code> crate, but the default CLI path self re-execs.
        </Callout>
      </Grid>
      <EvidenceButtons
        paths={[
          { label: "aiperf modules", path: "rust/aiperf/src/lib.rs" },
          { label: "Executable features", path: "rust/cli/Cargo.toml" },
          { label: "Library features", path: "rust/aiperf/Cargo.toml" },
        ]}
      />
    </Stack>
  );
}

function RuntimeView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        This is the one-run hot path. Startup uses frozen registries and strict DTOs; request execution then stays on
        transport-native request types and local observer graphs.
      </Text>
      <svg viewBox="0 0 940 620" style={SVG_STYLE}>
        <defs>
          <Arrow id="run" color={line} />
          <Arrow id="run-green" color={green} />
          <Arrow id="run-orange" color={orange} />
        </defs>

        <Band t={t} x={12} y={10} w={916} h={92} label="1 · Author and bootstrap" accent={blue} />
        <Box t={t} x={48} y={38} w={190} h={42} title="Config v2 / flags" />
        <Box t={t} x={290} y={34} w={230} h={50} title="AuthoredRunSpecV2" sub="serialized to execution-child stdin" accent={blue} />
        <Box t={t} x={578} y={34} w={300} h={50} title="RunnerApplication::stock" sub="freeze registries + resolvers + factories" accent={green} />
        <Edge d="M238,59 L290,59" color={line} marker="run" />
        <Edge d="M520,59 L578,59" color={line} marker="run" />

        <Band t={t} x={12} y={120} w={916} h={112} label="2 · Validate and prepare" accent={green} />
        <Box t={t} x={42} y={154} w={170} h={48} title="Coordinator" sub="resolve IDs; fail closed" />
        <Box t={t} x={250} y={150} w={190} h={56} title="Workload factory" sub="scheduled · graph" accent={green} />
        <Box t={t} x={478} y={150} w={190} h={56} title="Transport factory" sub="http · grpc · dynosim" accent={purple} />
        <Box t={t} x={706} y={150} w={180} h={56} title="Prepared operation" sub="one-shot executable" accent={orange} />
        <Edge d="M212,178 L250,178" color={line} marker="run" />
        <Edge d="M440,178 L478,178" color={line} marker="run" />
        <Edge d="M668,178 L706,178" color={line} marker="run" />

        <Band t={t} x={12} y={250} w={916} h={188} label="3 · Run phases and dispatch requests" accent={orange} />
        <Box t={t} x={42} y={286} w={175} h={54} title="Phase runtime" sub="warmup → profiling" accent={orange} />
        <Box t={t} x={258} y={286} w={190} h={54} title="Workload driver" sub="scheduled or graph" accent={green} />
        <Box t={t} x={489} y={286} w={180} h={54} title="Admission + pacing" sub="SlotPool · arrivals · stop" />
        <Box t={t} x={710} y={286} w={176} h={54} title="Prepared endpoint" sub="request body + parser" accent={blue} />
        <Edge d="M217,313 L258,313" color={line} marker="run" />
        <Edge d="M448,313 L489,313" color={line} marker="run" />
        <Edge d="M669,313 L710,313" color={line} marker="run" />

        <Box t={t} x={102} y={368} w={198} h={44} title="Clock" sub="RealClock or SimClock" accent={purple} />
        <Box t={t} x={371} y={364} w={220} h={52} title="RequestSink<R>::dispatch" sub="HTTP · gRPC · DirectRequest" accent={purple} />
        <Box t={t} x={662} y={364} w={210} h={52} title="RequestObserver" sub="arrival · admit · token · usage · terminal" accent={orange} />
        <Edge d="M300,390 L371,390" color={purple} marker="run" dashed />
        <Edge d="M591,390 L662,390" color={orange} marker="run-orange" width={1.8} />
        <Edge d="M798,340 L520,364" color={line} marker="run" />

        <Band t={t} x={12} y={456} w={916} h={150} label="4 · Reduce, join side channels, commit" accent={green} />
        <Box t={t} x={42} y={492} w={175} h={54} title="Per-worker capture" sub="records or t-digest sketch" />
        <Box t={t} x={257} y={492} w={185} h={54} title="Metrics accumulator" sub="merge once after drain" accent={green} />
        <Box t={t} x={482} y={492} w={185} h={54} title="Side channels" sub="GPU · server · network" accent={purple} />
        <Box t={t} x={707} y={492} w={179} h={54} title="Native exporters" sub="commit report + artifacts" accent={orange} />
        <Edge d="M767,416 C767,458 184,452 130,492" color={orange} marker="run-orange" />
        <Edge d="M217,519 L257,519" color={line} marker="run" />
        <Edge d="M442,519 L482,519" color={line} marker="run" />
        <Edge d="M667,519 L707,519" color={green} marker="run-green" />
        <text x={797} y={580} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>native-v2.json + compatibility exports</text>
      </svg>

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Startup vs hot path">
          Type erasure and registry lookups happen during validation/preparation, not per token.
        </Callout>
        <Callout tone="neutral" title="Timing authority">
          Arrival, admission, token, cancellation, and phase timing come from the injected <Code>Clock</Code>.
        </Callout>
        <Callout tone="success" title="Lock avoidance">
          Worker-local <Code>Rc/RefCell</Code> observer state avoids an <Code>Arc/Mutex</Code> on each token.
        </Callout>
      </Grid>
      <EvidenceButtons
        paths={[
          { label: "Application composition", path: "rust/aiperf/src/runner_protocol/application.rs" },
          { label: "Registry contracts", path: "rust/aiperf/src/runner_protocol/registry.rs" },
          { label: "Request seam", path: "rust/loadgen-core/src/sink.rs" },
        ]}
      />
    </Stack>
  );
}

function ProtocolView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Each benchmark gets a fresh process boundary without a second product binary: the entry point resolves
        <Code>current_exe()</Code>, starts <Code>aiperf --execute</Code>, writes one protocol-v2 envelope, and waits for
        one terminal JSON line.
      </Text>
      <svg viewBox="0 0 920 470" style={SVG_STYLE}>
        <defs><Arrow id="proto" color={line} /><Arrow id="proto-green" color={green} /></defs>
        <Band t={t} x={12} y={12} w={896} h={92} label="Parent process" accent={blue} />
        <Box t={t} x={42} y={38} w={180} h={44} title="profile / sweep / search" />
        <Box t={t} x={276} y={34} w={190} h={52} title="RunnerRequest v2" sub="operation + AuthoredRunSpecV2" accent={blue} />
        <Box t={t} x={520} y={34} w={160} h={52} title="exec_bin::resolve" sub="override or current_exe" />
        <Box t={t} x={734} y={34} w={145} h={52} title="spawn child" sub="--execute" accent={green} />
        <Edge d="M222,60 L276,60" color={line} marker="proto" />
        <Edge d="M466,60 L520,60" color={line} marker="proto" />
        <Edge d="M680,60 L734,60" color={line} marker="proto" />

        <Band t={t} x={12} y={124} w={896} h={238} label="Execution child" accent={green} />
        <Box t={t} x={42} y={154} w={155} h={48} title="stdin to EOF" sub="strict JSON decode" accent={purple} />
        <Box t={t} x={239} y={150} w={175} h={56} title="bootstrap" sub="protocol=2 · validate|execute" />
        <Box t={t} x={456} y={150} w={195} h={56} title="RunnerApplication" sub="frozen distribution universe" accent={green} />
        <Box t={t} x={693} y={150} w={170} h={56} title="Coordinator" sub="validate → prepare" accent={orange} />
        <Edge d="M197,178 L239,178" color={line} marker="proto" />
        <Edge d="M414,178 L456,178" color={line} marker="proto" />
        <Edge d="M651,178 L693,178" color={line} marker="proto" />

        <Box t={t} x={182} y={268} w={190} h={52} title="validate operation" sub="side-effect-free result" accent={blue} />
        <Box t={t} x={514} y={268} w={190} h={52} title="execute operation" sub="run + commit report" accent={green} />
        <Edge d="M778,206 C720,238 400,230 277,268" color={line} marker="proto" dashed />
        <Edge d="M778,206 C765,244 642,247 609,268" color={line} marker="proto" />

        <Band t={t} x={12} y={382} w={896} h={74} label="Terminal contract" accent={orange} />
        <Box t={t} x={80} y={398} w={230} h={42} title="stderr" sub="diagnostics and lifecycle only" />
        <Box t={t} x={345} y={394} w={230} h={50} title="stdout" sub="exactly one typed JSONL envelope" accent={orange} />
        <Box t={t} x={610} y={398} w={230} h={42} title="parent parses terminal" sub="success · report_path · error" />
        <Edge d="M460,362 L460,394" color={green} marker="proto-green" />
        <Edge d="M575,419 L610,419" color={line} marker="proto" />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Isolation">Signals and panics are contained in the child; the parent remains the presentation shell.</Callout>
        <Callout tone="neutral" title="Capabilities">The linked catalog is composed in-process; it is not a public CLI subcommand.</Callout>
        <Callout tone="warning" title="Override"><Code>AIPERF_EXEC_BIN</Code> may point development runs at a differently featured executable.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Self-exec resolver", path: "rust/cli/src/exec_bin.rs" },
        { label: "Parent protocol client", path: "rust/cli/src/execute.rs" },
        { label: "Execution child", path: "rust/cli/src/execute_mode.rs" },
        { label: "Coordinator", path: "rust/aiperf/src/runner_protocol/coordinator.rs" },
      ]} />
    </Stack>
  );
}

function ScheduledView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        The scheduled workload lowers datasets into conversations, applies one arrival policy, admits work through
        bounded slots, and dispatches prepared turns over HTTP, gRPC, or DynoSim.
      </Text>
      <svg viewBox="0 0 930 520" style={SVG_STYLE}>
        <defs><Arrow id="sched" color={line} /></defs>
        <Band t={t} x={12} y={12} w={906} h={96} label="Prepare data and policy" accent={blue} />
        <Box t={t} x={38} y={38} w={160} h={46} title="dataset input" sub="synthetic · file · public" />
        <Box t={t} x={238} y={38} w={150} h={46} title="loader" sub="Dataset + conversations" accent={blue} />
        <Box t={t} x={428} y={38} w={150} h={46} title="sampler" sub="sequential · shuffle · random" />
        <Box t={t} x={618} y={38} w={260} h={46} title="NativeRunSpec" sub="phases · limits · arrival · endpoint profiles" accent={green} />
        <Edge d="M198,61 L238,61" color={line} marker="sched" />
        <Edge d="M388,61 L428,61" color={line} marker="sched" />
        <Edge d="M578,61 L618,61" color={line} marker="sched" />

        <Band t={t} x={12} y={126} w={906} h={150} label="Drive phases and arrivals" accent={green} />
        <Box t={t} x={38} y={162} w={165} h={54} title="PhaseOrchestrator" sub="warmup → profiling" accent={orange} />
        <Box t={t} x={243} y={162} w={165} h={54} title="workload policy" sub="request-rate · user-centric · fixed" accent={green} />
        <Box t={t} x={448} y={162} w={165} h={54} title="arrival schedule" sub="constant · Poisson · Gamma · burst" />
        <Box t={t} x={653} y={162} w={225} h={54} title="SlotPool + StopChecker" sub="admission · request/duration bounds" accent={purple} />
        <Edge d="M203,189 L243,189" color={line} marker="sched" />
        <Edge d="M408,189 L448,189" color={line} marker="sched" />
        <Edge d="M613,189 L653,189" color={line} marker="sched" />
        <text x={120} y={246} fontSize={9} fill={t.text.tertiary}>ramp · cancellation · grace · drain</text>
        <text x={765} y={246} fontSize={9} fill={t.text.tertiary}>continuations receive FIFO priority</text>

        <Band t={t} x={12} y={294} w={906} h={126} label="Place and dispatch" accent={purple} />
        <Box t={t} x={38} y={328} w={165} h={54} title="PreparedTurn" sub="materialized conversation turn" />
        <Box t={t} x={243} y={328} w={180} h={54} title="TurnDispatcher" sub="placement abstraction" accent={purple} />
        <Box t={t} x={463} y={328} w={190} h={54} title="worker-local endpoint table" sub="prepare_worker once" />
        <Box t={t} x={693} y={328} w={185} h={54} title="RequestSink<R>" sub="HTTP · gRPC · DirectRequest" accent={blue} />
        <Edge d="M203,355 L243,355" color={line} marker="sched" />
        <Edge d="M423,355 L463,355" color={line} marker="sched" />
        <Edge d="M653,355 L693,355" color={line} marker="sched" />

        <Band t={t} x={12} y={438} w={906} h={68} label="Worker topology" accent={orange} />
        <Box t={t} x={95} y={452} w={270} h={38} title="workers = 1" sub="coordinator current-thread runtime" />
        <Box t={t} x={565} y={452} w={270} h={38} title="workers > 1" sub="OS threads · current_thread + LocalSet" accent={orange} />
        <Edge d="M458,420 C410,438 330,442 285,452" color={line} marker="sched" dashed />
        <Edge d="M568,420 C610,438 666,442 700,452" color={line} marker="sched" />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Same workload ID">Transport selection does not create separate HTTP, gRPC, and DynoSim workload registrations.</Callout>
        <Callout tone="neutral" title="Accuracy">Static accuracy is configuration on the scheduled path, with canonical Python evaluators behind a subprocess seam.</Callout>
        <Callout tone="success" title="Local hot path">Each worker co-locates scheduler, prepared endpoints, transport, and observers without per-token cross-thread locking.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Phase runtime", path: "rust/aiperf/src/phase_runtime.rs" },
        { label: "Scheduled bridge", path: "rust/aiperf/src/scheduled.rs" },
        { label: "Sharded workers", path: "rust/aiperf/src/runner_protocol/sharded_scheduled.rs" },
        { label: "Turn placement", path: "rust/aiperf/src/runner_protocol/turn_execution.rs" },
      ]} />
    </Stack>
  );
}

function GraphView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Trace datasets bypass the linear dataset loader: one graph resolver strictly decodes the source, compiles it
        into shared segments, then derives phase-specific programs for warmup and profiling.
      </Text>
      <svg viewBox="0 0 930 520" style={SVG_STYLE}>
        <defs><Arrow id="graph" color={line} /></defs>
        <Band t={t} x={12} y={12} w={906} h={98} label="Decode and compile once" accent={blue} />
        <Box t={t} x={34} y={38} w={185} h={48} title="trace source" sub="dag_jsonl · WEKA · Dynamo" />
        <Box t={t} x={264} y={34} w={190} h={56} title="GraphInputAdapterResolver" sub="identity selection + strict decode" accent={blue} />
        <Box t={t} x={499} y={34} w={170} h={56} title="compiler" sub="LCP trie + dense interning" />
        <Box t={t} x={714} y={34} w={180} h={56} title="GraphInputBundle" sub="program + SegmentStore" accent={green} />
        <Edge d="M219,62 L264,62" color={line} marker="graph" />
        <Edge d="M454,62 L499,62" color={line} marker="graph" />
        <Edge d="M669,62 L714,62" color={line} marker="graph" />

        <Band t={t} x={12} y={128} w={906} h={156} label="Derive phase programs" accent={purple} />
        <Box t={t} x={38} y={164} w={180} h={54} title="TStarSampler" sub="seeded trajectory start" accent={purple} />
        <Box t={t} x={268} y={158} w={175} h={66} title="warmup rewrite" sub="prime prefixes before the frontier" accent={orange} />
        <Box t={t} x={493} y={158} w={175} h={66} title="handoff frontier" sub="resume exactly once after warmup" />
        <Box t={t} x={718} y={158} w={175} h={66} title="profiling chop" sub="replay from sampled t*" accent={green} />
        <Edge d="M218,191 L268,191" color={line} marker="graph" />
        <Edge d="M443,191 L493,191" color={line} marker="graph" />
        <Edge d="M668,191 L718,191" color={line} marker="graph" />
        <text x={355} y={253} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>optional cache-pressure recycle</text>
        <text x={805} y={253} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>default window preserves full replay</text>

        <Band t={t} x={12} y={302} w={906} h={128} label="Execute graph" accent={green} />
        <Box t={t} x={38} y={338} w={165} h={54} title="graph policies" sub="root · arrival · admission · failure" />
        <Box t={t} x={243} y={338} w={175} h={54} title="graph executor" sub="firing gates + dependencies" accent={green} />
        <Box t={t} x={458} y={338} w={190} h={54} title="placement factory" sub="trace to worker-local sink" accent={purple} />
        <Box t={t} x={688} y={338} w={190} h={54} title="RequestSink<R>" sub="one dispatch per graph node" accent={blue} />
        <Edge d="M203,365 L243,365" color={line} marker="graph" />
        <Edge d="M418,365 L458,365" color={line} marker="graph" />
        <Edge d="M648,365 L688,365" color={line} marker="graph" />

        <Band t={t} x={12} y={448} w={906} h={58} label="Outputs" accent={orange} />
        <Box t={t} x={175} y={459} w={220} h={34} title="per-node CapturedRecord" />
        <Box t={t} x={535} y={459} w={220} h={34} title="phase metrics + warmup handoff" accent={orange} />
        <Edge d="M590,392 C520,430 370,434 310,459" color={line} marker="graph" />
        <Edge d="M765,392 C755,430 680,440 645,459" color={line} marker="graph" />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="One compiler">Python does not parse or lower WEKA/Dynamo graph inputs on the Rust path.</Callout>
        <Callout tone="neutral" title="Shared execution seam">Graph placement still ends at the same Clock and RequestSink/Observer interfaces as scheduled work.</Callout>
        <Callout tone="warning" title="Warmup failure">A terminal trajectory-warmup failure stops before profiling and returns a typed v2 failure.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Graph input", path: "rust/aiperf/src/runner_protocol/graph_input.rs" },
        { label: "Graph phases", path: "rust/aiperf/src/runner_protocol/graph_phase_runtime.rs" },
        { label: "Snapshot transforms", path: "rust/aiperf/src/graph/snapshot.rs" },
        { label: "Graph executor", path: "rust/aiperf/src/graph/executor.rs" },
      ]} />
    </Stack>
  );
}

function EndpointsView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Endpoint dialects own payload and response semantics. Validation resolves authored profiles once; each worker
        builds a dense prepared table so request dispatch avoids repeated registry and configuration work.
      </Text>
      <svg viewBox="0 0 930 500" style={SVG_STYLE}>
        <defs><Arrow id="ep" color={line} /></defs>
        <Band t={t} x={12} y={12} w={906} h={100} label="Startup validation" accent={blue} />
        <Box t={t} x={34} y={38} w={180} h={50} title="endpoint profiles" sub="id + model + raw config" />
        <Box t={t} x={259} y={34} w={180} h={58} title="EndpointRegistry" sub="factory lookup by EndpointId" accent={blue} />
        <Box t={t} x={484} y={34} w={180} h={58} title="strict validation" sub="raw → effective config" />
        <Box t={t} x={709} y={34} w={185} h={58} title="profile identities" sub="stable dense EndpointKey" accent={green} />
        <Edge d="M214,63 L259,63" color={line} marker="ep" />
        <Edge d="M439,63 L484,63" color={line} marker="ep" />
        <Edge d="M664,63 L709,63" color={line} marker="ep" />

        <Band t={t} x={12} y={130} w={906} h={112} label="Worker preparation" accent={green} />
        <Box t={t} x={70} y={162} w={210} h={54} title="PreparedEndpointTableFactory" sub="shared startup blueprint" />
        <Box t={t} x={360} y={158} w={210} h={62} title="prepare_worker()" sub="worker-local tokenizer + bindings" accent={green} />
        <Box t={t} x={650} y={162} w={210} h={54} title="PreparedEndpointTable" sub="dense lookup by EndpointKey" accent={purple} />
        <Edge d="M280,189 L360,189" color={line} marker="ep" />
        <Edge d="M570,189 L650,189" color={line} marker="ep" />

        <Band t={t} x={12} y={260} w={906} h={150} label="Per-turn request and response" accent={purple} />
        <Box t={t} x={34} y={304} w={155} h={56} title="PreparedTurn" sub="content + token counts" />
        <Box t={t} x={229} y={300} w={170} h={64} title="Endpoint dialect" sub="format payload · headers · parser" accent={purple} />
        <Box t={t} x={449} y={296} w={190} h={72} title="transport binding" sub="HTTP URI/body or gRPC tensors" accent={blue} />
        <Box t={t} x={689} y={300} w={195} h={64} title="observations" sub="tokens · usage · endpoint metrics" accent={orange} />
        <Edge d="M189,332 L229,332" color={line} marker="ep" />
        <Edge d="M399,332 L449,332" color={line} marker="ep" />
        <Edge d="M639,332 L689,332" color={line} marker="ep" />
        <Edge d="M786,364 C736,397 350,398 314,364" color={orange} marker="ep" dashed />

        <Band t={t} x={12} y={428} w={906} h={58} label="Dialect families" accent={orange} />
        <Box t={t} x={38} y={439} w={195} h={34} title="OpenAI + Anthropic" />
        <Box t={t} x={263} y={439} w={180} h={34} title="KServe HTTP/gRPC" />
        <Box t={t} x={473} y={439} w={180} h={34} title="NVIDIA Riva" />
        <Box t={t} x={683} y={439} w={195} h={34} title="vLLM + specialized" />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Open registry">New dialects register factories; the core transport does not gain an endpoint-type switch.</Callout>
        <Callout tone="neutral" title="Transport-native binding">HTTP and gRPC share endpoint identity but prepare different wire representations.</Callout>
        <Callout tone="success" title="Usage authority">Endpoint parsers reconcile provider usage and token classification before emitting observer facts.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Endpoint trait", path: "rust/aiperf/src/endpoints/endpoints.rs" },
        { label: "Endpoint registry", path: "rust/aiperf/src/endpoints/registry.rs" },
        { label: "HTTP preparation", path: "rust/aiperf/src/runner_protocol/turn_execution.rs" },
        { label: "gRPC binding", path: "rust/aiperf/src/transport_grpc/binding.rs" },
      ]} />
    </Stack>
  );
}

function MetricsView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Measurement is an event stream, not transport-specific reporting. Request callbacks feed local collectors and
        native metrics; side channels join later, and exporters consume the finalized report.
      </Text>
      <svg viewBox="0 0 940 540" style={SVG_STYLE}>
        <defs><Arrow id="met" color={line} /></defs>
        <Band t={t} x={12} y={12} w={916} h={92} label="Hot-path observations" accent={orange} />
        <Box t={t} x={35} y={36} w={190} h={48} title="RequestSink<R>" sub="transport completes request" />
        <Box t={t} x={280} y={32} w={240} h={56} title="RequestObserver callbacks" sub="arrival · admit · token · usage · terminal" accent={orange} />
        <Box t={t} x={575} y={32} w={160} h={56} title="ObserverTee" sub="preserve event order" />
        <Box t={t} x={790} y={36} w={105} h={48} title="records" sub="optional raw" accent={blue} />
        <Edge d="M225,60 L280,60" color={line} marker="met" />
        <Edge d="M520,60 L575,60" color={line} marker="met" />
        <Edge d="M735,60 L790,60" color={line} marker="met" dashed />

        <Band t={t} x={12} y={122} w={916} h={130} label="Worker-local accumulation" accent={green} />
        <Box t={t} x={80} y={158} w={210} h={58} title="CollectorObserver" sub="timing trace + request lifecycle" accent={blue} />
        <Box t={t} x={365} y={158} w={210} h={58} title="NativeMetricsObserver" sub="catalog RecordIngest facts" accent={green} />
        <Box t={t} x={650} y={158} w={210} h={58} title="storage policy" sub="exact retain or t-digest sketch" accent={purple} />
        <Edge d="M655,88 C600,125 275,130 185,158" color={line} marker="met" />
        <Edge d="M655,88 C610,125 510,132 470,158" color={line} marker="met" />
        <Edge d="M575,187 L650,187" color={line} marker="met" />

        <Band t={t} x={12} y={270} w={916} h={116} label="Post-drain reduction" accent={green} />
        <Box t={t} x={45} y={302} w={180} h={50} title="worker partitions" sub="plain data after callbacks stop" />
        <Box t={t} x={275} y={298} w={200} h={58} title="MetricsAccumulator" sub="merge stores + derived metrics" accent={green} />
        <Box t={t} x={525} y={298} w={170} h={58} title="side channels" sub="GPU · server · network" accent={purple} />
        <Box t={t} x={745} y={298} w={150} h={58} title="NativeReport" sub="typed schema v2" accent={orange} />
        <Edge d="M225,327 L275,327" color={line} marker="met" />
        <Edge d="M475,327 L745,327" color={line} marker="met" />
        <Edge d="M695,327 L745,327" color={line} marker="met" />

        <Band t={t} x={12} y={404} w={916} h={120} label="Persistence and fan-out" accent={blue} />
        <Box t={t} x={34} y={442} w={175} h={50} title="native-v2.json" sub="durable report commit" accent={orange} />
        <Box t={t} x={249} y={442} w={175} h={50} title="compat reports" sub="aiperf JSON + CSV + console" />
        <Box t={t} x={464} y={442} w={175} h={50} title="columnar artifacts" sub="records + server metrics" />
        <Box t={t} x={679} y={442} w={215} h={50} title="network exporters" sub="OTLP · MLflow · W&B" accent={blue} />
        <Edge d="M820,356 C760,405 175,399 121,442" color={line} marker="met" />
        <Edge d="M209,467 L249,467" color={line} marker="met" />
        <Edge d="M424,467 L464,467" color={line} marker="met" />
        <Edge d="M639,467 L679,467" color={line} marker="met" dashed />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Exact mode">Retained rows support raw records, timeslices, and byte-exact percentile reporting.</Callout>
        <Callout tone="neutral" title="Sketch mode">Rows are folded and dropped; counts and extrema stay exact while percentiles are approximate.</Callout>
        <Callout tone="warning" title="Separate artifact path">Per-record files are written at capture sites because those rows do not live in the finalized report.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Observer adapter", path: "rust/aiperf/src/metrics.rs" },
        { label: "Metrics core", path: "rust/aiperf/src/metrics_core/accumulator.rs" },
        { label: "Report commit", path: "rust/aiperf/src/report.rs" },
        { label: "Export registry", path: "rust/aiperf/src/export/mod.rs" },
      ]} />
    </Stack>
  );
}

function CellularView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        A request with <Code>cells &gt; 1</Code> promotes its execution child into a controller. Cells receive sliced
        envelopes over Velo, run the ordinary single-process engine, and return mergeable partitions.
      </Text>
      <svg viewBox="0 0 940 520" style={SVG_STYLE}>
        <defs><Arrow id="cells" color={line} /></defs>
        <Band t={t} x={12} y={12} w={916} h={84} label="Controller promotion" accent={orange} />
        <Box t={t} x={55} y={32} w={190} h={44} title="aiperf --execute" sub="detect cells > 1" />
        <Box t={t} x={330} y={28} w={230} h={52} title="cell launcher" sub="partition budgets + envelope" accent={orange} />
        <Box t={t} x={645} y={28} w={230} h={52} title="controller transport" sub="Velo endpoints + lifecycle" accent={purple} />
        <Edge d="M245,54 L330,54" color={line} marker="cells" />
        <Edge d="M560,54 L645,54" color={line} marker="cells" />

        <Band t={t} x={12} y={114} w={916} h={184} label="Cell execution" accent={green} />
        <Box t={t} x={45} y={154} w={165} h={54} title="aiperf --cell 0" sub="fetch sliced envelope" accent={green} />
        <Box t={t} x={45} y={224} w={165} h={54} title="aiperf --cell N" sub="fetch sliced envelope" accent={green} />
        <Box t={t} x={275} y={180} w={200} h={70} title="ordinary execute path" sub="prepare · phases · dispatch · metrics" accent={blue} />
        <Box t={t} x={540} y={154} w={170} h={54} title="records partition" sub="global-order merge input" />
        <Box t={t} x={540} y={224} w={170} h={54} title="folded store" sub="exact-fold or sketch input" accent={purple} />
        <Box t={t} x={775} y={180} w={120} h={70} title="heartbeats" sub="progress + health" accent={orange} />
        <Edge d="M210,181 L275,205" color={line} marker="cells" />
        <Edge d="M210,251 L275,225" color={line} marker="cells" />
        <Edge d="M475,205 L540,181" color={line} marker="cells" />
        <Edge d="M475,225 L540,251" color={line} marker="cells" />
        <Edge d="M710,181 L775,205" color={line} marker="cells" dashed />

        <Band t={t} x={12} y={316} w={916} h={112} label="Flat controller merge" accent={green} />
        <Box t={t} x={65} y={348} w={190} h={50} title="cell messages" sub="partitions + artifacts" />
        <Box t={t} x={330} y={344} w={210} h={58} title="hierarchy request" sub="refused before startup" accent={purple} />
        <Box t={t} x={615} y={344} w={245} h={58} title="controller merge" sub="global order or associative store merge" accent={green} />
        <Edge d="M255,373 L615,373" color={line} marker="cells" />

        <Band t={t} x={12} y={446} w={916} h={60} label="Single commit point" accent={orange} />
        <Box t={t} x={180} y={457} w={220} h={36} title="sidecars on primary cell only" />
        <Box t={t} x={540} y={457} w={220} h={36} title="final report + exporters" accent={orange} />
        <Edge d="M738,402 C720,435 680,446 650,457" color={line} marker="cells" />
      </svg>
      <Grid columns={4} gap={10}>
        <Callout tone="neutral" title="S1">Issuance authority assigns aggregate dispatch ordinals.</Callout>
        <Callout tone="neutral" title="S2">Records shards expose mergeable partitions.</Callout>
        <Callout tone="neutral" title="S3">Metrics heartbeats carry live snapshots.</Callout>
        <Callout tone="neutral" title="S4">Cell partitions define deterministic ownership.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Controller", path: "rust/runtime/src/engine/cellular_controller.rs" },
        { label: "Cell mode", path: "rust/runtime/src/engine/cellular_cell.rs" },
        { label: "Hierarchy refusal", path: "rust/runtime/src/engine/cellular_aggregator.rs" },
        { label: "Cellular seams", path: "rust/runtime/src/cellular/mod.rs" },
      ]} />
    </Stack>
  );
}

function FeaturesView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        The executable’s feature set defines the available implementation universe. The lean CLI remains
        sibling-free; optional features add persistence, scale-out, embedded Python, or Dynamo integration.
      </Text>
      <svg viewBox="0 0 940 520" style={SVG_STYLE}>
        <defs><Arrow id="feat" color={line} /></defs>
        <Band t={t} x={12} y={12} w={916} h={90} label="Base executable" accent={blue} />
        <Box t={t} x={52} y={34} w={225} h={48} title="aiperf-cli default = []" sub="lean entry point + execution child" accent={blue} />
        <Box t={t} x={355} y={34} w={225} h={48} title="runner-protocol" sub="always enabled on aiperf dependency" accent={green} />
        <Box t={t} x={658} y={34} w={225} h={48} title="base execution" sub="HTTP · gRPC · scheduled · graph" />
        <Edge d="M277,58 L355,58" color={line} marker="feat" />
        <Edge d="M580,58 L658,58" color={line} marker="feat" />

        <Band t={t} x={12} y={120} w={916} h={188} label="Orthogonal feature branches" accent={purple} />
        <Box t={t} x={35} y={154} w={190} h={56} title="parquet" sub="columnar datasets + artifacts" accent={orange} />
        <Box t={t} x={255} y={154} w={190} h={56} title="velo" sub="controller · cell · hierarchy refusal" accent={purple} />
        <Box t={t} x={475} y={154} w={190} h={56} title="dynosim" sub="Dynamo mocker sibling checkout" accent={green} />
        <Box t={t} x={695} y={154} w={190} h={56} title="pyo3-embed" sub="in-process Python delegation" accent={blue} />
        <Box t={t} x={145} y={240} w={190} h={46} title="search-pyo3" sub="scipy + optuna planners" />
        <Box t={t} x={375} y={240} w={190} h={46} title="dynamo-full" sub="router · ZMQ · KV · AIC" />
        <Box t={t} x={605} y={240} w={190} h={46} title="full" sub="dynosim + parquet + velo" accent={green} />
        <Edge d="M790,210 C700,228 330,220 240,240" color={line} marker="feat" dashed />
        <Edge d="M570,210 L470,240" color={line} marker="feat" dashed />
        <Edge d="M570,210 L700,240" color={line} marker="feat" />

        <Band t={t} x={12} y={326} w={916} h={98} label="Runtime capability result" accent={green} />
        <Box t={t} x={35} y={352} w={190} h={48} title="transport catalog" sub="http · grpc · optional dynosim" />
        <Box t={t} x={255} y={352} w={190} h={48} title="cell count policy" sub="cells > 1 requires velo" />
        <Box t={t} x={475} y={352} w={190} h={48} title="artifact policy" sub="Parquet requires parquet" />
        <Box t={t} x={695} y={352} w={190} h={48} title="delegation policy" sub="embedded or Python subprocess" />

        <Band t={t} x={12} y={442} w={916} h={64} label="Packaging transition" accent={orange} />
        <Box t={t} x={80} y={454} w={270} h={38} title="wheel entry point" sub="interned aiperf-native" accent={blue} />
        <Box t={t} x={590} y={454} w={270} h={38} title="legacy packaged runner" sub="still present; default CLI self re-execs" muted />
        <Edge d="M350,473 L590,473" color={line} marker="feat" dashed />
      </svg>
      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Fail closed">Authored transports, artifacts, or cells unavailable in the current image are rejected during validation.</Callout>
        <Callout tone="neutral" title="No runtime discovery">Capabilities describe statically linked factories; enabling a config value cannot load missing code.</Callout>
        <Callout tone="warning" title="DynoSim dependency">DynoSim and Dynamo-full builds require the sibling <Code>dynamo-aiperf-native</Code> checkout.</Callout>
      </Grid>
      <EvidenceButtons paths={[
        { label: "Executable features", path: "rust/cli/Cargo.toml" },
        { label: "Library features", path: "rust/aiperf/Cargo.toml" },
        { label: "Capability composition", path: "rust/aiperf/src/runner_protocol/application.rs" },
        { label: "Wheel bundling", path: "Makefile" },
      ]} />
    </Stack>
  );
}

function SeamsView() {
  const t = useHostTheme();
  const line = t.stroke.primary;
  const blue = t.category.blue;
  const green = t.category.green;
  const purple = t.category.purple;
  const orange = t.category.orange;

  return (
    <Stack gap={16}>
      <Text tone="secondary">
        The architecture stays open in two directions: compile-time product composition at startup, and
        transport/clock substitution on the execution path. Cellular mode scales around the same single-run core.
      </Text>
      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Compile-time extension universe</H2>
          <svg viewBox="0 0 450 350" style={SVG_STYLE}>
            <defs><Arrow id="ext" color={line} /></defs>
            <Box t={t} x={120} y={16} w={210} h={48} title="AIPerfExtension" sub="transactional registration" accent={blue} />
            <Box t={t} x={105} y={102} w={240} h={58} title="AIPerfRegistry" sub="frozen once per executable image" accent={green} />
            <Edge d="M225,64 L225,102" color={line} marker="ext" />
            <Box t={t} x={10} y={205} w={130} h={46} title="datasets" sub="loaders + samplers" />
            <Box t={t} x={160} y={205} w={130} h={46} title="endpoints" sub="body + response" />
            <Box t={t} x={310} y={205} w={130} h={46} title="exporters" sub="report sinks" />
            <Box t={t} x={85} y={286} w={130} h={46} title="transports" sub="HTTP · gRPC · DynoSim" />
            <Box t={t} x={235} y={286} w={130} h={46} title="workloads" sub="scheduled · graph…" />
            <Edge d="M170,160 L75,205" color={line} marker="ext" />
            <Edge d="M205,160 L225,205" color={line} marker="ext" />
            <Edge d="M270,160 L375,205" color={line} marker="ext" />
            <Edge d="M190,160 C150,216 150,250 150,286" color={line} marker="ext" />
            <Edge d="M260,160 C300,216 300,250 300,286" color={line} marker="ext" />
          </svg>
        </Stack>

        <Stack gap={10}>
          <H2>Execution substitution</H2>
          <svg viewBox="0 0 450 350" style={SVG_STYLE}>
            <defs><Arrow id="seam" color={line} /></defs>
            <Box t={t} x={105} y={16} w={240} h={48} title="Workload / graph executor" sub="transport-neutral orchestration" accent={green} />
            <Box t={t} x={25} y={104} w={180} h={52} title="Clock" sub="RealClock | SimClock" accent={purple} />
            <Box t={t} x={245} y={104} w={180} h={52} title="RequestSink<R>" sub="transport-native R" accent={purple} />
            <Edge d="M185,64 L115,104" color={line} marker="seam" />
            <Edge d="M265,64 L335,104" color={line} marker="seam" />
            <Box t={t} x={8} y={213} w={128} h={46} title="HTTP / SSE" sub="Hyper" accent={blue} />
            <Box t={t} x={161} y={213} w={128} h={46} title="gRPC" sub="Tonic" accent={blue} />
            <Box t={t} x={314} y={213} w={128} h={46} title="DynoSim" sub="DirectRequest" accent={green} />
            <Edge d="M300,156 L72,213" color={line} marker="seam" />
            <Edge d="M335,156 L225,213" color={line} marker="seam" />
            <Edge d="M370,156 L378,213" color={line} marker="seam" />
            <Box t={t} x={105} y={292} w={240} h={42} title="RequestObserver event stream" accent={orange} />
            <Edge d="M72,259 L150,292" color={line} marker="seam" />
            <Edge d="M225,259 L225,292" color={line} marker="seam" />
            <Edge d="M378,259 L300,292" color={line} marker="seam" />
          </svg>
        </Stack>
      </Grid>

      <Divider />
      <H2>Cellular scaling wraps the same run core</H2>
      <svg viewBox="0 0 900 210" style={SVG_STYLE}>
        <defs><Arrow id="cell" color={line} /></defs>
        <Box t={t} x={20} y={62} w={185} h={60} title="controller process" sub="slice budgets + distribute envelope" accent={orange} />
        <Box t={t} x={270} y={20} w={155} h={50} title="cell 0" sub="ordinary execute path" accent={green} />
        <Box t={t} x={270} y={80} w={155} h={50} title="cell 1" sub="ordinary execute path" accent={green} />
        <Box t={t} x={270} y={140} w={155} h={50} title="cell N" sub="ordinary execute path" accent={green} />
        <Box t={t} x={500} y={62} w={170} h={60} title="hierarchy request" sub="refused before startup" accent={purple} />
        <Box t={t} x={735} y={62} w={145} h={60} title="final report" sub="controller commit" accent={orange} />
        <Edge d="M205,84 L270,45" color={line} marker="cell" />
        <Edge d="M205,92 L270,105" color={line} marker="cell" />
        <Edge d="M205,100 L270,165" color={line} marker="cell" />
        <Edge d="M425,45 C560,45 650,75 735,92" color={line} marker="cell" />
        <Edge d="M425,105 L735,92" color={line} marker="cell" />
        <Edge d="M425,165 C560,165 650,112 735,92" color={line} marker="cell" />
      </svg>

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="No runtime plugin discovery">
          Extensions are statically linked and duplicate names fail registration transactionally.
        </Callout>
        <Callout tone="neutral" title="No pair matrix">
          Transport and workload registries are independent; workloads resolve an execution factory from the prepared transport.
        </Callout>
        <Callout tone="warning" title="Cellular gate">
          Cross-process cells use the opt-in <Code>velo</Code> feature. Lean builds preserve <Code>cells=1</Code> and reject larger runs.
        </Callout>
      </Grid>
      <EvidenceButtons
        paths={[
          { label: "Extension registry", path: "rust/aiperf/src/extensions/mod.rs" },
          { label: "Cell controller", path: "rust/aiperf/src/runner_protocol/cellular_controller.rs" },
          { label: "Observer implementation", path: "rust/loadgen-core/src/observer.rs" },
        ]}
      />
    </Stack>
  );
}

export default function RustAiperfArchitecture() {
  const [view, setView] = useCanvasState<View>("rust-aiperf.view", "system");

  return (
    <Stack gap={18} style={{ padding: 22, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>Rust AIPerf architecture</H1>
        <Text tone="secondary">
          Four zoom levels, from product boundaries to the hot-path seams. Grounded in the current workspace code and
          Cargo feature graph.
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {VIEWS.map((item) => (
          <Pill
            active={view === item.id}
            onClick={() => setView(item.id)}
            title={item.hint}
          >
            {item.label}
          </Pill>
        ))}
      </Row>

      <Card size="lg">
        <CardHeader trailing={<Pill size="sm" active>{VIEWS.find((item) => item.id === view)?.hint}</Pill>}>
          {VIEWS.find((item) => item.id === view)?.label}
        </CardHeader>
        <CardBody>
          {view === "system" && <SystemView />}
          {view === "processes" && <ProcessesView />}
          {view === "runtime" && <RuntimeView />}
          {view === "protocol" && <ProtocolView />}
          {view === "scheduled" && <ScheduledView />}
          {view === "graph" && <GraphView />}
          {view === "endpoints" && <EndpointsView />}
          {view === "metrics" && <MetricsView />}
          {view === "cellular" && <CellularView />}
          {view === "features" && <FeaturesView />}
          {view === "seams" && <SeamsView />}
        </CardBody>
      </Card>

      <Text size="small" tone="tertiary">
        Reading convention: solid edges are primary paths; dashed edges are optional, delegated, or feature-gated.
        Source buttons open the implementation files used to anchor each view.
      </Text>
    </Stack>
  );
}
