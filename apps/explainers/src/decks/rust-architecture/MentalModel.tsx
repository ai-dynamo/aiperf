import { Text, useHostTheme } from "../../core/ui";
import { SLIDES } from "./content";

const SCENE_LABELS = [
  "ONE BINARY",
  "WORKSPACE CRATES",
  "STARTUP ORDER",
  "COMMAND DISPATCH",
  "CONFIG V2",
  "SELF EXECUTE",
  "PROTOCOL V2",
  "REGISTRY BOOTSTRAP",
  "COORDINATOR",
  "CLOCK SEAM",
  "DATASET PIPELINE",
  "WORKLOAD + PHASES",
  "OBSERVER SEAM",
  "WORKER MODEL",
  "CELLULAR MODE",
  "METRICS + FEATURES",
] as const;

const SCENE_NOTES = [
  "cli is entry + engine",
  "cli → runtime → loadgen-core",
  "execute_mode before dispatch",
  "native profile/config/cell",
  "BenchmarkRun wire object",
  "spawn current_exe()",
  "Application::handle_v2",
  "frozen capabilities",
  "validate → prepare → execute",
  "RealClock / SimClock",
  "load → sample → materialize",
  "scheduled + phase_runtime",
  "RequestSink + Observer",
  "thread-per-core sub-cells",
  "controller + cells + Velo",
  "native-v2.json + exporters",
] as const;

const MOTION_ROUTES: readonly (string | null)[] = [
  "M170 190 H530",
  "M120 190 H580",
  "M90 190 H610",
  "M110 190 H590",
  "M130 190 H570",
  "M150 190 H550",
  "M170 190 H530",
  "M190 190 H510",
  "M210 190 H490",
  "M230 190 H470",
  "M250 190 H450",
  "M270 190 H430",
  "M290 190 H410",
  "M310 190 H390",
  "M330 190 H370",
  "M350 190 H350",
];


function MotionSignal({ slideIndex }: { slideIndex: number }) {
  const t = useHostTheme();
  const path = MOTION_ROUTES[slideIndex];
  if (!path) return null;
  const stops = [
    { x: 70 + slideIndex * 8, y: 150, w: 150, h: 80 },
    { x: 480 - slideIndex * 6, y: 150, w: 150, h: 80 },
  ];
  return (
    <g className="rust-arch-motion" aria-hidden="true">
      {stops.map((box, i) => (
        <rect
          key={i}
          x={box.x}
          y={box.y}
          width={box.w}
          height={box.h}
          rx={10}
          fill="none"
          stroke={i === 0 ? t.category.blue : t.category.green}
          strokeWidth={1.5}
          className="rust-arch-box-pulse"
          style={{ animationDelay: `${0.8 + i * 1.1}s` }}
        />
      ))}
      <circle r={5} fill={t.category.green}>
        <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" begin="0.8s" dur="2.2s" repeatCount="indefinite" />
        <animateMotion path={path} begin="0.8s" dur="2.2s" repeatCount="indefinite" />
      </circle>
    </g>
  );
}

function Box({
  x,
  y,
  width,
  height,
  title,
  detail,
  accent,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  title: string;
  detail: string;
  accent?: keyof ReturnType<typeof useHostTheme>["category"];
}) {
  const t = useHostTheme();
  const stroke = accent ? t.category[accent] : t.stroke.secondary;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx={10} fill={t.bg.elevated} stroke={stroke} strokeWidth={accent ? 1.8 : 1.3} />
      <text x={x + width / 2} y={y + 34} textAnchor="middle" fill={t.text.primary} fontSize={14} fontWeight={700}>
        {title}
      </text>
      <text x={x + width / 2} y={y + 58} textAnchor="middle" fill={t.text.secondary} fontSize={11}>
        {detail}
      </text>
    </g>
  );
}

export function MentalModel({ slideIndex }: { slideIndex: number; slide: unknown }) {
  const t = useHostTheme();
  const arrow = (d: string, color = t.category.green) => (
    <path d={d} fill="none" stroke={color} strokeWidth={2.2} markerEnd="url(#rust-green)" />
  );

  return (
    <div style={{ border: `1px solid ${t.stroke.secondary}`, borderRadius: 8, background: t.bg.editor }}>
      <svg className="rust-arch-live" viewBox="0 0 700 400" role="img" aria-label="Evolving Rust architecture diagram" style={{ display: "block", width: "100%" }}>
        <defs>
          <marker id="rust-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.green} />
          </marker>
          <marker id="rust-blue" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.blue} />
          </marker>
        </defs>
        <rect x={18} y={16} width={664} height={44} rx={8} fill={t.fill.quaternary} />
        <text x={38} y={43} fill={t.text.primary} fontSize={14} fontWeight={700}>
          {SCENE_LABELS[slideIndex]}
        </text>
        <text x={662} y={43} textAnchor="end" fill={t.text.secondary} fontSize={12}>
          {SCENE_NOTES[slideIndex]}
        </text>

        {slideIndex === 0 ? (
          <g>
            <Box x={250} y={130} width={200} height={96} title="aiperf binary" detail="cli/src/main.rs" accent="green" />
            <text x={350} y={270} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              public CLI on the outside · execution child on the inside
            </text>
          </g>
        ) : null}

        {slideIndex === 1 ? (
          <g>
            <Box x={60} y={130} width={150} height={96} title="aiperf-cli" detail="commands + execute" accent="blue" />
            {arrow("M210 178 H250")}
            <Box x={250} y={130} width={170} height={96} title="aiperf-runtime" detail="engine + transports" accent="green" />
            {arrow("M420 178 H460")}
            <Box x={460} y={130} width={180} height={96} title="loadgen-core" detail="observer seam" accent="purple" />
            <Box x={520} y={280} width={120} height={58} title="mock-server" detail="separate target" />
          </g>
        ) : null}

        {slideIndex === 2 ? (
          <g>
            {["logging", "execute_mode?", "dispatch::run"].map((label, i) => (
              <g key={label}>
                <Box x={70 + i * 190} y={130} width={150} height={96} title={label} detail={i === 1 ? "hidden argv" : "startup"} accent={i === 1 ? "yellow" : undefined} />
                {i < 2 ? arrow(`M${220 + i * 190} 178 H${250 + i * 190}`) : null}
              </g>
            ))}
          </g>
        ) : null}

        {slideIndex === 3 ? (
          <g>
            <Box x={70} y={120} width={220} height={110} title="Native commands" detail="profile · config · cell" accent="green" />
            <Box x={410} y={120} width={220} height={110} title="Delegated commands" detail="python -m aiperf" accent="orange" />
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              dispatch.rs chooses the path from argv[0]
            </text>
          </g>
        ) : null}

        {slideIndex === 4 ? (
          <g>
            <Box x={50} y={130} width={150} height={96} title="benchmark.yaml" detail="Config v2" />
            {arrow("M200 178 H250")}
            <Box x={250} y={130} width={170} height={96} title="yaml + load" detail="expand + validate" accent="blue" />
            {arrow("M420 178 H470")}
            <Box x={470} y={130} width={180} height={96} title="BenchmarkRun" detail="typed wire object" accent="green" />
          </g>
        ) : null}

        {slideIndex === 5 ? (
          <g>
            <Box x={70} y={130} width={170} height={96} title="profile parent" detail="operator process" accent="blue" />
            {arrow("M240 178 H290")}
            <Box x={290} y={130} width={170} height={96} title="spawn child" detail="current_exe()" accent="yellow" />
            {arrow("M460 178 H510")}
            <Box x={510} y={130} width={120} height={96} title="--execute" detail="stdio JSON" accent="green" />
          </g>
        ) : null}

        {slideIndex === 6 ? (
          <g>
            <Box x={90} y={130} width={150} height={96} title="Envelope v2" detail="protocol boundary" />
            {arrow("M240 178 H290")}
            <Box x={290} y={130} width={150} height={96} title="Application" detail="stock composition" accent="green" />
            {arrow("M440 178 H490")}
            <Box x={490} y={130} width={120} height={96} title="terminal" detail="JSONL out" accent="purple" />
          </g>
        ) : null}

        {slideIndex === 7 ? (
          <g>
            <Box x={250} y={90} width={200} height={70} title="AIPerfRegistry" detail="transactional build" accent="green" />
            {[["loaders", 80], ["samplers", 220], ["endpoints", 360], ["transports", 500]].map(([name, x]) => (
              <g key={name}>
                <path d={`M350 160 V190 H${Number(x) + 55} V220`} fill="none" stroke={t.category.green} strokeWidth={2} markerEnd="url(#rust-green)" />
                <Box x={Number(x)} y={220} width={110} height={70} title={String(name)} detail="registered" />
              </g>
            ))}
          </g>
        ) : null}

        {slideIndex === 8 ? (
          <g>
            {["validate", "prepare", "execute", "persist"].map((stage, i) => (
              <g key={stage}>
                <Box x={40 + i * 160} y={130} width={130} height={96} title={stage} detail="coordinator" accent={i === 2 ? "green" : undefined} />
                {i < 3 ? arrow(`M${170 + i * 160} 178 H${200 + i * 160}`) : null}
              </g>
            ))}
          </g>
        ) : null}

        {slideIndex === 9 ? (
          <g>
            <Box x={120} y={130} width={180} height={96} title="RealClock" detail="HTTP / gRPC online" accent="blue" />
            <Box x={400} y={130} width={180} height={96} title="SimClock" detail="dynosim + graph" accent="purple" />
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              scheduling, backoff, and measurement all route through Clock
            </text>
          </g>
        ) : null}

        {slideIndex === 10 ? (
          <g>
            {[
              ["load", 40],
              ["compose", 190],
              ["sample", 340],
              ["materialize", 490],
            ].map(([name, x], i) => (
              <g key={name}>
                <Box x={Number(x)} y={130} width={120} height={96} title={String(name)} detail="dataset" accent={i === 3 ? "green" : undefined} />
                {i < 3 ? arrow(`M${Number(x) + 120} 178 H${Number(x) + 150}`) : null}
              </g>
            ))}
          </g>
        ) : null}

        {slideIndex === 11 ? (
          <g>
            <Box x={80} y={120} width={180} height={110} title="Workload" detail="request-rate / graph" accent="blue" />
            {arrow("M260 175 H310")}
            <Box x={310} y={120} width={180} height={110} title="phase_runtime" detail="warmup · drain" accent="yellow" />
            {arrow("M490 175 H540")}
            <Box x={540} y={120} width={90} height={110} title="Turns" detail="dispatch" accent="green" />
          </g>
        ) : null}

        {slideIndex === 12 ? (
          <g>
            <Box x={70} y={130} width={150} height={96} title="TurnDispatcher" detail="scheduler side" accent="blue" />
            {arrow("M220 178 H270")}
            <Box x={270} y={130} width={150} height={96} title="RequestSink" detail="transport side" accent="green" />
            {arrow("M420 178 H470")}
            <Box x={470} y={130} width={160} height={96} title="RequestObserver" detail="measurement" accent="purple" />
          </g>
        ) : null}

        {slideIndex === 13 ? (
          <g>
            <Box x={250} y={90} width={200} height={70} title="Coordinator thread" detail="workers > 1" accent="yellow" />
            {[120, 290, 460].map((x, i) => (
              <g key={x}>
                <path d={`M350 160 V190 H${x + 55} V220`} fill="none" stroke={t.category.green} strokeWidth={2} markerEnd="url(#rust-green)" />
                <Box x={x} y={220} width={110} height={70} title={`Worker ${i}`} detail="sub-cell" accent="green" />
              </g>
            ))}
          </g>
        ) : null}

        {slideIndex === 14 ? (
          <g>
            <Box x={70} y={130} width={150} height={96} title="Controller" detail="rank 0" accent="yellow" />
            {arrow("M220 178 H270", t.category.green)}
            <Box x={270} y={130} width={150} height={96} title="Velo" detail="control plane" accent="green" />
            {arrow("M420 178 H470", t.category.green)}
            <Box x={470} y={130} width={160} height={96} title="Cells" detail="load + results" accent="blue" />
          </g>
        ) : null}

        {slideIndex === 15 ? (
          <g>
            <Box x={60} y={130} width={130} height={96} title="Workers" detail="local metrics" />
            {arrow("M190 178 H240")}
            <Box x={240} y={130} width={150} height={96} title="native-v2.json" detail="authoritative" accent="purple" />
            {arrow("M390 178 H440")}
            <Box x={440} y={130} width={200} height={96} title="Exporters" detail="JSON · CSV · Parquet" accent="green" />
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={12}>
              features: grpc · cellular · dynosim · parquet · pyo3-embed
            </text>
          </g>
        ) : null}

        <MotionSignal slideIndex={slideIndex} />
      </svg>
      <div style={{ padding: "12px 16px", borderTop: `1px solid ${t.stroke.tertiary}` }}>
        <Text tone="secondary" weight="medium">
          {SLIDES[slideIndex].caption}
        </Text>
      </div>
    </div>
  );
}
