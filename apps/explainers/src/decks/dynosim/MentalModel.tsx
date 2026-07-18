import { Text, useHostTheme } from "../../core/ui";
import type { SlideDefinition } from "../../core/types";
import { SLIDES } from "./content";

const SCENE_LABELS = [
  "NO SERVING SOCKET",
  "FEATURE GATE",
  "TRANSPORT → CLOCK",
  "REGISTRY ROUTE",
  "HOST + SINK",
  "OFFLINE · SIMCLOCK",
  "ONLINE · REALCLOCK",
  "CLOCK COMPARE",
  "DUAL QUEUES",
  "SIM PUMP LOOP",
  "CLOCK WINS TIES",
  "STEP BOUNDS",
  "DIRECT REQUEST",
  "OBSERVER PATH",
  "METRICS MERGE",
  "DELIVERY MODE",
  "DRAIN + STALL",
  "MODE RECAP",
] as const;

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
  accent?: "green" | "blue" | "purple" | "yellow" | "orange";
}) {
  const t = useHostTheme();
  const stroke = accent ? t.category[accent] : t.stroke.secondary;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={10}
        fill={t.bg.elevated}
        stroke={stroke}
        strokeWidth={accent ? 1.8 : 1.3}
      />
      <text
        x={x + width / 2}
        y={y + 34}
        textAnchor="middle"
        fill={t.text.primary}
        fontSize={14}
        fontWeight={700}
      >
        {title}
      </text>
      <text
        x={x + width / 2}
        y={y + 58}
        textAnchor="middle"
        fill={t.text.secondary}
        fontSize={11}
      >
        {detail}
      </text>
    </g>
  );
}

function PumpMotion({ path }: { path: string }) {
  const t = useHostTheme();
  return (
    <g className="deck-dynosim-motion" aria-hidden="true">
      <circle r={5} fill={t.category.purple}>
        <animate
          attributeName="opacity"
          values="0;1;1;0"
          keyTimes="0;0.08;0.9;1"
          begin="0.8s"
          dur="2.2s"
          repeatCount="indefinite"
        />
        <animateMotion path={path} begin="0.8s" dur="2.2s" repeatCount="indefinite" />
      </circle>
    </g>
  );
}

export function MentalModel({
  slideIndex,
}: {
  slideIndex: number;
  slide: SlideDefinition;
}) {
  const t = useHostTheme();
  const arrow = (d: string, color: string = t.category.green) => (
    <path d={d} fill="none" stroke={color} strokeWidth={2.2} markerEnd="url(#dyno-green)" />
  );

  return (
    <div
      style={{
        border: `1px solid ${t.stroke.secondary}`,
        borderRadius: 8,
        background: t.bg.editor,
      }}
    >
      <svg
        className="deck-dynosim-live"
        viewBox="0 0 700 400"
        role="img"
        aria-label="Dynosim architecture diagram"
        style={{ display: "block", width: "100%" }}
      >
        <defs>
          <marker id="dyno-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.green} />
          </marker>
          <marker id="dyno-purple" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.purple} />
          </marker>
        </defs>
        <rect x={18} y={16} width={664} height={44} rx={8} fill={t.fill.quaternary} />
        <text x={38} y={43} fill={t.text.primary} fontSize={14} fontWeight={700}>
          {SCENE_LABELS[slideIndex]}
        </text>
        <text x={662} y={43} textAnchor="end" fill={t.text.secondary} fontSize={12}>
          {SLIDES[slideIndex]?.caption}
        </text>

        {slideIndex === 0 ? (
          <g>
            <Box x={80} y={130} width={160} height={96} title="AIPerf workload" detail="schedule + observe" accent="green" />
            <Box x={280} y={130} width={160} height={96} title="DynosimSink" detail="in-process" accent="blue" />
            <Box x={480} y={130} width={140} height={96} title="SteppableReplay" detail="Dynamo mocker" accent="purple" />
            {arrow("M240 178 H280")}
            {arrow("M440 178 H480", t.category.purple)}
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              HTTP/gRPC socket omitted on the hot path
            </text>
          </g>
        ) : null}

        {slideIndex === 1 ? (
          <g>
            <Box x={120} y={120} width={200} height={110} title="Cargo feature" detail="dynosim" accent="yellow" />
            {arrow("M320 175 H380")}
            <Box x={380} y={120} width={200} height={110} title="Registry IDs" detail="offline · online" accent="green" />
          </g>
        ) : null}

        {slideIndex === 2 ? (
          <g>
            <Box x={60} y={130} width={150} height={96} title="BenchmarkRun" detail="Config v2" />
            {arrow("M210 178 H260")}
            <Box x={260} y={130} width={180} height={96} title="transport.type" detail="selects driver" accent="blue" />
            {arrow("M440 178 H490")}
            <Box x={490} y={130} width={150} height={96} title="Clock" detail="Sim or Real" accent="green" />
          </g>
        ) : null}

        {slideIndex === 3 ? (
          <g>
            <Box x={90} y={120} width={180} height={110} title="HTTP factory" detail="not used" />
            <Box x={430} y={120} width={180} height={110} title="Dynosim factory" detail="ExecutionSinkBuilder" accent="green" />
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              transport ID routes before Hyper/Tonic clients build
            </text>
          </g>
        ) : null}

        {slideIndex === 4 ? (
          <g>
            <Box x={70} y={130} width={150} height={96} title="EngineHost" detail="SteppableReplay" accent="purple" />
            {arrow("M220 178 H270")}
            <Box x={270} y={130} width={150} height={96} title="DynosimSink" detail="RequestSink" accent="green" />
            {arrow("M420 178 H470")}
            <Box x={470} y={130} width={160} height={96} title="RequestObserver" detail="metrics" accent="blue" />
          </g>
        ) : null}

        {slideIndex === 5 ? (
          <g>
            <Box x={250} y={120} width={200} height={110} title="dynosim_offline" detail="SimClock" accent="purple" />
            {arrow("M350 230 V270")}
            <Box x={250} y={270} width={200} height={70} title="drive_sim_with_source" detail="virtual pump" accent="green" />
          </g>
        ) : null}

        {slideIndex === 6 ? (
          <g>
            <Box x={250} y={120} width={200} height={110} title="dynosim_online" detail="RealClock" accent="blue" />
            {arrow("M350 230 V270")}
            <Box x={250} y={270} width={200} height={70} title="drive_real_with_source" detail="wall pump" accent="green" />
          </g>
        ) : null}

        {slideIndex === 7 ? (
          <g>
            <Box x={100} y={130} width={200} height={96} title="RealClock" detail="wall nanoseconds" accent="blue" />
            <Box x={400} y={130} width={200} height={96} title="SimClock" detail="virtual nanoseconds" accent="purple" />
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              Clock::is_virtual() selects reactor vs. simulation driver
            </text>
          </g>
        ) : null}

        {slideIndex === 8 ? (
          <g>
            <Box x={80} y={130} width={220} height={96} title="SimClock queue" detail="arrivals · phases" accent="blue" />
            <Box x={400} y={130} width={220} height={96} title="SimEventSource" detail="engine events" accent="purple" />
            <text x={350} y={280} textAnchor="middle" fill={t.category.green} fontSize={13} fontWeight={700}>
              min(clock, source) · clock wins ties
            </text>
          </g>
        ) : null}

        {slideIndex === 9 ? (
          <g>
            {["poll future", "idle?", "advance t", "repeat"].map((label, i) => (
              <g key={label}>
                <Box x={40 + i * 160} y={130} width={130} height={96} title={label} detail="sim pump" accent={i === 2 ? "purple" : undefined} />
                {i < 3 ? arrow(`M${170 + i * 160} 178 H${200 + i * 160}`) : null}
              </g>
            ))}
            <PumpMotion path="M105 178 H585" />
          </g>
        ) : null}

        {slideIndex === 10 ? (
          <g>
            <Box x={120} y={140} width={180} height={80} title="arrival @ t" detail="clock queue" accent="blue" />
            <Box x={400} y={140} width={180} height={80} title="engine pass @ t" detail="source queue" accent="purple" />
            <text x={350} y={260} textAnchor="middle" fill={t.category.green} fontSize={14} fontWeight={700}>
              clock fires first at equal t
            </text>
          </g>
        ) : null}

        {slideIndex === 11 ? (
          <g>
            <Box x={80} y={130} width={170} height={96} title="source.step" detail="[at, end_ns)" accent="purple" />
            {arrow("M250 178 H300")}
            <Box x={300} y={130} width={170} height={96} title="guards" detail="regression · overshoot" accent="yellow" />
            {arrow("M470 178 H520")}
            <Box x={520} y={130} width={100} height={96} title="advance" detail="SimClock" accent="green" />
          </g>
        ) : null}

        {slideIndex === 12 ? (
          <g>
            <Box x={70} y={130} width={150} height={96} title="Turn dispatch" detail="workload" accent="blue" />
            {arrow("M220 178 H270")}
            <Box x={270} y={130} width={150} height={96} title="DirectRequest" detail="submit" accent="green" />
            {arrow("M420 178 H470", t.category.purple)}
            <Box x={470} y={130} width={160} height={96} title="waiter mailbox" detail="completion" accent="purple" />
          </g>
        ) : null}

        {slideIndex === 13 ? (
          <g>
            <Box x={60} y={130} width={160} height={96} title="EngineEvent" detail="token · usage" accent="purple" />
            {arrow("M220 178 H280")}
            <Box x={280} y={130} width={160} height={96} title="DynosimSink" detail="translate" accent="green" />
            {arrow("M440 178 H500")}
            <Box x={500} y={130} width={140} height={96} title="Observer" detail="TTFT · terminal" accent="blue" />
          </g>
        ) : null}

        {slideIndex === 14 ? (
          <g>
            <Box x={80} y={130} width={150} height={96} title="Observer events" detail="per worker" />
            {arrow("M230 178 H280")}
            <Box x={280} y={130} width={170} height={96} title="metrics_core" detail="accumulate" accent="green" />
            {arrow("M450 178 H500")}
            <Box x={500} y={130} width={120} height={96} title="records" detail="exact / sketch" accent="purple" />
          </g>
        ) : null}

        {slideIndex === 15 ? (
          <g>
            <Box x={100} y={130} width={220} height={96} title="Incremental" detail="token → token" accent="green" />
            <Box x={380} y={130} width={220} height={96} title="Coalesced" detail="terminal batch" accent="yellow" />
          </g>
        ) : null}

        {slideIndex === 16 ? (
          <g>
            <Box x={120} y={120} width={180} height={110} title="Ready future" detail="drain source" accent="green" />
            <Box x={400} y={120} width={180} height={110} title="No progress" detail="stall / deadlock" accent="orange" />
          </g>
        ) : null}

        {slideIndex === 17 ? (
          <g>
            <Box x={60} y={130} width={150} height={96} title="Offline" detail="SimClock" accent="purple" />
            <Box x={275} y={130} width={150} height={96} title="Online" detail="RealClock" accent="blue" />
            {arrow("M210 178 H275")}
            {arrow("M425 178 H490")}
            <Box x={490} y={130} width={150} height={96} title="native-v2.json" detail="same persist" accent="green" />
          </g>
        ) : null}
      </svg>
      <div style={{ padding: "12px 16px", borderTop: `1px solid ${t.stroke.tertiary}` }}>
        <Text tone="secondary" weight="medium">
          {SLIDES[slideIndex].caption}
        </Text>
      </div>
    </div>
  );
}
