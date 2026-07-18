import { FlowArrow } from "../../core/diagram/FlowArrow";
import { MotionSignal } from "../../core/diagram/MotionSignal";
import { SceneBox } from "../../core/diagram/SceneBox";
import { useHostTheme } from "../../core/ui";
import type { SlideDefinition } from "../../core/types";
import { SLIDES } from "./content";

const SCENE_LABELS = [
  "SYSTEM LANDSCAPE",
  "PROCESS / CRATE GRAPH",
  "ONE-RUN HOT PATH",
  "PROTOCOL V2 CHILD",
  "SCHEDULED WORKLOAD",
  "GRAPH TRACE PATH",
  "ENDPOINT DIALECTS",
  "METRICS PLANE",
  "CELLULAR SCALE-OUT",
  "FEATURE BUILDS",
  "EXTENSION SEAMS",
] as const;

const MOTION: readonly (string | null)[] = [
  "M350 120 V200",
  "M140 178 H560",
  "M90 178 H610",
  "M100 120 H600",
  "M80 160 H620",
  "M90 160 H610",
  "M100 160 H600",
  "M80 160 H620",
  "M120 160 H580",
  "M140 160 H560",
  "M200 100 V280",
];

function Markers({ green, blue, purple }: { green: string; blue: string; purple: string }) {
  return (
    <defs>
      <marker id="atlas-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={green} />
      </marker>
      <marker id="atlas-blue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={blue} />
      </marker>
      <marker id="atlas-purple" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={purple} />
      </marker>
    </defs>
  );
}

function Scene({ slideIndex }: { slideIndex: number }) {
  const t = useHostTheme();
  const green = "atlas-green";
  const blue = "atlas-blue";
  const purple = "atlas-purple";

  switch (slideIndex) {
    case 0:
      return (
        <g>
          <SceneBox x={60} y={90} width={160} height={70} title="Config v2" detail="cli flags" accent="blue" />
          <FlowArrow d="M220 125 H270" markerId={blue} color={t.category.blue} />
          <SceneBox x={270} y={80} width={180} height={90} title="aiperf" detail="cli/src/main.rs" accent="blue" />
          <FlowArrow d="M360 170 V210" markerId={green} />
          <SceneBox x={250} y={210} width={220} height={80} title="aiperf --execute" detail="protocol v2 child" accent="green" />
          <FlowArrow d="M470 250 H520" markerId={green} color={t.category.orange} />
          <SceneBox x={520} y={210} width={140} height={80} title="--cell N" detail="optional" accent="orange" />
          <SceneBox x={60} y={310} width={150} height={60} title="real server" detail="HTTP / gRPC" />
          <SceneBox x={250} y={310} width={170} height={60} title="mock-server" detail="standalone" accent="purple" />
          <SceneBox x={460} y={310} width={180} height={60} title="Dynosim" detail="feature gate" accent="green" />
        </g>
      );
    case 1:
      return (
        <g>
          <SceneBox x={40} y={100} width={160} height={90} title="aiperf-cli" detail="binary + commands" accent="blue" />
          <FlowArrow d="M200 145 H250" markerId={green} />
          <SceneBox x={250} y={100} width={180} height={90} title="aiperf-runtime" detail="engine + transports" accent="green" />
          <FlowArrow d="M430 145 H480" markerId={purple} color={t.category.purple} />
          <SceneBox x={480} y={100} width={180} height={90} title="loadgen-core" detail="sink · observer" accent="purple" />
          <SceneBox x={250} y={240} width={160} height={70} title="mock-server" detail="→ runtime" />
          <SceneBox x={450} y={240} width={140} height={70} title="pyext / e2e" detail="off hot path" />
        </g>
      );
    case 2:
      return (
        <g>
          <SceneBox x={30} y={90} width={140} height={70} title="Application" detail="freeze registry" accent="blue" />
          <FlowArrow d="M170 125 H210" markerId={green} />
          <SceneBox x={210} y={90} width={140} height={70} title="Coordinator" detail="validate/prepare" accent="green" />
          <FlowArrow d="M350 125 H390" markerId={green} />
          <SceneBox x={390} y={90} width={140} height={70} title="Phase runtime" detail="warmup→profile" accent="orange" />
          <FlowArrow d="M530 125 H570" markerId={purple} color={t.category.purple} />
          <SceneBox x={570} y={90} width={110} height={70} title="Sink" detail="dispatch" accent="purple" />
          <SceneBox x={120} y={220} width={150} height={70} title="Clock" detail="Real / Sim" accent="purple" />
          <SceneBox x={320} y={220} width={160} height={70} title="Observer" detail="token · usage" accent="orange" />
          <SceneBox x={520} y={220} width={150} height={70} title="native-v2" detail="report commit" accent="green" />
        </g>
      );
    case 3:
      return (
        <g>
          <SceneBox x={40} y={90} width={150} height={70} title="profile" detail="parent shell" />
          <FlowArrow d="M190 125 H230" markerId={blue} color={t.category.blue} />
          <SceneBox x={230} y={90} width={160} height={70} title="RunnerRequest" detail="protocol v2" accent="blue" />
          <FlowArrow d="M390 125 H430" markerId={green} />
          <SceneBox x={430} y={90} width={140} height={70} title="exec_bin" detail="current_exe" />
          <FlowArrow d="M570 125 H610" markerId={green} />
          <SceneBox x={610} y={90} width={70} height={70} title="spawn" detail="--execute" accent="green" />
          <SceneBox x={80} y={210} width={140} height={70} title="stdin JSON" detail="to EOF" accent="purple" />
          <SceneBox x={280} y={210} width={160} height={70} title="Application" detail="frozen catalog" accent="green" />
          <SceneBox x={500} y={210} width={160} height={70} title="stdout JSONL" detail="one terminal" accent="orange" />
        </g>
      );
    case 4:
      return (
        <g>
          <SceneBox x={30} y={90} width={130} height={70} title="dataset" detail="loader" accent="blue" />
          <FlowArrow d="M160 125 H200" markerId={green} />
          <SceneBox x={200} y={90} width={140} height={70} title="PhaseOrchestrator" detail="lifecycle" accent="orange" />
          <FlowArrow d="M340 125 H380" markerId={green} />
          <SceneBox x={380} y={90} width={140} height={70} title="SlotPool" detail="admission" accent="purple" />
          <FlowArrow d="M520 125 H560" markerId={green} />
          <SceneBox x={560} y={90} width={120} height={70} title="TurnDispatcher" detail="place turn" accent="green" />
          <SceneBox x={120} y={220} width={180} height={70} title="workers=1" detail="co-located LocalSet" />
          <SceneBox x={400} y={220} width={200} height={70} title="workers>1" detail="OS thread sub-cells" accent="orange" />
        </g>
      );
    case 5:
      return (
        <g>
          <SceneBox x={30} y={90} width={140} height={70} title="trace source" detail="dag · WEKA · Dynamo" />
          <FlowArrow d="M170 125 H210" markerId={blue} color={t.category.blue} />
          <SceneBox x={210} y={90} width={160} height={70} title="graph_input" detail="strict decode" accent="blue" />
          <FlowArrow d="M370 125 H410" markerId={green} />
          <SceneBox x={410} y={90} width={150} height={70} title="compiler" detail="LCP + intern" />
          <FlowArrow d="M560 125 H600" markerId={green} />
          <SceneBox x={600} y={90} width={80} height={70} title="bundle" detail="program" accent="green" />
          <SceneBox x={80} y={210} width={150} height={70} title="warmup rewrite" detail="prime prefixes" accent="orange" />
          <SceneBox x={280} y={210} width={150} height={70} title="profiling chop" detail="from t*" accent="green" />
          <SceneBox x={480} y={210} width={160} height={70} title="graph executor" detail="RequestSink" accent="purple" />
        </g>
      );
    case 6:
      return (
        <g>
          <SceneBox x={40} y={90} width={150} height={70} title="profiles" detail="EndpointId" />
          <FlowArrow d="M190 125 H230" markerId={blue} color={t.category.blue} />
          <SceneBox x={230} y={90} width={160} height={70} title="EndpointRegistry" detail="factory lookup" accent="blue" />
          <FlowArrow d="M390 125 H430" markerId={green} />
          <SceneBox x={430} y={90} width={160} height={70} title="prepare_worker" detail="dense table" accent="green" />
          <FlowArrow d="M590 125 H620" markerId={purple} color={t.category.purple} />
          <SceneBox x={620} y={90} width={60} height={70} title="key" detail="table" accent="purple" />
          <SceneBox x={80} y={220} width={150} height={70} title="dialect" detail="BodyPlan" accent="purple" />
          <SceneBox x={280} y={220} width={160} height={70} title="HTTP / gRPC bind" detail="URI · tensors" accent="blue" />
          <SceneBox x={500} y={220} width={150} height={70} title="observations" detail="usage · tokens" accent="orange" />
        </g>
      );
    case 7:
      return (
        <g>
          <SceneBox x={40} y={90} width={140} height={70} title="RequestSink" detail="complete request" />
          <FlowArrow d="M180 125 H220" markerId={green} color={t.category.orange} />
          <SceneBox x={220} y={90} width={180} height={70} title="RequestObserver" detail="arrival→terminal" accent="orange" />
          <FlowArrow d="M400 125 H440" markerId={green} />
          <SceneBox x={440} y={90} width={140} height={70} title="worker store" detail="exact / sketch" accent="purple" />
          <FlowArrow d="M580 125 H610" markerId={green} />
          <SceneBox x={610} y={90} width={70} height={70} title="merge" detail="drain" accent="green" />
          <SceneBox x={100} y={220} width={150} height={70} title="native-v2.json" detail="commit" accent="orange" />
          <SceneBox x={300} y={220} width={150} height={70} title="compat exports" detail="JSON · CSV" />
          <SceneBox x={500} y={220} width={160} height={70} title="OTLP · MLflow" detail="network sinks" accent="blue" />
        </g>
      );
    case 8:
      return (
        <g>
          <SceneBox x={40} y={90} width={160} height={70} title="--execute" detail="cells > 1" />
          <FlowArrow d="M200 125 H240" markerId={green} color={t.category.orange} />
          <SceneBox x={240} y={90} width={170} height={70} title="controller" detail="partition + Velo" accent="orange" />
          <FlowArrow d="M410 125 H450" markerId={purple} color={t.category.purple} />
          <SceneBox x={450} y={90} width={140} height={70} title="cell launcher" detail="sliced envelope" accent="purple" />
          <SceneBox x={60} y={210} width={130} height={70} title="--cell 0" detail="ordinary exec" accent="green" />
          <SceneBox x={230} y={210} width={130} height={70} title="--cell N" detail="ordinary exec" accent="green" />
          <SceneBox x={400} y={210} width={140} height={70} title="aggregators" detail="optional" accent="purple" />
          <SceneBox x={560} y={210} width={120} height={70} title="merge" detail="one commit" accent="orange" />
        </g>
      );
    case 9:
      return (
        <g>
          <SceneBox x={40} y={90} width={180} height={70} title="default" detail="grpc + cellular" accent="blue" />
          <SceneBox x={260} y={90} width={130} height={70} title="parquet" detail="columnar" accent="orange" />
          <SceneBox x={420} y={90} width={130} height={70} title="dynosim" detail="Dynamo mocker" accent="green" />
          <SceneBox x={580} y={90} width={100} height={70} title="pyo3" detail="embed" accent="blue" />
          <SceneBox x={120} y={220} width={160} height={70} title="search-pyo3" detail="planners" />
          <SceneBox x={340} y={220} width={160} height={70} title="full" detail="dynosim+parquet+…" accent="green" />
          <SceneBox x={540} y={220} width={120} height={70} title="fail closed" detail="validate" accent="orange" />
        </g>
      );
    case 10:
      return (
        <g>
          <SceneBox x={240} y={80} width={200} height={60} title="AIPerfExtension" detail="transactional register" accent="blue" />
          <FlowArrow d="M340 140 V170" markerId={green} />
          <SceneBox x={230} y={170} width={220} height={60} title="AIPerfRegistry" detail="frozen once" accent="green" />
          <SceneBox x={40} y={270} width={110} height={60} title="datasets" detail="loaders" />
          <SceneBox x={170} y={270} width={110} height={60} title="endpoints" detail="dialects" />
          <SceneBox x={300} y={270} width={110} height={60} title="transports" detail="HTTP·gRPC" accent="purple" />
          <SceneBox x={430} y={270} width={110} height={60} title="workloads" detail="sched·graph" />
          <SceneBox x={560} y={270} width={110} height={60} title="exporters" detail="sinks" />
        </g>
      );
    default:
      return null;
  }
}

export function MentalModel({
  slideIndex,
}: {
  slideIndex: number;
  slide: SlideDefinition;
}) {
  const t = useHostTheme();
  const motion = MOTION[slideIndex];

  return (
    <div
      style={{
        border: `1px solid ${t.stroke.secondary}`,
        borderRadius: 8,
        background: t.bg.editor,
      }}
    >
      <svg
        className="deck-rust-atlas-live"
        viewBox="0 0 700 400"
        role="img"
        aria-label="Rust architecture atlas diagram"
        style={{ display: "block", width: "100%" }}
      >
        <Markers green={t.category.green} blue={t.category.blue} purple={t.category.purple} />
        <rect x={18} y={16} width={664} height={44} rx={8} fill={t.fill.quaternary} />
        <text x={38} y={43} fill={t.text.primary} fontSize={14} fontWeight={700}>
          {SCENE_LABELS[slideIndex]}
        </text>
        <text x={662} y={43} textAnchor="end" fill={t.text.secondary} fontSize={12}>
          {SLIDES[slideIndex]?.caption}
        </text>
        <Scene slideIndex={slideIndex} />
        {motion ? (
          <g className="deck-rust-atlas-motion" aria-hidden="true">
            <MotionSignal path={motion} delay="0.8s" />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
