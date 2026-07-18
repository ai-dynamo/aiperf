import { FlowArrow } from "../../core/diagram/FlowArrow";
import { MotionSignal } from "../../core/diagram/MotionSignal";
import { SceneBox } from "../../core/diagram/SceneBox";
import { useHostTheme } from "../../core/ui";
import type { SlideDefinition } from "../../core/types";
import { SLIDES } from "./content";

const SCENE_LABELS = [
  "LAUNCH · ONE RUN, MANY CELLS",
  "LAUNCH · CONFIG V2 SOURCE",
  "LAUNCH · SELF-EXEC BOUNDARY",
  "LAUNCH · CONTROLLER PROMOTION",
  "LAUNCH · VALIDATE / FAIL CLOSED",
  "DISTRIBUTE · MODULO PARTITION",
  "DISTRIBUTE · START BARRIER",
  "DISTRIBUTE · START POLICIES",
  "DISTRIBUTE · DATASET DELIVERY",
  "DISTRIBUTE · OWNERSHIP INDEX",
  "EXECUTE · AUTONOMOUS CELL",
  "EXECUTE · WORKER SHARDS",
  "EXECUTE · GLOBAL ORDINAL",
  "EXECUTE · ONLINE DISPATCH",
  "EXECUTE · CAPTURED RECORD",
  "REDUCE · RETAIN ROWS",
  "REDUCE · EXACT FOLD",
  "REDUCE · SKETCH",
  "SCALE · MERGE & PUBLISH",
  "SCALE · FULL SYSTEM ATLAS",
] as const;

// green = request/load (execution), purple = results, yellow = coordination (control), blue = timing/runtime (data).
const MOTION: readonly (string | null)[] = [
  "M180 175 H560",
  null,
  "M110 175 H600",
  "M180 160 H560",
  null,
  "M110 175 H580",
  "M350 300 V160",
  null,
  "M180 175 H560",
  "M110 175 H580",
  "M110 175 H560",
  "M190 175 H540",
  null,
  "M110 175 H560",
  "M110 175 H600",
  "M110 175 H560",
  "M110 175 H560",
  "M110 175 H560",
  "M110 175 H580",
  null,
];

function Markers({
  green,
  blue,
  purple,
  yellow,
}: {
  green: string;
  blue: string;
  purple: string;
  yellow: string;
}) {
  return (
    <defs>
      <marker id="cell-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={green} />
      </marker>
      <marker id="cell-blue" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={blue} />
      </marker>
      <marker id="cell-purple" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={purple} />
      </marker>
      <marker id="cell-yellow" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={yellow} />
      </marker>
    </defs>
  );
}

function Scene({ slideIndex }: { slideIndex: number }) {
  const t = useHostTheme();
  const green = "cell-green";
  const blue = "cell-blue";
  const purple = "cell-purple";
  const yellow = "cell-yellow";

  switch (slideIndex) {
    case 0:
      return (
        <g>
          <SceneBox x={40} y={150} width={160} height={80} title="Authored run" detail="one identity" accent="yellow" />
          <FlowArrow d="M200 190 H280" markerId={green} color={t.category.green} />
          <SceneBox x={280} y={150} width={150} height={80} title="Cell k / N" detail="autonomous" accent="green" />
          <FlowArrow d="M430 190 H510" markerId={purple} color={t.category.purple} />
          <SceneBox x={510} y={150} width={150} height={80} title="One report" detail="authoritative" accent="purple" />
          <text x={120} y={300} fill={t.text.tertiary} fontSize={12}>
            placement changes · measurement contract does not
          </text>
        </g>
      );
    case 1:
      return (
        <g>
          <SceneBox x={230} y={120} width={240} height={90} title="Config v2" detail="ProfileFlags → BenchmarkRun" accent="yellow" />
          {[0, 1, 2, 3].map((i) => (
            <SceneBox key={i} x={40 + i * 160} y={280} width={130} height={60} title={`cell ${i}`} detail="same request" accent="green" />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M350 210 C${180 + i * 120} 240 ${105 + i * 160} 250 ${105 + i * 160} 280`} markerId={yellow} color={t.category.yellow} />
          ))}
        </g>
      );
    case 2:
      return (
        <g>
          <SceneBox x={30} y={150} width={140} height={80} title="profile" detail="parent shell" accent="yellow" />
          <FlowArrow d="M170 190 H220" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={220} y={150} width={140} height={80} title="exec_bin" detail="current_exe()" accent="yellow" />
          <FlowArrow d="M360 190 H410" markerId={green} color={t.category.green} />
          <SceneBox x={410} y={150} width={150} height={80} title="--execute" detail="protocol-v2 child" accent="green" />
          <SceneBox x={230} y={290} width={200} height={56} title="stdin: one envelope" detail="stdout: one terminal" accent="blue" />
          <FlowArrow d="M485 230 V290 H430" markerId={blue} color={t.category.blue} />
        </g>
      );
    case 3:
      return (
        <g>
          <SceneBox x={30} y={150} width={150} height={80} title="--execute" detail="cells > 1" accent="green" />
          <FlowArrow d="M180 190 H240" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={240} y={150} width={180} height={80} title="Controller" detail="coordinate · no load" accent="yellow" />
          <FlowArrow d="M420 175 C480 120 560 120 610 140" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={480} y={280} width={190} height={60} title="controller / cell / aggregator" detail="native K8s roles" accent="yellow" />
          <SceneBox x={60} y={280} width={140} height={60} title="single cell" detail="stays direct" accent="green" />
        </g>
      );
    case 4:
      return (
        <g>
          <SceneBox x={250} y={110} width={200} height={80} title="Controller" detail="validate_cellular_run_shape" accent="yellow" />
          <SceneBox x={40} y={270} width={170} height={70} title="scheduled infinite" detail="rejected" accent="red" />
          <SceneBox x={260} y={270} width={170} height={70} title="DynoSim cellular" detail="rejected" accent="red" />
          <SceneBox x={480} y={270} width={180} height={70} title="eligible shape" detail="launch" accent="green" />
          <FlowArrow d="M340 190 C240 220 180 240 150 270" markerId={yellow} color={t.category.red} />
          <FlowArrow d="M360 190 V270" markerId={yellow} color={t.category.red} />
          <FlowArrow d="M400 190 C520 220 560 240 570 270" markerId={green} color={t.category.green} />
        </g>
      );
    case 5:
      return (
        <g>
          <SceneBox x={30} y={150} width={150} height={80} title="Controller" detail="global budget" accent="yellow" />
          <FlowArrow d="M180 190 H230" markerId={blue} color={t.category.blue} />
          <SceneBox x={230} y={150} width={160} height={80} title="ModuloCellPartition" detail="i % cells == id" accent="blue" />
          <FlowArrow d="M390 190 H440" markerId={green} color={t.category.green} />
          {[0, 1, 2].map((i) => (
            <SceneBox key={i} x={440} y={100 + i * 70} width={220} height={54} title={`cell ${i} owns`} detail={`i % 3 == ${i}`} accent="green" />
          ))}
          {[0, 1, 2].map((i) => (
            <FlowArrow key={i} d={`M470 190 C480 ${127 + i * 70} 490 ${127 + i * 70} 440 ${127 + i * 70}`} markerId={green} color={t.category.green} />
          ))}
        </g>
      );
    case 6:
      return (
        <g>
          {[0, 1, 2, 3].map((i) => (
            <SceneBox key={i} x={30 + i * 100} y={280} width={80} height={60} title={`c${i}`} detail="register" accent="green" />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M${70 + i * 100} 280 C${70 + i * 100} 240 350 220 440 195`} markerId={yellow} color={t.category.yellow} />
          ))}
          <SceneBox x={440} y={155} width={190} height={80} title="Barrier" detail="await_all → trigger" accent="yellow" />
          <SceneBox x={200} y={110} width={220} height={52} title="START" detail="synchronized dispatch" accent="yellow" />
          <FlowArrow d="M530 155 C500 125 450 130 420 136" markerId={yellow} color={t.category.yellow} />
        </g>
      );
    case 7:
      return (
        <g>
          <SceneBox x={40} y={110} width={180} height={64} title="synchronized" detail="default barrier" accent="yellow" />
          <SceneBox x={40} y={190} width={180} height={64} title="phaser" detail="opt-in generations" accent="yellow" />
          <SceneBox x={40} y={270} width={180} height={64} title="barrier-free" detail="k6-class" accent="yellow" />
          <SceneBox x={400} y={150} width={260} height={80} title="shared timing origin" detail="zero at barrier · opt-in" accent="blue" />
          <FlowArrow d="M220 142 C320 150 340 170 400 180" markerId={blue} color={t.category.blue} />
          <FlowArrow d="M220 302 C320 280 340 220 400 205" markerId={blue} color={t.category.blue} />
        </g>
      );
    case 8:
      return (
        <g>
          <SceneBox x={30} y={150} width={150} height={80} title="Ownership" detail="request-id chunks" accent="blue" />
          <FlowArrow d="M180 175 C240 130 300 130 350 145" markerId={blue} color={t.category.blue} />
          <FlowArrow d="M180 205 C240 250 300 250 350 235" markerId={blue} color={t.category.blue} />
          <SceneBox x={350} y={110} width={200} height={70} title="Velo overlay" detail="fan-out verify" accent="blue" />
          <SceneBox x={350} y={210} width={200} height={70} title="Stage G serve" detail="HTTP + zstd" accent="blue" />
          <FlowArrow d="M550 145 H610" markerId={green} color={t.category.green} />
          <FlowArrow d="M550 245 H610" markerId={green} color={t.category.green} />
          <SceneBox x={575} y={160} width={110} height={70} title="cell" detail="owned only" accent="green" />
        </g>
      );
    case 9:
      return (
        <g>
          <SceneBox x={40} y={150} width={160} height={80} title="DatasetIndex" detail="cell-local" accent="blue" />
          <FlowArrow d="M200 190 H260" markerId={blue} color={t.category.blue} />
          <SceneBox x={260} y={100} width={150} height={64} title="Indexed" detail="owned ids" accent="blue" />
          <SceneBox x={260} y={180} width={150} height={64} title="InFlight" detail="dispatched" accent="green" />
          <SceneBox x={260} y={260} width={150} height={64} title="Completed" detail="captured" accent="purple" />
          <text x={450} y={195} fill={t.text.secondary} fontSize={13}>
            one identity → exactly one cell
          </text>
        </g>
      );
    case 10:
      return (
        <g>
          <SceneBox x={30} y={90} width={150} height={56} title="sliced envelope" detail="control" accent="yellow" />
          <SceneBox x={30} y={172} width={150} height={56} title="owned inputs" detail="data" accent="blue" />
          <SceneBox x={30} y={254} width={150} height={56} title="shared origin" detail="timing" accent="blue" />
          <FlowArrow d="M180 118 C240 150 260 170 300 185" markerId={green} color={t.category.green} />
          <FlowArrow d="M180 200 H300" markerId={green} color={t.category.green} />
          <FlowArrow d="M180 282 C240 250 260 220 300 205" markerId={green} color={t.category.green} />
          <SceneBox x={300} y={150} width={200} height={90} title="Autonomous cell" detail="own runtime + metrics" accent="green" />
          <text x={330} y={300} fill={t.text.tertiary} fontSize={12}>
            no hot-path collector lock
          </text>
        </g>
      );
    case 11:
      return (
        <g>
          <SceneBox x={40} y={150} width={160} height={80} title="Cell slice" detail="owned positions" accent="green" />
          <FlowArrow d="M200 190 H260" markerId={green} color={t.category.green} />
          <SceneBox x={260} y={100} width={160} height={64} title="worker 0" detail="local 0,2,…" accent="green" />
          <SceneBox x={260} y={210} width={160} height={64} title="worker 1" detail="local 1,3,…" accent="green" />
          <FlowArrow d="M420 132 H480" markerId={purple} color={t.category.purple} />
          <FlowArrow d="M420 242 H480" markerId={purple} color={t.category.purple} />
          <SceneBox x={480} y={150} width={170} height={80} title="worker-local store" detail="thread-per-core" accent="purple" />
        </g>
      );
    case 12:
      return (
        <g>
          <SceneBox x={140} y={110} width={420} height={70} title="global_ordinal" detail="phase_base + local × cell_count + cell_id" accent="green" />
          {[0, 1, 2, 3].map((i) => (
            <SceneBox key={i} x={40 + i * 160} y={250} width={130} height={64} title={`ordinal ${i}`} detail={`cell + ${i}·N`} accent="green" />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M350 180 C${180 + i * 100} 210 ${105 + i * 160} 220 ${105 + i * 160} 250`} markerId={green} color={t.category.green} />
          ))}
        </g>
      );
    case 13:
      return (
        <g>
          <SceneBox x={40} y={150} width={150} height={80} title="worker shard" detail="RequestSink" accent="green" />
          <FlowArrow d="M190 175 C250 130 300 130 350 150" markerId={green} color={t.category.green} />
          <FlowArrow d="M190 205 C250 250 300 250 350 230" markerId={green} color={t.category.green} />
          <SceneBox x={350} y={115} width={180} height={64} title="HTTP / gRPC" detail="clock-injected" accent="green" />
          <SceneBox x={350} y={215} width={180} height={64} title="DynoSim cellular" detail="fail closed" accent="red" />
          <FlowArrow d="M530 147 H600" markerId={purple} color={t.category.purple} />
          <SceneBox x={560} y={115} width={110} height={64} title="server" detail="response" accent="blue" />
        </g>
      );
    case 14:
      return (
        <g>
          {["arrival", "admission", "token", "usage", "terminal"].map((label, i) => (
            <SceneBox key={label} x={20 + i * 108} y={110} width={96} height={60} title={label} detail="event" accent="green" />
          ))}
          <FlowArrow d="M350 200 V250" markerId={purple} color={t.category.purple} />
          <SceneBox x={230} y={250} width={240} height={80} title="CapturedRecord" detail="finalized · storage begins" accent="purple" />
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M${68 + i * 108} 170 C${120 + i * 60} 210 320 220 350 250`} markerId={purple} color={t.category.purple} />
          ))}
        </g>
      );
    case 15:
      return (
        <g>
          <SceneBox x={40} y={150} width={160} height={80} title="dispatch" detail="CapturedRecord" accent="green" />
          <FlowArrow d="M200 190 H270" markerId={purple} color={t.category.purple} />
          <SceneBox x={270} y={150} width={180} height={80} title="Retained rows" detail="Vec<Record> · O(records)" accent="purple" />
          <FlowArrow d="M450 190 H520" markerId={purple} color={t.category.purple} />
          <SceneBox x={520} y={150} width={150} height={80} title="Partition" detail="global-order merge" accent="purple" />
          <text x={230} y={300} fill={t.text.tertiary} fontSize={12}>
            keeps raw record artifacts available
          </text>
        </g>
      );
    case 16:
      return (
        <g>
          <SceneBox x={40} y={150} width={160} height={80} title="dispatch" detail="each record" accent="green" />
          <FlowArrow d="M200 190 H270" markerId={purple} color={t.category.purple} />
          <SceneBox x={270} y={150} width={200} height={80} title="fold_streaming" detail="exact ColumnStore" accent="purple" />
          <FlowArrow d="M470 190 H540" markerId={purple} color={t.category.purple} />
          <SceneBox x={540} y={150} width={130} height={80} title="drop row" detail="bounded" accent="purple" />
          <text x={230} y={300} fill={t.text.tertiary} fontSize={12}>
            exact aggregates without retaining rows
          </text>
        </g>
      );
    case 17:
      return (
        <g>
          <SceneBox x={40} y={150} width={160} height={80} title="finite values" detail="metric stream" accent="green" />
          <FlowArrow d="M200 190 H270" markerId={purple} color={t.category.purple} />
          <SceneBox x={270} y={150} width={200} height={80} title="TagSketch" detail="t-digest + moments" accent="purple" />
          <SceneBox x={150} y={280} width={200} height={54} title="exact" detail="count · sum · extrema · std" accent="green" />
          <SceneBox x={380} y={280} width={200} height={54} title="approximate" detail="percentiles" accent="orange" />
          <FlowArrow d="M370 230 C320 255 300 260 280 280" markerId={green} color={t.category.green} />
          <FlowArrow d="M400 230 C440 255 460 260 470 280" markerId={purple} color={t.category.orange} />
        </g>
      );
    case 18:
      return (
        <g>
          {[0, 1, 2, 3].map((i) => (
            <SceneBox key={i} x={30} y={90 + i * 60} width={120} height={46} title={`cell ${i}`} detail="store" accent="green" />
          ))}
          <SceneBox x={210} y={110} width={150} height={64} title="aggregator 0" detail="subtree fold" accent="yellow" />
          <SceneBox x={210} y={240} width={150} height={64} title="aggregator 1" detail="subtree fold" accent="yellow" />
          {[0, 1].map((i) => (
            <FlowArrow key={i} d={`M150 ${116 + i * 60} H210`} markerId={yellow} color={t.category.yellow} />
          ))}
          {[2, 3].map((i) => (
            <FlowArrow key={i} d={`M150 ${116 + i * 60} C180 ${116 + i * 60} 190 272 210 272`} markerId={yellow} color={t.category.yellow} />
          ))}
          <FlowArrow d="M360 142 C420 160 430 180 460 190" markerId={purple} color={t.category.purple} />
          <FlowArrow d="M360 272 C420 250 430 220 460 210" markerId={purple} color={t.category.purple} />
          <SceneBox x={460} y={150} width={140} height={80} title="Controller merge" detail="one report" accent="purple" />
          <SceneBox x={430} y={300} width={200} height={44} title="external sink" detail="planned" accent="gray" />
        </g>
      );
    case 19:
      return (
        <g>
          <SceneBox x={30} y={70} width={640} height={44} title="CONTROL — coordinate & authority" detail="controller · START · roles" accent="yellow" />
          <SceneBox x={30} y={140} width={640} height={44} title="DATA — distribute & timing" detail="partition · dataset · origin" accent="blue" />
          <SceneBox x={30} y={210} width={640} height={44} title="EXECUTE — request / load" detail="cells · shards · dispatch" accent="green" />
          <SceneBox x={30} y={280} width={640} height={44} title="RESULTS — reduce & publish" detail="retain · fold · merge · report" accent="purple" />
          <FlowArrow d="M350 114 V140" markerId={blue} color={t.category.blue} />
          <FlowArrow d="M350 184 V210" markerId={green} color={t.category.green} />
          <FlowArrow d="M350 254 V280" markerId={purple} color={t.category.purple} />
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
        className="deck-cellular-live"
        viewBox="0 0 700 400"
        role="img"
        aria-label="Cellular internals diagram"
        style={{ display: "block", width: "100%" }}
      >
        <Markers
          green={t.category.green}
          blue={t.category.blue}
          purple={t.category.purple}
          yellow={t.category.yellow}
        />
        <rect x={18} y={16} width={664} height={44} rx={8} fill={t.fill.quaternary} />
        <text x={38} y={43} fill={t.text.primary} fontSize={14} fontWeight={700}>
          {SCENE_LABELS[slideIndex]}
        </text>
        <text x={662} y={43} textAnchor="end" fill={t.text.secondary} fontSize={12}>
          {SLIDES[slideIndex]?.caption}
        </text>
        <Scene slideIndex={slideIndex} />
        {motion ? (
          <g className="deck-cellular-motion" aria-hidden="true">
            <MotionSignal path={motion} delay="0.8s" color={t.category.green} />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
