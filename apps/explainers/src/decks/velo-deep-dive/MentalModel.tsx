import { FlowArrow } from "../../core/diagram/FlowArrow";
import { MotionSignal } from "../../core/diagram/MotionSignal";
import { SceneBox } from "../../core/diagram/SceneBox";
import { useHostTheme } from "../../core/ui";
import type { SlideDefinition } from "../../core/types";
import { SLIDES } from "./content";

const SCENE_LABELS = [
  "CONNECTION RESOLVE",
  "REGISTRATION REPLY",
  "SYNCHRONIZED START",
  "MESSAGEPACK CODEC",
  "HEARTBEAT LANE",
  "PARTITION SHIP",
  "MERGE CENTER",
  "PHASER REPLAY",
  "DATASET FLOODGATE",
  "AGGREGATOR TREE",
] as const;

// green = request/load, purple = results, yellow = coordination authority, blue = timing/runtime.
const MOTION: readonly (string | null)[] = [
  "M110 175 H560",
  "M110 170 H560",
  "M350 300 V150",
  "M90 180 H600",
  "M120 130 H560",
  "M110 180 H590",
  "M110 175 H560",
  "M110 130 H590",
  "M120 175 H560",
  "M350 320 V150",
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
      <marker id="velo-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={green} />
      </marker>
      <marker id="velo-blue" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={blue} />
      </marker>
      <marker id="velo-purple" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={purple} />
      </marker>
      <marker id="velo-yellow" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={yellow} />
      </marker>
    </defs>
  );
}

function Scene({ slideIndex }: { slideIndex: number }) {
  const t = useHostTheme();
  const green = "velo-green";
  const blue = "velo-blue";
  const purple = "velo-purple";
  const yellow = "velo-yellow";

  switch (slideIndex) {
    case 0:
      return (
        <g>
          <SceneBox x={40} y={140} width={160} height={80} title="Cell k" detail="known coordinate" accent="blue" />
          <FlowArrow d="M200 180 H300" markerId={blue} color={t.category.blue} />
          <text x={250} y={168} textAnchor="middle" fill={t.text.tertiary} fontSize={11} fontWeight={700}>
            _hello
          </text>
          <SceneBox x={300} y={140} width={170} height={80} title="Velo transport" detail="resolve PeerInfo" accent="blue" />
          <FlowArrow d="M470 180 H540" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={540} y={140} width={140} height={80} title="Controller" detail="register_peer" accent="yellow" />
          <SceneBox x={210} y={280} width={280} height={60} title="tcp:// · uds:// · SLURM_* · k8s" detail="coordinate sources" accent="blue" />
        </g>
      );
    case 1:
      return (
        <g>
          <SceneBox x={30} y={150} width={150} height={70} title="CellRegister" detail="cell_id: u32" accent="green" />
          <FlowArrow d="M180 185 H225" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={225} y={150} width={150} height={70} title="register_peer" detail="return route" accent="yellow" />
          <FlowArrow d="M375 185 H420" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={420} y={150} width={150} height={70} title="spec_for(id)" detail="pure lookup" accent="yellow" />
          <FlowArrow d="M570 185 H610" markerId={purple} color={t.category.purple} />
          <SceneBox x={470} y={280} width={200} height={70} title="RegisterReply" detail="envelope + start_event" accent="purple" />
          <FlowArrow d="M495 220 V280" markerId={purple} color={t.category.purple} />
        </g>
      );
    case 2:
      return (
        <g>
          {[0, 1, 2, 3].map((i) => (
            <SceneBox
              key={i}
              x={30 + i * 100}
              y={280}
              width={80}
              height={60}
              title={`c${i}`}
              detail="register"
              accent="green"
            />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M${70 + i * 100} 280 C${70 + i * 100} 230 350 220 460 190`} markerId={yellow} color={t.category.yellow} />
          ))}
          <SceneBox x={460} y={150} width={180} height={80} title="Barrier" detail="all_registered → trigger" accent="yellow" />
          <SceneBox x={210} y={110} width={220} height={54} title="START event" detail="wake all awaiters" accent="yellow" />
          <FlowArrow d="M550 150 C520 120 470 130 430 137" markerId={yellow} color={t.category.yellow} />
        </g>
      );
    case 3:
      return (
        <g>
          <SceneBox x={30} y={150} width={150} height={80} title="CellMessage" detail="ttft: NaN · max +∞" accent="blue" />
          <FlowArrow d="M180 190 H225" markerId={blue} color={t.category.blue} />
          <SceneBox x={225} y={150} width={130} height={80} title="to_vec" detail="rmp_serde" accent="blue" />
          <FlowArrow d="M355 190 H400" markerId={blue} color={t.category.blue} />
          <SceneBox x={400} y={150} width={130} height={80} title="raw bytes" detail="Velo payload" accent="green" />
          <FlowArrow d="M530 190 H575" markerId={blue} color={t.category.blue} />
          <SceneBox x={555} y={280} width={130} height={70} title="from_slice" detail="typed again" accent="blue" />
          <FlowArrow d="M600 230 V280" markerId={blue} color={t.category.blue} />
        </g>
      );
    case 4:
      return (
        <g>
          {[0, 1, 2].map((i) => (
            <SceneBox
              key={i}
              x={40}
              y={100 + i * 90}
              width={140}
              height={64}
              title={`cell ${i}`}
              detail={i === 2 ? "pulse missing" : "snapshot"}
              accent={i === 2 ? "orange" : "green"}
            />
          ))}
          {[0, 1, 2].map((i) => (
            <FlowArrow key={i} d={`M180 ${132 + i * 90} C300 ${132 + i * 90} 360 200 440 200`} markerId={purple} color={i === 2 ? t.category.orange : t.category.purple} />
          ))}
          <SceneBox x={440} y={160} width={220} height={80} title="Controller aggregate" detail="counters + sketches · lag" accent="purple" />
          <text x={110} y={360} fill={t.text.tertiary} fontSize={12}>
            fire-and-forget · never blocks dispatch
          </text>
        </g>
      );
    case 5:
      return (
        <g>
          <SceneBox x={30} y={150} width={160} height={80} title="Finished cell" detail="ship peer + partition" accent="green" />
          <FlowArrow d="M190 190 H250" markerId={purple} color={t.category.purple} />
          <SceneBox x={250} y={150} width={150} height={80} title="unary payload" detail="fresh PeerInfo" accent="purple" />
          <FlowArrow d="M400 190 H460" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={460} y={150} width={200} height={80} title="Controller" detail="register_peer(shipper)" accent="yellow" />
          <FlowArrow d="M560 230 C560 290 300 300 190 290" markerId={green} color={t.category.green} />
          <SceneBox x={210} y={280} width={130} height={54} title="CellAck" detail="delivered" accent="green" />
        </g>
      );
    case 6:
      return (
        <g>
          {[0, 1, 2, 3].map((i) => (
            <SceneBox
              key={i}
              x={30}
              y={90 + i * 68}
              width={130}
              height={52}
              title={`partition ${i}`}
              detail={i % 2 === 0 ? "records" : "store"}
              accent="purple"
            />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M160 ${116 + i * 68} C280 ${116 + i * 68} 340 200 420 200`} markerId={purple} color={t.category.purple} />
          ))}
          <SceneBox x={420} y={160} width={150} height={80} title="Merge center" detail="associative fold" accent="purple" />
          <FlowArrow d="M570 200 H620" markerId={green} color={t.category.green} />
          <SceneBox x={520} y={290} width={160} height={60} title="one result" detail="order · append · merge" accent="green" />
          <FlowArrow d="M600 240 V290" markerId={green} color={t.category.green} />
        </g>
      );
    case 7:
      return (
        <g>
          {[1, 2, 3, 4, 5].map((g, i) => {
            const replay = g <= 3;
            return (
              <SceneBox
                key={g}
                x={30 + i * 95}
                y={110}
                width={80}
                height={56}
                title={`g${g}`}
                detail={replay ? "replay" : "live"}
                accent={replay ? "yellow" : "green"}
              />
            );
          })}
          <SceneBox x={220} y={250} width={180} height={70} title="attach @ g3" detail="capture boundary" accent="yellow" />
          <FlowArrow d="M310 250 C310 210 250 190 200 168" markerId={yellow} color={t.category.yellow} />
          <FlowArrow d="M400 260 C480 240 500 200 510 168" markerId={green} color={t.category.green} />
          <text x={150} y={356} fill={t.text.tertiary} fontSize={12}>
            reply replays ≤ g3 · active-message pushes &gt; g3
          </text>
        </g>
      );
    case 8:
      return (
        <g>
          <SceneBox x={30} y={140} width={140} height={90} title="chunk stream" detail="MessagePack + zstd" accent="green" />
          <FlowArrow d="M170 185 H230" markerId={green} color={t.category.green} />
          <SceneBox x={230} y={150} width={120} height={70} title="broadcast" detail="publish once" accent="blue" />
          {[0, 1, 2].map((i) => (
            <SceneBox
              key={i}
              x={430}
              y={90 + i * 90}
              width={230}
              height={64}
              title={`cell ${i}`}
              detail={`keep id % 3 == ${i}`}
              accent="green"
            />
          ))}
          {[0, 1, 2].map((i) => (
            <FlowArrow key={i} d={`M350 185 C390 ${122 + i * 90} 400 ${122 + i * 90} 430 ${122 + i * 90}`} markerId={green} color={t.category.green} />
          ))}
        </g>
      );
    case 9:
      return (
        <g>
          <SceneBox x={270} y={90} width={160} height={64} title="Controller" detail="one report" accent="purple" />
          <SceneBox x={120} y={210} width={150} height={60} title="aggregator 0" detail="cells 0–3" accent="yellow" />
          <SceneBox x={430} y={210} width={150} height={60} title="aggregator 1" detail="cells 4–7" accent="yellow" />
          <FlowArrow d="M195 210 C220 180 300 160 330 154" markerId={purple} color={t.category.purple} />
          <FlowArrow d="M505 210 C480 180 400 160 370 154" markerId={purple} color={t.category.purple} />
          {[0, 1, 2, 3].map((i) => (
            <SceneBox
              key={i}
              x={30 + i * 90}
              y={320}
              width={70}
              height={50}
              title={`c${i}`}
              detail="store"
              accent="green"
            />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <FlowArrow key={i} d={`M${65 + i * 90} 320 C${65 + i * 90} 300 170 290 195 270`} markerId={purple} color={t.category.purple} />
          ))}
          <text x={430} y={356} fill={t.text.tertiary} fontSize={12}>
            fold only · retain stays flat
          </text>
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
        className="deck-velo-deep-live"
        viewBox="0 0 700 400"
        role="img"
        aria-label="Velo deep dive diagram"
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
          <g className="deck-velo-deep-motion" aria-hidden="true">
            <MotionSignal path={motion} delay="0.8s" color={t.category.blue} />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
