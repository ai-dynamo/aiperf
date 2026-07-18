import { FlowArrow } from "../../core/diagram/FlowArrow";
import { MotionSignal } from "../../core/diagram/MotionSignal";
import { SceneBox } from "../../core/diagram/SceneBox";
import { useHostTheme } from "../../core/ui";
import type { SlideDefinition } from "../../core/types";
import { SLIDES } from "./content";

const SCENE_LABELS = [
  "BUILD · FREEZE · DISPATCH",
  "SEGMENT POOL INTERN",
  "PAYLOAD DOMAINS",
  "BODYPLAN SPLICE",
  "PREFIX CHAINS",
  "DISPATCH PRECEDENCE",
] as const;

const MOTION: readonly (string | null)[] = [
  "M120 200 H580",
  "M200 180 H500",
  "M100 200 H600",
  "M350 120 V280",
  "M190 167 H470",
  "M210 180 H590",
];

function Markers({ green, purple, yellow }: { green: string; purple: string; yellow: string }) {
  return (
    <defs>
      <marker id="seg-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={green} />
      </marker>
      <marker id="seg-purple" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={purple} />
      </marker>
      <marker id="seg-yellow" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={yellow} />
      </marker>
    </defs>
  );
}

function Scene({ slideIndex }: { slideIndex: number }) {
  const t = useHostTheme();
  const green = "seg-green";
  const purple = "seg-purple";
  const yellow = "seg-yellow";

  switch (slideIndex) {
    case 0:
      return (
        <g>
          <text x={90} y={88} fill={t.text.tertiary} fontSize={11} fontWeight={700}>
            BUILD
          </text>
          <text x={320} y={88} fill={t.text.tertiary} fontSize={11} fontWeight={700}>
            FREEZE
          </text>
          <text x={530} y={88} fill={t.text.tertiary} fontSize={11} fontWeight={700}>
            DISPATCH
          </text>
          <SceneBox x={30} y={110} width={150} height={70} title="Composer" detail="intern rows" accent="blue" />
          <FlowArrow d="M180 145 H230" markerId={purple} color={t.category.purple} />
          <SceneBox x={230} y={100} width={160} height={90} title="SegmentPool" detail="arena + ids map" accent="purple" />
          <FlowArrow d="M390 145 H430" markerId={purple} color={t.category.purple} />
          <SceneBox x={430} y={110} width={110} height={70} title=".freeze()" detail="drop ids" accent="purple" />
          <FlowArrow d="M540 145 H570" markerId={green} />
          <SceneBox x={570} y={90} width={110} height={60} title="Dataset" detail="Arc store" accent="green" />
          <SceneBox x={570} y={170} width={110} height={60} title="BodyPlan" detail="precompute" accent="green" />
          <SceneBox x={570} y={250} width={110} height={60} title="Materializer" detail="→ Bytes" accent="yellow" />
          <SceneBox x={230} y={250} width={160} height={70} title="InMemoryStore" detail="Box<[Segment]>" accent="purple" />
        </g>
      );
    case 1:
      return (
        <g>
          <SceneBox x={40} y={100} width={160} height={80} title="intern(msg)" detail="parent + bytes" accent="blue" />
          <FlowArrow d="M200 140 H250" markerId={purple} color={t.category.purple} />
          <SceneBox x={250} y={90} width={200} height={100} title="SegmentPool" detail="HashMap → Handle" accent="purple" />
          <FlowArrow d="M450 140 H500" markerId={green} />
          <SceneBox x={500} y={100} width={160} height={80} title="dedup hit?" detail="reuse Handle" accent="green" />
          <SceneBox x={120} y={240} width={140} height={70} title="H0 system" detail="shared" accent="blue" />
          <SceneBox x={300} y={240} width={140} height={70} title="H1 user" detail="shared" accent="blue" />
          <SceneBox x={480} y={240} width={140} height={70} title="H2 / H3" detail="novel replies" accent="orange" />
        </g>
      );
    case 2:
      return (
        <g>
          {[
            { x: 30, title: "Message", detail: "wire + tokens", accent: "blue" as const },
            { x: 145, title: "Text", detail: "text-only", accent: "blue" as const },
            { x: 260, title: "Raw", detail: "complete body", accent: "purple" as const },
            { x: 375, title: "TokenIds", detail: "native path", accent: "green" as const },
            { x: 490, title: "Media", detail: "multimodal", accent: "orange" as const },
            { x: 605, title: "TraceHash", detail: "sim identity", accent: "orange" as const },
          ].map((box) => (
            <SceneBox
              key={box.title}
              x={box.x}
              y={110}
              width={100}
              height={90}
              title={box.title}
              detail={box.detail}
              accent={box.accent}
            />
          ))}
          <SceneBox
            x={150}
            y={250}
            width={400}
            height={70}
            title="HASH_VERSION + domain + parent id"
            detail="blake3 → SegmentId([u8;32])"
            accent="purple"
          />
        </g>
      );
    case 3:
      return (
        <g>
          <SceneBox x={40} y={100} width={160} height={80} title="BodyPlan" detail="shape + slots" accent="green" />
          <FlowArrow d="M200 140 H250" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={250} y={90} width={200} height={100} title="JsonBodyMaterializer" detail="walk once" accent="yellow" />
          <FlowArrow d="M450 140 H500" markerId={green} />
          <SceneBox x={500} y={100} width={160} height={80} title="Bytes" detail="wire body" accent="green" />
          <SceneBox x={80} y={240} width={160} height={70} title="literals" detail="serde_json" accent="blue" />
          <SceneBox x={280} y={240} width={160} height={70} title="segment wires" detail="clone Bytes" accent="green" />
          <SceneBox x={480} y={240} width={160} height={70} title="Overrides" detail="stream · limits" accent="orange" />
        </g>
      );
    case 4:
      return (
        <g>
          <SceneBox x={40} y={140} width={140} height={70} title="H0 system" detail="shared once" accent="blue" />
          <FlowArrow d="M180 175 H220" markerId={green} color={t.category.blue} />
          <SceneBox x={220} y={140} width={140} height={70} title="H1 user" detail="shared once" accent="blue" />
          <FlowArrow d="M360 155 C400 120 420 100 460 100" markerId={green} color={t.category.blue} />
          <FlowArrow d="M360 195 C400 230 420 250 460 250" markerId={green} />
          <SceneBox x={460} y={70} width={150} height={70} title="H2 assistant" detail='C1 "4"' accent="blue" />
          <SceneBox x={460} y={220} width={150} height={70} title="H3 assistant" detail="C2 novel" accent="green" />
          <text x={350} y={340} textAnchor="middle" fill={t.text.secondary} fontSize={12}>
            LCP trie / compose parent chains reuse H0·H1
          </text>
        </g>
      );
    case 5:
      return (
        <g>
          <SceneBox x={30} y={140} width={150} height={80} title="Turn.body" detail="SmallVec handles" accent="blue" />
          <FlowArrow d="M180 160 C220 120 240 100 280 100" markerId={purple} color={t.category.purple} />
          <FlowArrow d="M180 180 H280" markerId={green} />
          <FlowArrow d="M180 200 C220 240 240 260 280 260" markerId={green} color={t.category.blue} />
          <SceneBox x={280} y={70} width={170} height={70} title="Raw first?" detail="complete body" accent="purple" />
          <SceneBox x={280} y={155} width={170} height={70} title="TokenIds?" detail="native / validate" accent="green" />
          <SceneBox x={280} y={240} width={170} height={70} title="Messages" detail="format array" accent="blue" />
          <FlowArrow d="M450 180 H500" markerId={yellow} color={t.category.yellow} />
          <SceneBox x={500} y={140} width={160} height={80} title="BodyPlan → Bytes" detail="domain path" accent="yellow" />
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
        className="deck-segment-pools-live"
        viewBox="0 0 700 400"
        role="img"
        aria-label="Segment pools and body plans diagram"
        style={{ display: "block", width: "100%" }}
      >
        <Markers green={t.category.green} purple={t.category.purple} yellow={t.category.yellow} />
        <rect x={18} y={16} width={664} height={44} rx={8} fill={t.fill.quaternary} />
        <text x={38} y={43} fill={t.text.primary} fontSize={14} fontWeight={700}>
          {SCENE_LABELS[slideIndex]}
        </text>
        <text x={662} y={43} textAnchor="end" fill={t.text.secondary} fontSize={12}>
          {SLIDES[slideIndex]?.caption}
        </text>
        <Scene slideIndex={slideIndex} />
        {motion ? (
          <g className="deck-segment-pools-motion" aria-hidden="true">
            <MotionSignal path={motion} delay="0.8s" color={t.category.purple} />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
