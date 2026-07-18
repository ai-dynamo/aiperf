/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { FlowArrow } from "../../core/diagram/FlowArrow";
import { MotionSignal } from "../../core/diagram/MotionSignal";
import { SceneBox } from "../../core/diagram/SceneBox";
import type { SlideDefinition } from "../../core/types";
import { useHostTheme } from "../../core/ui";
import { SLIDES } from "./content";

type Status = "BUILT" | "PARTIAL" | "FEATURE-GATED";

const SCENE_LABELS = [
  "WORKBOOK · CHAPTER MAP",
  "WORKBOOK · STATUS & EVIDENCE",
  "ELIGIBILITY · FAIL-CLOSED GATES",
  "OWNERSHIP · DISJOINT TILING",
  "CONTROL · REPLAY + LIVE + START",
  "DISTRIBUTION · CANONICAL + OVERLAY",
  "EXECUTION · TWO-LEVEL PARTITION",
  "CAPTURE · STORAGE MODES",
  "MERGE · EXPLICIT LAWS",
  "ARTIFACTS · AUTHORITY + FAILURE",
  "COMPOSITION · CANONICAL ROUTE",
  "COMPOSITION · STATUS BRANCHES",
  "DECISION · EFFECTIVE RUN KIND",
  "DECISION · FIDELITY + OWNERSHIP",
  "DECISION · PLACEMENT + SHIPPING",
  "REFERENCE · EXHAUSTIVE CATALOG",
] as const;

const MOTION = [
  "M70 132 H630",
  null,
  "M85 188 H610",
  "M115 190 H585",
  "M90 205 H610",
  "M80 195 H620",
  "M80 195 H620",
  "M80 195 H620",
  "M80 195 H620",
  "M80 195 H620",
  "M55 195 H645",
  null,
  "M350 125 V320",
  "M350 125 V320",
  "M350 125 V320",
  null,
] as const;

function Markers({ green, purple, yellow }: { green: string; purple: string; yellow: string }) {
  return (
    <defs>
      <marker id="algo-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={green} />
      </marker>
      <marker id="algo-purple" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={purple} />
      </marker>
      <marker id="algo-yellow" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill={yellow} />
      </marker>
    </defs>
  );
}

function StatusLabel({ x, y, status }: { x: number; y: number; status: Status }) {
  const t = useHostTheme();
  const color =
    status === "BUILT"
      ? t.category.green
      : status === "PARTIAL"
        ? t.category.orange
        : t.category.yellow;
  const width = status === "FEATURE-GATED" ? 116 : 70;

  return (
    <g aria-label={`Status: ${status}`}>
      <rect x={x} y={y} width={width} height={24} rx={12} fill={t.bg.elevated} stroke={color} />
      <text
        x={x + width / 2}
        y={y + 16}
        textAnchor="middle"
        fill={color}
        fontSize={10}
        fontWeight={800}
      >
        {status}
      </text>
    </g>
  );
}

function Algorithm({
  x,
  y,
  width,
  title,
  detail,
  status,
}: {
  x: number;
  y: number;
  width: number;
  title: string;
  detail: string;
  status: Status;
}) {
  return (
    <g>
      <SceneBox
        x={x}
        y={y}
        width={width}
        height={72}
        title={title}
        detail={detail}
        accent={status === "BUILT" ? "green" : status === "PARTIAL" ? "orange" : "yellow"}
      />
      <StatusLabel x={x + width - (status === "FEATURE-GATED" ? 116 : 70)} y={y - 12} status={status} />
    </g>
  );
}

function ChapterMap() {
  const chapters = [
    ["eligibility", "fail closed"],
    ["ownership", "tile work"],
    ["control", "coordinate"],
    ["distribution", "deliver"],
    ["execution", "run shards"],
    ["capture", "store"],
    ["merge", "reduce"],
    ["artifacts", "publish"],
  ] as const;

  return (
    <g>
      {chapters.map(([title, detail], index) => (
        <SceneBox
          key={title}
          x={25 + (index % 4) * 168}
          y={104 + Math.floor(index / 4) * 128}
          width={145}
          height={72}
          title={title}
          detail={detail}
          accent={index < 4 ? "blue" : "purple"}
        />
      ))}
      <text x={350} y={330} textAnchor="middle" fontSize={12} fill="currentColor">
        route order is evidence order
      </text>
    </g>
  );
}

function Decision({
  title,
  left,
  right,
  invariant,
}: {
  title: string;
  left: string;
  right: string;
  invariant: string;
}) {
  const t = useHostTheme();
  return (
    <g>
      <SceneBox x={245} y={92} width={210} height={64} title={title} detail="route selector" accent="yellow" />
      <FlowArrow d="M300 156 C230 195 190 215 165 245" markerId="algo-green" color={t.category.green} />
      <FlowArrow d="M400 156 C470 195 510 215 535 245" markerId="algo-purple" color={t.category.purple} />
      <SceneBox x={45} y={245} width={240} height={70} title={left} detail="left route" accent="green" />
      <SceneBox x={415} y={245} width={240} height={70} title={right} detail="right route" accent="purple" />
      <text x={350} y={355} textAnchor="middle" fill={t.text.secondary} fontSize={11}>
        {invariant}
      </text>
    </g>
  );
}

function Scene({ slideIndex }: { slideIndex: number }) {
  const t = useHostTheme();
  const arrow = (d: string, markerId = "algo-green", color: string = t.category.green) => (
    <FlowArrow d={d} markerId={markerId} color={color} />
  );

  switch (slideIndex) {
    case 0:
      return <ChapterMap />;
    case 1:
      return (
        <g>
          <Algorithm x={35} y={132} width={190} title="modulo-cell-ownership" detail="ordinary route" status="BUILT" />
          <Algorithm x={255} y={132} width={190} title="final-report-assembly" detail="named boundary" status="PARTIAL" />
          <Algorithm x={475} y={132} width={190} title="phaser-start" detail="explicit admission" status="FEATURE-GATED" />
          <SceneBox x={110} y={275} width={480} height={54} title="Evidence" detail="boundary · unit · integration · end-to-end" accent="blue" />
        </g>
      );
    case 2:
      return (
        <g>
          <Algorithm x={22} y={150} width={150} title="execution-mode-dispatch" detail="private protocol" status="BUILT" />
          {arrow("M172 186 H205", "algo-yellow", t.category.yellow)}
          <Algorithm x={205} y={150} width={150} title="controller-promotion" detail="cells > 1" status="BUILT" />
          {arrow("M355 186 H388", "algo-yellow", t.category.yellow)}
          <Algorithm x={388} y={150} width={150} title="run-shape-validation" detail="fail closed" status="BUILT" />
          {arrow("M538 186 H570", "algo-yellow", t.category.yellow)}
          <Algorithm x={570} y={150} width={108} title="Velo roles" detail="admit" status="FEATURE-GATED" />
        </g>
      );
    case 3:
      return (
        <g>
          <Algorithm x={30} y={145} width={210} title="modulo-cell-ownership" detail="i % N == k" status="BUILT" />
          {arrow("M240 181 H290")}
          <SceneBox x={290} y={105} width={170} height={58} title="cell 0" detail="0, 3, 6, …" accent="green" />
          <SceneBox x={290} y={181} width={170} height={58} title="cell 1" detail="1, 4, 7, …" accent="green" />
          <SceneBox x={290} y={257} width={170} height={58} title="cell 2" detail="2, 5, 8, …" accent="green" />
          <Algorithm x={500} y={145} width={170} title="conversation-ownership" detail="whole sessions" status="BUILT" />
          <text x={350} y={350} textAnchor="middle" fill={t.text.secondary} fontSize={12}>
            disjoint + complete
          </text>
        </g>
      );
    case 4:
      return (
        <g>
          <Algorithm x={22} y={150} width={185} title="broadcast-attach-replay" detail="close seam race" status="BUILT" />
          {arrow("M207 186 H250", "algo-yellow", t.category.yellow)}
          <Algorithm x={250} y={150} width={195} title="phaser-generation-advance" detail="monotonic" status="BUILT" />
          {arrow("M445 186 H488", "algo-yellow", t.category.yellow)}
          <Algorithm x={488} y={150} width={190} title="phaser-start" detail="opt-in START" status="FEATURE-GATED" />
          <text x={350} y={302} textAnchor="middle" fill={t.text.secondary} fontSize={12}>
            coordination changes · ownership does not
          </text>
        </g>
      );
    case 5:
      return (
        <g>
          <Algorithm x={35} y={122} width={220} title="canonical-regeneration" detail="measured data" status="BUILT" />
          <Algorithm x={35} y={245} width={220} title="controller-fanout-generation" detail="verification overlay" status="FEATURE-GATED" />
          {arrow("M255 158 H440")}
          {arrow("M255 281 C340 275 375 230 440 210", "algo-yellow", t.category.yellow)}
          <Algorithm x={440} y={174} width={225} title="distribution-miss" detail="count · surface · recover" status="BUILT" />
        </g>
      );
    case 6:
      return (
        <g>
          <Algorithm x={30} y={150} width={175} title="two-level-partition" detail="cell ownership" status="BUILT" />
          {arrow("M205 186 H260")}
          <SceneBox x={260} y={108} width={180} height={64} title="worker 0" detail="!Send local sink" accent="green" />
          <SceneBox x={260} y={225} width={180} height={64} title="worker 1" detail="!Send local sink" accent="green" />
          {arrow("M440 140 H500", "algo-purple", t.category.purple)}
          {arrow("M440 257 H500", "algo-purple", t.category.purple)}
          <Algorithm x={500} y={150} width={170} title="issuance-dispatch-injection" detail="global identity" status="BUILT" />
        </g>
      );
    case 7:
      return (
        <g>
          <Algorithm x={20} y={150} width={165} title="terminal-record-finalization" detail="CapturedRecord" status="BUILT" />
          {arrow("M185 186 H225", "algo-purple", t.category.purple)}
          <SceneBox x={225} y={92} width={210} height={60} title="retain-record-capture" detail="rows + raw artifacts" accent="purple" />
          <SceneBox x={225} y={170} width={210} height={60} title="streaming-exact-fold" detail="exact store · drop row" accent="purple" />
          <SceneBox x={225} y={248} width={210} height={60} title="tagged-sketch-merge" detail="bounded percentiles" accent="orange" />
          <Algorithm x={485} y={150} width={190} title="storage compatibility" detail="artifact authority" status="BUILT" />
        </g>
      );
    case 8:
      return (
        <g>
          <Algorithm x={25} y={116} width={220} title="scheduled-global-ordinal-merge" detail="dense permutation" status="BUILT" />
          <Algorithm x={25} y={246} width={220} title="merged-report-fidelity-boundary" detail="graph · fold · sketch" status="PARTIAL" />
          {arrow("M245 152 C350 155 385 182 445 195", "algo-purple", t.category.purple)}
          {arrow("M245 282 C350 275 385 228 445 210", "algo-purple", t.category.purple)}
          <Algorithm x={445} y={165} width={225} title="final-report-assembly" detail="publish merged report" status="PARTIAL" />
        </g>
      );
    case 9:
      return (
        <g>
          <Algorithm x={20} y={150} width={160} title="artifact-authority-allowlist" detail="relative paths" status="FEATURE-GATED" />
          {arrow("M180 186 H215", "algo-purple", t.category.purple)}
          <Algorithm x={215} y={150} width={150} title="HTTP zstd upload" detail=".partial file" status="FEATURE-GATED" />
          {arrow("M365 186 H400", "algo-purple", t.category.purple)}
          <Algorithm x={400} y={150} width={130} title="atomic replace" detail="publish" status="FEATURE-GATED" />
          {arrow("M530 186 H565", "algo-purple", t.category.purple)}
          <Algorithm x={565} y={150} width={115} title="completion barrier" detail="all files" status="FEATURE-GATED" />
        </g>
      );
    case 10:
      return (
        <g>
          {["eligibility", "ownership", "control", "distribution", "execution", "capture", "merge", "artifacts"].map(
            (label, index) => (
              <g key={label}>
                <SceneBox
                  x={18 + index * 85}
                  y={160}
                  width={74}
                  height={64}
                  title={label}
                  detail={index === 0 ? "gate" : "route"}
                  accent={index < 4 ? "blue" : "purple"}
                />
                {index < 7 ? arrow(`M${92 + index * 85} 192 H${103 + index * 85}`) : null}
              </g>
            ),
          )}
          <text x={350} y={292} textAnchor="middle" fill={t.text.secondary} fontSize={12}>
            canonical scheduled · exact fold · flat merge
          </text>
        </g>
      );
    case 11:
      return (
        <g>
          <Algorithm x={35} y={118} width={195} title="canonical route" detail="ownership + execution" status="BUILT" />
          <Algorithm x={252} y={118} width={195} title="phaser / fan-out / Stage E" detail="explicit branches" status="FEATURE-GATED" />
          <Algorithm x={469} y={118} width={195} title="report / telemetry / cancel" detail="named limitations" status="PARTIAL" />
          <text x={350} y={295} textAnchor="middle" fill={t.text.primary} fontSize={14} fontWeight={700}>
            status text travels with every selected algorithm
          </text>
        </g>
      );
    case 12:
      return (
        <Decision
          title="run-kind-classification"
          left="synthetic → scheduled"
          right="WEKA → graph"
          invariant="dataset shape selects the effective run kind"
        />
      );
    case 13:
      return (
        <Decision
          title="storage + ownership"
          left="exact fold · requests"
          right="sketch · conversations"
          invariant="one owner remains invariant"
        />
      );
    case 14:
      return (
        <Decision
          title="deployment placement"
          left="same host · flat/local"
          right="cross host · Stage E/G"
          invariant="placement changes shipping, not measurement"
        />
      );
    case 15:
      return (
        <g>
          <SceneBox x={70} y={110} width={560} height={82} title="~100 algorithm pages" detail="pseudocode · frames · gates · failures · evidence" accent="blue" />
          {arrow("M350 192 V245", "algo-purple", t.category.purple)}
          <SceneBox
            x={70}
            y={245}
            width={560}
            height={72}
            title="docs/canvases/cellular-algorithm-workbook.canvas.tsx"
            detail="exhaustive maintainer lookup"
            accent="purple"
          />
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
        className="deck-algorithm-live"
        viewBox="0 0 700 400"
        role="img"
        aria-label="Cellular algorithm workbook diagram"
        style={{ display: "block", width: "100%", height: "auto" }}
      >
        <Markers
          green={t.category.green}
          purple={t.category.purple}
          yellow={t.category.yellow}
        />
        <rect x={18} y={16} width={664} height={44} rx={8} fill={t.fill.quaternary} />
        <text x={38} y={43} fill={t.text.primary} fontSize={14} fontWeight={700}>
          {SCENE_LABELS[slideIndex]}
        </text>
        <text x={662} y={43} textAnchor="end" fill={t.text.secondary} fontSize={11}>
          {SLIDES[slideIndex]?.caption}
        </text>
        <Scene slideIndex={slideIndex} />
        {motion ? (
          <g className="deck-algorithm-motion" aria-hidden="true">
            <MotionSignal path={motion} delay="0.7s" color={t.category.green} />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
