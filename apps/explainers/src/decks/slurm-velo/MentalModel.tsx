import { Text, useHostTheme } from "../../core/ui";
import type { SlideDefinition } from "../../core/types";

const SCENE_LABELS = [
  "BENCHMARK GOAL",
  "SLURM RESERVES RESOURCES",
  "ONE COMMAND, MANY TASKS",
  "RANKS BECOME AIPERF ROLES",
  "EVERY CELL FINDS RANK 0",
  "VELO = CONTROL MESSAGING",
  "VELO HANDSHAKE",
  "REGISTER + START OVER VELO",
  "HOT PATH IS NOT VELO",
  "THREE TRAFFIC PLANES",
  "RESULTS OVER VELO",
  "BULK ARTIFACTS SEPARATE",
  "CONTROLLER ROLE VS NODE",
  "FAN-OUT: ONE PLAN → MANY SLICES",
  "FAN-IN: MANY RESULTS → ONE REPORT",
  "FROM CONFIG TO REPORT",
] as const;

const SCENE_NOTES = [
  "many machines generate one combined load",
  "the scheduler lends your job a set of machines",
  "SLURM_PROCID gives each copy a unique rank",
  "rank 0 coordinates · ranks 1…N generate load",
  "the controller address is derived, not discovered",
  "SLURM launches processes · Velo lets them talk",
  "address in · peer connection out",
  "Velo carries the readiness choreography",
  "measured requests use HTTP / gRPC only",
  "control · load · bulk files",
  "partitions return to the original rank-0 controller",
  "large files use HTTP/1 + zstd, not Velo",
  "role is required · whole node is optional",
  "rank 0 distributes distinct cell-owned work",
  "every cell returns one terminal result",
  "the generated script launches the complete run",
] as const;

const MOTION_ROUTES: readonly (string | null)[] = [
  "M294 186 H418",
  "M254 176 H330",
  "M350 136 V168 H150 V198 V168 H350 V198 V168 H550 V198",
  "M292 178 H408",
  "M572 164 H331 H572",
  "M228 155 H472 V195 H228",
  "M98 178 H602 V248 H98",
  "M76 170 H620 H76",
  "M210 160 H430 H210",
  "M80 122 H620 V214 H80 V306 H620",
  "M162 178 H536",
  null,
  null,
  "M210 178 H410",
  "M290 178 H454",
  "M128 173 H564",
];

type SignalLane = "control" | "load" | "results" | "mixed";

const SIGNAL_LANES: readonly SignalLane[] = [
  "load",
  "control",
  "control",
  "control",
  "control",
  "control",
  "control",
  "control",
  "load",
  "mixed",
  "control",
  "results",
  "control",
  "control",
  "control",
  "mixed",
];

type FlowStop = {
  x: number;
  y: number;
  width: number;
  height: number;
  at: number;
  rx?: number;
};

type PulseTone = "green" | "yellow" | "purple" | "neutral";

const PULSE_TONES: readonly (readonly PulseTone[])[] = [
  ["neutral", "green"],
  ["yellow", "neutral"],
  ["green", "neutral", "neutral", "neutral"],
  ["yellow", "green"],
  ["green", "green", "green"],
  ["yellow", "green", "green", "green", "yellow"],
  ["neutral", "green", "neutral", "neutral", "neutral", "green", "neutral"],
  ["green", "green", "green", "green", "green", "green", "green", "green", "green"],
  ["green", "green", "green"],
  ["green", "green", "purple"],
  ["green", "green", "green", "purple"],
  [],
  [],
  ["yellow", "green"],
  ["green", "purple"],
  ["neutral", "neutral", "neutral", "neutral", "purple"],
];

const FLOW_STOPS: readonly (readonly FlowStop[])[] = [
  [
    { x: 74, y: 138, width: 220, height: 96, rx: 10, at: 0 },
    { x: 418, y: 138, width: 210, height: 96, rx: 36, at: 1 },
  ],
  [
    { x: 64, y: 128, width: 190, height: 96, rx: 10, at: 0 },
    { x: 330, y: 98, width: 304, height: 156, rx: 10, at: 1 },
  ],
  [
    { x: 190, y: 78, width: 320, height: 58, rx: 8, at: 0 },
    { x: 72, y: 198, width: 156, height: 88, rx: 8, at: 0.33 },
    { x: 272, y: 198, width: 156, height: 88, rx: 8, at: 0.66 },
    { x: 472, y: 198, width: 156, height: 88, rx: 8, at: 1 },
  ],
  [
    { x: 72, y: 122, width: 220, height: 112, rx: 10, at: 0 },
    { x: 408, y: 98, width: 220, height: 160, rx: 10, at: 1 },
  ],
  [
    { x: 482, y: 102, width: 180, height: 124, rx: 8, at: 0 },
    { x: 236, y: 116, width: 190, height: 96, rx: 8, at: 0.5 },
    { x: 482, y: 102, width: 180, height: 124, rx: 8, at: 1 },
  ],
  [
    { x: 48, y: 100, width: 180, height: 150, rx: 10, at: 0 },
    { x: 270, y: 132, width: 160, height: 86, rx: 10, at: 0.25 },
    { x: 472, y: 100, width: 180, height: 150, rx: 10, at: 0.5 },
    { x: 270, y: 132, width: 160, height: 86, rx: 10, at: 0.75 },
    { x: 48, y: 100, width: 180, height: 150, rx: 10, at: 1 },
  ],
  [
    { x: 28, y: 128, width: 140, height: 100, rx: 9, at: 0 },
    { x: 196, y: 128, width: 140, height: 100, rx: 9, at: 0.17 },
    { x: 364, y: 128, width: 140, height: 100, rx: 9, at: 0.34 },
    { x: 532, y: 128, width: 140, height: 100, rx: 9, at: 0.5 },
    { x: 364, y: 128, width: 140, height: 100, rx: 9, at: 0.67 },
    { x: 196, y: 128, width: 140, height: 100, rx: 9, at: 0.84 },
    { x: 28, y: 128, width: 140, height: 100, rx: 9, at: 1 },
  ],
  [
    { x: 20, y: 120, width: 112, height: 100, rx: 8, at: 0 },
    { x: 156, y: 120, width: 112, height: 100, rx: 8, at: 0.13 },
    { x: 292, y: 120, width: 112, height: 100, rx: 8, at: 0.25 },
    { x: 428, y: 120, width: 112, height: 100, rx: 8, at: 0.38 },
    { x: 564, y: 120, width: 112, height: 100, rx: 8, at: 0.5 },
    { x: 428, y: 120, width: 112, height: 100, rx: 8, at: 0.63 },
    { x: 292, y: 120, width: 112, height: 100, rx: 8, at: 0.75 },
    { x: 156, y: 120, width: 112, height: 100, rx: 8, at: 0.88 },
    { x: 20, y: 120, width: 112, height: 100, rx: 8, at: 1 },
  ],
  [
    { x: 40, y: 100, width: 170, height: 120, rx: 10, at: 0 },
    { x: 430, y: 100, width: 170, height: 120, rx: 36, at: 0.5 },
    { x: 40, y: 100, width: 170, height: 120, rx: 10, at: 1 },
  ],
  [
    { x: 60, y: 84, width: 580, height: 76, rx: 10, at: 0 },
    { x: 60, y: 176, width: 580, height: 76, rx: 10, at: 0.5 },
    { x: 60, y: 268, width: 580, height: 76, rx: 10, at: 1 },
  ],
  [
    { x: 26, y: 126, width: 136, height: 104, rx: 9, at: 0 },
    { x: 196, y: 126, width: 136, height: 104, rx: 9, at: 0.33 },
    { x: 366, y: 126, width: 136, height: 104, rx: 9, at: 0.66 },
    { x: 536, y: 126, width: 136, height: 104, rx: 9, at: 1 },
  ],
  [],
  [],
  [
    { x: 48, y: 124, width: 162, height: 108, rx: 10, at: 0 },
    { x: 410, y: 153, width: 186, height: 50, rx: 8, at: 1 },
  ],
  [
    { x: 104, y: 153, width: 186, height: 50, rx: 8, at: 0 },
    { x: 454, y: 124, width: 166, height: 108, rx: 10, at: 1 },
  ],
  [
    { x: 20, y: 126, width: 108, height: 94, rx: 8, at: 0 },
    { x: 156, y: 126, width: 108, height: 94, rx: 8, at: 0.25 },
    { x: 292, y: 126, width: 108, height: 94, rx: 8, at: 0.5 },
    { x: 428, y: 126, width: 108, height: 94, rx: 8, at: 0.75 },
    { x: 564, y: 126, width: 108, height: 94, rx: 8, at: 1 },
  ],
];


function MotionSignals({ slideIndex }: { slideIndex: number }) {
  if (slideIndex === 2 || slideIndex === 13 || slideIndex === 14) {
    return <ParallelMotionSignals slideIndex={slideIndex} />;
  }
  const t = useHostTheme();
  const path = MOTION_ROUTES[slideIndex];
  if (!path) return null;
  const lane = SIGNAL_LANES[slideIndex] ?? "control";
  const signalColor =
    lane === "load"
      ? t.category.green
      : lane === "results"
        ? t.category.purple
        : t.category.green;
  const stops = FLOW_STOPS[slideIndex] ?? [];
  const pulseTones = PULSE_TONES[slideIndex] ?? [];
  const pulseColor = (index: number) => {
    const tone = pulseTones[index] ?? "neutral";
    return tone === "green"
      ? t.category.green
      : tone === "yellow"
        ? t.category.yellow
        : tone === "purple"
          ? t.category.purple
          : t.stroke.primary;
  };
  const interior = stops
    .map((stop) => stop.at)
    .filter((at) => at > 0.05 && at < 0.95);
  const opacityKeyTimes = [
    0,
    0.045,
    ...interior.flatMap((at) => [at - 0.025, at, at + 0.025]),
    0.955,
    1,
  ];
  const opacityValues = [
    0,
    1,
    ...interior.flatMap(() => [1, 0, 1]),
    1,
    0,
  ];
  return (
    <g className="slurm101-motion" aria-hidden="true">
      {stops.map((stop, index) => (
        <rect
          key={`stop-${slideIndex}-${index}`}
          x={stop.x}
          y={stop.y}
          width={stop.width}
          height={stop.height}
          rx={stop.rx ?? 8}
          fill="none"
          stroke={pulseColor(index)}
          strokeWidth={1.5}
          className="slurm101-box-pulse"
          style={{
            animationDuration: "2.2s",
            animationDelay: `${1 + stop.at * 2.2}s`,
            animationIterationCount: "infinite",
          }}
          pointerEvents="none"
        />
      ))}
      <circle r={5} fill={signalColor}>
        {lane === "mixed" ? (
          <animate
            attributeName="fill"
            values={`${t.category.green};${t.category.green};${t.category.purple}`}
            keyTimes="0;0.5;1"
            begin="1s"
            dur="2.2s"
            repeatCount="indefinite"
          />
        ) : null}
        <animate
          attributeName="opacity"
          values={opacityValues.join(";")}
          keyTimes={opacityKeyTimes.join(";")}
          begin="1s"
          dur="2.2s"
          repeatCount="indefinite"
        />
        <animateMotion path={path} begin="1s" dur="2.2s" repeatCount="indefinite" />
      </circle>
    </g>
  );
}

function ParallelMotionSignals({ slideIndex }: { slideIndex: 2 | 13 | 14 }) {
  const t = useHostTheme();
  const color = t.category.green;
  const isTaskFork = slideIndex === 2;
  const isFanOut = slideIndex === 13;
  const sourceColor = isTaskFork ? t.category.green : isFanOut ? t.category.yellow : t.category.green;
  const targetColor = isTaskFork ? t.stroke.primary : isFanOut ? t.category.green : t.category.purple;
  const source =
    isTaskFork
      ? { x: 190, y: 78, width: 320, height: 58, rx: 8 }
      : { x: 48, y: 124, width: 162, height: 108, rx: 10 };
  const forkTrunk = isTaskFork ? "M350 136 V168" : "M210 178 H310";
  const branchPaths = isTaskFork
    ? ["M350 168 H150 V198", "M350 168 V198", "M350 168 H550 V198"]
    : isFanOut
      ? ["M310 178 H382 V118 H410", "M310 178 H410", "M310 178 H382 V238 H410"]
      : ["M290 118 H318 V178 H390", "M290 178 H390", "M290 238 H318 V178 H390"];
  const branchBoxes = isTaskFork
    ? [
        { x: 72, y: 198, width: 156, height: 88, rx: 8 },
        { x: 272, y: 198, width: 156, height: 88, rx: 8 },
        { x: 472, y: 198, width: 156, height: 88, rx: 8 },
      ]
    : isFanOut
      ? [
          { x: 410, y: 93, width: 186, height: 50, rx: 8 },
          { x: 410, y: 153, width: 186, height: 50, rx: 8 },
          { x: 410, y: 213, width: 186, height: 50, rx: 8 },
        ]
      : [
          { x: 104, y: 93, width: 186, height: 50, rx: 8 },
          { x: 104, y: 153, width: 186, height: 50, rx: 8 },
          { x: 104, y: 213, width: 186, height: 50, rx: 8 },
        ];

  if (slideIndex === 14) {
    return (
      <g className="slurm101-motion" aria-hidden="true">
        {branchBoxes.map((box, index) => (
          <rect
            key={`source-${index}`}
            {...box}
            fill="none"
            stroke={sourceColor}
            strokeWidth={1.5}
            className="slurm101-box-pulse"
            style={{ animationDuration: "2.2s", animationDelay: "0.8s", animationIterationCount: 3 }}
          />
        ))}
        {branchPaths.map((path, index) => (
          <circle key={`branch-${index}`} r={5} fill={color} opacity={0}>
            <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" begin="1.2s;3.4s;5.6s" dur="0.7s" />
            <animateMotion path={path} begin="1.2s;3.4s;5.6s" dur="0.7s" />
          </circle>
        ))}
        <circle r={5.5} fill={color} opacity={0}>
          <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" begin="1.9s;4.1s;6.3s" dur="0.4s" />
          <animateMotion path="M390 178 H454" begin="1.9s;4.1s;6.3s" dur="0.4s" />
        </circle>
        <rect
          x={454}
          y={124}
          width={166}
          height={108}
          rx={10}
          fill="none"
          stroke={targetColor}
          strokeWidth={1.5}
          className="slurm101-box-pulse"
          style={{ animationDuration: "2.2s", animationDelay: "2.25s", animationIterationCount: 3 }}
        />
      </g>
    );
  }

  return (
    <g className="slurm101-motion" aria-hidden="true">
      <rect
        {...source}
        fill="none"
        stroke={sourceColor}
        strokeWidth={1.5}
        className="slurm101-box-pulse"
        style={{ animationDuration: "2.2s", animationDelay: "0.8s", animationIterationCount: 3 }}
      />
      <circle r={5.5} fill={color} opacity={0}>
        <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" begin="1.05s;3.25s;5.45s" dur="0.4s" />
        <animateMotion path={forkTrunk} begin="1.05s;3.25s;5.45s" dur="0.4s" />
      </circle>
      {branchPaths.map((path, index) => (
        <circle key={`branch-${index}`} r={5} fill={color} opacity={0}>
          <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" begin="1.45s;3.65s;5.85s" dur="0.7s" />
          <animateMotion path={path} begin="1.45s;3.65s;5.85s" dur="0.7s" />
        </circle>
      ))}
      {branchBoxes.map((box, index) => (
        <rect
          key={`target-${index}`}
          {...box}
          fill="none"
          stroke={targetColor}
          strokeWidth={1.5}
          className="slurm101-box-pulse"
          style={{ animationDuration: "2.2s", animationDelay: "2.1s", animationIterationCount: 3 }}
        />
      ))}
    </g>
  );
}

export function MentalModel({ slide, slideIndex }: { slide: SlideDefinition; slideIndex: number }) {
  const t = useHostTheme();

  return (
    <div
      style={{
        border: `1px solid ${t.stroke.secondary}`,
        borderRadius: 8,
        background: t.bg.editor,
      }}
    >
      <svg className="slurm101-live" viewBox="0 0 700 400" role="img" aria-label="Evolving picture of the SLURM and Velo benchmark" style={{ display: "block", width: "100%" }}>
        <defs>
          <marker id="s101-blue" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.green} />
          </marker>
          <marker id="s101-green" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.green} />
          </marker>
          <marker id="s101-purple" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.purple} />
          </marker>
          <marker id="s101-yellow" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.category.yellow} />
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
            <rect x={74} y={138} width={220} height={96} rx={10} fill={t.bg.elevated} stroke={t.stroke.secondary} />
            <text x={184} y={176} textAnchor="middle" fill={t.text.primary} fontSize={16} fontWeight={700}>Many load generators</text>
            <text x={184} y={199} textAnchor="middle" fill={t.text.secondary} fontSize={12}>more machines = more traffic</text>
            <path d="M294 186 H418" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-green)" />
            <text x={356} y={172} textAnchor="middle" fill={t.category.green} fontSize={11} fontWeight={700}>REQUESTS</text>
            <rect x={418} y={138} width={210} height={96} rx={36} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.6} />
            <text x={523} y={178} textAnchor="middle" fill={t.text.primary} fontSize={16} fontWeight={700}>Inference server</text>
            <text x={523} y={201} textAnchor="middle" fill={t.text.secondary} fontSize={12}>the AI model endpoint</text>
          </g>
        ) : null}

        {slideIndex === 1 ? (
          <g>
            <rect x={64} y={128} width={190} height={96} rx={10} fill={t.bg.elevated} stroke={t.category.yellow} strokeWidth={1.6} />
            <text x={159} y={166} textAnchor="middle" fill={t.text.primary} fontSize={16} fontWeight={700}>SLURM scheduler</text>
            <text x={159} y={190} textAnchor="middle" fill={t.text.secondary} fontSize={12}>finds and reserves machines</text>
            <path d="M254 176 H330" fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" />
            <rect x={330} y={98} width={304} height={156} rx={10} fill={t.bg.elevated} stroke={t.stroke.secondary} />
            <text x={482} y={126} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Your allocation</text>
            {[0, 1, 2, 3].map((task) => (
              <g key={task}>
                <rect x={350 + task * 68} y={150} width={54} height={58} rx={6} fill={t.fill.quaternary} stroke={t.stroke.secondary} />
                <text x={377 + task * 68} y={176} textAnchor="middle" fill={t.text.primary} fontSize={12} fontWeight={650}>{`Task ${task}`}</text>
                <text x={377 + task * 68} y={195} textAnchor="middle" fill={t.text.secondary} fontSize={10}>machine</text>
              </g>
            ))}
          </g>
        ) : null}

        {slideIndex === 2 ? (
          <g>
            <rect x={190} y={78} width={320} height={58} rx={8} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.8} />
            <text x={350} y={102} textAnchor="middle" fill={t.text.primary} fontSize={13} fontWeight={700}>aiperf slurm run --config benchmark.yaml</text>
            <text x={350} y={122} textAnchor="middle" fill={t.text.secondary} fontSize={12}>one identical command on every task</text>
            {[150, 350, 550].map((cx, task) => (
              <g key={task}>
                <rect x={cx - 78} y={198} width={156} height={88} rx={8} fill={t.bg.elevated} stroke={t.stroke.secondary} strokeWidth={1.4} />
                <text x={cx} y={230} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>{`Task ${task}`}</text>
                <text x={cx} y={254} textAnchor="middle" fill={t.text.secondary} fontSize={12}>{`SLURM_PROCID = ${task}`}</text>
                <text x={cx} y={274} textAnchor="middle" fill={t.text.tertiary} fontSize={11}>rank decides the role</text>
              </g>
            ))}
            <path d="M350 136 V168" fill="none" stroke={t.category.green} strokeWidth={2.5} />
            <path d="M150 168 H550" fill="none" stroke={t.category.green} strokeWidth={2.5} />
            {[150, 350, 550].map((cx) => (
              <path key={cx} d={`M${cx} 168 V198`} fill="none" stroke={t.category.green} strokeWidth={2.5} markerEnd="url(#s101-blue)" />
            ))}
          </g>
        ) : null}

        {slideIndex === 3 ? (
          <g>
            <rect x={72} y={122} width={220} height={112} rx={10} fill={t.bg.elevated} stroke={t.category.yellow} strokeWidth={1.8} />
            <text x={182} y={158} textAnchor="middle" fill={t.text.primary} fontSize={17} fontWeight={700}>Rank 0</text>
            <text x={182} y={185} textAnchor="middle" fill={t.category.yellow} fontSize={14} fontWeight={700}>CONTROLLER</text>
            <text x={182} y={211} textAnchor="middle" fill={t.text.secondary} fontSize={12}>coordinates the benchmark</text>
            <rect x={408} y={98} width={220} height={160} rx={10} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.8} />
            <text x={518} y={131} textAnchor="middle" fill={t.text.primary} fontSize={17} fontWeight={700}>Ranks 1, 2, 3…</text>
            <text x={518} y={157} textAnchor="middle" fill={t.category.green} fontSize={14} fontWeight={700}>LOAD CELLS</text>
            <text x={518} y={183} textAnchor="middle" fill={t.text.secondary} fontSize={12}>cell 0 · cell 1 · cell 2…</text>
            <text x={518} y={208} textAnchor="middle" fill={t.text.secondary} fontSize={12}>send benchmark requests</text>
            <path d="M292 178 H408" fill="none" stroke={t.stroke.secondary} strokeWidth={2} />
            <text x={350} y={165} textAnchor="middle" fill={t.text.secondary} fontSize={11} fontWeight={650}>SAME RUN</text>
          </g>
        ) : null}

        {slideIndex === 4 ? (
          <g>
            <rect x={38} y={116} width={142} height={96} rx={8} fill={t.bg.elevated} stroke={t.stroke.secondary} />
            <text x={109} y={150} textAnchor="middle" fill={t.text.primary} fontSize={14} fontWeight={700}>SLURM nodelist</text>
            <text x={109} y={175} textAnchor="middle" fill={t.text.secondary} fontSize={11}>node01, node02…</text>
            <text x={109} y={195} textAnchor="middle" fill={t.text.secondary} fontSize={11}>+ port 9500</text>
            <path d="M180 164 H236" fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" />
            <rect x={236} y={116} width={190} height={96} rx={8} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.6} />
            <text x={331} y={150} textAnchor="middle" fill={t.text.primary} fontSize={14} fontWeight={700}>Controller coordinate</text>
            <text x={331} y={179} textAnchor="middle" fill={t.category.green} fontSize={13} fontWeight={700}>tcp://node01:9500</text>
            <path d="M426 164 H482" fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" />
            <rect x={482} y={102} width={180} height={124} rx={8} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.6} />
            <text x={572} y={137} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Every cell dials</text>
            <text x={572} y={163} textAnchor="middle" fill={t.text.secondary} fontSize={12}>the same address</text>
            <text x={572} y={188} textAnchor="middle" fill={t.text.secondary} fontSize={12}>and reaches rank 0</text>
            <text x={572} y={210} textAnchor="middle" fill={t.text.tertiary} fontSize={10.5}>no discovery service</text>
          </g>
        ) : null}

        {slideIndex === 5 ? (
          <g>
            <rect x={48} y={100} width={180} height={150} rx={10} fill={t.bg.elevated} stroke={t.category.yellow} strokeWidth={1.8} />
            <text x={138} y={140} textAnchor="middle" fill={t.text.primary} fontSize={16} fontWeight={700}>Controller</text>
            <text x={138} y={165} textAnchor="middle" fill={t.text.secondary} fontSize={12}>rank 0</text>
            <text x={138} y={195} textAnchor="middle" fill={t.text.secondary} fontSize={11}>binds Velo listener</text>
            <text x={138} y={218} textAnchor="middle" fill={t.text.secondary} fontSize={11}>on port 9500</text>

            <rect x={270} y={132} width={160} height={86} rx={10} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={2} />
            <text x={350} y={166} textAnchor="middle" fill={t.category.green} fontSize={18} fontWeight={700}>VELO</text>
            <text x={350} y={192} textAnchor="middle" fill={t.text.secondary} fontSize={12}>control walkie-talkie</text>

            <rect x={472} y={100} width={180} height={150} rx={10} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.8} />
            <text x={562} y={140} textAnchor="middle" fill={t.text.primary} fontSize={16} fontWeight={700}>Load cells</text>
            <text x={562} y={165} textAnchor="middle" fill={t.text.secondary} fontSize={12}>ranks 1…N</text>
            <text x={562} y={195} textAnchor="middle" fill={t.text.secondary} fontSize={11}>dial controller</text>
            <text x={562} y={218} textAnchor="middle" fill={t.text.secondary} fontSize={11}>over Velo</text>

            <path d="M228 155 H270" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" />
            <path d="M430 155 H472" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" />
            <path d="M472 195 H430" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" />
            <path d="M270 195 H228" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" />
            <text x={249} y={143} textAnchor="middle" fill={t.category.green} fontSize={9.5} fontWeight={700}>SEND</text>
            <text x={451} y={215} textAnchor="middle" fill={t.category.green} fontSize={9.5} fontWeight={700}>REPLY</text>
            <text x={350} y={300} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              Velo carries short control messages — not the AI requests.
            </text>
          </g>
        ) : null}

        {slideIndex === 6 ? (
          <g>
            {[
              ["Cell knows", "tcp://node01:9500"],
              ["velo.connect", "hello handshake"],
              ["Peers registered", "controller ↔ cell"],
              ["Ready to talk", "named handlers"],
            ].map(([title, detail], index) => {
              const x = 28 + index * 168;
              return (
                <g key={title}>
                  <rect x={x} y={128} width={140} height={100} rx={9} fill={t.bg.elevated} stroke={index === 1 ? t.category.green : t.stroke.secondary} strokeWidth={index === 1 ? 2 : 1.3} />
                  <text x={x + 70} y={168} textAnchor="middle" fill={t.text.primary} fontSize={13} fontWeight={700}>{title}</text>
                  <text x={x + 70} y={196} textAnchor="middle" fill={t.text.secondary} fontSize={11}>{detail}</text>
                  {index < 3 ? <path d={`M${x + 140} 178 H${x + 168}`} fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" /> : null}
                </g>
              );
            })}
            <path d="M602 248 H98" fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" />
            <text x={350} y={268} textAnchor="middle" fill={t.category.green} fontSize={10.5} fontWeight={700}>HELLO REPLY RETURNS TO THE CELL</text>
            <text x={350} y={285} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              No DNS discovery service. One a-priori address is enough.
            </text>
          </g>
        ) : null}

        {slideIndex === 7 ? (
          <g>
            {[
              ["Connect", "Velo hello"],
              ["Register", "aiperf.cell.register"],
              ["Get slice", "sliced envelope"],
              ["Await START", "EventHandle"],
              ["START!", "all cells release"],
            ].map(([title, detail], index) => {
              const x = 20 + index * 136;
              return (
                <g key={title}>
                  <rect x={x} y={120} width={112} height={100} rx={8} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={index === 4 ? 2 : 1.4} />
                  <text x={x + 56} y={158} textAnchor="middle" fill={t.text.primary} fontSize={12.5} fontWeight={700}>{title}</text>
                  <text x={x + 56} y={186} textAnchor="middle" fill={t.text.secondary} fontSize={10}>{detail}</text>
                  {index < 4 ? <path d={`M${x + 112} 170 H${x + 136}`} fill="none" stroke={t.category.green} strokeWidth={2} markerEnd="url(#s101-blue)" /> : null}
                </g>
              );
            })}
            <text x={350} y={280} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              All of this readiness choreography rides on Velo.
            </text>
          </g>
        ) : null}

        {slideIndex === 8 ? (
          <g>
            <rect x={40} y={100} width={170} height={120} rx={10} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.8} />
            <text x={125} y={140} textAnchor="middle" fill={t.text.primary} fontSize={16} fontWeight={700}>Load cells</text>
            <text x={125} y={168} textAnchor="middle" fill={t.text.secondary} fontSize={12}>send requests</text>
            <text x={125} y={192} textAnchor="middle" fill={t.text.secondary} fontSize={12}>measure replies</text>

            <path d="M210 160 H430" fill="none" stroke={t.category.green} strokeWidth={3} markerEnd="url(#s101-green)" />
            <text x={320} y={145} textAnchor="middle" fill={t.category.green} fontSize={12} fontWeight={700}>HTTP / gRPC</text>
            <text x={320} y={180} textAnchor="middle" fill={t.text.secondary} fontSize={11}>NOT Velo</text>

            <rect x={430} y={100} width={170} height={120} rx={36} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.8} />
            <text x={515} y={148} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Inference server</text>
            <text x={515} y={176} textAnchor="middle" fill={t.text.secondary} fontSize={12}>AI endpoint</text>

            <rect x={230} y={268} width={240} height={56} rx={8} fill={t.fill.quaternary} stroke={t.category.green} strokeWidth={1.4} />
            <text x={350} y={292} textAnchor="middle" fill={t.category.green} fontSize={13} fontWeight={700}>Velo heartbeats only</text>
            <text x={350} y={311} textAnchor="middle" fill={t.text.secondary} fontSize={11}>progress summaries → rank 0</text>
          </g>
        ) : null}

        {slideIndex === 9 ? (
          <g>
            {[
              [t.category.green, "VELO", "control", "register · START · heartbeat · partition"],
              [t.category.green, "HTTP / gRPC", "load", "benchmark requests to the AI server"],
              [t.category.purple, "HTTP/1 + zstd", "bulk files", "large per-record artifact uploads"],
            ].map(([color, title, role, detail], index) => {
              const y = 84 + index * 92;
              return (
                <g key={title}>
                  <rect x={60} y={y} width={580} height={76} rx={10} fill={t.bg.elevated} stroke={color} strokeWidth={1.8} />
                  <text x={90} y={y + 32} fill={color} fontSize={16} fontWeight={700}>{title}</text>
                  <text x={260} y={y + 32} fill={t.text.primary} fontSize={14} fontWeight={650}>{role}</text>
                  <text x={90} y={y + 56} fill={t.text.secondary} fontSize={12}>{detail}</text>
                </g>
              );
            })}
          </g>
        ) : null}

        {slideIndex === 10 ? (
          <g>
            {[
              ["Load cells", "finish local work"],
              ["Velo ship", "partition / store"],
              ["Rank-0 controller", "receives + merges"],
              ["One report", "global benchmark"],
            ].map(([title, detail], index) => {
              const x = 26 + index * 170;
              const color = index < 3 ? t.category.green : t.category.purple;
              return (
                <g key={title}>
                  <rect x={x} y={126} width={136} height={104} rx={9} fill={t.bg.elevated} stroke={color} strokeWidth={1.7} />
                  <text x={x + 68} y={164} textAnchor="middle" fill={t.text.primary} fontSize={13.5} fontWeight={700}>{title}</text>
                  <text x={x + 68} y={190} textAnchor="middle" fill={t.text.secondary} fontSize={11}>{detail}</text>
                  {index < 3 ? <path d={`M${x + 136} 178 H${x + 170}`} fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" /> : null}
                </g>
              );
            })}
            <text x={350} y={280} textAnchor="middle" fill={t.text.primary} fontSize={13} fontWeight={650}>
              Result summaries travel on Velo back to the original rank-0 controller.
            </text>
            <text x={350} y={305} textAnchor="middle" fill={t.text.secondary} fontSize={12}>
              Merge is work inside that controller process — not a separate “merge machine.”
            </text>
          </g>
        ) : null}

        {slideIndex === 11 ? (
          <g>
            <rect x={48} y={110} width={250} height={150} rx={10} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.8} />
            <text x={173} y={148} textAnchor="middle" fill={t.category.green} fontSize={15} fontWeight={700}>VELO PATH</text>
            <text x={173} y={178} textAnchor="middle" fill={t.text.primary} fontSize={13}>small messages</text>
            <text x={173} y={204} textAnchor="middle" fill={t.text.secondary} fontSize={12}>register · START</text>
            <text x={173} y={228} textAnchor="middle" fill={t.text.secondary} fontSize={12}>heartbeat · partition</text>

            <rect x={402} y={110} width={250} height={150} rx={10} fill={t.bg.elevated} stroke={t.category.purple} strokeWidth={1.8} />
            <text x={527} y={148} textAnchor="middle" fill={t.category.purple} fontSize={15} fontWeight={700}>HTTP ARTIFACT PATH</text>
            <text x={527} y={178} textAnchor="middle" fill={t.text.primary} fontSize={13}>large files</text>
            <text x={527} y={204} textAnchor="middle" fill={t.text.secondary} fontSize={12}>per-record exports</text>
            <text x={527} y={228} textAnchor="middle" fill={t.text.secondary} fontSize={12}>zstd upload + concat</text>

            <text x={350} y={310} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              Same controller host. Different purpose. Different port.
            </text>
          </g>
        ) : null}

        {slideIndex === 12 ? (
          <g>
            <rect x={60} y={110} width={260} height={150} rx={10} fill={t.bg.elevated} stroke={t.category.yellow} strokeWidth={1.8} />
            <text x={190} y={148} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Dedicated ROLE</text>
            <text x={190} y={178} textAnchor="middle" fill={t.category.yellow} fontSize={13} fontWeight={700}>always useful</text>
            <text x={190} y={208} textAnchor="middle" fill={t.text.secondary} fontSize={12}>keeps merge + sync</text>
            <text x={190} y={232} textAnchor="middle" fill={t.text.secondary} fontSize={12}>off the measured path</text>

            <rect x={380} y={110} width={260} height={150} rx={10} fill={t.bg.elevated} stroke={t.stroke.secondary} strokeWidth={1.5} />
            <text x={510} y={148} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Dedicated NODE</text>
            <text x={510} y={178} textAnchor="middle" fill={t.text.secondary} fontSize={13} fontWeight={700}>optional</text>
            <text x={510} y={208} textAnchor="middle" fill={t.text.secondary} fontSize={12}>default script uses one</text>
            <text x={510} y={232} textAnchor="middle" fill={t.text.secondary} fontSize={12}>task per node</text>

            <text x={350} y={310} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              Co-locate controller + cell when machines are scarce.
            </text>
          </g>
        ) : null}

        {slideIndex === 13 ? (
          <g>
            <rect x={48} y={124} width={162} height={108} rx={10} fill={t.bg.elevated} stroke={t.category.yellow} strokeWidth={1.8} />
            <text x={129} y={160} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Rank 0</text>
            <text x={129} y={184} textAnchor="middle" fill={t.category.yellow} fontSize={12} fontWeight={700}>GLOBAL PLAN</text>
            <text x={129} y={208} textAnchor="middle" fill={t.text.secondary} fontSize={11}>requests 0…N</text>

            <circle cx={310} cy={178} r={18} fill={t.fill.quaternary} stroke={t.category.green} strokeWidth={1.7} />
            <text x={310} y={182} textAnchor="middle" fill={t.category.green} fontSize={11} fontWeight={700}>Velo</text>
            <path d="M210 178 H292" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" />

            {[
              ["Cell 0", "slice 0", 118],
              ["Cell 1", "slice 1", 178],
              ["Cell 2", "slice 2", 238],
            ].map(([cell, slice, y]) => (
              <g key={cell}>
                <path d={`M328 178 H382 V${y} H410`} fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" />
                <rect x={410} y={Number(y) - 25} width={186} height={50} rx={8} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.5} />
                <text x={432} y={Number(y) - 3} fill={t.text.primary} fontSize={12.5} fontWeight={700}>{cell}</text>
                <text x={578} y={Number(y) - 3} textAnchor="end" fill={t.category.green} fontSize={11.5} fontWeight={650}>{slice}</text>
                <text x={432} y={Number(y) + 15} fill={t.text.secondary} fontSize={10}>disjoint owned work</text>
              </g>
            ))}
            <text x={350} y={305} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              One plan branches into different slices — not copies of the same work.
            </text>
          </g>
        ) : null}

        {slideIndex === 14 ? (
          <g>
            {[
              ["Cell 0", "partition 0", 118],
              ["Cell 1", "partition 1", 178],
              ["Cell 2", "partition 2", 238],
            ].map(([cell, partition, y]) => (
              <g key={cell}>
                <rect x={104} y={Number(y) - 25} width={186} height={50} rx={8} fill={t.bg.elevated} stroke={t.category.green} strokeWidth={1.5} />
                <text x={126} y={Number(y) - 3} fill={t.text.primary} fontSize={12.5} fontWeight={700}>{cell}</text>
                <text x={272} y={Number(y) - 3} textAnchor="end" fill={t.category.green} fontSize={11.5} fontWeight={650}>{partition}</text>
                <text x={126} y={Number(y) + 15} fill={t.text.secondary} fontSize={10}>finished local result</text>
                <path d={`M290 ${y} H318 V178 H372`} fill="none" stroke={t.category.green} strokeWidth={2.2} markerEnd="url(#s101-blue)" />
              </g>
            ))}

            <circle cx={390} cy={178} r={18} fill={t.fill.quaternary} stroke={t.category.green} strokeWidth={1.7} />
            <text x={390} y={182} textAnchor="middle" fill={t.category.green} fontSize={11} fontWeight={700}>Velo</text>
            <path d="M408 178 H454" fill="none" stroke={t.category.green} strokeWidth={2.4} markerEnd="url(#s101-blue)" />

            <rect x={454} y={124} width={166} height={108} rx={10} fill={t.bg.elevated} stroke={t.category.purple} strokeWidth={1.8} />
            <text x={537} y={158} textAnchor="middle" fill={t.text.primary} fontSize={15} fontWeight={700}>Rank 0</text>
            <text x={537} y={183} textAnchor="middle" fill={t.category.purple} fontSize={12} fontWeight={700}>MERGE</text>
            <text x={537} y={208} textAnchor="middle" fill={t.text.secondary} fontSize={11}>one global report</text>
            <text x={350} y={305} textAnchor="middle" fill={t.text.secondary} fontSize={13}>
              Three terminal results converge; rank 0 combines them once all arrive.
            </text>
          </g>
        ) : null}

        {slideIndex === 15 ? (
          <g>
            {[
              ["Config v2", "benchmark.yaml"],
              ["Generate", "job.sbatch"],
              ["Submit", "sbatch job.sbatch"],
              ["SLURM + Velo", "roles + messaging"],
              ["AIPerf report", "merged result"],
            ].map(([title, detail], index) => {
              const x = 20 + index * 136;
              return (
                <g key={title}>
                  <rect x={x} y={126} width={108} height={94} rx={8} fill={t.bg.elevated} stroke={index === 4 ? t.category.purple : t.stroke.secondary} strokeWidth={index === 4 ? 1.8 : 1.3} />
                  <text x={x + 54} y={162} textAnchor="middle" fill={t.text.primary} fontSize={12.5} fontWeight={700}>{title}</text>
                  <text x={x + 54} y={188} textAnchor="middle" fill={t.text.secondary} fontSize={10}>{detail}</text>
                  {index < 4 ? <path d={`M${x + 108} 173 H${x + 136}`} fill="none" stroke={t.category.green} strokeWidth={2} markerEnd="url(#s101-blue)" /> : null}
                </g>
              );
            })}
          </g>
        ) : null}
        <MotionSignals slideIndex={slideIndex} />
      </svg>
      <div style={{ padding: "12px 16px", borderTop: `1px solid ${t.stroke.tertiary}` }}>
        <Text tone="secondary" weight="medium">{slide.caption}</Text>
      </div>
    </div>
  );
}
