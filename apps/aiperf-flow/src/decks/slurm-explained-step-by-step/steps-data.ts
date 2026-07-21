/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Content ground truth for the "SLURM + Velo from scratch" step-by-step explainer, ported
//! verbatim from the Cursor canvas `slurm-explained-step-by-step.canvas.tsx`. A 16-step
//! walkthrough of how AIPerf uses SLURM to launch a cellular benchmark and Velo to coordinate
//! the controller (rank 0) and its load-generating cells. Prose (eyebrow/title/lede/term/points/
//! caption), one-line narration, and the per-step scene labels/notes all live here so the deck
//! and diagram components share one authoritative copy.

/** One walkthrough step, mirroring the canvas source's `Step` shape. */
export type Step = {
  eyebrow: string;
  title: string;
  lede: string;
  term?: { word: string; meaning: string };
  points: readonly string[];
  caption: string;
};

export const STEPS: readonly Step[] = [
  {
    eyebrow: "The problem",
    title: "You want to load-test a big AI server",
    lede:
      "AIPerf sends many requests to an inference server and measures how fast it answers. To push a really large server hard, one computer sending traffic is not enough — you need many computers sending at once.",
    term: {
      word: "Inference server",
      meaning: "The service that runs an AI model and answers requests (for example, a chat completion endpoint).",
    },
    points: [
      "One laptop can only send so many requests per second.",
      "To stress a large server you need a fleet of machines generating load together.",
      "That fleet has to act like one coordinated test, not many disconnected ones.",
    ],
    caption: "Goal: many machines, one benchmark, one result.",
  },
  {
    eyebrow: "The tool",
    title: "SLURM hands you a cluster of machines",
    lede:
      "SLURM is the software that shares a big cluster among many people. You ask it for machines; it finds free ones, reserves them for you, and runs your command on every one of them at the same time.",
    term: {
      word: "SLURM",
      meaning:
        "A job scheduler for compute clusters. You submit a request for N machines; SLURM grants an allocation and launches your program on each task.",
    },
    points: [
      "`sbatch` submits a batch job; `srun` launches tasks right now.",
      "An allocation is your reserved set of machines for this job.",
      "SLURM runs the exact same command on every task in the allocation.",
    ],
    caption: "SLURM = the landlord that lends you machines for a while.",
  },
  {
    eyebrow: "The key trick",
    title: "Every machine runs the identical command",
    lede:
      "This is the part that surprises newcomers. SLURM does not run a different program on each machine. It launches the very same line — `aiperf slurm run` — on all of them at once.",
    term: {
      word: "Task",
      meaning: "One running copy of your command inside the allocation. A 4-machine job has 4 tasks.",
    },
    points: [
      "So how does each copy know what to do differently?",
      "SLURM gives each task a numbered identity called its rank.",
      "The program reads that number and decides its own job.",
    ],
    caption: "Same command everywhere — the rank number breaks the tie.",
  },
  {
    eyebrow: "Splitting the roles",
    title: "Rank 0 leads; everyone else does the work",
    lede:
      "AIPerf reads the rank SLURM assigned. Rank 0 becomes the controller — the coordinator. Every other rank becomes a cell — a worker that actually sends load to the server.",
    term: {
      word: "Rank",
      meaning: "Each task's index in the allocation, from 0 up. AIPerf uses it to pick controller vs. cell.",
    },
    points: [
      "Rank 0 → the single controller (it coordinates, it does not send benchmark load).",
      "Ranks 1, 2, 3, … → cells, numbered cell_id = rank − 1.",
      "So a 4-task job = 1 controller + 3 cells.",
    ],
    caption: "controller = rank 0 · cell_id = rank − 1 · cell_count = tasks − 1.",
  },
  {
    eyebrow: "Finding each other",
    title: "Cells dial the controller with one shared fact",
    lede:
      "The cells need to talk to the controller, but nobody set up a directory service. Instead, every task computes the same address from the SLURM environment: the first machine in the allocation, on a known port.",
    term: {
      word: "Coordinate",
      meaning: "The controller's address, like tcp://node01:9500, derived identically on every task from SLURM_* variables.",
    },
    points: [
      "SLURM tells every task the list of machines and its own rank.",
      "All tasks agree rank 0 lives on the first machine in that list.",
      "Default connection port is 9500, so every cell knows exactly where to call.",
    ],
    caption: "One fact, computed the same everywhere — no discovery service needed.",
  },
  {
    eyebrow: "Meet Velo",
    title: "Velo is the walkie-talkie between controller and cells",
    lede:
      "Once cells know the controller address, they still need a messaging system. That system is Velo: a small control-plane library AIPerf uses so the controller and cells can exchange short messages across machines.",
    term: {
      word: "Velo",
      meaning:
        "An async messaging framework used only for cellular control traffic — registration, START sync, heartbeats, and result summaries. It is never on the per-request hot path.",
    },
    points: [
      "Velo is how AIPerf processes talk to each other across hosts.",
      "It carries small control messages, not the benchmark requests themselves.",
      "Without cellular mode (`--cells`), Velo is not constructed at all.",
    ],
    caption: "SLURM launches the processes. Velo lets those processes talk.",
  },
  {
    eyebrow: "Velo bootstrap",
    title: "A cell connects once, then Velo learns the peer",
    lede:
      "Each cell calls Velo with the one known address: tcp://HOST:9500. Velo's hello handshake discovers the controller's real peer identity, then the two sides can send named messages to each other.",
    term: {
      word: "Handshake",
      meaning: "Velo's connect step that turns a host:port into a live, mutually registered peer connection.",
    },
    points: [
      "The cell only needs AIPERF_CELL_CONTROLLER_ADDR.",
      "Velo connects, shakes hands, and registers both peers.",
      "After that, named handlers like register / heartbeat / partition work.",
    ],
    caption: "Address in → peer connection out. No service discovery backend.",
  },
  {
    eyebrow: "Getting ready together",
    title: "Register and START travel over Velo",
    lede:
      "Before any traffic flies, each cell registers over Velo. The controller replies with that cell's work slice and a START event handle. When every expected cell has registered, the controller triggers START and they begin together.",
    term: {
      word: "START barrier",
      meaning: "A Velo event gate that holds every cell until all of them have registered, so the benchmark starts in sync.",
    },
    points: [
      "aiperf.cell.register — cell joins and receives its sliced envelope.",
      "The controller waits until cell_count cells have registered.",
      "One START trigger releases every waiting cell at once.",
    ],
    caption: "Line everyone up over Velo, then start the race together.",
  },
  {
    eyebrow: "Doing the work",
    title: "Benchmark requests do NOT use Velo",
    lede:
      "After START, cells send their share of requests straight to the inference server over HTTP or gRPC. Velo stays out of that path — it only carries lightweight heartbeats back to the controller.",
    term: {
      word: "Hot path",
      meaning: "The per-request and per-token traffic that measures latency. In AIPerf that path is HTTP/gRPC, never Velo.",
    },
    points: [
      "Each cell sends only its assigned slice — no overlap, no gaps.",
      "Request traffic goes cell → inference server directly.",
      "The controller is not a bottleneck: it never sits in the request path.",
    ],
    caption: "Velo coordinates. HTTP/gRPC generates the measured load.",
  },
  {
    eyebrow: "Three planes",
    title: "Three completely different kinds of traffic",
    lede:
      "It helps to keep three roads separate in your head. Velo is only the control road. The load road hits the AI server. Large result files take a third bulk-upload road.",
    term: {
      word: "Traffic plane",
      meaning: "A separate purpose for network traffic: control messages, benchmark load, or bulk artifact files.",
    },
    points: [
      "Velo — register, START, heartbeats, result partitions / stores.",
      "HTTP / gRPC — the real benchmark requests to the inference server.",
      "HTTP/1 + zstd — large per-record artifact files, not carried on Velo.",
    ],
    caption: "Mixing these up is the main source of confusion.",
  },
  {
    eyebrow: "One answer",
    title: "Result partitions return to rank 0 over Velo",
    lede:
      "When the run ends, each cell ships its measurements back to the original rank-0 controller over Velo. The controller merges every cell's numbers into one report.",
    term: {
      word: "Partition",
      meaning: "One cell's bundle of results (raw records or a pre-summarized store) sent to the controller at the end over Velo.",
    },
    points: [
      "aiperf.cell.partition or store_partition — one terminal ship per cell.",
      "Merge happens inside the original rank-0 controller process.",
      "The report looks like one benchmark, not N separate jobs.",
    ],
    caption: "Cells measure. Rank 0 merges. One authoritative report.",
  },
  {
    eyebrow: "Bulk files",
    title: "Huge per-record files take a different road",
    lede:
      "If the run keeps large per-record artifact files, those bytes do not ride Velo. They upload over a separate HTTP/1 path with zstd compression. The controller concatenates them after every cell finishes.",
    term: {
      word: "Artifact plane",
      meaning: "A second network path for bulk files, derived from the same controller host with a different port.",
    },
    points: [
      "Velo stays small: control messages and metric summaries.",
      "Bulk files use HTTP/1 + zstd so they do not clog the control plane.",
      "Synthetic / summary-only runs may never need this path.",
    ],
    caption: "Small control on Velo. Big files on HTTP.",
  },
  {
    eyebrow: "Controller cost",
    title: "Why spend a whole rank on a non-loading process?",
    lede:
      "A dedicated controller rank keeps coordination, START sync, merging, and artifact handling away from measured load. That does not always mean a dedicated node — only a dedicated role.",
    term: {
      word: "Dedicated rank",
      meaning: "A process role, not necessarily a whole machine. Rank 0 can share a node with a cell if isolation is less important than allocation efficiency.",
    },
    points: [
      "Dedicated role: yes — keeps measurement clean and merge simple.",
      "Dedicated node: optional — the script defaults to one task per node.",
      "Co-locate controller + cell when machine count is scarce.",
    ],
    caption: "Pay for coordination. Do not always pay for a whole idle node.",
  },
  {
    eyebrow: "Fan-out",
    title: "Rank 0 fans distinct work slices out to the cells",
    lede:
      "The controller begins with one global benchmark plan. As each cell registers over Velo, rank 0 replies with that cell's sliced envelope so the cells divide the work without overlap or omissions.",
    term: {
      word: "Fan-out",
      meaning: "One source distributes different pieces of a larger job to several workers.",
    },
    points: [
      "The controller owns the one resolved Config v2 benchmark plan.",
      "Cell 0 gets slice 0, cell 1 gets slice 1, and so on.",
      "Together the slices tile the global request or conversation budget exactly.",
    ],
    caption: "One benchmark plan fans out into disjoint cell-owned slices.",
  },
  {
    eyebrow: "Fan-in",
    title: "Cells fan their finished results back into rank 0",
    lede:
      "At the end, direction reverses. Every cell sends exactly one terminal partition or folded store over Velo. Rank 0 collects all expected children and merges them into the global result.",
    term: {
      word: "Fan-in",
      meaning: "Several workers return their outputs to one collector, which combines them.",
    },
    points: [
      "Each cell ships one result partition or one folded metric store.",
      "Rank 0 waits for every expected cell before completing the merge.",
      "The merged output becomes one authoritative AIPerf report.",
    ],
    caption: "Many cell results fan in to the original rank-0 controller.",
  },
  {
    eyebrow: "Try it",
    title: "The two commands you actually type",
    lede:
      "You rarely wire this up by hand. AIPerf generates a ready-to-submit SLURM script for you, and a single command runs the whole cellular benchmark inside the allocation.",
    points: [
      "Generate a submission script from your benchmark config.",
      "Submit it with SLURM; every task launches the same run command.",
      "Rank assignment, Velo wiring, and merging all happen automatically.",
    ],
    caption: "You describe the benchmark; AIPerf handles the cluster choreography.",
  },
] as const;

export const NARRATION: readonly string[] = [
  "Large AI servers need many load generators acting together as one benchmark.",
  "SLURM reserves cluster machines and launches your program across all of them.",
  "Every task runs the same AIPerf command, then its rank determines its role.",
  "Rank zero coordinates the benchmark. Every other rank becomes a load-generating cell.",
  "Each cell derives the rank-zero controller address from the shared SLURM allocation.",
  "Velo carries coordination messages between the controller and its remote cells.",
  "Each cell connects once, allowing Velo to establish the peer relationship.",
  "Cells register through Velo, then wait until the controller broadcasts START.",
  "Benchmark requests travel directly from cells to the inference server, never through Velo.",
  "Control, benchmark traffic, and bulk artifacts use three deliberately separate paths.",
  "When work finishes, each cell returns its result partition to rank zero over Velo.",
  "Large per-request artifacts use compressed HTTP instead of crowding the control plane.",
  "A dedicated controller rank keeps coordination responsive while cells generate maximum load.",
  "The controller partitions one global plan into distinct slices and fans them out together.",
  "Cells return their completed slices in parallel, and rank zero merges one final report.",
  "Generate the batch script, submit it, and AIPerf handles ranks, Velo, load, and results.",
] as const;

export const SCENE_LABELS: readonly string[] = [
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

export const SCENE_NOTES: readonly string[] = [
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
