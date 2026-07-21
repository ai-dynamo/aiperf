/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `graph-subsystem-overview.canvas.tsx` (a real Cursor Canvas) onto aiperf-flow's
//! component vocabulary. The single long scroll becomes five `PageTabs`-switched pages —
//! Overview, Credit Flow, Deduplication, Scheduling, Execution — plus a shared manager/developer
//! audience toggle in the TopBar and a closing glossary. Diagram boxes are real React Flow
//! node/edge graphs; interactive walkthroughs use `useStepSimulator` or plain `useState`.

import { useState } from "react";
import clsx from "clsx";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Divider } from "../../layout/Divider.js";
import { CollapsibleSection } from "../../prose/CollapsibleSection.js";
import { inkClassName } from "../../theme/tokens.js";
import type { Audience } from "./audience.js";
import { OverviewPage } from "./OverviewPage.js";
import { CreditFlowPage } from "./CreditFlowPage.js";
import { DeduplicationPage } from "./DeduplicationPage.js";
import { SchedulingPage } from "./SchedulingPage.js";
import { ExecutionPage } from "./ExecutionPage.js";

type GraphPageId = "overview" | "credit" | "dedup" | "scheduling" | "execution";

const PAGES: ReadonlyArray<PageTabDefinition<GraphPageId>> = [
  { id: "overview", label: "Overview" },
  { id: "credit", label: "Credit Flow" },
  { id: "dedup", label: "Deduplication" },
  { id: "scheduling", label: "Scheduling" },
  { id: "execution", label: "Execution" },
];

export const GLOSSARY = [
  { term: "ParsedGraph", def: "The canonical in-memory workload: nodes, edges, channels, traces, and (for trie builds) a SegmentPool. Every adapter produces one." },
  { term: "SegmentPool / segment_id", def: "The deduplicated content store. segment_id is a blake2b-16 hash over (parent_id, role, tokens), chained root-to-tip so shared prefixes share ids." },
  { term: "node_ordinal", def: "The build-time index of a node within its base trace (dense, sorted by arrival offset then id). The stable key a worker uses to look up a request envelope." },
  { term: "phase_variant", def: "Either profiling or warmup. Warmup reuses the profiling envelope but caps output to one token, so no duplicate stores are needed." },
  { term: "VersionedChannelStore", def: "The per-trace dataflow store. Nodes write versioned log entries; ChannelRequirement.count gates fan-in (count=all waits for every declared producer)." },
  { term: "CreditDispatchAdapter", def: "The per-instance bridge between the async executor and the v1 credit system. Parks a Future per dispatched node and resolves it when the worker returns." },
  { term: "t* (t-star)", def: "A sampled split point in a trace's timeline. Arrivals before t* are warmup; at/after t* are profiled and rebased. chop_trie_at_tstar trims pre-t* nodes for resume." },
  { term: "Unified store", def: "One directory that duck-types both the addressing (delta) and content (segment) readers. Its A2 interned layout enables zero-copy pre-serialized dispatch." },
];

function AudienceToggle({ audience, onChange }: { audience: Audience; onChange: (a: Audience) => void }): React.JSX.Element {
  return (
    <div className="flex items-center gap-1.5">
      <span className={clsx("text-xs", inkClassName("tertiary"))}>View</span>
      {(["manager", "developer"] as Audience[]).map((a) => (
        <button
          key={a}
          type="button"
          aria-pressed={audience === a}
          onClick={() => onChange(a)}
          className={clsx(
            "rounded-none border px-3 py-1 text-xs font-medium capitalize",
            audience === a
              ? "border-accent-primary bg-accent-primary text-white"
              : clsx("border-stroke-secondary", inkClassName("secondary")),
          )}
        >
          {a}
        </button>
      ))}
    </div>
  );
}

/**
 * Top-level composing component for the Graph Subsystem Overview deck. Holds the current page id
 * and the shared audience discriminant, renders the `PageTabs`, one page component per tab, and a
 * closing glossary shared across every page.
 */
export function GraphSubsystemOverviewDeck(): React.JSX.Element {
  const [page, setPage] = useState<GraphPageId>("overview");
  const [audience, setAudience] = useState<Audience>("manager");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Graph Subsystem" actions={<AudienceToggle audience={audience} onChange={setAudience} />} />
      <div className="border-b border-stroke-secondary bg-surface-page px-8 py-3">
        <PageTabs pages={PAGES} current={page} onChange={setPage} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-5xl bg-surface-page px-10 py-8">
          {page === "overview" && <OverviewPage audience={audience} />}
          {page === "credit" && <CreditFlowPage />}
          {page === "dedup" && <DeduplicationPage audience={audience} />}
          {page === "scheduling" && <SchedulingPage />}
          {page === "execution" && <ExecutionPage />}

          <Divider className="my-8" />
          <Stack gap={8}>
            <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Glossary</h2>
            <p className={clsx("text-sm", inkClassName("secondary"))}>The terms that come up most when reading this subsystem.</p>
            <Stack gap={2}>
              {GLOSSARY.map((g) => (
                <CollapsibleSection key={g.term} title={g.term}>
                  <p className={clsx("pt-1 text-sm", inkClassName("secondary"))}>{g.def}</p>
                </CollapsibleSection>
              ))}
            </Stack>
          </Stack>
          <Divider className="my-6" />
          <p className={clsx("text-xs", inkClassName("quaternary"))}>
            Anchored to code verified on the weka-ir-v1 branch · build plane, schedule plane, runtime plane, worker
            plane.
          </p>
        </div>
      </div>
    </div>
  );
}
