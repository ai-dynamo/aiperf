/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { inkClassName } from "../../theme/tokens.js";
import { OverviewPage } from "./OverviewPage.js";
import { SourcePage } from "./SourcePage.js";
import { SessionPage } from "./SessionPage.js";
import { PipelinePage } from "./PipelinePage.js";
import { ShadowReplayPage } from "./ShadowReplayPage.js";

type PageId = "overview" | "source" | "session" | "pipeline" | "shadow-replay";

interface PageMeta extends PageTabDefinition<PageId> {
  hint: string;
}

const PAGES: ReadonlyArray<PageMeta> = [
  { id: "overview", label: "1 · Overview", hint: "end-to-end pipeline" },
  { id: "source", label: "2 · Source", hint: "acquire & decode trace files" },
  { id: "session", label: "3 · Session", hint: "join fragments into conversations" },
  { id: "pipeline", label: "4 · Pipeline & Results", hint: "deliver, compact, export" },
  { id: "shadow-replay", label: "5 · Shadow Replay", hint: "re-execute recorded requests" },
];

/**
 * Five-page deck explaining the native Rust streaming dataset and shadow-replay workload built
 * across tasks P1–P4, 5D-5F, 6B-6D, C1-C3, A-REG, and V1 on the ajc/native-rust-runtime-plugins branch.
 */
export function StreamingDynamoShadowReplayDeck(): React.JSX.Element {
  const [page, setPage] = useState<PageId>("overview");
  const currentHint = PAGES.find((p) => p.id === page)?.hint;

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Streaming · Dynamo Shadow Replay" />
      <div className="border-b border-stroke-secondary bg-surface-page py-4">
        <div className="mx-auto max-w-6xl 2xl:max-w-[1728px] px-10">
          <h1 className="text-xl font-bold">Streaming Dynamo shadow replay</h1>
          <p className={`mt-1 max-w-4xl text-sm ${inkClassName("secondary")}`}>
            Native Rust streaming dataset and shadow-replay workload: acquire Dynamo trace files from local disk or S3,
            decode them, join request fragments into sessions, issue each recorded request against a live endpoint, and
            export per-session results — all resumable from checkpoint.
          </p>
          <div className="mt-3">
            <PageTabs pages={PAGES} current={page} onChange={setPage} />
          </div>
          {currentHint !== undefined && (
            <div className={`mt-2 text-xs uppercase tracking-wide ${inkClassName("tertiary")}`}>{currentHint}</div>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        <div className="mx-auto max-w-6xl 2xl:max-w-[1728px] px-10 py-6">
          {page === "overview" && <OverviewPage />}
          {page === "source" && <SourcePage />}
          {page === "session" && <SessionPage />}
          {page === "pipeline" && <PipelinePage />}
          {page === "shadow-replay" && <ShadowReplayPage />}
        </div>
      </div>
    </div>
  );
}
