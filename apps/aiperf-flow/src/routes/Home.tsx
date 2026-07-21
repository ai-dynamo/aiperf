/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Link } from "react-router-dom";
import { TopBar } from "../shell/TopBar.js";
import { Grid } from "../layout/Grid.js";
import { Stack } from "../layout/Stack.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../theme/tokens.js";

export type DeckListing = {
  path: string;
  title: string;
  description: string;
};

/** Every browsable deck. Add an entry here whenever a new deck route is wired into `App.tsx`. */
export const DECKS: readonly DeckListing[] = [
  {
    path: "/segment-pools",
    title: "Segment Pools",
    description:
      "Content-addressed interning, freeze-to-store, and BodyPlan materialization — a six-page walkthrough with two live simulators, ported from a real Cursor Canvas.",
  },
];

function DeckCard({ deck }: { deck: DeckListing }): React.JSX.Element {
  return (
    <Link
      to={deck.path}
      className={`block rounded-none border p-6 transition-colors hover:border-accent-primary ${surfaceClassName("elevated")} ${strokeClassName("primary")}`}
    >
      <Stack gap={8}>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>{deck.title}</h2>
        <p className={`text-sm ${inkClassName("secondary")}`}>{deck.description}</p>
      </Stack>
    </Link>
  );
}

/** Landing page: browse every deck currently wired into the app. */
export function Home(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Home" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto max-w-6xl px-10 py-12">
          <Stack gap={4} className="mb-10">
            <h1 className={`text-3xl font-extrabold ${inkClassName("primary")}`}>
              Explainer decks
            </h1>
            <p className={`max-w-2xl text-sm ${inkClassName("secondary")}`}>
              Interactive diagrams and walkthroughs of AIPerf subsystems, built as plain React
              components on React Flow, Motion, and Tailwind — no custom DSL.
            </p>
          </Stack>
          <Grid columns={2} gap={20}>
            {DECKS.map((deck) => (
              <DeckCard key={deck.path} deck={deck} />
            ))}
          </Grid>
        </div>
      </div>
    </div>
  );
}
