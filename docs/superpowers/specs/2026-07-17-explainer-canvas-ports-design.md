# Explainer Canvas Ports Design

## Goal

Port all seven AIPerf product canvases from `docs/canvases/` into the unified
`apps/explainers/` website as narrated decks. Existing decks remain available;
overlapping ports are intentionally retained as deeper, source-oriented
walkthroughs.

## New decks

1. **Rust architecture atlas** (`#/rust-architecture-atlas`)
   - Eleven slides following the canvas tabs: system, processes, runtime,
     protocol, scheduled execution, graph execution, endpoints, metrics,
     cellular, builds, and extension seams.
   - Positioned as the source-oriented companion to the existing beginner Rust
     architecture deck.
2. **Velo deep dive** (`#/velo-deep-dive`)
   - Ten mechanism slides: connection, registration, START, MessagePack,
     heartbeats, partitions, merge, phasers, dataset distribution, and
     aggregator hierarchy.
3. **Cellular internals** (`#/cellular-internals`)
   - Twenty slides preserving the canvas chapters: launch, distribute, execute,
     reduce, and scale.
4. **Cellular algorithm workbook** (`#/cellular-algorithms`)
   - A narrated chapter deck rather than a 100-slide transcription.
   - Sixteen slides cover eligibility, ownership, control, distribution,
     execution, capture, merge, artifacts, composition, and decision points.
   - The narration explicitly directs maintainers to the source canvas for the
     exhaustive per-algorithm catalog.
5. **Dynosim offline deep dive** (`#/dynosim-offline`)
   - Seven slides preserving overview, launch, architecture seams, simulation
     loop, dispatch/token flow, parity, and topology builder internals.
   - Positioned as the offline-specific companion to the existing broad
     Dynosim deck.
6. **Segment pools and body plans** (`#/segment-pools`)
   - Six slides preserving build/freeze/dispatch, interning, payload domains,
     BodyPlan splicing, prefix addressing, and dispatch precedence.
7. **Mock server architecture** (`#/mock-server`)
   - Ten chapter slides preserving orientation, ingress, LLM protocols,
     specialized endpoints, gRPC/Riva, timing, scheduler/cache, faults,
     observability/deployment, and proof/boundaries.
   - This is an orientation deck; the 64-page canvas remains the exhaustive
     feature reference.

## Architecture

Every new route exports a `DeckDefinition` and uses the existing
`ExplainerShell`. Each slide co-locates title, explanatory copy, narration,
points, caption, and its deck-local SVG scene. The central deck registry remains
the single source for hub cards and routing.

Common visual primitives may be extracted when at least two new decks need the
same box, arrow, lane, or motion signal. Deck-specific diagrams stay local so
the core does not grow a canvas-specific scene language.

## Content policy

- Ground claims in current executable Rust and manifests, using canvas text as
  an editorial source rather than runtime truth.
- Preserve the canvas learning sequence while rewriting dense reference prose
  for spoken narration.
- Label overlapping decks as “deep dive” or “atlas” on the hub.
- Keep narration concise enough for approximately 20–45 seconds per slide.
- Keep each scene legible at the existing 700×400 SVG viewport and mobile
  responsive through the shared shell.

## Behavior

All new decks inherit:

- play with or without audio;
- persistent, selectable voice pills;
- synchronized subtitles;
- narration-timed advancement;
- Back/pill/keyboard navigation restarting the selected slide;
- reduced-motion support;
- deck-scoped local storage.

## Testing and deployment

- Extend registry tests to assert ten total decks, unique routes/IDs, and
  non-empty narration.
- Add content checks for the expected slide count of every new deck.
- Run the complete Vitest suite and strict production build.
- Smoke-test hub cards and all new hash routes from the production preview.
- Publish the single `dist/` artifact through the existing Pages script.

## Out of scope

- Porting the meta `canvas-repo-layout` canvas.
- Removing or replacing the source canvases.
- Reproducing every interactive simulator or every algorithm page verbatim.
- Changing the existing Rust, SLURM/Velo, or broad Dynosim deck routes.
