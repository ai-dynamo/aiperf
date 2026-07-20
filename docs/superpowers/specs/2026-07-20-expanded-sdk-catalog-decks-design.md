# Expanded SDK Catalog Decks Design

## Goal

Replace the dense, single-page generic and diagram SDK catalogs with polished, near-exhaustive explainer decks. Every registered primitive receives a focused page that teaches what it is, shows useful variants, and demonstrates it through the production Flow SDK.

## Scope

This work expands:

- `apps/explainers/decks-flow/sdk-generic-catalog.flow`
- `apps/explainers/decks-flow/sdk-diagram-catalog.flow`

The existing routes and deck identities remain unchanged. SDK factories and renderer code change only when a concrete deck requirement exposes a verified limitation; visual stand-ins must not replace the primitive being demonstrated.

## Deck Architecture

Each catalog becomes a chaptered deck with:

1. An opening page that establishes the catalog and its visual language.
2. Chapter-divider pages that introduce each primitive family.
3. One focused page for every registered primitive.
4. A final composition page showing several families working together.

With the current 45 generic and 41 diagram registrations, the generic deck has 54 pages and the diagram deck has 49 pages. These totals include the opening, chapter dividers, focused primitive pages, and finale. The coverage check derives expected counts from the registry if the catalog changes before implementation.

### Generic Chapters

1. Foundations and shapes
2. Typography and rich content
3. Status and identity
4. Data display
5. Navigation and actions
6. Indicators
7. Containers and composition

### Diagram Chapters

1. Actors and compute
2. Storage
3. Messaging and network
4. Control flow
5. Boundaries and grouping
6. Symbols and annotations

Primitive order follows a teaching sequence within each family: simple foundations first, then specialized forms, then composition-oriented components.

## Primitive Page Template

Every primitive page uses a consistent hero-stage structure:

- A family eyebrow and primitive name establish context.
- A large hero composition demonstrates the primitive in a believable UI or system scene.
- Two to four smaller, evenly sized variant specimens show meaningful states, forms, or configurations.
- Labels attach directly to specimens.
- A concise takeaway explains the primitive's authoring or behavioral contract.

The hero composition is the focal point. Variant specimens remain visually secondary and must not compete with the hero. Pages share alignment and spacing conventions, but their examples are authored intentionally rather than generated as mechanically identical grids.

## Visual System

The decks use:

- A dark graphite stage.
- Restrained NVIDIA green accents for focus and active state.
- A consistent alignment grid.
- Generous negative space.
- Clear hierarchy between page title, hero, variants, and takeaway.
- Believable UI and system content instead of placeholder labels.
- Motion that reveals the hero before the variants.

The decks must not contain:

- Dense kitchen-sink scenes.
- Floating or orphaned labels.
- Broken-image placeholders.
- Unlabeled empty boxes.
- Accidental overlap or clipped content.
- Decorative motion unrelated to explanation.

## SDK and Data Flow

Each demonstrated primitive expands through the production SDK registry into Scene IR:

1. The `.flow` page invokes the registered primitive.
2. The Flow compiler resolves and expands the SDK component.
3. The existing scene-lowering path produces Scene IR.
4. `SceneRenderer` renders the resulting scene.

The primitive under demonstration cannot be replaced with hand-authored SVG or a visually similar core node. Supporting context may use other SDK primitives and existing core scene capabilities.

## Coverage Contract

Implementation maintains an explicit primitive-to-slide coverage map for each deck. The map must prove that:

- Every registered generic primitive has a focused generic-catalog page.
- Every registered diagram primitive has a focused diagram-catalog page.
- Each focused page invokes its named primitive through the SDK.
- No primitive is omitted because it appears only in a chapter divider or finale.

Primitives may appear on additional composition pages without weakening the one-focused-page requirement.

## Error Handling

Unsupported properties, malformed component invocations, missing registrations, invalid geometry, and unresolved scene nodes must fail parsing, compilation, or verification. The decks must not silently substitute placeholders for failed components.

## Verification

Verification includes:

1. Flow parsing and IR verification for both decks.
2. Existing catalog registration and expansion tests.
3. Coverage checks against the registered generic and diagram component lists.
4. Full Playwright walkthroughs of both catalog routes.
5. Screenshot inspection of the opening, every chapter boundary, representative primitive pages, and each finale.
6. Production build of the explainer application.

All pages must remain within safe scene bounds, preserve readable labels, avoid overlap, and render without browser errors or warnings at supported viewport sizes.

## Non-Goals

- Adding AIPerf-specific SDK primitives.
- Adding full charting primitives.
- Merging the generic and diagram catalogs.
- Redesigning the global explainer shell.
- Replacing the production SDK expansion path with catalog-only rendering.
