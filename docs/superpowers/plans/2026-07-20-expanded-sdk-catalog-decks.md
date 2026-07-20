# Expanded SDK Catalog Decks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace both one-page SDK primitive catalogs with polished, chaptered decks containing one contextual hero page per registered primitive.

**Architecture:** A text-level coverage test treats each focused slide's canonical SDK id as its internal slide name and verifies that the slide invokes that primitive. After the test is red, the generic and diagram decks can be authored in parallel because they edit separate files; final verification runs against the integrated working tree.

**Tech Stack:** Flow DSL, TypeScript, Vitest, the production Flow SDK registry, Flow IR verifier, Playwright/Vite deck walker.

## Global Constraints

- Keep `/sdk-generic-catalog` and `/sdk-diagram-catalog` and their existing deck identities.
- The generic deck has 54 slides: one opening, seven chapter dividers, 45 focused primitive slides, and one finale.
- The diagram deck has 49 slides: one opening, six chapter dividers, 41 focused primitive slides, and one finale.
- Every focused slide is named with the primitive's canonical id, such as `slide "sdk.shape"`.
- Every focused slide invokes its named primitive through the production SDK registry.
- Use the approved hero-stage layout: one large contextual hero, two to four secondary variants, and one concise authoring takeaway.
- Use a dark graphite stage, restrained NVIDIA green accents, a consistent alignment grid, generous negative space, attached labels, and believable content.
- Do not use broken-image placeholders, floating labels, unlabeled empty boxes, accidental overlap, clipped content, dense kitchen-sink scenes, or decorative motion.
- Renderer and SDK implementation files change only if a concrete failing deck example proves a real limitation.
- Preserve the NVIDIA SPDX header in every source file.
- Do not create commits; the user requested implementation, not git commits.

---

### Task 1: Add the red deck coverage contract

**Files:**
- Create: `apps/explainers/src/flow/sdk/catalog-decks.test.ts`

**Interfaces:**
- Consumes: `GENERIC_CATALOG_COMPONENTS` and `DIAGRAM_SDK_COMPONENTS`.
- Produces: a focused-slide coverage contract used by both deck implementations.

- [ ] **Step 1: Write the failing coverage test**

Create `catalog-decks.test.ts` with the repository SPDX header and this behavior:

```ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { DIAGRAM_SDK_COMPONENTS } from "./diagram/catalog.js";
import { GENERIC_CATALOG_COMPONENTS } from "./generic/catalog.js";

function readDeck(name: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../../../decks-flow/${name}`, import.meta.url)),
    "utf8",
  );
}

function slideBodies(source: string): Map<string, string> {
  const starts = [...source.matchAll(/^\s*slide "([^"]+)" \{/gm)];
  return new Map(
    starts.map((match, index) => [
      match[1],
      source.slice(
        match.index,
        starts[index + 1]?.index ?? source.length,
      ),
    ]),
  );
}

function invocationFor(id: string): string {
  const name = id.slice("sdk.".length);
  return `sdk.${name[0].toUpperCase()}${name.slice(1)}(`;
}

describe("SDK catalog decks", () => {
  it.each([
    {
      name: "generic",
      source: readDeck("sdk-generic-catalog.flow"),
      components: GENERIC_CATALOG_COMPONENTS,
      slideCount: 54,
    },
    {
      name: "diagram",
      source: readDeck("sdk-diagram-catalog.flow"),
      components: DIAGRAM_SDK_COMPONENTS,
      slideCount: 49,
    },
  ])("$name deck gives every registered primitive one focused slide", ({
    source,
    components,
    slideCount,
  }) => {
    const slides = slideBodies(source);

    expect(slides.size).toBe(slideCount);
    for (const component of components) {
      const id = component.descriptor.id;
      const body = slides.get(id);
      expect(body, `missing focused slide ${id}`).toBeDefined();
      expect(body).toContain(invocationFor(id));
    }
  });
});
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/explainers
npx vitest run src/flow/sdk/catalog-decks.test.ts
```

Expected: both cases fail because each existing deck has one slide and no canonical focused-slide names.

- [ ] **Step 3: Leave the test red for the parallel deck tasks**

Report the exact failing assertions. Do not weaken expected counts or infer coverage from the old kitchen-sink scene.

### Task 2: Rewrite the generic catalog deck

**Files:**
- Modify: `apps/explainers/decks-flow/sdk-generic-catalog.flow`
- Test: `apps/explainers/src/flow/sdk/catalog-decks.test.ts`

**Interfaces:**
- Consumes: the canonical-slide contract from Task 1.
- Produces: the complete 54-slide generic catalog.

- [ ] **Step 1: Establish the fixed slide sequence**

Author one opening, these seven chapter dividers, 45 focused slides, and one finale:

1. Foundations and shapes: `sdk.shape`, `sdk.text`, `sdk.richText`, `sdk.icon`, `sdk.image`, `sdk.line`, `sdk.arrow`, `sdk.spacer`, `sdk.inset`.
2. Typography and rich content: `sdk.title`, `sdk.paragraph`, `sdk.caption`, `sdk.codeBlock`, `sdk.quote`, `sdk.list`, `sdk.keyValue`, `sdk.propertyList`.
3. Status and identity: `sdk.badge`, `sdk.statusDot`, `sdk.avatar`, `sdk.iconLabel`, `sdk.alert`, `sdk.statusCard`, `sdk.emptyState`.
4. Data display: `sdk.stat`, `sdk.metric`, `sdk.table`, `sdk.tableRow`, `sdk.tableCell`, `sdk.tagList`.
5. Navigation and actions: `sdk.breadcrumb`, `sdk.tabs`, `sdk.pagination`, `sdk.timeline`, `sdk.timelineItem`.
6. Indicators: `sdk.progress`, `sdk.meter`, `sdk.gauge`, `sdk.sparkline`, `sdk.rating`, `sdk.semaphore`.
7. Containers and composition: `sdk.section`, `sdk.toolbar`, `sdk.splitPane`, `sdk.mediaObject`.

The focused slide names must be the canonical ids exactly. Opening, divider, and finale names must not begin with `sdk.`.

- [ ] **Step 2: Author every focused page with the approved template**

For each focused page:

- Invoke the named primitive in a large hero region.
- Use realistic content appropriate to the component.
- Add two to four smaller variants when the primitive has meaningful states or forms.
- Attach labels to the specimens.
- Include one concise takeaway in the slide copy.
- Reveal the hero first and variants second.
- Keep scene content within the established 700-by-400 stage and use repeated x/y/width/height anchors for alignment.

Image examples must use an existing valid repository asset. Empty or spacing primitives must use visible framing and attached annotation without replacing the primitive itself.

- [ ] **Step 3: Author the opening, chapter dividers, and finale**

The opening explains the deck's reference-and-teaching purpose. Each divider previews its family with ample negative space. The finale composes primitives from at least four generic families into one believable interface and does not serve as a substitute for any focused page.

- [ ] **Step 4: Run the generic coverage test and verify GREEN**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/explainers
npx vitest run src/flow/sdk/catalog-decks.test.ts
```

Expected: the generic case passes; the diagram case remains red until Task 3 completes.

- [ ] **Step 5: Run Flow IR verification**

Run `npm run flow-verifier:ir`. Expected: no errors attributable to `sdk-generic-catalog.flow`.

### Task 3: Rewrite the diagram catalog deck

**Files:**
- Modify: `apps/explainers/decks-flow/sdk-diagram-catalog.flow`
- Test: `apps/explainers/src/flow/sdk/catalog-decks.test.ts`

**Interfaces:**
- Consumes: the canonical-slide contract from Task 1.
- Produces: the complete 49-slide diagram catalog.

- [ ] **Step 1: Establish the fixed slide sequence**

Author one opening, these six chapter dividers, 41 focused slides, and one finale:

1. Actors and compute: `sdk.user`, `sdk.client`, `sdk.service`, `sdk.server`, `sdk.process`, `sdk.worker`, `sdk.function`, `sdk.container`, `sdk.cloud`.
2. Storage: `sdk.database`, `sdk.dataStore`, `sdk.cache`, `sdk.file`, `sdk.objectStore`, `sdk.volume`.
3. Messaging and network: `sdk.queue`, `sdk.topic`, `sdk.stream`, `sdk.eventBus`, `sdk.gateway`, `sdk.endpoint`, `sdk.loadBalancer`, `sdk.firewall`.
4. Control flow: `sdk.start`, `sdk.end`, `sdk.processStep`, `sdk.decision`, `sdk.merge`, `sdk.delay`, `sdk.retry`, `sdk.loop`.
5. Boundaries and grouping: `sdk.boundary`, `sdk.zone`, `sdk.cluster`, `sdk.trustBoundary`.
6. Symbols and annotations: `sdk.document`, `sdk.terminal`, `sdk.clock`, `sdk.lock`, `sdk.key`, `sdk.warning`.

The focused slide names must be the canonical ids exactly. Opening, divider, and finale names must not begin with `sdk.`.

- [ ] **Step 2: Author every focused page with the approved template**

For each focused page:

- Invoke the named primitive as the hero in a believable system topology or flow.
- Demonstrate its semantic role and relevant connection ports through supporting primitives and connectors where useful.
- Add two to four secondary variants or states without turning the page into a dense topology.
- Attach labels to the specimens.
- Include one concise takeaway in the slide copy.
- Reveal the hero first and variants second.
- Keep scene content within the established 700-by-400 stage and use repeated anchors for alignment.

- [ ] **Step 3: Author the opening, chapter dividers, and finale**

The opening explains the node grammar and semantic-port purpose. Each divider previews its family with ample negative space. The finale composes compute, storage, messaging/network, control, and boundary primitives into one coherent architecture without replacing any focused page.

- [ ] **Step 4: Run the diagram coverage test and verify GREEN**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/explainers
npx vitest run src/flow/sdk/catalog-decks.test.ts
```

Expected: the diagram case passes; when Task 2 is present, both cases pass.

- [ ] **Step 5: Run Flow IR verification**

Run `npm run flow-verifier:ir`. Expected: no errors attributable to `sdk-diagram-catalog.flow`.

### Task 4: Integrated visual and build verification

**Files:**
- Verify: `apps/explainers/decks-flow/sdk-generic-catalog.flow`
- Verify: `apps/explainers/decks-flow/sdk-diagram-catalog.flow`
- Verify: `apps/explainers/src/flow/sdk/catalog-decks.test.ts`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: final evidence that both decks compile, render, and satisfy coverage.

- [ ] **Step 1: Run focused tests**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/explainers
npx vitest run \
  src/flow/sdk/catalog-decks.test.ts \
  src/flow/sdk/generic/catalog.test.ts \
  src/flow/sdk/diagram/catalog.test.ts
```

Expected: all tests pass.

- [ ] **Step 2: Run Flow verification**

Run `npm run flow-verifier:ir`. Expected: no catalog-deck errors.

- [ ] **Step 3: Run isolated full Playwright walkthroughs**

Use the repository's existing deck screenshot/walkthrough tooling against `/sdk-generic-catalog` and `/sdk-diagram-catalog`. Visit every slide in both routes. Expected: no page errors, console errors, or warnings.

- [ ] **Step 4: Inspect screenshots**

Inspect the opening, every chapter divider, every focused primitive page, and both finales. Reject pages with overlap, clipped labels, broken images, empty unlabeled boxes, floating labels, weak hierarchy, or kitchen-sink density. Correct defects in the owning deck and repeat Steps 1–4.

- [ ] **Step 5: Run the production build**

Run `npm run build`. Expected: TypeScript and Vite build succeed. If unrelated concurrent work blocks the broad build, record the exact pre-existing error and verify the changed files with the focused commands above.
