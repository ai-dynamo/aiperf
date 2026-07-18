# AIPerf Flow Theme System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a typed AIPerf Flow theme system (authoring → IR → runtime resolution → evaluation → Canvas/SVG → player chrome) with the first bundled theme **Systems Chalk**, while keeping unthemed documents visually compatible with today's defaults.

**Architecture:** Schema owns the closed role vocabulary and strict theme IR. Language parses `theme` / `use theme` / `theme(role)`. Compiler validates inheritance and emits unresolved role references. Runtime registers bundled and document themes transactionally, resolves a frozen `ResolvedTheme`, and injects it into `CapabilityEvaluationContext`. Evaluators and foundation nodes resolve semantic roles into literal display-list paint values; Canvas and SVG never branch on theme identity. Player chrome maps a restrained subset of the active theme onto `.aiperf-flow` CSS custom properties.

**Tech Stack:** TypeScript strict mode, Zod 4, Chevrotain 12, Vitest 4, React 19, Playwright, `@fontsource/nunito-sans`, `@fontsource/ibm-plex-mono`, `svg-path-properties`, `culori`, npm workspaces under `apps/aiperf-flow/`.

**Design record:** [`../specs/2026-07-18-aiperf-flow-theme-system-design.md`](../specs/2026-07-18-aiperf-flow-theme-system-design.md)

## Global Constraints

- Implement only the approved design slice: typed themes, Systems Chalk, `theme(role)`, runtime override, stroke reveal, restrained chrome mapping. Do not add arbitrary CSS in `.flow`, shaders, materials, light/print/high-contrast named themes, scene-local themes, multiple inheritance, or nondeterministic jitter.
- Package ownership order is mandatory to avoid cross-package conflicts: `@aiperf/flow-schema` → `@aiperf/flow-language` → `@aiperf/flow-compiler` → `@aiperf/flow-runtime` display-list/theme modules → evaluation/contributions → backends → `FlowApp`/chrome → preview/E2E/docs.
- Strict TDD on every task: write the failing test, run it and confirm the expected failure, implement the minimum code, rerun and confirm pass.
- Activate the project environment before every command:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
```

- Verify package work with workspace scripts from `apps/aiperf-flow/`:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-schema
npm test -w @aiperf/flow-language
npm test -w @aiperf/flow-compiler
npm test -w @aiperf/flow-runtime
npm run flow:check
```

- Do not create git commits unless the user explicitly requests them.
- Do not implement production code outside the task currently being executed.
- Preserve existing `token(name)` compile-time lowering to literals. `theme(role)` must remain unresolved in IR.
- Display lists stay backend-neutral literals. Canvas and SVG must not import Systems Chalk or read CSS variables for scene paint.
- Reduced motion, high contrast, and quality tiers remain orthogonal axes that may constrain theme defaults after resolution.

---

## Current baseline (2026-07-18)

| Area | Current state |
|---|---|
| Flow IR | `irVersion: 1` in `packages/schema/src/ir.ts`. `FlowIr` has `tokens` and `scenes` only; no `themes` / `defaultTheme`. |
| Style values | `RenderNodeBaseIr.style` is `Record<string, string \| number \| boolean>` via `z.record(z.string(), scalarSchema)`. |
| Language | `DocumentAst` has tokens/symbols/scenes. `ValueAst = LiteralAst \| TokenReferenceAst`. No theme AST. `Duration` keyword exists only for timeline cues (`duration <number>`), not `420ms` literals. |
| Compiler | `lower()` emits `irVersion: 1` and resolves `token(...)` to scalars. No theme collection or validation. |
| Display list | `PathDrawCommand` has `fill` / `stroke` / `strokeWidth` only. No `lineCap`, `lineJoin`, or `strokeReveal`. |
| Evaluation | `CapabilityEvaluationContext = { atMs }`. Foundation `pathStyle()` copies string/number style fields only. |
| Contributions | Hard-coded fallbacks: queue `#111827`/`#64748b`/`#22c55e`; waterfall `#7dcfff`/`#38bdf8`/`#f8fafc`/`#fbbf24`; segment-strip `#334155`/`#f8fafc`/`#38bdf8`; span-map `#ef4444`/`#94a3b8`/`#38bdf8`; fonts often `"sans-serif"`. |
| Chrome | `theme.css` uses Semantic-Depthfield-like CSS vars (`--flow-plane-deep: #07111f`, etc.). `FlowAppProps` has no theme override. |
| Bundled fonts | None; chrome references Inter / IBM Plex without bundling. |
| Cinematic fixture | `examples/cinematic/request-lifecycle.flow` uses document tokens, not themes. |

---

## Compatibility constraints

1. **Unthemed documents keep today's look.** If a document omits `use theme` and the host omits `themeOverride`, evaluation uses the existing legacy literal fallbacks (current contribution defaults and current chrome CSS). Do not silently restyle to Systems Chalk.
2. **Explicit author overrides win.** Literal styles, authored props, and `token(name)` (already lowered to literals) beat theme roles. `theme(role)` resolves against the active theme.
3. **IR v1 consumers.** Provide `upgradeFlowIrV1ToV2` that adds `themes: []`, omits `defaultTheme`, and leaves scalar styles unchanged. Runtime/test helpers that build IR fixtures must emit `irVersion: 2` after this plan, or call the upgrader.
4. **Tests that assert hard-coded contribution colors** must either keep asserting legacy fallbacks when `theme` is absent, or pass an explicit Systems Chalk / test theme and assert resolved role values.
5. **Reserved bundled IDs.** Authors cannot declare `theme systems_chalk { ... }`. Bundled definitions live only in the runtime registry.
6. **Import linking.** Imported modules may export themes; a linked document still has at most one active `use theme` default.

---

## File map

| File | Responsibility |
|---|---|
| `apps/aiperf-flow/packages/schema/src/theme.ts` | Closed `ThemeRole` vocabulary, value kinds, Zod parsers, contrast pair list |
| `apps/aiperf-flow/packages/schema/src/ir.ts` | `FlowThemeIr`, style-value union, `FlowIr` v2 fields, `upgradeFlowIrV1ToV2` |
| `apps/aiperf-flow/packages/schema/src/index.ts` | Re-exports |
| `apps/aiperf-flow/packages/language/src/tokens.ts` | `Theme`, `Use`, `Extends`, `ColorKind`, `NumberKind`, `DurationLiteral` |
| `apps/aiperf-flow/packages/language/src/ast.ts` | Theme AST nodes + `ThemeRoleReferenceAst` in `ValueAst` |
| `apps/aiperf-flow/packages/language/src/parser.ts` | Parse theme decls, `use theme`, `theme(role)`, typed assignments |
| `apps/aiperf-flow/packages/language/src/formatter.ts` | Round-trip theme syntax |
| `apps/aiperf-flow/packages/compiler/src/themes.ts` | Collect/validate themes across linked modules |
| `apps/aiperf-flow/packages/compiler/src/link.ts` | Carry theme decls / default through `LinkedDocument` |
| `apps/aiperf-flow/packages/compiler/src/lower.ts` | Emit `FlowThemeIr[]`, `defaultTheme`, unresolved `theme-role` style values, `irVersion: 2` |
| `apps/aiperf-flow/packages/compiler/src/validate.ts` | Invoke theme validation in the compile pipeline |
| `apps/aiperf-flow/packages/runtime/src/theme/types.ts` | `ResolvedTheme`, typed getters |
| `apps/aiperf-flow/packages/runtime/src/theme/registry.ts` | Transactional `ThemeRegistry` |
| `apps/aiperf-flow/packages/runtime/src/theme/contrast.ts` | WCAG AA checks via `culori` |
| `apps/aiperf-flow/packages/runtime/src/theme/systems-chalk.ts` | Bundled Systems Chalk values + shape constants |
| `apps/aiperf-flow/packages/runtime/src/theme/legacy-defaults.ts` | Current unthemed fallback literals |
| `apps/aiperf-flow/packages/runtime/src/theme/chrome-css.ts` | Map restrained roles → CSS custom properties |
| `apps/aiperf-flow/packages/runtime/src/theme/visual-roles.ts` | Shared component role mappings |
| `apps/aiperf-flow/packages/runtime/src/theme/path-metrics.ts` | Path length via `svg-path-properties` |
| `apps/aiperf-flow/packages/runtime/src/display-list.ts` | `lineCap`, `lineJoin`, `strokeReveal` on paths |
| `apps/aiperf-flow/packages/runtime/src/evaluate/registry.ts` | `theme?: ResolvedTheme` on evaluation context |
| `apps/aiperf-flow/packages/runtime/src/evaluate/scene-evaluator.ts` | Resolve `theme-role` styles; pass theme into evaluators |
| `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/*.ts` | Consume visual-role helpers with legacy fallbacks |
| `apps/aiperf-flow/packages/runtime/src/backends/canvas/canvas-renderer.ts` | Cap/join/dash reveal |
| `apps/aiperf-flow/packages/runtime/src/backends/svg/svg-fallback.tsx` | Cap/join/`pathLength`/`strokeDashoffset` reveal |
| `apps/aiperf-flow/packages/runtime/src/app.tsx` | `themeOverride`, active theme id, chrome CSS vars |
| `apps/aiperf-flow/packages/runtime/src/theme.css` | Consume mapped vars; keep legacy defaults as `:root` fallback |
| `apps/aiperf-flow/preview/App.tsx` | Theme selector for registry IDs |
| `apps/aiperf-flow/e2e/request-lifecycle-cinematic.spec.ts` | Systems Chalk Canvas/SVG assertions |
| `docs/superpowers/plans/2026-07-17-aiperf-flow-roadmap.md` | Index this plan in the artifact map |

---

## Task order and ownership

```text
Task 1  schema          (alone)
Task 2  language parse  (after 1)
Task 3  language format (after 2)
Task 4  compiler validate themes (after 2)
Task 5  compiler lower + IR emit (after 1+4)
Task 6  runtime ThemeRegistry + Systems Chalk (after 1; parallel with 7)
Task 7  display-list path contract + path metrics (after baseline; parallel with 6)
Task 8  evaluation integration (after 5+6+7)
Task 9  contribution migration (after 8)
Task 10 Canvas/SVG stroke reveal (after 7; can start once 7 lands)
Task 11 FlowApp chrome mapping (after 6+8)
Task 12 preview + E2E + roadmap index (after 9+10+11)
```

---

### Task 1: Schema theme vocabulary and Flow IR v2

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/theme.ts`
- Create: `apps/aiperf-flow/packages/schema/test/theme.test.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/ir.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Modify: `apps/aiperf-flow/packages/schema/test/ir.test.ts`

**Interfaces:**
- Consumes: existing `SourceRange`, Zod strict helpers in `ir.ts`.
- Produces:

```ts
export const THEME_ROLES = [
  "surface.canvas", "surface.panel", "surface.raised", "surface.control",
  "ink.primary", "ink.muted", "ink.inverse",
  "line.structural", "line.guide",
  "accent.control", "accent.execution", "accent.compute", "accent.attention",
  "accent.success", "accent.danger", "accent.focus",
  "font.display", "font.body", "font.data",
  "weight.regular", "weight.label", "weight.emphasis",
  "size.caption", "size.body", "size.label", "size.title",
  "stroke.hairline", "stroke.standard", "stroke.emphasis",
  "stroke.cap", "stroke.join",
  "motion.draw", "motion.enter", "motion.emphasis", "motion.stagger",
  "motion.easing",
] as const;
export type ThemeRole = (typeof THEME_ROLES)[number];

export type ThemeValueIr =
  | Readonly<{ kind: "color"; value: string }>
  | Readonly<{ kind: "font"; value: readonly string[] }>
  | Readonly<{ kind: "number"; value: number }>
  | Readonly<{ kind: "duration"; valueMs: number }>
  | Readonly<{ kind: "enum"; value: string }>;

export type FlowThemeIr = Readonly<{
  id: string;
  extends: string;
  values: Readonly<Partial<Record<ThemeRole, ThemeValueIr>>>;
  sourceMap: SourceRange;
}>;

export type ThemeRoleReferenceIr = Readonly<{
  kind: "theme-role";
  role: ThemeRole;
}>;

export type StyleScalarIr = string | number | boolean;
export type StyleValueIr = StyleScalarIr | ThemeRoleReferenceIr;

export type FlowIr = Readonly<{
  irVersion: 2;
  id: string;
  title: string;
  capabilities: readonly CapabilityRequirement[];
  tokens: Readonly<Record<string, string | number | boolean>>;
  themes: readonly FlowThemeIr[];
  defaultTheme?: string;
  scenes: readonly SceneIr[];
  sourceMap: SourceRange;
}>;

export function upgradeFlowIrV1ToV2(input: unknown): unknown;
export function parseThemeRole(input: string): ThemeRole;
export const REQUIRED_CONTRAST_PAIRS: readonly Readonly<{
  foreground: ThemeRole;
  background: ThemeRole;
  minRatio: number;
}>[];
```

- [ ] **Step 1: Write the failing schema tests**

Create `apps/aiperf-flow/packages/schema/test/theme.test.ts`:

```ts
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  REQUIRED_CONTRAST_PAIRS,
  THEME_ROLES,
  parseThemeRole,
  themeValueIrSchema,
  themeRoleReferenceIrSchema,
} from "../src/theme.js";
import { parseFlowIr, safeParseFlowIr, upgradeFlowIrV1ToV2 } from "../src/ir.js";

const sourceMap = {
  source: "theme.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

describe("theme schema", () => {
  test("accepts the closed role vocabulary and rejects unknown roles", () => {
    expect(THEME_ROLES).toContain("accent.execution");
    expect(parseThemeRole("motion.easing")).toBe("motion.easing");
    expect(() => parseThemeRole("glow.ambient")).toThrow(/unknown theme role/i);
  });

  test("parses typed theme values and rejects bad discriminants", () => {
    expect(
      themeValueIrSchema.parse({ kind: "color", value: "#71D8D0" }),
    ).toEqual({ kind: "color", value: "#71D8D0" });
    expect(
      themeValueIrSchema.parse({ kind: "duration", valueMs: 420 }),
    ).toEqual({ kind: "duration", valueMs: 420 });
    expect(
      themeValueIrSchema.safeParse({ kind: "color", valueMs: 1 }).success,
    ).toBe(false);
    expect(
      themeValueIrSchema.safeParse({ kind: "number", value: Number.NaN })
        .success,
    ).toBe(false);
  });

  test("parses theme-role style references and rejects unknown fields", () => {
    expect(
      themeRoleReferenceIrSchema.parse({
        kind: "theme-role",
        role: "surface.raised",
      }),
    ).toEqual({ kind: "theme-role", role: "surface.raised" });
    expect(
      themeRoleReferenceIrSchema.safeParse({
        kind: "theme-role",
        role: "surface.raised",
        extra: true,
      }).success,
    ).toBe(false);
  });

  test("lists required WCAG AA contrast pairs", () => {
    expect(REQUIRED_CONTRAST_PAIRS.length).toBeGreaterThanOrEqual(8);
    expect(REQUIRED_CONTRAST_PAIRS).toContainEqual({
      foreground: "ink.primary",
      background: "surface.canvas",
      minRatio: 4.5,
    });
  });
});

describe("Flow IR v2 themes", () => {
  test("parses themes, defaultTheme, and theme-role style values", () => {
    const flow = parseFlowIr({
      irVersion: 2,
      id: "themed",
      title: "Themed",
      capabilities: [],
      tokens: {},
      themes: [
        {
          id: "lab_chalk",
          extends: "systems_chalk",
          values: {
            "accent.control": { kind: "color", value: "#78dce8" },
            "stroke.standard": { kind: "number", value: 2 },
            "motion.draw": { kind: "duration", valueMs: 420 },
          },
          sourceMap,
        },
      ],
      defaultTheme: "lab_chalk",
      scenes: [
        {
          id: "main",
          title: "Main",
          summary: "s",
          roots: [
            {
              kind: "rect",
              id: "r",
              geometry: { x: 0, y: 0, width: 10, height: 10 },
              style: {
                fill: { kind: "theme-role", role: "surface.raised" },
              },
              accessibility: { label: "r" },
              fallback: "r",
              sourceMap,
            },
          ],
          camera: [],
          timeline: [],
          narration: "n",
          interactions: [],
          responsive: [],
          accessibility: { label: "main", readingOrder: ["r"] },
          fallback: "f",
          sourceMap,
        },
      ],
      sourceMap,
    });
    expect(flow.irVersion).toBe(2);
    expect(flow.defaultTheme).toBe("lab_chalk");
    expect(flow.themes[0]?.values["accent.control"]).toEqual({
      kind: "color",
      value: "#78dce8",
    });
    expect(flow.scenes[0]?.roots[0]?.style.fill).toEqual({
      kind: "theme-role",
      role: "surface.raised",
    });
  });

  test("rejects irVersion 1 without upgrade and upgrades v1 payloads", () => {
    const v1 = {
      irVersion: 1,
      id: "legacy",
      title: "Legacy",
      capabilities: [],
      tokens: { accent: "#7aa2f7" },
      scenes: [],
      sourceMap,
    };
    expect(safeParseFlowIr(v1).ok).toBe(false);
    const upgraded = parseFlowIr(upgradeFlowIrV1ToV2(v1));
    expect(upgraded.irVersion).toBe(2);
    expect(upgraded.themes).toEqual([]);
    expect(upgraded.defaultTheme).toBeUndefined();
    expect(upgraded.tokens.accent).toBe("#7aa2f7");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-schema -- theme.test.ts ir.test.ts
```

Expected: FAIL with module-not-found / `THEME_ROLES` undefined / `irVersion` still literal `1`.

- [ ] **Step 3: Implement schema modules**

Add `theme.ts` with the closed role list, hex color regex `^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$`, font array schema (`z.array(z.string().min(1)).min(1)`), number finite nonnegative where required, duration nonnegative int ms, enums for `stroke.cap` (`butt|round|square`), `stroke.join` (`bevel|round|miter`), `motion.easing` (`linear|ease_in|ease_out|ease_in_out`), and `REQUIRED_CONTRAST_PAIRS` covering:

- `ink.primary` on every surface role (`canvas`, `panel`, `raised`, `control`) at 4.5
- `ink.muted` on `surface.canvas` and `surface.panel` at 4.5
- interactive accents (`control`, `execution`, `compute`, `attention`, `success`, `danger`) on `surface.control` at 3.0 (UI component / non-text boundary floor used by the design's non-text pair set; document the ratio in the module comment)
- `accent.focus` on `surface.canvas`, `surface.panel`, and `surface.control` at 3.0

Update `ir.ts`:

- change `styleSchema` to `z.record(z.string(), z.union([scalarSchema, themeRoleReferenceIrSchema]))`
- change `RenderNodeBaseIr.style` to `Readonly<Record<string, StyleValueIr>>`
- add `themes` + optional `defaultTheme` to `FlowIr`
- set `irVersion: z.literal(2)`
- implement `upgradeFlowIrV1ToV2` that copies v1 objects, sets `irVersion: 2`, `themes: []`, and deletes unknown fields only as needed for parse success

Export new symbols from `index.ts`. Update existing `ir.test.ts` fixtures from `irVersion: 1` to `2` and add `themes: []`.

- [ ] **Step 4: Run schema tests to verify they pass**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-schema
npm run build -w @aiperf/flow-schema
```

Expected: PASS. Downstream packages may fail typecheck until later tasks; do not fix them yet except unavoidable `irVersion` literal updates required to keep schema tests green.

---

### Task 2: Language AST and parser for themes

**Files:**
- Modify: `apps/aiperf-flow/packages/language/src/tokens.ts`
- Modify: `apps/aiperf-flow/packages/language/src/ast.ts`
- Modify: `apps/aiperf-flow/packages/language/src/parser.ts`
- Modify: `apps/aiperf-flow/packages/language/src/index.ts`
- Create: `apps/aiperf-flow/packages/language/test/theme.test.ts`

**Interfaces:**
- Consumes: `ThemeRole` string validation deferred to compiler; parser accepts dotted role identifiers.
- Produces:

```ts
export type ThemeRoleReferenceAst = AstNode<"theme-role-reference"> &
  Readonly<{ role: string }>;

export type ValueAst = LiteralAst | TokenReferenceAst | ThemeRoleReferenceAst;

export type ThemeValueKindAst = "color" | "number" | "duration" | "font" | "enum";

export type ThemeAssignmentAst = AstNode<"theme-assignment"> &
  Readonly<{
    valueKind: ThemeValueKindAst;
    role: string;
    value: LiteralAst | ThemeFontLiteralAst;
  }>;

export type ThemeFontLiteralAst = AstNode<"theme-font-literal"> &
  Readonly<{ families: readonly string[] }>;

export type ThemeDeclarationAst = AstNode<"theme-declaration"> &
  Readonly<{
    id: string;
    extends: string;
    assignments: readonly ThemeAssignmentAst[];
  }>;

export type UseThemeAst = AstNode<"use-theme"> & Readonly<{ themeId: string }>;

export type DocumentAst = AstNode<"document"> &
  Readonly<{
    // existing fields...
    themes: readonly ThemeDeclarationAst[];
    useTheme?: UseThemeAst;
  }>;
```

Duration literal token: `DurationLiteral = createToken({ name: "DurationLiteral", pattern: /[0-9]+ms/ })` with higher priority than `NumberLiteral` / `Identifier`.

Keywords: `Theme` (`theme`), `Use` (`use`), `Extends` (`extends`), `ColorKind` (`color`), `NumberKind` (`number`), `FontKind` (`font`), `EnumKind` (`enum`). Reuse existing `Duration` keyword for assignment kind `duration <role> = <DurationLiteral>` so timeline `duration <NumberLiteral>` remains unambiguous by rule context.

- [ ] **Step 1: Write failing parser tests**

```ts
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { parseDocument } from "../src/parser.js";

const source = `
flow "Lab" as lab {
  language 1

  theme lab_chalk extends systems_chalk {
    color accent.control = "#78dce8"
    color accent.execution = "#ffd866"
    number stroke.standard = 2
    duration motion.draw = 420ms
    font font.body = ["Nunito Sans", "Segoe UI", "sans-serif"]
    enum stroke.cap = "round"
  }

  use theme lab_chalk

  require core.rect "^1.0.0"

  scene "Main" as main {
    summary "s"
    rect router {
      x 0
      y 0
      width 10
      height 10
      fill theme(surface.raised)
      stroke theme(accent.control)
      label "Router"
      role "group"
      description "router"
      fallback "router"
    }
    readingOrder router
    fallback "f"
  }
}
`;

describe("theme grammar", () => {
  test("parses theme declaration, use theme, and theme(role) style refs", () => {
    const result = parseDocument(source, "lab.flow");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const doc = result.value;
    expect(doc.themes).toHaveLength(1);
    expect(doc.themes[0]?.id).toBe("lab_chalk");
    expect(doc.themes[0]?.extends).toBe("systems_chalk");
    expect(doc.themes[0]?.assignments.map((a) => a.role)).toEqual([
      "accent.control",
      "accent.execution",
      "stroke.standard",
      "motion.draw",
      "font.body",
      "stroke.cap",
    ]);
    expect(doc.useTheme?.themeId).toBe("lab_chalk");
    const rect = doc.scenes[0]?.renderDeclarations[0];
    expect(rect?.kind).toBe("rect");
    if (rect?.kind !== "rect") return;
    expect(rect.fill).toEqual(
      expect.objectContaining({
        kind: "theme-role-reference",
        role: "surface.raised",
      }),
    );
    expect(rect.stroke).toEqual(
      expect.objectContaining({
        kind: "theme-role-reference",
        role: "accent.control",
      }),
    );
  });

  test("rejects theme(role) with empty role", () => {
    const result = parseDocument(
      `flow "X" as x { language 1
        scene "S" as s {
          summary "s"
          rect a { x 0 y 0 width 1 height 1 fill theme() label "a" role "g" description "a" fallback "a" }
          readingOrder a
          fallback "f"
        }
      }`,
      "bad.flow",
    );
    expect(result.ok).toBe(false);
  });
});
```

Also extend `RectAst` with optional `stroke?: ValueAst` (required for the design example). Keep connector `stroke` unchanged.

- [ ] **Step 2: Run parser tests to verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-language -- theme.test.ts
```

Expected: FAIL (unexpected token `theme` / missing `themes` on document).

- [ ] **Step 3: Implement tokens, AST, parser rules**

Parse dotted roles as `Identifier (Dot Identifier)*` joined with `.`.
Parse `theme(role)` inside the existing value rule alongside `token(name)`.
Parse document members in any order consistent with current document rule: collect `themes[]` and at most one `useTheme` (parser may accept duplicates; compiler rejects).
Parse font literals as string-array `[ StringLiteral (, StringLiteral)* ]`.
Parse `420ms` via `DurationLiteral` → `LiteralAst` with numeric `value` already converted to milliseconds number in AST (`value: 420`) and rely on assignment `valueKind: "duration"` for typing.

- [ ] **Step 4: Run language theme tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-language
npm run build -w @aiperf/flow-language
```

Expected: PASS, including existing parser/symbol/formatter suites still green after formatter stub tolerance (Task 3 owns round-trip).

---

### Task 3: Formatter round-trip for themes

**Files:**
- Modify: `apps/aiperf-flow/packages/language/src/formatter.ts`
- Modify: `apps/aiperf-flow/packages/language/test/formatter.test.ts`

**Interfaces:**
- Consumes: `ThemeDeclarationAst`, `UseThemeAst`, `ThemeRoleReferenceAst`
- Produces: deterministic `.flow` text preserving theme blocks before scenes:

```flow
theme lab_chalk extends systems_chalk {
  color accent.control = "#78dce8"
  duration motion.draw = 420ms
}
use theme lab_chalk
```

- [ ] **Step 1: Write failing formatter test**

Add a test that formats the Task 2 fixture AST (or re-parses the Task 2 source) and asserts the emitted text contains `theme lab_chalk extends systems_chalk`, `duration motion.draw = 420ms`, `use theme lab_chalk`, and `fill theme(surface.raised)`.

- [ ] **Step 2: Run to verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-language -- formatter.test.ts
```

Expected: FAIL on missing theme formatting branches.

- [ ] **Step 3: Implement formatter branches** for theme declarations, use-theme, theme-role references, duration literals (`${value}ms`), and font arrays.

- [ ] **Step 4: Re-run language tests** — Expected: PASS.

---

### Task 4: Compiler theme collection and validation

**Files:**
- Create: `apps/aiperf-flow/packages/compiler/src/themes.ts`
- Create: `apps/aiperf-flow/packages/compiler/test/themes.test.ts`
- Modify: `apps/aiperf-flow/packages/compiler/src/link.ts`
- Modify: `apps/aiperf-flow/packages/compiler/src/validate.ts`
- Modify: `apps/aiperf-flow/packages/compiler/src/index.ts`

**Interfaces:**
- Consumes: linked document themes + bundled ID set passed in as `readonly string[]` (default `["systems_chalk"]`).
- Produces:

```ts
export const BUNDLED_THEME_IDS = ["systems_chalk"] as const;

export type ThemeValidationInput = Readonly<{
  themes: readonly ThemeDeclarationAst[];
  useTheme?: UseThemeAst;
  bundledThemeIds?: readonly string[];
}>;

export function validateThemes(
  input: ThemeValidationInput,
): Result<Readonly<{
  themes: readonly ThemeDeclarationAst[];
  defaultTheme?: string;
}>>;
```

Diagnostics (stable codes + actionable messages including theme id, role/base, expected kind, received value, source range):

| Code | Condition |
|---|---|
| `THEME_DUPLICATE_ID` | duplicate authored id |
| `THEME_RESERVED_ID` | authored id in bundled set |
| `THEME_UNKNOWN_BASE` | `extends` not authored and not bundled |
| `THEME_INHERITANCE_CYCLE` | cycle among authored themes |
| `THEME_UNKNOWN_ROLE` | role not in `THEME_ROLES` |
| `THEME_ROLE_KIND_MISMATCH` | assignment kind incompatible with role |
| `THEME_INVALID_VALUE` | bad hex, non-finite number, bad enum, empty font stack |
| `THEME_UNKNOWN_DEFAULT` | `use theme` id missing |
| `THEME_DUPLICATE_DEFAULT` | more than one `use theme` after link |

Kind expectations:

- color roles → `color`
- font.* → `font`
- weight.* / size.* / stroke widths → `number`
- stroke.cap / stroke.join / motion.easing → `enum`
- motion.draw|enter|emphasis|stagger → `duration`

- [ ] **Step 1: Write failing compiler theme tests** covering each diagnostic code with exact message substrings, plus a valid inheritance case `lab_chalk extends systems_chalk`.

- [ ] **Step 2: Run**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-compiler -- themes.test.ts
```

Expected: FAIL module not found.

- [ ] **Step 3: Implement `validateThemes` and wire it from `validate()` after symbol/component checks.** Extend `LinkedDocument` to include `themes` and `useTheme` collected from the primary document and imported module exports (imported theme declarations flatten into the validation set with source maps preserved).

- [ ] **Step 4: Re-run compiler theme tests** — Expected: PASS.

---

### Task 5: Lower themes and preserve `theme(role)` in IR

**Files:**
- Modify: `apps/aiperf-flow/packages/compiler/src/lower.ts`
- Create: `apps/aiperf-flow/packages/compiler/test/lower-themes.test.ts`
- Modify: `apps/aiperf-flow/packages/compiler/test/compile.test.ts`
- Modify: `apps/aiperf-flow/packages/compiler/test/fixture.ts` (add `themes: []` / `irVersion: 2` as needed)

**Interfaces:**
- Consumes: validated `ThemeDeclarationAst`, `ValueAst`
- Produces: `FlowIr` with `irVersion: 2`, `themes`, optional `defaultTheme`, and style entries that are either scalars or `{ kind: "theme-role", role }`

```ts
function lowerStyleValue(
  value: ValueAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): StyleValueIr {
  switch (value.kind) {
    case "literal":
      return value.value;
    case "token-reference":
      return resolveTokenLiteral(value.token, tokens);
    case "theme-role-reference":
      return { kind: "theme-role", role: value.role as ThemeRole };
  }
}
```

- [ ] **Step 1: Write failing lower/compile tests**

```ts
test("preserves theme(role) and lowers custom themes without inlining systems_chalk", () => {
  const ir = compileSource({
    source: /* Task 2 source */,
    sourceName: "lab.flow",
    capabilities: /* minimal manifest including core.rect */,
    strict: true,
  });
  expect(ir.ok).toBe(true);
  if (!ir.ok) return;
  expect(ir.value.irVersion).toBe(2);
  expect(ir.value.defaultTheme).toBe("lab_chalk");
  expect(ir.value.themes.map((t) => t.id)).toEqual(["lab_chalk"]);
  expect(ir.value.themes[0]?.extends).toBe("systems_chalk");
  const rect = ir.value.scenes[0]?.roots.find((n) => n.id === "router");
  expect(rect?.style.fill).toEqual({
    kind: "theme-role",
    role: "surface.raised",
  });
});

test("still lowers token(name) to literals", () => {
  // existing accent token fixture remains scalar "#7aa2f7"
});
```

- [ ] **Step 2: Run** — Expected: FAIL on `irVersion` / missing themes.

- [ ] **Step 3: Implement lowering** for theme assignments → `ThemeValueIr`, document themes array, defaultTheme, rect optional stroke, and bump `irVersion` to `2`. Do **not** embed bundled Systems Chalk values in compiler output.

- [ ] **Step 4: Run**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-compiler
npm run build -w @aiperf/flow-compiler
```

Expected: PASS.

---

### Task 6: Runtime `ThemeRegistry`, contrast, and Systems Chalk

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/theme/types.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/contrast.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/registry.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/systems-chalk.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/legacy-defaults.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/index.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/theme/registry.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/theme/systems-chalk-contrast.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/package.json` (add `culori`, `@fontsource/nunito-sans`, `@fontsource/ibm-plex-mono`)
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`

**Interfaces:**
- Consumes: `FlowThemeIr`, `THEME_ROLES`, `REQUIRED_CONTRAST_PAIRS`
- Produces:

```ts
export type ResolvedTheme = Readonly<{
  id: string;
  values: Readonly<Record<ThemeRole, ThemeValueIr>>;
}>;

export class ThemeRegistry {
  registerBundled(themes: readonly FlowThemeIr[]): void;
  registerDocumentThemes(themes: readonly FlowThemeIr[]): void; // atomic
  freeze(): FrozenThemeRegistry;
}

export class FrozenThemeRegistry {
  ids(): readonly string[];
  resolve(id: string): ResolvedTheme;
  has(id: string): boolean;
}

export function selectActiveThemeId(input: Readonly<{
  overrideId?: string;
  documentDefault?: string;
  legacyId?: string; // default undefined meaning "no theme"
}>): string | undefined;

export class DuplicateThemeIdError extends Error {}
export class ReservedThemeIdError extends Error {}
export class UnknownThemeIdError extends Error {}
export class ThemeInheritanceCycleError extends Error {}
export class IncompleteThemeError extends Error {}
export class ThemeContrastError extends Error {}
export class ThemeRoleKindError extends Error {}

export function createBootstrapThemeRegistry(): ThemeRegistry;
/** Bundled root theme. `extends` is the internal sentinel `BUNDLED_ROOT_BASE`. */
export const BUNDLED_ROOT_BASE = "__bundled_root__" as const;
export const SYSTEMS_CHALK: FlowThemeIr;
export const SYSTEMS_CHALK_SHAPE: Readonly<{ cornerRadiusPx: 12 }>;
export const LEGACY_VISUAL_FALLBACKS: Readonly<{
  queueLane: "#111827";
  queueWaiting: "#64748b";
  queueServing: "#22c55e";
  waterfallPoint: "#7dcfff";
  waterfallInterval: "#38bdf8";
  waterfallText: "#f8fafc";
  waterfallPlayhead: "#fbbf24";
  segmentFill: "#334155";
  segmentText: "#f8fafc";
  segmentContinuation: "#38bdf8";
  spanUncovered: "#ef4444";
  spanCovered: "#94a3b8";
  spanEdge: "#38bdf8";
  glyphFill: "#f8fafc";
  morphFill: "#38bdf8";
}>;
```

Authors cannot use `BUNDLED_ROOT_BASE` as an `extends` target; only `registerBundled` accepts it. `SYSTEMS_CHALK.extends` must equal `BUNDLED_ROOT_BASE`.

Systems Chalk palette/type/stroke/motion values must match the design record exactly, including:

- colors `#232526` … `#9BDBF5` as listed in the design
- fonts: `font.display` / `font.body` → `["Nunito Sans", "Segoe UI", "sans-serif"]`; `font.data` → `["IBM Plex Mono", "Cascadia Code", "monospace"]`
- weights: regular 400, label 500, emphasis 600
- sizes: caption 11, body 13, label 12, title 18
- strokes: hairline 1, standard 2, emphasis 3; cap/join `round`
- motion: draw 420, enter 240, emphasis 180, stagger 60; easing `ease_out`

`resolve(id)` walks single-parent chain, detects cycles, requires every `THEME_ROLES` entry present after merge, revalidates kinds, runs contrast checks, deep-freezes, and caches by id.

- [ ] **Step 1: Write failing registry/contrast tests** for atomic document registration (duplicate fails before mutation), freeze immutability, inheritance override precedence, cycle errors, unknown override id, Systems Chalk contrast pass, and a custom theme that fails contrast.

- [ ] **Step 2: Run**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- theme/registry.test.ts theme/systems-chalk-contrast.test.ts
```

Expected: FAIL.

- [ ] **Step 3: Install deps and implement modules**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm install culori @fontsource/nunito-sans @fontsource/ibm-plex-mono -w @aiperf/flow-runtime
```

Implement contrast with `culori` WCAG contrast. Record font licenses in `apps/aiperf-flow/packages/runtime/src/theme/FONTS.md` (OFL for both families). Export theme API from runtime `index.ts`.

- [ ] **Step 4: Re-run focused runtime theme tests** — Expected: PASS.

---

### Task 7: Display-list path contract and path metrics

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/display-list.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/path-metrics.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/display-list.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/theme/path-metrics.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/package.json` (add `svg-path-properties`)

**Interfaces:**
- Consumes: SVG path strings already used by contributions
- Produces:

```ts
export type PathLineCap = "butt" | "round" | "square";
export type PathLineJoin = "bevel" | "round" | "miter";

export type PathDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "path";
    path: string;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
    lineCap?: PathLineCap;
    lineJoin?: PathLineJoin;
    strokeReveal?: number; // finite, clamped to [0, 1] at build time
  }>;

export function pathLength(path: string): number;
export function strokeDashForReveal(
  length: number,
  reveal: number,
): Readonly<{ dashArray: string; dashOffset: number }>;
```

`buildDisplayList` clamps `strokeReveal` into `[0, 1]` and rejects non-finite values.

- [ ] **Step 1: Write failing tests** for clamp/reject behavior and dash math (`reveal: 0` → full offset, `reveal: 1` → zero offset, `reveal: 0.25` → offset `0.75 * length`).

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Implement path-metrics with `svg-path-properties` and extend `PathDrawCommand`.** Cache lengths in a module-local `Map<string, number>` keyed by path string.

- [ ] **Step 4: Re-run display-list + path-metrics tests** — Expected: PASS.

---

### Task 8: Evaluation context and style resolution

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/registry.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/resolve-style.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/theme/visual-roles.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/scene-evaluator.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/theme/resolve-style.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/scene-evaluator.test.ts`

**Interfaces:**
- Consumes: `ResolvedTheme`, `StyleValueIr`
- Produces:

```ts
export type CapabilityEvaluationContext = Readonly<{
  atMs: number;
  theme?: ResolvedTheme;
  reducedMotion?: boolean;
}>;

export type EvaluateSceneOptions = Readonly<{
  evaluators?: FrozenCapabilityEvaluatorRegistry;
  theme?: ResolvedTheme;
  reducedMotion?: boolean;
}>;

export function resolveStyleColor(
  value: StyleValueIr | undefined,
  theme: ResolvedTheme | undefined,
  legacyFallback: string,
): string;

export function themeColor(theme: ResolvedTheme, role: ThemeRole): string;
export function themeNumber(theme: ResolvedTheme, role: ThemeRole): number;
export function themeFont(theme: ResolvedTheme, role: ThemeRole): readonly string[];
export function themeDurationMs(theme: ResolvedTheme, role: ThemeRole): number;
export function themeEnum(theme: ResolvedTheme, role: ThemeRole): string;
```

Precedence helper for contributions:

```ts
authoredProp ?? (theme ? themeColor(theme, role) : legacyFallback)
```

Foundation `pathStyle` must resolve `theme-role` entries before emitting draw commands. When `reducedMotion === true`, any path that would set progressive reveal emits `strokeReveal: 1`.

Causal draw-on for connectors: when theme is active and a timeline `trace` cue targets a connector, compute reveal from cue window using `motion.draw` / `motion.easing`; without theme, keep current immediate geometry (legacy).

- [ ] **Step 1: Write failing tests**

1. Foundation rect with `fill: { kind: "theme-role", role: "surface.raised" }` + Systems Chalk theme → command fill `#303334`.
2. Same rect with literal `#ff0000` → `#ff0000` even when theme present.
3. No theme → legacy behavior unchanged for literal-only fixtures.
4. Kind mismatch on `themeColor(theme, "stroke.standard")` throws `ThemeRoleKindError`.

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Implement resolve helpers and thread `theme` through `evaluateScene` → component evaluators (`context.theme`).**

- [ ] **Step 4: Re-run scene-evaluator + resolve-style tests** — Expected: PASS.

---

### Task 9: Migrate P0 contributions to semantic visual roles

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/theme/visual-roles.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/queue.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/waterfall.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/segment-strip.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/span-map.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/glyph-run.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/semantic-morph.ts`
- Modify: corresponding `packages/runtime/test/evaluate/*-contribution.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/registry.ts` evaluator wrappers to pass `context.theme`

**Interfaces:**
- Consumes: `CapabilityEvaluationContext.theme`, `LEGACY_VISUAL_FALLBACKS`
- Produces: shared mappings:

| Contribution | Role mapping when themed |
|---|---|
| Queue lane | `surface.panel` |
| Queue waiting | `ink.muted` |
| Queue serving | `accent.success` |
| Waterfall point | `accent.compute` |
| Waterfall interval | `accent.execution` |
| Waterfall text | `ink.primary` |
| Waterfall playhead | `accent.attention` |
| Segment strip fill | `surface.raised` |
| Segment text | `ink.primary` |
| Segment continuation | `accent.execution` |
| Span uncovered | `accent.danger` |
| Span covered | `line.structural` |
| Span edge | `accent.execution` |
| GlyphRun fill default | `ink.primary` |
| GlyphRun font default | `font.body` + `size.body` + `weight.regular` |
| Semantic morph fill default | `accent.compute` |

Stroke defaults when themed: `stroke.standard`, `stroke.cap`, `stroke.join`.

- [ ] **Step 1: Write/adjust failing contribution tests** that assert legacy colors with `theme` absent, and Systems Chalk colors/fonts/strokes with theme present.

- [ ] **Step 2: Run contribution suites** — Expected: FAIL on themed assertions.

- [ ] **Step 3: Implement mappings; do not import `systems-chalk.ts` from contribution files—only read `ResolvedTheme` / helpers.**

- [ ] **Step 4: Run**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- evaluate/queue-contribution.test.ts evaluate/waterfall-contribution.test.ts evaluate/segment-strip-contribution.test.ts evaluate/span-map-contribution.test.ts evaluate/glyph-run-contribution.test.ts evaluate/semantic-morph-contribution.test.ts evaluate/component-scene-evaluator.test.ts
```

Expected: PASS.

---

### Task 10: Canvas and SVG stroke reveal parity

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/canvas/canvas-renderer.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/svg/svg-fallback.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/canvas-renderer.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/svg-fallback.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/backend-conformance.test.tsx`

**Interfaces:**
- Consumes: `PathDrawCommand.lineCap|lineJoin|strokeReveal`, `pathLength`, `strokeDashForReveal`
- Produces: equivalent visible stroke progress on both backends

SVG:

```tsx
<path
  d={command.path}
  strokeLinecap={command.lineCap}
  strokeLinejoin={command.lineJoin}
  pathLength={command.strokeReveal === undefined ? undefined : 1}
  strokeDasharray={dash?.dashArray}
  strokeDashoffset={dash?.dashOffset}
/>
```

Normalize with `pathLength={1}` and dash values in `[0,1]` space **or** use absolute lengths consistently; pick absolute lengths from `pathLength(path)` for both backends.

Canvas: set `lineCap` / `lineJoin`, `setLineDash([length * reveal == 0 ? ...])` using cached length; fill-only paths ignore reveal.

Hit regions and semantic projection remain complete at every reveal fraction (assert in conformance test).

- [ ] **Step 1: Write failing backend tests** for `strokeReveal: 0`, `0.5`, `1`, reduced-motion forced `1`, and cap/join property application.

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Implement Canvas + SVG reveal using shared path-metrics helpers only.**

- [ ] **Step 4: Re-run backend suites** — Expected: PASS.

---

### Task 11: `FlowApp` theme override and chrome CSS mapping

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/theme/chrome-css.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/theme.css`
- Create: `apps/aiperf-flow/packages/runtime/test/theme/chrome-css.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/app.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/scripts/build-site.mjs` if font CSS must be bundled into `site.js` / static output

**Interfaces:**
- Consumes: `FrozenThemeRegistry`, `ResolvedTheme`
- Produces:

```ts
export type FlowAppProps = Readonly<{
  flow: FlowIr;
  registry?: CapabilityRegistry;
  themeRegistry?: FrozenThemeRegistry;
  themeOverride?: string;
  onThemeIdChange?(themeId: string | undefined): void;
  // existing props unchanged...
}>;

export function themeToChromeCssVars(
  theme: ResolvedTheme,
): Readonly<Record<string, string>>;
```

Mapped CSS variables (restrained subset only):

| CSS var | Role |
|---|---|
| `--flow-plane-deep` / `--flow-plane` | `surface.canvas` / `surface.panel` |
| `--flow-plane-raised` | `surface.raised` |
| `--flow-ink` / `--flow-muted` | `ink.primary` / `ink.muted` |
| `--flow-control` | `accent.control` |
| `--flow-execution` | `accent.execution` |
| `--flow-danger` | `accent.danger` |
| `--flow-focus` | `accent.focus` |
| `--flow-font-body` | `font.body` joined |
| `--flow-font-data` | `font.data` joined |
| `--flow-stroke-standard` | `stroke.standard` px |

Active theme selection inside `FlowApp`:

```ts
const activeThemeId = selectActiveThemeId({
  overrideId: themeOverride,
  documentDefault: flow.defaultTheme,
});
```

Unknown `themeOverride` throws/renders the existing error surface with `UnknownThemeIdError` message. When `activeThemeId` is undefined, do not rewrite CSS vars (legacy `theme.css` defaults remain). When set, apply `themeToChromeCssVars` as inline style on the `.aiperf-flow` root. Pass resolved theme into `evaluateScene`. Expose active id via `data-theme-id` attribute and `onThemeIdChange`.

Import fontsource CSS from runtime entry used by site/preview builds so Systems Chalk does not depend on host fonts.

- [ ] **Step 1: Write failing chrome-css + app tests** for override precedence, document default, legacy default, CSS var mapping, and “switching theme does not recompile / does not require new FlowIr object identity beyond registry resolve”.

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Implement mapping and `FlowApp` wiring.** Keep control geometry/layout CSS unchanged.

- [ ] **Step 4: Run runtime app + chrome tests** — Expected: PASS.

---

### Task 12: Preview selector, cinematic E2E, roadmap index

**Files:**
- Modify: `apps/aiperf-flow/preview/App.tsx`
- Modify: `apps/aiperf-flow/examples/cinematic/request-lifecycle.flow` (add `use theme systems_chalk` **or** keep tokens and rely on E2E host override—prefer adding `use theme systems_chalk` in a dedicated themed fixture copy if changing the flagship fixture would break unthemed expectations)
- Create: `apps/aiperf-flow/examples/cinematic/request-lifecycle-systems-chalk.flow` (preferred: themed fixture copy)
- Modify: `apps/aiperf-flow/e2e/request-lifecycle-cinematic.spec.ts` (or add `e2e/systems-chalk-cinematic.spec.ts`)
- Modify: `docs/superpowers/plans/2026-07-17-aiperf-flow-roadmap.md` artifact map entry for this plan
- Modify: any CLI/site packaging tests that assume `irVersion: 1`

**Interfaces:**
- Preview selects among `themeRegistry.ids()` and passes `themeOverride`.
- E2E asserts:
  - `.aiperf-flow[data-theme-id="systems_chalk"]`
  - computed CSS `--flow-plane` ≈ `#292C2D`
  - Canvas and forced SVG paths both mount
  - semantic twin entities/relations remain complete during early timeline beats where connectors are mid-reveal
  - raw display-list sampling via a test-only hook **or** DOM `data-draw-command-id` stroke attributes for a known connector id

- [ ] **Step 1: Write failing E2E assertions** for Systems Chalk chrome + semantic completeness under reveal.

- [ ] **Step 2: Run Playwright E2E** (after `npm run flow:build`):

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:build
npx playwright test e2e/systems-chalk-cinematic.spec.ts
```

Expected: FAIL until fixture/app wiring complete.

- [ ] **Step 3: Implement preview selector + themed fixture + E2E. Update roadmap artifact map with:**

```markdown
- typed theme system and Systems Chalk:
  [`2026-07-18-aiperf-flow-theme-system.md`](2026-07-18-aiperf-flow-theme-system.md);
```

- [ ] **Step 4: Full verification gate**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:check
npx playwright test e2e/live-cinematic-runtime.spec.ts e2e/request-lifecycle-cinematic.spec.ts e2e/systems-chalk-cinematic.spec.ts
cd /home/anthony/nvidia/projects/aiperf/ajc/rust
/usr/bin/python3 tools/check_docs_current.py
```

Expected: all PASS.

---

## Verification gate (definition of done)

Feature complete when all design completion criteria hold and:

1. `npm run flow:check` is green.
2. Schema rejects unknown roles/fields; compiler emits source-aware theme diagnostics.
3. Runtime supports document default + explicit override + legacy unthemed path.
4. Queue, Waterfall, SegmentStrip, SpanMap, GlyphRun, semantic morph, and foundation nodes consume semantic roles when themed.
5. Canvas/SVG stroke reveal parity tests pass; reduced motion forces complete strokes.
6. Player chrome reflects restrained active-theme CSS vars.
7. Unthemed fixtures still match legacy colors.
8. Systems Chalk cinematic E2E passes on Canvas and SVG fallback.

---

## Spec coverage checklist

| Design section | Task(s) |
|---|---|
| `use theme` / custom `theme ... extends` | 2, 3, 4, 5 |
| `theme(role)` unresolved in IR | 2, 5, 8 |
| Precedence (authored > theme > schema default/legacy) | 8, 9 |
| Typed role vocabulary | 1 |
| Schema/IR + version bump | 1, 5 |
| Compilation diagnostics | 4 |
| ThemeRegistry / resolve / freeze / cache | 6 |
| Evaluation helpers + contribution migration | 8, 9 |
| Display-list `lineCap`/`lineJoin`/`strokeReveal` | 7, 10 |
| Player chrome CSS subset + override | 11, 12 |
| Systems Chalk palette/type/shape/motion | 6, 9 |
| Contrast AA enforcement | 6 |
| Reduced motion complete strokes | 8, 10 |
| Compatibility / unthemed docs | 1 (upgrade), 8–9 (legacy fallbacks), 12 |
| Testing strategy matrix | 1–12 |

---

## Blockers / decisions locked by this plan

1. **IR version becomes `2`.** All hand-built IR fixtures in runtime/compiler/CLI tests must bump or call `upgradeFlowIrV1ToV2`.
2. **Duration literal `420ms` is new syntax**; timeline cue `duration <number>` stays unitless milliseconds as today.
3. **Corner radius `12` is a Systems Chalk shape constant**, not a theme role in v1 vocabulary (`SYSTEMS_CHALK_SHAPE.cornerRadiusPx`).
4. **Fonts:** Nunito Sans + IBM Plex Mono via `@fontsource/*` (OFL). No handwriting fonts.
5. **Path metrics:** `svg-path-properties` (no custom SVG path parser).
6. **Contrast library:** `culori`.
7. **Unthemed path keeps legacy literals**, not an implicit Systems Chalk activation.
8. **Bundled root sentinel:** Systems Chalk uses `extends: "__bundled_root__"` (`BUNDLED_ROOT_BASE`); authors may not extend or declare that id.
9. **No commit steps** in execution; commit only if the user explicitly asks.

No open product blockers remain for implementation to start at Task 1.
