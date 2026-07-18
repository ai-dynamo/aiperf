# AIPerf Flow Browser Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype an original hierarchical Flow browser and compact
upper-left scene header while preserving the stage as the dominant surface for
the live cinematic renderer.

**Architecture:** The preview is temporary shell scaffolding around the runtime,
not the final scene implementation. `App.tsx` may own fixture browser data while
the manifest/navigation contracts are incomplete; `styles.css` provides a
narrow instrument-browser shell, compact stage header, and responsive
drawer-like mobile presentation. The production shell must consume packed Flow
IR and manifests, host the Canvas 2D visual renderer, keep the semantic HTML
twin available, and expose the SVG/HTML fallback without document-specific UI.

**Tech Stack:** React, TypeScript, CSS, Vite.

## Global Constraints

- Do not copy the Architecture Atlas sidebar.
- Preserve the approved graphite flight-deck palette and horizontally scrollable chapter rail.
- Browse Flow files, chapters, and scenes in one collapsible hierarchy.
- Keep navigation accessible with semantic buttons, labels, focus states, and responsive behavior.
- The live scene, not application chrome, is the product. Browser and transport
  controls collapse or auto-hide without removing keyboard access, captions,
  transcript, or navigation.
- The shell must not assume SVG, inspect Canvas pixels for semantics, or
  duplicate scene state. It consumes runtime navigation, timeline, camera,
  semantic-twin, and inspector interfaces.
- Decorative preview controls may exist only when visibly marked as prototype
  scaffolding. Production controls must execute real runtime actions.
- Viewer interaction pauses at the current beat by default. Resume continues
  from that beat and restores the authored camera according to scene policy.
- The 3840×2160 profile verifies visual fidelity; the shell remains responsive
  and is never baked into scene output.
- Do not create a git commit unless the user explicitly requests one.

## Status and boundary

Task 1 is a completed visual prototype. Its fixture-owned file tree, hard-coded
chapter index, and inactive canvas controls are not production contracts. The
production browser is driven by the packed manifest and narrative graph; the
production stage is driven by the evaluated scene and deterministic clock.

---

### Task 1: Add the Flow browser and compact scene composition

**Files:**
- Modify: `apps/aiperf-flow/preview/App.tsx`
- Modify: `apps/aiperf-flow/preview/styles.css`

**Interfaces:**
- Consumes: existing `SceneRenderer`, chapter fixture, and playback state.
- Produces: `flow-workspace`, `flow-browser`, and `story-content` layout regions.

- [x] **Step 1: Add the hierarchical browser fixture and collapse state**

Define typed file, chapter, and scene rows in `App.tsx`; render them as nested semantic lists with the active scene marked using `aria-current`.

- [x] **Step 2: Move narrative content into a compact stage header**

Replace the large two-column brief with a compact upper-left header inside the stage and allow `story-figure` to consume the remaining area.

- [x] **Step 3: Style the original instrument browser and expanded scene**

Use a narrow rail with compact typography, tree guides, cyan active edge, search affordance, and a collapse control. Keep the scene palette and chapter rail visually unchanged.

- [x] **Step 4: Add responsive behavior**

Collapse the browser at tablet widths and expose its identity as a compact top control without reducing the scene below a usable minimum.

- [ ] **Step 5: Verify**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:build
```

Expected: all Flow workspace packages compile successfully.

Run IDE diagnostics for `preview/App.tsx` and `preview/styles.css`.

Preview IDE diagnostics report no errors in `preview/App.tsx` or
`preview/styles.css`. The current workspace gate is not green: the 2026-07-18
run reached 43 passing and 3 failing runtime test files, with failures in Canvas
text-atlas measurement, audio-consent dialog naming, and semantic-twin table
fallback plus narrator-worker rejections in jsdom. Rerun Step 5 after those
runtime changes stabilize.

---

### Task 2: Replace prototype ownership with runtime contracts

**Files:**
- Modify: `apps/aiperf-flow/preview/App.tsx`
- Modify: `apps/aiperf-flow/preview/styles.css`
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`
- Test: `apps/aiperf-flow/packages/runtime/test/app.test.tsx`

**Interfaces:**
- Consumes: packed manifest navigation, deterministic timeline state, camera
  takeover/resume actions, semantic-twin focus state, and renderer quality
  profile.
- Produces: a shell that can host Canvas, semantic HTML, and SVG/HTML fallback
  backends without knowing scene geometry.

- [ ] Replace hard-coded files, chapters, scenes, title, and progress with
  manifest and narrative-graph data.
- [ ] Wire play, pause, seek, select, pan, zoom, fit, inspect, and resume to
  runtime actions; remove controls that have no implementation.
- [ ] Pause the timeline when exploration begins and verify resume from the
  exact beat with authored-camera restoration.
- [ ] Mount the semantic HTML twin adjacent to the visual surface and verify
  keyboard focus and visual selection remain synchronized.
- [ ] Verify the shell at 3840×2160, desktop, tablet, and mobile sizes with
  Canvas, SVG/HTML fallback, reduced motion, high contrast, and no depth.
