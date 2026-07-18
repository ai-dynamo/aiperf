<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Flow Immersive Preview Design

## Status

Approved design for replacing the preview's video-player grammar with a native,
fullscreen **Causal Field** experience. All four immersive capabilities ship.
**Causal Replay** and **Command Constellation** are the primary interaction
spine; **Context Lens** and **Focus World** are contextual tools.

## Goal

The evaluated scene is the application surface rather than content framed
inside a player. AIPerf Flow should feel like an interactive instrument for
understanding inference systems:

- the scene owns the viewport;
- controls live in a restrained edge HUD;
- authored causal beats replace generic media progress;
- entities, evidence, narration, and semantic navigation remain coequal;
- exploration changes the view without changing authored scene truth.

The preview and packed `FlowApp` use the same visual language. Hosts may arrange
their outer shell differently, but the scene field, causal navigation, semantic
surface, and interaction vocabulary do not become separate products.

## Product principles

### Scene is the app

The stage is full bleed within the available viewport. It has no glass card,
decorative frame, drop shadow, or persistent dashboard columns. Scene geometry
continues to come exclusively from evaluated scene and display-list contracts.

### Causality replaces playback

The primary progress model is an ordered set of authored beats such as arrival,
admission, first token, stream, and terminal. Direct beat selection seeks the
integer virtual clock. Continuous playback and direct seek to a beat produce
equal evaluated state.

### Power without permanent chrome

Common actions remain immediately available in a compact edge HUD. Dense
navigation, entity search, evidence access, and theme/accessibility controls
live in Command Constellation. The HUD may auto-hide during uninterrupted
playback, but reappears on pointer movement, focus, pause, exploration, or
keyboard invocation.

### Semantics remain coequal

Canvas is never the source of truth. The semantic twin stays mounted whenever
the visual stage is mounted. Context Lens and Focus World operate on stable
semantic IDs and serializable interaction state. SVG/HTML fallback preserves
the same entities, beats, focus, selection, evidence, and controls.

### Systems Chalk alignment

The preview chrome consumes the restrained host-facing subset of the resolved
theme. Scene paints remain literal evaluated display-list values. Backends do
not branch on theme IDs.

## Visual system

### Chrome roles

The initial Systems Chalk mapping is:

- **Board** `#232526`: `surface.canvas`;
- **Panel** `#292C2D`: `surface.panel`;
- **Chalk** `#F1F3F2`: `ink.primary`;
- **Guide** `#AEB4B5`: `ink.muted`;
- **Signal** `#71D8D0`: `accent.control` and focus;
- **Beat** `#F0CF58`: `accent.attention` for the active causal beat.

The chrome does not use acid green, ambient glow, glass blur, or theme-specific
backend conditionals. High contrast and forced colors remain orthogonal runtime
policies.

### Typography

- Body, titles, and command labels use the resolved body/display stack.
- Beat indices, metrics, IDs, and command categories use the resolved data
  stack with tabular numerals.
- Handwriting is not an identity device.

### Signature

The memorable element is the **causal path HUD**: a thin authored-beat path
whose completed, active, and future states mirror runtime timeline state.
It is not a free-form media scrubber and does not introduce a second clock.

## Layout

### Desktop

```text
┌─────────────────────────────────────────────────────────────┐
│ AIPerf Flow / scene title                 scene · beat · live│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                 EVALUATED SCENE FIELD                       │
│                                                             │
│ [context lens when selected]          [captions safe area]  │
│                                                             │
│ [pause] [explore] [twin] [command]       [fullscreen]       │
├─────────────────────────────────────────────────────────────┤
│ CAUSAL PATH  ●──●──●──○──○                     BEAT 3 / 5   │
└─────────────────────────────────────────────────────────────┘
```

The document browser is an overlay or edge drawer. Opening it must not shrink
the scene into a framed card.

### Mobile

```text
┌───────────────────────┐
│ ≡ Flow      beat · live│
│                       │
│      SCENE FIELD      │
│                       │
│ [lens / captions]     │
│ [▶] [explore] [⌘] [⋯] │
│ ●──●──●──○──○         │
└───────────────────────┘
```

Primary controls remain within thumb reach. Secondary controls move into
Command Constellation. The page has no horizontal overflow at 390×844.

## Capability 1: Context Lens

Context Lens is an edge-attached in-scene inspector for the selected semantic
entity.

It shows only data already present in evaluated or linked runtime state:

- label, role, and description;
- directly related entities;
- available metrics and timeline anchors;
- evidence references and source locations;
- actions to focus, compare, or open the semantic twin.

The lens:

- opens from Canvas hit selection or semantic activation;
- uses the same selected entity ID as all backends;
- is keyboard reachable and dismissible;
- collapses into a bottom sheet on narrow screens;
- never computes domain semantics independently.

Missing optional data is omitted rather than represented by invented values.

## Capability 2: Causal Replay

Causal Replay is the primary navigation and playback model.

### Beat model

Each visible tick references an existing authored cue or stable timeline
anchor. The runtime derives:

- stable beat ID;
- label and optional description;
- integer `timeMs`;
- completion, active, and future state;
- linked semantic entity IDs when authored.

When explicit named beats are absent, the shell may expose scene cue boundaries.
It must not infer domain-specific names from draw commands.

### Interaction

- Click or activate a beat to seek directly.
- Left/right traverse beats; Home/End select first/last.
- Play advances virtual time through the same evaluator.
- Exploration pauses at the exact current integer timestamp.
- Resume continues from that timestamp and restores authored camera policy.
- The URL may encode scene and beat IDs, never wall-clock progress.

Reduced motion changes transitions, not resulting semantic state.

## Capability 3: Focus World

Focus World temporarily makes one entity or trace the visual center of the
experience.

It is serializable exploration state containing:

- focused semantic entity ID;
- prior selection and camera takeover state;
- optional comparison entity ID;
- entry timestamp and restoration policy.

Unrelated entities may be visually muted, but remain present in the semantic
twin and fallback. Focus World does not delete semantic content or mutate the
authored scene. Escape or Resume restores the exact prior beat, focus target,
and authored camera according to scene policy.

## Capability 4: Command Constellation

Command Constellation is the native power surface, invoked by a visible command
button or keyboard shortcut.

It indexes:

- scenes;
- causal beats;
- semantic entities and relations;
- evidence references;
- runtime actions such as play, pause, explore, resume, fit, and fullscreen;
- semantic twin, captions, narration, theme, contrast, and motion controls.

### Command contract

Commands have stable IDs, user-facing labels, categories, optional shortcuts,
availability state, and one runtime action. Search is deterministic,
case-insensitive, and ordered by exact prefix, token prefix, then authored
order. Command execution uses existing runtime actions rather than duplicating
state mutation.

The palette:

- traps focus while open and restores the invoking element on close;
- supports arrow navigation, Enter, and Escape;
- exposes disabled reasons when an action is unavailable;
- uses plain language, not internal capability IDs;
- becomes a full-height sheet on mobile.

## Fullscreen and HUD behavior

The fullscreen action uses the browser Fullscreen API when available and a
layout-only immersive mode otherwise. Fullscreen failure is recoverable and
does not interrupt playback.

The HUD has three visibility states:

- **present** while paused, exploring, focused, or keyboard-active;
- **quiet** during normal playback;
- **hidden decorative chrome** after inactivity during playback.

Captions, focus indicators, semantic controls, errors, and active dialogs never
auto-hide. Pointer, touch, or keyboard activity restores the HUD immediately.
Tests use injected or explicit activity events rather than timing sleeps.

## Runtime data flow

```text
validated Flow IR + resolved theme + interaction snapshot + integer time
                              │
                              ▼
                        evaluateFrame
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
          EvaluatedScene   DisplayList   Quality report
                 │            │
      ┌──────────┼────────────┼──────────┐
      ▼          ▼            ▼          ▼
  semantic    Canvas/SVG   hit/damage   causal HUD
    twin       backend      helpers      + commands
      │
      └──── selection / focus / exploration actions ────┘
```

`evaluateFrame` is the public pure composition seam. It evaluates the scene,
builds the display list, applies quality policy, and returns diagnostics. Hit
indexes and damage computation consume its output. Neither preview nor packed
site reinterprets Flow IR independently.

## URL and restoration state

The URL may encode:

- document or packed flow identifier;
- scene ID;
- causal beat ID;
- selected semantic entity ID when explicitly shareable;
- exploration mode only when the state is serializable.

Transient pointer position, dialog focus, wall time, and backend-specific state
do not enter the URL. Invalid IDs fail closed to the first valid scene/beat and
produce a recoverable diagnostic.

## Error behavior

- Canvas failure renders SVG/HTML from the same evaluated frame.
- Fullscreen denial keeps the current layout and announces the result.
- Unsupported commands remain discoverable only when a useful disabled reason
  exists; otherwise they are omitted.
- Missing evidence shows that no evidence is attached; it does not create a
  dead link.
- Evaluation or load failures remain inside the existing error boundaries and
  leave navigation, transcript, and semantic controls reachable.

## Accessibility

- The semantic twin is always mounted and never `display: none` or
  `aria-hidden`.
- Causal beats form one keyboard-operable navigation control with an accessible
  current-beat state.
- Context Lens has a labelled region and does not steal focus on pointer-only
  selection.
- Focus World preserves semantic reading order.
- Command Constellation uses dialog/listbox semantics with deterministic focus
  restoration.
- Captions remain in a safe area and are not hidden with the HUD.
- Forced colors, high contrast, reduced motion, reduced transparency, no depth,
  Canvas failure, and print retain meaning and controls.

## Scope

### In scope

- `preview/App.tsx` and `preview/styles.css`;
- mounted runtime shell and theme CSS convergence;
- `evaluateFrame` composition and public exports;
- causal beat projection;
- Context Lens, Focus World, Command Constellation, fullscreen, and HUD policy;
- URL restoration;
- Playwright harness, deterministic screenshots, accessibility and responsive
  tests;
- runtime measurement sample collection and quality report generation.

### Out of scope

- new domain-specific scene geometry;
- backend-specific theme branches;
- WebGPU;
- arbitrary CSS or scripts in `.flow`;
- a general animation language;
- inferred domain causality from pixels or draw command order;
- changing narrator synthesis internals.

## Verification

### Unit and integration

- direct seek equals continuous playback at every causal beat;
- `evaluateFrame` composition order and deterministic hashes;
- command search/order/action dispatch;
- Context Lens selection/focus parity;
- Focus World entry/restoration;
- HUD state transitions without arbitrary sleeps;
- fullscreen success, denial, and fallback;
- URL parse/serialize round trips;
- Canvas/SVG/semantic parity.

### Screenshot matrix

1. 1440×900 playing, browser closed;
2. 1440×900 browser drawer open;
3. 1440×900 Context Lens open;
4. 1440×900 Focus World active;
5. 1440×900 Command Constellation open;
6. 1440×900 captions at a named cue;
7. 390×844 playing;
8. 390×844 command sheet;
9. SVG fallback;
10. reduced motion;
11. forced colors/high contrast;
12. 3840×2160 reference fidelity.

Snapshots use fixed fonts, viewport, device scale, timeline time, random seed,
and authored fixture. Performance reports record evaluation, draw, total frame,
and memory metrics with environment metadata. Semantic, determinism,
accessibility, and visual regressions fail CI; hardware-sensitive performance
changes report their environment.

## Acceptance criteria

1. The scene is the dominant viewport surface with no player card or media
   scrubber.
2. Causal Replay and Command Constellation are the primary navigation model.
3. Context Lens and Focus World operate through stable semantic IDs and
   serializable exploration state.
4. Preview and packed `FlowApp` share Causal Field vocabulary and Systems Chalk
   chrome roles.
5. Canvas, semantic twin, and SVG fallback preserve equal entities, relations,
   beats, focus, selection, evidence, and actions.
6. Direct beat seek and continuous playback produce equal evaluated state.
7. Desktop, mobile, fullscreen, reduced-motion, forced-colors, and fallback
   verification pass.
8. Authors still commit only `.flow` and referenced assets; no
   document-specific React, TypeScript, JavaScript, or CSS is introduced.
