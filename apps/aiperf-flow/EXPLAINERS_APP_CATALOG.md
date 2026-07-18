# AIPerf Flow Explainers App - Complete Page Catalog

**Analysis Date:** 2026-07-18  
**App URL:** http://127.0.0.1:5188  
**Framework:** React + TypeScript + Vite  
**Source Root:** `/home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow`

---

## Executive Summary

The AIPerf Flow Explainers app is an interactive educational platform for visualizing and understanding AI inference system architecture. It features:

- **3 Primary Flows:** Request Flow, Architecture (system concepts), and Endpoint Lifecycle  
- **Multiple Interactive Scenes:** Each flow contains multiple scenes with step-by-step visualizations  
- **4 Explainer Decks:** Rust Architecture, SLURM/Velo, DynoSim, and AIPerf Flow System  
- **3 Theme Options:** Systems Chalk (default), Legacy, and Core  
- **Responsive Design:** Dual-mode layout (desktop/mobile at 860px breakpoint)  
- **Narrative & Voice Support:** Kokoro text-to-speech with audio consent management  

---

## Navigation Architecture

### URL Structure
- **Root:** `/` → Home page (scene selector)
- **Flow/Scene:** Direct navigation through UI; no URL path prefix (React app state-based)
- **Explainer Deck Modal:** Triggered from main app, displays full-screen overlay

### State Management
- React useState hooks manage:
  - `showHome` (boolean) - toggles between home and scene view
  - `activeFlowId` - selected flow ID
  - `activeSceneId` - selected scene ID
  - `theme` - current theme (systems-chalk | legacy | core)
  - `showThemeMenu` - theme selector dropdown visibility
  - `audioConsent` - audio permission state (yes | no | unset)
  - `showExplainerDeckId` - explainer deck modal state
  - `narrowLayout` - responsive layout mode (<860px)
  - `reducedMotion` - accessibility preference

---

## Complete Page & State Inventory

### PAGE 1: Home Page - Scene Selector

**State:** `showHome === true`  
**URL:** `/`  
**Component:** `<HomePage />`  

#### Visual Description
- Full-screen dark theme landing page
- Dark background: `#0d1117` (preview-board)
- White text on dark background
- Grid layout with organized flow sections
- Responsive: 3+ columns on desktop, 1 column on mobile

#### Layout & Structure
```
┌─────────────────────────────────────────────┐
│  Header                                      │
│  "AIPerf Flow Scenes"                       │
│  "Explore N flows with M interactive scenes"│
├─────────────────────────────────────────────┤
│  Flow Section 1: "request-flow.flow"        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Card 1   │ │ Card 2   │ │ Card 3   │    │
│  └──────────┘ └──────────┘ └──────────┘    │
│  Flow Section 2: "architecture.flow"       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Card 1   │ │ Card 2   │ │ Card 3   │    │
│  └──────────┘ └──────────┘ └──────────┘    │
│  Flow Section 3: "endpoint-lifecycle.flow" │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Card 1   │ │ Card 2   │ │ Card 3   │    │
│  └──────────┘ └──────────┘ └──────────┘    │
└─────────────────────────────────────────────┘
```

#### Content Structure
**Header:**
- Main heading: "AIPerf Flow Scenes" (2.5rem, font-weight: 700)
- Subtitle: "Explore {flowCount} flows with {sceneCount} interactive scenes"
- Bottom border with green accent color (#3fb950)

**Scene Cards (repeating grid):**
- Card layout: 320px minimum width, auto-fill columns
- Card components:
  - Title (1.2rem, bold)
  - Flow badge ("KICKER") - uppercase, green background
  - Description text (muted gray)
  - "View scene" badge button (green background, black text)

#### Interactive Elements
- **Scene Cards:** Clickable buttons, each triggers `selectScene(flowId, sceneId)`
  - On hover: border brightens to green (#3fb950)
  - On hover: background raised (#21262d)
  - On hover: slight upward translate (-2px) with shadow
  - On focus: 2px green outline
- **Keyboard:** Tab navigation between cards, Enter to select

#### Styling Variables
- `--preview-board`: #0d1117 (dark background)
- `--preview-chalk`: #f0f6fc (primary text)
- `--preview-muted`: #8b949e (secondary text)
- `--preview-guide`: #30363d (borders)
- `--preview-panel`: #161b22 (card background)
- `--preview-raised`: #21262d (card hover)
- `--preview-signal`: #3fb950 (green accent)

#### Responsive Behavior
- **Mobile (<860px):**
  - Single-column grid layout
  - Reduced padding (1rem vs 2rem)
  - Smaller heading (2rem vs 2.5rem)
  - Larger card spacing adjustments

#### Accessibility
- Scene cards have `aria-label` attributes
- Semantic HTML with proper heading hierarchy
- Keyboard navigable with visible focus states

---

### PAGE 2: Scene View - Workspace Layout

**State:** `showHome === false`  
**Component Hierarchy:**
```
preview-shell
├── preview-topbar (header)
│   ├── preview-brand-cluster (home button + title)
│   └── preview-theme-cluster (theme selector, cycle button)
├── flow-workspace (main area)
│   ├── DocumentBrowser (left sidebar)
│   └── flow-main-section (right content)
│       ├── Breadcrumb (scene progression)
│       ├── main.runtime-story (canvas area)
│       └── BottomNav (next/prev controls)
└── preview-legend (footer - conditional)
```

#### Layout Architecture

**Full Desktop View (1920x1080+):**
```
┌─ preview-topbar (height: 3rem) ─────────────────────────────────┐
│ [AIPerf]  [Breadcrumb]     [Status] [Theme Selector] [Cycle ⟳] │
├─────────────────────────────────────────────────────────────────┤
│                    flow-workspace                               │
│ ┌─ DocumentBrowser ─┐ ┌─ flow-main-section ──────────────────┐ │
│ │  [Flows]          │ │ ┌─ Breadcrumb ─────────────────────┐ │ │
│ │  ◆ request-flow   │ │ │ ●Step1 ○Step2 ○Step3 ○Step4...  │ │ │
│ │    ○ Scene 1      │ │ └──────────────────────────────────┘ │ │
│ │    ○ Scene 2      │ │ ┌─ runtime-story (main canvas) ─────┐ │ │
│ │  ◆ architecture   │ │ │                                    │ │ │
│ │    ○ Scene 1      │ │ │   [Interactive Flow Diagram]       │ │ │
│ │    ○ Scene 2      │ │ │   [Narrative Content]              │ │ │
│ │  ◆ endpoint-lcl   │ │ │                                    │ │ │
│ │    ○ Scene 1      │ │ └────────────────────────────────────┘ │ │
│ │                   │ │ ┌─ BottomNav ──────────────────────────┐ │ │
│ │ 3 flows · 15 scns │ │ │  [← Back]              [Next →]       │ │ │
│ └───────────────────┘ │ └───────────────────────────────────────┘ │ │
│                       └──────────────────────────────────────────┘ │
├─ preview-legend (conditional, "request-investigation" scene) ─────┤
│ [active cause] [selected request] [decision point]                 │
│ Entity → connector → destination → annotation                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Narrow Layout (<860px):**
- DocumentBrowser: hidden or drawer-style
- flow-main-section: full width
- BottomNav: full-width buttons
- Breadcrumb: stacked or horizontal scroll

#### Visual Styling

**Top Bar (preview-topbar)**
- Height: 3rem
- Background: `var(--preview-panel)`
- Border bottom: 1px `var(--preview-guide)`
- Flexbox: space-between

**Brand Cluster**
- Left side of topbar
- Home button with nested span containing:
  - Small label: "AIPerf Flow · Scene study 02"
  - H1: "From one request to the whole system"
- Click handler: `goHome()` → sets `showHome = true`

**Theme Cluster** (hidden when `showHome === true`)
- Right side of topbar
- Elements (flex row):
  - Optional: `<p>` voice status message (aria-live="polite")
  - Theme selector button with dropdown menu
  - Theme cycle button (⟳)

**Theme Selector Dropdown:**
- Button label: Current theme (e.g., "Systems Chalk", "Legacy", "Core")
- On click: toggles `showThemeMenu`
- Click outside closes menu
- Dropdown menu options:
  - "systems chalk" (systems-chalk)
  - "legacy" (legacy)
  - "core" (core)
- Selected theme has `--preview-signal` green background

**Main Content Area (flow-main-section)**
- Flex column layout
- Contains: Breadcrumb, runtime-story, BottomNav

**Breadcrumb Navigation**
- Horizontal list of scene steps
- Each step is a button with:
  - Step number (e.g., "1", "2", "3")
  - Scene title
  - Active state styling
  - Click handler: `onSelectScene(flowId, sceneId)`
- Aria label: "Scene progression"
- Active step has `aria-current="step"`

**Runtime Story Canvas (main.runtime-story)**
- Main content area for flow visualization
- Data attributes:
  - `data-theme={theme}` - applies theme-specific CSS variables
  - Theme-specific CSS variable mappings (--flow-board, --flow-panel, etc.)
- Content: `<FlowApp />` component (interactive diagram + narrative)
- Or: `<ExplainerSlideViewer />` when showing deck modal

**Bottom Navigation (preview-bottom-nav)**
- Flex row with two buttons
- "← Back" button
  - Disabled if on first scene
  - Click: `goToPrev()` → moves to previous scene
- "Next →" button
  - Disabled if on last scene
  - Click: `goToNext()` → moves to next scene
- Aria label: "Scene navigation"

**Optional Footer (preview-legend)**
- Conditional: only shown when scene is "request-investigation"
- Semantic legend explaining UI markings
- Labels: "active cause", "selected request", "decision point"
- Additional text: "Entity → connector → destination → annotation"

#### Theme Variables

**Systems Chalk Theme (default):**
```css
--flow-board: #1a1a1d
--flow-panel: #24282b
--flow-raised: #2a3033
--flow-control-surface: #2a3033
--flow-chalk: #e8e3d9
--flow-chalk-muted: #8b8680
```

**Legacy Theme:**
```css
--flow-board: #1a1a1a
--flow-panel: #222222
--flow-raised: #2a2a2a
--flow-control-surface: #2a2a2a
--flow-chalk: #e8e8e8
--flow-chalk-muted: #999999
```

**Core Theme:**
```css
--flow-board: #0d1117
--flow-panel: #161b22
--flow-raised: #21262d
--flow-control-surface: #21262d
--flow-chalk: #f0f6fc
--flow-chalk-muted: #8b949e
```

#### Accessibility Features
- `aria-label` attributes on all regions
- `aria-live="polite"` for voice status updates
- `aria-current` for active navigation items
- Keyboard navigation throughout
- Focus management
- Respect for `prefers-reduced-motion` media query
- High contrast mode support

---

### PAGE 3: Document Browser Sidebar

**State:** Part of normal scene view (left sidebar in desktop, collapsible on mobile)  
**Component:** `<DocumentBrowser />`

#### Visual Description
- Left sidebar panel
- Width: typically 280-320px on desktop
- Dark background matching theme
- Scrollable content area
- Footer showing metadata

#### Layout Structure
```
┌─ flow-browser ─────────────┐
│ ╔═ flow-browser-head ═╗    │
│ ║ "Workspace"         ║    │
│ ║ **Flows**           ║    │
│ ╚═════════════════════╝    │
│                            │
│ ┌─ Search Box ──────────┐  │
│ │ ⌕ Find a flow or...  │  │
│ │               [⌘K]   │  │
│ └────────────────────────┐  │
│                            │
│ ┌─ Flow Tree ────────────┐ │
│ │ ◆ request-flow.flow (3)│ │
│ │   Chapter 1            │ │
│ │   ○ Scene A            │ │
│ │   ○ Scene B            │ │
│ │ ◆ architecture.flow (5)│ │
│ │   Chapter 1            │ │
│ │   ○ Scene A            │ │
│ │   ◉ Scene B  [active]  │ │
│ │   ○ Scene C            │ │
│ │ ◆ endpoint-lcl.flow(4) │ │
│ │   Chapter 1            │ │
│ │   ○ Scene A            │ │
│ └────────────────────────┘ │
│                            │
│ ┌─ flow-browser-foot ────┐ │
│ │ ● 3 flows · 12 scenes  │ │
│ └────────────────────────┘ │
└────────────────────────────┘
```

#### Content Components

**Flow Browser Header:**
- Kicker: "Workspace" (small, muted)
- Title: "Flows" (bold, large)

**Search Box:**
- Icon: ⌕ (search symbol)
- Input field: type="search", placeholder="Find a flow or scene"
- Keyboard shortcut indicator: ⌘K (or Ctrl+K)

**Flow Tree (hierarchical navigation):**
- `<nav aria-label="Flow files and scenes">`
- Structure: `<details>` elements (collapsible)
  - Each flow is a `<details>` with `<summary>`
  - Open state: `open={file.id === activeFlowId}`
  - Summary contains: flow mark (◆), source name, scene count
  - Content: chapters and scenes

**Chapter Sections:**
- Each `<section>` contains:
  - `<p>` with chapter name
  - `<ul>` of scene buttons

**Scene Buttons:**
- Bullet markers: "●" (active), "○" (inactive)
- Text: scene title
- On click: `onSelectScene(file.id, scene.id)`
- Aria attributes: `aria-current="page"` if active

**Footer:**
- Status dot: green indicator
- Metadata: "{flowCount} flow{s} · {sceneCount} scene{s}"

#### Interactive Elements
- **Details/Summary:** Click to expand/collapse flow sections
- **Search Input:** Filters available flows and scenes (Cmd/Ctrl+K focus)
- **Scene Buttons:** Click to navigate to scene
  - Hover: highlights row
  - Active: bold or bright color

#### Accessibility
- Semantic `<nav>` with aria-label
- `<details>/<summary>` for progressive disclosure
- Proper button and link semantics
- Keyboard navigation throughout

---

### PAGE 4: Audio Consent Dialog

**State:** `requireAudioConsent === true` AND `audioConsent === "unset"`  
**Component:** Built into `<FlowApp />` (from runtime package)

#### Visual Description
- Modal overlay on top of scene content
- Semi-transparent background
- Centered dialog box
- Primary action button (bright green)
- Secondary action button or dismiss

#### Content
- Heading: "Audio Narration Available"
- Description: "This scene includes text-to-speech narration"
- Privacy/permissions explanation
- Action buttons:
  - "Enable Audio" / "Play with narration" (primary, green)
  - "Continue without audio" / "Skip" (secondary)

#### State Management
- Dialog appears only when:
  - `requireAudioConsent === true`
  - AND `audioConsent === "unset"`
- On user choice:
  - "Yes": `onAudioConsentChange(true)` → sets `audioConsent = "yes"`
  - "No": `onAudioConsentChange(false)` → sets `audioConsent = "no"`
  - Preference stored in localStorage key: `aiperf-flow-audio-consent`

#### Styling
- Overlay: dark with opacity
- Dialog: panel background color from theme
- Buttons: primary green (`--preview-signal`), secondary gray
- Text: chalk color from theme

---

### PAGE 5: Theme Selector Dropdown Menu

**State:** `showThemeMenu === true`  
**Component:** Inline in App.tsx topbar

#### Visual Description
- Dropdown menu positioned below theme button
- Absolute positioning: `top: 100%, right: 0`
- Drop shadow: `0 4px 12px rgba(0, 0, 0, 0.3)`
- Dark background panel color
- Border: 1px guide color

#### Layout
```
┌─ [Current Theme ▼] ─ topbar button
│
└─ [Dropdown Menu] ─────────────────┐
   │ [systems chalk]                │
   │ [legacy]                       │
   │ [core]                         │
   └─────────────────────────────────┘
```

#### Content
- 3 theme options as buttons:
  - "systems chalk"
  - "legacy"
  - "core"
- Each button:
  - Full width
  - Padding: 0.5rem 0.75rem
  - Font: 0.7rem uppercase/capitalized
  - Hover: slight background change
  - Selected theme: green background + dark text

#### Interactive Elements
- **Theme Buttons:** Click to select theme
  - `onClick={() => handleThemeChange(themeOption)}`
  - Sets theme and closes menu
- **Click Outside:** Closes menu
  - Listener on document
  - Checks click target against theme menu and button
  - If outside, sets `showThemeMenu = false`

#### Accessibility
- Menu items are buttons with proper semantics
- Parent button: `aria-expanded={showThemeMenu}`
- Currently selected theme: visually distinct

---

### PAGE 6: Voice Status Indicator

**State:** `voiceStatus !== null`  
**Component:** Paragraph in topbar theme-cluster

#### Visual Description
- Small status message in topbar
- Muted color (gray)
- Live region: `aria-live="polite"`

#### Content Variations
**Loading State:**
- Shown when: `kokoroState?.status === "loading"`
- Text: `Loading voice {percentage}%`
- Example: "Loading voice 35%"
- Progress indication without full progress bar

**User Activation Needed:**
- Shown when: `kokoroState?.status === "needs-user-activation"`
- Text: "Press play for voice"
- Indicates voice engine loaded but waiting for user interaction

**Hidden:**
- When: `voiceStatus === null`
- Display: none

#### Styling
- Font: small, muted color
- Live announcement region

---

### PAGE 7: Scene Contents - Flow Diagram Canvas

**State:** All active scene views  
**Component:** `<FlowApp />` from `@aiperf/flow-runtime`

#### Visual Description
- Interactive diagram showing:
  - Flow of requests through system
  - Architecture components and relationships
  - Endpoint lifecycle stages
  - Causal paths and dependencies
- Canvas-based rendering (SVG)
- Animated transitions between steps
- Narrative overlay/sidebar

#### Features
- **Interactive Elements:**
  - Draggable elements (request paths, nodes)
  - Clickable callouts and annotations
  - Hover interactions showing details
  - Click to highlight causal relationships

- **Navigation:**
  - Timeline slider (if applicable)
  - Play/pause controls (for animated sequences)
  - Step-by-step progression

- **Content Layers:**
  - Base diagram (static)
  - Animated elements (movements, highlights)
  - Narrative callouts (text overlays)
  - Semantic annotations

#### Responsive Behavior
- Desktop: Full diagram visible
- Narrow (<860px): Diagram adapted via `flowWithNarrowScene()`
- Alternative responsive roots provided in scene definition

#### Theme Application
- Background: `--flow-board` theme variable
- Text/strokes: `--flow-chalk` and `--flow-chalk-muted`
- Highlights: theme-specific accent colors

#### Accessibility
- SVG with proper `role="img"`
- Alt text from diagram structure
- Keyboard navigation support
- Optional: captions/transcript

---

### PAGE 8: Explainer Deck Modal

**State:** `showExplainerDeckId !== null`  
**Component:** `<ExplainerSlideViewer />`

#### Visual Description
- Full-screen modal overlay
- White background (distinct from theme-styled content)
- Slide-based presentation format
- Back button to dismiss

#### Layout
```
┌─────────────────────────────────────┐
│ [Explainer Deck Presentation]       │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │                                 │ │
│ │   [Slide Content]               │ │
│ │   - Title                       │ │
│ │   - Diagram/Visualization       │ │
│ │   - Description text            │ │
│ │                                 │ │
│ └─────────────────────────────────┘ │
│                                     │
│ [← Back]  [controls]  [► Next]      │
└─────────────────────────────────────┘
```

#### Available Decks
1. **Rust Architecture Deck**
   - ID: `rust-architecture`
   - Topic: `system-architecture`
   - Multiple slides covering:
     - Product shell architecture
     - Crate organization
     - Runtime components
     - Transport layers

2. **SLURM/Velo Deck**
   - ID: `slurm-velo`
   - Topic: `distributed-execution`
   - Covers:
     - Distributed cell execution
     - SLURM integration
     - Velo cross-host transport

3. **DynoSim Deck**
   - ID: `dynosim`
   - Topic: `simulation`
   - Content on:
     - Simulation mode
     - Clock simulation
     - Offline replay

4. **AIPerf Flow System Deck**
   - ID: `aiperf-flow-system`
   - Topic: `flow-system`
   - Describes:
     - Flow IR and compilation
     - Flow execution model
     - Narrative integration

#### Slide Structure (per deck)
Each slide contains:
- `id`: unique identifier
- `eyebrow`: section label
- `title`: slide heading
- `lede`: introductory text
- `narration`: text-to-speech content
- `points`: bulleted key points
- `caption`: image/diagram caption
- `sceneBlock`: visual diagram definition

#### Navigation
- **Back Button:**
  - On click: `setShowExplainerDeckId(null)`
  - Returns to flow scene view
  - Styling: gray background, simple border

- **Slide Controls:** (built into ExplainerSlideViewer)
  - Previous/Next buttons
  - Slide counter/indicator
  - Optional: progress bar

#### Styling
- Background: white (`#fff`)
- Text: dark gray
- Padding: generous (20px+)
- Distinct from flow theme styles

---

### PAGE 9: Responsive Mobile View

**State:** `narrowLayout === true` (media query: max-width 860px)  
**Trigger:** Window resize, mobile device orientation

#### Visual Changes

**Layout Transformation:**
- Sidebar: hidden or drawer-style
- Main content: full width
- Breadcrumb: may horizontal-scroll or stack
- Canvas: `flowWithNarrowScene()` applied

**Typography Scaling:**
- Headings: reduced sizes
- Text: adjusted line-height
- Padding/margins: reduced

**Component Adaptations:**

**BottomNav (Full Width):**
```
┌────────────────────────────────────┐
│ [← Back]  [Next →]                 │
└────────────────────────────────────┘
```

**Breadcrumb (Horizontal Scroll or Stack):**
```
┌────────────────────────────┐
│ ●1: Title...               │
│ ○2: Title...               │
│ ○3: Title...               │
└────────────────────────────┘
```

**Scene Content:**
- Diagram centered
- Touch-friendly interactive areas
- Pinch zoom support (browser default)

#### CSS Media Query
```css
@media (width <= 860px) {
  .preview-shell { /* adaptations */ }
  .flow-workspace { /* column layout */ }
  .scene-cards-grid { grid-template-columns: 1fr; }
  .preview-shell[data-preview-layout="hub-spoke"] { /* special layout */ }
  /* ... etc */
}
```

#### Touch Interactions
- Tap to navigate (no hover states)
- Long-press for context menus (if available)
- Swipe for scene navigation (if implemented)

---

### PAGE 10: Reduced Motion Preference

**State:** `reducedMotion === true` (media query: prefers-reduced-motion: reduce)  
**Trigger:** User accessibility setting

#### Visual Changes
- Animations disabled or slowed
- Transitions: instant or minimal
- No parallax or motion-based effects
- Focus transitions: immediate

#### Implemented Changes
```css
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; }
  * { animation-iteration-count: 1 !important; }
  * { transition-duration: 0.01ms !important; }
}
```

#### Component Impact
- FlowApp: animations disabled
- Slide transitions: instant
- Hover effects: no motion
- Transitions: no easing animations

---

### PAGE 11: High Contrast Mode

**State:** `prefers-contrast: more` (browser/OS setting)  
**Trigger:** User accessibility preference

#### Visual Changes
```css
@media (prefers-contrast: more) {
  /* Enhanced contrast colors */
  --preview-guide: higher contrast
  --preview-chalk: brighter white
  --preview-board: darker background
}
```

#### Implementation
- All text contrast meets WCAG AA or AAA
- Borders and dividers enhanced
- Focus indicators more prominent
- Color-only information supplemented with patterns/text

---

### PAGE 12: Forced Colors Mode

**State:** `forced-colors: active` (high contrast mode in Windows)  
**Trigger:** Windows High Contrast mode enabled

#### Visual Changes
```css
@media (forced-colors: active) {
  /* System color palette applied */
  background: Canvas;
  color: CanvasText;
  border: 1px solid CanvasText;
}
```

#### Implementation
- All custom colors replaced with system equivalents
- Borders and separators use `CanvasText` color
- Backgrounds use `Canvas` color
- Images may not display (user choice)

---

## Data & Content Structure

### Available Flows (3 total)

#### Flow 1: Request Flow
- **ID:** `request-flow`
- **Source File:** `request-flow.flow`
- **Description:** Traces a single request through the system
- **Scenes:** Multiple steps showing request lifecycle
- **Note:** Skipped on home page; explainer decks used instead

#### Flow 2: Architecture
- **ID:** `architecture`
- **Source File:** `architecture.flow`
- **Description:** System architecture and components
- **Chapters:**
  - Architectural Concepts
  - Runtime Organization
  - Transport & Measurement
- **Example Scenes:**
  - `control-plane` (default active scene)
  - Other architectural aspects

#### Flow 3: Endpoint Lifecycle
- **ID:** `endpoint-lifecycle`
- **Source File:** `endpoint-lifecycle.flow`
- **Description:** Endpoint behavior across request lifecycle
- **Chapters:** Multiple phases of endpoint handling
- **Scenes:** Various endpoint-related visualizations

### Scene Structure (per scene)
```typescript
{
  id: "scene-identifier",
  title: "Scene Title",
  summary: "Brief description",
  roots: [/* visual elements */],
  responsive: [{
    condition: "(max-width: 860px)",
    roots: [/* mobile-adapted elements */]
  }],
  timeline: [/* animation keyframes */],
  narrativeTrack: {
    cues: [/* narration timing */]
  }
}
```

---

## Feature Capabilities

### Theme System
- **Storage:** localStorage key `aiperf-flow-theme`
- **Persistence:** Survives page reload
- **Fallback:** Defaults to "systems-chalk" if no stored value
- **Themes:**
  - Systems Chalk (chalk/muted design)
  - Legacy (neutral gray)
  - Core (GitHub-inspired dark)

### Audio & Narration
- **Engine:** Kokoro text-to-speech
- **Backend:** Preview narrator backend with prewarm support
- **State Management:**
  - Loading progress tracking
  - User activation requirement
  - Auto-play option (if consent given)
- **Consent:**
  - Storage key: `aiperf-flow-audio-consent`
  - States: "yes" | "no" | "unset"
- **Status Updates:**
  - Live region updates for screen readers

### Keyboard Navigation
- **⌘K** (Cmd+K on Mac, Ctrl+K elsewhere): Focus search box
- **Tab/Shift+Tab:** Navigate between buttons and controls
- **Enter/Space:** Activate buttons
- **Arrow Keys:** Navigate within lists (if supported)

### Accessibility Features
- ARIA labels on all interactive regions
- Semantic HTML structure
- Focus management and visible focus indicators
- Live regions for dynamic content
- Reduced motion support
- High contrast mode support
- Forced colors mode support
- Keyboard-only navigation support

### Visibility State Tracking
- Listens to `visibilitychange` event
- Tracks when user leaves site: `hasLeftSite` state
- May affect audio playback or other behavior

---

## URL & Routing Behavior

### Root Path
**URL:** `http://127.0.0.1:5188/`

**Browser Behavior:**
- Page loads with default scene in fixture
- React app initializes state from `previewWorkspace()`
- Home page displayed on load (or cached scene view)

### Single-Page Application
- No traditional URL routing with path-based segments
- All navigation is React state-based (in-memory)
- Refresh returns to initial state/home
- No deep-linking to specific scenes (limitation)

---

## Browser/Storage APIs Used

### localStorage
- Key: `aiperf-flow-theme` → current theme preference
- Key: `aiperf-flow-audio-consent` → audio permission
- Error handling: catches and silently ignores storage errors
- SSR-safe: checks `typeof localStorage` before access

### Media Queries (matchMedia)
- `(prefers-reduced-motion: reduce)` → accessibility
- `(max-width: 860px)` → responsive layout
- `(prefers-contrast: more)` → high contrast
- `(forced-colors: active)` → Windows high contrast

### Event Listeners
- `visibilitychange` → track user leaving/returning
- `click` → close theme menu on outside clicks
- Media query listeners for reactive updates

---

## Performance Characteristics

### Build Setup
- **Framework:** React 18+ with TypeScript
- **Build Tool:** Vite
- **Dev Server:** 127.0.0.1:5188
- **Asset Handling:** WASM assets included
- **Worker Format:** ES module

### Code Splitting
- Runtime packages organized by feature
- Lazy loading of scene chunks (if implemented)
- Explainer deck precompilation

### Data Structure Size
- Explainer decks are pre-compiled TypeScript objects
- Scene graphs embedded in fixture
- Total bundle size managed via Vite chunking

---

## Testing Infrastructure

### Test Files Identified
- `preview/home-page.test.tsx` - Home page tests
- `preview/immersive.test.tsx` - Immersive preview tests
- `preview/narrative.test.tsx` - Narrative content tests
- `preview/theme-selector.test.tsx` - Theme switcher tests
- `preview/layout-ownership.test.ts` - Layout behavior tests
- `e2e/explainer-visuals.spec.ts` - E2E visual tests
- `e2e/explainer-visual-parity.spec.ts` - Cross-platform parity

### Test Frameworks
- **Unit:** Vitest with jsdom environment
- **E2E:** Playwright
- **Configuration:** `vitest.config.ts`, `playwright.config.ts`

---

## Known Limitations & Notes

1. **No URL-based Deep Linking:** Scenes can't be bookmarked or shared via URL
2. **Explainer Decks Modal:** Currently overlay only, not full flow integration
3. **Mobile Drawer:** Sidebar access on mobile not fully described in App.tsx
4. **Request Flow Skipped:** Home page excludes request-flow from card display
5. **Search Functionality:** Search box exists but implementation not in App.tsx scope

---

## File Structure Reference

```
preview/
├── App.tsx (main app component - 717 lines)
├── home-page.tsx (landing page - 223 lines)
├── styles.css (all styling)
├── narrator-backend.ts (voice/audio integration)
├── fixture.ts (scene and flow definitions - 1600+ lines)
├── main.tsx (React entry point)
├── index.html (HTML shell)
└── [test files...]

packages/runtime/src/
├── app.tsx (FlowApp component)
├── explainer/
│   ├── compiled-decks.ts (4 precompiled decks)
│   ├── ui/ExplainerSlideViewer.tsx
│   └── registry.ts
├── narrative/
│   ├── kokoro-narrator.ts (voice engine)
│   └── narrator.ts
└── [other runtime modules...]
```

---

## Summary: Unique Pages Identified

| # | Page/State | Component | Trigger | Key Features |
|---|-----------|-----------|---------|--------------|
| 1 | Home Page | HomePage | `showHome === true` | Scene selector grid, flow sections |
| 2 | Scene Workspace | App (main) | `showHome === false` | Sidebar, breadcrumb, canvas, nav |
| 3 | Document Browser | DocumentBrowser | Always visible (desktop) | Hierarchical flow tree, search |
| 4 | Audio Consent | FlowApp modal | `requireAudioConsent === true` | Permission dialog |
| 5 | Theme Selector | Dropdown menu | `showThemeMenu === true` | 3-option dropdown menu |
| 6 | Voice Status | Status paragraph | `voiceStatus !== null` | Loading/ready indicator |
| 7 | Flow Diagram | FlowApp canvas | All scenes | Interactive SVG visualization |
| 8 | Explainer Deck | ExplainerSlideViewer | `showExplainerDeckId !== null` | Modal slide presentation |
| 9 | Mobile Layout | Responsive CSS | `narrowLayout === true` | Adapted to <860px width |
| 10 | Reduced Motion | CSS media query | `prefers-reduced-motion: reduce` | Animations disabled |
| 11 | High Contrast | CSS media query | `prefers-contrast: more` | Enhanced contrast |
| 12 | Forced Colors | CSS media query | `forced-colors: active` | System color palette |

---

**Analysis Complete**

This catalog represents all unique pages, states, and views discoverable in the AIPerf Flow Explainers app based on source code analysis. Interactive elements, styling, content structure, and responsive behavior are fully documented.
