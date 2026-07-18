# Explainers App - Quick Reference Guide

**Generated:** 2026-07-18  
**App Location:** http://127.0.0.1:5188  
**Source:** `/home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow`

---

## 12 Unique Pages/States Identified

### Visual Map

```
┌─────────────────────────────────────────────────────────────┐
│ ENTRY POINT                                                 │
│ http://127.0.0.1:5188/                                      │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─ STATE: showHome === true
         │  └─> PAGE 1: HOME PAGE - SCENE SELECTOR
         │      Layout: Full-screen dark theme grid
         │      Content: Scene cards organized by flow
         │      Actions: Click card → navigate to scene
         │
         └─ STATE: showHome === false
            └─> PAGE 2: SCENE WORKSPACE (Main Application)
                Layout: 3-column (sidebar | breadcrumb | canvas | nav)
                Content: Interactive flow diagram + narrative
                Components:
                  ├─ PAGE 3: DOCUMENT BROWSER SIDEBAR
                  │   Left navigation tree, scene search
                  │
                  ├─ Breadcrumb Step Navigation
                  │   Visual indicator of current scene position
                  │
                  ├─ PAGE 7: FLOW DIAGRAM CANVAS
                  │   Interactive SVG visualization with animations
                  │   Hidden when showing:
                  │   └─ PAGE 8: EXPLAINER DECK MODAL
                  │       Full-screen slide presentation overlay
                  │
                  └─ BottomNav (Next/Previous)
                      Linear scene navigation

                Conditional Overlays:
                ├─ PAGE 4: AUDIO CONSENT DIALOG
                │   Shows when: audioConsent === "unset"
                │   Actions: Enable/disable narration
                │
                └─ PAGE 5: THEME SELECTOR DROPDOWN
                    Shows when: showThemeMenu === true
                    Options: Systems Chalk | Legacy | Core

                Status Indicators:
                ├─ PAGE 6: VOICE STATUS
                │   Loading progress or activation prompt
                │
                └─ PAGE 12: FOOTER LEGEND
                    Only on "request-investigation" scene
                    Semantic key for diagram markings

            Responsive Adaptations:
            ├─ PAGE 9: MOBILE LAYOUT (<860px)
            │   Single column, hidden sidebar, full-width buttons
            │
            ├─ PAGE 10: REDUCED MOTION
            │   Animations disabled per prefers-reduced-motion
            │
            ├─ PAGE 11: HIGH CONTRAST
            │   Enhanced contrast per prefers-contrast: more
            │
            └─ PAGE 12: FORCED COLORS
                System palette per forced-colors: active
```

---

## State Transition Diagram

```
Home Page (showHome=true)
        ↓
   Click card → selectScene(flowId, sceneId)
        ↓
Scene Workspace (showHome=false)
        ↓
   ├─ Click theme button → showThemeMenu=true
   │        ↓
   │   PAGE 5: Theme Dropdown
   │        ↓
   │   Click outside → showThemeMenu=false
   │
   ├─ Narrative plays → audioConsent="unset"
   │        ↓
   │   PAGE 4: Audio Consent Dialog
   │        ↓
   │   Choose option → audioConsent="yes"|"no"
   │
   ├─ Click explainer link → showExplainerDeckId="deck-id"
   │        ↓
   │   PAGE 8: Explainer Modal
   │        ↓
   │   Click back → showExplainerDeckId=null
   │
   ├─ Window resize <860px → narrowLayout=true
   │        ↓
   │   PAGE 9: Mobile Layout
   │
   └─ Click home button → showHome=true
            ↓
        Home Page
```

---

## 12 Unique Pages - Quick List

| # | Name | Trigger | Key Component | Visual Style | Navigation |
|---|------|---------|----------------|--------------|------------|
| 1 | **Home Page** | `showHome===true` | `<HomePage/>` | Full-screen grid, dark theme | Click cards |
| 2 | **Workspace** | `showHome===false` | `<App/>` main | Sidebar+canvas+nav layout | Buttons, breadcrumb |
| 3 | **Document Browser** | Desktop mode | `<DocumentBrowser/>` | Left sidebar tree | Details/summary controls |
| 4 | **Audio Consent** | First narration play | Modal (FlowApp) | Center dialog, overlay | Buttons (yes/no) |
| 5 | **Theme Selector** | `showThemeMenu===true` | Dropdown menu | Absolute dropdown | Click option |
| 6 | **Voice Status** | `voiceStatus !== null` | Status `<p>` | Topbar indicator | Live region |
| 7 | **Flow Diagram** | All scenes | `<FlowApp/>` canvas | SVG interactive | Click/drag/hover |
| 8 | **Explainer Deck** | `showExplainerDeckId!==null` | `<ExplainerSlideViewer/>` | Modal, white bg | Slide nav buttons |
| 9 | **Mobile Layout** | `narrowLayout===true` | CSS media query | Single column | Touch-friendly |
| 10 | **Reduced Motion** | User accessibility setting | CSS media query | No animations | N/A (automatic) |
| 11 | **High Contrast** | `prefers-contrast:more` | CSS media query | Enhanced colors | N/A (automatic) |
| 12 | **Forced Colors** | `forced-colors:active` | CSS media query | System palette | N/A (automatic) |

---

## Data Flows

### Available Content

**3 Primary Flows:**
1. **Request Flow** (`request-flow`)
   - Request lifecycle visualization
   - Skipped on home (explainer decks instead)

2. **Architecture** (`architecture`)
   - System components and relationships
   - Default flow on app load
   - Chapter structure: Architectural Concepts, Runtime, Transport

3. **Endpoint Lifecycle** (`endpoint-lifecycle`)
   - Endpoint behavior through request phases
   - Per-endpoint state management

**4 Explainer Decks (compiled to TypeScript):**
1. `rust-architecture` - Crate organization, CLI, runtime
2. `slurm-velo` - Distributed cell execution
3. `dynosim` - Simulation mode
4. `aiperf-flow-system` - Flow IR and execution

**Scenes per Flow:**
- Each flow contains 3-5+ scenes
- Each scene has:
  - ID, title, summary
  - Visual roots (SVG elements)
  - Optional responsive variants (<860px)
  - Timeline (animation keyframes)
  - Narrative track (voice timing)

---

## Theme Variables

### Systems Chalk (Default)
```
Board:    #1a1a1d  (dark background)
Panel:    #24282b  (card/panel bg)
Chalk:    #e8e3d9  (primary text)
Muted:    #8b8680  (secondary text)
Signal:   #3fb950  (accent - green)
```

### Legacy
```
Board:    #1a1a1a  (very dark)
Panel:    #222222  (dark gray)
Chalk:    #e8e8e8  (light gray text)
Muted:    #999999  (medium gray)
```

### Core (GitHub Dark)
```
Board:    #0d1117  (darkest)
Panel:    #161b22  (dark blue-gray)
Chalk:    #f0f6fc  (bright white)
Muted:    #8b949e  (muted blue-gray)
```

---

## Interactive Elements Summary

### Buttons & Controls
- Scene cards (home page) → navigate to scene
- Breadcrumb steps → jump to scene
- Back/Next nav buttons → linear navigation
- Theme selector → choose theme
- Theme cycle button (⟳) → rotate themes
- Home button (top-left) → return to home
- Explainer deck back → close modal

### Forms & Inputs
- Search box → filter flows/scenes (Cmd/Ctrl+K)
- Audio consent dialog → enable/disable narration

### Dropdowns & Menus
- Flow tree (details/summary) → expand/collapse flows
- Theme dropdown → 3-option menu
- Scene chapters (nested lists) → collapsible sections

### Keyboard Shortcuts
- **Cmd/Ctrl+K:** Focus search box in sidebar
- **Tab/Shift+Tab:** Navigate buttons
- **Enter/Space:** Activate button
- **Arrow keys:** Navigate lists (browser default)

---

## Storage & Persistence

### localStorage Keys
- `aiperf-flow-theme`: "systems-chalk" | "legacy" | "core"
- `aiperf-flow-audio-consent`: "yes" | "no" | "unset"

### Browser Events
- `visibilitychange` → detect when user leaves/returns
- Media query listeners:
  - `(prefers-reduced-motion: reduce)`
  - `(max-width: 860px)`
  - `(prefers-contrast: more)`
  - `(forced-colors: active)`

---

## Accessibility Features

✓ ARIA labels on all regions  
✓ Semantic HTML structure  
✓ Keyboard navigation (Tab, Enter, Space)  
✓ Focus management with visible indicators  
✓ Live regions for dynamic updates  
✓ Reduced motion support  
✓ High contrast mode support  
✓ Forced colors (Windows High Contrast)  
✓ Screen reader compatible  
✓ SVG alt text from diagram structure  

---

## Testing & Development

### Run App
```bash
npm run dev
# Server at http://127.0.0.1:5188
```

### Run Tests
```bash
npm run flow:test        # All tests
npm run preview:test     # Preview-only tests
npm run e2e             # Playwright E2E tests
```

### Files Modified in Session
- Created: `EXPLAINERS_APP_CATALOG.md` (1016 lines - comprehensive)
- Created: `CATALOG_QUICK_REFERENCE.md` (this file)
- Created: `catalog-app-pages.mjs` (Playwright automation script)

---

## Known Limitations

1. ❌ No URL-based deep linking (state-based only)
2. ❌ Explainer decks are modal-only (not integrated in flow)
3. ❌ Mobile drawer not fully implemented in App.tsx
4. ❌ Request flow hidden from home page
5. ⚠️  Search box exists but filtering not in App.tsx scope

---

## File Locations

**Key Source Files:**
- App main: `preview/App.tsx` (717 lines)
- Home page: `preview/home-page.tsx` (223 lines)
- Styles: `preview/styles.css`
- Fixtures: `preview/fixture.ts` (1600+ lines)
- Runtime: `packages/runtime/src/`

**Test Files:**
- Unit: `preview/*.test.tsx`
- E2E: `e2e/*.spec.ts`

**Configuration:**
- Vite: `vite.config.ts`
- Vitest: `vitest.config.ts`
- Playwright: `playwright.config.ts`

---

## Summary

This explainers app presents **12 distinct pages/states** covering:
- Landing & navigation
- Content display & interaction
- Theme/audio preferences
- Responsive layouts
- Accessibility modes

All pages/states are thoroughly documented in the companion `EXPLAINERS_APP_CATALOG.md` file with layouts, components, styling, and interactivity details.

---
