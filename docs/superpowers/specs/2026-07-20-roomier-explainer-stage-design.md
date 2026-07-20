# Roomier Explainer Stage Design

## Goal

Give explainer diagrams more usable stage area, reduce visual crowding, and prevent diagram or footer content from being cut off across desktop, short laptop, and mobile viewports.

## Design

Keep the existing full-viewport cinematic shell and two-row stage structure. Make the footer denser so the diagram row receives more of the viewport:

- Reduce footer padding and the gap between subtitles and slide copy.
- Tighten title, lede, and eyebrow spacing without hiding any content.
- Keep the lede visible and allow it to wrap to its full height.
- Add short-viewport rules that compact footer typography and spacing further.

Fit the diagram within the space left by the footer:

- Preserve safe edge insets around the diagram.
- Constrain rendered SVGs to the available width and height while preserving their aspect ratio.
- Keep diagram-internal overflow visible so labels and connector decorations at SVG edges are not clipped.
- Retain the shell's full-screen presentation behavior and avoid introducing page scrolling during normal presentation.

Reduce all text rendered inside scene SVGs to 90% of its authored or default size. Apply the scale centrally in the scene renderer so SDK-generated and directly authored labels behave consistently. Scale automatic line height with the font size, and leave shell titles, ledes, subtitles, controls, and other HTML text unchanged.

## Responsive Behavior

Desktop and presentation modes use the roomier default layout. Short viewports reduce vertical stage and footer padding before reducing typography. Mobile viewports use smaller safe insets and compact title and lede sizing, but continue to show the complete lede and diagram.

## Scope

This is a shared shell and scene-renderer presentation change. It does not add per-deck zoom controls or alter scene geometry, narration, playback, or deck content.

## Verification

- Run the explainer shell tests.
- Run the explainer package type check or build.
- Run scene-renderer tests that verify authored and default SVG font sizes use the shared 90% scale.
- Inspect representative desktop, short laptop, and mobile viewport sizes.
- Confirm the diagram, subtitles, title, and full wrapped lede remain visible without overlap or cutoff.
