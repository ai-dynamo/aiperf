# Callout design note

Prop shape mirrors the reference Cursor Canvas SDK's `CalloutProps` (tone, title,
children) but drops `icon`/`style` — this app has no icon primitive yet and styling
goes through Tailwind classes + `clsx`, not inline `style`, per the rest of the
component set (see `Panel.tsx`). `tone` defaults to `"info"`.

There is no "tone" concept in `theme/tokens.ts` yet, only `CategoryRole`. Rather than
add a new token axis, `Callout` maps its four tones onto existing `CategoryRole`
values via a small local `toneCategory` record: `info -> blue`, `warning -> yellow`,
`danger -> red`, `success -> green`. This reuses `categoryClassName` for the accent
color (left border + title text) instead of inventing a parallel tone system.

Visually: a flat bordered box (`rounded-none`, no shadow, no gradient) with a thicker
left border in the tone color as the accent bar, default `stroke-secondary` on the
remaining three sides, `surface-elevated` background, optional bold title line in the
tone color, then body content in `ink-primary`.
