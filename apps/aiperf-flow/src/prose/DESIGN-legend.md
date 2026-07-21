# Legend / Swatch design note

Prop shape is adapted from the reference Cursor Canvas SDK's `SwatchProps` (`color:
Color`, optional `style`), but ported to this app's conventions: `color` is typed as
the existing `CategoryRole` from `theme/tokens.ts` (not a separate `Color` union), and
styling goes through Tailwind classes via `categoryClassName`/`clsx`, not inline
`style` objects. `Swatch` renders a small filled square (`h-3 w-3`, `rounded-none` per
the app's flat visual language) using `bg-category-{role}`; since `tokens.ts` only
exposes a `text-category-*` helper, `Swatch` builds its own `bg-category-${color}`
class string directly rather than adding a new token helper for one caller.

`Legend` is a horizontal row of `{ color: CategoryRole; label: string }` entries, each
rendered as a `Swatch` next to its label text in `ink-secondary`. Row layout reuses
`layout/Row.tsx` if present at build time (checked at implementation time — it did not
exist yet), so `Legend` falls back to an inline `flex flex-wrap gap-4` wrapper rather
than introducing a second layout primitive.
