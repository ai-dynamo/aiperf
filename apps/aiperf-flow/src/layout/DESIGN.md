# Layout primitives

`Stack`, `Row`, and `Grid` are thin `<div>` wrappers that mirror the prop shapes of the
`cursor/canvas` UI primitives (`~/.cursor/skills-cursor/canvas/sdk/ui-primitives.d.ts`), but render
with Tailwind utility classes instead of that SDK's inline-style approach.

- **Stack**: `{ children, gap?, className? }`. Renders `flex flex-col`. No alignment props — callers
  compose alignment via `className`.
- **Row**: `{ children, gap?, align?, justify?, wrap?, className? }`. Renders `flex flex-row`.
  `align`/`justify` map to static `items-*`/`justify-*` Tailwind classes; `wrap` toggles
  `flex-wrap` vs the default no-wrap (omitted, matching `flex`'s default).
- **Grid**: `{ children, columns, gap?, align?, className? }`. Renders `grid`. `columns` is the one
  prop that can't be pure Tailwind: a numeric `columns` (1-12) is resolved through a static
  `grid-cols-N` lookup table so the literal class strings are visible to Tailwind's compiler (you
  cannot interpolate `` `grid-cols-${n}` `` at runtime — Tailwind only picks up classes it can see as
  whole strings in source). A string `columns` (e.g. `"1fr 2fr"`) falls through to an inline
  `style={{ gridTemplateColumns: columns }}`, since arbitrary CSS Grid template strings have no
  finite Tailwind class enumeration.

Across all three, `gap` is a pixel number applied via inline `style={{ gap }}` rather than a
Tailwind spacing class, again because gap is a caller-supplied runtime number, not one of a fixed
enum of values. `className` is merged with `clsx(rootClasses, className)`, matching the
`Header.tsx` pattern of the node components: fixed classes first, caller-supplied `className` last
so it wins on conflicting utilities.
