# Table design note

Prop shape is data-driven rather than mirroring the reference Cursor Canvas SDK's
positional `headers: ReactNode[]` / `rows: ReactNode[][]` arrays. Real decks reference
columns by name repeatedly, so `columns: Array<{ key; label; align? }>` plus
`rows: Array<Record<string, ReactNode> & { tone? }>` avoids index-matching bugs between
headers and row-array columns. Per-column `align` (`"start" | "center" | "end"`, logical
rather than the reference's `"left" | "right"`) drives both header and body cell text
alignment. Per-row `tone` (`"neutral" | "success" | "warning" | "danger"`, dropping the
reference's `"info"` since this app has no blue-as-info convention) replaces the
reference's marker-dot with a subtle full-row background tint, mapped onto existing
`CategoryRole` values (`success -> green`, `warning -> yellow`, `danger -> red`) via
`bg-category-<x>/10`, matching how `Callout.tsx` reuses `CategoryRole` instead of adding a
new tone token axis. No `framed`/`striped`/`stickyHeader`/`emptyMessage` — kept minimal
for a first cut; callers wrap in `Panel`/`Card` for framing per this app's composition
convention.

Renders a real `<table>` with `<thead>`/`<tbody>`, thin `stroke-secondary` row dividers,
`rounded-none`, no shadows, consistent with the rest of the flat/boxy component set.
