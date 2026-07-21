# Stat design note

Prop shape adapts the reference Cursor Canvas SDK's `StatProps` (`value`, `label`,
`tone`) to this app's conventions: `value: string | number` (the SDK's `ReactNode`
is narrowed since this is a plain-text metric tile, not an arbitrary-content slot),
plus an added `trend?: string` for a delta chip (e.g. `"+8.2%"`), and `tone`'s
semantics move from "color of the value" to "color of the trend text" since the big
value itself should stay `ink-primary` for legibility and only the trend needs a
directional color. The SDK's `success/danger/warning/info` tone set collapses to
`"neutral" | "positive" | "negative"`, matching how a KPI trend is actually judged
(up/down/flat), not the four-way severity scale a `Callout` needs. `style` is
dropped in favor of `className` + `clsx`, per the rest of this component set.

`tone` colors reuse existing tokens rather than inventing a tone-to-category map:
`positive -> accent-primary` (NVIDIA green, the app's existing "good" accent),
`negative -> category-red`, `neutral -> ink-secondary`. No new token axis needed.

Visually: an uppercase, tracking-wide, small `ink-secondary` label above a large
bold `ink-primary` value, with the optional trend rendered smaller/secondary beside
the value in the tone color. Flat box, `rounded-none`, no shadow, `surface-elevated`
background with a `stroke-secondary` border, matching `Card`/`Header`.
