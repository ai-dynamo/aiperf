# P0 flagship IR fixtures

Hand-authored Flow IR v1 documents that prove the three P0 wrapper scenes before
`.flow` lowering lands. Each file is strict-schema valid: `irVersion`, source
maps, accessibility, and fallbacks on every scene and node.

| Fixture | Capability | Proof |
| --- | --- | --- |
| [`token-span-morph.ir.json`](./token-span-morph.ir.json) | `core.span-map` (+ `core.glyph-run` child) | Café 🚀 grapheme-to-token morphs |
| [`prompt-segment-composer.ir.json`](./prompt-segment-composer.ir.json) | `core.segment-strip` | Seven-segment prompt strip, layout seed 42 |
| [`request-lifecycle-waterfall.ir.json`](./request-lifecycle-waterfall.ir.json) | `viz.waterfall` | Arrival → admission → connect → first-token lanes |

## Validate with `safeParseFlowIr`

Build the schema package once, then parse each fixture from the `apps/aiperf-flow`
workspace root:

```bash
cd apps/aiperf-flow
npm run build -w @aiperf/flow-schema
node --input-type=module -e "
import { readFileSync } from 'node:fs';
import { safeParseFlowIr } from '@aiperf/flow-schema';

for (const path of [
  'examples/p0/token-span-morph.ir.json',
  'examples/p0/prompt-segment-composer.ir.json',
  'examples/p0/request-lifecycle-waterfall.ir.json',
]) {
  const parsed = safeParseFlowIr(JSON.parse(readFileSync(path, 'utf8')));
  if (!parsed.ok) {
    console.error(path, parsed.diagnostics);
    process.exitCode = 1;
    continue;
  }
  console.log(path, 'ok', parsed.value.id);
}
"
```

Expected output:

```text
examples/p0/token-span-morph.ir.json ok token-span-morph
examples/p0/prompt-segment-composer.ir.json ok prompt-segment-composer
examples/p0/request-lifecycle-waterfall.ir.json ok request-lifecycle-waterfall
```

## Automated test

```bash
npm test -w @aiperf/flow-schema
```

The `flagship-ir-fixtures` suite reads these JSON files and asserts
`safeParseFlowIr(...).ok === true` for each.
