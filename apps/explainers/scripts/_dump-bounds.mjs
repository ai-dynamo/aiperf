/* SPDX-License-Identifier: Apache-2.0 */
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

const wanted = process.argv.slice(2);
const { stdout } = await execFileAsync(
  "npx",
  ["vite-node", resolve(ROOT, "scripts/compile-decks.ts")],
  { cwd: ROOT, maxBuffer: 128 * 1024 * 1024 },
);
const bundle = JSON.parse(stdout);
const scenes = bundle.resolvedScenes.filter((s) => s.deckId === "sdk-generic-catalog");
for (const s of scenes) {
  const nodes = s.snapshot.nodes.filter((n) =>
    wanted.some((w) => n.id.includes(w)),
  );
  if (nodes.length === 0) continue;
  console.log(`--- scene ${s.sceneId ?? s.slideId ?? "?"} ---`);
  for (const n of nodes) {
    const b = n.bounds;
    console.log(
      `${n.id}\t${n.capability}\tx=${b.x} y=${b.y} w=${b.width} h=${b.height} (r=${b.x + b.width}, b=${b.y + b.height})`,
    );
  }
}
