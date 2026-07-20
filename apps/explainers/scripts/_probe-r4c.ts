import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { compileExplainerSource } from "../src/flow/compiler/compile-explainer.ts";
import { FOUNDATION_CAPABILITIES } from "../src/flow/schema/index.ts";
import { resolveScene } from "../src/core/diagram/resolution/resolve-scene.ts";

function walk(nodes: any[], out: any[] = [], parents = new Map<string, string | undefined>(), parentId?: string) {
  for (const n of nodes ?? []) {
    out.push(n);
    parents.set(n.id, parentId);
    walk(n.children ?? [], out, parents, n.id);
  }
  return { out, parents };
}

function compile(rel: string) {
  const sourcePath = resolve(rel);
  const compiled = compileExplainerSource({
    source: readFileSync(sourcePath, "utf8"),
    sourceName: sourcePath,
    capabilities: FOUNDATION_CAPABILITIES,
    strict: true,
    strictSdkAuthoring: true,
  });
  if (!compiled.ok) {
    console.error("fail", rel, compiled.diagnostics?.slice?.(0, 5));
    process.exit(1);
  }
  return compiled.value;
}

const cellular = compile("decks-flow/cellular-algorithms.flow");
console.log("slide ids", cellular.slides.map((s: any) => s.id));
const veloSlide = cellular.slides.find((s: any) => s.id === "fail-closed-before-cells-launch");
const { out: nodes, parents } = walk(veloSlide?.render?.scene?.roots ?? []);

for (const id of ["velo", "st-velo", "badge-velo"]) {
  const n = nodes.find((x) => x.id === id);
  console.log(id, {
    cap: n?.capabilityId ?? n?.capability,
    parentId: parents.get(id),
    geom: n?.geometry,
  });
}

const resolved = resolveScene(veloSlide.render.scene);
console.log(
  "velo slide overlaps",
  resolved.diagnostics.filter((d: any) => d.code === "SCENE_ABSOLUTE_SIBLING_OVERLAP").map((d: any) => d.nodeIds),
);
console.log(
  "velo world geometry",
  ["velo", "st-velo", "badge-velo"].map((id) => [id, resolved.worldGeometryById?.get?.(id)]),
);
