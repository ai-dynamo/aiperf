import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { compileExplainerSource } from "../src/flow/compiler/compile-explainer.ts";
import { FOUNDATION_CAPABILITIES } from "../src/flow/schema/index.ts";
import { resolveScene } from "../src/core/diagram/resolution/resolve-scene.ts";

function walk(nodes, out = []) {
  for (const n of nodes ?? []) {
    out.push(n);
    walk(n.children ?? [], out);
  }
  return out;
}

function compile(rel) {
  const sourcePath = resolve(rel);
  const compiled = compileExplainerSource({
    source: readFileSync(sourcePath, "utf8"),
    sourceName: sourcePath,
    capabilities: FOUNDATION_CAPABILITIES,
    strict: true,
    strictSdkAuthoring: true,
  });
  if (!compiled.ok) {
    console.error("fail", rel, compiled.diagnostics?.slice?.(0, 2));
    process.exit(1);
  }
  return compiled.value;
}

const catalog = compile("decks-flow/sdk-generic-catalog.flow");
const slide = catalog.slides[42];
const nodes = walk(slide?.render?.scene?.roots ?? []);
console.log(
  "progress track/value origins",
  JSON.stringify(
    nodes
      .filter(
        (n) =>
          n.id === "progress-hero__track" || n.id === "progress-hero__value",
      )
      .map((n) => ({
        id: n.id,
        cap: n.capabilityId ?? n.capability,
        sdkOrigin: n.sdkOrigin,
        geom: n.geometry,
      })),
    null,
    2,
  ),
);

const cellular = compile("decks-flow/cellular-algorithms.flow");
const cslide = cellular.slides[1];
const cnodes = walk(cslide?.render?.scene?.roots ?? []);
console.log(
  "chip/panel",
  JSON.stringify(
    cnodes
      .filter((n) => n.id === "algo-built" || n.id === "st-built")
      .map((n) => ({
        id: n.id,
        cap: n.capabilityId ?? n.capability,
        sdkOrigin: n.sdkOrigin,
        geom: n.geometry,
      })),
    null,
    2,
  ),
);

const segment = compile("decks-flow/segment-pools.flow");
const snodes = walk(segment.slides[0]?.render?.scene?.roots ?? []);
const bracket = snodes.find((n) => n.id === "dispatch-bracket");
console.log(
  "bracket",
  JSON.stringify(
    {
      id: bracket?.id,
      cap: bracket?.capabilityId ?? bracket?.capability,
      kind: bracket?.kind,
      sdkOrigin: bracket?.sdkOrigin,
      geom: bracket?.geometry,
      inConnectors: resolveScene(segment.slides[0].render.scene).connectorsById.has(
        "dispatch-bracket",
      ),
    },
    null,
    2,
  ),
);

console.log(
  "progress overlaps",
  resolveScene(slide.render.scene)
    .diagnostics.filter((d) => d.code === "SCENE_ABSOLUTE_SIBLING_OVERLAP")
    .map((d) => d.nodeIds),
);
