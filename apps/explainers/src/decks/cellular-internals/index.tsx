import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import type { DeckDefinition } from "../../core/types";
import { SLIDES } from "./content";
import { MentalModel } from "./MentalModel";
import { CSS } from "./styles";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">cellular runtime</Pill>}>Live evidence files</CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code>rust/runtime/src/engine/cellular_controller.rs</Code>
          <Code>rust/runtime/src/engine/cellular_cell.rs</Code>
          <Code>rust/runtime/src/engine/sharded_scheduled.rs</Code>
          <Code>rust/runtime/src/cellular/partition.rs</Code>
          <Code>rust/runtime/src/cellular/shard.rs</Code>
          <Code>rust/runtime/src/engine/cellular_aggregator.rs</Code>
          <Code>docs/canvases/cellular-architecture.canvas.tsx</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const cellularInternalsDeck: DeckDefinition = {
  id: "cellular-internals",
  route: "/cellular-internals",
  storagePrefix: "cellular-internals-explainer",
  classPrefix: "deck-cellular",
  eyebrowLabel: "CELLULAR INTERNALS",
  startGateTitle: "Cellular internals",
  hub: {
    highlight: "Cellular internals",
    title: "launch · distribute · execute · reduce · scale",
    description:
      "Twenty-slide walkthrough of AIPerf's cellular runtime in five chapters: promotion and validation, deterministic distribution, autonomous execution, retain/fold/sketch reduction, and flat-or-tree merge to one report. Grounded in rust/runtime source, deeper than the SLURM & Velo tour.",
  },
  slides: SLIDES,
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
