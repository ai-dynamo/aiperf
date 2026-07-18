import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import type { DeckDefinition } from "../../core/types";
import { SLIDES } from "./content";
import { MentalModel } from "./MentalModel";
import { CSS } from "./styles";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">velo transport</Pill>}>Live evidence files</CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code>rust/runtime/src/cellular/transport/mod.rs</Code>
          <Code>rust/runtime/src/cellular/transport/velo_transport.rs</Code>
          <Code>rust/runtime/src/cellular/transport/phaser_velo.rs</Code>
          <Code>rust/runtime/src/cellular/transport/dataset_velo.rs</Code>
          <Code>rust/runtime/src/cellular/heartbeat.rs</Code>
          <Code>rust/runtime/src/cellular/shard.rs</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const veloDeepDiveDeck: DeckDefinition = {
  id: "velo-deep-dive",
  route: "/velo-deep-dive",
  storagePrefix: "velo-deep-dive-explainer",
  classPrefix: "deck-velo-deep",
  eyebrowLabel: "VELO DEEP DIVE",
  startGateTitle: "Velo deep dive",
  hub: {
    highlight: "Velo deep dive",
    title: "cellular transport mechanisms",
    description:
      "Ten mechanisms behind AIPerf's cellular Velo plane: connection resolve, registration reply, synchronized START, MessagePack, heartbeat, partition shipping, merge, phaser replay, dataset floodgate, and the aggregator tree. Deeper than the beginner SLURM & Velo tour.",
  },
  slides: SLIDES,
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
