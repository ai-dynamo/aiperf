import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import type { DeckDefinition } from "../../core/types";
import { SLIDES } from "./content";
import { MentalModel } from "./MentalModel";
import { CSS } from "./styles";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">dataset path</Pill>}>Live evidence files</CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code>rust/runtime/src/dataset/segment.rs</Code>
          <Code>rust/runtime/src/dataset/model.rs</Code>
          <Code>rust/runtime/src/dataset/compose.rs</Code>
          <Code>rust/runtime/src/body_plan.rs</Code>
          <Code>rust/runtime/src/graph/recorded/trie/</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const segmentPoolsDeck: DeckDefinition = {
  id: "segment-pools",
  route: "/segment-pools",
  storagePrefix: "segment-pools-explainer",
  classPrefix: "deck-segment-pools",
  eyebrowLabel: "SEGMENT POOLS",
  startGateTitle: "Segment pools & body plans",
  hub: {
    highlight: "Segment pools",
    title: "intern → freeze → splice",
    description:
      "How dataset rows become content-addressed handles and wire Bytes: SegmentPool, six payload domains, BodyPlan materialization, prefix chains, and Turn.body dispatch precedence.",
  },
  slides: SLIDES,
  glossary: [],
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
