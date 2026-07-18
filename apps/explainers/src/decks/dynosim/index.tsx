import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import { SLIDES } from "./content";
import { CSS } from "./styles";
import { MentalModel } from "./MentalModel";
import type { DeckDefinition } from "../../core/types";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">config v2</Pill>}>Try it in config</CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code style={{ fontSize: 13 }}>transport.type: dynosim_offline</Code>
          <Code style={{ fontSize: 13 }}>transport.type: dynosim_online</Code>
          <Code style={{ fontSize: 13 }}>cargo build -p aiperf-cli --features dynosim</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const dynosimDeck: DeckDefinition = {
  id: "dynosim",
  route: "/dynosim",
  storagePrefix: "dynosim-explainer",
  classPrefix: "deck-dynosim",
  eyebrowLabel: "DYNOSIM",
  startGateTitle: "Dynosim co-simulation walkthrough",
  hub: {
    highlight: "Dynosim",
    title: "offline & online replay",
    description:
      "Concise tour of dynosim transports, SimClock vs RealClock, the sim pump, token flow through RequestObserver, and metric parity with HTTP runs.",
  },
  slides: SLIDES,
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
