import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import { SLIDES } from "./content";
import { CSS } from "./styles";
import { MentalModel } from "./MentalModel";
import type { DeckDefinition } from "../../core/types";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">follow the code</Pill>}>Suggested live files</CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code>rust/cli/src/main.rs</Code>
          <Code>rust/cli/src/profile.rs</Code>
          <Code>rust/runtime/src/engine/coordinator.rs</Code>
          <Code>rust/runtime/src/engine/execute.rs</Code>
          <Code>rust/loadgen-core/src/sink.rs</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const rustArchitectureDeck: DeckDefinition = {
  id: "rust-architecture",
  route: "/rust-architecture",
  storagePrefix: "rust-arch-explainer",
  classPrefix: "rust-arch",
  eyebrowLabel: "RUST ARCHITECTURE",
  startGateTitle: "Rust architecture walkthrough",
  hub: {
    highlight: "Rust architecture",
    title: "from scratch",
    description:
      "Narrated walkthrough of the native workspace, self-execution seam, registry bootstrap, request lifecycle, workers, cellular mode, and feature gates.",
  },
  slides: SLIDES,
  glossary: [],
  MentalModel: ({ slideIndex }) => <MentalModel slideIndex={slideIndex} slide={SLIDES[slideIndex]} />,
  css: CSS,
  FinalCard,
};
