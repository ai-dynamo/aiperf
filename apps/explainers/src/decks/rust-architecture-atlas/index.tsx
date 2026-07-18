import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import type { DeckDefinition } from "../../core/types";
import { SLIDES } from "./content";
import { MentalModel } from "./MentalModel";
import { CSS } from "./styles";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">source atlas</Pill>}>Live evidence files</CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code>rust/cli/src/dispatch.rs</Code>
          <Code>rust/cli/src/execute_mode.rs</Code>
          <Code>rust/runtime/src/engine/coordinator.rs</Code>
          <Code>rust/runtime/src/engine/application.rs</Code>
          <Code>rust/loadgen-core/src/sink.rs</Code>
          <Code>rust/cli/Cargo.toml</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const rustArchitectureAtlasDeck: DeckDefinition = {
  id: "rust-architecture-atlas",
  route: "/rust-architecture-atlas",
  storagePrefix: "rust-arch-atlas-explainer",
  classPrefix: "deck-rust-atlas",
  eyebrowLabel: "RUST ARCHITECTURE ATLAS",
  startGateTitle: "Rust architecture atlas",
  hub: {
    highlight: "Rust architecture atlas",
    title: "source-oriented map",
    description:
      "Eleven-slide companion to the overview deck: system landscape, crate graph, hot path, protocol v2, scheduled and graph execution, endpoints, metrics, cellular, features, and extension seams.",
  },
  slides: SLIDES,
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
