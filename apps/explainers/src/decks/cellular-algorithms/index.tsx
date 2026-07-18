/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckDefinition } from "../../core/types";
import { Card, CardBody, CardHeader, Code, Pill, Stack } from "../../core/ui";
import { SLIDES } from "./content";
import { MentalModel } from "./MentalModel";
import { CSS } from "./styles";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">exhaustive catalog</Pill>}>
        Continue in the source workbook
      </CardHeader>
      <CardBody>
        <Stack gap={8}>
          <Code>docs/canvases/cellular-algorithm-workbook.canvas.tsx</Code>
          <Code>rust/runtime/src/engine/cellular_controller.rs</Code>
          <Code>rust/runtime/src/cellular/partition.rs</Code>
          <Code>rust/runtime/src/cellular/shard.rs</Code>
          <Code>rust/runtime/src/metrics_core/store.rs</Code>
          <Code>rust/runtime/src/engine/artifact_shipping.rs</Code>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const cellularAlgorithmsDeck: DeckDefinition = {
  id: "cellular-algorithms",
  route: "/cellular-algorithms",
  storagePrefix: "cellular-algorithms-explainer",
  classPrefix: "deck-algorithm",
  eyebrowLabel: "CELLULAR ALGORITHM WORKBOOK",
  startGateTitle: "Cellular algorithm workbook",
  hub: {
    highlight: "Cellular algorithms",
    title: "compose · compare · trace evidence",
    description:
      "Sixteen-slide maintainer map of the cellular algorithm workbook: eight evidence chapters, representative built/partial/feature-gated algorithms, route composition, and the decisions that change ownership, fidelity, topology, and shipping.",
  },
  slides: SLIDES,
  glossary: [],
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
