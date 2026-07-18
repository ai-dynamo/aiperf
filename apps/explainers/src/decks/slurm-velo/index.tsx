import { Card, CardBody, CardHeader, Code, Pill, Stack, Text } from "../../core/ui";
import { SLIDES } from "./content";
import { CSS } from "./styles";
import { MentalModel } from "./MentalModel";
import type { DeckDefinition } from "../../core/types";

function FinalCard() {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">copy &amp; run</Pill>}>The commands</CardHeader>
      <CardBody>
        <Stack gap={12}>
          <Stack gap={5}>
            <div style={{ color: "#e4e4e48d", fontSize: 14, fontWeight: 700 }}>
              1 · GENERATE A SUBMISSION SCRIPT
            </div>
            <Code style={{ fontSize: 14 }}>
              aiperf slurm generate --config benchmark.yaml --cells 4 --output job.sbatch
            </Code>
          </Stack>
          <Stack gap={5}>
            <div style={{ color: "#e4e4e48d", fontSize: 14, fontWeight: 700 }}>
              2 · SUBMIT IT TO SLURM
            </div>
            <Code style={{ fontSize: 14 }}>sbatch job.sbatch</Code>
            <Text tone="secondary">
              Every task launches <Code>aiperf slurm run</Code>. Rank picks controller vs. cell; Velo
              wires the control plane automatically.
            </Text>
          </Stack>
        </Stack>
      </CardBody>
    </Card>
  );
}

export const slurmVeloDeck: DeckDefinition = {
  id: "slurm-velo",
  route: "/slurm-velo",
  storagePrefix: "slurm-explainer",
  classPrefix: "slurm101",
  eyebrowLabel: "SLURM + VELO FROM SCRATCH",
  startGateTitle: "SLURM + Velo from scratch",
  hub: {
    highlight: "SLURM + Velo",
    title: "from scratch",
    description:
      "Step-by-step slideshow for how SLURM launches cellular AIPerf runs and how Velo wires the control plane.",
  },
  slides: SLIDES,
  MentalModel: ({ slideIndex, slide }) => (
    <MentalModel slideIndex={slideIndex} slide={slide} />
  ),
  css: CSS,
  FinalCard,
};
