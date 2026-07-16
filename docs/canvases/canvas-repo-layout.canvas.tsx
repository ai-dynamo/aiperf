import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Spacer,
  Stack,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

type FlowStep = {
  id: string;
  title: string;
  detail: string;
  tone: "built" | "local" | "git";
};

const canvases = [
  { name: "cellular-algorithm-workbook", topic: "Cellular algorithm workbook" },
  { name: "cellular-architecture", topic: "Cellular controller / cell topology" },
  { name: "dynosim-offline-flow", topic: "Dynosim offline replay flow" },
  { name: "mock-server-architecture", topic: "aiperf-mock-server surface map" },
  { name: "rust-aiperf-architecture", topic: "Rust product architecture" },
  { name: "segment-pools-and-body-plans", topic: "Segment pools and body plans" },
  { name: "velo-in-aiperf", topic: "Velo transport in cellular mode" },
];

const flow: FlowStep[] = [
  {
    id: "edit",
    title: "Edit in repo",
    detail:
      "Source files live in docs/canvases/*.canvas.tsx and are versioned with the project.",
    tone: "git",
  },
  {
    id: "symlink",
    title: "IDE bridge",
    detail:
      "Each file is symlinked into ~/.cursor/projects/<workspace>/canvases/ so Cursor can compile and preview it beside chat.",
    tone: "built",
  },
  {
    id: "sidecar",
    title: "Local runtime state",
    detail:
      "*.canvas.data.json and *.canvas.status.json stay in the managed directory and are gitignored.",
    tone: "local",
  },
];

const tonePill = {
  built: "success" as const,
  local: "warning" as const,
  git: "info" as const,
};

export default function CanvasRepoLayout() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 980 }}>
      <Stack gap={8}>
        <H1>Canvas repo layout</H1>
        <Text color="secondary">
          Committed canvases for the AIPerf Rust workspace. Source of truth is in
          the repo; Cursor still previews through its managed canvases directory.
        </Text>
        <Row gap={8} wrap>
          <Pill tone="success">7 canvases migrated</Pill>
          <Pill tone="info">docs/canvases/</Pill>
          <Pill tone="neutral">symlink bridge</Pill>
        </Row>
      </Stack>

      <Callout tone="info" title="Why not commit only to the repo path?">
        Cursor detects canvases only when they appear as direct children of
        ~/.cursor/projects/&lt;workspace&gt;/canvases/. Symlinks satisfy that
        rule while keeping git history on the real files under docs/canvases/.
      </Callout>

      <Grid columns={3} gap={12}>
        {flow.map((step) => (
          <Card key={step.id} variant="outline">
            <CardHeader
              title={step.title}
              trailing={<Pill tone={tonePill[step.tone]}>{step.tone}</Pill>}
            />
            <CardBody>
              <Text size="sm" color="secondary">
                {step.detail}
              </Text>
            </CardBody>
          </Card>
        ))}
      </Grid>

      <Stack gap={10}>
        <H2>Directory map</H2>
        <Card>
          <CardBody>
            <Code block>{`repo/
  docs/canvases/
    *.canvas.tsx          # committed source (edit here)
    .gitignore            # ignores runtime sidecars if they land here

managed (per machine)/
  ~/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-rust/canvases/
    *.canvas.tsx -> repo/docs/canvases/*.canvas.tsx
    *.canvas.data.json    # local UI state (not committed)
    tsconfig.json         # IDE tooling only`}</Code>
          </CardBody>
        </Card>
      </Stack>

      <Stack gap={10}>
        <H2>Committed canvases</H2>
        <Table
          caption="Source: docs/canvases/ in the AIPerf Rust repo"
          columns={[
            { key: "file", header: "File", width: "42%" },
            { key: "topic", header: "Topic" },
            {
              key: "open",
              header: "Open",
              align: "right",
              width: 100,
            },
          ]}
          rows={canvases.map((canvas) => ({
            key: canvas.name,
            file: `${canvas.name}.canvas.tsx`,
            topic: canvas.topic,
            open: (
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  dispatch({
                    type: "openFile",
                    path: `docs/canvases/${canvas.name}.canvas.tsx`,
                  })
                }
              >
                Open
              </Button>
            ),
          }))}
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Adding a new canvas</H2>
        <Grid columns={2} gap={12}>
          <Card variant="outline">
            <CardHeader title="1. Create source in repo" />
            <CardBody>
              <Text size="sm" color="secondary">
                Add docs/canvases/my-topic.canvas.tsx. Import only from
                cursor/canvas and default-export one component.
              </Text>
            </CardBody>
          </Card>
          <Card variant="outline">
            <CardHeader title="2. Bridge to Cursor" />
            <CardBody>
              <Code block>{`ln -s "$PWD/docs/canvases/my-topic.canvas.tsx" \\
  ~/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-rust/canvases/`}</Code>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Card
        style={{
          background: theme.fill.tertiary,
          borderColor: theme.stroke.secondary,
        }}
      >
        <CardBody>
          <H3>Companion planning docs</H3>
          <Spacer size={6} />
          <Text size="sm" color="secondary">
            Markdown storyboards for some canvases already live under
            docs/superpowers/plans/. Keep narrative/planning text there; keep
            interactive architecture views here as .canvas.tsx files.
          </Text>
        </CardBody>
      </Card>
    </Stack>
  );
}
