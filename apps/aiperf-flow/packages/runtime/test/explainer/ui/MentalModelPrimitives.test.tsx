// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";
import {
  ArchitectureBox,
  DataFlowArrow,
  GridLayout,
  SequenceTimeline,
  ThemePalette,
  ContrastMatrix,
  LegendBlock,
} from "../../../src/explainer/ui/MentalModelPrimitives.js";
import type { ResolvedTheme } from "../../../src/theme/types.js";

afterEach(() => {
  cleanup();
});

const mockTheme: ResolvedTheme = {
  id: "test-theme",
  values: {
    "surface.primary": { kind: "color", value: "#303334" },
    "surface.secondary": { kind: "color", value: "#383c3e" },
    "ink.primary": { kind: "color", value: "#f1f3f2" },
    "ink.muted": { kind: "color", value: "#aeb4b5" },
    "accent.execute": { kind: "color", value: "#7dce82" },
    "accent.alert": { kind: "color", value: "#f07972" },
    "accent.attention": { kind: "color", value: "#f0cf58" },
    "structure.divider": { kind: "color", value: "#d7dada" },
  } as any,
};

describe("MentalModelPrimitives", () => {
  describe("ArchitectureBox", () => {
    test("renders basic architecture box", () => {
      render(
        <ArchitectureBox
          theme={mockTheme}
          label="Service"
          description="A sample service"
        />
      );

      expect(screen.getByText("Service")).toBeInTheDocument();
      expect(screen.getByText("A sample service")).toBeInTheDocument();
    });

    test("renders with custom box type styling", () => {
      const { container } = render(
        <ArchitectureBox
          theme={mockTheme}
          label="Process"
          boxType="process"
        />
      );

      const box = container.firstChild as HTMLElement;
      expect(box).toHaveStyle({ borderRadius: "8px" });
    });

    test("renders with custom dimensions", () => {
      const { container } = render(
        <ArchitectureBox
          theme={mockTheme}
          label="Box"
          width={200}
          height={100}
        />
      );

      const box = container.firstChild as HTMLElement;
      expect(box.style.width).toBe("200px");
      expect(box.style.height).toBe("100px");
    });

    test("renders with children", () => {
      render(
        <ArchitectureBox theme={mockTheme} label="Container">
          <div>Child Content</div>
        </ArchitectureBox>
      );

      expect(screen.getByText("Child Content")).toBeInTheDocument();
    });

    test("renders with icon", () => {
      render(
        <ArchitectureBox
          theme={mockTheme}
          label="WithIcon"
          icon="📦"
        />
      );

      expect(screen.getByText("📦")).toBeInTheDocument();
    });

    test("renders all box types", () => {
      const types: Array<"service" | "process" | "boundary" | "default"> = [
        "service",
        "process",
        "boundary",
        "default",
      ];

      types.forEach((type) => {
        const { unmount } = render(
          <ArchitectureBox
            theme={mockTheme}
            label={`Box-${type}`}
            boxType={type}
          />
        );
        expect(screen.getByText(`Box-${type}`)).toBeInTheDocument();
        unmount();
      });
    });

    test("applies custom class name and style", () => {
      const { container } = render(
        <ArchitectureBox
          theme={mockTheme}
          label="Styled"
          className="custom-class"
          style={{ opacity: 0.5 }}
        />
      );

      const box = container.firstChild as HTMLElement;
      expect(box.className).toBe("custom-class");
      expect(box.style.opacity).toBe("0.5");
    });
  });

  describe("DataFlowArrow", () => {
    test("renders arrow without error", () => {
      const { container } = render(
        <DataFlowArrow
          theme={mockTheme}
          label="Request"
        />
      );

      expect(container.querySelector("svg")).toBeInTheDocument();
      expect(screen.getByText("Request")).toBeInTheDocument();
    });

    test("renders all arrow directions", () => {
      const directions: Array<"up" | "down" | "left" | "right" | "diagonal-down-right"> = [
        "up",
        "down",
        "left",
        "right",
        "diagonal-down-right",
      ];

      directions.forEach((dir) => {
        const { unmount } = render(
          <DataFlowArrow
            theme={mockTheme}
            direction={dir}
            label={`Arrow-${dir}`}
          />
        );
        expect(screen.getByText(`Arrow-${dir}`)).toBeInTheDocument();
        unmount();
      });
    });

    test("renders different arrow variants", () => {
      const variants: Array<"solid" | "dashed" | "dotted"> = [
        "solid",
        "dashed",
        "dotted",
      ];

      variants.forEach((variant) => {
        const { unmount } = render(
          <DataFlowArrow
            theme={mockTheme}
            variant={variant}
            label={variant}
          />
        );
        expect(screen.getByText(variant)).toBeInTheDocument();
        unmount();
      });
    });

    test("renders arrow with custom length", () => {
      const { container } = render(
        <DataFlowArrow
          theme={mockTheme}
          length={100}
        />
      );

      const svg = container.querySelector("svg");
      expect(svg).toHaveAttribute("width", "100");
    });

    test("renders arrow without label", () => {
      const { container } = render(
        <DataFlowArrow theme={mockTheme} />
      );

      expect(container.querySelector("svg")).toBeInTheDocument();
      expect(screen.queryByText(/^Arrow/)).not.toBeInTheDocument();
    });
  });

  describe("GridLayout", () => {
    test("renders grid with children", () => {
      render(
        <GridLayout theme={mockTheme} columns={2}>
          <div>Item 1</div>
          <div>Item 2</div>
          <div>Item 3</div>
        </GridLayout>
      );

      expect(screen.getByText("Item 1")).toBeInTheDocument();
      expect(screen.getByText("Item 2")).toBeInTheDocument();
      expect(screen.getByText("Item 3")).toBeInTheDocument();
    });

    test("renders grid with custom column count", () => {
      const { container } = render(
        <GridLayout theme={mockTheme} columns={4} gap={20}>
          <div>Item</div>
        </GridLayout>
      );

      const grid = container.firstChild as HTMLElement;
      expect(grid.style.gridTemplateColumns).toBe("repeat(4, 1fr)");
      expect(grid.style.gap).toBe("20px");
    });

    test("applies custom alignment", () => {
      const { container } = render(
        <GridLayout theme={mockTheme} align="end">
          <div>Item</div>
        </GridLayout>
      );

      const grid = container.firstChild as HTMLElement;
      expect(grid.style.alignItems).toBe("end");
      expect(grid.style.justifyContent).toBe("end");
    });
  });

  describe("SequenceTimeline", () => {
    const steps = [
      { id: "step1", label: "Step 1", description: "First step" },
      { id: "step2", label: "Step 2", description: "Second step" },
      { id: "step3", label: "Step 3" },
    ];

    test("renders timeline with all steps", () => {
      render(
        <SequenceTimeline
          theme={mockTheme}
          steps={steps}
        />
      );

      expect(screen.getByText("Step 1")).toBeInTheDocument();
      expect(screen.getByText("Step 2")).toBeInTheDocument();
      expect(screen.getByText("Step 3")).toBeInTheDocument();
      expect(screen.getByText("First step")).toBeInTheDocument();
      expect(screen.getByText("Second step")).toBeInTheDocument();
    });

    test("renders with active step indicator", () => {
      const { container } = render(
        <SequenceTimeline
          theme={mockTheme}
          steps={steps}
          activeStep={1}
        />
      );

      // The active step should show a number, completed steps show checkmarks
      const circles = container.querySelectorAll("[style*='border-radius: 50%']");
      expect(circles).toHaveLength(3);
    });

    test("renders vertical orientation", () => {
      const { container } = render(
        <SequenceTimeline
          theme={mockTheme}
          steps={steps}
          orientation="vertical"
        />
      );

      const timeline = container.firstChild as HTMLElement;
      expect(timeline.style.flexDirection).toBe("column");
    });

    test("renders horizontal orientation", () => {
      const { container } = render(
        <SequenceTimeline
          theme={mockTheme}
          steps={steps}
          orientation="horizontal"
        />
      );

      const timeline = container.firstChild as HTMLElement;
      expect(timeline.style.flexDirection).toBe("row");
    });

    test("renders with custom step size", () => {
      const { container } = render(
        <SequenceTimeline
          theme={mockTheme}
          steps={steps}
          stepSize={80}
        />
      );

      const circles = container.querySelectorAll("[style*='border-radius: 50%']");
      expect(circles[0].getAttribute("style")).toContain("width: 80px");
      expect(circles[0].getAttribute("style")).toContain("height: 80px");
    });
  });

  describe("ThemePalette", () => {
    const colors = [
      { role: "surface.primary", label: "Primary Surface" },
      { role: "ink.primary", label: "Primary Text" },
      { role: "accent.execute", label: "Execute" },
    ];

    test("renders palette with all colors", () => {
      render(
        <ThemePalette
          theme={mockTheme}
          colors={colors}
        />
      );

      expect(screen.getByText("Primary Surface")).toBeInTheDocument();
      expect(screen.getByText("Primary Text")).toBeInTheDocument();
      expect(screen.getByText("Execute")).toBeInTheDocument();
    });

    test("renders in grid format", () => {
      const { container } = render(
        <ThemePalette
          theme={mockTheme}
          colors={colors}
          format="grid"
        />
      );

      const palette = container.firstChild as HTMLElement;
      expect(palette.style.display).toBe("grid");
    });

    test("renders in row format", () => {
      const { container } = render(
        <ThemePalette
          theme={mockTheme}
          colors={colors}
          format="row"
        />
      );

      const palette = container.firstChild as HTMLElement;
      expect(palette.style.display).toBe("flex");
    });

    test("renders with different sizes", () => {
      const sizes: Array<"small" | "medium" | "large"> = [
        "small",
        "medium",
        "large",
      ];

      sizes.forEach((size) => {
        const { unmount } = render(
          <ThemePalette
            theme={mockTheme}
            colors={colors}
            size={size}
          />
        );
        expect(screen.getByText("Primary Surface")).toBeInTheDocument();
        unmount();
      });
    });

    test("renders color hex values", () => {
      render(
        <ThemePalette
          theme={mockTheme}
          colors={colors}
        />
      );

      expect(screen.getByText("#303334")).toBeInTheDocument();
      expect(screen.getByText("#f1f3f2")).toBeInTheDocument();
      expect(screen.getByText("#7dce82")).toBeInTheDocument();
    });
  });

  describe("ContrastMatrix", () => {
    const pairs = [
      {
        foreground: "#f1f3f2",
        background: "#303334",
        label: "High Contrast",
        ratio: 8.2,
      },
      {
        foreground: "#aeb4b5",
        background: "#383c3e",
        label: "Medium Contrast",
        ratio: 3.5,
      },
    ];

    test("renders contrast matrix with pairs", () => {
      render(
        <ContrastMatrix
          theme={mockTheme}
          pairs={pairs}
        />
      );

      expect(screen.getByText("High Contrast")).toBeInTheDocument();
      expect(screen.getByText("Medium Contrast")).toBeInTheDocument();
    });

    test("displays contrast ratios", () => {
      render(
        <ContrastMatrix
          theme={mockTheme}
          pairs={pairs}
        />
      );

      expect(screen.getByText(/Ratio: 8\.20:1/)).toBeInTheDocument();
      expect(screen.getByText(/Ratio: 3\.50:1/)).toBeInTheDocument();
    });

    test("applies pass/fail styling based on ratio", () => {
      const { container } = render(
        <ContrastMatrix
          theme={mockTheme}
          pairs={pairs}
          minRatio={4.5}
        />
      );

      const contrastBoxes = container.querySelectorAll("[style*='border']");
      expect(contrastBoxes.length).toBeGreaterThan(0);
    });

    test("renders pairs without ratio", () => {
      const pairsNoRatio = [
        {
          foreground: "#f1f3f2",
          background: "#303334",
          label: "No Ratio",
        },
      ];

      render(
        <ContrastMatrix
          theme={mockTheme}
          pairs={pairsNoRatio}
        />
      );

      expect(screen.getByText("No Ratio")).toBeInTheDocument();
    });
  });

  describe("LegendBlock", () => {
    const entries = [
      {
        id: "entry1",
        label: "Entry 1",
        color: "#7dce82",
        description: "First entry",
      },
      {
        id: "entry2",
        label: "Entry 2",
        icon: "🔷",
      },
      { id: "entry3", label: "Entry 3" },
    ];

    test("renders legend with all entries", () => {
      render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
        />
      );

      expect(screen.getByText("Entry 1")).toBeInTheDocument();
      expect(screen.getByText("Entry 2")).toBeInTheDocument();
      expect(screen.getByText("Entry 3")).toBeInTheDocument();
      expect(screen.getByText("First entry")).toBeInTheDocument();
    });

    test("renders legend with title", () => {
      render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
          title="Legend Title"
        />
      );

      expect(screen.getByText("Legend Title")).toBeInTheDocument();
    });

    test("renders vertical orientation", () => {
      const { container } = render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
          orientation="vertical"
        />
      );

      const legend = container.firstChild as HTMLElement;
      expect(legend.style.flexDirection).toBe("column");
    });

    test("renders horizontal orientation", () => {
      const { container } = render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
          orientation="horizontal"
        />
      );

      const legend = container.firstChild as HTMLElement;
      expect(legend.style.flexDirection).toBe("row");
    });

    test("renders entries with color swatches", () => {
      const { container } = render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
        />
      );

      const swatches = container.querySelectorAll(
        "[style*='border-radius: 4px']"
      );
      expect(swatches.length).toBeGreaterThan(0);
    });

    test("renders entries with icons", () => {
      render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
        />
      );

      expect(screen.getByText("🔷")).toBeInTheDocument();
    });

    test("renders entries without color or icon", () => {
      render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
        />
      );

      expect(screen.getByText("Entry 3")).toBeInTheDocument();
    });
  });

  describe("Theme Integration", () => {
    test("all components respect custom theme", () => {
      const customTheme: ResolvedTheme = {
        id: "custom",
        values: {
          "surface.primary": { kind: "color", value: "#ff0000" },
          "ink.primary": { kind: "color", value: "#00ff00" },
          "accent.execute": { kind: "color", value: "#0000ff" },
          "accent.alert": { kind: "color", value: "#ffff00" },
          "accent.attention": { kind: "color", value: "#ff00ff" },
          "structure.divider": { kind: "color", value: "#00ffff" },
          "surface.secondary": { kind: "color", value: "#aabbcc" },
          "ink.muted": { kind: "color", value: "#ddeeff" },
        } as any,
      };

      const { container } = render(
        <div>
          <ArchitectureBox theme={customTheme} label="Box" />
          <DataFlowArrow theme={customTheme} />
          <GridLayout theme={customTheme}>
            <div>Item</div>
          </GridLayout>
          <SequenceTimeline theme={customTheme} steps={[{ id: "1", label: "Step 1" }]} />
          <ThemePalette theme={customTheme} colors={[{ role: "surface.primary", label: "Surface" }]} />
          <ContrastMatrix theme={customTheme} pairs={[{ foreground: "#000", background: "#fff", label: "Test" }]} />
          <LegendBlock theme={customTheme} entries={[{ id: "1", label: "Entry 1" }]} />
        </div>
      );

      // All components should render without error
      expect(container.firstChild).toBeInTheDocument();
    });
  });

  describe("Composability", () => {
    test("components can be nested together", () => {
      render(
        <GridLayout theme={mockTheme} columns={2}>
          <ArchitectureBox theme={mockTheme} label="Box 1" />
          <ArchitectureBox theme={mockTheme} label="Box 2" />
          <DataFlowArrow theme={mockTheme} label="Flow" />
          <LegendBlock
            theme={mockTheme}
            entries={[
              { id: "1", label: "Legend" },
            ]}
          />
        </GridLayout>
      );

      expect(screen.getByText("Box 1")).toBeInTheDocument();
      expect(screen.getByText("Box 2")).toBeInTheDocument();
      expect(screen.getByText("Flow")).toBeInTheDocument();
      expect(screen.getByText("Legend")).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    test("architecture box has semantic structure", () => {
      const { container } = render(
        <ArchitectureBox
          theme={mockTheme}
          label="Service"
          description="Description"
        />
      );

      const box = container.firstChild as HTMLElement;
      expect(box.textContent).toContain("Service");
      expect(box.textContent).toContain("Description");
    });

    test("timeline steps are properly labeled", () => {
      const steps = [
        { id: "1", label: "First Step" },
        { id: "2", label: "Second Step" },
      ];

      render(
        <SequenceTimeline
          theme={mockTheme}
          steps={steps}
        />
      );

      expect(screen.getByText("First Step")).toBeInTheDocument();
      expect(screen.getByText("Second Step")).toBeInTheDocument();
    });

    test("legend entries are readable", () => {
      const entries = [
        { id: "1", label: "Important", description: "This is important" },
      ];

      render(
        <LegendBlock
          theme={mockTheme}
          entries={entries}
        />
      );

      expect(screen.getByText("Important")).toBeInTheDocument();
      expect(screen.getByText("This is important")).toBeInTheDocument();
    });
  });
});
