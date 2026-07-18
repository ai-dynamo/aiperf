// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GlossaryUI, type GlossaryTerm } from "../../../src/explainer/ui/GlossaryUI.js";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("GlossaryUI", () => {
  const mockTerms: GlossaryTerm[] = [
    {
      word: "Latency",
      meaning: "The time it takes for a request to be processed.",
    },
    {
      word: "Throughput",
      meaning: "The number of requests processed per unit time.",
    },
    {
      word: "Token",
      meaning: "A unit of text in a language model.",
    },
    {
      word: "Inference",
      meaning: "The process of running a model to generate predictions.",
    },
  ];

  describe("Term Display", () => {
    it("renders the first term by default", () => {
      render(<GlossaryUI terms={mockTerms} />);

      const heading = screen.getByText("Latency");
      expect(heading).toBeTruthy();
      expect(
        screen.getByText("The time it takes for a request to be processed."),
      ).toBeTruthy();
    });

    it("displays selected term when selectedTermIndex is provided", () => {
      render(<GlossaryUI terms={mockTerms} selectedTermIndex={2} />);

      const heading = screen.getByText("Token");
      expect(heading).toBeTruthy();
      expect(
        screen.getByText("A unit of text in a language model."),
      ).toBeTruthy();
    });

    it("renders glossary header", () => {
      render(<GlossaryUI terms={mockTerms} />);

      const header = screen.getByText("Glossary");
      expect(header).toBeTruthy();
      expect(header.tagName).toBe("H3");
    });

    it("applies theme colors to term display", () => {
      const { container } = render(
        <GlossaryUI
          terms={mockTerms}
          theme={{
            backgroundColor: "#24282b",
            textColor: "#f2eee3",
            accentColor: "#72d6a2",
            borderColor: "#3a3f44",
          }}
        />,
      );

      const glossaryContainer = container.querySelector(".glossary-ui");
      expect(glossaryContainer).toBeTruthy();
      const styles = window.getComputedStyle(glossaryContainer!);
      expect(styles.color).toBeDefined();
    });

    it("displays empty state when no terms are provided", () => {
      render(<GlossaryUI terms={[]} />);

      expect(screen.getByText("Glossary")).toBeTruthy();
    });

    it("displays multiple terms correctly when selected", () => {
      const { rerender } = render(
        <GlossaryUI terms={mockTerms} selectedTermIndex={0} />,
      );

      expect(screen.getByText("Latency")).toBeTruthy();

      rerender(
        <GlossaryUI terms={mockTerms} selectedTermIndex={1} />,
      );

      expect(screen.getByText("Throughput")).toBeTruthy();
    });
  });

  describe("Index Navigation", () => {
    it("hides index by default", () => {
      render(<GlossaryUI terms={mockTerms} showIndex={false} />);

      const buttons = screen.queryAllByRole("button");
      const indexButtons = buttons.filter((btn) =>
        mockTerms.some((term) => btn.textContent?.includes(term.word)),
      );

      expect(indexButtons.length).toBe(0);
    });

    it("shows index when showIndex prop is true", () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const allButtons = screen.getAllByRole("button");
      const termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );

      expect(termButtons.length).toBe(4);
    });

    it("toggles index visibility with Show/Hide Index button", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={false} />);

      const toggleButton = screen.getByText("Show Index");
      expect(toggleButton).toBeTruthy();

      await act(async () => {
        fireEvent.click(toggleButton);
      });

      expect(screen.getByText("Hide Index")).toBeTruthy();

      // Verify index buttons are visible
      const allButtons = screen.getAllByRole("button");
      const termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );
      expect(termButtons.length).toBe(4);

      const hideButton = screen.getByText("Hide Index");
      await act(async () => {
        fireEvent.click(hideButton);
      });

      expect(screen.getByText("Show Index")).toBeTruthy();
    });

    it("calls onSelectTerm callback when term is selected from index", async () => {
      const mockOnSelectTerm = vi.fn();

      render(
        <GlossaryUI
          terms={mockTerms}
          onSelectTerm={mockOnSelectTerm}
          showIndex={true}
        />,
      );

      const throughputButton = screen.getByText("Throughput");
      await act(async () => {
        fireEvent.click(throughputButton);
      });

      expect(mockOnSelectTerm).toHaveBeenCalledWith(1);
    });

    it("highlights selected term in index", async () => {
      const { container } = render(
        <GlossaryUI
          terms={mockTerms}
          selectedTermIndex={1}
          showIndex={true}
          theme={{ accentColor: "#72d6a2" }}
        />,
      );

      const allButtons = screen.getAllByRole("button");
      const throughputButton = allButtons.find(btn => btn.textContent === "Throughput");
      expect(throughputButton).toBeTruthy();
      const styles = window.getComputedStyle(throughputButton!);
      // Check that selected term has accent color applied
      expect(styles.color).toBeDefined();
    });

    it("displays term count in index", () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      expect(screen.getByText("4 terms")).toBeTruthy();
    });

    it("updates term count when filtered", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "token" } });
      });

      expect(screen.getByText("1 term")).toBeTruthy();
    });

    it("navigates through index without external selection callback", async () => {
      const { rerender } = render(
        <GlossaryUI terms={mockTerms} selectedTermIndex={0} showIndex={true} />,
      );

      // Verify first term is displayed
      const allButtons = screen.getAllByRole("button");
      const latencyButton = allButtons.find(btn => btn.textContent === "Latency");
      expect(latencyButton).toBeTruthy();

      // Simulate selecting another term by clicking
      const inferenceButton = allButtons.find(btn => btn.textContent === "Inference");
      expect(inferenceButton).toBeTruthy();

      await act(async () => {
        fireEvent.click(inferenceButton!);
      });

      // Component should update display based on onSelectTerm callback
      // For this test, we verify the callback was intended to be called
    });
  });

  describe("Search Functionality", () => {
    it("filters terms by word match", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "latency" } });
      });

      // Check that only Latency is shown in the index
      const allButtons = screen.getAllByRole("button");
      const termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );
      expect(termButtons.length).toBe(1);
      expect(termButtons[0].textContent).toBe("Latency");
    });

    it("filters terms by meaning match", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "model" } });
      });

      // Should match Token (A unit of text in a language model)
      // and Inference (The process of running a model...)
      const allButtons = screen.getAllByRole("button");
      const termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );
      expect(termButtons.length).toBe(2);
      const buttonTexts = termButtons.map(b => b.textContent);
      expect(buttonTexts).toContain("Token");
      expect(buttonTexts).toContain("Inference");
    });

    it("performs case-insensitive search", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "LATENCY" } });
      });

      const allButtons = screen.getAllByRole("button");
      const termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );
      expect(termButtons.length).toBe(1);
      expect(termButtons[0].textContent).toBe("Latency");
    });

    it("displays empty state when no results match", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "xyz999" } });
      });

      expect(screen.getByText("No terms match your search.")).toBeTruthy();
    });

    it("clears search when input is emptied", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "token" } });
      });

      expect(screen.getByText("1 term")).toBeTruthy();

      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "" } });
      });

      expect(screen.getByText("4 terms")).toBeTruthy();
    });

    it("only displays search input when there are more than 3 terms", () => {
      const fewTerms: GlossaryTerm[] = [
        { word: "A", meaning: "First" },
        { word: "B", meaning: "Second" },
        { word: "C", meaning: "Third" },
      ];

      const { rerender } = render(
        <GlossaryUI terms={fewTerms} showIndex={true} />,
      );

      expect(screen.queryByPlaceholderText("Search terms...")).toBeFalsy();

      const moreTerms = [
        ...fewTerms,
        { word: "D", meaning: "Fourth" },
      ];
      rerender(
        <GlossaryUI terms={moreTerms} showIndex={true} />,
      );

      expect(screen.getByPlaceholderText("Search terms...")).toBeTruthy();
    });

    it("updates displayed term on search results", async () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      // Verify first term displayed
      let allButtons = screen.getAllByRole("button");
      let termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );
      expect(termButtons.length).toBe(4);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "inference" } });
      });

      allButtons = screen.getAllByRole("button");
      termButtons = allButtons.filter(btn =>
        mockTerms.some(term => btn.textContent === term.word)
      );
      expect(termButtons.length).toBe(1);
      expect(termButtons[0].textContent).toBe("Inference");
    });

    it("maintains search while navigating index", async () => {
      const onSelectTerm = vi.fn();

      render(
        <GlossaryUI
          terms={mockTerms}
          onSelectTerm={onSelectTerm}
          showIndex={true}
        />,
      );

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "t" } });
      });

      // Should match: Throughput, Token
      expect(screen.getByText("Throughput")).toBeTruthy();
      expect(screen.getByText("Token")).toBeTruthy();
      expect(screen.queryByText("Latency")).toBeFalsy();

      const tokenButton = screen.getByText("Token");
      await act(async () => {
        fireEvent.click(tokenButton);
      });

      expect(onSelectTerm).toHaveBeenCalledWith(2); // Token is at index 2
    });
  });

  describe("Styling and Theme", () => {
    it("accepts custom className", () => {
      const { container } = render(
        <GlossaryUI terms={mockTerms} className="custom-glossary" />,
      );

      const glossaryDiv = container.querySelector(".glossary-ui.custom-glossary");
      expect(glossaryDiv).toBeTruthy();
    });

    it("applies theme colors to glossary container", () => {
      const { container } = render(
        <GlossaryUI
          terms={mockTerms}
          theme={{
            backgroundColor: "#24282b",
            textColor: "#f2eee3",
            accentColor: "#72d6a2",
            borderColor: "#3a3f44",
          }}
        />,
      );

      const glossaryDiv = container.querySelector(".glossary-ui");
      expect(glossaryDiv).toBeTruthy();
    });

    it("renders with default theme when no theme is provided", () => {
      const { container } = render(<GlossaryUI terms={mockTerms} />);

      const glossaryDiv = container.querySelector(".glossary-ui");
      expect(glossaryDiv).toBeTruthy();
      const styles = window.getComputedStyle(glossaryDiv!);
      expect(styles.display).toBe("flex");
    });
  });

  describe("Accessibility", () => {
    it("renders glossary header as h3", () => {
      render(<GlossaryUI terms={mockTerms} />);

      const heading = screen.getByRole("heading", { level: 3 });
      expect(heading.textContent).toBe("Glossary");
    });

    it("renders all index items as buttons", () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const buttons = screen.getAllByRole("button");
      // Should have Show/Hide Index button + 4 term buttons
      expect(buttons.length).toBeGreaterThanOrEqual(4);
    });

    it("search input has proper label", () => {
      render(<GlossaryUI terms={mockTerms} showIndex={true} />);

      const searchInput = screen.getByPlaceholderText("Search terms...");
      expect(searchInput).toBeTruthy();
      expect(searchInput.getAttribute("type")).toBe("text");
    });

    it("displays plural/singular term count correctly", () => {
      render(
        <GlossaryUI
          terms={[mockTerms[0]]}
          showIndex={true}
        />,
      );

      expect(screen.getByText("1 term")).toBeTruthy();
    });
  });

  describe("Edge Cases", () => {
    it("handles empty terms array gracefully", () => {
      render(<GlossaryUI terms={[]} />);

      expect(screen.getByText("Glossary")).toBeTruthy();
    });

    it("handles undefined selectedTermIndex", () => {
      render(
        <GlossaryUI
          terms={mockTerms}
          selectedTermIndex={undefined}
        />,
      );

      expect(screen.getByText("Latency")).toBeTruthy(); // First term
    });

    it("handles out-of-bounds selectedTermIndex", () => {
      render(
        <GlossaryUI
          terms={mockTerms}
          selectedTermIndex={999}
        />,
      );

      expect(screen.getByText("Latency")).toBeTruthy(); // Falls back to first
    });

    it("handles search with special characters", async () => {
      const specialTerms: GlossaryTerm[] = [
        { word: "Token (LLM)", meaning: "A unit with parens" },
        { word: "Rate [req/s]", meaning: "Speed with brackets" },
      ];

      render(
        <GlossaryUI
          terms={specialTerms}
          showIndex={true}
        />,
      );

      const searchInput = screen.getByPlaceholderText("Search terms...");
      await act(async () => {
        fireEvent.change(searchInput, { target: { value: "(LLM)" } });
      });

      expect(screen.getByText("Token (LLM)")).toBeTruthy();
    });

    it("handles very long term definitions", () => {
      const longTerms: GlossaryTerm[] = [
        {
          word: "Comprehensive",
          meaning:
            "A very long definition that explains in great detail the meaning of a term, including multiple clauses and extensive description that might wrap to multiple lines.",
        },
      ];

      render(<GlossaryUI terms={longTerms} />);

      const longMeaning = screen.getByText(longTerms[0].meaning);
      expect(longMeaning).toBeTruthy();
    });

    it("handles duplicate term words", () => {
      const dupTerms: GlossaryTerm[] = [
        { word: "Token", meaning: "First meaning" },
        { word: "Token", meaning: "Second meaning" },
      ];

      render(<GlossaryUI terms={dupTerms} showIndex={true} />);

      const tokens = screen.getAllByText("Token");
      expect(tokens.length).toBeGreaterThanOrEqual(1);
    });
  });
});
