// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

/** Definition of a single glossary term with word and meaning. */
export interface GlossaryTerm {
  word: string;
  meaning: string;
}

/** Props for the GlossaryUI component. */
export interface GlossaryUIProps {
  /** Array of glossary terms to display. */
  terms: GlossaryTerm[];
  /** Currently selected term index, if any. */
  selectedTermIndex?: number;
  /** Callback when a term is selected from the index. */
  onSelectTerm?: (index: number) => void;
  /** Optional CSS classes for styling. */
  className?: string;
  /** Theme colors for the glossary UI. */
  theme?: {
    backgroundColor?: string;
    textColor?: string;
    accentColor?: string;
    borderColor?: string;
  };
  /** Show term index by default. */
  showIndex?: boolean;
}

/** Renders a glossary term definition on slide with index/search capabilities. */
export function GlossaryUI({
  terms,
  selectedTermIndex,
  onSelectTerm,
  className = "",
  theme = {},
  showIndex = false,
}: GlossaryUIProps): ReactNode {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isIndexOpen, setIsIndexOpen] = useState<boolean>(showIndex);

  // Compute filtered terms based on search query
  const filteredTerms = useMemo(() => {
    if (!searchQuery.trim()) {
      return terms;
    }
    const lowerQuery = searchQuery.toLowerCase();
    return terms.filter(
      (term) =>
        term.word.toLowerCase().includes(lowerQuery) ||
        term.meaning.toLowerCase().includes(lowerQuery),
    );
  }, [terms, searchQuery]);

  // Get the current term to display
  const currentTerm =
    selectedTermIndex !== undefined && terms[selectedTermIndex]
      ? terms[selectedTermIndex]
      : filteredTerms[0];

  const themeStyles: CSSProperties = {
    backgroundColor: theme.backgroundColor || "rgba(255, 255, 255, 0.1)",
    color: theme.textColor || "#f2eee3",
    borderColor: theme.borderColor || "rgba(255, 255, 255, 0.2)",
  };

  const handleTermSelect = (index: number): void => {
    if (onSelectTerm) {
      onSelectTerm(index);
    }
  };

  const handleSearch = (query: string): void => {
    setSearchQuery(query);
  };

  return (
    <div
      className={`glossary-ui ${className}`.trim()}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
        padding: "1rem",
        borderRadius: "0.5rem",
        border: `1px solid ${theme.borderColor || "rgba(255, 255, 255, 0.2)"}`,
        ...themeStyles,
      }}
    >
      {/* Glossary Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3
          style={{
            margin: 0,
            fontSize: "0.875rem",
            fontWeight: 600,
            letterSpacing: "0.05em",
            textTransform: "uppercase",
            opacity: 0.8,
          }}
        >
          Glossary
        </h3>
        <button
          onClick={() => setIsIndexOpen(!isIndexOpen)}
          style={{
            padding: "0.25rem 0.5rem",
            fontSize: "0.75rem",
            backgroundColor: "transparent",
            border: `1px solid ${theme.accentColor || "#72d6a2"}`,
            color: theme.accentColor || "#72d6a2",
            borderRadius: "0.25rem",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            if (e.currentTarget instanceof HTMLElement) {
              e.currentTarget.style.backgroundColor = theme.accentColor || "#72d6a2";
              e.currentTarget.style.color = theme.backgroundColor || "#24282b";
            }
          }}
          onMouseLeave={(e) => {
            if (e.currentTarget instanceof HTMLElement) {
              e.currentTarget.style.backgroundColor = "transparent";
              e.currentTarget.style.color = theme.accentColor || "#72d6a2";
            }
          }}
        >
          {isIndexOpen ? "Hide Index" : "Show Index"}
        </button>
      </div>

      {/* Term Definition Display */}
      {currentTerm && (
        <div
          style={{
            backgroundColor: "rgba(0, 0, 0, 0.2)",
            padding: "0.75rem",
            borderRadius: "0.375rem",
            borderLeft: `3px solid ${theme.accentColor || "#72d6a2"}`,
          }}
        >
          <div
            style={{
              fontSize: "1rem",
              fontWeight: 600,
              color: theme.accentColor || "#72d6a2",
              marginBottom: "0.5rem",
            }}
          >
            {currentTerm.word}
          </div>
          <div
            style={{
              fontSize: "0.875rem",
              lineHeight: 1.5,
              color: theme.textColor || "#f2eee3",
            }}
          >
            {currentTerm.meaning}
          </div>
        </div>
      )}

      {/* Search Input */}
      {terms.length > 3 && (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <input
            type="text"
            placeholder="Search terms..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            style={{
              flex: 1,
              padding: "0.5rem",
              fontSize: "0.875rem",
              backgroundColor: "rgba(0, 0, 0, 0.3)",
              border: `1px solid ${theme.borderColor || "rgba(255, 255, 255, 0.2)"}`,
              borderRadius: "0.25rem",
              color: theme.textColor || "#f2eee3",
              outline: "none",
            } as CSSProperties}
            onFocus={(e) => {
              if (e.target instanceof HTMLInputElement) {
                e.target.style.borderColor = theme.accentColor || "#72d6a2";
              }
            }}
            onBlur={(e) => {
              if (e.target instanceof HTMLInputElement) {
                e.target.style.borderColor =
                  theme.borderColor || "rgba(255, 255, 255, 0.2)";
              }
            }}
          />
        </div>
      )}

      {/* Glossary Index */}
      {isIndexOpen && (
        <div
          style={{
            maxHeight: "12rem",
            overflowY: "auto",
            borderTop: `1px solid ${theme.borderColor || "rgba(255, 255, 255, 0.2)"}`,
            paddingTop: "0.5rem",
          }}
        >
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: "0.5rem" }}>
            {filteredTerms.length} term{filteredTerms.length !== 1 ? "s" : ""}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            {filteredTerms.map((term, idx) => {
              const originalIndex = terms.indexOf(term);
              const isSelected = originalIndex === selectedTermIndex;
              return (
                <button
                  key={idx}
                  onClick={() => handleTermSelect(originalIndex)}
                  style={{
                    padding: "0.375rem 0.5rem",
                    textAlign: "left",
                    fontSize: "0.8125rem",
                    backgroundColor: isSelected
                      ? `${theme.accentColor || "#72d6a2"}20`
                      : "transparent",
                    border: "none",
                    borderRadius: "0.25rem",
                    color: isSelected
                      ? theme.accentColor || "#72d6a2"
                      : theme.textColor || "#f2eee3",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                  onMouseEnter={(e) => {
                    if (e.currentTarget instanceof HTMLElement && !isSelected) {
                      e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.1)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (e.currentTarget instanceof HTMLElement && !isSelected) {
                      e.currentTarget.style.backgroundColor = "transparent";
                    }
                  }}
                >
                  {term.word}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty State */}
      {filteredTerms.length === 0 && (
        <div
          style={{
            textAlign: "center",
            padding: "1rem",
            opacity: 0.5,
            fontSize: "0.875rem",
          }}
        >
          No terms match your search.
        </div>
      )}
    </div>
  );
}
