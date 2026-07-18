// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import {
  type FlowCommand,
  searchCommands,
} from "../commands.js";

/** Controlled props for the Command Constellation dialog. */
export type CommandConstellationProps = Readonly<{
  commands: readonly FlowCommand[];
  open: boolean;
  onClose(): void;
  initialQuery?: string;
}>;

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

const backdropStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  zIndex: 40,
  display: "grid",
  placeItems: "start center",
  padding:
    "max(1rem, env(safe-area-inset-top, 0px)) max(1rem, env(safe-area-inset-right, 0px)) max(1rem, env(safe-area-inset-bottom, 0px)) max(1rem, env(safe-area-inset-left, 0px))",
  background: "rgb(7 17 31 / 72%)",
};

const mobileBackdropStyle: CSSProperties = {
  ...backdropStyle,
  padding: 0,
};

const dialogStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.75rem",
  width: "min(36rem, 100%)",
  maxHeight: "min(36rem, calc(100vh - 2rem))",
  marginBlockStart: "min(12vh, 4.5rem)",
  padding: "0.85rem 0.9rem 0.95rem",
  border: "1px solid rgb(104 168 255 / 28%)",
  borderRadius: "0.25rem",
  background: "rgb(7 17 31 / 96%)",
  color: "var(--flow-ink, #eef4ff)",
};

const dialogMobileStyle: CSSProperties = {
  ...dialogStyle,
  width: "100%",
  maxHeight: "none",
  height: "100%",
  marginBlockStart: 0,
  borderRadius: "0.75rem 0.75rem 0 0",
  alignSelf: "end",
};

const eyebrowStyle: CSSProperties = {
  margin: 0,
  color: "var(--flow-control, #68a8ff)",
  fontFamily: '"IBM Plex Mono", "Cascadia Code", monospace',
  fontSize: "0.75rem",
  letterSpacing: "0.11em",
  textTransform: "uppercase",
};

const titleStyle: CSSProperties = {
  margin: "0.15rem 0 0",
  fontFamily: '"IBM Plex Mono", "Cascadia Code", monospace',
  fontSize: "1.05rem",
  fontWeight: 520,
  letterSpacing: "-0.03em",
};

const inputStyle: CSSProperties = {
  width: "100%",
  minHeight: "2.4rem",
  marginBlockStart: "0.35rem",
  padding: "0.55rem 0.7rem",
  border: "1px solid rgb(104 168 255 / 45%)",
  borderRadius: "0.25rem",
  color: "var(--flow-ink, #eef4ff)",
  background: "var(--flow-plane-raised, #13243a)",
  font: "inherit",
};

const listboxStyle: CSSProperties = {
  margin: 0,
  padding: 0,
  listStyle: "none",
  overflow: "auto",
  minHeight: 0,
  flex: 1,
  border: "1px solid rgb(104 168 255 / 18%)",
  background: "var(--flow-plane, #0c1a2c)",
};

const optionStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1fr) auto",
  gap: "0.35rem 0.75rem",
  alignItems: "start",
  padding: "0.65rem 0.75rem",
  borderBlockEnd: "1px solid rgb(104 168 255 / 12%)",
  cursor: "pointer",
};

const optionActiveStyle: CSSProperties = {
  ...optionStyle,
  background: "rgb(104 168 255 / 14%)",
  outline: "0.12rem solid var(--flow-control, #68a8ff)",
  outlineOffset: "-0.12rem",
};

const optionDisabledStyle: CSSProperties = {
  ...optionStyle,
  opacity: 0.72,
  cursor: "not-allowed",
};

const optionActiveDisabledStyle: CSSProperties = {
  ...optionActiveStyle,
  opacity: 0.72,
  cursor: "not-allowed",
};

const categoryStyle: CSSProperties = {
  margin: 0,
  color: "var(--flow-muted, #91a4be)",
  fontFamily: '"IBM Plex Mono", "Cascadia Code", monospace',
  fontSize: "0.68rem",
  letterSpacing: "0.08em",
  textTransform: "uppercase",
};

const labelStyle: CSSProperties = {
  margin: "0.15rem 0 0",
  fontSize: "0.95rem",
};

const reasonStyle: CSSProperties = {
  margin: "0.25rem 0 0",
  color: "var(--flow-danger, #ff9c9c)",
  fontSize: "0.82rem",
};

const shortcutStyle: CSSProperties = {
  color: "var(--flow-muted, #91a4be)",
  fontFamily: '"IBM Plex Mono", "Cascadia Code", monospace',
  fontSize: "0.75rem",
  whiteSpace: "nowrap",
};

const emptyStyle: CSSProperties = {
  margin: 0,
  padding: "1rem 0.75rem",
  color: "var(--flow-muted, #91a4be)",
};

function optionDomId(prefix: string, commandId: string): string {
  return `${prefix}-${commandId.replace(/[^A-Za-z0-9_-]/gu, "-")}`;
}

function resolveActiveId(
  matches: readonly FlowCommand[],
  activeId: string | null,
): string | null {
  if (matches.length === 0) {
    return null;
  }
  if (activeId !== null && matches.some((command) => command.id === activeId)) {
    return activeId;
  }
  return matches[0]?.id ?? null;
}

function moveActiveId(
  matches: readonly FlowCommand[],
  activeId: string | null,
  delta: number,
): string | null {
  if (matches.length === 0) {
    return null;
  }
  const currentIndex = matches.findIndex((command) => command.id === activeId);
  const start = currentIndex < 0 ? (delta > 0 ? -1 : 0) : currentIndex;
  const nextIndex = (start + delta + matches.length) % matches.length;
  return matches[nextIndex]?.id ?? null;
}

function focusablesWithin(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter(
    (element) =>
      !element.hasAttribute("disabled") &&
      element.getAttribute("aria-hidden") !== "true",
  );
}

function useMobileSheet(): boolean {
  const [mobile, setMobile] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") {
      return;
    }
    const media = window.matchMedia("(width < 720px)");
    const sync = (): void => {
      setMobile(media.matches);
    };
    sync();
    media.addEventListener("change", sync);
    return () => {
      media.removeEventListener("change", sync);
    };
  }, []);

  return mobile;
}

function useFocusTrap(
  open: boolean,
  containerRef: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!open) {
      return;
    }

    function onKeyDown(event: globalThis.KeyboardEvent): void {
      if (event.key !== "Tab") {
        return;
      }
      const container = containerRef.current;
      if (container === null) {
        return;
      }
      const focusables = focusablesWithin(container);
      if (focusables.length === 0) {
        event.preventDefault();
        return;
      }
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (first === undefined || last === undefined) {
        return;
      }
      const active = document.activeElement;
      if (event.shiftKey) {
        if (active === first || !container.contains(active)) {
          event.preventDefault();
          last.focus();
        }
        return;
      }
      if (active === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [containerRef, open]);
}

/**
 * Accessible command palette for Causal Field power actions.
 *
 * Tracks the active command by stable ID (never list index), traps focus while
 * open, and restores the invoking element when closed.
 */
export function CommandConstellation({
  commands,
  open,
  onClose,
  initialQuery = "",
}: CommandConstellationProps): ReactNode {
  const inputId = useId();
  const listboxId = useId();
  const optionIdPrefix = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const openSessionRef = useRef({ commands, initialQuery });
  openSessionRef.current = { commands, initialQuery };
  const mobile = useMobileSheet();
  const [query, setQuery] = useState(initialQuery);
  const [activeId, setActiveId] = useState<string | null>(null);

  const matches = searchCommands(commands, query);
  const resolvedActiveId = resolveActiveId(matches, activeId);
  const activeCommand =
    matches.find((command) => command.id === resolvedActiveId) ?? null;

  useFocusTrap(open, dialogRef);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }

    const { commands: catalog, initialQuery: seed } = openSessionRef.current;
    const active = document.activeElement;
    restoreFocusRef.current =
      active instanceof HTMLElement ? active : null;
    setQuery(seed);
    setActiveId(searchCommands(catalog, seed)[0]?.id ?? null);

    const frame = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });

    return () => {
      cancelAnimationFrame(frame);
      const restore = restoreFocusRef.current;
      restoreFocusRef.current = null;
      if (restore !== null && typeof restore.focus === "function") {
        restore.focus();
      }
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setActiveId((current) =>
      resolveActiveId(searchCommands(commands, query), current),
    );
  }, [commands, open, query]);

  useEffect(() => {
    if (!open || resolvedActiveId === null) {
      return;
    }
    const option = document.getElementById(
      optionDomId(optionIdPrefix, resolvedActiveId),
    );
    option?.scrollIntoView?.({ block: "nearest" });
  }, [open, optionIdPrefix, resolvedActiveId]);

  if (!open) {
    return null;
  }

  function close(): void {
    onClose();
  }

  function executeActive(): void {
    if (activeCommand === null || activeCommand.disabledReason !== undefined) {
      return;
    }
    activeCommand.execute();
    onClose();
  }

  function onDialogKeyDown(event: KeyboardEvent<HTMLDivElement>): void {
    switch (event.key) {
      case "Escape":
        event.preventDefault();
        event.stopPropagation();
        close();
        return;
      case "ArrowDown":
        event.preventDefault();
        setActiveId(moveActiveId(matches, resolvedActiveId, 1));
        return;
      case "ArrowUp":
        event.preventDefault();
        setActiveId(moveActiveId(matches, resolvedActiveId, -1));
        return;
      case "Home":
        event.preventDefault();
        setActiveId(matches[0]?.id ?? null);
        return;
      case "End":
        event.preventDefault();
        setActiveId(matches[matches.length - 1]?.id ?? null);
        return;
      case "Enter":
        if (event.target instanceof HTMLInputElement) {
          event.preventDefault();
          executeActive();
        }
        return;
      default:
        return;
    }
  }

  return (
    <div
      className="aiperf-flow__command-constellation-backdrop"
      style={mobile ? mobileBackdropStyle : backdropStyle}
    >
      <div
        ref={dialogRef}
        aria-label="Command Constellation"
        aria-modal="true"
        className="aiperf-flow__command-constellation"
        data-mobile={mobile ? "true" : "false"}
        onKeyDown={onDialogKeyDown}
        role="dialog"
        style={mobile ? dialogMobileStyle : dialogStyle}
      >
        <header className="aiperf-flow__command-constellation-header">
          <p className="aiperf-flow__eyebrow" style={eyebrowStyle}>
            Command constellation
          </p>
          <h2 style={titleStyle}>
            Jump to a scene, beat, entity, or action
          </h2>
          <label
            className="aiperf-flow__command-constellation-label"
            htmlFor={inputId}
            style={{
              position: "absolute",
              width: 1,
              height: 1,
              padding: 0,
              margin: -1,
              overflow: "hidden",
              clip: "rect(0, 0, 0, 0)",
              whiteSpace: "nowrap",
              border: 0,
            }}
          >
            Search commands
          </label>
          <input
            ref={inputRef}
            aria-activedescendant={
              resolvedActiveId === null
                ? undefined
                : optionDomId(optionIdPrefix, resolvedActiveId)
            }
            aria-autocomplete="list"
            aria-controls={listboxId}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            className="aiperf-flow__command-constellation-input"
            id={inputId}
            onChange={(event) => {
              const nextQuery = event.target.value;
              setQuery(nextQuery);
              const nextMatches = searchCommands(commands, nextQuery);
              setActiveId(nextMatches[0]?.id ?? null);
            }}
            placeholder="Search scenes, beats, entities, evidence…"
            spellCheck={false}
            style={inputStyle}
            type="search"
            value={query}
          />
        </header>

        {matches.length === 0 ? (
          <p
            className="aiperf-flow__command-constellation-empty"
            style={emptyStyle}
          >
            No matching commands.
          </p>
        ) : null}
        <ul
          aria-label="Commands"
          className="aiperf-flow__command-constellation-list"
          hidden={matches.length === 0}
          id={listboxId}
          role="listbox"
          style={listboxStyle}
        >
          {matches.map((command) => {
            const selected = command.id === resolvedActiveId;
            const disabled = command.disabledReason !== undefined;
            const optionId = optionDomId(optionIdPrefix, command.id);
            const reasonId = `${optionId}-reason`;
            let style = optionStyle;
            if (selected && disabled) {
              style = optionActiveDisabledStyle;
            } else if (selected) {
              style = optionActiveStyle;
            } else if (disabled) {
              style = optionDisabledStyle;
            }

            return (
              <li
                key={command.id}
                aria-describedby={disabled ? reasonId : undefined}
                aria-disabled={disabled ? true : undefined}
                aria-selected={selected}
                className="aiperf-flow__command-constellation-option"
                data-active={selected ? "true" : "false"}
                data-category={command.category}
                data-command-id={command.id}
                id={optionId}
                onClick={() => {
                  if (disabled) {
                    setActiveId(command.id);
                    return;
                  }
                  command.execute();
                  onClose();
                }}
                onMouseEnter={() => {
                  setActiveId(command.id);
                }}
                role="option"
                style={style}
              >
                <div>
                  <p style={categoryStyle}>{command.category}</p>
                  <p style={labelStyle}>{command.label}</p>
                  {disabled ? (
                    <p id={reasonId} style={reasonStyle}>
                      {command.disabledReason}
                    </p>
                  ) : null}
                </div>
                {command.shortcut !== undefined ? (
                  <kbd style={shortcutStyle}>{command.shortcut}</kbd>
                ) : null}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
