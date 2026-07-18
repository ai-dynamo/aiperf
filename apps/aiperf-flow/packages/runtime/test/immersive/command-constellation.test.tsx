// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { useState, type ReactNode } from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { CommandConstellation } from "../../src/immersive/command-constellation.js";
import type { FlowCommand } from "../../src/commands.js";

afterEach(cleanup);

function command(overrides: Partial<FlowCommand> = {}): FlowCommand {
  return {
    id: overrides.id ?? "focus-request",
    label: overrides.label ?? "Focus request",
    category: overrides.category ?? "entity",
    keywords: overrides.keywords ?? [],
    ...(overrides.shortcut === undefined ? {} : { shortcut: overrides.shortcut }),
    ...(overrides.disabledReason === undefined
      ? {}
      : { disabledReason: overrides.disabledReason }),
    execute: overrides.execute ?? vi.fn(),
  };
}

function catalog(): readonly FlowCommand[] {
  return [
    command({ id: "scene:play", label: "Play scene", category: "scene" }),
    command({ id: "scene:pause", label: "Pause playback", category: "scene" }),
    command({
      id: "beat:first-token",
      label: "Jump to first token",
      category: "beat",
      shortcut: "G then T",
    }),
    command({
      id: "accessibility:twin",
      label: "Open semantic twin",
      category: "accessibility",
      keywords: ["screen reader", "table"],
    }),
  ];
}

function searchbox(): HTMLInputElement {
  return screen.getByLabelText("Search commands") as HTMLInputElement;
}

function selectedOptionLabels(): readonly string[] {
  return screen
    .getAllByRole("option")
    .filter((option) => option.getAttribute("aria-selected") === "true")
    .map((option) => option.textContent ?? "");
}

// Controlled host so focus can be asserted against a real invoking button.
function TriggerHarness({
  commands,
}: Readonly<{ commands: readonly FlowCommand[] }>): ReactNode {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)} type="button">
        Open constellation
      </button>
      <CommandConstellation
        commands={commands}
        onClose={() => setOpen(false)}
        open={open}
      />
    </>
  );
}

describe("CommandConstellation dialog", () => {
  test("renders an accessible dialog with a labelled searchbox and listbox", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);

    const dialog = screen.getByRole("dialog", { name: "Command Constellation" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(searchbox()).not.toBeNull();
    expect(screen.getByRole("listbox", { name: "Commands" })).not.toBeNull();
    expect(screen.getAllByRole("option")).toHaveLength(4);
  });

  test("renders nothing while closed", () => {
    const { container } = render(
      <CommandConstellation commands={catalog()} onClose={vi.fn()} open={false} />,
    );

    expect(container.firstChild).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  test("marks the first authored match active on open", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);

    expect(selectedOptionLabels()).toEqual(["scenePlay scene"]);
    expect(searchbox().getAttribute("aria-activedescendant")).not.toBeNull();
  });
});

describe("CommandConstellation keyboard navigation", () => {
  test("ArrowDown and ArrowUp move the active command by stable identity", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);
    const input = searchbox();

    fireEvent.keyDown(input, { key: "ArrowDown" });
    expect(selectedOptionLabels()).toEqual(["scenePause playback"]);

    fireEvent.keyDown(input, { key: "ArrowUp" });
    expect(selectedOptionLabels()).toEqual(["scenePlay scene"]);
  });

  test("Home and End jump to the first and last matches", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);
    const input = searchbox();

    fireEvent.keyDown(input, { key: "End" });
    expect(selectedOptionLabels()).toEqual(["accessibilityOpen semantic twin"]);

    fireEvent.keyDown(input, { key: "Home" });
    expect(selectedOptionLabels()).toEqual(["scenePlay scene"]);
  });

  test("ArrowUp from the first match wraps to the last", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);

    fireEvent.keyDown(searchbox(), { key: "ArrowUp" });
    expect(selectedOptionLabels()).toEqual(["accessibilityOpen semantic twin"]);
  });
});

describe("CommandConstellation activation", () => {
  test("Enter executes the active command and closes the dialog", () => {
    const execute = vi.fn();
    const onClose = vi.fn();
    const commands = [
      command({ id: "scene:play", label: "Play scene", execute }),
      ...catalog().slice(1),
    ];
    render(<CommandConstellation commands={commands} onClose={onClose} open />);

    fireEvent.keyDown(searchbox(), { key: "Enter" });

    expect(execute).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("clicking an option executes that command and closes the dialog", () => {
    const execute = vi.fn();
    const onClose = vi.fn();
    const commands = [
      catalog()[0] as FlowCommand,
      command({ id: "beat:first-token", label: "Jump to first token", execute }),
    ];
    render(<CommandConstellation commands={commands} onClose={onClose} open />);

    fireEvent.click(screen.getByRole("option", { name: /Jump to first token/u }));

    expect(execute).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("Escape closes without executing any command", () => {
    const execute = vi.fn();
    const onClose = vi.fn();
    const commands = [command({ id: "scene:play", label: "Play scene", execute })];
    render(<CommandConstellation commands={commands} onClose={onClose} open />);

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(execute).not.toHaveBeenCalled();
  });
});

describe("CommandConstellation disabled commands", () => {
  test("describes the disabled reason and refuses execution on Enter", () => {
    const execute = vi.fn();
    const onClose = vi.fn();
    const commands = [
      command({
        id: "beat:first-token",
        label: "Jump to first token",
        disabledReason: "No first-token beat in this scene",
        execute,
      }),
    ];
    render(<CommandConstellation commands={commands} onClose={onClose} open />);

    const option = screen.getByRole("option", { name: /Jump to first token/u });
    expect(option.getAttribute("aria-disabled")).toBe("true");
    expect(
      within(option).getByText("No first-token beat in this scene"),
    ).not.toBeNull();

    fireEvent.keyDown(searchbox(), { key: "Enter" });
    expect(execute).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  test("clicking a disabled option neither executes nor closes", () => {
    const execute = vi.fn();
    const onClose = vi.fn();
    const commands = [
      command({
        id: "beat:first-token",
        label: "Jump to first token",
        disabledReason: "No first-token beat in this scene",
        execute,
      }),
    ];
    render(<CommandConstellation commands={commands} onClose={onClose} open />);

    fireEvent.click(screen.getByRole("option", { name: /Jump to first token/u }));

    expect(execute).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });
});

describe("CommandConstellation filtering", () => {
  test("narrows the listbox as the query changes", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);

    fireEvent.change(searchbox(), { target: { value: "pause" } });

    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(1);
    expect(options[0]?.getAttribute("data-command-id")).toBe("scene:pause");
  });

  test("reports an empty state when nothing matches", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);

    fireEvent.change(searchbox(), { target: { value: "nonexistent" } });

    expect(screen.getByText("No matching commands.")).not.toBeNull();
    expect(screen.queryAllByRole("option")).toHaveLength(0);
  });
});

describe("CommandConstellation focus management", () => {
  test("traps Tab and Shift+Tab within the dialog", () => {
    render(<CommandConstellation commands={catalog()} onClose={vi.fn()} open />);
    const input = searchbox();
    act(() => {
      input.focus();
    });

    const tab = fireEvent.keyDown(document, { key: "Tab" });
    expect(tab).toBe(false);
    expect(document.activeElement).toBe(input);

    const shiftTab = fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(shiftTab).toBe(false);
    expect(document.activeElement).toBe(input);
  });

  test("restores focus to the invoking button after closing", () => {
    render(<TriggerHarness commands={catalog()} />);
    const trigger = screen.getByRole("button", { name: "Open constellation" });
    act(() => {
      trigger.focus();
    });

    fireEvent.click(trigger);
    expect(screen.getByRole("dialog")).not.toBeNull();

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });
});
