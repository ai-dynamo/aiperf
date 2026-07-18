/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it, vi } from "vitest";
import {
  deckDefinitionsFromPackageModules,
  isDeckPackageModule,
  loadDeckPackages,
  type DeckPackageModule,
} from "../core/load-deck-packages";
import type { DeckDefinition, SlideDefinition } from "../core/types";

const slide: SlideDefinition = {
  eyebrow: "Overview",
  title: "Title",
  lede: "Lede",
  narration: "Narration for the slide.",
  points: ["Point one"],
  caption: "Caption",
};

function stubDeck(id: string, route: string): DeckDefinition {
  return {
    id,
    route,
    storagePrefix: `${id}-storage`,
    classPrefix: id,
    eyebrowLabel: id.toUpperCase(),
    startGateTitle: `${id} gate`,
    hub: {
      title: "from scratch",
      highlight: id,
      description: `${id} description`,
    },
    slides: [slide],
    MentalModel: () => null,
    css: "",
  };
}

function validPackage(id: string, route: string): DeckPackageModule {
  return {
    schemaVersion: 1,
    id,
    route,
  };
}

describe("isDeckPackageModule", () => {
  it("accepts schemaVersion 1 packages with id and route", () => {
    expect(isDeckPackageModule(validPackage("rust-architecture", "/rust-architecture"))).toBe(
      true,
    );
  });

  it("rejects missing or invalid packages", () => {
    expect(isDeckPackageModule(null)).toBe(false);
    expect(isDeckPackageModule({ schemaVersion: 2, id: "x", route: "/x" })).toBe(false);
    expect(isDeckPackageModule({ schemaVersion: 1, id: "", route: "/x" })).toBe(false);
    expect(isDeckPackageModule({ schemaVersion: 1, id: "x", route: "" })).toBe(false);
  });
});

describe("deckDefinitionsFromPackageModules", () => {
  it("unwraps default exports, sorts by path, and converts via packageToDeckDefinition", () => {
    const packageToDeckDefinition = vi.fn((pkg: DeckPackageModule) =>
      stubDeck(pkg.id, pkg.route),
    );

    const decks = deckDefinitionsFromPackageModules(
      {
        "../decks-generated/slurm-velo.package.json": {
          default: validPackage("slurm-velo", "/slurm-velo"),
        },
        "../decks-generated/rust-architecture.package.json": validPackage(
          "rust-architecture",
          "/rust-architecture",
        ),
      },
      packageToDeckDefinition,
    );

    expect(packageToDeckDefinition).toHaveBeenCalledTimes(2);
    expect(decks.map((deck) => deck.id)).toEqual(["rust-architecture", "slurm-velo"]);
    expect(decks[0]?.route).toBe("/rust-architecture");
    expect(decks[1]?.route).toBe("/slurm-velo");
  });

  it("throws when a module is not a DeckPackage", () => {
    expect(() =>
      deckDefinitionsFromPackageModules(
        {
          "../decks-generated/bad.package.json": { schemaVersion: 1, id: "bad" },
        },
        (pkg) => stubDeck(pkg.id, pkg.route),
      ),
    ).toThrow(/Invalid DeckPackage module/);
  });
});

describe("loadDeckPackages", () => {
  it("returns an empty list when decks-generated has no packages yet", () => {
    const packageToDeckDefinition = vi.fn((pkg: DeckPackageModule) =>
      stubDeck(pkg.id, pkg.route),
    );

    expect(loadDeckPackages(packageToDeckDefinition)).toEqual([]);
    expect(packageToDeckDefinition).not.toHaveBeenCalled();
  });
});
