// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useState, type KeyboardEvent } from "react";

import type { Audience } from "../../domain/audience";
import {
  runtimeStory,
  runtimeStoryLength,
  type StoryChapter,
  type StoryConfigMode,
  type StoryTrait,
  type TraitImpl,
} from "./story-content";

interface RuntimeStoryProps {
  /** Drives progressive detail: executive → developer → maintainer. */
  readonly audience: Audience;
}

/** Detail tiers unlocked per audience, so more depth means fewer hidden facts, not more prose. */
function detailFor(audience: Audience): { technical: boolean; evidence: boolean } {
  return {
    technical: audience !== "executive",
    evidence: audience === "maintainer",
  };
}

function TraitFanArrows({ implCount }: { implCount: number }) {
  if (implCount === 0) {
    return null;
  }

  const hubY = 50;
  const bendX = 34;
  const endX = 100;
  const paths = Array.from({ length: implCount }, (_, index) => {
    const targetY = ((index + 0.5) / implCount) * 100;
    return `M 0 ${hubY} H ${bendX} L ${endX} ${targetY}`;
  });

  return (
    <svg
      aria-hidden="true"
      className="story-fan-arrows"
      preserveAspectRatio="none"
      viewBox="0 0 100 100"
    >
      <defs>
        <marker
          id="story-fan-head"
          markerHeight="6"
          markerUnits="strokeWidth"
          markerWidth="6"
          orient="auto"
          refX="5"
          refY="3"
        >
          <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" />
        </marker>
      </defs>
      {paths.map((path, pathIndex) => (
        <path
          key={pathIndex}
          className="story-fan-path"
          d={path}
          fill="none"
          markerEnd="url(#story-fan-head)"
        />
      ))}
    </svg>
  );
}

function TraitFan({
  trait,
  audience,
}: {
  trait: StoryTrait;
  audience: Audience;
}) {
  const detail = detailFor(audience);
  return (
    <div className="story-fan">
      <div className="story-hub">
        <span className="story-hub-kind">{trait.kind}</span>
        <strong className="story-hub-name">{trait.name}</strong>
        {detail.technical ? (
          <code className="story-hub-sig">{trait.signature}</code>
        ) : null}
      </div>
      <TraitFanArrows implCount={trait.impls.length} />
      <ul className="story-impls" aria-label={`${trait.name} implementations`}>
        {trait.impls.map((impl: TraitImpl) => (
          <li className="story-impl" key={impl.name}>
            <div className="story-impl-head">
              <strong className="story-impl-name">{impl.name}</strong>
              <span className="story-impl-tag">{impl.tag}</span>
            </div>
            {detail.technical ? (
              <span className="story-impl-crate">{impl.crate}</span>
            ) : null}
            {detail.evidence ? (
              <span className="story-impl-note">{impl.note}</span>
            ) : null}
            {detail.evidence ? (
              <span className="story-impl-path">
                crates/{impl.crate}/src/{impl.file}
              </span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ConfigModeCompare({
  modes,
  audience,
}: {
  modes: readonly StoryConfigMode[];
  audience: Audience;
}) {
  const detail = detailFor(audience);
  return (
    <div className="story-config-compare">
      <div className="story-hub story-config-hub">
        <span className="story-hub-kind">authoring</span>
        <strong className="story-hub-name">Config v2</strong>
        {detail.technical ? (
          <code className="story-hub-sig">Python CLI + YAML file</code>
        ) : null}
      </div>
      <TraitFanArrows implCount={modes.length} />
      <ul className="story-config-modes" aria-label="Config v2 authoring modes">
        {modes.map((mode) => (
          <li className="story-config-mode" data-mode={mode.id} key={mode.id}>
            <div className="story-impl-head">
              <strong className="story-impl-name">{mode.label}</strong>
              <span className="story-impl-tag">{mode.tag}</span>
            </div>
            <pre className="story-config-snippet">
              <code>{mode.lines.join("\n")}</code>
            </pre>
          </li>
        ))}
      </ul>
    </div>
  );
}

function NarrativeFigure({
  chapter,
  index,
}: {
  chapter: StoryChapter;
  index: number;
}) {
  const previous = index > 0 ? runtimeStory[index - 1] : null;
  const next = index < runtimeStoryLength - 1 ? runtimeStory[index + 1] : null;
  return (
    <div className="story-focus">
      {previous ? (
        <span className="story-focus-node is-adjacent">{previous.title}</span>
      ) : (
        <span className="story-focus-node is-origin">Config v2</span>
      )}
      <span className="story-focus-arrow" aria-hidden="true" />
      <span className="story-focus-node is-current" data-accent={chapter.accent}>
        {chapter.title}
      </span>
      <span className="story-focus-arrow" aria-hidden="true" />
      {next ? (
        <span className="story-focus-node is-adjacent">{next.title}</span>
      ) : (
        <span className="story-focus-node is-origin">Native report</span>
      )}
    </div>
  );
}

export function RuntimeStory({ audience }: RuntimeStoryProps) {
  const [index, setIndex] = useState(0);
  const reduceMotion = useReducedMotion();
  const chapter = runtimeStory[index];
  const detail = detailFor(audience);

  const goTo = useCallback((target: number) => {
    setIndex(Math.max(0, Math.min(runtimeStoryLength - 1, target)));
  }, []);

  useEffect(() => {
    setIndex((current) => Math.min(current, runtimeStoryLength - 1));
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      switch (event.key) {
        case "ArrowRight":
        case "PageDown":
          event.preventDefault();
          goTo(index + 1);
          break;
        case "ArrowLeft":
        case "PageUp":
          event.preventDefault();
          goTo(index - 1);
          break;
        case "Home":
          event.preventDefault();
          goTo(0);
          break;
        case "End":
          event.preventDefault();
          goTo(runtimeStoryLength - 1);
          break;
        default:
          break;
      }
    },
    [goTo, index],
  );

  const atStart = index === 0;
  const atEnd = index === runtimeStoryLength - 1;

  return (
    <section
      aria-label="Guided runtime flow"
      aria-roledescription="carousel"
      className="runtime-story"
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      <ol className="story-rail" aria-label="Runtime flow steps">
        {runtimeStory.map((step, stepIndex) => (
          <li className="story-rail-item" key={step.id}>
            <button
              aria-current={stepIndex === index ? "step" : undefined}
              className="story-rail-dot"
              data-accent={step.accent}
              data-state={
                stepIndex === index
                  ? "active"
                  : stepIndex < index
                    ? "done"
                    : "upcoming"
              }
              onClick={() => goTo(stepIndex)}
              type="button"
            >
              <span className="story-rail-index">{stepIndex + 1}</span>
              <span className="story-rail-label">{step.title}</span>
            </button>
          </li>
        ))}
      </ol>

      <motion.article
        animate={{ opacity: 1, y: 0 }}
        className="story-stage"
        data-accent={chapter.accent}
        initial={reduceMotion ? false : { opacity: 0, y: 12 }}
        key={chapter.id}
        transition={reduceMotion ? { duration: 0 } : { duration: 0.32, ease: "easeOut" }}
      >
        <div className="story-brief">
          <p className="story-kicker">
            <span className="story-kicker-chapter">
              {String(index + 1).padStart(2, "0")} / {String(runtimeStoryLength).padStart(2, "0")}
            </span>
            <span className="story-kicker-text">{chapter.kicker}</span>
          </p>
          <h1 className="story-title" aria-live="polite">
            {chapter.title}
          </h1>
          <p className="story-blurb">{chapter.blurb}</p>
          {!chapter.trait && detail.evidence && chapter.evidence ? (
            <p className="story-brief-evidence">{chapter.evidence}</p>
          ) : null}
        </div>

        <div className="story-figure">
          {chapter.trait ? (
            <TraitFan trait={chapter.trait} audience={audience} />
          ) : chapter.configModes ? (
            <ConfigModeCompare modes={chapter.configModes} audience={audience} />
          ) : (
            <NarrativeFigure chapter={chapter} index={index} />
          )}
        </div>
      </motion.article>

      <nav className="story-controls" aria-label="Flow navigation">
        <button
          className="story-nav-button"
          disabled={atStart}
          onClick={() => goTo(index - 1)}
          type="button"
        >
          <span aria-hidden="true">&larr;</span> Back
        </button>
        <p className="story-progress" role="status">
          Chapter {index + 1} of {runtimeStoryLength}
        </p>
        <button
          className="story-nav-button story-nav-button-primary"
          disabled={atEnd}
          onClick={() => goTo(index + 1)}
          type="button"
        >
          Next <span aria-hidden="true">&rarr;</span>
        </button>
      </nav>
    </section>
  );
}
