// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode } from "react";
import type { SceneCardInfo } from "./fixture";

type SceneCardProps = Readonly<{
  scene: SceneCardInfo;
  onSelect(flowId: string, sceneId: string): void;
}>;

function SceneCard({ scene, onSelect }: SceneCardProps): ReactNode {
  return (
    <button
      className="scene-card"
      onClick={() => onSelect(scene.flowId, scene.sceneId)}
      type="button"
      aria-label={`Load ${scene.title} scene`}
    >
      <div className="scene-card-header">
        <h3 className="scene-card-title">{scene.title}</h3>
        <span className="scene-card-kicker">{scene.flowTitle}</span>
      </div>
      <p className="scene-card-description">{scene.description}</p>
      <span className="scene-card-badge">View scene</span>
    </button>
  );
}

type HomePageProps = Readonly<{
  scenesByFlow: readonly Readonly<{
    flowId: string;
    flowTitle: string;
    scenes: readonly SceneCardInfo[];
  }>[];
  onSelectScene(flowId: string, sceneId: string): void;
  onOpenExplainers(): void;
}>;

/**
 * Home page displaying all available Flow scenes and decks.
 * Organized by flow with clickable scene cards.
 */
export function HomePage({ scenesByFlow, onSelectScene, onOpenExplainers }: HomePageProps): ReactNode {
  const totalScenes = scenesByFlow.reduce(
    (sum, flow) => sum + flow.scenes.length,
    0,
  );

  return (
    <div className="home-page">
      <header className="home-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>AIPerf Flow Scenes</h1>
            <p className="home-subtitle">
              Explore {scenesByFlow.length} flows with {totalScenes} interactive scenes
            </p>
          </div>
          <button
            onClick={onOpenExplainers}
            type="button"
            aria-label="Open explainer decks"
            style={{
              padding: '0.75rem 1.5rem',
              background: 'var(--preview-signal, #3fb950)',
              color: 'var(--preview-board, #0d1117)',
              border: 'none',
              borderRadius: '6px',
              fontSize: '0.95rem',
              fontWeight: 600,
              cursor: 'pointer',
              textTransform: 'uppercase',
              letterSpacing: '0.03em',
              transition: 'all 0.2s ease',
              whiteSpace: 'nowrap',
              marginLeft: '1rem',
            }}
            onMouseEnter={(e) => {
              const target = e.currentTarget as HTMLButtonElement;
              target.style.opacity = '0.9';
              target.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={(e) => {
              const target = e.currentTarget as HTMLButtonElement;
              target.style.opacity = '1';
              target.style.transform = 'translateY(0)';
            }}
          >
            📚 Explainers
          </button>
        </div>
      </header>

      <div className="flows-container">
        {scenesByFlow.map((flow) => (
          <section key={flow.flowId} className="flow-section">
            <h2 className="flow-title">{flow.flowTitle}</h2>
            <div className="scene-cards-grid">
              {flow.scenes.map((scene) => (
                <SceneCard
                  key={scene.sceneId}
                  scene={scene}
                  onSelect={onSelectScene}
                />
              ))}
            </div>
          </section>
        ))}
      </div>

      <style>{`
        .home-page {
          padding: 2rem;
          background: var(--preview-board, #0d1117);
          color: var(--preview-chalk, #f0f6fc);
          min-height: 100vh;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .home-header {
          max-width: 1200px;
          margin: 0 auto 3rem;
          padding-bottom: 2rem;
          border-bottom: 1px solid var(--preview-guide, #30363d);
        }

        .home-header > div {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 2rem;
        }

        .home-header > div > div {
          flex: 1;
          text-align: center;
        }

        .home-header h1 {
          font-size: 2.5rem;
          font-weight: 700;
          margin: 0 0 0.5rem;
          letter-spacing: -0.02em;
        }

        .home-subtitle {
          font-size: 1.1rem;
          color: var(--preview-muted, #8b949e);
          margin: 0;
        }

        .flows-container {
          max-width: 1200px;
          margin: 0 auto;
        }

        .flow-section {
          margin-bottom: 3rem;
        }

        .flow-title {
          font-size: 1.5rem;
          font-weight: 600;
          margin: 0 0 1.5rem;
          color: var(--preview-chalk, #f0f6fc);
          padding-bottom: 0.75rem;
          border-bottom: 2px solid var(--preview-signal, #3fb950);
        }

        .scene-cards-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 1.5rem;
        }

        .scene-card {
          all: unset;
          cursor: pointer;
          padding: 1.5rem;
          background: var(--preview-panel, #161b22);
          border: 1px solid var(--preview-guide, #30363d);
          border-radius: 8px;
          transition: all 0.2s ease;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          text-align: left;
        }

        .scene-card:hover {
          border-color: var(--preview-signal, #3fb950);
          background: var(--preview-raised, #21262d);
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .scene-card:focus-visible {
          outline: 2px solid var(--preview-signal, #3fb950);
          outline-offset: 2px;
        }

        .scene-card-header {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .scene-card-title {
          font-size: 1.2rem;
          font-weight: 600;
          margin: 0;
          color: var(--preview-chalk, #f0f6fc);
        }

        .scene-card-kicker {
          font-size: 0.8rem;
          font-weight: 500;
          color: var(--preview-signal, #3fb950);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .scene-card-description {
          font-size: 0.95rem;
          color: var(--preview-muted, #8b949e);
          margin: 0;
          line-height: 1.5;
          flex: 1;
        }

        .scene-card-badge {
          align-self: flex-start;
          padding: 0.4rem 0.8rem;
          background: var(--preview-signal, #3fb950);
          color: var(--preview-board, #0d1117);
          font-size: 0.85rem;
          font-weight: 600;
          border-radius: 4px;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }

        @media (max-width: 860px) {
          .home-header h1 {
            font-size: 2rem;
          }

          .home-page {
            padding: 1rem;
          }

          .home-header {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
          }

          .home-header > div {
            flex-direction: column;
            align-items: center;
          }

          .home-header > div > div {
            width: 100%;
          }

          .scene-cards-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
          }

          .flow-section {
            margin-bottom: 2rem;
          }
        }
      `}</style>
    </div>
  );
}
