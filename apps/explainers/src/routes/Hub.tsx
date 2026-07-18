import { Link } from "react-router-dom";
import { DECK_REGISTRY } from "../core/deck-registry";
import { useHostTheme } from "../core/ui";

export function Hub() {
  const t = useHostTheme();

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "48px 24px" }}>
      <div
        style={{
          color: t.text.secondary,
          fontSize: 13,
          fontWeight: 650,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 10,
        }}
      >
        AIPerf · Explainers
      </div>
      <h1 style={{ margin: "0 0 12px", fontSize: 32, lineHeight: 1.15, color: t.text.primary }}>
        Interactive walkthroughs
      </h1>
      <p style={{ color: t.text.secondary, fontSize: 17, lineHeight: 1.55, margin: "0 0 28px" }}>
        Short, narrated slideshows that explain how AIPerf pieces fit together. Pick a deck to start.
      </p>
      <div style={{ display: "grid", gap: 12 }}>
        {DECK_REGISTRY.map((deck) => (
          <Link
            key={deck.id}
            to={deck.route}
            style={{
              display: "block",
              textDecoration: "none",
              color: "inherit",
              background: t.bg.elevated,
              border: `1px solid ${t.stroke.secondary}`,
              borderRadius: 10,
              padding: "18px 18px 16px",
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>
              <span style={{ color: t.category.green }}>{deck.hub.highlight}</span>{" "}
              {deck.hub.title}
            </div>
            <div style={{ color: t.text.secondary, fontSize: 15, lineHeight: 1.5 }}>
              {deck.hub.description}
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}
