import { Navigate, useParams } from "react-router-dom";
import { ExplainerShell } from "../core/ExplainerShell";
import { deckByRoute } from "../core/deck-registry";

export function DeckRoute() {
  const { deckId } = useParams<{ deckId: string }>();
  const route = deckId ? `/${deckId}` : "/";
  const deck = deckByRoute(route);

  if (!deck) {
    return <Navigate to="/" replace />;
  }

  return <ExplainerShell deck={deck} />;
}
