import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import { ThemeProvider } from "./core/ui";
import { Hub } from "./routes/Hub";
import { DeckRoute } from "./routes/DeckRoute";

export function App() {
  return (
    <ThemeProvider>
      <HashRouter>
        <Routes>
          <Route path="/" element={<Hub />} />
          <Route path="/:deckId" element={<DeckRoute />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HashRouter>
    </ThemeProvider>
  );
}
