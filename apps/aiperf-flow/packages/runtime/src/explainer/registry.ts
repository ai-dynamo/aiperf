import type { ExplainerDefinition } from '@aiperf/flow-compiler';

export class ExplainerRegistry {
  private static decks = new Map<string, ExplainerDefinition>();
  private static routes = new Map<string, string>(); // route -> id

  static register(deck: ExplainerDefinition): void {
    if (this.decks.has(deck.id)) {
      throw new Error(`Explainer with ID "${deck.id}" is already registered (duplicate ID)`);
    }

    if (this.routes.has(deck.route)) {
      throw new Error(`Route "${deck.route}" is already registered (conflict with another deck)`);
    }

    this.decks.set(deck.id, deck);
    this.routes.set(deck.route, deck.id);
  }

  static getDeck(id: string): ExplainerDefinition | undefined {
    return this.decks.get(id);
  }

  static getDeckByRoute(route: string): ExplainerDefinition | undefined {
    const id = this.routes.get(route);
    return id ? this.decks.get(id) : undefined;
  }

  static getAllDecks(): readonly ExplainerDefinition[] {
    return Array.from(this.decks.values());
  }

  static getRouteMap(): Map<string, string> {
    return new Map(this.routes);
  }

  static clear(): void {
    this.decks.clear();
    this.routes.clear();
  }
}
