export interface ImmersiveScene {
  layout: { fullViewport: boolean };
  overlayContent: {
    narrationUI: { position: string };
    title: string;
  };
}

export interface ImmersiveControls {
  playButton: { label: string };
  speedControl: { speeds: number[] };
  causalTraceToggle: { label: string };
}

export class ImmersiveExplainerContext {
  expandSlideToViewport(sceneId: string): ImmersiveScene {
    return {
      layout: { fullViewport: true },
      overlayContent: {
        narrationUI: {
          position: 'top-right',
        },
        title: `Exploring ${sceneId}`,
      },
    };
  }

  applyImmersiveControls(): ImmersiveControls {
    return {
      playButton: { label: 'Play' },
      speedControl: { speeds: [0.5, 1, 1.5, 2] },
      causalTraceToggle: { label: 'Show Causal Trace' },
    };
  }
}
