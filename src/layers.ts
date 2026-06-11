/**
 * layers.ts -- Four-layer Buleyean system as gnosis BoundaryStack
 *
 * The God Formula w_i = R - min(v_i, R) + 1 operates at every
 * timescale. The four layers are four instances of the same formula
 * running at different R (observation rounds per layer):
 *
 *   Layer 0 (deepest):    Retrocausal  -- R ~ lifetime   (Law 7: terminal constraints)
 *   Layer 1:              Bayesian     -- R ~ weeks      (Law 2: prior-posterior ordering)
 *   Layer 2:              Frequentist  -- R ~ minutes    (Law 5: conservation of counts)
 *   Layer 3 (shallowest): Solomonoff   -- R ~ generational (Law 1: complexity never zeroes out)
 *
 * The 3x3 matrix (conservation × irreversibility × ground state) maps to
 * the inter-layer flows: upward = conservation (truth persists), downward =
 * irreversibility (observations commit), cross-resonance = ground state
 * (retrocausal ↔ solomonoff seek minimum).
 *
 * Law 4 (every observation is a cave): each layer sees a projection of the
 * full multi-timescale reality. The semiotic deficit between layers is the
 * information lost by compressing all timescales through one attention stream.
 */

import {
  type BoundaryStack,
  type TimescaleBoundary,
  createTimescaleBoundary,
  createBoundaryStack,
  createResonance,
  tickBoundaryStack,
  measureStack,
  upwardConstraint,
  downwardContext,
} from '@a0n/gnosis/src/void';

// ============================================================================
// Layer Definitions
// ============================================================================

export const BULEYEAN_LAYERS = [
  {
    name: 'retrocausal',
    timescale: 'lifetime' as const,
    description:
      'Terminal state constrains trajectory -- what must be avoided shapes what is chosen',
  },
  {
    name: 'bayesian',
    timescale: 'weeks' as const,
    description:
      'Prior beliefs updated by evidence -- posterior as complement distribution',
  },
  {
    name: 'frequentist',
    timescale: 'minutes' as const,
    description:
      'Direct observation counting -- the raw void boundary accumulation',
  },
  {
    name: 'solomonoff',
    timescale: 'generational' as const,
    description:
      'Complexity-weighted initialization -- simpler hypotheses start with less void',
  },
] as const;

// ============================================================================
// Stack Creation
// ============================================================================

/**
 * Create the four-layer Buleyean BoundaryStack.
 *
 * @param dimensions Number of dimensions per layer (shared across all four)
 * @param dimensionLabels Optional labels for each dimension
 */
export function createBuleyeanStack(
  dimensions: number,
  dimensionLabels?: readonly string[]
): BoundaryStack {
  const layers: TimescaleBoundary[] = BULEYEAN_LAYERS.map((def) =>
    createTimescaleBoundary(
      def.name,
      def.timescale,
      dimensions,
      dimensionLabels
    )
  );

  // Cross-layer resonance: retrocausal (0) couples directly to solomonoff (3)
  // The terminal constraint resonates with the initial complexity prior
  const resonances = [
    createResonance(
      0,
      3,
      0.08,
      'retrocausal-solomonoff: terminal fate shapes initial complexity prior'
    ),
    createResonance(
      3,
      0,
      0.03,
      'solomonoff-retrocausal: complexity prior influences fate constraints'
    ),
  ];

  return createBoundaryStack('buleyean-four-layer', layers, resonances);
}

/**
 * Get a named layer from the Buleyean stack.
 */
export function getLayer(
  stack: BoundaryStack,
  name: 'retrocausal' | 'bayesian' | 'frequentist' | 'solomonoff'
): TimescaleBoundary {
  const idx = BULEYEAN_LAYERS.findIndex((l) => l.name === name);
  return stack.layers[idx];
}

/**
 * Tick the Buleyean stack: natural decay + upward constraints + downward context.
 * Delegates to gnosis tickBoundaryStack which handles all inter-layer flows.
 */
export function tickBuleyeanStack(stack: BoundaryStack): void {
  tickBoundaryStack(stack);
}

/**
 * Measure the full Buleyean stack state.
 */
export function measureBuleyeanStack(stack: BoundaryStack) {
  return {
    global: measureStack(stack),
    layers: stack.layers.map((layer) => ({
      name: layer.name,
      timescale: layer.timescale,
      totalEntries: layer.boundary.totalEntries,
      counts: [...layer.boundary.counts],
    })),
  };
}

/**
 * Apply the four-layer constraint cascade explicitly.
 * Returns the bias vectors for each layer pair.
 */
export function constraintCascade(stack: BoundaryStack): {
  retrocausalToBayesian: number[];
  bayesianToFrequentist: number[];
  frequentistToSolomonoff: number[];
} {
  return {
    retrocausalToBayesian: upwardConstraint(stack.layers[0], stack.layers[1]),
    bayesianToFrequentist: upwardConstraint(stack.layers[1], stack.layers[2]),
    frequentistToSolomonoff: upwardConstraint(stack.layers[2], stack.layers[3]),
  };
}

/**
 * Apply the four-layer contextualization cascade explicitly.
 * Returns the modulation vectors for each layer pair.
 */
export function contextualizationCascade(stack: BoundaryStack): {
  solomonoffToFrequentist: number[];
  frequentistToBayesian: number[];
  bayesianToRetrocausal: number[];
} {
  return {
    solomonoffToFrequentist: downwardContext(stack.layers[3], stack.layers[2]),
    frequentistToBayesian: downwardContext(stack.layers[2], stack.layers[1]),
    bayesianToRetrocausal: downwardContext(stack.layers[1], stack.layers[0]),
  };
}
