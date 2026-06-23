/**
 * solomonoff.ts -- Complexity-weighted initialization via the God Formula
 *
 * The Solomonoff layer sets initial v_i (rejection counts) proportional
 * to Kolmogorov complexity estimates. The God Formula then computes:
 *
 *   w_i = R - min(v_i, R) + 1
 *
 * Simpler hypotheses have lower v_i, hence higher w_i (more weight).
 * Complex hypotheses start deeper in the void (less weight), but
 * Law 1 (impossibility of zero) guarantees they are never zeroed out.
 * Even the most complex hypothesis retains the sliver: w_i >= 1.
 *
 * This is the Buleyean analogue of Solomonoff's universal prior:
 * instead of 2^{-K(x)} weighting, we use void initialization
 * proportional to complexity. The God Formula then ensures all seven
 * laws hold from the first observation.
 */

import type { VoidBoundary } from '@a0n/gnosis/src/void';
import { updateVoidBoundary } from '@a0n/gnosis/src/void';

// ============================================================================
// Complexity Estimation
// ============================================================================

/**
 * Estimate relative complexity of a hypothesis.
 * Uses description length as a proxy for Kolmogorov complexity.
 *
 * @param description String description of the hypothesis
 * @returns Complexity score (higher = more complex = more initial void)
 */
export function estimateComplexity(description: string): number {
  // Byte length as crude K(x) proxy
  const byteLength = new TextEncoder().encode(description).length;
  // Normalize to log scale -- complexity grows slowly
  return Math.log2(byteLength + 1);
}

/**
 * Estimate complexity from a structured object.
 * Deeper nesting and more keys = more complex.
 */
export function estimateStructuralComplexity(obj: unknown): number {
  if (obj === null || obj === undefined) return 0;
  if (typeof obj === 'string') return estimateComplexity(obj);
  if (typeof obj === 'number' || typeof obj === 'boolean') return 1;
  if (Array.isArray(obj)) {
    return (
      1 +
      obj.reduce(
        (sum: number, item) => sum + estimateStructuralComplexity(item),
        0
      )
    );
  }
  if (typeof obj === 'object') {
    const entries = Object.entries(obj as Record<string, unknown>);
    return (
      1 +
      entries.reduce(
        (sum, [key, val]) =>
          sum + estimateComplexity(key) + estimateStructuralComplexity(val),
        0
      )
    );
  }
  return 1;
}

// ============================================================================
// Solomonoff Initialization
// ============================================================================

/**
 * Initialize a void boundary with Solomonoff complexity priors.
 * Each dimension gets initial void proportional to its complexity.
 *
 * @param boundary The void boundary to initialize
 * @param complexities Per-dimension complexity scores
 * @param scale Scaling factor for complexity-to-void mapping
 */
export function solomonoffInit(
  boundary: VoidBoundary,
  complexities: number[],
  scale: number = 1.0
): void {
  // Cannon rotation: each dimension's void update is independent
  // (different slot, no cross-dimension dependency).
  // Under gnode --strategy cannon, dimensions distribute across lanes.
  const limit = Math.min(complexities.length, boundary.counts.length);
  for (let i = 0; i < limit; i++) {
    const complexity = complexities[i];
    if (complexity === undefined) {
      throw new Error('solomonoffInit: missing complexity');
    }
    const voidAmount = complexity * scale;
    if (voidAmount > 0) {
      updateVoidBoundary(boundary, i, voidAmount);
    }
  }
}

/**
 * Initialize from string descriptions of each dimension.
 * Shorter/simpler descriptions = less initial void = higher Buleyean probability.
 */
export function solomonoffInitFromDescriptions(
  boundary: VoidBoundary,
  descriptions: string[],
  scale: number = 1.0
): void {
  const complexities = descriptions.map(estimateComplexity);
  solomonoffInit(boundary, complexities, scale);
}

/**
 * Compute the Solomonoff-Buleyean prior distribution.
 * This is what the Buleyean distribution looks like immediately
 * after Solomonoff initialization, before any observations.
 *
 * Returns complexities paired with their resulting Buleyean weights.
 */
export function solomonoffPrior(
  complexities: number[],
  scale: number = 1.0
): { complexity: number; voidInit: number; weight: number }[] {
  const n = complexities.length;
  const voidInits = complexities.map((c) => c * scale);
  const T = voidInits.reduce((a, b) => a + b, 0);

  return complexities.map((c, i) => {
    const voidInit = voidInits[i];
    if (voidInit === undefined) {
      throw new Error('solomonoffPrior: missing void init');
    }
    const w = T - voidInit + 1;
    return { complexity: c, voidInit, weight: w };
  });
}
