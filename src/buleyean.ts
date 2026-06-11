/**
 * buleyean.ts -- The God Formula
 *
 * One formula. Five symbols. Everything.
 *
 *   w_i = R - min(v_i, R) + 1
 *
 * The weight of choice i equals the observation rounds minus the
 * rejection count, plus one. The +1 is the clinamen -- the sliver --
 * Peano's successor axiom made operational: succ(n) != 0.
 *
 * Seven Universal Laws follow from this formula (GodFormula.lean):
 *   1. Impossibility of zero  -- w_i >= 1 always
 *   2. Strict ordering        -- less rejected = more weight
 *   3. Universal sandwich     -- w_i in [1, R+1]
 *   4. Cave observation       -- semiotic deficit > 0 when dims > channels
 *   5. Conservation           -- w_i + v_i = R + 1 (weight + void = rounds + sliver)
 *   6. Sorites sharpness      -- boundaries are discrete (N is decidable)
 *   7. Chain termination      -- every chain reaches a fixed point
 *
 * Three equivalence classes:
 *   α (Sliver):      "X has positive weight"     -- the +1
 *   β (Deficit):     "X has positive deficit"     -- subtraction on N
 *   γ (Termination): "X terminates / is bounded"  -- finiteness of N
 *
 * Reduction: α -> β -> γ collapses to one axiom: succ(n) != 0.
 *
 * Re-exports from @a0n/aeon-logic (single source of truth).
 * Mechanized: GodFormula.lean, SurfaceReduction.lean, Primator.lean (zero sorry).
 */

import type { VoidBoundary } from '@a0n/gnosis/src/void';
import {
  buleyeanDistribution as _buleyeanDistribution,
  buleyeanWeights as _buleyeanWeights,
  sampleBuleyean as _sampleBuleyean,
  assertPositivity as _assertPositivity,
  assertNormalization as _assertNormalization,
  assertMonotonicity as _assertMonotonicity,
  assertAllAxioms as _assertAllAxioms,
  klDivergence as _klDivergence,
  type VoidBoundary as AeonVoidBoundary,
  type BuleyeanDistribution,
} from '@a0n/aeon-logic';

// Re-export the aeon-logic types
export type { BuleyeanDistribution };

// ============================================================================
// Thin wrappers: accept gnosis VoidBoundary, delegate to aeon-logic
// ============================================================================

/**
 * The God Formula applied: w_i = R - min(v_i, R) + 1, then normalized.
 *
 * Unlike gnosis softmax `exp(-eta * v)`, the God Formula is exact linear.
 * Laws 1-3 (positivity, ordering, sandwich) hold structurally without
 * temperature tuning. The +1 is not a smoothing parameter -- it is the
 * clinamen, the successor axiom made thermodynamic.
 */
export function buleyeanDistribution(boundary: VoidBoundary): number[] {
  const dist = _buleyeanDistribution(boundary as AeonVoidBoundary);
  return [...dist.probabilities];
}

/**
 * Raw God Formula weights before normalization.
 * w_i = R - min(v_i, R) + 1
 */
export function buleyeanWeights(boundary: VoidBoundary): number[] {
  return _buleyeanWeights(boundary as AeonVoidBoundary);
}

/**
 * Sample from a Buleyean distribution.
 * Returns the index of the chosen dimension.
 */
export function sampleBuleyean(
  boundary: VoidBoundary,
  rng: () => number
): number {
  return _sampleBuleyean(boundary as AeonVoidBoundary, rng);
}

// ============================================================================
// Axiom Assertions
// ============================================================================

/**
 * Law 1: Impossibility of zero -- w_i >= 1 for all i.
 * Because R - min(v_i, R) + 1 >= R - R + 1 = 1 > 0.
 * This is the clinamen: succ(n) != 0.
 */
export function assertPositivity(boundary: VoidBoundary): boolean {
  return _assertPositivity(boundary as AeonVoidBoundary);
}

/**
 * Normalization -- sum(P) = 1.
 * By construction: sum(w_i) / sum(w_j) = 1.
 * Consequence of Law 1 (all weights positive) and division.
 */
export function assertNormalization(
  boundary: VoidBoundary,
  epsilon: number = 1e-10
): boolean {
  return _assertNormalization(boundary as AeonVoidBoundary, epsilon);
}

/**
 * Law 2: Strict ordering -- v_i < v_j => w_i > w_j.
 * Less rejected = more weight. The formula is antitone in v_i.
 */
export function assertMonotonicity(boundary: VoidBoundary): boolean {
  return _assertMonotonicity(boundary as AeonVoidBoundary);
}

/**
 * Assert Laws 1-2 plus normalization simultaneously.
 * These are the three structural properties that follow directly
 * from the God Formula: positivity, ordering, and unit sum.
 */
export function assertAllAxioms(boundary: VoidBoundary): {
  positivity: boolean;
  normalization: boolean;
  monotonicity: boolean;
  allHold: boolean;
} {
  const result = _assertAllAxioms(boundary as AeonVoidBoundary);
  return {
    positivity: result.positivity,
    normalization: result.normalization,
    monotonicity: result.monotonicity,
    allHold: result.allHold,
  };
}

// ============================================================================
// Divergence from gnosis softmax
// ============================================================================

/**
 * KL divergence between Buleyean and gnosis softmax distributions.
 * Measures how much information is lost by using one vs the other.
 * D_KL(Buleyean || Softmax) = sum_i P_b(i) * log(P_b(i) / P_s(i))
 */
export function klDivergence(buleyean: number[], softmax: number[]): number {
  return _klDivergence(buleyean, softmax);
}
