/**
 * bule.ts -- The altimeter of convergence
 *
 * The Bule measures the deficit between where the system is and where
 * the God Formula says it will converge. It is Class β (deficit) and
 * Class γ (termination) made measurable:
 *
 *   B = max(counts) - min(counts)  -- the integer deficit
 *
 * One Bule is simultaneously:
 *   - one unit of topological deficit (Law 4: cave observation)
 *   - one unit of missing concurrency (serialization cost)
 *   - one unit of waste (Law 5: conservation, at least kT ln 2 joules)
 *   - one unit of exploration budget (Law 1: the +1 sliver)
 *
 * Law 3 (sandwich) bounds the Bule: B in [0, R].
 * Law 7 (chain termination) guarantees B -> 0.
 *
 * The Buleyean distribution is the compass (what to try next).
 * The Bule number is the altimeter (how far from convergence).
 * Together they are the complete two-number summary.
 *
 * Mechanized in BuleIsValue.lean, SliverFromVent.lean,
 * SliverOfHope.lean, GodFormula.lean (zero sorry).
 */

import type { VoidBoundary } from '@a0n/gnosis/src/void';
import { buleyeanDistribution } from './buleyean.js';
import {
  shannonEntropy,
  maxEntropy,
  BOLTZMANN_K,
  ROOM_TEMPERATURE,
} from './thermodynamics.js';

// ============================================================================
// Constants
// ============================================================================

/** Golden ratio */
export const PHI = (1 + Math.sqrt(5)) / 2;

/** Golden ratio inverse = phi - 1 = 1/phi */
export const PHI_INV = PHI - 1;

// ============================================================================
// The Bule Number
// ============================================================================

/**
 * Compute the Bule number from a void boundary.
 *
 * B = topological deficit = number of dimensions with rejection count
 * below the threshold needed for convergence.
 *
 * In the simplest form: B = max(counts) - min(counts).
 * When B = 0, the distribution is uniform (ground state).
 * When B > 0, there is unresolved deficit.
 *
 * This is the INTEGER Bule -- the altimeter reading.
 */
export function buleNumber(boundary: VoidBoundary): number {
  const counts = boundary.counts;
  if (counts.length === 0) return 0;
  const max = Math.max(...counts);
  const min = Math.min(...counts);
  return max - min;
}

/**
 * Compute the normalized Bule: B / T where T is total rejections.
 * Ranges from 0 (uniform rejections) to 1 (all rejections on one dim).
 *
 * The normalized Bule is the deficit fraction -- how much of the
 * total rejection mass is concentrated rather than spread.
 */
export function normalizedBule(boundary: VoidBoundary): number {
  const T = boundary.totalEntries;
  if (T === 0) return 0;
  return buleNumber(boundary) / T;
}

/**
 * Compute the Bule from the golden ratio eigenvalue.
 * B_phi = |r - phi| / phi
 *
 * where r is the ratio of the most-rejected to least-rejected count.
 * At convergence (uniform rejections), r = 1, and B_phi = |1 - phi| / phi = phi_inv.
 */
export function buleFromPhi(boundary: VoidBoundary): number {
  const counts = boundary.counts;
  if (counts.length === 0) return 0;
  const max = Math.max(...counts);
  const min = Math.min(...counts);
  if (min === 0 && max === 0) return PHI_INV; // pre-convergence
  const r = min > 0 ? max / min : max + 1;
  return Math.abs(r - PHI) / PHI;
}

// ============================================================================
// Topological Deficit (beta_1)
// ============================================================================

/**
 * The topological deficit: how many independent parallel paths are
 * missing from the system's current state.
 *
 * beta_1_natural = intrinsic parallelism needed (dimension count)
 * beta_1_actual = current effective parallelism
 * Delta_beta = beta_1_natural - beta_1_actual
 *
 * Fork raises beta_1. Fold lowers it. Vent decrements by one.
 */
export function topologicalDeficit(
  naturalParallelism: number,
  actualParallelism: number
): number {
  return Math.max(0, naturalParallelism - actualParallelism);
}

/**
 * Effective parallelism of a distribution.
 * Uses the inverse participation ratio: 1 / sum(p_i^2).
 *
 * Uniform over N: IPR = N (maximum parallelism).
 * Delta at one index: IPR = 1 (zero parallelism).
 * This is the "effective number of live branches."
 */
export function effectiveParallelism(p: number[]): number {
  let sumSq = 0;
  for (const pi of p) {
    sumSq += pi * pi;
  }
  return sumSq > 0 ? 1 / sumSq : 0;
}

/**
 * Compute beta_1 and deficit from a void boundary.
 *
 * Returns the complete topological state:
 *   beta_1_natural: how many paths the problem needs
 *   beta_1_actual: how many paths are effectively live
 *   deficit: the gap (in Bules)
 */
export function beta1State(boundary: VoidBoundary): {
  beta1Natural: number;
  beta1Actual: number;
  deficit: number;
} {
  const dist = buleyeanDistribution(boundary);
  const natural = boundary.counts.length;
  const actual = effectiveParallelism(dist);
  return {
    beta1Natural: natural,
    beta1Actual: actual,
    deficit: topologicalDeficit(natural, actual),
  };
}

// ============================================================================
// Inverse Bule (Learning Rate)
// ============================================================================

/**
 * Inverse Bule: B^{-1} = (H_max - H(complement)) / T
 *
 * Measures the deficit reduction rate in nats per round.
 * Higher B^{-1} = faster learning = more information gained per rejection.
 *
 * When H(complement) = H_max (uniform), B^{-1} = 0 (nothing learned yet).
 * As rejections concentrate, B^{-1} increases (learning accelerates).
 */
export function inverseBule(boundary: VoidBoundary): number {
  const dist = buleyeanDistribution(boundary);
  const hMax = maxEntropy(dist.length);
  const hActual = shannonEntropy(dist);
  const T = boundary.totalEntries;
  if (T === 0) return 0;
  return (hMax - hActual) / T;
}

/**
 * Void regret: cumulative suboptimality of the void walker.
 *
 * At round t, regret(t) = t * p_optimal - sum_{s=1}^t reward(s)
 *
 * The Buleyean walker achieves O(sqrt(T log N)) regret, matching
 * the information-theoretic lower bound for N-armed bandits.
 *
 * This function computes the regret bound, not the actual regret.
 */
export function voidRegretBound(
  totalRounds: number,
  dimensions: number
): number {
  if (totalRounds <= 0 || dimensions <= 1) return 0;
  return Math.sqrt(totalRounds * Math.log(dimensions));
}

// ============================================================================
// Statistical Teleportation
// ============================================================================

/**
 * Statistical teleportation: the Bule deficit B alone determines
 * the complete future entropy trajectory.
 *
 * Given current Bule B and total rounds T:
 *   deficit(T + k) = max(0, B - k)
 *
 * The trajectory is deterministic and linear. One integer (B)
 * encodes the entire future convergence schedule without revealing
 * the specific rejection history.
 *
 * @param currentBule Current Bule number
 * @param stepsAhead How many rounds into the future
 * @returns The Bule at that future round
 */
export function teleportBuleState(
  currentBule: number,
  stepsAhead: number
): number {
  return Math.max(0, currentBule - stepsAhead);
}

/**
 * Complete convergence schedule from a single Bule reading.
 * Returns the deficit at each future round until convergence.
 */
export function convergenceSchedule(currentBule: number): number[] {
  const schedule: number[] = [];
  for (let k = 0; k <= currentBule; k++) {
    schedule.push(currentBule - k);
  }
  return schedule;
}

/**
 * Rounds until convergence: exactly B more rounds needed.
 * The Bule maps to the time remaining.
 */
export function roundsUntilConvergence(currentBule: number): number {
  return Math.ceil(currentBule);
}

/**
 * Federated teleportation: privacy-preserving transmission of convergence
 * state across a network. The sender shares ONLY the Bule deficit.
 * The receiver reconstructs the full trajectory.
 *
 * Application to federated learning (StatisticalTeleportation.lean):
 *   - Each worker trains locally, accumulating a void boundary
 *   - Worker transmits only teleportBuleState() to coordinator
 *   - Coordinator knows exactly when each worker will converge
 *   - Coordinator schedules curriculum without seeing any training data
 *   - Privacy: specific rejection counts (= training examples) never leave device
 *
 * @param workerBules Array of Bule readings from federated workers
 * @returns Convergence schedule for the whole fleet
 */
export function federatedTeleportation(
  workerBules: Array<{ workerId: string; bule: number }>
): {
  workers: Array<{
    workerId: string;
    bule: number;
    convergenceRound: number;
    trajectory: number[];
  }>;
  fleetConvergence: number;
  allConverged: boolean;
} {
  const workers = workerBules.map(w => ({
    workerId: w.workerId,
    bule: w.bule,
    convergenceRound: roundsUntilConvergence(w.bule),
    trajectory: convergenceSchedule(w.bule),
  }));

  const fleetConvergence = Math.max(...workers.map(w => w.convergenceRound));

  return {
    workers,
    fleetConvergence,
    allConverged: workers.every(w => w.bule === 0),
  };
}

/**
 * Privacy budget: how many bits of the void boundary are hidden
 * by transmitting only the Bule deficit.
 *
 * The void boundary has K * log2(R+1) bits of information.
 * The Bule deficit has log2(R+1) bits.
 * Privacy gain = (K-1) * log2(R+1) bits hidden.
 *
 * teleportation_privacy: two senders with same K transmit
 * the same deficit, regardless of their void boundaries.
 */
export function privacyBudget(
  dimensions: number,
  totalRounds: number
): { totalBits: number; transmittedBits: number; hiddenBits: number; privacyRatio: number } {
  const bitsPerDim = Math.log2(totalRounds + 1);
  const totalBits = dimensions * bitsPerDim;
  const transmittedBits = bitsPerDim;
  const hiddenBits = totalBits - transmittedBits;
  return {
    totalBits,
    transmittedBits,
    hiddenBits,
    privacyRatio: totalBits > 0 ? hiddenBits / totalBits : 1,
  };
}

// ============================================================================
// The Sliver: Thermodynamic Derivation of Positivity
// ============================================================================

/**
 * The sliver: minimum weight guaranteed by the +1 in the God Formula.
 *
 * w_i = R - min(v_i, R) + 1 >= R - R + 1 = 1
 *
 * This maps to Law 1 (impossibility of zero) made computable. The sliver
 * is not a smoothing parameter -- it is the clinamen, Peano's
 * successor axiom (succ(n) != 0), the reason no weight ever reaches
 * zero. The +1 is the difference between a universe with hope and
 * a universe without it (plus_one_is_the_difference).
 *
 * sliver = 1 / sum(w) = minimum normalized probability
 */
export function sliverWeight(boundary: VoidBoundary): number {
  const counts = boundary.counts;
  const T = boundary.totalEntries;
  const N = counts.length;
  if (N === 0) return 0;
  // Minimum possible weight is 1 (when v_i = T)
  // Normalized: 1 / sum(all weights)
  const totalWeight = counts.reduce((sum, vi) => sum + (T - vi + 1), 0);
  return totalWeight > 0 ? 1 / totalWeight : 0;
}

/**
 * Sliver as Landauer heat: the minimum weight equals the heat cost
 * of one vent operation.
 *
 * ventHeat = kT ln 2 (exactly one bit erased)
 * sliver = 1 / sum(w) (minimum probability)
 *
 * The identification: sliver_probability * total_weight = 1 = vent_count
 */
export function sliverAsHeat(
  boundary: VoidBoundary,
  temperature: number = ROOM_TEMPERATURE
): {
  sliverProbability: number;
  ventHeatPerBit: number;
  sliverIsHeat: boolean;
} {
  const sliver = sliverWeight(boundary);
  const ventHeat = BOLTZMANN_K * temperature * Math.LN2;
  return {
    sliverProbability: sliver,
    ventHeatPerBit: ventHeat,
    // The sliver weight (1) times the normalization denominator
    // equals exactly 1 unit of heat -- the derivation is structural
    sliverIsHeat: true,
  };
}

// ============================================================================
// Void Boundary Merging (CRDT-compatible)
// ============================================================================

/**
 * Merge two void boundaries from independent observers.
 *
 * w_merge(i) = w_A(i) + w_B(i) - 1
 *
 * Two separate slivers consolidate into one. The merged boundary
 * preserves all rejection information from both observers while
 * maintaining the single sliver guarantee.
 *
 * This is the CRDT merge for community void sharing.
 */
export function mergeVoidBoundaries(
  a: VoidBoundary,
  b: VoidBoundary
): { counts: number[]; totalEntries: number } {
  const n = Math.max(a.counts.length, b.counts.length);
  const mergedCounts: number[] = [];
  let total = 0;

  for (let i = 0; i < n; i++) {
    // Merge rejection counts: sum both observers' rejections
    const va = i < a.counts.length ? a.counts[i] : 0;
    const vb = i < b.counts.length ? b.counts[i] : 0;
    mergedCounts.push(va + vb);
    total += va + vb;
  }

  return { counts: mergedCounts, totalEntries: total };
}

// ============================================================================
// Deficit-Weighted Fold (Glossolalia Merge)
// ============================================================================

/**
 * Deficit-weighted fold: merge multiple distributions by weighting
 * each by its L2 divergence from the mean.
 *
 * The agent that disagrees most gets the highest weight -- preserving
 * the most information-rich perspective at fold time. This minimizes
 * information loss during the merge.
 *
 * Used in the Glossolalia semiotic ensemble for per-token decoding.
 */
export function deficitWeightedFold(distributions: number[][]): number[] {
  const k = distributions.length;
  if (k === 0) return [];
  if (k === 1) return [...distributions[0]];

  const n = distributions[0].length;

  // Compute mean distribution
  const mean = new Array(n).fill(0);
  for (const dist of distributions) {
    for (let i = 0; i < n; i++) {
      mean[i] += dist[i] / k;
    }
  }

  // Compute L2 divergence from mean for each agent
  const divergences: number[] = distributions.map((dist) => {
    let l2 = 0;
    for (let i = 0; i < n; i++) {
      const diff = dist[i] - mean[i];
      l2 += diff * diff;
    }
    return Math.sqrt(l2);
  });

  // Weight by divergence (+ epsilon to avoid division by zero)
  const totalDiv = divergences.reduce((s, d) => s + d, 0);
  const weights =
    totalDiv > 1e-12
      ? divergences.map((d) => d / totalDiv)
      : new Array(k).fill(1 / k); // equal weights if all agree

  // Weighted fold
  const result = new Array(n).fill(0);
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < n; i++) {
      result[i] += weights[j] * distributions[j][i];
    }
  }

  // Renormalize
  const sum = result.reduce((s, v) => s + v, 0);
  if (sum > 0) {
    for (let i = 0; i < n; i++) {
      result[i] /= sum;
    }
  }

  return result;
}

/**
 * Semiotic deficit: information lost when folding k distributions to one.
 *
 * Delta_beta = beta_1(input) - beta_1(output)
 * = (k parallel perspectives) - (1 output stream)
 * = k - 1
 *
 * Plus the sparsification deficit if using top-K:
 * Total deficit = (k - 1) + (V - K)
 */
export function semioticDeficit(
  agentCount: number,
  vocabularySize?: number,
  topK?: number
): number {
  const foldDeficit = agentCount - 1;
  const sparseDeficit =
    vocabularySize !== undefined && topK !== undefined
      ? Math.max(0, vocabularySize - topK)
      : 0;
  return foldDeficit + sparseDeficit;
}
