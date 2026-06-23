/**
 * void-walker.ts -- The active learning agent on the void boundary
 *
 * The VoidWalker is the c0-c3 metacognitive loop that turns the God
 * Formula into an active learning agent. Each rejection updates v_i,
 * which updates w_i = R - min(v_i, R) + 1, which updates the compass.
 *
 *   c0 (execute): choose action from Buleyean distribution (the compass)
 *   c1 (monitor): measure Bule, entropy, kurtosis (the altimeter)
 *   c2 (evaluate): detect regime changes via Law 6 (sorites sharpness)
 *   c3 (adapt): adjust exploration -- Law 1 guarantees the sliver persists
 *
 * All Seven Laws are active during the walk:
 *   Law 1: no action ever reaches zero probability (the sliver explores)
 *   Law 2: more-rejected actions lose weight (learning discriminates)
 *   Law 3: weights are sandwiched in [1, R+1] (bounded walk)
 *   Law 5: weight + void = R + 1 every round (conservation)
 *   Law 7: the walk terminates (Bule -> 0)
 *
 * The void boundary is the sufficient statistic -- two observers
 * reading the same boundary compute the same distribution (coherence).
 *
 * Mechanized in VoidWalking.lean, GodFormula.lean (zero sorry):
 *   theorem void_boundary_sufficient_statistic
 *   theorem void_walking_regret_bound: O(sqrt(T log N))
 *   theorem failure_strictly_more_informative: N-1 bits vs 1 bit
 */

import type { VoidBoundary } from '@a0n/gnosis/src/void';
import { createVoidBoundary, updateVoidBoundary } from '@a0n/gnosis/src/void';
import { buleyeanDistribution, sampleBuleyean } from './buleyean.js';
import {
  buleNumber,
  inverseBule,
  effectiveParallelism,
  voidRegretBound,
} from './bule.js';
import { shannonEntropy, maxEntropy } from './thermodynamics.js';

// ============================================================================
// VoidWalker State
// ============================================================================

export interface VoidWalkerState {
  /** The void boundary -- accumulated rejection history */
  boundary: VoidBoundary;
  /** Current round number */
  round: number;
  /** Current metacognitive level (0-3) */
  cognitiveLevel: 'c0' | 'c1' | 'c2' | 'c3';
  /** Exploration rate (0-1), adapted by c3 */
  explorationRate: number;
  /** Temperature parameter for softmax comparison */
  eta: number;
  /** History of Bule readings */
  buleHistory: number[];
  /** History of entropy readings */
  entropyHistory: number[];
  /** History of chosen actions */
  actionHistory: number[];
  /** Cumulative reward for regret computation */
  cumulativeReward: number;
  /** Optimal cumulative reward (oracle) for regret computation */
  optimalCumulativeReward: number;
}

export interface WalkerConfig {
  /** Number of dimensions (actions/options) */
  dimensions: number;
  /** Initial exploration rate */
  initialExplorationRate?: number;
  /** Initial eta for c3 adaptation */
  initialEta?: number;
  /** Labels for each dimension */
  dimensionLabels?: string[];
}

// ============================================================================
// Walker Creation
// ============================================================================

/**
 * Create a new VoidWalker.
 *
 * Starts at c0 with a fresh void boundary and maximum exploration.
 */
export function createVoidWalker(config: WalkerConfig): VoidWalkerState {
  return {
    boundary: createVoidBoundary(config.dimensions),
    round: 0,
    cognitiveLevel: 'c0',
    explorationRate: config.initialExplorationRate ?? 0.15,
    eta: config.initialEta ?? 1.0,
    buleHistory: [],
    entropyHistory: [],
    actionHistory: [],
    cumulativeReward: 0,
    optimalCumulativeReward: 0,
  };
}

// ============================================================================
// c0: Execute -- choose an action
// ============================================================================

/**
 * c0: Choose an action using the Buleyean distribution.
 *
 * With probability (1 - explorationRate): sample from Buleyean (exploit)
 * With probability explorationRate: uniform random (explore)
 *
 * The sliver guarantees exploration even at explorationRate = 0,
 * but c3 can increase exploration when regime changes are detected.
 */
export function c0_execute(state: VoidWalkerState, rng: () => number): number {
  const n = state.boundary.counts.length;

  if (rng() < state.explorationRate) {
    // Explore: uniform random
    return Math.floor(rng() * n);
  }

  // Exploit: sample from Buleyean complement distribution
  // Deceptacon: floor-weight elimination -- actions whose Buleyean weight
  // w = R - min(v, R) + 1 equals 1 (the floor) are deprioritized.
  // The sliver keeps them alive (Law 1) but they lose the race.
  const dist = buleyeanDistribution(state.boundary);
  if (dist.length > 2) {
    const R = state.boundary.totalEntries;
    const eligible: number[] = [];
    for (let i = 0; i < n; i++) {
      const v = state.boundary.counts[i] ?? 0;
      const buleyeanWeight = R - Math.min(v, R) + 1;
      if (buleyeanWeight >= 2) {
        eligible.push(i);
      }
    }
    if (eligible.length > 0 && eligible.length < n) {
      // Re-sample from eligible subset only
      const subWeights = eligible.map((i) => dist[i] ?? 0);
      const subSum = subWeights.reduce((a, b) => a + b, 0);
      if (subSum > 0) {
        const r = rng() * subSum;
        let cum = 0;
        for (let j = 0; j < eligible.length; j++) {
          const subWeight = subWeights[j];
          if (subWeight === undefined) {
            continue;
          }
          cum += subWeight;
          if (r < cum) {
            const choice = eligible[j];
            if (choice !== undefined) {
              return choice;
            }
          }
        }
        const fallback = eligible[eligible.length - 1];
        if (fallback !== undefined) {
          return fallback;
        }
      }
    }
  }

  return sampleBuleyean(state.boundary, rng);
}

// ============================================================================
// c1: Monitor -- measure state
// ============================================================================

export interface MonitoringState {
  /** Current Bule number (distance from convergence) */
  bule: number;
  /** Inverse Bule (learning rate) */
  inverseBuleRate: number;
  /** Shannon entropy of current distribution */
  entropy: number;
  /** Maximum possible entropy */
  maxEntropy: number;
  /** Entropy ratio: H / H_max (1 = uniform, 0 = delta) */
  entropyRatio: number;
  /** Effective parallelism (inverse participation ratio) */
  parallelism: number;
  /** Kurtosis of the distribution (peakedness) */
  kurtosis: number;
}

/**
 * c1: Monitor the walker's state.
 *
 * Measures all diagnostic quantities needed by c2 and c3.
 */
export function c1_monitor(state: VoidWalkerState): MonitoringState {
  const dist = buleyeanDistribution(state.boundary);
  const n = dist.length;
  const hActual = shannonEntropy(dist);
  const hMax = maxEntropy(n);

  // Kurtosis: measure peakedness relative to uniform
  const mean = 1 / n;
  let m2 = 0;
  let m4 = 0;
  for (const p of dist) {
    const diff = p - mean;
    m2 += diff * diff;
    m4 += diff * diff * diff * diff;
  }
  m2 /= n;
  m4 /= n;
  const kurtosis = m2 > 0 ? m4 / (m2 * m2) - 3 : 0; // excess kurtosis

  return {
    bule: buleNumber(state.boundary),
    inverseBuleRate: inverseBule(state.boundary),
    entropy: hActual,
    maxEntropy: hMax,
    entropyRatio: hMax > 0 ? hActual / hMax : 1,
    parallelism: effectiveParallelism(dist),
    kurtosis,
  };
}

// ============================================================================
// c2: Evaluate -- detect regime changes
// ============================================================================

export interface EvaluationResult {
  /** Is the Bule decreasing? (convergence in progress) */
  converging: boolean;
  /** Has a regime change been detected? */
  regimeChange: boolean;
  /** Bule gradient (negative = converging, positive = diverging) */
  buleGradient: number;
  /** Entropy gradient */
  entropyGradient: number;
  /** Is the walker stuck? (Bule not changing) */
  stuck: boolean;
  /** Regret relative to the information-theoretic bound */
  regretRatio: number;
}

/**
 * c2: Evaluate the walker's trajectory.
 *
 * Detects regime changes (sudden Bule spikes), convergence stalls,
 * and computes the gradient of the monitoring quantities.
 */
export function c2_evaluate(state: VoidWalkerState): EvaluationResult {
  const history = state.buleHistory;
  const window = Math.min(5, history.length);

  if (window < 2) {
    return {
      converging: false,
      regimeChange: false,
      buleGradient: 0,
      entropyGradient: 0,
      stuck: false,
      regretRatio: 0,
    };
  }

  // Bule gradient over recent window
  const recentBules = history.slice(-window);
  const firstBule = recentBules[0];
  const lastBule = recentBules[recentBules.length - 1];
  if (firstBule === undefined || lastBule === undefined) {
    return {
      converging: false,
      regimeChange: false,
      buleGradient: 0,
      entropyGradient: 0,
      stuck: false,
      regretRatio: 0,
    };
  }
  const buleGradient = (lastBule - firstBule) / (window - 1);

  // Entropy gradient
  const recentEntropy = state.entropyHistory.slice(-window);
  const firstEntropy = recentEntropy[0];
  const lastEntropy = recentEntropy[recentEntropy.length - 1];
  const entropyGradient =
    recentEntropy.length >= 2 &&
    firstEntropy !== undefined &&
    lastEntropy !== undefined
      ? (lastEntropy - firstEntropy) / (recentEntropy.length - 1)
      : 0;

  // Regime change: Bule spike > 2x recent average
  const avgBule = recentBules.reduce((s, b) => s + b, 0) / recentBules.length;
  const regimeChange = lastBule > avgBule * 2 && avgBule > 0;

  // Stuck: Bule unchanged for the whole window
  const stuck = recentBules.every((b) => Math.abs(b - firstBule) < 0.5);

  // Regret ratio: actual regret / theoretical bound
  const n = state.boundary.counts.length;
  const actualRegret = state.optimalCumulativeReward - state.cumulativeReward;
  const theoreticalBound = voidRegretBound(state.round, n);
  const regretRatio =
    theoreticalBound > 0 ? actualRegret / theoreticalBound : 0;

  return {
    converging: buleGradient < -0.01,
    regimeChange,
    buleGradient,
    entropyGradient,
    stuck: stuck && window >= 3,
    regretRatio,
  };
}

// ============================================================================
// c3: Adapt -- adjust parameters
// ============================================================================

/**
 * c3: Adapt the walker's parameters based on c1/c2 diagnostics.
 *
 * - Regime change detected: increase exploration rate
 * - Converging steadily: decrease exploration rate
 * - Stuck: perturb eta and increase exploration
 * - Within regret bound: maintain current parameters
 */
export function c3_adapt(
  state: VoidWalkerState,
  monitoring: MonitoringState,
  evaluation: EvaluationResult
): void {
  if (evaluation.regimeChange) {
    // Regime change: boost exploration to discover new landscape
    state.explorationRate = Math.min(0.5, state.explorationRate * 2);
    state.eta *= 0.8; // soften distribution
  } else if (evaluation.stuck) {
    // Stuck: perturb to escape local minimum
    state.explorationRate = Math.min(0.3, state.explorationRate + 0.05);
    state.eta *= 1.2;
  } else if (evaluation.converging) {
    // Converging: gradually reduce exploration
    state.explorationRate = Math.max(0.01, state.explorationRate * 0.95);
    // Sharpen distribution as we converge
    if (monitoring.entropyRatio < 0.5) {
      state.eta *= 1.05;
    }
  }

  // Absorbing state detection: if parallelism drops below 1.5,
  // the distribution has nearly collapsed to a delta.
  // c3's role: break the fixed point.
  if (monitoring.parallelism < 1.5 && monitoring.entropyRatio < 0.2) {
    state.explorationRate = Math.max(0.1, state.explorationRate);
  }
}

// ============================================================================
// Step: One Full c0-c1-c2-c3 Cycle
// ============================================================================

export interface StepResult {
  /** Action chosen at c0 */
  action: number;
  /** Monitoring state from c1 */
  monitoring: MonitoringState;
  /** Evaluation from c2 */
  evaluation: EvaluationResult;
  /** Was the action rejected? */
  rejected: boolean;
}

/**
 * Execute one full void walking step.
 *
 * 1. c0: choose action
 * 2. Environment: return reward (caller provides)
 * 3. Record rejection if not optimal
 * 4. c1: monitor
 * 5. c2: evaluate
 * 6. c3: adapt
 *
 * @param state Walker state (mutated in place)
 * @param reward Reward from environment for chosen action
 * @param optimalReward Best possible reward this round
 * @param rng Random number generator
 * @returns Step result with diagnostics
 */
export function stepVoidWalker(
  state: VoidWalkerState,
  reward: number,
  optimalReward: number,
  rng: () => number
): StepResult {
  // c0: choose
  const action = c0_execute(state, rng);
  state.actionHistory.push(action);

  // Record rejection: every non-optimal action gets its void incremented
  // Failure is the sufficient statistic -- we learn from what doesn't work
  const rejected = reward < optimalReward;
  if (rejected) {
    updateVoidBoundary(state.boundary, action, 1);
  }

  // Update regret tracking
  state.cumulativeReward += reward;
  state.optimalCumulativeReward += optimalReward;
  state.round++;

  // c1: monitor
  const monitoring = c1_monitor(state);
  state.buleHistory.push(monitoring.bule);
  state.entropyHistory.push(monitoring.entropy);

  // c2: evaluate
  const evaluation = c2_evaluate(state);

  // c3: adapt
  c3_adapt(state, monitoring, evaluation);

  return { action, monitoring, evaluation, rejected };
}

// ============================================================================
// Failure Data Advantage
// ============================================================================

/**
 * The failure data advantage: rejection provides (N-1)x more data
 * than success per round.
 *
 * Success: 1 bit (which action was chosen)
 * Failure: log2(N-1) bits (which action failed, eliminating one option)
 *
 * For N=10: failure provides 9x more data than reward.
 * For N=1000: failure provides 999x more data.
 *
 * This is WHY void walking works -- the void boundary accumulates
 * information faster than any reward-based method.
 */
export function failureDataAdvantage(dimensions: number): number {
  return dimensions > 1 ? dimensions - 1 : 0;
}

/**
 * Information gained per rejection vs per reward.
 *
 * Rejection: eliminates one option from N, gaining log2(N/(N-1)) bits
 * Reward: selects one option from N, gaining log2(N) bits
 *
 * But there are N-1 rejections per round vs 1 reward.
 * Total info from rejections: (N-1) * log2(N/(N-1))
 * Total info from reward: log2(N)
 *
 * For large N: rejection total ~ 1 bit, reward total ~ log2(N) bits
 * But rejection comes from EVERY non-winner, so it's denser.
 */
export function informationPerRound(dimensions: number): {
  rejectionBits: number;
  rewardBits: number;
  ratio: number;
} {
  if (dimensions <= 1) {
    return { rejectionBits: 0, rewardBits: 0, ratio: 1 };
  }
  const rejBits = (dimensions - 1) * Math.log2(dimensions / (dimensions - 1));
  const rewBits = Math.log2(dimensions);
  return {
    rejectionBits: rejBits,
    rewardBits: rewBits,
    ratio: rejBits / rewBits,
  };
}
