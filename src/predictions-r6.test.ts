/**
 * predictions-r6.test.ts -- Round 6: Algebraic rate theorems
 *
 * Predictions 157-161: convergence rates, simplex interior guarantees,
 * minimum probability bounds, kurtosis-gait composition, and the
 * algebraic vs exponential convergence dichotomy.
 */

import { describe, expect, it } from 'bun:test';
import {
  createVoidBoundary,
  updateVoidBoundary,
  complementDistribution,
  shannonEntropy,
  excessKurtosis,
  selectGait,
  type Gait,
} from '@a0n/gnosis/src/void';
import { buleyeanDistribution, buleyeanWeights } from './buleyean';
import { fisherRaoDistance } from './manifold';

// ============================================================================
// Prediction 157: The Buleyean Simplex Interior Guarantee
//
// The Buleyean distribution is ALWAYS in the open interior of the
// simplex (never on the boundary). min(p_i) = 1/S > 0 where
// S = T(n-1) + n. The Fisher metric is therefore always finite --
// no regularization needed, no edge cases, no numerical instability.
// This is the geometric consequence of buleyean_positivity.
// ============================================================================

describe('Prediction 157: Simplex Interior Guarantee', () => {
  it('minimum probability is exactly 1/S for maximally rejected dim', () => {
    for (const T of [1, 10, 100, 1000]) {
      const n = 4;
      const boundary = createVoidBoundary(n);
      // All rejections on dim 0
      updateVoidBoundary(boundary, 0, T);

      const dist = buleyeanDistribution(boundary);
      const weights = buleyeanWeights(boundary);
      const S = weights.reduce((a, b) => a + b, 0);

      // min(p_i) = min(w_i) / S = 1 / S
      expect(dist[0]).toBeCloseTo(1 / S, 10);
      expect(dist[0]).toBeGreaterThan(0);
    }
  });

  it('minimum probability decreases as 1/(T(n-1)+n)', () => {
    const n = 5;
    const minProbs: number[] = [];

    for (const T of [10, 50, 100, 500]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, T);
      const dist = buleyeanDistribution(boundary);
      minProbs.push(Math.min(...dist));
    }

    // min probability should decrease with T
    for (let i = 1; i < minProbs.length; i++) {
      expect(minProbs[i]).toBeLessThan(minProbs[i - 1]);
    }

    // But NEVER reaches zero
    for (const p of minProbs) {
      expect(p).toBeGreaterThan(0);
    }
  });

  it('Fisher metric is always finite (no infinities)', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 10000); // extreme asymmetry

    const dist = buleyeanDistribution(boundary);
    // Fisher metric diagonal: 1/p_i
    const fisherDiag = dist.map((p) => 1 / p);
    for (const g of fisherDiag) {
      expect(isFinite(g)).toBe(true);
      expect(g).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Prediction 158: Algebraic vs Exponential Convergence Dichotomy
//
// The Buleyean distribution converges to concentration at rate O(1/T)
// (algebraic). The softmax complement converges at rate O(exp(-eta*T))
// (exponential). This means:
// - Softmax concentrates fast but stagnates (all remaining mass in sliver)
// - Buleyean concentrates slowly but maintains sensitivity (sliver = 1/S)
// The convergence rate is a structural choice, not a tuning parameter.
// ============================================================================

describe('Prediction 158: Algebraic vs Exponential Convergence', () => {
  it('Buleyean max probability grows as O(1 - 1/T)', () => {
    const n = 4;
    const maxProbs: { T: number; pMax: number }[] = [];

    for (const T of [10, 50, 100, 500, 1000]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, T); // all on dim 0
      const dist = buleyeanDistribution(boundary);
      // Max prob should be on dim 1 (never rejected)
      const pMax = Math.max(...dist);
      maxProbs.push({ T, pMax });
    }

    // pMax = (T+1) / S = (T+1) / (T(n-1)+n)
    // As T -> inf: pMax -> 1/(n-1) (NOT 1!)
    // For n=4: limit is 1/3
    const limit = 1 / (n - 1);
    for (const { T, pMax } of maxProbs) {
      const expected = (T + 1) / (T * (n - 1) + n);
      expect(pMax).toBeCloseTo(expected, 8);
    }
    // Approaches limit from below
    expect(maxProbs[maxProbs.length - 1].pMax).toBeLessThan(limit);
    expect(maxProbs[maxProbs.length - 1].pMax).toBeGreaterThan(limit - 0.01);
  });

  it('gnosis softmax limit is 1/(n-1+exp(-eta)), lower than Buleyean limit', () => {
    const n = 4;
    const eta = 3.0;
    const T = 100000;

    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, T);

    const bMax = Math.max(...buleyeanDistribution(boundary));
    const sMax = Math.max(...complementDistribution(boundary, eta));

    // Buleyean limit: 1/(n-1) = 1/3
    const bLimit = 1 / (n - 1);
    expect(bMax).toBeCloseTo(bLimit, 3);

    // Gnosis softmax limit: 1/(n-1+exp(-eta)) < 1/(n-1)
    const sLimit = 1 / (n - 1 + Math.exp(-eta));
    expect(sMax).toBeCloseTo(sLimit, 2);

    // Buleyean concentrates MORE in the limit (higher max prob)
    expect(bMax).toBeGreaterThan(sMax);
  });

  it('neither Buleyean nor gnosis softmax reaches delta; both have finite limits', () => {
    const n = 3;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 1_000_000);

    const bDist = buleyeanDistribution(boundary);
    const bMax = Math.max(...bDist);
    const sDist = complementDistribution(boundary, 3.0);
    const sMax = Math.max(...sDist);

    // Both have limits < 1 (neither reaches delta)
    expect(bMax).toBeLessThan(1);
    expect(sMax).toBeLessThan(1);

    // Buleyean limit: 1/(n-1) = 0.5
    expect(bMax).toBeCloseTo(1 / (n - 1), 3);
    // Softmax limit: 1/(n-1+exp(-eta))
    expect(sMax).toBeCloseTo(1 / (n - 1 + Math.exp(-3)), 2);

    // Both maintain positivity on the rejected dimension
    expect(Math.min(...bDist)).toBeGreaterThan(0);
    expect(Math.min(...sDist)).toBeGreaterThan(0);
  });
});

// ============================================================================
// Prediction 159: Hellinger Distance = Closed-Form Function of Fisher-Rao
//
// The Hellinger distance H(p,q) = sqrt(1 - BC(p,q)) between two
// Buleyean distributions is a closed-form function of the Fisher-Rao
// distance: H = sqrt(1 - cos(d_FR/2)). This converts between the
// metric space and the divergence in O(1) after computing either.
// ============================================================================

describe('Prediction 159: Hellinger-Fisher Closed-Form Conversion', () => {
  it('Hellinger = sqrt(1 - cos(d_FR/2))', () => {
    for (let trial = 0; trial < 10; trial++) {
      const n = 4;
      const b1 = createVoidBoundary(n);
      const b2 = createVoidBoundary(n);
      updateVoidBoundary(b1, trial % n, 5 + trial * 3);
      updateVoidBoundary(b2, (trial + 1) % n, 8 + trial * 2);

      const p = buleyeanDistribution(b1);
      const q = buleyeanDistribution(b2);

      // Compute Bhattacharyya coefficient
      let bc = 0;
      for (let i = 0; i < n; i++) bc += Math.sqrt(p[i] * q[i]);
      bc = Math.min(bc, 1);

      // Hellinger distance
      const hellinger = Math.sqrt(1 - bc);

      // Fisher-Rao distance
      const dFR = fisherRaoDistance(p, q);

      // Closed-form conversion: H = sqrt(1 - cos(d_FR/2))
      const hellingerFromFR = Math.sqrt(Math.max(0, 1 - Math.cos(dFR / 2)));

      expect(hellinger).toBeCloseTo(hellingerFromFR, 8);
    }
  });

  it('Hellinger = 0 iff Fisher-Rao = 0 (identical distributions)', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 10);
    const p = buleyeanDistribution(boundary);

    let bc = 0;
    for (let i = 0; i < n; i++) bc += Math.sqrt(p[i] * p[i]);
    const hellinger = Math.sqrt(1 - Math.min(bc, 1));
    const dFR = fisherRaoDistance(p, p);

    expect(hellinger).toBeCloseTo(0, 10);
    expect(dFR).toBeCloseTo(0, 10);
  });

  it('Hellinger is bounded [0, 1] for all Buleyean pairs', () => {
    const n = 3;
    for (let trial = 0; trial < 20; trial++) {
      const b1 = createVoidBoundary(n);
      const b2 = createVoidBoundary(n);
      updateVoidBoundary(b1, trial % n, 1 + trial * 5);
      updateVoidBoundary(b2, (trial + 2) % n, 1 + trial * 7);

      const p = buleyeanDistribution(b1);
      const q = buleyeanDistribution(b2);
      let bc = 0;
      for (let i = 0; i < n; i++) bc += Math.sqrt(p[i] * q[i]);
      const hellinger = Math.sqrt(1 - Math.min(bc, 1));

      expect(hellinger).toBeGreaterThanOrEqual(0);
      expect(hellinger).toBeLessThanOrEqual(1);
    }
  });
});

// ============================================================================
// Prediction 160: Buleyean Entropy is Strictly Decreasing Under Biased Rejection
//
// Each biased rejection strictly reduces Shannon entropy of the
// Buleyean distribution. The entropy decrease per step is bounded
// above and approaches zero as T -> inf (diminishing returns).
// Under uniform rejection, entropy remains constant at log(n).
// ============================================================================

describe('Prediction 160: Entropy Strictly Decreasing Under Bias', () => {
  it('each biased rejection decreases entropy', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    const entropies: number[] = [
      shannonEntropy(buleyeanDistribution(boundary)),
    ];

    for (let step = 0; step < 20; step++) {
      updateVoidBoundary(boundary, 0, 3);
      entropies.push(shannonEntropy(buleyeanDistribution(boundary)));
    }

    // Strictly decreasing
    for (let i = 1; i < entropies.length; i++) {
      expect(entropies[i]).toBeLessThan(entropies[i - 1]);
    }
  });

  it('entropy decrease per step diminishes (diminishing returns)', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    const decreases: number[] = [];

    let prevH = shannonEntropy(buleyeanDistribution(boundary));
    for (let step = 0; step < 30; step++) {
      updateVoidBoundary(boundary, 0, 5);
      const H = shannonEntropy(buleyeanDistribution(boundary));
      decreases.push(prevH - H);
      prevH = H;
    }

    // Early decreases > late decreases (diminishing returns)
    const earlyAvg = decreases.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const lateAvg = decreases.slice(20).reduce((a, b) => a + b, 0) / 10;
    expect(earlyAvg).toBeGreaterThan(lateAvg);
  });

  it('uniform rejection preserves entropy at log(n)', () => {
    const n = 4;
    const maxH = Math.log(n);
    const boundary = createVoidBoundary(n);

    for (let step = 0; step < 20; step++) {
      for (let d = 0; d < n; d++) updateVoidBoundary(boundary, d, 1);
      const H = shannonEntropy(buleyeanDistribution(boundary));
      expect(H).toBeCloseTo(maxH, 8);
    }
  });
});

// ============================================================================
// Prediction 161: Buleyean Max Probability Limit Theorem
//
// Under concentrated rejection (all on one dimension), the Buleyean
// max probability converges to 1/(n-1), NOT to 1. This is the
// fundamental difference from softmax: the Buleyean distribution
// can never fully concentrate. The limit 1/(n-1) is the "democratic
// floor" -- the n-1 non-rejected dimensions share equally.
// ============================================================================

describe('Prediction 161: Max Probability Limit = 1/(n-1)', () => {
  it('max probability converges to 1/(n-1) for all n', () => {
    for (const n of [3, 4, 5, 8, 10]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, 1_000_000);

      const dist = buleyeanDistribution(boundary);
      const pMax = Math.max(...dist);
      const limit = 1 / (n - 1);

      expect(Math.abs(pMax - limit)).toBeLessThan(0.0001);
    }
  });

  it('the limit is exact: pMax = (T+1)/(T(n-1)+n) -> 1/(n-1)', () => {
    const n = 5;
    const limit = 1 / (n - 1);
    const errors: number[] = [];

    for (const T of [10, 100, 1000, 10000]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, T);
      const pMax = Math.max(...buleyeanDistribution(boundary));
      errors.push(Math.abs(pMax - limit));
    }

    // Error should decrease as O(1/T)
    for (let i = 1; i < errors.length; i++) {
      expect(errors[i]).toBeLessThan(errors[i - 1]);
    }
    // Last error should be very small
    expect(errors[errors.length - 1]).toBeLessThan(0.001);
  });

  it('min probability converges to 0 but never reaches it', () => {
    const n = 4;
    for (const T of [100, 1000, 100000]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, T);
      const pMin = Math.min(...buleyeanDistribution(boundary));
      expect(pMin).toBeGreaterThan(0);
      // pMin = 1/(T(n-1)+n) -> 0 as T -> inf
      expect(pMin).toBeLessThan(1 / T);
    }
  });

  it('Buleyean limit 1/(n-1) exceeds gnosis softmax limit 1/(n-1+exp(-eta))', () => {
    const n = 4;
    const T = 100000;
    const eta = 3.0;

    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, T);

    const bMax = Math.max(...buleyeanDistribution(boundary));
    const sMax = Math.max(...complementDistribution(boundary, eta));

    // Buleyean limit: 1/(n-1) = 1/3
    expect(bMax).toBeCloseTo(1 / (n - 1), 3);
    // Softmax limit: 1/(n-1+exp(-eta)) < 1/(n-1)
    expect(sMax).toBeCloseTo(1 / (n - 1 + Math.exp(-eta)), 2);
    // Buleyean concentrates strictly more
    expect(bMax).toBeGreaterThan(sMax);
  });
});
