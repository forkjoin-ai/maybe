/**
 * predictions-r7.test.ts -- Round 7: Walker dynamics + decay composition
 *
 * Predictions 227-231: composing the c0-c3 walker loop and decay
 * operations with the Buleyean distribution on the Fisher manifold.
 * These are TRAJECTORY theorems, not state theorems.
 */

import { describe, expect, it } from 'bun:test';
import {
  createVoidBoundary,
  updateVoidBoundary,
  decayVoidBoundary,
  boundaryDimensions,
  createWalker,
  c0Choose,
  c0Update,
  c1Measure,
  c2c3Adapt,
  shannonEntropy,
  createTimescaleBoundary,
  tickTimescaleBoundary,
  TIMESCALE_DECAY,
} from '@a0n/gnosis/src/void';
import {
  buleyeanDistribution,
  buleyeanWeights,
  assertAllAxioms,
} from './buleyean';
import { fisherRaoDistance } from './manifold';

// ============================================================================
// Prediction 227: Decay Moves Buleyean Distribution Toward Uniform
//
// decayVoidBoundary(factor) reduces all counts proportionally.
// This moves the Buleyean distribution TOWARD uniform (reducing
// Fisher-Rao distance from uniform). Decay is the geometric
// inverse of rejection: rejection moves away from uniform,
// decay moves back. The decay-rejection balance determines the
// steady-state Fisher distance.
// ============================================================================

describe('Prediction 227: Decay Moves Toward Uniform', () => {
  it('each decay step reduces Fisher distance from uniform', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);

    // Build up asymmetry
    updateVoidBoundary(boundary, 0, 50);
    updateVoidBoundary(boundary, 1, 10);

    const distances: number[] = [];
    for (let step = 0; step < 15; step++) {
      distances.push(
        fisherRaoDistance(buleyeanDistribution(boundary), uniform)
      );
      decayVoidBoundary(boundary, 0.2);
    }

    // Each decay step should reduce distance (move toward uniform)
    for (let i = 1; i < distances.length; i++) {
      expect(distances[i]).toBeLessThanOrEqual(distances[i - 1] + 1e-6);
    }
  });

  it('sufficient decay drives distance to zero (uniform recovery)', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);

    updateVoidBoundary(boundary, 0, 100);
    updateVoidBoundary(boundary, 2, 50);

    // Heavy decay
    for (let i = 0; i < 100; i++) {
      decayVoidBoundary(boundary, 0.5);
    }

    const d = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    // After 100 rounds of 50% decay, should be very close to uniform
    expect(d).toBeLessThan(0.01);
  });

  it('axioms hold after any amount of decay', () => {
    const n = 6;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 1000);

    for (let i = 0; i < 50; i++) {
      decayVoidBoundary(boundary, 0.3);
      expect(assertAllAxioms(boundary).allHold).toBe(true);
    }
  });
});

// ============================================================================
// Prediction 228: The Walker's Buleyean Trajectory is Measurably Different
// from its Softmax Trajectory on the Same Boundary
//
// A Walker using c0Choose (softmax complement) traces one trajectory.
// The same boundary state evaluated via buleyeanDistribution traces
// a different trajectory. The two trajectories diverge on the Fisher
// manifold, and the divergence grows with the number of steps.
// ============================================================================

describe('Prediction 228: Walker Softmax vs Buleyean Trajectory Divergence', () => {
  it('same boundary produces different distributions', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 20);
    updateVoidBoundary(boundary, 2, 5);

    const walker = createWalker(boundary);
    const softmaxMeasure = c1Measure(walker);
    const buleyeanDist = buleyeanDistribution(boundary);
    const buleyeanEntropy = shannonEntropy(buleyeanDist);

    // The softmax and Buleyean entropies will differ
    // (softmax uses exp(-eta*v), Buleyean uses T-v+1)
    expect(Math.abs(softmaxMeasure.entropy - buleyeanEntropy)).toBeGreaterThan(
      0.001
    );
  });

  it('divergence is always positive for non-uniform boundaries', () => {
    const n = 4;

    for (const mag of [5, 20, 50, 100]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, mag);

      const walker = createWalker(boundary);
      const m = c1Measure(walker);
      const bH = shannonEntropy(buleyeanDistribution(boundary));

      // The two formulas always produce different entropies
      expect(Math.abs(m.entropy - bH)).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Prediction 229: Timescale Decay Rates Create a Fisher Distance Hierarchy
//
// Different timescales (instant, seconds, minutes, ..., generational)
// have different decay rates. After equal initial bias, faster-decaying
// timescales have smaller Fisher distance from uniform. The hierarchy
// is preserved: d_FR(instant) <= d_FR(seconds) <= ... <= d_FR(lifetime).
// ============================================================================

describe('Prediction 229: Timescale Fisher Distance Hierarchy', () => {
  it('faster timescales decay to smaller Fisher distance', () => {
    const n = 4;
    const uniform = new Array(n).fill(1 / n);

    // Create TimescaleBoundaries at different rates
    const timescales: Array<{ name: string; decay: number }> = [
      { name: 'minutes', decay: TIMESCALE_DECAY.minutes }, // 0.2
      { name: 'days', decay: TIMESCALE_DECAY.days }, // 0.05
      { name: 'years', decay: TIMESCALE_DECAY.years }, // 0.002
      { name: 'lifetime', decay: TIMESCALE_DECAY.lifetime }, // 0.0005
    ];

    const results: { name: string; finalDistance: number }[] = [];

    for (const ts of timescales) {
      const boundary = createVoidBoundary(n);
      // Same initial bias
      updateVoidBoundary(boundary, 0, 100);

      // Apply decay 20 times
      for (let i = 0; i < 20; i++) {
        decayVoidBoundary(boundary, ts.decay);
      }

      const d = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
      results.push({ name: ts.name, finalDistance: d });
    }

    // Faster decay -> smaller final distance
    for (let i = 1; i < results.length; i++) {
      expect(results[i].finalDistance).toBeGreaterThanOrEqual(
        results[i - 1].finalDistance - 0.01
      );
    }
  });
});

// ============================================================================
// Prediction 230: The c2c3 Adaptation Loop is a Contractive Map on Walker State
//
// After c2c3Adapt, the walker's exploration and eta parameters are
// bounded. Exploration stays in [0.01, 0.4]. Eta stays in [1.0, 8.0].
// These bounds are absorbing: once entered, never escaped.
// The walker parameter space is a compact set.
// ============================================================================

describe('Prediction 230: Walker Parameter Bounds are Absorbing', () => {
  it('exploration stays in [0.01, 0.4] after adaptation', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    const walker = createWalker(boundary, 2.0, 0.3);

    // Run many steps and adaptations
    const rng = (() => {
      let s = 42;
      return () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };
    })();

    for (let step = 0; step < 200; step++) {
      const dim = c0Choose(walker, rng);
      c0Update(walker, dim, 1);
      c2c3Adapt(walker);

      expect(walker.exploration).toBeGreaterThanOrEqual(0.01);
      expect(walker.exploration).toBeLessThanOrEqual(0.4);
    }
  });

  it('eta stays in [1.0, 8.0] after adaptation', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    const walker = createWalker(boundary, 2.0, 0.3);

    const rng = (() => {
      let s = 77;
      return () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };
    })();

    for (let step = 0; step < 200; step++) {
      const dim = c0Choose(walker, rng);
      c0Update(walker, dim, 1);
      c2c3Adapt(walker);

      expect(walker.eta).toBeGreaterThanOrEqual(1.0);
      expect(walker.eta).toBeLessThanOrEqual(8.0);
    }
  });

  it('walker steps count monotonically', () => {
    const n = 3;
    const boundary = createVoidBoundary(n);
    const walker = createWalker(boundary);

    const rng = (() => {
      let s = 99;
      return () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };
    })();

    for (let step = 0; step < 50; step++) {
      c0Update(walker, step % n, 1);
      expect(walker.steps).toBe(step + 1);
    }
  });
});

// ============================================================================
// Prediction 231: Decay + Rejection Reaches a Steady-State Fisher Distance
//
// When decay and rejection operate simultaneously (as in a
// TimescaleBoundary under tick), the Fisher distance from uniform
// reaches a steady state where the two forces balance. The steady
// state depends on the ratio of rejection rate to decay rate.
// Higher rejection/decay ratio -> farther from uniform (more curvature).
// ============================================================================

describe('Prediction 231: Decay-Rejection Steady State', () => {
  it('simultaneous decay and rejection converge to steady state', () => {
    const n = 4;
    const uniform = new Array(n).fill(1 / n);
    const boundary = createVoidBoundary(n);

    const distances: number[] = [];
    for (let step = 0; step < 60; step++) {
      // Rejection: add 5 to dim 0 each step
      updateVoidBoundary(boundary, 0, 5);
      // Decay: 10% per step
      decayVoidBoundary(boundary, 0.1);
      distances.push(
        fisherRaoDistance(buleyeanDistribution(boundary), uniform)
      );
    }

    // Should converge: late distances should be stable
    const lateDistances = distances.slice(40);
    const mean =
      lateDistances.reduce((a, b) => a + b, 0) / lateDistances.length;
    const variance =
      lateDistances.reduce((s, d) => s + (d - mean) ** 2, 0) /
      lateDistances.length;

    // Low variance = steady state reached
    expect(Math.sqrt(variance)).toBeLessThan(0.05);
    // Steady state is positive (not uniform -- rejection keeps it away)
    expect(mean).toBeGreaterThan(0.01);
  });

  it('higher rejection/decay ratio -> farther from uniform', () => {
    const n = 4;
    const uniform = new Array(n).fill(1 / n);

    const steadyStates: number[] = [];

    for (const rejectionRate of [1, 5, 20]) {
      const boundary = createVoidBoundary(n);
      // Run to steady state
      for (let step = 0; step < 100; step++) {
        updateVoidBoundary(boundary, 0, rejectionRate);
        decayVoidBoundary(boundary, 0.1);
      }
      steadyStates.push(
        fisherRaoDistance(buleyeanDistribution(boundary), uniform)
      );
    }

    // Higher rejection rate -> farther from uniform at steady state
    for (let i = 1; i < steadyStates.length; i++) {
      expect(steadyStates[i]).toBeGreaterThan(steadyStates[i - 1]);
    }
  });

  it('axioms hold at steady state', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    for (let step = 0; step < 100; step++) {
      updateVoidBoundary(boundary, 0, 10);
      decayVoidBoundary(boundary, 0.15);
    }
    expect(assertAllAxioms(boundary).allHold).toBe(true);
  });
});
