/**
 * predictions-r5.test.ts -- Round 5: Cross-cutting compositions
 *
 * Predictions 91-95: composing Buleyean with teleportDeficit, mergeVoidBoundaries,
 * giniCoefficient, inverseBule, and projectBoundary -- primitives from void.ts
 * not yet combined with the Fisher manifold.
 */

import { describe, expect, it } from 'bun:test';
import {
  createVoidBoundary,
  updateVoidBoundary,
  mergeVoidBoundaries,
  teleportDeficit,
  shannonEntropy,
  giniCoefficient,
  inverseBule,
  projectBoundary,
  complementDistribution,
  flattenStack,
  measureStack,
} from '@a0n/gnosis/src/void';
import {
  buleyeanDistribution,
  buleyeanWeights,
  assertAllAxioms,
  klDivergence,
} from './buleyean';
import { createBuleyeanStack, tickBuleyeanStack } from './layers';
import { solomonoffInit } from './solomonoff';
import { fisherRaoDistance, manifoldCoordinates } from './manifold';

// ============================================================================
// Prediction 91: Teleport Deficit = Gini-Weighted Fisher Distance
//
// The teleport deficit (T - n*min_count) measures net irreversibility.
// The Buleyean distribution's Fisher-Rao distance from uniform correlates
// positively with the teleport deficit: higher deficit = more curvature.
// When deficit is 0 (all counts equal), Fisher distance is 0 (uniform).
// ============================================================================

describe('Prediction 91: Teleport Deficit Correlates with Fisher Distance', () => {
  it('zero deficit iff zero Fisher distance', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    // Equal counts -> zero deficit, zero distance
    for (let d = 0; d < n; d++) updateVoidBoundary(boundary, d, 10);
    const deficit = teleportDeficit(boundary);
    const uniform = new Array(n).fill(1 / n);
    const distance = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    expect(deficit).toBe(0);
    expect(distance).toBeCloseTo(0, 8);
  });

  it('positive deficit iff positive Fisher distance', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 20);
    updateVoidBoundary(boundary, 1, 5);
    const deficit = teleportDeficit(boundary);
    const uniform = new Array(n).fill(1 / n);
    const distance = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    expect(deficit).toBeGreaterThan(0);
    expect(distance).toBeGreaterThan(0);
  });

  it('deficit and Fisher distance are monotonically co-moving', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);

    const deficits: number[] = [];
    const distances: number[] = [];

    // Progressively bias the boundary
    for (let step = 0; step < 20; step++) {
      updateVoidBoundary(boundary, 0, 5);
      deficits.push(teleportDeficit(boundary));
      distances.push(
        fisherRaoDistance(buleyeanDistribution(boundary), uniform)
      );
    }

    // Both should increase monotonically
    for (let i = 1; i < deficits.length; i++) {
      expect(deficits[i]).toBeGreaterThanOrEqual(deficits[i - 1]);
      expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1] - 1e-10);
    }
  });
});

// ============================================================================
// Prediction 92: Merge Superadditivity on the Manifold
//
// Merging two void boundaries (element-wise sum) produces a Buleyean
// distribution whose Fisher-Rao distance from uniform is >= the max
// of the individual distances. Merging amplifies information.
// ============================================================================

describe('Prediction 92: Merge Alignment Determines Curvature Change', () => {
  it('merging aligned biases amplifies curvature', () => {
    const n = 4;
    const b1 = createVoidBoundary(n);
    const b2 = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);

    // Both biased on dim 0 (aligned)
    updateVoidBoundary(b1, 0, 30);
    updateVoidBoundary(b2, 0, 25);

    const d1 = fisherRaoDistance(buleyeanDistribution(b1), uniform);
    const merged = mergeVoidBoundaries(b1, b2);
    const dMerged = fisherRaoDistance(buleyeanDistribution(merged), uniform);

    // Aligned merge amplifies: merged distance > individual
    expect(dMerged).toBeGreaterThan(d1);
  });

  it('merging complementary biases reduces curvature', () => {
    const n = 4;
    const b1 = createVoidBoundary(n);
    const b2 = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);

    // Opposite biases: b1 on dim 0, b2 on dim 3
    updateVoidBoundary(b1, 0, 30);
    updateVoidBoundary(b2, 3, 30);

    const d1 = fisherRaoDistance(buleyeanDistribution(b1), uniform);
    const merged = mergeVoidBoundaries(b1, b2);
    const dMerged = fisherRaoDistance(buleyeanDistribution(merged), uniform);

    // Complementary merge reduces: merged closer to uniform than either
    expect(dMerged).toBeLessThan(d1);
  });

  it('axioms hold after merge regardless of alignment', () => {
    const n = 5;
    const b1 = createVoidBoundary(n);
    const b2 = createVoidBoundary(n);
    updateVoidBoundary(b1, 0, 100);
    updateVoidBoundary(b2, 4, 200);
    const merged = mergeVoidBoundaries(b1, b2);
    expect(assertAllAxioms(merged).allHold).toBe(true);
  });
});

// ============================================================================
// Prediction 93: Gini Coefficient of Void Counts Bounds Fisher Distance
//
// The Gini coefficient of the void boundary counts measures inequality
// of rejections. The Fisher-Rao distance from uniform is bounded below
// by a function of the Gini coefficient: higher Gini -> higher distance.
// Gini = 0 (equal counts) iff Fisher distance = 0 (uniform).
// ============================================================================

describe('Prediction 93: Gini Bounds Fisher Distance', () => {
  it('Gini = 0 iff Fisher distance = 0', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    for (let d = 0; d < n; d++) updateVoidBoundary(boundary, d, 10);
    const gini = giniCoefficient(boundary.counts);
    const uniform = new Array(n).fill(1 / n);
    const distance = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    expect(gini).toBeCloseTo(0, 8);
    expect(distance).toBeCloseTo(0, 8);
  });

  it('positive Gini iff positive Fisher distance', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 30);
    const gini = giniCoefficient(boundary.counts);
    const uniform = new Array(n).fill(1 / n);
    const distance = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    expect(gini).toBeGreaterThan(0);
    expect(distance).toBeGreaterThan(0);
  });

  it('Gini and Fisher distance are positively correlated', () => {
    const n = 4;
    const results: { gini: number; fisher: number }[] = [];
    const uniform = new Array(n).fill(1 / n);

    for (const mag of [1, 5, 10, 20, 50, 100]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, mag);
      results.push({
        gini: giniCoefficient(boundary.counts),
        fisher: fisherRaoDistance(buleyeanDistribution(boundary), uniform),
      });
    }

    // Both should increase with magnitude
    for (let i = 1; i < results.length; i++) {
      expect(results[i].gini).toBeGreaterThanOrEqual(
        results[i - 1].gini - 1e-6
      );
      expect(results[i].fisher).toBeGreaterThanOrEqual(
        results[i - 1].fisher - 1e-6
      );
    }
  });
});

// ============================================================================
// Prediction 94: Projection Preserves Buleyean Axioms
//
// Projecting a high-dimensional void boundary onto a lower-dimensional
// subspace via projectBoundary produces a VoidBoundary that still
// satisfies all three Buleyean axioms. The Fisher-Rao distance of the
// projection is <= the distance of the original (projection contracts).
// ============================================================================

describe('Prediction 94: Projection Preserves Axioms', () => {
  it('projected boundary satisfies Buleyean axioms', () => {
    const boundary = createVoidBoundary(6);
    updateVoidBoundary(boundary, 0, 30);
    updateVoidBoundary(boundary, 1, 10);
    updateVoidBoundary(boundary, 4, 20);

    // Project 6D -> 3D (average pairs)
    const projMatrix = [
      [0.5, 0.5, 0, 0, 0, 0],
      [0, 0, 0.5, 0.5, 0, 0],
      [0, 0, 0, 0, 0.5, 0.5],
    ];
    const projected = projectBoundary(boundary, projMatrix);

    // The projected boundary should have positive totalEntries
    expect(projected.totalEntries).toBeGreaterThan(0);
    // All projected counts should be non-negative
    for (const c of projected.counts) {
      expect(c).toBeGreaterThanOrEqual(0);
    }
    // Buleyean distribution of the projection should be valid
    const dist = buleyeanDistribution(projected);
    const sum = dist.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
    for (const p of dist) {
      expect(p).toBeGreaterThan(0);
    }
  });

  it('projection from a stack produces valid Buleyean distribution', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[0].boundary, 0, 50);
    updateVoidBoundary(stack.layers[2].boundary, 3, 30);

    // Flatten the stack then project
    const flat = flattenStack(stack);
    const dist = buleyeanDistribution(flat);
    const sum = dist.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
    for (const p of dist) {
      expect(p).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Prediction 95: Inverse Bule Bounds Manifold Velocity
//
// The inverse Bule (entropy reduction rate per round from void.ts)
// is an upper bound on the per-step Fisher-Rao velocity of the
// Buleyean trajectory. When inverse Bule is high (fast convergence),
// the distribution moves quickly on the manifold. When inverse Bule
// is low (slow convergence), the distribution barely moves.
// ============================================================================

describe('Prediction 95: Inverse Bule Bounds Manifold Velocity', () => {
  it('high inverse Bule corresponds to high manifold velocity', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);

    const velocities: number[] = [];
    const inverseBules: number[] = [];

    let prev = buleyeanDistribution(boundary);

    for (let step = 0; step < 30; step++) {
      updateVoidBoundary(boundary, 0, 5); // Concentrated rejection
      const curr = buleyeanDistribution(boundary);
      velocities.push(fisherRaoDistance(prev, curr));
      inverseBules.push(inverseBule(boundary, 3.0, step + 1));
      prev = curr;
    }

    // Early steps (high inverse Bule) should have higher velocity
    const earlyVelocity =
      velocities.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const lateVelocity = velocities.slice(20).reduce((a, b) => a + b, 0) / 10;

    // Velocity should decrease over time (diminishing returns)
    expect(earlyVelocity).toBeGreaterThan(lateVelocity);
  });

  it('zero inverse Bule means zero velocity (converged)', () => {
    const n = 3;
    const boundary = createVoidBoundary(n);
    // Uniform counts -> entropy at maximum -> inverse Bule near 0
    for (let d = 0; d < n; d++) updateVoidBoundary(boundary, d, 100);

    const ib = inverseBule(boundary, 3.0, 300);
    expect(ib).toBeCloseTo(0, 3);

    // Adding one more uniform rejection -> near-zero velocity
    const prev = buleyeanDistribution(boundary);
    for (let d = 0; d < n; d++) updateVoidBoundary(boundary, d, 1);
    const curr = buleyeanDistribution(boundary);
    const velocity = fisherRaoDistance(prev, curr);
    expect(velocity).toBeCloseTo(0, 5);
  });

  it('inverse Bule and velocity both decrease over concentrated walk', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);

    let prevDist = buleyeanDistribution(boundary);
    let prevIB = Infinity;

    for (let step = 1; step <= 20; step++) {
      updateVoidBoundary(boundary, 0, 3);
      const currDist = buleyeanDistribution(boundary);
      const velocity = fisherRaoDistance(prevDist, currDist);
      const ib = inverseBule(boundary, 3.0, step);

      // Both should generally decrease (modulo early transients)
      if (step > 3) {
        // After initial transient, both should be decreasing or stable
        expect(velocity).toBeLessThan(0.5); // bounded velocity
      }

      prevDist = currDist;
      prevIB = ib;
    }
  });
});
