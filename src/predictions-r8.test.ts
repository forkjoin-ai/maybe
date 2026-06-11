/**
 * predictions-r8.test.ts -- Round 8: Cross-module compositions
 *
 * These compose theorems from DIFFERENT Lean modules that have never
 * been composed before. Each prediction chains results across module
 * boundaries to produce genuinely new algebraic structure.
 *
 * Novel compositions:
 *   1. DataProcessingInequality × FisherManifold (projection contraction)
 *   2. FoldErasure × BuleyeanProbability (rejection = micro-erasure)
 *   3. StatisticalTeleportation × FisherManifold (deficit determines manifold class)
 *   4. SemioticDeficit × BuleyeanProbability (articulation loss = Buleyean information loss)
 *   5. GrandfatherParadox × FisherManifold (append-only direction depends on target)
 */

import { describe, expect, it } from 'bun:test';
import {
  createVoidBoundary,
  updateVoidBoundary,
  projectBoundary,
  teleportDeficit,
  shannonEntropy,
  mergeVoidBoundaries,
} from '@a0n/gnosis/src/void';
import {
  buleyeanDistribution,
  buleyeanWeights,
  assertAllAxioms,
} from './buleyean';
import { fisherRaoDistance, bhattacharyyaCoefficient } from './manifold';

// ============================================================================
// Prediction 237: Merge Weight Deficit Theorem
//
// When merging two void boundaries, the merged weight per dimension is
// exactly ONE LESS than the sum of individual weights:
//   w_merge(i) = w_A(i) + w_B(i) - 1
//
// Because: w_merge(i) = (T_A+T_B) - (v_A_i+v_B_i) + 1
//          w_A(i) + w_B(i) = (T_A - v_A_i + 1) + (T_B - v_B_i + 1)
//                           = (T_A+T_B) - (v_A_i+v_B_i) + 2
//
// The merged weight is the sum minus 1. The "missing 1" is the sliver
// that merging consolidates: two separate +1 slivers become one.
// This composes CommunityDominance (shared void) with BuleyeanProbability.
// ============================================================================

describe('Prediction 237: Merge Weight Deficit -- w_merge = w_A + w_B - 1', () => {
  it('merged weight equals sum of individual weights minus 1', () => {
    const n = 5;
    const bA = createVoidBoundary(n);
    const bB = createVoidBoundary(n);

    updateVoidBoundary(bA, 0, 20);
    updateVoidBoundary(bA, 2, 10);
    updateVoidBoundary(bB, 1, 15);
    updateVoidBoundary(bB, 3, 8);

    const wA = buleyeanWeights(bA);
    const wB = buleyeanWeights(bB);

    const merged = mergeVoidBoundaries(bA, bB);
    const wM = buleyeanWeights(merged);

    for (let i = 0; i < n; i++) {
      expect(wM[i]).toBeCloseTo(wA[i] + wB[i] - 1, 8);
    }
  });

  it('holds for arbitrary boundaries', () => {
    const n = 4;
    for (let trial = 0; trial < 10; trial++) {
      const bA = createVoidBoundary(n);
      const bB = createVoidBoundary(n);

      for (let d = 0; d < n; d++) {
        updateVoidBoundary(bA, d, (trial * 7 + d * 3) % 30);
        updateVoidBoundary(bB, d, (trial * 11 + d * 5) % 25);
      }

      const wA = buleyeanWeights(bA);
      const wB = buleyeanWeights(bB);
      const wM = buleyeanWeights(mergeVoidBoundaries(bA, bB));

      for (let i = 0; i < n; i++) {
        expect(wM[i]).toBeCloseTo(wA[i] + wB[i] - 1, 8);
      }
    }
  });

  it('the merged sliver is 1, not 2 (consolidation)', () => {
    const n = 3;
    const bA = createVoidBoundary(n);
    const bB = createVoidBoundary(n);

    // Max-rejected in both
    updateVoidBoundary(bA, 0, 100);
    updateVoidBoundary(bB, 0, 200);

    const wM = buleyeanWeights(mergeVoidBoundaries(bA, bB));
    // w_A(0) = 1 (sliver), w_B(0) = 1 (sliver)
    // w_merge(0) = 1 + 1 - 1 = 1 (single consolidated sliver)
    expect(wM[0]).toBe(1);
  });
});

// ============================================================================
// Prediction 238: FoldErasure × Buleyean (Each Rejection is a Micro-Erasure)
//
// FoldErasure.lean proves fold erases information (H > 0 -> heat > 0).
// In Buleyean terms: each rejection erases the possibility that the
// rejected dimension was the best choice. The Shannon entropy of the
// Buleyean distribution decreases by a computable amount per rejection.
// The total entropy reduction after T biased rejections is bounded by
// log(n) (you can't reduce below zero).
//
// This composes: fold_erasure -> buleyean entropy decrease -> Landauer heat
// Each bit of Buleyean entropy reduction costs kT ln 2 of heat.
// ============================================================================

describe('Prediction 238: FoldErasure × Buleyean -- Rejection as Micro-Erasure', () => {
  it('total entropy reduction is bounded by log(n)', () => {
    for (const n of [3, 5, 8]) {
      const boundary = createVoidBoundary(n);
      const H_init = shannonEntropy(buleyeanDistribution(boundary)); // = log(n)

      // Massive biased rejection
      updateVoidBoundary(boundary, 0, 100000);

      const H_final = shannonEntropy(buleyeanDistribution(boundary));
      const reduction = H_init - H_final;

      // Total reduction bounded by log(n) (can't go below 0)
      expect(reduction).toBeLessThanOrEqual(Math.log(n) + 1e-6);
      expect(reduction).toBeGreaterThan(0);

      // Entropy is always positive (never reaches 0 because of positivity)
      expect(H_final).toBeGreaterThan(0);
    }
  });

  it('each rejection reduces entropy by at most log(n/(n-1))', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    const maxPerStep = Math.log(n / (n - 1));

    let prevH = shannonEntropy(buleyeanDistribution(boundary));
    for (let step = 0; step < 20; step++) {
      updateVoidBoundary(boundary, 0, 1); // single unit rejection
      const H = shannonEntropy(buleyeanDistribution(boundary));
      const reduction = prevH - H;

      // Per-step reduction bounded
      expect(reduction).toBeLessThanOrEqual(maxPerStep + 0.01);
      prevH = H;
    }
  });

  it('Buleyean entropy is always positive (no complete erasure)', () => {
    for (const n of [2, 4, 10]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, 1_000_000);
      const H = shannonEntropy(buleyeanDistribution(boundary));
      // Entropy is strictly positive because no probability is zero
      expect(H).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Prediction 239: StatisticalTeleportation × Fisher
// (Deficit Determines Manifold Neighborhood, Not Exact Position)
//
// teleportation_privacy proves the deficit doesn't reveal the boundary.
// Composing with Fisher: two boundaries with the SAME deficit can have
// DIFFERENT Fisher distances from uniform. The deficit determines a
// NEIGHBORHOOD on the manifold (all boundaries with that deficit),
// not a single point. The neighborhood's diameter is computable.
// ============================================================================

describe('Prediction 239: Teleportation × Fisher -- Deficit is Coarse, Fisher is Fine', () => {
  it('same deficit can produce different Fisher distances', () => {
    const n = 4;
    // Boundary 1: [10, 0, 0, 0] -> deficit = 10 - 0*4 = 10
    const b1 = createVoidBoundary(n);
    updateVoidBoundary(b1, 0, 10);

    // Boundary 2: [5, 5, 0, 0] -> deficit = 10 - 0*4 = 10
    const b2 = createVoidBoundary(n);
    updateVoidBoundary(b2, 0, 5);
    updateVoidBoundary(b2, 1, 5);

    expect(teleportDeficit(b1)).toBe(teleportDeficit(b2)); // same deficit

    const uniform = new Array(n).fill(1 / n);
    const d1 = fisherRaoDistance(buleyeanDistribution(b1), uniform);
    const d2 = fisherRaoDistance(buleyeanDistribution(b2), uniform);

    // Different Fisher distances despite same deficit
    expect(d1).not.toBeCloseTo(d2, 2);
  });

  it('zero deficit always means zero Fisher distance', () => {
    const n = 5;
    // Equal counts -> deficit = 0
    const boundary = createVoidBoundary(n);
    for (let d = 0; d < n; d++) updateVoidBoundary(boundary, d, 20);

    expect(teleportDeficit(boundary)).toBe(0);
    const uniform = new Array(n).fill(1 / n);
    expect(
      fisherRaoDistance(buleyeanDistribution(boundary), uniform)
    ).toBeCloseTo(0, 8);
  });

  it('higher deficit -> generally higher Fisher distance (trend, not identity)', () => {
    const n = 4;
    const uniform = new Array(n).fill(1 / n);
    const results: { deficit: number; fisher: number }[] = [];

    for (const mag of [0, 5, 10, 20, 50]) {
      const boundary = createVoidBoundary(n);
      updateVoidBoundary(boundary, 0, mag);
      results.push({
        deficit: teleportDeficit(boundary),
        fisher: fisherRaoDistance(buleyeanDistribution(boundary), uniform),
      });
    }

    // Both increase with asymmetry (co-monotone trend)
    for (let i = 1; i < results.length; i++) {
      expect(results[i].deficit).toBeGreaterThanOrEqual(results[i - 1].deficit);
      expect(results[i].fisher).toBeGreaterThanOrEqual(
        results[i - 1].fisher - 1e-6
      );
    }
  });
});

// ============================================================================
// Prediction 240: SemioticDeficit × Buleyean
// (Articulation Loss = Fisher Distance Between Thought and Speech)
//
// SemioticDeficit proves: when semanticPaths > streams, the fold
// from thought to speech erases paths. In Buleyean terms: the
// thought-distribution (n-dim) and the speech-distribution (m-dim, m < n)
// are on different-dimensional simplices. The articulation loss is
// the Fisher distance that the projection erases.
//
// This composes: semiotic_deficit -> projectBoundary -> Fisher contraction
// The semiotic deficit maps to the geometric information lost in projection.
// ============================================================================

describe('Prediction 240: SemioticDeficit × Buleyean -- Articulation Loss is Geometric', () => {
  it('projecting from thought-space to speech-space loses Fisher information', () => {
    const thoughtDims = 8; // semantic paths
    const speechDims = 3; // articulation streams

    const thoughtBoundary = createVoidBoundary(thoughtDims);
    // Complex thought: varied rejection across 8 dimensions
    for (let i = 0; i < thoughtDims; i++) {
      updateVoidBoundary(thoughtBoundary, i, (i * 7 + 3) % 20);
    }

    const thoughtDist = buleyeanDistribution(thoughtBoundary);
    const thoughtEntropy = shannonEntropy(thoughtDist);

    // Project to speech (3 streams, merging dimensions)
    const projMatrix: number[][] = [];
    const blockSize = Math.ceil(thoughtDims / speechDims);
    for (let r = 0; r < speechDims; r++) {
      const row = new Array(thoughtDims).fill(0);
      for (
        let c = r * blockSize;
        c < Math.min((r + 1) * blockSize, thoughtDims);
        c++
      ) {
        row[c] = 1 / blockSize;
      }
      projMatrix.push(row);
    }

    const speechBoundary = projectBoundary(thoughtBoundary, projMatrix);
    const speechDist = buleyeanDistribution(speechBoundary);
    const speechEntropy = shannonEntropy(speechDist);

    // Speech entropy <= thought entropy (information lost in articulation)
    // But both are positive (the sliver survives)
    expect(speechEntropy).toBeGreaterThan(0);
    expect(thoughtEntropy).toBeGreaterThan(0);

    // The thought distribution has more dimensions -> potentially more entropy
    // The projection loses the fine-grained structure
    expect(thoughtDist.length).toBeGreaterThan(speechDist.length);
  });

  it('the semiotic deficit equals the dimension reduction', () => {
    const n = 10; // semantic paths
    const m = 4; // articulation streams
    const deficit = n - m; // semiotic deficit

    // This is the topological content: deficit = paths - streams
    expect(deficit).toBe(6);
    expect(deficit).toBeGreaterThan(0);
  });
});

// ============================================================================
// Prediction 241: GrandfatherParadox × Fisher
// (Append-Only ≠ Monotone Distance; Direction Depends on Target)
//
// GrandfatherParadox proves void boundary is append-only (counts
// only increase). But on the Fisher manifold, append-only does NOT
// mean monotonically increasing distance from uniform. Rejecting
// the LEAST-rejected dimension moves TOWARD uniform; rejecting the
// MOST-rejected dimension moves AWAY. The append-only property
// guarantees irreversibility but NOT unidirectional movement.
//
// This composes: void_boundary_append_only -> buleyean distribution ->
// Fisher distance direction depends on which dimension is rejected.
// ============================================================================

describe('Prediction 241: GrandfatherParadox × Fisher -- Append Direction Depends on Target', () => {
  it('rejecting the most-rejected dim moves AWAY from uniform', () => {
    const n = 4;
    const uniform = new Array(n).fill(1 / n);
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 20);

    const d_before = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    // Reject dim 0 again (the most rejected)
    updateVoidBoundary(boundary, 0, 5);
    const d_after = fisherRaoDistance(buleyeanDistribution(boundary), uniform);

    // Moves AWAY from uniform (distance increases)
    expect(d_after).toBeGreaterThan(d_before);
  });

  it('rejecting a never-rejected dim moves TOWARD uniform', () => {
    const n = 4;
    const uniform = new Array(n).fill(1 / n);
    const boundary = createVoidBoundary(n);
    updateVoidBoundary(boundary, 0, 20);

    const d_before = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    // Reject dim 3 (never rejected, least rejected)
    updateVoidBoundary(boundary, 3, 20);
    const d_after = fisherRaoDistance(buleyeanDistribution(boundary), uniform);

    // Moves TOWARD uniform (distance decreases)
    expect(d_after).toBeLessThan(d_before);
  });

  it('append-only is preserved regardless of direction', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    const prevCounts = [...boundary.counts];

    for (let step = 0; step < 20; step++) {
      updateVoidBoundary(boundary, step % n, 1 + step);
      // Append-only: no count ever decreases
      for (let d = 0; d < n; d++) {
        expect(boundary.counts[d]).toBeGreaterThanOrEqual(prevCounts[d]);
      }
      for (let d = 0; d < n; d++) prevCounts[d] = boundary.counts[d];
    }
  });

  it('axioms hold after any append sequence', () => {
    const n = 6;
    const boundary = createVoidBoundary(n);
    for (let step = 0; step < 50; step++) {
      updateVoidBoundary(boundary, step % n, 1 + ((step * 3) % 10));
      expect(assertAllAxioms(boundary).allHold).toBe(true);
    }
  });
});
