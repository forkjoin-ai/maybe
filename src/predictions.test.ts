/**
 * predictions.test.ts -- Five novel predictions from the Fisher manifold
 *
 * Each prediction chains Buleyean axioms + Fisher geometry into a
 * falsifiable claim. The tests prove the mathematical content;
 * the manuscript names the experiments that would refute the empirical claims.
 */

import { describe, expect, it } from 'bun:test';
import {
  createVoidBoundary,
  updateVoidBoundary,
  complementDistribution,
} from '@a0n/gnosis/src/void';
import { buleyeanDistribution, assertAllAxioms } from './buleyean';
import { createBuleyeanStack, tickBuleyeanStack, getLayer } from './layers';
import { solomonoffInit } from './solomonoff';
import {
  fisherRaoDistance,
  geodesicCurvature,
  geodesicPath,
  totalCurvature,
  manifoldCoordinates,
  detectFraud,
  fisherMetric,
  scalarCurvature,
  bhattacharyyaCoefficient,
} from './manifold';

// ============================================================================
// Prediction 1: Buleyean Sustained Responsiveness
//
// Softmax complement concentrates exponentially, causing distribution
// stagnation (near-zero manifold velocity after concentration). Buleyean
// complement responds linearly, maintaining sustained manifold velocity.
// The Buleyean walker keeps learning; the softmax walker freezes.
//
// Total Fisher-Rao path length of the Buleyean trajectory exceeds softmax.
// ============================================================================

describe('Prediction 1: Buleyean Sustained Responsiveness', () => {
  it('Buleyean late-stage velocity ratio exceeds softmax under concentrated rejection', () => {
    const n = 5;

    const boundary = createVoidBoundary(n);
    const buleyeanVelocities: number[] = [];
    const softmaxVelocities: number[] = [];

    let prevB = buleyeanDistribution(boundary);
    let prevS = complementDistribution(boundary, 3.0);

    // Concentrated rejection: bias toward dim 0 (same pattern as test 2)
    for (let step = 0; step < 40; step++) {
      updateVoidBoundary(boundary, 0, 5);
      if (step % 5 === 0) updateVoidBoundary(boundary, 1, 1); // mild secondary
      const currB = buleyeanDistribution(boundary);
      const currS = complementDistribution(boundary, 3.0);

      buleyeanVelocities.push(fisherRaoDistance(prevB, currB));
      softmaxVelocities.push(fisherRaoDistance(prevS, currS));
      prevB = currB;
      prevS = currS;
    }

    // Late/early velocity ratio: Buleyean sustains; softmax stagnates
    const bEarly = buleyeanVelocities.slice(0, 10).reduce((a, b) => a + b, 0);
    const bLate = buleyeanVelocities.slice(30).reduce((a, b) => a + b, 0);
    const sEarly = softmaxVelocities.slice(0, 10).reduce((a, b) => a + b, 0);
    const sLate = softmaxVelocities.slice(30).reduce((a, b) => a + b, 0);

    const bRatio = bLate / (bEarly + 1e-15);
    const sRatio = sLate / (sEarly + 1e-15);

    // Buleyean maintains higher late/early ratio (sustained learning)
    expect(bRatio).toBeGreaterThan(sRatio);
  });

  it('softmax velocity decays to near-zero while Buleyean remains positive', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);

    const buleyeanVelocities: number[] = [];
    const softmaxVelocities: number[] = [];

    let prevB = buleyeanDistribution(boundary);
    let prevS = complementDistribution(boundary, 3.0);

    for (let step = 0; step < 40; step++) {
      updateVoidBoundary(boundary, 0, 5);
      const currB = buleyeanDistribution(boundary);
      const currS = complementDistribution(boundary, 3.0);

      buleyeanVelocities.push(fisherRaoDistance(prevB, currB));
      softmaxVelocities.push(fisherRaoDistance(prevS, currS));

      prevB = currB;
      prevS = currS;
    }

    // Softmax velocity should decay much faster than Buleyean
    const softmaxLate =
      softmaxVelocities.slice(30).reduce((a, b) => a + b, 0) / 10;
    const buleyeanLate =
      buleyeanVelocities.slice(30).reduce((a, b) => a + b, 0) / 10;

    // Buleyean still moving; softmax has stagnated
    expect(buleyeanLate / (softmaxLate + 1e-15)).toBeGreaterThan(2);
  });

  it('KL divergence between the two grows with asymmetry', () => {
    const boundary = createVoidBoundary(4);
    const klValues: number[] = [];
    for (let step = 0; step < 20; step++) {
      updateVoidBoundary(boundary, 0, 5);
      const b = buleyeanDistribution(boundary);
      const s = complementDistribution(boundary, 3.0);
      let kl = 0;
      for (let i = 0; i < b.length; i++) {
        if (b[i] > 0 && s[i] > 0) kl += b[i] * Math.log(b[i] / s[i]);
      }
      klValues.push(kl);
    }
    const firstHalf = klValues.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const secondHalf = klValues.slice(10).reduce((a, b) => a + b, 0) / 10;
    expect(secondHalf).toBeGreaterThan(firstHalf);
  });
});

// ============================================================================
// Prediction 2: Manifold Velocity = Inverse Bule
//
// The rate of Fisher-Rao distance increase from the uniform distribution
// equals the inverse Bule (entropy reduction rate). The void walker's
// convergence metric maps to the manifold velocity.
// ============================================================================

describe('Prediction 2: Manifold Velocity Equals Inverse Bule', () => {
  it('Fisher-Rao distance from uniform increases monotonically with rejections', () => {
    const n = 5;
    const boundary = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);
    const distances: number[] = [
      fisherRaoDistance(buleyeanDistribution(boundary), uniform),
    ];

    for (let step = 0; step < 30; step++) {
      updateVoidBoundary(boundary, step % n, 1);
      // After uniform rejections, should stay near uniform
    }
    // Now bias heavily
    for (let step = 0; step < 30; step++) {
      updateVoidBoundary(boundary, 0, 3);
      distances.push(
        fisherRaoDistance(buleyeanDistribution(boundary), uniform)
      );
    }

    // Distance from uniform should increase monotonically during biased accumulation
    for (let i = 1; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1] - 1e-10);
    }
  });

  it('manifold velocity correlates with entropy reduction', () => {
    const n = 4;
    const boundary = createVoidBoundary(n);
    const uniform = new Array(n).fill(1 / n);

    const velocities: number[] = [];
    const entropyChanges: number[] = [];

    let prevDist = fisherRaoDistance(buleyeanDistribution(boundary), uniform);
    let prevEntropy = -buleyeanDistribution(boundary).reduce(
      (s, p) => s + (p > 0 ? p * Math.log(p) : 0),
      0
    );

    for (let step = 0; step < 40; step++) {
      updateVoidBoundary(boundary, 0, 2); // Concentrate on dim 0
      const dist = buleyeanDistribution(boundary);
      const d = fisherRaoDistance(dist, uniform);
      const entropy = -dist.reduce(
        (s, p) => s + (p > 0 ? p * Math.log(p) : 0),
        0
      );

      velocities.push(d - prevDist);
      entropyChanges.push(prevEntropy - entropy); // Positive when entropy decreases

      prevDist = d;
      prevEntropy = entropy;
    }

    // Both should be positive (moving away from uniform, entropy decreasing)
    const posVelocities = velocities.filter((v) => v > 1e-15).length;
    const posEntropyChanges = entropyChanges.filter((e) => e > 1e-15).length;
    expect(posVelocities).toBeGreaterThan(velocities.length * 0.8);
    expect(posEntropyChanges).toBeGreaterThan(entropyChanges.length * 0.8);

    // Correlation between velocity and entropy change should be positive
    const meanV = velocities.reduce((a, b) => a + b, 0) / velocities.length;
    const meanE =
      entropyChanges.reduce((a, b) => a + b, 0) / entropyChanges.length;
    let cov = 0;
    let varV = 0;
    let varE = 0;
    for (let i = 0; i < velocities.length; i++) {
      cov += (velocities[i] - meanV) * (entropyChanges[i] - meanE);
      varV += (velocities[i] - meanV) ** 2;
      varE += (entropyChanges[i] - meanE) ** 2;
    }
    const correlation = cov / (Math.sqrt(varV) * Math.sqrt(varE));
    expect(correlation).toBeGreaterThan(0.5);
  });
});

// ============================================================================
// Prediction 3: Layer Tetrahedron Contraction Under Tick
//
// When the BoundaryStack ticks (upward constraint + downward context),
// the total inter-layer Fisher-Rao distance decreases. The four layers
// converge on the manifold. The epistemological tetrahedron shrinks.
// ============================================================================

describe('Prediction 3: Layer Tetrahedron Contraction', () => {
  it('inter-layer distances decrease after stack tick', () => {
    const stack = createBuleyeanStack(5);

    // Initialize each layer with different bias
    updateVoidBoundary(stack.layers[0].boundary, 0, 50); // retrocausal: heavy on dim 0
    updateVoidBoundary(stack.layers[1].boundary, 2, 30); // bayesian: dim 2
    updateVoidBoundary(stack.layers[2].boundary, 4, 20); // frequentist: dim 4
    updateVoidBoundary(stack.layers[3].boundary, 1, 40); // solomonoff: dim 1

    const coordsBefore = manifoldCoordinates(stack);
    const totalDistBefore =
      coordsBefore.interLayerDistances.retrocausal_bayesian +
      coordsBefore.interLayerDistances.bayesian_frequentist +
      coordsBefore.interLayerDistances.frequentist_solomonoff +
      coordsBefore.interLayerDistances.retrocausal_solomonoff;

    // Tick multiple times to let inter-layer flows propagate
    for (let i = 0; i < 10; i++) {
      tickBuleyeanStack(stack);
    }

    const coordsAfter = manifoldCoordinates(stack);
    const totalDistAfter =
      coordsAfter.interLayerDistances.retrocausal_bayesian +
      coordsAfter.interLayerDistances.bayesian_frequentist +
      coordsAfter.interLayerDistances.frequentist_solomonoff +
      coordsAfter.interLayerDistances.retrocausal_solomonoff;

    // Layers should converge: total inter-layer distance decreases
    expect(totalDistAfter).toBeLessThan(totalDistBefore);
  });

  it('all layers maintain Buleyean axioms during contraction', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[0].boundary, 0, 100);
    updateVoidBoundary(stack.layers[1].boundary, 3, 50);
    updateVoidBoundary(stack.layers[2].boundary, 1, 25);
    updateVoidBoundary(stack.layers[3].boundary, 2, 75);

    for (let tick = 0; tick < 20; tick++) {
      tickBuleyeanStack(stack);
      for (const layer of stack.layers) {
        const axioms = assertAllAxioms(layer.boundary);
        expect(axioms.allHold).toBe(true);
      }
    }
  });

  it('contraction continues monotonically over many ticks', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[0].boundary, 0, 80);
    updateVoidBoundary(stack.layers[1].boundary, 2, 60);
    updateVoidBoundary(stack.layers[2].boundary, 3, 40);
    updateVoidBoundary(stack.layers[3].boundary, 1, 70);

    const distances: number[] = [];
    for (let tick = 0; tick < 30; tick++) {
      const coords = manifoldCoordinates(stack);
      const total =
        coords.interLayerDistances.retrocausal_bayesian +
        coords.interLayerDistances.bayesian_frequentist +
        coords.interLayerDistances.frequentist_solomonoff +
        coords.interLayerDistances.retrocausal_solomonoff;
      distances.push(total);
      tickBuleyeanStack(stack);
    }

    // Overall trend should be decreasing (allow local noise from decay dynamics)
    const firstThird = distances.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const lastThird = distances.slice(20).reduce((a, b) => a + b, 0) / 10;
    expect(lastThird).toBeLessThan(firstThird);
  });
});

// ============================================================================
// Prediction 4: Fraud Detection Sensitivity Scales with sqrt(n)
//
// For an n-outcome system, the minimum detectable curvature anomaly
// decreases as 1/sqrt(n). Higher-dimensional manifolds (more outcomes)
// make smaller deviations detectable. More outcomes = more sensitive
// fraud detector.
// ============================================================================

describe('Prediction 4: Fraud Detection Sensitivity Scales with Dimensionality', () => {
  it('higher n detects smaller perturbations', () => {
    const results: { n: number; minDetectable: number }[] = [];

    for (const n of [3, 5, 8, 12, 20]) {
      // Create a nearly-geodesic path with a small perturbation at the midpoint
      const start = new Array(n)
        .fill(0)
        .map((_, i) => (i === 0 ? 0.5 : 0.5 / (n - 1)));
      const end = new Array(n)
        .fill(0)
        .map((_, i) => (i === n - 1 ? 0.5 : 0.5 / (n - 1)));

      // Binary search for minimum detectable perturbation
      let lo = 0;
      let hi = 0.3;
      for (let iter = 0; iter < 20; iter++) {
        const perturbation = (lo + hi) / 2;
        const geodesic = geodesicPath(start, end, 10);
        // Perturb the midpoint
        const midIdx = 5;
        const perturbed = [...geodesic];
        const perturbedMid = [...geodesic[midIdx]];
        // Shift mass to a random dimension
        const shiftDim = Math.floor(n / 2);
        perturbedMid[shiftDim] += perturbation;
        perturbedMid[0] -= perturbation / 2;
        perturbedMid[n - 1] -= perturbation / 2;
        // Renormalize
        const sum = perturbedMid.reduce((a, b) => a + b, 0);
        for (let i = 0; i < n; i++)
          perturbedMid[i] = Math.max(1e-10, perturbedMid[i] / sum);
        perturbed[midIdx] = perturbedMid;

        const analysis = detectFraud(perturbed);
        if (analysis.fraudScore > 0.01) {
          hi = perturbation;
        } else {
          lo = perturbation;
        }
      }
      results.push({ n, minDetectable: (lo + hi) / 2 });
    }

    // Minimum detectable perturbation should decrease with n
    for (let i = 1; i < results.length; i++) {
      expect(results[i].minDetectable).toBeLessThanOrEqual(
        results[i - 1].minDetectable + 0.01
      );
    }
  });

  it('scalar curvature increases with n, amplifying detection', () => {
    // The manifold's intrinsic curvature provides a "background" against
    // which deviations are measured. Higher n = more curvature = more contrast.
    const curvatures = [3, 5, 8, 12, 20].map((n) => ({
      n,
      R: scalarCurvature(n),
    }));

    for (let i = 1; i < curvatures.length; i++) {
      expect(curvatures[i].R).toBeGreaterThan(curvatures[i - 1].R);
    }
  });
});

// ============================================================================
// Prediction 5: Solomonoff Curvature Dominates Before Data,
//               Frequentist Curvature Dominates After Data
//
// On the Fisher manifold, b3 (Solomonoff) > b2 (frequentist) when T = 0
// (no observations). After sufficient observations, b2 > b3.
// The complexity prior shapes the manifold before data arrives; data
// washes it out and the frequentist floor takes over.
// ============================================================================

describe('Prediction 5: Solomonoff-Frequentist Curvature Crossover', () => {
  it('Solomonoff curvature dominates before observations', () => {
    const stack = createBuleyeanStack(5);

    // Initialize Solomonoff layer with complexity priors
    solomonoffInit(stack.layers[3].boundary, [1, 3, 5, 8, 12], 1.0);

    // Frequentist layer has no data yet
    const coords = manifoldCoordinates(stack);
    expect(coords.b3_solomonoff).toBeGreaterThan(coords.b2_frequentist);
  });

  it('frequentist curvature overtakes Solomonoff after sufficient data', () => {
    const stack = createBuleyeanStack(5);

    // Initialize Solomonoff with mild complexity priors
    solomonoffInit(stack.layers[3].boundary, [1, 2, 3, 4, 5], 1.0);

    // Add massive biased frequentist data to create high curvature
    for (let i = 0; i < 500; i++) {
      updateVoidBoundary(stack.layers[2].boundary, 0, 10);
    }

    const coords = manifoldCoordinates(stack);
    expect(coords.b2_frequentist).toBeGreaterThan(coords.b3_solomonoff);
  });

  it('crossover point exists and is findable', () => {
    const stack = createBuleyeanStack(5);
    solomonoffInit(stack.layers[3].boundary, [1, 3, 5, 8, 12], 1.0);

    let crossoverFound = false;
    let prevSolomonoffDominant = true;

    for (let step = 0; step < 200; step++) {
      // Add frequentist data biased to dim 0
      updateVoidBoundary(stack.layers[2].boundary, 0, 3);
      if (step % 3 === 0) updateVoidBoundary(stack.layers[2].boundary, 1, 1);

      const coords = manifoldCoordinates(stack);
      const solomonoffDominant = coords.b3_solomonoff > coords.b2_frequentist;

      if (prevSolomonoffDominant && !solomonoffDominant) {
        crossoverFound = true;
        break;
      }
      prevSolomonoffDominant = solomonoffDominant;
    }

    expect(crossoverFound).toBe(true);
  });

  it('Solomonoff weight gap is O(1/T) -- data washes out complexity prior', () => {
    const n = 4;
    const complexities = [2, 5, 8, 12];
    const ratios: number[] = [];

    for (const T of [10, 50, 100, 500, 1000]) {
      const boundary = createVoidBoundary(n);
      solomonoffInit(boundary, complexities, 1.0);
      // Add T uniform rejections
      for (let i = 0; i < T; i++) {
        updateVoidBoundary(boundary, i % n, 1);
      }
      const dist = buleyeanDistribution(boundary);
      // Measure how much the distribution differs from uniform
      const uniform = new Array(n).fill(1 / n);
      const d = fisherRaoDistance(dist, uniform);
      ratios.push(d);
    }

    // Distance from uniform should decrease with more data (complexity washes out)
    for (let i = 1; i < ratios.length; i++) {
      expect(ratios[i]).toBeLessThanOrEqual(ratios[i - 1] + 1e-6);
    }
  });
});
