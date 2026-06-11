import { describe, expect, it } from 'bun:test';
import {
  buleyeanDistribution,
  buleyeanWeights,
  sampleBuleyean,
  assertPositivity,
  assertNormalization,
  assertMonotonicity,
  assertAllAxioms,
  klDivergence,
} from './buleyean';
import {
  createVoidBoundary,
  updateVoidBoundary,
  complementDistribution,
} from '@a0n/gnosis/src/void';

describe('buleyeanDistribution', () => {
  it('returns uniform distribution for empty boundary', () => {
    const boundary = createVoidBoundary(4);
    const dist = buleyeanDistribution(boundary);
    expect(dist).toHaveLength(4);
    for (const p of dist) {
      expect(p).toBeCloseTo(0.25, 10);
    }
  });

  it('returns empty array for zero-dimension boundary', () => {
    const boundary = createVoidBoundary(0);
    expect(buleyeanDistribution(boundary)).toEqual([]);
  });

  it('gives less weight to more-rejected dimensions', () => {
    const boundary = createVoidBoundary(3);
    updateVoidBoundary(boundary, 0, 10); // heavily rejected
    updateVoidBoundary(boundary, 1, 2); // lightly rejected
    // dimension 2: never rejected
    const dist = buleyeanDistribution(boundary);
    expect(dist[2]).toBeGreaterThan(dist[1]);
    expect(dist[1]).toBeGreaterThan(dist[0]);
  });

  it('always sums to 1', () => {
    const boundary = createVoidBoundary(5);
    updateVoidBoundary(boundary, 0, 100);
    updateVoidBoundary(boundary, 3, 50);
    updateVoidBoundary(boundary, 4, 1);
    const dist = buleyeanDistribution(boundary);
    const sum = dist.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it('all probabilities are positive', () => {
    const boundary = createVoidBoundary(4);
    // Even with extreme asymmetry
    updateVoidBoundary(boundary, 0, 10000);
    const dist = buleyeanDistribution(boundary);
    for (const p of dist) {
      expect(p).toBeGreaterThan(0);
    }
  });
});

describe('buleyeanWeights', () => {
  it('computes T - v_i + 1 for each dimension', () => {
    const boundary = createVoidBoundary(3);
    updateVoidBoundary(boundary, 0, 5);
    updateVoidBoundary(boundary, 1, 3);
    // T = 8, v = [5, 3, 0]
    const weights = buleyeanWeights(boundary);
    expect(weights[0]).toBe(8 - 5 + 1); // 4
    expect(weights[1]).toBe(8 - 3 + 1); // 6
    expect(weights[2]).toBe(8 - 0 + 1); // 9
  });
});

describe('sampleBuleyean', () => {
  it('samples from the distribution', () => {
    const boundary = createVoidBoundary(3);
    updateVoidBoundary(boundary, 0, 100);
    // dimension 0 heavily rejected, should be sampled least
    const counts = [0, 0, 0];
    const rng = () => Math.random();
    for (let i = 0; i < 1000; i++) {
      counts[sampleBuleyean(boundary, rng)]++;
    }
    // dimension 0 should have fewest samples
    expect(counts[0]).toBeLessThan(counts[1]);
    expect(counts[0]).toBeLessThan(counts[2]);
  });
});

describe('axioms', () => {
  it('positivity holds on empty boundary', () => {
    expect(assertPositivity(createVoidBoundary(5))).toBe(true);
  });

  it('positivity holds on extreme boundary', () => {
    const boundary = createVoidBoundary(3);
    updateVoidBoundary(boundary, 0, 1e6);
    expect(assertPositivity(boundary)).toBe(true);
  });

  it('normalization holds', () => {
    const boundary = createVoidBoundary(4);
    updateVoidBoundary(boundary, 1, 42);
    updateVoidBoundary(boundary, 3, 7);
    expect(assertNormalization(boundary)).toBe(true);
  });

  it('monotonicity holds', () => {
    const boundary = createVoidBoundary(4);
    updateVoidBoundary(boundary, 0, 10);
    updateVoidBoundary(boundary, 1, 5);
    updateVoidBoundary(boundary, 2, 1);
    expect(assertMonotonicity(boundary)).toBe(true);
  });

  it('monotonicity holds with equal counts', () => {
    const boundary = createVoidBoundary(3);
    updateVoidBoundary(boundary, 0, 5);
    updateVoidBoundary(boundary, 1, 5);
    updateVoidBoundary(boundary, 2, 5);
    expect(assertMonotonicity(boundary)).toBe(true);
  });

  it('all axioms hold after random walk', () => {
    const boundary = createVoidBoundary(6);
    const rng = () => Math.random();
    for (let i = 0; i < 100; i++) {
      const dim = Math.floor(rng() * 6);
      const mag = rng() * 10;
      updateVoidBoundary(boundary, dim, mag);
    }
    const result = assertAllAxioms(boundary);
    expect(result.allHold).toBe(true);
  });
});

describe('klDivergence', () => {
  it('Buleyean differs from gnosis softmax', () => {
    const boundary = createVoidBoundary(4);
    updateVoidBoundary(boundary, 0, 10);
    updateVoidBoundary(boundary, 2, 3);

    const buleyean = buleyeanDistribution(boundary);
    const softmax = complementDistribution(boundary, 3.0);
    const kl = klDivergence(buleyean, softmax);
    // They should differ -- KL > 0
    expect(kl).toBeGreaterThan(0);
  });

  it('KL is 0 for identical distributions', () => {
    const a = [0.25, 0.25, 0.25, 0.25];
    expect(klDivergence(a, a)).toBeCloseTo(0, 10);
  });
});
