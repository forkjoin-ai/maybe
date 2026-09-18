import { describe, expect, it } from 'bun:test';
import {
  urnTotal,
  urnRejections,
  buleyeanWeightFromUrn,
  laplacePosterior,
  buleyeanPosteriorFromUrn,
  empiricalFrequency,
  assertUrnLaplaceIdentity,
  type Urn,
} from './urn';

/** Deterministic xorshift32 so a failing case is reproducible. */
function rng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state / 0x100000000;
  };
}

describe('urn -- classic count identity', () => {
  it('reads rejections as N - counts[i] and the Buleyean weight as counts[i] + 1', () => {
    const counts = [0, 1, 7, 2];
    expect(urnTotal(counts)).toBe(10);
    expect(urnRejections(counts)).toEqual([10, 9, 3, 8]);
    for (let i = 0; i < counts.length; i++) {
      expect(buleyeanWeightFromUrn(counts, i)).toBe(counts[i] + 1);
    }
  });

  it('holds exactly for random count vectors (2000 cases)', () => {
    const rand = rng(0x5eed1234);
    for (let t = 0; t < 2000; t++) {
      const k = 1 + Math.floor(rand() * 12);
      const counts = Array.from({ length: k }, () => Math.floor(rand() * 50));
      const urn: Urn = { counts };
      const N = urnTotal(urn.counts);
      const rejections = urnRejections(urn.counts);

      let integerWeightSum = 0;
      for (let i = 0; i < k; i++) {
        expect(rejections[i]).toBe(N - urn.counts[i]);
        const weight = buleyeanWeightFromUrn(urn.counts, i);
        expect(Number.isInteger(weight)).toBe(true);
        expect(weight).toBe(urn.counts[i] + 1);
        integerWeightSum += urn.counts[i] + 1;
      }

      // The exact integer backbone: sum_i (counts[i] + 1) === N + K.
      expect(integerWeightSum).toBe(N + k);

      const report = assertUrnLaplaceIdentity(urn.counts);
      expect(report.ok).toBe(true);
      expect(report.detail).toContain(`N + K = ${N + k}`);
    }
  });

  it('reports failure exactly when the clamp stops tracking counts[i] + 1', () => {
    // Admissible urns always pass: v_i = N - counts[i] <= N, so the clamp is
    // inert and the weight is counts[i] + 1. A count above N (only reachable
    // with an invalid negative bucket) drives v_i > N and breaks it.
    const admissible = assertUrnLaplaceIdentity([2, 3, 4]);
    expect(admissible.ok).toBe(true);

    const broken = assertUrnLaplaceIdentity([-1, 5]);
    expect(broken.ok).toBe(false);
    expect(broken.detail).toContain('!==');
  });
});

describe('urn -- Laplace posterior', () => {
  it('gives the uniform 1/K posterior when N = 0', () => {
    for (const k of [1, 2, 3, 5, 11]) {
      const counts = Array.from({ length: k }, () => 0);
      expect(urnTotal(counts)).toBe(0);

      const laplace = laplacePosterior(counts);
      const buleyean = buleyeanPosteriorFromUrn(counts);
      for (let i = 0; i < k; i++) {
        expect(laplace[i]).toBeCloseTo(1 / k, 12);
        expect(buleyean[i]).toBeCloseTo(1 / k, 12);
      }

      // No draws means no empirical frequency at all.
      expect(empiricalFrequency(counts).every((frequency) => frequency === null)).toBe(true);
    }
  });

  it('buleyeanPosterior === laplacePosterior numerically (1000 cases)', () => {
    const rand = rng(4242);
    for (let t = 0; t < 1000; t++) {
      const k = 1 + Math.floor(rand() * 12);
      const counts = Array.from({ length: k }, () => Math.floor(rand() * 40));
      const laplace = laplacePosterior(counts);
      const buleyean = buleyeanPosteriorFromUrn(counts);

      // Both sides share exact integer numerator counts[i] + 1 and exact
      // denominator N + K, so the double is bit-identical.
      let mass = 0;
      for (let i = 0; i < k; i++) {
        expect(buleyean[i]).toBe(laplace[i]);
        expect(buleyean[i]).toBeCloseTo((counts[i] + 1) / (urnTotal(counts) + k), 12);
        mass += buleyean[i];
      }
      expect(mass).toBeCloseTo(1, 12);
      expect(laplace.reduce((a, p) => a + p, 0)).toBeCloseTo(1, 12);
    }
  });

  it('gives empirical frequency counts[i]/N and null only at N = 0', () => {
    expect(empiricalFrequency([1, 1, 2])).toEqual([0.25, 0.25, 0.5]);
    expect(empiricalFrequency([0, 0])).toEqual([null, null]);
    expect(empiricalFrequency([0, 4])).toEqual([0, 1]);
  });
});
