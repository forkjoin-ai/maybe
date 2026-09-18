import { describe, expect, it } from 'bun:test';
import {
  acceptancePosterior,
  rankDraftHeads,
  draftLengthPolicy,
  expectedAcceptance,
  blockAcceptanceProfile,
  recommendedBlockLength,
  assertLaplaceIdentity,
  ACCEPTANCE_FLOOR,
} from './speculative-acceptance';
import { posterior as voidCrdtPosterior } from './void-crdt';

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

function randomCounts(rand: () => number, maxK = 12, maxCount = 40): number[] {
  const k = 1 + Math.floor(rand() * maxK);
  return Array.from({ length: k }, () => Math.floor(rand() * maxCount));
}

describe('speculative acceptance -- exact posterior', () => {
  it('matches the closed-form add-one example', () => {
    const post = acceptancePosterior([3, 1]);
    expect(post.k).toBe(2);
    expect(post.total).toBe(4n);
    expect(post.denominator).toBe(6n);
    expect(post.uniform).toBe(false);
    expect(post.terms.map((term) => term.accepts)).toEqual([3n, 1n]);
    expect(post.terms.map((term) => term.numerator)).toEqual([4n, 2n]);
    expect(post.terms[0]!.posterior).toBeCloseTo(4 / 6, 12);
    expect(post.terms[1]!.posterior).toBeCloseTo(2 / 6, 12);
  });

  it('is uniform 1/K when A = 0', () => {
    for (const k of [1, 2, 3, 5, 11]) {
      const counts = Array.from({ length: k }, () => 0);
      const post = acceptancePosterior(counts);
      expect(post.total).toBe(0n);
      expect(post.denominator).toBe(BigInt(k));
      expect(post.uniform).toBe(true);
      for (let i = 0; i < k; i++) {
        expect(post.terms[i]!.numerator).toBe(1n);
        expect(post.terms[i]!.posterior).toBeCloseTo(1 / k, 12);
      }
    }
  });

  it('holds the urn/Laplace identity for random vectors (2000 cases)', () => {
    const rand = rng(0x5eed5eed);
    for (let t = 0; t < 2000; t++) {
      const counts = randomCounts(rand);
      const post = acceptancePosterior(counts);
      let expected = 0n;
      for (let i = 0; i < counts.length; i++) {
        expect(post.terms[i]!.numerator).toBe(BigInt(counts[i]!) + 1n);
        expected += BigInt(counts[i]!) + 1n;
      }
      expect(expected).toBe(post.denominator);

      const report = assertLaplaceIdentity(counts);
      expect(report.ok).toBe(true);
      expect(report.floorPreserved).toBe(true);
      expect(report.numeratorSum).toBe(post.denominator);

      // Consistency with the void-crdt posterior on the same counts.
      expect(post.terms.map((term) => term.posterior)).toEqual(voidCrdtPosterior(counts));
    }
  });

  it('stays exact and finite beyond 2^53', () => {
    const big = 10n ** 30n;
    const post = acceptancePosterior([big, 1n]);
    expect(post.total).toBe(big + 1n);
    expect(post.denominator).toBe(big + 3n);
    expect(post.terms[0]!.numerator).toBe(big + 1n);
    expect(Number.isFinite(post.terms[0]!.posterior)).toBe(true);
    expect(post.terms[0]!.posterior).toBeGreaterThan(0.99);

    const report = assertLaplaceIdentity([big, 1n]);
    expect(report.ok).toBe(true);
    expect(report.numeratorSum).toBe(big + 3n);
  });

  it('rejects malformed and empty inputs', () => {
    expect(() => acceptancePosterior([])).toThrow(RangeError);
    expect(() => acceptancePosterior([1, -1])).toThrow(RangeError);
    expect(() => acceptancePosterior([1, 1.5])).toThrow(RangeError);
  });
});

describe('speculative acceptance -- ranking', () => {
  it('ranks by descending count, ties by candidate index', () => {
    const ranking = rankDraftHeads([2, 2, 5]);
    expect(ranking.map((entry) => entry.candidate)).toEqual([2, 0, 1]);
    expect(ranking.map((entry) => entry.rank)).toEqual([0, 1, 2]);
    expect(ranking.map((entry) => entry.numerator)).toEqual([6n, 3n, 3n]);
    // A = 9, K = 3, so the shared denominator is A + K = 12.
    expect(ranking[0]!.denominator).toBe(12n);
    expect(ranking[0]!.posterior).toBeCloseTo(6 / 12, 12);
  });

  it('retains every candidate (the floor never prunes)', () => {
    const rand = rng(0xf100f100);
    for (let t = 0; t < 500; t++) {
      const counts = randomCounts(rand);
      const ranking = rankDraftHeads(counts);
      expect(ranking.length).toBe(counts.length);
      expect(new Set(ranking.map((entry) => entry.candidate)).size).toBe(counts.length);
      for (const entry of ranking) {
        expect(entry.numerator).toBeGreaterThanOrEqual(ACCEPTANCE_FLOOR);
        expect(entry.posterior).toBeGreaterThan(0);
      }
      for (let i = 1; i < ranking.length; i++) {
        expect(ranking[i - 1]!.posterior).toBeGreaterThanOrEqual(ranking[i]!.posterior);
      }
    }
  });

  it('reads one candidate through expectedAcceptance', () => {
    const entry = expectedAcceptance([3, 1], 1);
    expect(entry.candidate).toBe(1);
    expect(entry.accepts).toBe(1n);
    expect(entry.numerator).toBe(2n);
    expect(entry.denominator).toBe(6n);
    expect(entry.posterior).toBeCloseTo(1 / 3, 12);
    expect(() => expectedAcceptance([3, 1], 2)).toThrow(RangeError);
    expect(() => expectedAcceptance([3, 1], -1)).toThrow(RangeError);
  });
});

describe('speculative acceptance -- draft length policy', () => {
  it('matches the closed-form cumulative example', () => {
    const low = draftLengthPolicy([3, 1], { minLen: 1, maxLen: 2, targetMass: 0.6 });
    expect(low.length).toBe(1);
    expect(low.reached).toBe(true);
    expect(low.achievedMass).toBeCloseTo(4 / 6, 12);
    expect(low.cumulativeNumerator).toBe(4n);
    expect(low.denominator).toBe(6n);

    const high = draftLengthPolicy([3, 1], { minLen: 1, maxLen: 2, targetMass: 0.7 });
    expect(high.length).toBe(2);
    expect(high.reached).toBe(true);
    expect(high.achievedMass).toBeCloseTo(1, 12);

    const full = draftLengthPolicy([3, 1], { minLen: 1, maxLen: 2, targetMass: 1 });
    expect(full.length).toBe(2);
    expect(full.reached).toBe(true);

    const atFloor = draftLengthPolicy([3, 1], { minLen: 1, maxLen: 2, targetMass: 0 });
    expect(atFloor.length).toBe(1);
    expect(atFloor.reached).toBe(true);
  });

  it('is monotone non-decreasing in targetMass (500 cases)', () => {
    const rand = rng(0x1e5f1e5f);
    for (let t = 0; t < 500; t++) {
      const counts = randomCounts(rand, 16, 50);
      let previous = -1;
      for (let step = 0; step <= 20; step++) {
        const targetMass = step / 20;
        const decision = draftLengthPolicy(counts, {
          minLen: 1,
          maxLen: counts.length,
          targetMass,
        });
        expect(decision.length).toBeGreaterThanOrEqual(previous);
        expect(decision.length).toBeGreaterThanOrEqual(1);
        expect(decision.length).toBeLessThanOrEqual(counts.length);
        previous = decision.length;
      }
    }
  });

  it('respects minLen and maxLen, and reports an unreachable target', () => {
    const counts = [5, 0, 0, 0, 0];
    const bounded = draftLengthPolicy(counts, { minLen: 2, maxLen: 3, targetMass: 0.5 });
    expect(bounded.length).toBe(2);
    expect(bounded.minLen).toBe(2);
    expect(bounded.maxLen).toBe(3);

    const capped = draftLengthPolicy(counts, { minLen: 1, maxLen: 2, targetMass: 1 });
    expect(capped.length).toBe(2);
    expect(capped.reached).toBe(false);
    expect(capped.achievedMass).toBeCloseTo(7 / 10, 12);
  });

  it('rejects an empty candidate set', () => {
    expect(() => draftLengthPolicy([])).toThrow(RangeError);
  });
});

describe('speculative acceptance -- block drafter', () => {
  it('matches the closed-form block profile', () => {
    const profile = blockAcceptanceProfile([3, 1]);
    expect(profile.positions).toBe(2);
    expect(profile.total).toBe(4n);
    expect(profile.denominator).toBe(6n);
    expect(profile.uniform).toBe(false);
    expect(profile.terms[0]!.posterior).toBeCloseTo(4 / 6, 12);
    expect(profile.terms[1]!.posterior).toBeCloseTo(2 / 6, 12);
    expect(profile.terms[0]!.cumulative).toBeCloseTo(4 / 6, 12);
    expect(profile.terms[1]!.cumulative).toBeCloseTo(1, 12);
    expect(profile.terms[0]!.survival).toBeCloseTo(2 / 6, 12);
    expect(profile.terms[1]!.survival).toBeCloseTo(0, 12);
    expect(profile.expectedLength).toBeCloseTo(4 / 3, 12);
  });

  it('is uniform with expected length (B + 1) / 2 at A = 0', () => {
    for (const b of [1, 2, 4, 7, 11]) {
      const profile = blockAcceptanceProfile(Array.from({ length: b }, () => 0));
      expect(profile.uniform).toBe(true);
      for (const term of profile.terms) {
        expect(term.posterior).toBeCloseTo(1 / b, 12);
      }
      expect(profile.expectedLength).toBeCloseTo((b + 1) / 2, 12);
    }
  });

  it('recommends a block length monotone in targetMass (500 cases)', () => {
    const rand = rng(0xb10cb10c);
    for (let t = 0; t < 500; t++) {
      const counts = randomCounts(rand, 16, 60);
      let previous = -1;
      for (let step = 0; step <= 20; step++) {
        const targetMass = step / 20;
        const decision = recommendedBlockLength(counts, {
          minLen: 1,
          maxLen: counts.length,
          targetMass,
        });
        expect(decision.length).toBeGreaterThanOrEqual(previous);
        expect(decision.length).toBeLessThanOrEqual(counts.length);
        previous = decision.length;
      }
    }
  });

  it('keeps every block position alive at the floor', () => {
    const profile = blockAcceptanceProfile([0, 7, 0]);
    for (const term of profile.terms) {
      expect(term.numerator).toBeGreaterThanOrEqual(ACCEPTANCE_FLOOR);
      expect(term.posterior).toBeGreaterThan(0);
    }
    expect(profile.expectedLength).toBeGreaterThan(1);
    expect(Number.isFinite(profile.expectedLength)).toBe(true);

    const report = assertLaplaceIdentity([0, 7, 0]);
    expect(report.floorPreserved).toBe(true);
  });
});
