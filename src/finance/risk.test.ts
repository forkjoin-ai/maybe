import { describe, expect, it } from 'bun:test';
import {
  exactExpectation,
  neverZeroTail,
  mlePosterior,
  tailAdversarialDual,
  tailMassBound,
  assertLossSandwich,
  coerceLosses,
} from './risk';
import { compareFractions, exactFraction } from './rational';
import { assertLaplaceIdentity as assertVoidLaplaceIdentity, exactPosterior } from '../void-crdt';

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

function randomCounts(rand: () => number, maxK = 8, maxCount = 50): number[] {
  const k = 1 + Math.floor(rand() * maxK);
  return Array.from({ length: k }, () => Math.floor(rand() * maxCount));
}

function randomLosses(rand: () => number, k: number, maxLoss = 1000): number[] {
  return Array.from({ length: k }, () => Math.floor(rand() * maxLoss));
}

describe('risk -- exact expectation', () => {
  it('matches the closed-form add-one expectation', () => {
    // counts [3,1], losses [0,100]: N=4, K=2, D=6
    // E = (0*4 + 100*2)/6 = 200/6 = 100/3
    const expectation = exactExpectation([3, 1], [0, 100]);
    expect(expectation.numerator).toBe(100n);
    expect(expectation.denominator).toBe(3n);
    expect(expectation.rawNumerator).toBe(200n);
    expect(expectation.posteriorDenominator).toBe(6n);
    expect(expectation.total).toBe(4n);
    expect(expectation.categories).toBe(2);
    expect(expectation.terms[0]!.probability).toEqual({ numerator: 2n, denominator: 3n });
    expect(expectation.terms[1]!.probability).toEqual({ numerator: 1n, denominator: 3n });
    expect(expectation.terms[1]!.contribution).toEqual({ numerator: 100n, denominator: 3n });
  });

  it('reuses the void-crdt posterior and holds the urn/Laplace identity (2000 cases)', () => {
    const rand = rng(0x5eed0001);
    for (let t = 0; t < 2000; t++) {
      const counts = randomCounts(rand);
      const losses = randomLosses(rand, counts.length);
      const expectation = exactExpectation(counts, losses);
      const posterior = exactPosterior(counts);
      for (let i = 0; i < counts.length; i++) {
        expect(expectation.terms[i]!.count).toBe(BigInt(counts[i]!));
      }
      expect(expectation.posteriorDenominator).toBe(posterior.denominator);
      expect(expectation.rawNumerator).toBe(
        losses.reduce((sum, loss, i) => sum + BigInt(loss) * (BigInt(counts[i]!) + 1n), 0n),
      );
      const report = assertVoidLaplaceIdentity(counts);
      expect(report.ok).toBe(true);
    }
  });

  it('stays exact beyond 2^53', () => {
    const huge = 10n ** 30n;
    const expectation = exactExpectation([huge, 0n], [0n, 10n ** 20n]);
    // N = huge, K = 2, D = huge + 2, raw numerator = 1 * 10^20
    expect(expectation.rawNumerator).toBe(10n ** 20n);
    expect(expectation.posteriorDenominator).toBe(huge + 2n);
    expect(expectation.total).toBe(huge);
    expect(Number.isFinite(expectation.value)).toBe(true);
  });

  it('rejects malformed inputs', () => {
    expect(() => exactExpectation([], [])).toThrow(RangeError);
    expect(() => exactExpectation([1, 2], [1])).toThrow(RangeError);
    expect(() => coerceLosses([1.5])).toThrow(RangeError);
  });
});

describe('risk -- sandwich and monotonicity', () => {
  it('holds minLoss <= E[loss] <= maxLoss on random vectors (2000 cases)', () => {
    const rand = rng(0x5eed0002);
    for (let t = 0; t < 2000; t++) {
      const counts = randomCounts(rand);
      const losses = randomLosses(rand, counts.length).map((loss, i) => (i % 3 === 0 ? -loss : loss));
      const sandwich = assertLossSandwich(counts, losses);
      expect(sandwich.ok).toBe(true);
      expect(compareFractions(sandwich.expectation, exactFraction(sandwich.minLoss, 1n))).toBeGreaterThanOrEqual(0);
      expect(compareFractions(sandwich.expectation, exactFraction(sandwich.maxLoss, 1n))).toBeLessThanOrEqual(0);
    }
  });

  it('is monotone non-decreasing when one loss increases (1000 cases)', () => {
    const rand = rng(0x5eed0003);
    for (let t = 0; t < 1000; t++) {
      const counts = randomCounts(rand, 6, 30);
      const losses = randomLosses(rand, counts.length);
      const raised = [...losses];
      const index = Math.floor(rand() * counts.length);
      raised[index] = raised[index]! + 1 + Math.floor(rand() * 50);
      const before = exactExpectation(counts, losses);
      const after = exactExpectation(counts, raised);
      expect(compareFractions(after, before)).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('risk -- never-zero tail', () => {
  it('prices an unobserved catastrophe at 1/(N+K), never 0', () => {
    const report = neverZeroTail([5, 0, 0]);
    // N=5, K=3, D=8, weights [6,1,1]
    expect(report.posteriorDenominator).toBe(8n);
    expect(report.weights).toEqual([6n, 1n, 1n]);
    expect(report.minWeight).toBe(1n);
    expect(report.minWeightState).toBe(1);
    expect(report.everyStateAlive).toBe(true);
    expect(report.unobserved).toEqual([1, 2]);
    expect(report.unobservedFloor).toEqual({ numerator: 1n, denominator: 8n });
    expect(report.unobservedFloorValue).toBeCloseTo(1 / 8, 12);
  });

  it('keeps every weight >= 1 on random vectors (2000 cases)', () => {
    const rand = rng(0x5eed0004);
    for (let t = 0; t < 2000; t++) {
      const counts = randomCounts(rand);
      const report = neverZeroTail(counts);
      expect(report.everyStateAlive).toBe(true);
      expect(report.minWeight).toBeGreaterThanOrEqual(1n);
      expect(report.weights.length).toBe(counts.length);
      for (const weight of report.weights) expect(weight).toBeGreaterThanOrEqual(1n);
    }
  });

  it('treats N=0 as the uniform 1/K floor', () => {
    for (const k of [1, 2, 5]) {
      const counts = Array.from({ length: k }, () => 0);
      const report = neverZeroTail(counts);
      expect(report.posteriorDenominator).toBe(BigInt(k));
      expect(report.unobservedFloor).toEqual({ numerator: 1n, denominator: BigInt(k) });
      expect(report.unobserved.length).toBe(k);
    }
  });
});

describe('risk -- adversarial dual (MLE zero-prices the tail)', () => {
  it('fires on an unobserved state and names the infinite underpricing', () => {
    const dual = tailAdversarialDual([5, 0, 0]);
    expect(dual.fires).toBe(true);
    expect(dual.mleDefined).toBe(true);
    expect(dual.unobserved).toEqual([1, 2]);
    expect(dual.addOneFloor).toEqual({ numerator: 1n, denominator: 8n });
    expect(dual.mlePrice).toEqual({ numerator: 0n, denominator: 1n });
    expect(dual.underpricingInfinite).toBe(true);

    const mle = mlePosterior([5, 0, 0]);
    expect(mle.defined).toBe(true);
    expect(mle.zeroStates).toEqual([1, 2]);
    expect(mle.probabilities[1]).toEqual({ numerator: 0n, denominator: 1n });
  });

  it('fires on random vectors with an unobserved state (500 cases)', () => {
    const rand = rng(0x5eed0005);
    for (let t = 0; t < 500; t++) {
      const k = 2 + Math.floor(rand() * 6);
      const counts = Array.from({ length: k }, (_, i) => (i === k - 1 ? 0 : 1 + Math.floor(rand() * 20)));
      const dual = tailAdversarialDual(counts);
      expect(dual.fires).toBe(true);
      expect(dual.underpricingInfinite).toBe(true);
      expect(dual.mlePrice.numerator).toBe(0n);
      expect(dual.addOneFloor.numerator).toBe(1n);
    }
  });

  it('does not fire when every state was observed', () => {
    const dual = tailAdversarialDual([2, 3, 1]);
    expect(dual.fires).toBe(false);
    expect(dual.unobserved).toEqual([]);
    expect(dual.underpricingInfinite).toBe(false);
  });

  it('reports the MLE as undefined at N=0', () => {
    const mle = mlePosterior([0, 0]);
    expect(mle.defined).toBe(false);
    expect(mle.zeroStates).toEqual([0, 1]);
    const dual = tailAdversarialDual([0, 0]);
    expect(dual.fires).toBe(true);
    expect(dual.mleDefined).toBe(false);
  });
});

describe('risk -- tail mass bound', () => {
  it('computes the exact mass strictly above a threshold', () => {
    const counts = [1, 2, 1];
    const losses = [0, 10, 100];
    // N=4, K=3, D=7
    const above10 = tailMassBound(counts, losses, 10);
    expect(above10.tail).toEqual([2]);
    expect(above10.numerator).toBe(2n);
    expect(above10.denominator).toBe(7n);
    expect(above10.rawNumerator).toBe(2n);

    const above0 = tailMassBound(counts, losses, 0);
    expect(above0.tail).toEqual([1, 2]);
    expect(above0.numerator).toBe(5n);
    expect(above0.denominator).toBe(7n);

    const aboveNineHalf = tailMassBound(counts, losses, exactFraction(19n, 2n));
    expect(aboveNineHalf.tail).toEqual([1, 2]);
    expect(aboveNineHalf.numerator).toBe(5n);
  });

  it('is a non-negative mass bounded by one on random vectors (1000 cases)', () => {
    const rand = rng(0x5eed0006);
    for (let t = 0; t < 1000; t++) {
      const counts = randomCounts(rand);
      const losses = randomLosses(rand, counts.length, 100);
      const threshold = Math.floor(rand() * 100);
      const bound = tailMassBound(counts, losses, threshold);
      expect(compareFractions(bound, 0n)).toBeGreaterThanOrEqual(0);
      expect(compareFractions(bound, 1n)).toBeLessThanOrEqual(0);
      expect(bound.tail.length).toBe(losses.filter((loss) => loss > threshold).length);
    }
  });
});
