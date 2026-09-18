import { describe, expect, it } from 'bun:test';
import {
  kellyFraction,
  kellyDecision,
  buleyeanKelly,
  mleKelly,
  kellyAdversarialDual,
  fractionalKellyFromRange,
  robustKellyFromRange,
  kellyRangeFromPosterior,
} from './kelly';
import { compareFractions, exactFraction } from './rational';
import { exactPosterior } from '../void-crdt';

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

describe('kelly -- exact fraction', () => {
  it('matches (b p - q) / b in closed form', () => {
    expect(kellyFraction(3n, 5n, 1n, 1n)).toEqual({ numerator: 1n, denominator: 5n });
    expect(kellyFraction(1n, 2n, 2n, 1n)).toEqual({ numerator: 1n, denominator: 4n });
    expect(kellyFraction(1n, 2n, 1n, 1n)).toEqual({ numerator: 0n, denominator: 1n });
    expect(kellyFraction(1n, 4n, 1n, 1n)).toEqual({ numerator: -1n, denominator: 2n });
    expect(kellyFraction(2n, 3n, 3n, 2n)).toEqual({ numerator: 4n, denominator: 9n });
  });

  it('reports the recommended non-negative stake', () => {
    const positive = kellyDecision(3n, 5n, 1n, 1n);
    expect(positive.shouldBet).toBe(true);
    expect(positive.recommendedFraction).toEqual({ numerator: 1n, denominator: 5n });
    const negative = kellyDecision(1n, 4n, 1n, 1n);
    expect(negative.shouldBet).toBe(false);
    expect(negative.recommendedFraction).toEqual({ numerator: 0n, denominator: 1n });
  });

  it('rejects malformed probabilities and odds', () => {
    expect(() => kellyFraction(3n, 2n, 1n, 1n)).toThrow(RangeError);
    expect(() => kellyFraction(1n, 2n, 0n, 1n)).toThrow(RangeError);
    expect(() => kellyFraction(1n, 0n, 1n, 1n)).toThrow(RangeError);
  });
});

describe('kelly -- Buleyean add-one sizing', () => {
  it('uses (c_win + 1)/(N + K) and matches the posterior exactly', () => {
    const decision = buleyeanKelly([1, 0], 1n, 0);
    expect(decision.probability).toEqual({ numerator: 2n, denominator: 3n });
    expect(decision.fraction).toEqual({ numerator: 1n, denominator: 3n });
    expect(decision.shouldBet).toBe(true);

    const posterior = exactPosterior([1, 0]);
    expect(decision.probability.numerator * posterior.denominator).toBe(
      posterior.terms[0]!.numerator * decision.probability.denominator,
    );
  });

  it('is exact for random counts and odds (1000 cases)', () => {
    const rand = rng(0xee110001);
    for (let t = 0; t < 1000; t++) {
      const k = 1 + Math.floor(rand() * 6);
      const counts = Array.from({ length: k }, () => Math.floor(rand() * 40));
      const win = Math.floor(rand() * k);
      const bNum = 1n + BigInt(Math.floor(rand() * 4));
      const bDen = 1n + BigInt(Math.floor(rand() * 3));
      const decision = buleyeanKelly(counts, { numerator: bNum, denominator: bDen }, win);
      const posterior = exactPosterior(counts);
      // Probability must equal the add-one posterior for the win category.
      expect(decision.probability.numerator * posterior.denominator).toBe(
        posterior.terms[win]!.numerator * decision.probability.denominator,
      );
      // Fraction must equal (b p - q)/b recomputed from the exact pair.
      const p = decision.probability;
      const recomputed = exactFraction(
        bNum * p.numerator - bDen * (p.denominator - p.numerator),
        bNum * p.denominator,
      );
      expect(decision.fraction).toEqual(recomputed);
      expect(decision.total).toBe(posterior.total);
    }
  });

  it('still prices a zero-count win with positive numerator', () => {
    const decision = buleyeanKelly([0, 5], 5n, 0);
    expect(decision.probability.numerator).toBe(1n);
    expect(decision.probability.denominator).toBe(7n);
    // b=5, p=1/7: (5/7 - 6/7)/5 = -1/7
    expect(decision.fraction).toEqual({ numerator: -1n, denominator: 35n });
    expect(decision.shouldBet).toBe(false);
  });
});

describe('kelly -- adversarial dual (small-sample overconfidence)', () => {
  it('shows the MLE overbetting on 1 win in 1 trial', () => {
    const dual = kellyAdversarialDual([1, 0], 1n, 0);
    expect(dual.fires).toBe(true);
    expect(dual.mleOverbets).toBe(true);
    expect(dual.mle!.probability).toEqual({ numerator: 1n, denominator: 1n });
    expect(dual.mle!.fraction).toEqual({ numerator: 1n, denominator: 1n });
    expect(dual.addOne.probability).toEqual({ numerator: 2n, denominator: 3n });
    expect(dual.addOne.fraction).toEqual({ numerator: 1n, denominator: 3n });
    expect(dual.overbetRatio).toEqual({ numerator: 3n, denominator: 1n });
  });

  it('fires when the MLE stakes more than add-one (500 cases)', () => {
    const rand = rng(0xee110002);
    for (let t = 0; t < 500; t++) {
      const counts = [1, 0];
      const b = 1n + BigInt(Math.floor(rand() * 5));
      const dual = kellyAdversarialDual(counts, b, 0);
      expect(compareFractions(dual.mle!.fraction, dual.addOne.fraction)).toBeGreaterThanOrEqual(0);
      expect(dual.fires).toBe(dual.mleOverbets);
    }
  });

  it('does not fire at N=0 where the MLE is undefined', () => {
    const dual = kellyAdversarialDual([0, 0], 1n, 0);
    expect(dual.fires).toBe(false);
    expect(dual.mle).toBeNull();
  });
});

describe('kelly -- conservative fractional sizing', () => {
  it('sizes at the lower endpoint of a proven range', () => {
    const half = fractionalKellyFromRange(1n, 3n, { numerator: 1n, denominator: 2n });
    expect(half.size).toEqual({ numerator: 1n, denominator: 6n });
    expect(half.shouldBet).toBe(true);

    const negative = fractionalKellyFromRange(-1n, 2n, 1n);
    expect(negative.size).toEqual({ numerator: 0n, denominator: 1n });
    expect(negative.shouldBet).toBe(false);

    expect(() => fractionalKellyFromRange(1n, 2n, 2n)).toThrow(RangeError);
    expect(() => fractionalKellyFromRange(1n, 2n, -1n)).toThrow(RangeError);
  });

  it('robustKellyFromRange takes full Kelly at the lower bound then scales', () => {
    // p=2/3, b=1 -> f*=1/3; times 1/2 -> 1/6
    const robust = robustKellyFromRange(2n, 3n, 1n, 1n, { numerator: 1n, denominator: 2n });
    expect(robust.size).toEqual({ numerator: 1n, denominator: 6n });
    // the aggressive alternative would be 1/3
    const aggressive = robustKellyFromRange(2n, 3n, 1n, 1n, 1n);
    expect(aggressive.size).toEqual({ numerator: 1n, denominator: 3n });
    expect(compareFractions(aggressive.size, robust.size)).toBeGreaterThan(0);
  });

  it('reads the proven posterior range through Kelly', () => {
    const report = kellyRangeFromPosterior([1, 0], 1n, 0);
    expect(report.lowProbability).toEqual({ numerator: 1n, denominator: 3n });
    expect(report.highProbability).toEqual({ numerator: 2n, denominator: 3n });
    expect(report.lowFraction).toEqual({ numerator: -1n, denominator: 3n });
    expect(report.highFraction).toEqual({ numerator: 1n, denominator: 3n });
    expect(report.conservativeFraction).toEqual({ numerator: 0n, denominator: 1n });
    expect(report.aggressiveFraction).toEqual({ numerator: 1n, denominator: 3n });
    expect(report.overbetIfAggressive).toBe(true);
  });
});
