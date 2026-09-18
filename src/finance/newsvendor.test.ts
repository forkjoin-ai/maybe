import { describe, expect, it } from 'bun:test';
import {
  criticalFractile,
  newsvendorOrder,
  newsvendorCost,
  posteriorMeanDemand,
  meanOrderDemand,
  newsvendorMeanDual,
  robustNewsvendorOrder,
  coerceDemands,
} from './newsvendor';
import { compareFractions, exactFraction } from './rational';

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

describe('newsvendor -- critical fractile', () => {
  it('computes cu/(cu+co) exactly', () => {
    expect(criticalFractile(1n, 1n, 1n, 1n)).toEqual({ numerator: 1n, denominator: 2n });
    expect(criticalFractile(9n, 1n, 1n, 1n)).toEqual({ numerator: 9n, denominator: 10n });
    expect(criticalFractile(1n, 1n, 3n, 1n)).toEqual({ numerator: 1n, denominator: 4n });
    expect(criticalFractile(2n, 3n, 1n, 6n)).toEqual({ numerator: 4n, denominator: 5n });
  });

  it('rejects zero total and negative costs', () => {
    expect(() => criticalFractile(0n, 1n, 0n, 1n)).toThrow(RangeError);
    expect(() => criticalFractile(-1n, 1n, 1n, 1n)).toThrow(RangeError);
  });
});

describe('newsvendor -- discrete order quantity', () => {
  it('orders the critical-fractile demand level', () => {
    const balanced = newsvendorOrder([1, 1, 1], [0, 10, 20], 1n, 1n);
    expect(balanced.criticalFractile).toEqual({ numerator: 1n, denominator: 2n });
    expect(balanced.posteriorDenominator).toBe(6n);
    expect(balanced.orderDemand).toBe(10n);
    expect(balanced.achievedMass).toEqual({ numerator: 2n, denominator: 3n });
    expect(balanced.reached).toBe(true);

    const highService = newsvendorOrder([1, 1, 1], [0, 10, 20], 9n, 1n);
    expect(highService.criticalFractile).toEqual({ numerator: 9n, denominator: 10n });
    expect(highService.orderDemand).toBe(20n);
    expect(highService.achievedMass).toEqual({ numerator: 1n, denominator: 1n });
  });

  it('ranks by demand regardless of input order', () => {
    const order = newsvendorOrder([1, 1, 1], [20, 0, 10], 1n, 1n);
    expect(order.orderDemand).toBe(10n);
    expect(order.rungs.map((rung) => rung.demand)).toEqual([0n, 10n, 20n]);
  });

  it('rejects malformed inputs', () => {
    expect(() => newsvendorOrder([], [], 1n, 1n)).toThrow(RangeError);
    expect(() => newsvendorOrder([1, 2], [0], 1n, 1n)).toThrow(RangeError);
    expect(() => coerceDemands([1.5])).toThrow(RangeError);
  });
});

describe('newsvendor -- exact expected cost', () => {
  it('matches the closed-form cost on the uniform example', () => {
    const counts = [1, 1, 1];
    const demands = [0, 10, 20];
    // E[D] = 10, so ordering 0 costs cu * 10 = 90.
    expect(newsvendorCost(counts, demands, 9n, 1n, 0n).cost).toEqual({ numerator: 90n, denominator: 1n });
    // q=10: overage 10/3, underage 30 -> 100/3
    expect(newsvendorCost(counts, demands, 9n, 1n, 10n).cost).toEqual({ numerator: 100n, denominator: 3n });
    // q=20: overage 20/3 + 10/3 = 10
    expect(newsvendorCost(counts, demands, 9n, 1n, 20n).cost).toEqual({ numerator: 10n, denominator: 1n });
  });

  it('is non-negative on random inputs (500 cases)', () => {
    const rand = rng(0x11e05001);
    for (let t = 0; t < 500; t++) {
      const k = 1 + Math.floor(rand() * 6);
      const counts = Array.from({ length: k }, () => Math.floor(rand() * 20));
      const demands = Array.from({ length: k }, () => Math.floor(rand() * 100)).sort((a, b) => a - b);
      const cu = 1n + BigInt(Math.floor(rand() * 5));
      const co = 1n + BigInt(Math.floor(rand() * 5));
      const order = demands[Math.floor(rand() * k)]!;
      const cost = newsvendorCost(counts, demands, cu, co, order);
      expect(compareFractions(cost.cost, 0n)).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(cost.value)).toBe(true);
    }
  });
});

describe('newsvendor -- mean and adversarial dual', () => {
  it('computes the posterior mean and nearest demand', () => {
    const mean = posteriorMeanDemand([1, 1, 1], [0, 10, 20]);
    expect(mean).toEqual({ numerator: 10n, denominator: 1n });
    expect(meanOrderDemand([1, 1, 1], [0, 10, 20])).toBe(10n);
  });

  it('fires when ordering the mean loses to the critical fractile', () => {
    const dual = newsvendorMeanDual([1, 1, 1], [0, 10, 20], 9n, 1n);
    expect(dual.criticalOrder).toBe(20n);
    expect(dual.meanOrder).toBe(10n);
    expect(dual.meanIsSuboptimal).toBe(true);
    expect(dual.fires).toBe(true);
    expect(dual.meanCost).toEqual({ numerator: 100n, denominator: 3n });
    expect(dual.criticalCost).toEqual({ numerator: 10n, denominator: 1n });
    expect(dual.costPenalty).toEqual({ numerator: 70n, denominator: 3n });
  });

  it('does not fire when cu = co and the mean is the fractile', () => {
    const dual = newsvendorMeanDual([1, 1, 1], [0, 10, 20], 1n, 1n);
    expect(dual.fires).toBe(false);
    expect(dual.criticalOrder).toBe(10n);
    expect(dual.meanOrder).toBe(10n);
    expect(dual.costsDiffer).toBe(false);
  });
});

describe('newsvendor -- robust collapse band', () => {
  it('returns a sound order range that contains the point', () => {
    const balanced = robustNewsvendorOrder([1, 1, 1], [0, 10, 20], 1n, 1n);
    // R = 1, W = 6, gamma = 1/2
    // optimistic: (1+1) r / 6 >= 1/2 -> r = 2 (demand 10)
    expect(balanced.optimistic.orderDemand).toBe(10n);
    // guaranteed: r / 6 >= 1/2 -> r = 3 (demand 20)
    expect(balanced.guaranteed.orderDemand).toBe(20n);
    expect(balanced.point).toBe(10n);
    expect(balanced.sound).toBe(true);
    expect(balanced.guaranteed.reached).toBe(true);
  });

  it('reports an unreachable lower edge honestly', () => {
    const highService = robustNewsvendorOrder([1, 1, 1], [0, 10, 20], 9n, 1n);
    expect(highService.criticalFractile).toEqual({ numerator: 9n, denominator: 10n });
    // guaranteed requires r/6 >= 9/10, i.e. r >= 5.4 > 3: unreachable.
    expect(highService.guaranteed.reached).toBe(false);
    expect(highService.guaranteed.orderDemand).toBe(20n);
    expect(highService.point).toBe(20n);
    expect(highService.sound).toBe(true);
  });

  it('keeps the point inside the sound band on random inputs (500 cases)', () => {
    const rand = rng(0x11e05002);
    for (let t = 0; t < 500; t++) {
      const k = 2 + Math.floor(rand() * 5);
      const counts = Array.from({ length: k }, () => Math.floor(rand() * 15));
      const demands = Array.from({ length: k }, () => Math.floor(rand() * 100)).sort((a, b) => a - b);
      const cu = 1n + BigInt(Math.floor(rand() * 6));
      const co = 1n + BigInt(Math.floor(rand() * 6));
      const robust = robustNewsvendorOrder(counts, demands, cu, co);
      expect(robust.sound).toBe(true);
      const low = robust.optimistic.orderDemand < robust.guaranteed.orderDemand
        ? robust.optimistic.orderDemand
        : robust.guaranteed.orderDemand;
      const high = robust.optimistic.orderDemand < robust.guaranteed.orderDemand
        ? robust.guaranteed.orderDemand
        : robust.optimistic.orderDemand;
      expect(robust.point).toBeGreaterThanOrEqual(low);
      expect(robust.point).toBeLessThanOrEqual(high);
    }
  });
});
