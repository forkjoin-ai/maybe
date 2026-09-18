import { describe, expect, it } from 'bun:test';
import {
  findsArbitrage,
  consensusResidual,
  twoVenueArbitrage,
  directedJointMatrix,
  antisymmetricResidualMatrix,
  directedJointIsSymmetric,
  directionBlindFindsArbitrage,
  arbitrageAdversarialDual,
  asRationalMatrix,
} from './arbitrage';
import { compareFractions, exactFraction, type ExactFraction } from './rational';

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

describe('arbitrage -- consensus law', () => {
  it('accepts a consistent 2x2 matrix', () => {
    const cond = [
      [1, 0.5],
      [0.5, 1],
    ];
    const report = findsArbitrage(cond, [1, 1]);
    expect(report.arbitrage).toBe(false);
    expect(report.violations).toEqual([]);
    expect(report.witness).toBeNull();
    expect(report.checkedPairs).toBe(1);
    expect(directedJointIsSymmetric(cond, [1, 1])).toBe(true);
  });

  it('rejects an inconsistent 2x2 matrix with a concrete witness', () => {
    const cond = [
      [1, exactFraction(1n, 2n)],
      [exactFraction(1n, 4n), 1],
    ];
    const report = findsArbitrage(cond, [1, 1]);
    expect(report.arbitrage).toBe(true);
    expect(report.violations.length).toBe(1);
    expect(report.witness).not.toBeNull();
    const witness = report.witness!;
    expect(witness.i).toBe(0);
    expect(witness.j).toBe(1);
    expect(witness.left).toEqual({ numerator: 1n, denominator: 2n });
    expect(witness.right).toEqual({ numerator: 1n, denominator: 4n });
    expect(witness.residual).toEqual({ numerator: 1n, denominator: 4n });
    expect((witness.gap)).toBeCloseTo(0.5, 12);
  });

  it('computes the consensus residual exactly', () => {
    const cond = [
      [1, exactFraction(1n, 2n)],
      [exactFraction(1n, 4n), 1],
    ];
    expect(consensusResidual(cond, [1, 1], 0, 1)).toEqual({ numerator: 1n, denominator: 4n });
    expect(consensusResidual(cond, [1, 1], 1, 0)).toEqual({ numerator: -1n, denominator: 4n });
    expect(consensusResidual(cond, [2, 1], 0, 1)).toEqual({ numerator: 0n, denominator: 1n });
  });

  it('checks two venues directly', () => {
    expect(twoVenueArbitrage(0.5, 0.5, 1, 1).arbitrage).toBe(false);
    expect(twoVenueArbitrage(0.5, 0.25, 1, 1).arbitrage).toBe(true);
    expect(twoVenueArbitrage(0.5, 0.25, 2, 1).arbitrage).toBe(false);
  });

  it('exposes the directed joint and its antisymmetric residual', () => {
    const cond = [
      [1, exactFraction(1n, 2n)],
      [exactFraction(1n, 4n), 1],
    ];
    expect(directedJointMatrix(cond, [1, 1])).toEqual([
      [{ numerator: 1n, denominator: 1n }, { numerator: 1n, denominator: 2n }],
      [{ numerator: 1n, denominator: 4n }, { numerator: 1n, denominator: 1n }],
    ]);
    const residual = antisymmetricResidualMatrix(cond, [1, 1]);
    expect(residual[0]![0]).toEqual({ numerator: 0n, denominator: 1n });
    expect(residual[0]![1]).toEqual({ numerator: 1n, denominator: 4n });
    expect(residual[1]![0]).toEqual({ numerator: -1n, denominator: 4n });
    expect(residual[1]![1]).toEqual({ numerator: 0n, denominator: 1n });
  });

  it('rejects malformed matrices', () => {
    expect(() => asRationalMatrix([])).toThrow(RangeError);
    expect(() => asRationalMatrix([[1, 2, 3]])).toThrow(RangeError);
    expect(() => findsArbitrage([[1, 0.5], [0.5, 1]], [1])).toThrow(RangeError);
  });
});

describe('arbitrage -- random consistency construction', () => {
  it('never finds arbitrage on a symmetric joint read through the masses (300 cases)', () => {
    const rand = rng(0xa4b17a01);
    for (let t = 0; t < 300; t++) {
      const n = 2 + Math.floor(rand() * 4);
      const masses = Array.from({ length: n }, () => 1 + Math.floor(rand() * 5));
      const joint: number[][] = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          const value = Math.floor(rand() * 10);
          joint[i]![j] = value;
          joint[j]![i] = value;
        }
      }
      const cond: ExactFraction[][] = joint.map((row, i) =>
        row.map((value, j) => exactFraction(BigInt(value), BigInt(masses[j]!))),
      );
      const report = findsArbitrage(cond, masses);
      expect(report.arbitrage).toBe(false);
      expect(directedJointIsSymmetric(cond, masses)).toBe(true);

      // Perturbing one directed price must create a witness.
      const perturbed = cond.map((row) => [...row]);
      perturbed[0]![1] = exactFraction(perturbed[0]![1]!.numerator + 1n, perturbed[0]![1]!.denominator);
      const broken = findsArbitrage(perturbed, masses);
      expect(broken.arbitrage).toBe(true);
      expect(broken.witness).not.toBeNull();
    }
  });

  it('stays exact beyond 2^53', () => {
    const huge = 10n ** 30n;
    const balanced = findsArbitrage([[1, 0.5], [0.5, 1]], [huge, huge]);
    expect(balanced.arbitrage).toBe(false);
    const skewed = findsArbitrage([[1, 0.5], [0.5, 1]], [huge, huge + 1n]);
    expect(skewed.arbitrage).toBe(true);
    expect(skewed.witness!.residual).toEqual({ numerator: 1n, denominator: 2n });
  });
});

describe('arbitrage -- adversarial dual (the mass-blind checker)', () => {
  it('fires when symmetric prices hide an inconsistent venue mass', () => {
    const cond = [
      [1, 0.5],
      [0.5, 1],
    ];
    const masses = [1, 3];
    expect(directionBlindFindsArbitrage(cond)).toBe(false);
    const report = findsArbitrage(cond, masses);
    expect(report.arbitrage).toBe(true);
    expect(report.witness!.left).toEqual({ numerator: 3n, denominator: 2n });
    expect(report.witness!.right).toEqual({ numerator: 1n, denominator: 2n });
    expect(report.witness!.residual).toEqual({ numerator: 1n, denominator: 1n });

    const dual = arbitrageAdversarialDual(cond, masses);
    expect(dual.lawFindsArbitrage).toBe(true);
    expect(dual.directionBlindFinds).toBe(false);
    expect(dual.fires).toBe(true);
    expect(dual.witness!.i).toBe(0);
    expect(dual.witness!.j).toBe(1);
  });

  it('also fires as a false positive when asymmetric prices are mass-compensated', () => {
    const cond = [
      [1, exactFraction(1n, 2n)],
      [exactFraction(1n, 4n), 1],
    ];
    // masses [2, 1] compensate: (1/2)*1 = (1/4)*2
    const masses = [2, 1];
    expect(directionBlindFindsArbitrage(cond)).toBe(true);
    expect(findsArbitrage(cond, masses).arbitrage).toBe(false);
    const dual = arbitrageAdversarialDual(cond, masses);
    expect(dual.lawFindsArbitrage).toBe(false);
    expect(dual.directionBlindFinds).toBe(true);
    expect(dual.fires).toBe(false);
  });

  it('agrees when the masses are uniform', () => {
    const cond = [
      [1, 0.5],
      [0.5, 1],
    ];
    const dual = arbitrageAdversarialDual(cond, [1, 1]);
    expect(dual.lawFindsArbitrage).toBe(false);
    expect(dual.directionBlindFinds).toBe(false);
    expect(dual.fires).toBe(false);
  });
});
