import { describe, expect, it } from 'bun:test';
import {
  PLEROMA,
  KENOMA,
  godWeight,
  floorWeight,
  pairwise,
  triangular,
  fib,
  faceVolume,
  voidVolume,
  totalVolume,
  structuralReport,
  countTrue,
  articulationDeficit,
  consensusEntry,
  consensusMatrix,
  consensusIsSymmetric,
  fieldWeights,
  skyRmsPeak,
  peakIsMaxWeight,
  channelBlocks,
  pleromaIdentities,
  collapse,
  type CollapseInput,
  type Signal,
} from './tensor-bayes-55';

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

describe('tensor bayes 55 -- floor', () => {
  it('never reaches zero on a survivor and reaches exactly one at the cap', () => {
    for (let R = 0; R <= 40; R++) {
      for (let v = 0; v <= R + 5; v++) {
        expect(floorWeight({ budget: R, rejections: v, hardRefuted: false })).toBeGreaterThanOrEqual(1);
      }
      expect(floorWeight({ budget: R, rejections: R, hardRefuted: false })).toBe(1);
      expect(floorWeight({ budget: R, rejections: R, hardRefuted: true })).toBe(0);
    }
  });

  it('agrees with the God Formula on the clamped domain', () => {
    for (let R = 0; R <= 20; R++) {
      for (let v = 0; v <= R; v++) {
        expect(godWeight(R, v)).toBe(R - v + 1);
      }
      expect(godWeight(R, R + 7)).toBe(1);
    }
  });
});

describe('tensor bayes 55 -- face/void conservation', () => {
  it('face * void === total for random weights and signals (2000 cases)', () => {
    const rand = rng(0x12345678);
    for (let t = 0; t < 2000; t++) {
      const n = 1 + Math.floor(rand() * 60);
      const weights = Array.from({ length: n }, () => 1 + Math.floor(rand() * 9));
      const signal: Signal = Array.from({ length: n }, () => rand() < 0.5);
      const report = structuralReport(weights, signal);
      expect(report.conserved).toBe(true);
      expect(report.floorPreserved).toBe(true);
      expect(report.faceVolume * report.voidVolume).toBe(report.totalVolume);
    }
  });

  it('is rotation invariant: every signal re-splits the same total', () => {
    const rand = rng(7);
    for (let t = 0; t < 500; t++) {
      const n = 1 + Math.floor(rand() * 30);
      const weights = Array.from({ length: n }, () => 1 + Math.floor(rand() * 5));
      const a: Signal = Array.from({ length: n }, () => rand() < 0.5);
      const b: Signal = Array.from({ length: n }, () => rand() < 0.5);
      expect(faceVolume(weights, a) * voidVolume(weights, a)).toBe(totalVolume(weights));
      expect(faceVolume(weights, b) * voidVolume(weights, b)).toBe(totalVolume(weights));
    }
  });

  it('tracks the articulation deficit', () => {
    const full: Signal = [true, true, true];
    const empty: Signal = [false, false, false];
    expect(articulationDeficit(full)).toBe(0);
    expect(articulationDeficit(empty)).toBe(3);
    expect(countTrue([true, false, true])).toBe(2);
    expect(articulationDeficit([true, false, true])).toBe(1);
  });
});

describe('tensor bayes 55 -- consensus', () => {
  it('is symmetric and equals the rank-one joint', () => {
    const weights = [2, 3, 5, 7, 11];
    expect(consensusIsSymmetric(weights)).toBe(true);
    expect(consensusEntry(weights, 0, 1)).toBe(6);
    expect(consensusEntry(weights, 1, 2)).toBe(15);
    const matrix = consensusMatrix(weights);
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
        expect(matrix[i][j]).toBe(matrix[j][i]);
      }
    }
  });

  it('shows the Skyrms peak is the maximum Buleyean weight', () => {
    const rand = rng(99);
    for (let t = 0; t < 500; t++) {
      const budget = Math.floor(rand() * 40);
      const n = 1 + Math.floor(rand() * 20);
      const rejections = Array.from({ length: n }, () => Math.floor(rand() * (budget + 1)));
      const field = { budget, rejections };
      const peak = skyRmsPeak(field);
      const weights = fieldWeights(field);
      const min = Math.min(...rejections);
      expect(rejections[peak]).toBe(min);
      expect(peakIsMaxWeight(field)).toBe(true);
      expect(weights[peak]).toBe(Math.max(...weights));
    }
  });
});

describe('tensor bayes 55 -- pleroma counting', () => {
  it('is fib(10), triangular(10), and C(11,2) at once', () => {
    expect(PLEROMA).toBe(55);
    expect(KENOMA).toBe(10);
    expect(fib(10)).toBe(55);
    expect(triangular(10)).toBe(55);
    expect(pairwise(11)).toBe(55);
    expect(pairwise(55)).toBe(1485);
    expect(pairwise(5)).toBe(10);
  });

  it('splits 1485 channels into 495 adjacent plus 990 disjoint', () => {
    const blocks = channelBlocks(11);
    expect(blocks.adjacent).toBe(495);
    expect(blocks.disjoint).toBe(990);
    expect(blocks.channels).toBe(1485);
    expect(blocks.adjacent + blocks.disjoint).toBe(blocks.channels);
  });

  it('exposes the identities as one record', () => {
    const ids = pleromaIdentities();
    expect(ids.tenBosonChannels).toBe(10);
    expect(ids.elevenWalkerChannels).toBe(PLEROMA);
    expect(ids.pairwiseChannels).toBe(1485);
    expect(ids.triangularKenoma).toBe(PLEROMA);
    expect(ids.fibTen).toBe(PLEROMA);
  });
});

describe('tensor bayes 55 -- human collapse', () => {
  it('reads mass, width, total, and the floor/cap interval on a known vector', () => {
    const result = collapse({ weights: [1, 2, 3, 4, 5], target: [0, 2], budget: 4 });
    expect(result.mass).toBe(4n);
    expect(result.width).toBe(2);
    expect(result.total).toBe(15n);
    expect(result.pointNumerator).toBe(4n);
    expect(result.pointDenominator).toBe(15n);
    expect(result.pointPercent).toBeCloseTo((4 / 15) * 100, 9);
    expect(result.lowPercent).toBeCloseTo((2 / 15) * 100, 9);
    expect(result.highPercent).toBeCloseTo((10 / 15) * 100, 9);
    expect(result.inRange).toBe(true);
    expect(result.sentence).toContain('15');
    expect(result.sentence).toContain('%');
    expect(result.sentence.length).toBeGreaterThan(0);
  });

  it('hits the interval endpoints at the floor and the cap', () => {
    const floor = collapse({ weights: [1, 1, 1, 1], target: [1, 3], budget: 9 });
    expect(floor.mass).toBe(2n);
    expect(floor.pointPercent).toBeCloseTo(floor.lowPercent, 12);
    expect(floor.inRange).toBe(true);

    const cap = collapse({ weights: [10, 10, 1], target: [0, 1], budget: 9 });
    expect(cap.mass).toBe(20n);
    expect(cap.pointPercent).toBeCloseTo(cap.highPercent, 12);
    expect(cap.inRange).toBe(true);
  });

  it('always keeps the collapse point in [low, high] for random inputs (2000 cases)', () => {
    const rand = rng(0xc011a95e);
    for (let t = 0; t < 2000; t++) {
      const budget = Math.floor(rand() * 12);
      const n = 1 + Math.floor(rand() * 20);
      const weights = Array.from({ length: n }, () => 1 + Math.floor(rand() * (budget + 1)));

      // Distinct non-empty target subset.
      const targetCount = 1 + Math.floor(rand() * n);
      const pool = Array.from({ length: n }, (_, i) => i);
      const target: number[] = [];
      for (let picked = 0; picked < targetCount; picked++) {
        const at = Math.floor(rand() * pool.length);
        target.push(pool[at]);
        pool.splice(at, 1);
      }

      const input: CollapseInput = { weights, target, budget };
      const result = collapse(input);

      let expectedMass = 0n;
      for (const index of target) expectedMass += BigInt(weights[index]);
      let expectedTotal = 0n;
      for (const weight of weights) expectedTotal += BigInt(weight);

      expect(result.width).toBe(target.length);
      expect(result.mass).toBe(expectedMass);
      expect(result.total).toBe(expectedTotal);
      expect(result.pointNumerator).toBe(expectedMass);
      expect(result.pointDenominator).toBe(expectedTotal);

      // Exact integer sandwich: width <= mass <= (budget + 1) * width.
      expect(result.mass >= BigInt(result.width)).toBe(true);
      expect(result.mass <= BigInt(budget + 1) * BigInt(result.width)).toBe(true);
      expect(result.inRange).toBe(true);

      // Display floats agree with the same sandwich.
      expect(result.lowPercent <= result.pointPercent).toBe(true);
      expect(result.pointPercent <= result.highPercent).toBe(true);
      expect(result.lowPercent).toBeCloseTo((result.width / Number(result.total)) * 100, 9);
      expect(result.highPercent).toBeCloseTo(
        (((budget + 1) * result.width) / Number(result.total)) * 100,
        9,
      );
    }
  });
});
