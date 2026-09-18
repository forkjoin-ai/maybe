import { describe, expect, it } from 'bun:test';
import {
  executionField,
  rejectionTotal,
  godWeight,
  laplaceCounts,
  venueWeights,
  venuePosterior,
  skyRmsPeak,
  peakIsMaxWeight,
  assertFloor,
  assertLaplaceIdentity,
  collapseRange,
  rankVenues,
  seededRng,
  selectVenue,
  routingAdversarialDual,
  DEFAULT_REJECTION_WEIGHTS,
  type ExecutionVenue,
} from './execution-router';
import { compareFractions } from './rational';

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

function venue(id: string, failedFills: number, adverseSelection = 0, latencyMisses = 0): ExecutionVenue {
  return { id, rejections: { failedFills, adverseSelection, latencyMisses } };
}

describe('execution-router -- rejection ledger', () => {
  it('folds rejection kinds with default weights', () => {
    expect(rejectionTotal({ failedFills: 2, adverseSelection: 3, latencyMisses: 4 })).toBe(9);
    expect(DEFAULT_REJECTION_WEIGHTS.failedFills).toBe(1);
    expect(rejectionTotal(
      { failedFills: 2, adverseSelection: 3, latencyMisses: 4 },
      { failedFills: 2, adverseSelection: 0, latencyMisses: 1 },
    )).toBe(8);
  });

  it('defaults the budget to the largest rejection count', () => {
    const field = executionField([venue('a', 0), venue('b', 2), venue('c', 1)]);
    expect(field.budget).toBe(2);
    expect(field.rejections).toEqual([0, 2, 1]);
    expect(field.venues).toEqual(['a', 'b', 'c']);
  });

  it('clamps a venue rejected beyond the budget instead of killing it', () => {
    const field = executionField([venue('a', 0), venue('b', 5)], 2);
    expect(godWeight(2, 5)).toBe(1n);
    expect(godWeight(2, 0)).toBe(3n);
    expect(godWeight(2, 2)).toBe(1n);
    expect(laplaceCounts(field)).toEqual([2, 0]);
    expect(venueWeights(field)).toEqual([3n, 1n]);
  });
});

describe('execution-router -- exact posterior and floor', () => {
  it('computes the exact add-one posterior', () => {
    const field = executionField([venue('a', 0), venue('b', 2), venue('c', 2)]);
    const posterior = venuePosterior(field);
    expect(posterior.counts).toEqual([2, 0, 0]);
    expect(posterior.weights).toEqual([3n, 1n, 1n]);
    expect(posterior.total).toBe(2n);
    expect(posterior.totalWeight).toBe(5n);
    expect(posterior.terms[0]!.exact).toEqual({ numerator: 3n, denominator: 5n });
    expect(posterior.terms[1]!.exact).toEqual({ numerator: 1n, denominator: 5n });
    expect(posterior.uniform).toBe(false);
  });

  it('holds the urn/Laplace identity on random fields (2000 cases)', () => {
    const rand = rng(0x0eec0001);
    for (let t = 0; t < 2000; t++) {
      const k = 1 + Math.floor(rand() * 8);
      const rejections = Array.from({ length: k }, () => Math.floor(rand() * 30));
      const budget = Math.floor(rand() * 35);
      const venues = rejections.map((_, i) => venue('v' + String(i), rejections[i]!));
      const field = executionField(venues, budget);
      const report = assertLaplaceIdentity(field);
      expect(report.ok).toBe(true);
      expect(report.numeratorSum).toBe(report.denominator);
      const weights = venueWeights(field);
      for (const weight of weights) expect(weight).toBeGreaterThanOrEqual(1n);
    }
  });

  it('keeps the never-collapse floor on every venue', () => {
    const field = executionField([venue('a', 5), venue('b', 1), venue('c', 0)], 2);
    const floor = assertFloor(field);
    expect(floor.ok).toBe(true);
    expect(floor.minWeight).toBe(1n);
    expect(floor.minWeightVenue).toBe(0);
  });
});

describe('execution-router -- skyRms peak and ranking', () => {
  it('finds the least rejected venue and its max weight', () => {
    const field = executionField([venue('a', 2), venue('b', 0), venue('c', 1)], 2);
    expect(skyRmsPeak(field)).toBe(1);
    expect(peakIsMaxWeight(field)).toBe(true);
  });

  it('ranks venues by descending posterior, ties by index', () => {
    const field = executionField([venue('a', 2), venue('b', 0), venue('c', 1)], 2);
    // counts [0,2,1], weights [1,3,2]
    const ranking = rankVenues(field);
    expect(ranking.map((entry) => entry.venue)).toEqual([1, 2, 0]);
    expect(ranking.map((entry) => entry.weight)).toEqual([3n, 2n, 1n]);
    expect(ranking[0]!.id).toBe('b');
    expect(ranking[0]!.denominator).toBe(6n);
  });

  it('breaks equal-rejection ties by lowest index', () => {
    const field = executionField([venue('a', 1), venue('b', 1), venue('c', 0)]);
    const ranking = rankVenues(field);
    expect(ranking.map((entry) => entry.venue)).toEqual([2, 0, 1]);
  });
});

describe('execution-router -- proven collapse range', () => {
  it('returns the sound band [width/W, (R+1) width/W]', () => {
    const field = executionField([venue('a', 0), venue('b', 2), venue('c', 2)]);
    // counts [2,0,0], weights [3,1,1], W=5, R=2
    const band = collapseRange(field, [0, 1]);
    expect(band.width).toBe(2);
    expect(band.total).toBe(5n);
    expect(band.mass).toBe(4n);
    expect(band.point).toEqual({ numerator: 4n, denominator: 5n });
    expect(band.low).toEqual({ numerator: 2n, denominator: 5n });
    expect(band.high).toEqual({ numerator: 6n, denominator: 5n });
    expect(band.inRange).toBe(true);
  });

  it('always contains the point on random fields (1000 cases)', () => {
    const rand = rng(0x0eec0002);
    for (let t = 0; t < 1000; t++) {
      const k = 2 + Math.floor(rand() * 7);
      const rejections = Array.from({ length: k }, () => Math.floor(rand() * 20));
      const budget = Math.floor(rand() * 25);
      const venues = rejections.map((_, i) => venue('v' + String(i), rejections[i]!));
      const field = executionField(venues, budget);
      const target: number[] = [];
      for (let i = 0; i < k; i++) if (rand() > 0.5) target.push(i);
      if (target.length === 0) target.push(0);
      const band = collapseRange(field, target);
      expect(band.inRange).toBe(true);
      expect(compareFractions(band.point, band.low)).toBeGreaterThanOrEqual(0);
      expect(compareFractions(band.point, band.high)).toBeLessThanOrEqual(0);
    }
  });
});

describe('execution-router -- deterministic selection', () => {
  it('reproduces the same sequence for the same seed', () => {
    const field = executionField([venue('a', 0), venue('b', 2), venue('c', 3)]);
    const left = seededRng(0x5eed);
    const right = seededRng(0x5eed);
    for (let i = 0; i < 50; i++) expect(left()).toBe(right());
  });

  it('samples the least rejected venue most often', () => {
    const field = executionField([venue('a', 0), venue('b', 4), venue('c', 4)]);
    const rand = seededRng(0x5eed5eed);
    const counts = [0, 0, 0];
    for (let i = 0; i < 3000; i++) counts[selectVenue(field, rand)]! += 1;
    expect(counts[0]).toBeGreaterThan(counts[1]!);
    expect(counts[0]).toBeGreaterThan(counts[2]!);
  });
});

describe('execution-router -- adversarial dual (MLE abandons a zeroed venue)', () => {
  it('fires when a correctly-rejected venue is priced at zero by the MLE', () => {
    const field = executionField([venue('a', 0), venue('b', 2), venue('c', 3)], 2);
    // counts [2,0,0]: venues 1 and 2 sit at the floor
    const dual = routingAdversarialDual(field);
    expect(dual.fires).toBe(true);
    expect(dual.mleDefined).toBe(true);
    expect(dual.zeroedVenues).toEqual([1, 2]);
    expect(dual.mostRejected).toBe(2);
    expect(dual.floor).toEqual({ numerator: 1n, denominator: 5n });
    expect(dual.total).toBe(2n);
  });

  it('does not fire when the budget exceeds every rejection count', () => {
    const field = executionField([venue('a', 0), venue('b', 2), venue('c', 3)], 10);
    const dual = routingAdversarialDual(field);
    expect(dual.fires).toBe(false);
    expect(dual.zeroedVenues).toEqual([]);
  });

  it('fires on random fields with a survivor and a correctly-rejected venue (500 cases)', () => {
    const rand = rng(0x0eec0003);
    for (let t = 0; t < 500; t++) {
      const k = 2 + Math.floor(rand() * 6);
      const rejections = Array.from({ length: k }, () => 1 + Math.floor(rand() * 20));
      rejections[0] = 0; // guarantees a strictly-least-rejected venue
      const venues = rejections.map((_, i) => venue('v' + String(i), rejections[i]!));
      const field = executionField(venues);
      const dual = routingAdversarialDual(field);
      expect(dual.fires).toBe(true);
      expect(dual.floor.numerator).toBe(1n);
    }
  });

  it('does not fire when every venue is equally rejected (N=0, MLE undefined)', () => {
    const field = executionField([venue('a', 3), venue('b', 3)]);
    const dual = routingAdversarialDual(field);
    expect(dual.fires).toBe(false);
    expect(dual.mleDefined).toBe(false);
  });
});
