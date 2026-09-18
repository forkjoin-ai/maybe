import { describe, expect, it } from 'bun:test';
import {
  MAX_SAFE_COUNT,
  coerceCounts,
  mergeBigCounts,
  mergeCounts,
  evidenceTotal,
  exactPosterior,
  posteriorPairs,
  posterior,
  assertLaplaceIdentity,
  createReplica,
  delta,
  applyDelta,
  mergeReplicas,
  merge,
  mergeAll,
  replicaFromDeltas,
  replicaPosterior,
  probe,
  hasObserved,
  type CountVector,
  type Delta,
  type Replica,
} from './void-crdt';
import { laplacePosterior, assertUrnLaplaceIdentity } from './urn';

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

function shuffle<T>(items: T[], rand: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = out[i]!;
    out[i] = out[j]!;
    out[j] = tmp;
  }
  return out;
}

/**
 * A fixed id -> category pool. Every replica draws its subset from this pool,
 * so overlapping ids always carry the same category. That is the well-formed
 * case; conflicting ids are exercised separately.
 */
function deltaPool(k: number, n: number): Delta[] {
  return Array.from({ length: n }, (_, i) => delta(`obs-${i}`, i % k));
}

function randomReplica(rand: () => number, k: number, pool: readonly Delta[]): Replica {
  const chosen: Delta[] = [];
  for (const entry of pool) {
    if (rand() < 0.5) chosen.push(entry);
  }
  return replicaFromDeltas(k, chosen);
}

describe('void crdt -- count layer', () => {
  it('mergeCounts is componentwise addition (shorter vectors zero-pad)', () => {
    expect(mergeCounts([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
    expect(mergeCounts([1], [2, 3])).toEqual([3, 3]);
    expect(mergeCounts([], [2, 3])).toEqual([2, 3]);
  });

  it('mergeCounts is commutative on random vectors (2000 cases)', () => {
    const rand = rng(0xc0ffee);
    for (let t = 0; t < 2000; t++) {
      const a = randomCounts(rand);
      const b = randomCounts(rand);
      expect(mergeCounts(a, b)).toEqual(mergeCounts(b, a));
    }
  });

  it('mergeCounts is associative on random vectors (2000 cases)', () => {
    const rand = rng(0xa550c1a7);
    for (let t = 0; t < 2000; t++) {
      const a = randomCounts(rand);
      const b = randomCounts(rand);
      const c = randomCounts(rand);
      expect(mergeCounts(mergeCounts(a, b), c)).toEqual(mergeCounts(a, mergeCounts(b, c)));
    }
  });

  it('discloses that the raw count merge is NOT idempotent: re-merging doubles', () => {
    const a = [3, 0, 7];
    expect(mergeCounts(a, a)).toEqual([6, 0, 14]);
    expect(mergeCounts(a, a)).not.toEqual(a);
    // Idempotency requires observation identity; see the replica layer.
    const replica = replicaFromDeltas(3, [delta('d0', 0), delta('d2', 2)]);
    expect(merge(replica, replica).counts).toEqual(replica.counts);
  });
});

describe('void crdt -- exact BigInt layer', () => {
  it('merges counts beyond 2^53 exactly', () => {
    const big = 1n << 60n;
    expect(mergeBigCounts([big], [big])).toEqual([big * 2n]);
    expect(mergeBigCounts([big], [big])[0]).toBe(2n ** 61n);
  });

  it('keeps exact numerators/denominators beyond 2^53', () => {
    const big = 1n << 60n;
    const exact = exactPosterior([big, 0n]);
    expect(exact.total).toBe(big);
    expect(exact.denominator).toBe(big + 2n);
    expect(exact.terms[0]!.numerator).toBe(big + 1n);
    expect(exact.terms[1]!.numerator).toBe(1n);
    expect(assertLaplaceIdentity([big, 0n]).ok).toBe(true);
    expect(evidenceTotal([big, 0n])).toBe(big);
  });

  it('coerceCounts rejects non-safe, negative and non-integer counts', () => {
    expect(() => coerceCounts([1.5])).toThrow(RangeError);
    expect(() => coerceCounts([-1])).toThrow(RangeError);
    expect(() => coerceCounts([MAX_SAFE_COUNT + 1])).toThrow(RangeError);
    expect(() => coerceCounts([-1n])).toThrow(RangeError);
    expect(coerceCounts([0, 1, 2])).toEqual([0n, 1n, 2n]);
    expect(coerceCounts([MAX_SAFE_COUNT])).toEqual([BigInt(MAX_SAFE_COUNT)]);
  });

  it('mergeCounts refuses to leave the safe range instead of rounding', () => {
    expect(mergeCounts([MAX_SAFE_COUNT], [0])).toEqual([MAX_SAFE_COUNT]);
    expect(() => mergeCounts([MAX_SAFE_COUNT], [1])).toThrow(RangeError);
  });
});

describe('void crdt -- replica merge laws', () => {
  const pool = deltaPool(4, 24);

  it('applyDelta is idempotent by observation identity', () => {
    let replica = createReplica(3);
    replica = applyDelta(replica, delta('d1', 1));
    replica = applyDelta(replica, delta('d2', 2));
    const once = replica;
    const twice = applyDelta(replica, delta('d1', 1));
    expect(twice.counts).toEqual(once.counts);
    expect(twice.ledger).toEqual(once.ledger);
    expect(hasObserved(twice, 'd1')).toBe(true);
    expect(hasObserved(twice, 'missing')).toBe(false);
    // Immutability: the input replica is untouched.
    expect(replica.counts).toEqual([0, 1, 1]);
  });

  it('merge is commutative on random replicas (1000 cases)', () => {
    const rand = rng(0x1234abcd);
    for (let t = 0; t < 1000; t++) {
      const a = randomReplica(rand, 4, pool);
      const b = randomReplica(rand, 4, pool);
      const left = merge(a, b);
      const right = merge(b, a);
      expect(left.counts).toEqual(right.counts);
      expect(left.ledger).toEqual(right.ledger);
    }
  });

  it('merge is associative on random replicas (1000 cases)', () => {
    const rand = rng(0x5a5a5a5a);
    for (let t = 0; t < 1000; t++) {
      const a = randomReplica(rand, 4, pool);
      const b = randomReplica(rand, 4, pool);
      const c = randomReplica(rand, 4, pool);
      expect(merge(merge(a, b), c).counts).toEqual(merge(a, merge(b, c)).counts);
      expect(merge(merge(a, b), c).ledger).toEqual(merge(a, merge(b, c)).ledger);
    }
  });

  it('merge is idempotent: merge(a, a) === a', () => {
    const rand = rng(0x1de0b07);
    for (let t = 0; t < 500; t++) {
      const a = randomReplica(rand, 4, pool);
      const merged = merge(a, a);
      expect(merged.counts).toEqual(a.counts);
      expect(merged.ledger).toEqual(a.ledger);
    }
  });

  it('duplicates do not change counts, even across two replicas', () => {
    const a = replicaFromDeltas(3, [delta('d1', 0), delta('d2', 1)]);
    const b = replicaFromDeltas(3, [delta('d2', 1), delta('d3', 2)]);
    const merged = merge(a, b);
    // d2 is shared; the union has three observations, not four.
    expect(merged.counts).toEqual([1, 1, 1]);
    expect(merged.ledger.map((entry) => entry.id)).toEqual(['d1', 'd2', 'd3']);
  });

  it('resolves a conflicting id deterministically to the lowest category', () => {
    const left = replicaFromDeltas(3, [delta('same', 2)]);
    const right = replicaFromDeltas(3, [delta('same', 0)]);
    expect(merge(left, right).counts).toEqual([1, 0, 0]);
    expect(merge(right, left).counts).toEqual([1, 0, 0]);
  });

  it('refuses mismatched category counts and invalid categories', () => {
    expect(() => merge(createReplica(2), createReplica(3))).toThrow(RangeError);
    expect(() => createReplica(0)).toThrow(RangeError);
    expect(() => replicaFromDeltas(3, [delta('bad', 3)])).toThrow(RangeError);
    expect(() => delta('', 0)).toThrow(RangeError);
  });
});

describe('void crdt -- convergence', () => {
  it('same observations in any order give the same exact posterior (500 orders)', () => {
    const base: Delta[] = [
      delta('a', 0), delta('b', 2), delta('c', 1),
      delta('d', 0), delta('e', 2), delta('f', 1),
    ];
    const target = replicaFromDeltas(3, base);
    const expected = replicaPosterior(target);
    const rand = rng(0xc0a7e5ce);

    for (let t = 0; t < 500; t++) {
      const replica = replicaFromDeltas(3, shuffle(base, rand));
      const actual = replicaPosterior(replica);
      expect(replica.counts).toEqual(target.counts);
      expect(actual.total).toBe(expected.total);
      expect(actual.denominator).toBe(expected.denominator);
      for (let i = 0; i < actual.k; i++) {
        expect(actual.terms[i]!.numerator).toBe(expected.terms[i]!.numerator);
        expect(actual.terms[i]!.denominator).toBe(expected.terms[i]!.denominator);
      }
    }
  });

  it('merging in any binary tree shape converges to the same counts', () => {
    const rand = rng(0x7ee5);
    const pool = deltaPool(5, 30);

    // Every replica sees a different overlapping subset of the same evidence.
    for (let t = 0; t < 300; t++) {
      const replicas: Replica[] = [];
      const chosenAll: Delta[] = [];
      for (let r = 0; r < 4; r++) {
        const chosen: Delta[] = [];
        for (const entry of pool) {
          if (rand() < 0.5) chosen.push(entry);
        }
        chosenAll.push(...chosen);
        replicas.push(replicaFromDeltas(5, shuffle(chosen, rand)));
      }

      // The reference is the plain union of every observation that arrived,
      // independent of merge order or tree shape.
      const reference = replicaFromDeltas(5, chosenAll);
      const leftDeep = merge(merge(replicas[0]!, replicas[1]!), merge(replicas[2]!, replicas[3]!));
      const rightDeep = merge(replicas[0]!, merge(replicas[1]!, merge(replicas[2]!, replicas[3]!)));
      const folded = mergeAll(shuffle(replicas, rand));

      for (const merged of [leftDeep, rightDeep, folded]) {
        expect(merged.counts).toEqual(reference.counts);
        expect(merged.ledger).toEqual(reference.ledger);
        expect(replicaPosterior(merged).denominator).toBe(replicaPosterior(reference).denominator);
      }
    }
  });

  it('the Laplace identity holds after arbitrary merges (500 chains)', () => {
    const rand = rng(0x1a9a1ce);
    const pool = deltaPool(6, 40);

    for (let t = 0; t < 500; t++) {
      const replicas: Replica[] = [];
      const count = 1 + Math.floor(rand() * 5);
      for (let r = 0; r < count; r++) {
        const chosen: Delta[] = [];
        for (const entry of pool) {
          if (rand() < 0.5) chosen.push(entry);
        }
        replicas.push(replicaFromDeltas(6, shuffle(chosen, rand)));
      }

      const merged = mergeAll(replicas);
      const report = assertLaplaceIdentity(merged.counts);
      expect(report.ok).toBe(true);
      expect(report.numeratorSum).toBe(report.denominator);

      // The urn module reads the same counts and agrees exactly.
      const urnReport = assertUrnLaplaceIdentity(merged.counts);
      expect(urnReport.ok).toBe(true);
      const mine = posterior(merged.counts);
      const urn = laplacePosterior(merged.counts);
      for (let i = 0; i < mine.length; i++) {
        expect(mine[i]).toBe(urn[i]);
      }
    }
  });
});

describe('void crdt -- exact posterior', () => {
  it('N = 0 gives the uniform 1/K posterior exactly', () => {
    for (const k of [1, 2, 3, 5, 11]) {
      const counts = Array.from({ length: k }, () => 0);
      const exact = exactPosterior(counts);
      expect(exact.total).toBe(0n);
      expect(exact.denominator).toBe(BigInt(k));
      expect(exact.uniform).toBe(true);
      for (const term of exact.terms) {
        expect(term.numerator).toBe(1n);
        expect(term.denominator).toBe(BigInt(k));
        expect(term.ratio).toBeCloseTo(1 / k, 12);
      }
    }
  });

  it('returns the exact integer numerator/denominator pairs', () => {
    expect(posteriorPairs([2, 0, 5])).toEqual([
      { numerator: 3n, denominator: 10n },
      { numerator: 1n, denominator: 10n },
      { numerator: 6n, denominator: 10n },
    ]);
    expect(exactPosterior([2, 0, 5]).total).toBe(7n);
    expect(exactPosterior([2, 0, 5]).denominator).toBe(10n);
    expect(exactPosterior([2, 0, 5]).uniform).toBe(false);
  });

  it('posterior === urn laplacePosterior bit-for-bit (1000 cases)', () => {
    const rand = rng(0x5eed1234);
    for (let t = 0; t < 1000; t++) {
      const counts = randomCounts(rand, 12, 40);
      const mine = posterior(counts);
      const urn = laplacePosterior(counts);
      const total = Number(evidenceTotal(counts));
      let mass = 0;
      for (let i = 0; i < counts.length; i++) {
        expect(mine[i]).toBe(urn[i]);
        expect(mine[i]).toBeCloseTo((counts[i]! + 1) / (total + counts.length), 12);
        mass += mine[i]!;
      }
      expect(mass).toBeCloseTo(1, 12);
      expect(assertLaplaceIdentity(counts).ok).toBe(true);
      expect(assertUrnLaplaceIdentity(counts).ok).toBe(true);
    }
  });

  it('exposes a probe/query API over a replica', () => {
    const replica = replicaFromDeltas(3, [delta('x', 0), delta('y', 0), delta('z', 2)]);
    const exact = replicaPosterior(replica);
    expect(exact.total).toBe(3n);
    expect(exact.denominator).toBe(6n);

    const first = probe(replica, 0);
    expect(first.numerator).toBe(3n);
    expect(first.denominator).toBe(6n);
    expect(first.ratio).toBeCloseTo(0.5, 12);
    expect(probe(replica, 2).numerator).toBe(2n);
    expect(() => probe(replica, 3)).toThrow(RangeError);
  });
});
