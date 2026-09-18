/**
 * void-crdt.ts -- the conflict-free replicated posterior.
 *
 * A Bayesian posterior is a fold over evidence, and evidence is commutative:
 * observing category A then B is the same world as observing B then A. This
 * module makes that fold a CRDT -- a join-semilattice -- so independent
 * replicas can exchange evidence in any order, with any amount of duplication,
 * and still agree on the exact posterior.
 *
 * The arithmetic is the same Laplace/Buleyean add-one posterior the urn module
 * reads through the God Formula:
 *
 *   posterior(counts)[i] = (counts[i] + 1) / (N + K),   N = sum_i counts[i].
 *
 * The +1 is the sliver; the denominator N + K is its exact integer home.
 *
 * ---------------------------------------------------------------------------
 * Two layers, and the honest law for each
 * ---------------------------------------------------------------------------
 *
 * 1. COUNT LAYER -- mergeCounts / mergeBigCounts. Componentwise addition of
 *    count vectors. Addition is commutative and associative, so this is a
 *    commutative monoid for DISTINCT evidence. It is NOT idempotent: a + a = 2a.
 *    Re-delivering the same observation through this layer inflates the count.
 *    Sums are exact in BigInt; mergeCounts refuses to return a number that would
 *    leave the 2^53 - 1 safe range instead of silently rounding.
 *
 * 2. REPLICA LAYER -- Replica / mergeReplicas (exported as `merge`). A replica
 *    carries the count vector AND a canonical identity ledger of the distinct
 *    observations behind it. The merge is set union of those identities, so it
 *    is commutative, associative, AND idempotent: merge(a, a) = a. A duplicate
 *    delta id does not change any count. This is the layer that is genuinely a
 *    conflict-free replicated posterior.
 *
 * The ledger is what makes duplication harmless. The raw count vector alone
 * cannot distinguish "one observation seen twice" from "two observations", so
 * idempotency is only available once observations carry stable identities.
 *
 * An id collision with a conflicting category is a protocol violation; it is
 * resolved deterministically to the lowest category index so the merge stays a
 * join regardless of arrival order.
 *
 * ---------------------------------------------------------------------------
 * No decay / no compaction
 * ---------------------------------------------------------------------------
 *
 * Decay is deliberately omitted. A scalar decay does not commute with addition:
 * stack(merge(a, b)) != merge(stack(a), stack(b)), so aging counts would destroy
 * the join identity that makes the posterior conflict-free. Any aging policy
 * must run identically on every replica before divergence, or be modelled as an
 * explicit new observation with its own id. There is nothing to compact either:
 * the ledger already holds exactly the distinct observations.
 *
 * Nothing here proves anything. It is the executable arithmetic, checked
 * exactly against the urn/Laplace identity with integer BigInt.
 */

/** A count vector of non-negative safe integers. */
export type CountVector = readonly number[];

/** A count vector of non-negative BigInts, exact beyond 2^53. */
export type BigCountVector = readonly bigint[];

/** Either representation; each entry must be a non-negative integer. */
export type Counts = CountVector | BigCountVector;

/** Largest count exactly representable as a double: 2^53 - 1. */
export const MAX_SAFE_COUNT = Number.MAX_SAFE_INTEGER;

/**
 * Normalize any count vector to exact BigInts.
 *
 * Number entries must be non-negative safe integers; a value beyond 2^53 - 1
 * would already have lost precision, so it is rejected rather than rounded. Use
 * a BigInt vector when a single count can exceed that bound.
 */
export function coerceCounts(counts: Counts): bigint[] {
  const out: bigint[] = [];
  for (let i = 0; i < counts.length; i++) {
    const count = counts[i]!;
    if (typeof count === 'bigint') {
      if (count < 0n) {
        throw new RangeError(`counts[${i}] = ${count} must be non-negative`);
      }
      out.push(count);
    } else {
      if (!Number.isSafeInteger(count) || count < 0) {
        throw new RangeError(
          `counts[${i}] = ${count} is not a non-negative safe integer; use a BigInt count beyond 2^53 - 1`,
        );
      }
      out.push(BigInt(count));
    }
  }
  return out;
}

/**
 * Exact componentwise merge with no upper bound.
 *
 * Shorter vectors are zero-padded to the wider category count, matching the
 * void-boundary merge convention. Commutative and associative; not idempotent.
 */
export function mergeBigCounts(a: BigCountVector, b: BigCountVector): bigint[] {
  const k = Math.max(a.length, b.length);
  const out: bigint[] = [];
  for (let i = 0; i < k; i++) {
    out.push((a[i] ?? 0n) + (b[i] ?? 0n));
  }
  return out;
}

/**
 * Componentwise merge of number count vectors.
 *
 * Throws a RangeError when any merged count would exceed
 * Number.MAX_SAFE_INTEGER. That is the disclosure that this convenience layer
 * is bounded: keep accumulating with mergeBigCounts when the sum can grow past
 * 2^53 - 1.
 */
export function mergeCounts(a: CountVector, b: CountVector): number[] {
  const merged = mergeBigCounts(coerceCounts(a), coerceCounts(b));
  const out: number[] = [];
  for (let i = 0; i < merged.length; i++) {
    const count = merged[i]!;
    if (count > BigInt(MAX_SAFE_COUNT)) {
      throw new RangeError(
        `merged counts[${i}] = ${count} exceeds Number.MAX_SAFE_INTEGER; keep the exact merge with mergeBigCounts`,
      );
    }
    out.push(Number(count));
  }
  return out;
}

/** N = sum_i counts[i], exactly. */
export function evidenceTotal(counts: Counts): bigint {
  let total = 0n;
  for (const count of coerceCounts(counts)) total += count;
  return total;
}

/** One category of the exact posterior: numerator counts[i] + 1 and denominator N + K. */
export interface PosteriorTerm {
  readonly category: number;
  readonly numerator: bigint;
  readonly denominator: bigint;
  /** Display-only float; the truth claim lives in numerator/denominator. */
  readonly ratio: number;
}

/** The exact add-one posterior over K categories. */
export interface ExactPosterior {
  readonly k: number;
  /** N = sum_i counts[i]. */
  readonly total: bigint;
  /** N + K, the shared denominator. */
  readonly denominator: bigint;
  readonly terms: readonly PosteriorTerm[];
  /** True when N = 0, i.e. the uniform 1/K posterior. */
  readonly uniform: boolean;
}

/**
 * The exact posterior as integer numerator/denominator pairs.
 *
 *   numerator_i   = counts[i] + 1
 *   denominator   = N + K
 *
 * Every term shares the denominator, so the posterior is one integer vector
 * over one integer scalar. No floating point is involved in the claim.
 */
export function exactPosterior(counts: Counts): ExactPosterior {
  const big = coerceCounts(counts);
  const k = big.length;
  if (k === 0) {
    throw new RangeError('exactPosterior requires at least one category');
  }

  let total = 0n;
  for (const count of big) total += count;
  const denominator = total + BigInt(k);

  const terms: PosteriorTerm[] = big.map((count, category) => {
    const numerator = count + 1n;
    return {
      category,
      numerator,
      denominator,
      ratio: Number(numerator) / Number(denominator),
    };
  });

  return { k, total, denominator, terms, uniform: total === 0n };
}

/** The exact posterior reduced to its integer pairs. */
export function posteriorPairs(
  counts: Counts,
): Array<{ readonly numerator: bigint; readonly denominator: bigint }> {
  return exactPosterior(counts).terms.map(({ numerator, denominator }) => ({
    numerator,
    denominator,
  }));
}

/**
 * The posterior as display floats, computed from the exact BigInt fraction.
 * (counts[i] + 1) / (N + K), matching urn.ts laplacePosterior bit-for-bit for
 * counts inside the safe range.
 */
export function posterior(counts: Counts): number[] {
  return exactPosterior(counts).terms.map((term) => term.ratio);
}

/** The exact integer reading of the urn/Laplace identity. */
export interface LaplaceIdentity {
  readonly ok: boolean;
  readonly detail: string;
  readonly k: number;
  readonly total: bigint;
  readonly denominator: bigint;
  readonly numeratorSum: bigint;
}

/**
 * Check the integer backbone of the posterior with no floating-point truth
 * claim:
 *
 *   numerator_i === counts[i] + 1 for every i, and
 *   sum_i (counts[i] + 1) === N + K.
 *
 * Passing means the posterior's numerators and denominator are exactly the
 * Laplace numerator and denominator, at any magnitude BigInt can hold.
 */
export function assertLaplaceIdentity(counts: Counts): LaplaceIdentity {
  const big = coerceCounts(counts);
  const exact = exactPosterior(big);
  let numeratorSum = 0n;

  for (let i = 0; i < big.length; i++) {
    const expected = big[i]! + 1n;
    const actual = exact.terms[i]!.numerator;
    if (actual !== expected) {
      return {
        ok: false,
        detail: `numerator_${i} = ${actual} !== counts[${i}] + 1 = ${expected}`,
        k: big.length,
        total: exact.total,
        denominator: exact.denominator,
        numeratorSum,
      };
    }
    numeratorSum += expected;
  }

  if (numeratorSum !== exact.denominator) {
    return {
      ok: false,
      detail: `sum_i (counts[i] + 1) = ${numeratorSum} !== N + K = ${exact.denominator}`,
      k: big.length,
      total: exact.total,
      denominator: exact.denominator,
      numeratorSum,
    };
  }

  return {
    ok: true,
    detail: `sum_i (counts[i] + 1) = ${numeratorSum} = N + K = ${exact.total} + ${big.length}`,
    k: big.length,
    total: exact.total,
    denominator: exact.denominator,
    numeratorSum,
  };
}

// ===========================================================================
// Replica layer -- identity-carrying state, genuinely idempotent merge
// ===========================================================================

/** Stable identity of one observation. Two deliveries of the same id are one observation. */
export type DeltaId = string;

/** A single observation: which category was seen, under which identity. */
export interface Delta {
  readonly id: DeltaId;
  readonly category: number;
}

/**
 * A replica: the derived count vector plus the canonical identity ledger.
 *
 * Invariant (maintained by every constructor):
 *   ledger      is sorted by id, with distinct ids
 *   counts[i]   = number of ledger entries whose category is i
 *   counts      are non-negative safe integers (they count distinct deltas)
 */
export interface Replica {
  readonly k: number;
  readonly counts: readonly number[];
  readonly ledger: readonly Delta[];
}

function assertK(k: number): void {
  if (!Number.isInteger(k) || k < 1) {
    throw new RangeError(`replica categories K = ${k} must be a positive integer`);
  }
}

function assertCategory(k: number, category: number, where: string): void {
  if (!Number.isInteger(category) || category < 0 || category >= k) {
    throw new RangeError(`${where} category = ${category} is outside [0, ${k})`);
  }
}

/** Deduplicate by id, resolve conflicts to the lowest category, sort by id. */
function canonicalLedger(k: number, deltas: readonly Delta[]): Delta[] {
  const byId = new Map<DeltaId, number>();
  for (const entry of deltas) {
    if (typeof entry.id !== 'string' || entry.id.length === 0) {
      throw new RangeError('delta id must be a non-empty string');
    }
    assertCategory(k, entry.category, `delta "${entry.id}"`);
    const prior = byId.get(entry.id);
    if (prior === undefined || entry.category < prior) {
      byId.set(entry.id, entry.category);
    }
  }

  const ledger: Delta[] = [];
  for (const [id, category] of byId) ledger.push({ id, category });
  ledger.sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  return ledger;
}

/** counts[i] = number of distinct ledger observations in category i. */
function countsFromLedger(k: number, ledger: readonly Delta[]): number[] {
  const counts = Array.from({ length: k }, () => 0);
  for (const entry of ledger) {
    counts[entry.category] = counts[entry.category]! + 1;
  }
  return counts;
}

function makeReplica(k: number, deltas: readonly Delta[]): Replica {
  const ledger = canonicalLedger(k, deltas);
  return { k, counts: countsFromLedger(k, ledger), ledger };
}

/** A single observation. Category range is checked when it enters a replica. */
export function delta(id: DeltaId, category: number): Delta {
  if (typeof id !== 'string' || id.length === 0) {
    throw new RangeError('delta id must be a non-empty string');
  }
  return { id, category };
}

/** The empty replica over K categories (N = 0, uniform posterior). */
export function createReplica(k: number): Replica {
  assertK(k);
  return makeReplica(k, []);
}

/** A replica from a multiset of observations; duplicates collapse by id. */
export function replicaFromDeltas(k: number, deltas: readonly Delta[]): Replica {
  assertK(k);
  return makeReplica(k, deltas);
}

/**
 * Apply one observation. Applying the same id again is a no-op, so this is
 * idempotent: applyDelta(applyDelta(r, d), d) has the same counts and ledger as
 * applyDelta(r, d).
 */
export function applyDelta(replica: Replica, entry: Delta): Replica {
  return makeReplica(replica.k, [...replica.ledger, entry]);
}

/**
 * Merge two replicas: the join of their identity ledgers, then counts derived
 * from the union. Commutative, associative, and idempotent. Duplicate ids
 * shared by both sides are absorbed and never double-counted.
 */
export function mergeReplicas(a: Replica, b: Replica): Replica {
  if (a.k !== b.k) {
    throw new RangeError(`cannot merge replicas with K = ${a.k} and K = ${b.k}`);
  }
  return makeReplica(a.k, [...a.ledger, ...b.ledger]);
}

/** The task-level name for the replica join. */
export { mergeReplicas as merge };

/** Fold many replicas under the join. Requires at least one replica. */
export function mergeAll(replicas: readonly Replica[]): Replica {
  if (replicas.length === 0) {
    throw new RangeError('mergeAll requires at least one replica');
  }
  let acc = replicas[0]!;
  for (let i = 1; i < replicas.length; i++) acc = mergeReplicas(acc, replicas[i]!);
  return acc;
}

/** True when this observation identity is already in the ledger. */
export function hasObserved(replica: Replica, id: DeltaId): boolean {
  return replica.ledger.some((entry) => entry.id === id);
}

/** The exact posterior of a replica's evidence. */
export function replicaPosterior(replica: Replica): ExactPosterior {
  return exactPosterior(replica.counts);
}

/** The exact posterior term for one category of a replica. */
export function probe(replica: Replica, category: number): PosteriorTerm {
  assertCategory(replica.k, category, 'probe');
  return replicaPosterior(replica).terms[category]!;
}
