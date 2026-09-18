/**
 * speculative-acceptance.ts -- the draft/verify acceptance posterior.
 *
 * A speculative decoder has two models: a cheap *drafter* that proposes draft
 * tokens and a slower *target* that accepts or rejects them. The evidence that
 * survives a run is a per-candidate acceptance count: draft candidate i was
 * accepted a_i times, out of A = sum_i a_i acceptances in all.
 *
 * Laplace's rule of succession -- read, as everywhere in this package, through
 * the clamped God Formula -- turns those counts into the add-one acceptance
 * posterior over K draft candidates:
 *
 *   P(i) = (a_i + 1) / (A + K),        A = sum_i a_i.
 *
 * The clamp w = A - min(v_i, A) + 1 with v_i = A - a_i collapses exactly onto
 * a_i + 1, so the +1 is the same sliver as in urn.ts: a candidate that has
 * never been accepted still carries numerator 1 and is never permanently
 * pruned. The denominator A + K is the exact integer home of that sliver.
 *
 * acceptancePosterior returns exact BigInt numerator/denominator pairs with a
 * display-only float alongside; assertLaplaceIdentity checks the integer
 * backbone (numerator_i === a_i + 1 and sum_i numerator_i === A + K) with no
 * floating-point truth claim. rankDraftHeads orders candidates by descending
 * posterior; draftLengthPolicy and recommendedBlockLength pick the smallest
 * bounded length whose cumulative posterior reaches a target mass; and
 * blockAcceptanceProfile reads the same posterior indexed by block position.
 *
 * ---------------------------------------------------------------------------
 * Scope: this is a selection posterior, NOT a token sampler
 * ---------------------------------------------------------------------------
 *
 * The acceptance posterior answers "which draft candidate will the target
 * accept next?" It sizes drafts and orders draft heads. It must NOT replace the
 * target's softmax with the affine rule w = R - v + 1. The measured counterpoint
 * lives in ../experiments/softmax-vs-buleyean-attention/README.md: a normalized
 * affine ramp puts an O(1/N) ceiling
 * (C + s_max) / (N*C + sum_j s_j) on a single key that no inverse temperature
 * rescues. At N = 64 the affine target mass saturates at 0.0309 against the
 * analytic ceiling 0.0311 even at beta = 64, while softmax reaches 0.999999; the
 * affine rule matches softmax only in the near-uniform regime where both are at
 * chance. The Gnosis wiring note
 * open-source/gnosis/distributed-inference/SPECULATIVE_ACCEPTANCE_INTEGRATION.md
 * records that boundary explicitly.
 *
 * Nothing here proves anything. It is the executable arithmetic, checked exactly
 * against the urn/Laplace identity in BigInt.
 */

import { coerceCounts, type Counts } from './void-crdt.js';

/**
 * Acceptance counts: one non-negative integer per draft candidate (or per
 * block position). Number entries must be safe integers; use BigInt when a
 * single count can exceed 2^53 - 1.
 */
export type AcceptanceCounts = Counts;

/** The never-collapse floor: every candidate carries numerator >= 1. */
export const ACCEPTANCE_FLOOR = 1n;

/** Approximate bit length of a positive BigInt (0 for 0 or negative). */
function bigintBitLength(value: bigint): number {
  if (value <= 0n) return 0;
  let bits = 0;
  let v = value;
  while (v >= 1n << 64n) {
    v >>= 64n;
    bits += 64;
  }
  let small = Number(v);
  while (small >= 1) {
    small = Math.floor(small / 2);
    bits += 1;
  }
  return bits;
}

/**
 * Display-only ratio from an exact BigInt fraction. For fractions whose
 * numerator or denominator exceeds the double range, both sides are shifted
 * down by the same number of bits so the ratio stays finite instead of
 * collapsing to NaN. No truth claim is made on the float.
 */
function ratioToNumber(numerator: bigint, denominator: bigint): number {
  if (numerator <= 0n || denominator <= 0n) return 0;
  if (numerator === denominator) return 1;
  const bits = Math.max(bigintBitLength(numerator), bigintBitLength(denominator));
  if (bits <= 1023) return Number(numerator) / Number(denominator);
  const drop = BigInt(bits - 1023);
  const scaledDenominator = Number(denominator >> drop);
  if (!(scaledDenominator > 0)) return 0;
  return Number(numerator >> drop) / scaledDenominator;
}

/** One candidate of the exact acceptance posterior. */
export interface AcceptanceTerm {
  /** Draft candidate index in [0, K). */
  readonly candidate: number;
  /** a_i, the exact acceptance count. */
  readonly accepts: bigint;
  /** a_i + 1, the Laplace numerator. */
  readonly numerator: bigint;
  /** A + K, the shared denominator. */
  readonly denominator: bigint;
  /** Display-only float; the truth claim lives in numerator/denominator. */
  readonly posterior: number;
}

/** The exact add-one acceptance posterior over K draft candidates. */
export interface AcceptancePosterior {
  readonly k: number;
  /** A = sum_i a_i, the total acceptance evidence. */
  readonly total: bigint;
  /** A + K, the shared denominator. */
  readonly denominator: bigint;
  readonly terms: readonly AcceptanceTerm[];
  /** True when A = 0, i.e. the uniform 1/K posterior. */
  readonly uniform: boolean;
}

function sumCounts(counts: readonly bigint[]): bigint {
  let total = 0n;
  for (const count of counts) total += count;
  return total;
}

/**
 * The exact acceptance posterior as integer numerator/denominator pairs.
 *
 *   numerator_i = a_i + 1
 *   denominator = A + K
 *
 * Every term shares the denominator, so the posterior is one integer vector
 * over one integer scalar. No floating point is involved in the claim; each
 * posterior float is display only. Throws a RangeError on an empty candidate
 * set or on a non-integer / negative count.
 */
export function acceptancePosterior(accepts: AcceptanceCounts): AcceptancePosterior {
  const big = coerceCounts(accepts);
  const k = big.length;
  if (k === 0) {
    throw new RangeError('acceptancePosterior requires at least one draft candidate');
  }

  const total = sumCounts(big);
  const denominator = total + BigInt(k);
  const terms: AcceptanceTerm[] = big.map((count, candidate) => {
    const numerator = count + 1n;
    return {
      candidate,
      accepts: count,
      numerator,
      denominator,
      posterior: ratioToNumber(numerator, denominator),
    };
  });

  return { k, total, denominator, terms, uniform: total === 0n };
}

/** One draft head in the descending acceptance ranking. */
export interface RankedDraftHead {
  /** 0-based rank; 0 is the most accepted candidate. */
  readonly rank: number;
  readonly candidate: number;
  readonly accepts: bigint;
  readonly numerator: bigint;
  readonly denominator: bigint;
  readonly posterior: number;
}

/**
 * Rank draft candidates by descending acceptance posterior, ties broken by
 * lowest candidate index so the order is deterministic. The comparison is
 * exact (shared denominator, so numerator order is posterior order) and every
 * candidate is retained -- the floor means ranking cannot prune a head.
 */
export function rankDraftHeads(accepts: AcceptanceCounts): RankedDraftHead[] {
  const posterior = acceptancePosterior(accepts);
  const ordered = [...posterior.terms].sort((left, right) => {
    if (left.numerator !== right.numerator) return left.numerator > right.numerator ? -1 : 1;
    return left.candidate - right.candidate;
  });
  return ordered.map((term, rank) => ({
    rank,
    candidate: term.candidate,
    accepts: term.accepts,
    numerator: term.numerator,
    denominator: term.denominator,
    posterior: term.posterior,
  }));
}

/** Options for the bounded target-mass length policy. */
export interface LengthPolicyOptions {
  /** Smallest draft length the caller will accept. Defaults to 1. */
  readonly minLen?: number;
  /** Largest draft length the caller will accept. Defaults to K. */
  readonly maxLen?: number;
  /** Cumulative posterior mass the draft length must reach. Defaults to 1. */
  readonly targetMass?: number;
}

/** The chosen draft/block length plus the exact cumulative mass behind it. */
export interface LengthPolicyResult {
  readonly length: number;
  /** The effective (clamped) lower bound actually used. */
  readonly minLen: number;
  /** The effective (clamped) upper bound actually used. */
  readonly maxLen: number;
  readonly targetMass: number;
  /** Display-only cumulative posterior at length. */
  readonly achievedMass: number;
  /** Exact cumulative numerator sum_{i < length} (a_i + 1). */
  readonly cumulativeNumerator: bigint;
  readonly denominator: bigint;
  /** True when achievedMass >= targetMass (or targetMass <= 0). */
  readonly reached: boolean;
}

/**
 * Choose the smallest length L in the effective [minLen, maxLen] whose cumulative
 * posterior sum_{i < L} P(i) reaches targetMass. The bounds are clamped into
 * [0, K], and maxLen is never allowed below minLen. If targetMass is unreachable
 * within maxLen, maxLen is returned with reached: false.
 *
 * The posterior masses are exact integers; only the comparison threshold is the
 * caller's display float, so the chosen length is monotone non-decreasing in
 * targetMass.
 */
function selectLength(
  terms: readonly AcceptanceTerm[],
  options: LengthPolicyOptions,
): LengthPolicyResult {
  const k = terms.length;
  if (k === 0) {
    throw new RangeError('length policy requires at least one draft candidate');
  }
  const denominator = terms[0]!.denominator;

  const requestedMin = options.minLen ?? 1;
  const requestedMax = options.maxLen ?? k;
  const requestedTarget = options.targetMass ?? 1;

  const minLen = Math.max(
    0,
    Math.min(Math.floor(Number.isFinite(requestedMin) ? requestedMin : 1), k),
  );
  const maxLen = Math.max(
    minLen,
    Math.min(Math.floor(Number.isFinite(requestedMax) ? requestedMax : k), k),
  );
  const targetMass = Number.isFinite(requestedTarget) ? requestedTarget : 1;

  let cumulative = 0n;
  for (let i = 0; i < minLen; i++) cumulative += terms[i]!.numerator;

  const result = (length: number, reached: boolean): LengthPolicyResult => ({
    length,
    minLen,
    maxLen,
    targetMass,
    achievedMass: ratioToNumber(cumulative, denominator),
    cumulativeNumerator: cumulative,
    denominator,
    reached,
  });

  if (!(targetMass > 0)) return result(minLen, true);

  for (let length = minLen; length <= maxLen; length++) {
    if (length > minLen) cumulative += terms[length - 1]!.numerator;
    if (ratioToNumber(cumulative, denominator) >= targetMass) return result(length, true);
  }

  return result(maxLen, false);
}

/** The bounded target-mass draft-length policy over draft candidates. */
export function draftLengthPolicy(
  accepts: AcceptanceCounts,
  options: LengthPolicyOptions = {},
): LengthPolicyResult {
  return selectLength(acceptancePosterior(accepts).terms, options);
}

/** The bounded target-mass block-length policy over block positions. */
export function recommendedBlockLength(
  countsByPosition: AcceptanceCounts,
  options: LengthPolicyOptions = {},
): LengthPolicyResult {
  return selectLength(acceptancePosterior(countsByPosition).terms, options);
}

/** The exact expected acceptance of one draft candidate. */
export interface ExpectedAcceptance {
  readonly candidate: number;
  readonly accepts: bigint;
  readonly numerator: bigint;
  readonly denominator: bigint;
  readonly posterior: number;
}

/**
 * The expected acceptance of one candidate: the exact posterior pair that a
 * future accepted proposal is candidate i, plus the display float. Throws a
 * RangeError when candidate is outside [0, K).
 */
export function expectedAcceptance(
  accepts: AcceptanceCounts,
  candidate: number,
): ExpectedAcceptance {
  const posterior = acceptancePosterior(accepts);
  if (!Number.isInteger(candidate) || candidate < 0 || candidate >= posterior.k) {
    throw new RangeError(
      'candidate = ' + candidate + ' is outside [0, ' + posterior.k + ')',
    );
  }
  const term = posterior.terms[candidate]!;
  return {
    candidate: term.candidate,
    accepts: term.accepts,
    numerator: term.numerator,
    denominator: term.denominator,
    posterior: term.posterior,
  };
}

/** One block position of the block-drafter profile. */
export interface BlockPositionTerm {
  readonly position: number;
  readonly accepts: bigint;
  readonly numerator: bigint;
  readonly denominator: bigint;
  /** Display-only posterior share P(position ends the accepted run). */
  readonly posterior: number;
  /** Display-only P(accepted run ends at or before this position). */
  readonly cumulative: number;
  /** Display-only P(accepted run continues past this position). */
  readonly survival: number;
}

/** The block-drafter acceptance profile over block positions. */
export interface BlockAcceptanceProfile {
  readonly positions: number;
  readonly total: bigint;
  readonly denominator: bigint;
  readonly terms: readonly BlockPositionTerm[];
  /** Display-only expected length sum_j (j + 1) P(j). */
  readonly expectedLength: number;
  readonly uniform: boolean;
}

/**
 * Profile a block drafter's acceptance counts by block position.
 *
 * The same add-one posterior is read with the block position as the outcome:
 * P(j) = (a_j + 1) / (A + B), the posterior that an accepted block's deepest
 * accepted position is j (length j + 1). The +1 floor keeps every position
 * possible, so A = 0 yields the uniform 1/B profile and expected length
 * (B + 1)/2. Throws on an empty position list or a malformed count.
 */
export function blockAcceptanceProfile(countsByPosition: AcceptanceCounts): BlockAcceptanceProfile {
  const posterior = acceptancePosterior(countsByPosition);
  let cumulative = 0;
  let expectedLength = 0;
  const terms: BlockPositionTerm[] = posterior.terms.map((term) => {
    cumulative += term.posterior;
    expectedLength += (term.candidate + 1) * term.posterior;
    return {
      position: term.candidate,
      accepts: term.accepts,
      numerator: term.numerator,
      denominator: term.denominator,
      posterior: term.posterior,
      cumulative,
      survival: 1 - cumulative,
    };
  });
  return {
    positions: posterior.k,
    total: posterior.total,
    denominator: posterior.denominator,
    terms,
    expectedLength,
    uniform: posterior.uniform,
  };
}

/** The exact integer reading of the acceptance/Laplace identity. */
export interface AcceptanceLaplaceIdentity {
  readonly ok: boolean;
  readonly detail: string;
  readonly k: number;
  readonly total: bigint;
  readonly denominator: bigint;
  readonly numeratorSum: bigint;
  /** True when every numerator is >= 1 (no candidate permanently pruned). */
  readonly floorPreserved: boolean;
}

/**
 * Check the integer backbone of the acceptance posterior with no floating-point
 * truth claim:
 *
 *   numerator_i === a_i + 1 for every i, and
 *   sum_i (a_i + 1) === A + K.
 *
 * Passing means the posterior's numerators and denominator are exactly the
 * Laplace numerator and denominator at any magnitude BigInt can hold, and that
 * the +1 floor left no candidate at zero.
 */
export function assertLaplaceIdentity(accepts: AcceptanceCounts): AcceptanceLaplaceIdentity {
  const posterior = acceptancePosterior(accepts);
  let numeratorSum = 0n;
  let floorPreserved = true;

  for (let i = 0; i < posterior.k; i++) {
    const term = posterior.terms[i]!;
    const expected = term.accepts + 1n;
    if (term.numerator !== expected) {
      return {
        ok: false,
        detail: 'numerator_' + i + ' = ' + term.numerator + ' !== accepts[' + i + '] + 1 = ' + expected,
        k: posterior.k,
        total: posterior.total,
        denominator: posterior.denominator,
        numeratorSum,
        floorPreserved,
      };
    }
    if (term.numerator < ACCEPTANCE_FLOOR) floorPreserved = false;
    numeratorSum += expected;
  }

  if (numeratorSum !== posterior.denominator) {
    return {
      ok: false,
      detail: 'sum_i (accepts[i] + 1) = ' + numeratorSum + ' !== A + K = ' + posterior.denominator,
      k: posterior.k,
      total: posterior.total,
      denominator: posterior.denominator,
      numeratorSum,
      floorPreserved,
    };
  }

  return {
    ok: true,
    detail:
      'numerator_i = accepts[i] + 1 for ' + posterior.k + ' candidates; ' +
      'sum_i numerator_i = ' + numeratorSum + ' = A + K = ' + posterior.total + ' + ' + posterior.k + '; ' +
      'floor preserved (every numerator >= ' + ACCEPTANCE_FLOOR + ')',
    k: posterior.k,
    total: posterior.total,
    denominator: posterior.denominator,
    numeratorSum,
    floorPreserved,
  };
}
