/**
 * execution-router.ts -- rank execution venues by rejection counts, exactly.
 *
 * An execution venue (a route, a broker, a dark pool) is scored by its rejection
 * evidence: failed fills, adverse selection, latency misses. The evidence is a
 * shared observation budget R and one rejection count v_i per venue. The God
 * Formula
 *
 *   w_i = R - min(v_i, R) + 1
 *
 * clamps a venue rejected more often than the budget instead of punishing it.
 * Writing c_i = R - min(v_i, R) for the implied successes, the normalized weight
 * is the add-one posterior
 *
 *   p_i = (c_i + 1) / (N + K),   N = sum_i c_i.
 *
 * The +1 is the never-collapse floor: a venue that missed every probe still
 * carries weight 1, so the router can always bring it back. skyRmsPeak is the
 * least rejected venue, and collapseRange returns the PROVEN sound band
 * [width/W, (R+1) width/W] for any target subset of venues.
 *
 * Adversarial dual: the raw MLE c_i / N prices a never-surviving venue at 0, so
 * an MLE router would abandon it forever. routingAdversarialDual names that.
 *
 * The shape mirrors open-source/gnosis/mesh-local-mcp/src/buleyean-router.ts.
 * This file re-implements it for finance use and does not import across packages.
 *
 * Nothing here is financial advice; it ranks rejection evidence, nothing more.
 */

import {
  exactFraction,
  fractionToNumber,
  type ExactFraction,
} from './rational.js';

/** Rejections attributed to one venue, by kind. */
export interface RejectionCounts {
  readonly failedFills: number;
  readonly adverseSelection: number;
  readonly latencyMisses: number;
}

/** Per-kind weights used to fold a rejection ledger into one integer. */
export interface RejectionWeights {
  readonly failedFills: number;
  readonly adverseSelection: number;
  readonly latencyMisses: number;
}

/** All three kinds count once by default. */
export const DEFAULT_REJECTION_WEIGHTS: RejectionWeights = {
  failedFills: 1,
  adverseSelection: 1,
  latencyMisses: 1,
};

/** One venue with an id and its rejection ledger. */
export interface ExecutionVenue {
  readonly id: string;
  readonly rejections: RejectionCounts;
}

/** The router's observation field: budget R plus one rejection count per venue. */
export interface ExecutionField {
  readonly budget: number;
  readonly rejections: readonly number[];
  readonly venues: readonly string[];
}

function assertNonNegativeInteger(value: number, where: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(where + ' = ' + String(value) + ' must be a non-negative safe integer');
  }
}

function assertField(field: ExecutionField): void {
  assertNonNegativeInteger(field.budget, 'budget');
  if (field.rejections.length === 0) {
    throw new RangeError('execution field requires at least one venue (K >= 1)');
  }
  if (field.venues.length !== field.rejections.length) {
    throw new RangeError('venues length must equal rejections length');
  }
  for (let i = 0; i < field.rejections.length; i++) {
    assertNonNegativeInteger(field.rejections[i]!, 'rejections[' + i + ']');
  }
}

/** Fold a rejection ledger into one integer under per-kind weights. */
export function rejectionTotal(
  rejections: RejectionCounts,
  weights: RejectionWeights = DEFAULT_REJECTION_WEIGHTS,
): number {
  assertNonNegativeInteger(rejections.failedFills, 'failedFills');
  assertNonNegativeInteger(rejections.adverseSelection, 'adverseSelection');
  assertNonNegativeInteger(rejections.latencyMisses, 'latencyMisses');
  assertNonNegativeInteger(weights.failedFills, 'weights.failedFills');
  assertNonNegativeInteger(weights.adverseSelection, 'weights.adverseSelection');
  assertNonNegativeInteger(weights.latencyMisses, 'weights.latencyMisses');
  return (
    rejections.failedFills * weights.failedFills +
    rejections.adverseSelection * weights.adverseSelection +
    rejections.latencyMisses * weights.latencyMisses
  );
}

/**
 * Build the field from venues. The budget defaults to the largest folded
 * rejection count, which guarantees c_i = R - min(v_i, R) is non-negative and
 * that at least one venue sits at the floor.
 */
export function executionField(
  venues: readonly ExecutionVenue[],
  budget?: number,
  weights: RejectionWeights = DEFAULT_REJECTION_WEIGHTS,
): ExecutionField {
  if (venues.length === 0) {
    throw new RangeError('executionField requires at least one venue');
  }
  const rejections = venues.map((venue) => rejectionTotal(venue.rejections, weights));
  let resolvedBudget = budget;
  if (resolvedBudget === undefined) {
    resolvedBudget = rejections.reduce((max, value) => (value > max ? value : max), 0);
  }
  assertNonNegativeInteger(resolvedBudget, 'budget');
  return {
    budget: resolvedBudget,
    rejections,
    venues: venues.map((venue) => venue.id),
  };
}

/**
 * The God Formula with the Lean clamp, exact BigInt:
 *   w(R, v) = R - min(v, R) + 1.
 */
export function godWeight(budget: number, rejections: number): bigint {
  assertNonNegativeInteger(budget, 'budget');
  assertNonNegativeInteger(rejections, 'rejections');
  const rounds = BigInt(budget);
  const value = BigInt(rejections);
  const clamped = value < rounds ? value : rounds;
  return rounds - clamped + 1n;
}

/** c_i = R - min(v_i, R): the implied successes behind each weight. */
export function laplaceCounts(field: ExecutionField): number[] {
  assertField(field);
  return field.rejections.map((rejections) => field.budget - Math.min(rejections, field.budget));
}

/** The Buleyean weight of every venue, exact. */
export function venueWeights(field: ExecutionField): bigint[] {
  assertField(field);
  return field.rejections.map((rejections) => godWeight(field.budget, rejections));
}

/** One venue's exact weight and reduced posterior. */
export interface VenuePosterior {
  readonly venue: number;
  readonly id: string;
  readonly rejections: number;
  readonly count: bigint;
  readonly weight: bigint;
  readonly numerator: bigint;
  readonly denominator: bigint;
  readonly exact: ExactFraction;
  readonly ratio: number;
}

/** The exact Buleyean posterior over venues plus display floats. */
export interface ExecutionPosterior {
  readonly k: number;
  readonly budget: number;
  readonly counts: readonly number[];
  readonly weights: readonly bigint[];
  readonly total: bigint;
  readonly totalWeight: bigint;
  readonly terms: readonly VenuePosterior[];
  readonly floats: readonly number[];
  readonly uniform: boolean;
}

function reduceBig(numerator: bigint, denominator: bigint): ExactFraction {
  return exactFraction(numerator, denominator);
}

/** The exact add-one posterior over venues. */
export function venuePosterior(field: ExecutionField): ExecutionPosterior {
  assertField(field);
  const counts = laplaceCounts(field);
  const weights = counts.map((count) => BigInt(count) + 1n);
  let total = 0n;
  for (const count of counts) total += BigInt(count);
  let totalWeight = 0n;
  for (const weight of weights) totalWeight += weight;

  const terms: VenuePosterior[] = weights.map((weight, venue) => ({
    venue,
    id: field.venues[venue]!,
    rejections: field.rejections[venue]!,
    count: BigInt(counts[venue]!),
    weight,
    numerator: weight,
    denominator: totalWeight,
    exact: reduceBig(weight, totalWeight),
    ratio: Number(weight) / Number(totalWeight),
  }));

  return {
    k: counts.length,
    budget: field.budget,
    counts,
    weights,
    total,
    totalWeight,
    terms,
    floats: terms.map((term) => term.ratio),
    uniform: total === 0n,
  };
}

/** The Skyrms peak: least rejected venue, ties broken by lowest index. */
export function skyRmsPeak(field: ExecutionField): number {
  assertField(field);
  let peak = 0;
  for (let i = 1; i < field.rejections.length; i++) {
    if (field.rejections[i]! < field.rejections[peak]!) peak = i;
  }
  return peak;
}

/** True when the least-rejected venue carries the greatest weight. */
export function peakIsMaxWeight(field: ExecutionField): boolean {
  const weights = venueWeights(field);
  const peak = skyRmsPeak(field);
  const peakWeight = weights[peak]!;
  return weights.every((weight) => weight <= peakWeight);
}

/** The never-collapse floor verdict. */
export interface FloorReport {
  readonly ok: boolean;
  readonly detail: string;
  readonly k: number;
  readonly minWeight: bigint;
  readonly minWeightVenue: number;
}

/** Law 1: every venue weight is at least the sliver 1. */
export function assertFloor(field: ExecutionField): FloorReport {
  const weights = venueWeights(field);
  let minWeight = weights[0]!;
  let minWeightVenue = 0;
  for (let i = 1; i < weights.length; i++) {
    if (weights[i]! < minWeight) {
      minWeight = weights[i]!;
      minWeightVenue = i;
    }
  }
  const ok = minWeight >= 1n;
  return {
    ok,
    detail: ok
      ? 'every venue weight w_i >= 1 (min ' + String(minWeight) + ' at venue ' + String(minWeightVenue) + '); the floor holds'
      : 'venue weight w_' + String(minWeightVenue) + ' = ' + String(minWeight) + ' < 1; the floor is broken',
    k: weights.length,
    minWeight,
    minWeightVenue,
  };
}

/** The exact integer reading of the urn/Laplace identity for the field. */
export interface RoutingLaplaceIdentity {
  readonly ok: boolean;
  readonly detail: string;
  readonly k: number;
  readonly total: bigint;
  readonly denominator: bigint;
  readonly numeratorSum: bigint;
}

/** Check w_i === c_i + 1 and sum_i (c_i + 1) === N + K, exactly. */
export function assertLaplaceIdentity(field: ExecutionField): RoutingLaplaceIdentity {
  assertField(field);
  const counts = laplaceCounts(field);
  const k = counts.length;
  const weights = venueWeights(field);
  let total = 0n;
  for (const count of counts) total += BigInt(count);
  const denominator = total + BigInt(k);
  let numeratorSum = 0n;
  for (let i = 0; i < k; i++) {
    const expected = BigInt(counts[i]!) + 1n;
    const actual = weights[i]!;
    if (actual !== expected) {
      return {
        ok: false,
        detail: 'w_' + String(i) + ' = ' + String(actual) + ' !== c_' + String(i) + ' + 1 = ' + String(expected),
        k,
        total,
        denominator,
        numeratorSum,
      };
    }
    numeratorSum += expected;
  }
  if (numeratorSum !== denominator) {
    return {
      ok: false,
      detail: 'sum_i (c_i + 1) = ' + String(numeratorSum) + ' !== N + K = ' + String(denominator),
      k,
      total,
      denominator,
      numeratorSum,
    };
  }
  return {
    ok: true,
    detail:
      'w_i = c_i + 1 for ' + String(k) + ' venues; sum = ' + String(numeratorSum) +
      ' = N + K = ' + String(total) + ' + ' + String(k),
    k,
    total,
    denominator,
    numeratorSum,
  };
}

/** The PROVEN collapse range for a target subset of venues. */
export interface CollapseRangeReport {
  readonly width: number;
  readonly budget: number;
  readonly mass: bigint;
  readonly total: bigint;
  readonly point: ExactFraction;
  readonly low: ExactFraction;
  readonly high: ExactFraction;
  readonly lowValue: number;
  readonly highValue: number;
  readonly pointValue: number;
  readonly inRange: boolean;
  readonly sound: true;
  readonly detail: string;
}

/**
 * The proven sound band for a target subset of width venues:
 *
 *   [width / W, (R + 1) * width / W],   W = sum_i w_i.
 *
 * Every weight lies in [1, R + 1], so any admissible target mass lies in that
 * band. The band is sound and deliberately wide; it is not an estimate.
 */
export function collapseRange(
  field: ExecutionField,
  target: readonly number[],
): CollapseRangeReport {
  const weights = venueWeights(field);
  const k = weights.length;
  for (const index of target) {
    if (!Number.isInteger(index) || index < 0 || index >= k) {
      throw new RangeError('target index ' + String(index) + ' is outside [0, ' + String(k) + ')');
    }
  }
  let total = 0n;
  for (const weight of weights) total += weight;
  let mass = 0n;
  for (const index of target) mass += weights[index]!;

  const width = target.length;
  const low = exactFraction(BigInt(width), total);
  const high = exactFraction(BigInt(field.budget + 1) * BigInt(width), total);
  const point = exactFraction(mass, total);
  const inRange = mass >= BigInt(width) && mass <= BigInt(field.budget + 1) * BigInt(width);
  return {
    width,
    budget: field.budget,
    mass,
    total,
    point,
    low,
    high,
    lowValue: fractionToNumber(low),
    highValue: fractionToNumber(high),
    pointValue: fractionToNumber(point),
    inRange,
    sound: true,
    detail:
      'target width ' + String(width) + ' of ' + String(k) + ': point ' + String(mass) + '/' + String(total) +
      ' in proven band [' + String(low.numerator) + '/' + String(low.denominator) +
      ', ' + String(high.numerator) + '/' + String(high.denominator) + ']; ' +
      (inRange ? 'inside' : 'outside'),
  };
}

/** One venue in the exact ranking. */
export interface RankedVenue {
  readonly rank: number;
  readonly venue: number;
  readonly id: string;
  readonly rejections: number;
  readonly count: bigint;
  readonly weight: bigint;
  readonly numerator: bigint;
  readonly denominator: bigint;
  readonly exact: ExactFraction;
  readonly ratio: number;
}

/**
 * Rank venues by descending posterior (equivalently ascending rejection count),
 * ties broken by lowest venue index so the order is deterministic.
 */
export function rankVenues(field: ExecutionField): RankedVenue[] {
  const posterior = venuePosterior(field);
  const ordered = [...posterior.terms].sort((left, right) => {
    if (left.weight !== right.weight) return left.weight > right.weight ? -1 : 1;
    return left.venue - right.venue;
  });
  return ordered.map((term, rank) => ({
    rank,
    venue: term.venue,
    id: term.id,
    rejections: term.rejections,
    count: term.count,
    weight: term.weight,
    numerator: term.numerator,
    denominator: term.denominator,
    exact: term.exact,
    ratio: term.ratio,
  }));
}

/** Seed used when selectVenue is called without an explicit RNG. */
export const DEFAULT_SELECT_SEED = 0xc0ffee;

/** Deterministic xorshift32 stream in [0, 1); zero seeds are remapped. */
export function seededRng(seed: number): () => number {
  let state = seed >>> 0;
  if (state === 0) state = 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

/** Choose one venue with probability proportional to its Buleyean weight. */
export function selectVenue(
  field: ExecutionField,
  rng: () => number = seededRng(DEFAULT_SELECT_SEED),
): number {
  const posterior = venuePosterior(field);
  const roll = rng();
  let cumulative = 0;
  for (let i = 0; i < posterior.floats.length; i++) {
    cumulative += posterior.floats[i]!;
    if (roll < cumulative) return i;
  }
  return posterior.floats.length - 1;
}

/** The adversarial dual: the MLE prices a never-surviving venue at zero. */
export interface RoutingAdversarialDual {
  readonly fires: boolean;
  readonly total: bigint;
  /** Venues with c_i = 0: the MLE prices them at exactly 0. */
  readonly zeroedVenues: readonly number[];
  readonly mostRejected: number;
  readonly floor: ExactFraction;
  readonly mleDefined: boolean;
  readonly detail: string;
}

/**
 * Show that a venue whose implied successes are zero is priced at 0 by the MLE
 * c_i / N and at 1/(N + K) by the floor. An MLE router abandons it forever; the
 * Buleyean router always leaves a route back.
 */
export function routingAdversarialDual(field: ExecutionField): RoutingAdversarialDual {
  const counts = laplaceCounts(field);
  const k = counts.length;
  const zeroed: number[] = [];
  let total = 0n;
  for (let i = 0; i < k; i++) {
    total += BigInt(counts[i]!);
    if (counts[i] === 0 && field.rejections[i]! > 0) zeroed.push(i);
  }
  const denominator = total + BigInt(k);
  const floor = exactFraction(1n, denominator);
  const mleDefined = total > 0n;
  const fires = mleDefined && zeroed.length > 0;
  let mostRejected = 0;
  for (let i = 1; i < k; i++) {
    if (field.rejections[i]! > field.rejections[mostRejected]!) mostRejected = i;
  }
  return {
    fires,
    total,
    zeroedVenues: zeroed,
    mostRejected,
    floor,
    mleDefined,
    detail: fires
      ? 'MLE prices venue(s) ' + zeroed.join(', ') + ' at 0 while the floor is 1/' +
        String(denominator) + '; an MLE router would never retry them'
      : 'no correctly-rejected venue is priced at zero by the MLE here',
  };
}
