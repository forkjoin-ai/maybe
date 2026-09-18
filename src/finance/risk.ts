/**
 * risk.ts -- exact expectation and the never-zero tail for a loss distribution.
 *
 * The distribution is the add-one (Laplace/Buleyean) posterior over K loss
 * states. With counts c_i and N = sum_i c_i,
 *
 *   P(state i) = (c_i + 1) / (N + K).
 *
 * The +1 is the sliver. It is the whole point of this module: an UNOBSERVED
 * catastrophe -- a state with c_i = 0 -- is priced at exactly 1/(N + K), never
 * at 0. The MLE posterior c_i / N prices that same unobserved state at 0, an
 * infinite underpricing, which is the adversarial dual this module names and
 * demonstrates rather than hides.
 *
 * Nothing here is financial advice. These are exact arithmetic primitives over
 * a caller-supplied loss vector; the caller owns every interpretation.
 */

import { coerceCounts, type Counts } from '../void-crdt.js';
import {
  compareFractions,
  exactFraction,
  fractionToNumber,
  isZeroFraction,
  mulFractions,
  toFraction,
  type ExactFraction,
  type RationalInput,
} from './rational.js';

/** One loss magnitude per state: a safe integer or an exact bigint. */
export type LossVector = readonly (number | bigint)[];

/** Coerce loss magnitudes to exact BigInt, rejecting unsafe doubles. */
export function coerceLosses(losses: LossVector, where = 'losses'): bigint[] {
  const out: bigint[] = [];
  for (let i = 0; i < losses.length; i++) {
    const loss = losses[i]!;
    if (typeof loss === 'bigint') {
      out.push(loss);
      continue;
    }
    if (!Number.isSafeInteger(loss)) {
      throw new RangeError(
        where + '[' + i + '] = ' + String(loss) + ' is not a safe integer; use a bigint for exact extreme losses',
      );
    }
    out.push(BigInt(loss));
  }
  return out;
}

/** One state's posterior probability and its exact loss contribution. */
export interface LossContribution {
  readonly category: number;
  readonly loss: bigint;
  readonly count: bigint;
  /** (count + 1) / (N + K), exact and reduced. */
  readonly probability: ExactFraction;
  /** loss * probability, exact and reduced. */
  readonly contribution: ExactFraction;
}

/** The exact expected loss under the add-one posterior. */
export interface ExactExpectation {
  /** sum_i losses[i] * (counts[i] + 1) reduced over (N + K). */
  readonly numerator: bigint;
  readonly denominator: bigint;
  /** Display-only float; the truth claim lives in numerator/denominator. */
  readonly value: number;
  /** The unreduced numerator sum_i losses[i] * (counts[i] + 1). */
  readonly rawNumerator: bigint;
  /** N + K, the shared posterior denominator. */
  readonly posteriorDenominator: bigint;
  /** N = sum_i counts[i]. */
  readonly total: bigint;
  readonly categories: number;
  readonly terms: readonly LossContribution[];
}

/**
 * Exact expectation E[loss] = sum_i losses[i] * (counts[i] + 1) / (N + K).
 *
 * The add-one posterior is used unconditionally, so every state keeps weight
 * at least 1 and an unobserved catastrophe always contributes.
 */
export function exactExpectation(counts: Counts, losses: LossVector): ExactExpectation {
  const big = coerceCounts(counts);
  const k = big.length;
  if (k === 0) {
    throw new RangeError('exactExpectation requires at least one loss state');
  }
  if (losses.length !== k) {
    throw new RangeError(
      'losses length ' + losses.length + ' must equal counts length ' + k,
    );
  }
  const lossBig = coerceLosses(losses);

  let total = 0n;
  for (const count of big) total += count;
  const denominator = total + BigInt(k);

  const terms: LossContribution[] = [];
  let rawNumerator = 0n;
  for (let i = 0; i < k; i++) {
    const count = big[i]!;
    const probability = exactFraction(count + 1n, denominator);
    const contribution = mulFractions(lossBig[i]!, probability);
    rawNumerator += lossBig[i]! * (count + 1n);
    terms.push({ category: i, loss: lossBig[i]!, count, probability, contribution });
  }

  const expectation = exactFraction(rawNumerator, denominator);
  return {
    numerator: expectation.numerator,
    denominator: expectation.denominator,
    value: fractionToNumber(expectation),
    rawNumerator,
    posteriorDenominator: denominator,
    total,
    categories: k,
    terms,
  };
}

/** The exact sandwich min(loss) <= E[loss] <= max(loss), decided without floats. */
export interface LossSandwichReport {
  readonly ok: boolean;
  readonly minLoss: bigint;
  readonly maxLoss: bigint;
  readonly expectation: ExactFraction;
  readonly value: number;
  readonly total: bigint;
  readonly categories: number;
  readonly posteriorDenominator: bigint;
  readonly detail: string;
}

/**
 * Prove the sandwich minLoss <= E[loss] <= maxLoss exactly. Because the add-one
 * posterior has K positive weights summing to N + K, the expectation is a
 * convex combination of the losses, so the sandwich always holds. The report
 * decides it with integer cross-multiplication, never a float.
 */
export function assertLossSandwich(counts: Counts, losses: LossVector): LossSandwichReport {
  const expectation = exactExpectation(counts, losses);
  const lossBig = coerceLosses(losses);
  let minLoss = lossBig[0]!;
  let maxLoss = lossBig[0]!;
  for (const loss of lossBig) {
    if (loss < minLoss) minLoss = loss;
    if (loss > maxLoss) maxLoss = loss;
  }
  const lower = exactFraction(minLoss, 1n);
  const upper = exactFraction(maxLoss, 1n);
  const atLeastMin = compareFractions(expectation, lower) >= 0;
  const atMostMax = compareFractions(expectation, upper) <= 0;
  const ok = atLeastMin && atMostMax;
  return {
    ok,
    minLoss,
    maxLoss,
    expectation: { numerator: expectation.numerator, denominator: expectation.denominator },
    value: expectation.value,
    total: expectation.total,
    categories: expectation.categories,
    posteriorDenominator: expectation.posteriorDenominator,
    detail:
      'minLoss = ' + String(minLoss) + ', E[loss] = ' +
      String(expectation.numerator) + '/' + String(expectation.denominator) +
      ', maxLoss = ' + String(maxLoss) + '; sandwich ' + (ok ? 'holds' : 'broken'),
  };
}

/** The never-zero tail: every state's weight, and the price of the unobserved. */
export interface NeverZeroTailReport {
  readonly k: number;
  readonly total: bigint;
  readonly posteriorDenominator: bigint;
  readonly weights: readonly bigint[];
  readonly minWeight: bigint;
  readonly minWeightState: number;
  readonly everyStateAlive: boolean;
  /** Indices with count 0: states never observed. */
  readonly unobserved: readonly number[];
  readonly hasUnobserved: boolean;
  /** 1 / (N + K): the exact floor price of any state, observed or not. */
  readonly unobservedFloor: ExactFraction;
  readonly unobservedFloorValue: number;
  readonly detail: string;
}

/**
 * Show that every state has weight c_i + 1 >= 1, so an unobserved catastrophe
 * is priced at exactly 1/(N + K). The tail cannot be rounded to zero by the
 * arithmetic; only an explicit policy could do that, and this report exposes
 * the fact instead.
 */
export function neverZeroTail(counts: Counts): NeverZeroTailReport {
  const big = coerceCounts(counts);
  const k = big.length;
  if (k === 0) {
    throw new RangeError('neverZeroTail requires at least one state');
  }
  let total = 0n;
  for (const count of big) total += count;
  const denominator = total + BigInt(k);

  const weights: bigint[] = [];
  const unobserved: number[] = [];
  let minWeight = big[0]! + 1n;
  let minWeightState = 0;
  for (let i = 0; i < k; i++) {
    const weight = big[i]! + 1n;
    weights.push(weight);
    if (weight < minWeight) {
      minWeight = weight;
      minWeightState = i;
    }
    if (big[i] === 0n) unobserved.push(i);
  }

  const floor = exactFraction(1n, denominator);
  const everyStateAlive = minWeight >= 1n;
  return {
    k,
    total,
    posteriorDenominator: denominator,
    weights,
    minWeight,
    minWeightState,
    everyStateAlive,
    unobserved,
    hasUnobserved: unobserved.length > 0,
    unobservedFloor: floor,
    unobservedFloorValue: fractionToNumber(floor),
    detail:
      'every weight >= 1 (min ' + String(minWeight) + ' at state ' + String(minWeightState) + '); ' +
      unobserved.length + ' unobserved state(s) priced at 1/(N+K) = 1/' + String(denominator) +
      ' = ' + String(floor.numerator) + '/' + String(floor.denominator),
  };
}

/** The maximum-likelihood posterior c_i / N, kept only to expose its failure. */
export interface MlePosterior {
  readonly k: number;
  readonly total: bigint;
  /** False when N = 0, where the MLE has no value at all. */
  readonly defined: boolean;
  readonly probabilities: readonly ExactFraction[];
  /** Indices the MLE prices at exactly zero. */
  readonly zeroStates: readonly number[];
  readonly detail: string;
}

/**
 * The raw MLE posterior p_i = c_i / N. It is NOT used for any decision in this
 * package; it exists so the adversarial dual can demonstrate that it prices an
 * unobserved state at zero. When N = 0 it is undefined and reported as such.
 */
export function mlePosterior(counts: Counts): MlePosterior {
  const big = coerceCounts(counts);
  const k = big.length;
  if (k === 0) {
    throw new RangeError('mlePosterior requires at least one state');
  }
  let total = 0n;
  for (const count of big) total += count;

  if (total === 0n) {
    const zeroStates: number[] = [];
    for (let i = 0; i < k; i++) zeroStates.push(i);
    return {
      k,
      total,
      defined: false,
      probabilities: big.map(() => exactFraction(0n, 1n)),
      zeroStates,
      detail: 'N = 0: the MLE posterior is undefined and prices every state at 0',
    };
  }

  const probabilities = big.map((count) => exactFraction(count, total));
  const zeroStates: number[] = [];
  for (let i = 0; i < k; i++) {
    if (big[i] === 0n) zeroStates.push(i);
  }
  return {
    k,
    total,
    defined: true,
    probabilities,
    zeroStates,
    detail:
      'MLE p_i = c_i / N; ' + zeroStates.length +
      ' unobserved state(s) priced at exactly 0',
  };
}

/** The adversarial dual: the MLE underprices an unobserved tail at zero. */
export interface TailAdversarialDual {
  readonly fires: boolean;
  readonly unobserved: readonly number[];
  readonly mleDefined: boolean;
  /** 1/(N + K), the add-one floor the MLE does not have. */
  readonly addOneFloor: ExactFraction;
  readonly addOneFloorValue: number;
  /** The MLE price of the first unobserved state; exactly 0. */
  readonly mlePrice: ExactFraction;
  /** True when the MLE price is 0 and the add-one floor is positive. */
  readonly underpricingInfinite: boolean;
  readonly detail: string;
}

/**
 * Demonstrate the failure mode of the MLE posterior: when some state has never
 * been observed, the MLE prices it at 0, which is an infinite underpricing of a
 * tail relative to the add-one floor 1/(N + K). A decision rule that trusts the
 * MLE would never budget for that catastrophe at all.
 */
export function tailAdversarialDual(counts: Counts): TailAdversarialDual {
  const mle = mlePosterior(counts);
  const neverZero = neverZeroTail(counts);
  const unobserved = neverZero.unobserved;
  const fires = unobserved.length > 0;
  const firstUnobserved = fires ? unobserved[0]! : 0;
  const mlePrice = mle.probabilities[firstUnobserved] ?? exactFraction(0n, 1n);
  const underpricingInfinite =
    fires && isZeroFraction(mlePrice) && !isZeroFraction(neverZero.unobservedFloor);
  return {
    fires,
    unobserved,
    mleDefined: mle.defined,
    addOneFloor: neverZero.unobservedFloor,
    addOneFloorValue: neverZero.unobservedFloorValue,
    mlePrice,
    underpricingInfinite,
    detail: fires
      ? 'MLE prices unobserved state ' + String(firstUnobserved) +
        ' at ' + String(mlePrice.numerator) + '/' + String(mlePrice.denominator) +
        ' while the add-one floor is 1/' + String(neverZero.posteriorDenominator) +
        '; the underpricing is ' + (underpricingInfinite ? 'infinite' : 'finite')
      : 'no unobserved state; the adversarial dual does not fire',
  };
}

/** The exact add-one posterior mass on losses strictly above a threshold. */
export interface TailMassBound {
  readonly threshold: ExactFraction;
  readonly strictAbove: true;
  /** Tail mass reduced to lowest terms. */
  readonly numerator: bigint;
  readonly denominator: bigint;
  readonly rawNumerator: bigint;
  readonly posteriorDenominator: bigint;
  readonly mass: number;
  /** Indices whose loss is strictly above the threshold. */
  readonly tail: readonly number[];
  /** Tail probability numerator before reduction. */
  readonly tailWeight: bigint;
  readonly detail: string;
}

/**
 * Exact mass above a loss threshold:
 *
 *   P(loss > threshold) = sum_{i: losses[i] > threshold} (counts[i] + 1) / (N + K).
 *
 * The comparison is exact integer cross-multiplication; the returned fraction is
 * reduced. This is a tight exact tail, not a probabilistic estimate.
 */
export function tailMassBound(
  counts: Counts,
  losses: LossVector,
  threshold: RationalInput,
): TailMassBound {
  const big = coerceCounts(counts);
  const k = big.length;
  if (k === 0) {
    throw new RangeError('tailMassBound requires at least one loss state');
  }
  const lossBig = coerceLosses(losses);
  if (lossBig.length !== k) {
    throw new RangeError(
      'losses length ' + lossBig.length + ' must equal counts length ' + k,
    );
  }
  const t = toFraction(threshold, 'threshold');

  let total = 0n;
  for (const count of big) total += count;
  const denominator = total + BigInt(k);

  const tail: number[] = [];
  let rawNumerator = 0n;
  for (let i = 0; i < k; i++) {
    // loss/1 > t.numerator/t.denominator  <=>  loss * t.denominator > t.numerator
    if (lossBig[i]! * t.denominator > t.numerator) {
      tail.push(i);
      rawNumerator += big[i]! + 1n;
    }
  }

  const mass = exactFraction(rawNumerator, denominator);
  return {
    threshold: t,
    strictAbove: true,
    numerator: mass.numerator,
    denominator: mass.denominator,
    rawNumerator,
    posteriorDenominator: denominator,
    mass: fractionToNumber(mass),
    tail,
    tailWeight: rawNumerator,
    detail:
      'P(loss > ' + String(t.numerator) + '/' + String(t.denominator) + ') = ' +
      String(mass.numerator) + '/' + String(mass.denominator) +
      ' over ' + tail.length + ' tail state(s)',
  };
}
