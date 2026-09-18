/**
 * kelly.ts -- exact bet sizing on the Buleyean/Laplace posterior.
 *
 * Kelly sizing maximises the expected logarithm of wealth for a binary bet with
 * net odds b (win b per unit staked, lose 1 per unit). For win probability p and
 * loss probability q = 1 - p, the Kelly fraction is
 *
 *   f* = (b p - q) / b.
 *
 * This module computes f* as an exact rational. buleyeanKelly plugs in the
 * add-one posterior p = (c_win + 1) / (N + K) instead of the raw success rate,
 * and kellyRangeFromPosterior reads the PROVEN collapse range
 * [1/(N+K), (N+1)/(N+K)] for that same posterior. fractionalKellyFromRange and
 * robustKellyFromRange size at the CONSERVATIVE lower endpoint: an
 * ambiguity-averse Kelly that never bets the optimistic end of a proven range.
 *
 * Adversarial dual: on a small sample the raw MLE plugs in an overconfident p
 * and overbets relative to add-one. kellyAdversarialDual returns that concrete
 * comparison (see 1 win in 1 trial, second outcome unobserved: MLE stakes 1,
 * add-one stakes 1/3 at even odds).
 *
 * Nothing here is financial advice and no profitability is claimed. Kelly is an
 * arithmetic rule for a stated model; the model and the odds are the caller's.
 */

import { coerceCounts, type Counts } from '../void-crdt.js';
import {
  compareFractions,
  divFractions,
  exactFraction,
  fractionToNumber,
  isPositiveFraction,
  mulFractions,
  toFraction,
  type ExactFraction,
  type RationalInput,
} from './rational.js';

/** Net odds or a multiplier: a number, bigint, or exact fraction. */
export type OddsInput = RationalInput;

/** The full Kelly decision for one stated probability and net odds. */
export interface KellyDecision {
  /** p, the win probability used. */
  readonly probability: ExactFraction;
  /** b, the net odds. */
  readonly odds: ExactFraction;
  /** f* = (b p - q) / b, signed and exact. */
  readonly fraction: ExactFraction;
  /** Display-only float of f*. */
  readonly value: number;
  /** max(f*, 0): the recommended stake, never negative. */
  readonly recommendedFraction: ExactFraction;
  readonly shouldBet: boolean;
  readonly detail: string;
}

function kellyCore(probability: ExactFraction, odds: ExactFraction): ExactFraction {
  if (odds.numerator <= 0n) {
    throw new RangeError('net odds b must be strictly positive');
  }
  if (probability.numerator < 0n || compareFractions(probability, exactFraction(1n, 1n)) > 0) {
    throw new RangeError('win probability p must lie in [0, 1]');
  }
  const p = probability;
  // f* = (b p - q) / b = [bNum pNum - bDen (pDen - pNum)] / (bNum pDen)
  const numerator = odds.numerator * p.numerator - odds.denominator * (p.denominator - p.numerator);
  const denominator = odds.numerator * p.denominator;
  return exactFraction(numerator, denominator);
}

/**
 * The exact Kelly fraction (b p - q) / b. The result is signed: a negative
 * fraction means the edge does not exist and no bet is warranted.
 */
export function kellyFraction(
  pNum: bigint,
  pDen: bigint,
  bNum: bigint,
  bDen: bigint,
): ExactFraction {
  const probability = exactFraction(pNum, pDen);
  const odds = exactFraction(bNum, bDen);
  return kellyCore(probability, odds);
}

/** The full decision (fraction, recommended non-negative stake, shouldBet). */
export function kellyDecision(
  pNum: bigint,
  pDen: bigint,
  bNum: bigint,
  bDen: bigint,
): KellyDecision {
  const probability = exactFraction(pNum, pDen);
  const odds = exactFraction(bNum, bDen);
  const fraction = kellyCore(probability, odds);
  return {
    probability,
    odds,
    fraction,
    value: fractionToNumber(fraction),
    recommendedFraction: isPositiveFraction(fraction) ? fraction : exactFraction(0n, 1n),
    shouldBet: isPositiveFraction(fraction),
    detail:
      'p = ' + String(probability.numerator) + '/' + String(probability.denominator) +
      ', b = ' + String(odds.numerator) + '/' + String(odds.denominator) +
      ', f* = ' + String(fraction.numerator) + '/' + String(fraction.denominator) +
      (isPositiveFraction(fraction) ? ' (bet)' : ' (no bet)'),
  };
}

/** A Buleyean Kelly decision, carrying the posterior it used. */
export interface BuleyeanKellyDecision extends KellyDecision {
  readonly win: number;
  readonly counts: readonly bigint[];
  readonly total: bigint;
  readonly k: number;
}

function winPosterior(counts: Counts, win: number, addOne: boolean): {
  countsBig: bigint[];
  total: bigint;
  k: number;
  probability: ExactFraction;
} {
  const countsBig = coerceCounts(counts);
  const k = countsBig.length;
  if (k === 0) {
    throw new RangeError('Kelly requires at least one outcome');
  }
  if (!Number.isInteger(win) || win < 0 || win >= k) {
    throw new RangeError('win = ' + String(win) + ' is outside [0, ' + String(k) + ')');
  }
  let total = 0n;
  for (const count of countsBig) total += count;
  const numerator = countsBig[win]! + (addOne ? 1n : 0n);
  const denominator = addOne ? total + BigInt(k) : total;
  if (denominator === 0n) {
    throw new RangeError('the MLE is undefined when N = 0; use an add-one prior');
  }
  return { countsBig, total, k, probability: exactFraction(numerator, denominator) };
}

/**
 * Kelly sizing on the add-one posterior p = (c_win + 1) / (N + K). The +1 floor
 * keeps the size finite and never lets a zero count collapse the probability.
 */
export function buleyeanKelly(
  counts: Counts,
  b: OddsInput,
  win = 0,
): BuleyeanKellyDecision {
  const odds = toFraction(b, 'b');
  const state = winPosterior(counts, win, true);
  const fraction = kellyCore(state.probability, odds);
  return {
    win,
    counts: state.countsBig,
    total: state.total,
    k: state.k,
    probability: state.probability,
    odds,
    fraction,
    value: fractionToNumber(fraction),
    recommendedFraction: isPositiveFraction(fraction) ? fraction : exactFraction(0n, 1n),
    shouldBet: isPositiveFraction(fraction),
    detail:
      'add-one p = ' + String(state.probability.numerator) + '/' + String(state.probability.denominator) +
      ', b = ' + String(odds.numerator) + '/' + String(odds.denominator) +
      ', f* = ' + String(fraction.numerator) + '/' + String(fraction.denominator),
  };
}

/** The raw MLE Kelly decision, kept to expose its small-sample failure. */
export function mleKelly(
  counts: Counts,
  b: OddsInput,
  win = 0,
): KellyDecision {
  const odds = toFraction(b, 'b');
  const state = winPosterior(counts, win, false);
  const fraction = kellyCore(state.probability, odds);
  return {
    probability: state.probability,
    odds,
    fraction,
    value: fractionToNumber(fraction),
    recommendedFraction: isPositiveFraction(fraction) ? fraction : exactFraction(0n, 1n),
    shouldBet: isPositiveFraction(fraction),
    detail:
      'MLE p = ' + String(state.probability.numerator) + '/' + String(state.probability.denominator) +
      ', f* = ' + String(fraction.numerator) + '/' + String(fraction.denominator),
  };
}

/** The conservative fractional-Kelly size from a proven lower bound. */
export interface FractionalKellyFromRange {
  /** The lower endpoint of the proven bet-fraction range. */
  readonly low: ExactFraction;
  readonly multiplier: ExactFraction;
  /** multiplier * low, signed. */
  readonly rawSize: ExactFraction;
  /** max(multiplier * low, 0): the recommended stake. */
  readonly size: ExactFraction;
  readonly value: number;
  readonly shouldBet: boolean;
  readonly detail: string;
}

/**
 * Size at the CONSERVATIVE lower endpoint of a proven range:
 *
 *   size = multiplier * low,
 *
 * with lowNum/lowDen the lower end of a proven bet-fraction (Kelly) range and
 * multiplier in [0, 1] the fractional-Kelly guard. Negative lower endpoints
 * clamp to a zero stake. Sizing at the upper endpoint instead is the adversarial
 * failure named by kellyAdversarialDual-style reasoning: it bets the optimistic
 * end of a range the arithmetic only proved as an interval.
 */
export function fractionalKellyFromRange(
  lowNum: bigint,
  lowDen: bigint,
  multiplier: RationalInput,
): FractionalKellyFromRange {
  const low = exactFraction(lowNum, lowDen);
  const mult = toFraction(multiplier, 'multiplier');
  if (mult.numerator < 0n || compareFractions(mult, exactFraction(1n, 1n)) > 0) {
    throw new RangeError('multiplier must lie in [0, 1]');
  }
  const rawSize = mulFractions(mult, low);
  const size = isPositiveFraction(rawSize) ? rawSize : exactFraction(0n, 1n);
  return {
    low,
    multiplier: mult,
    rawSize,
    size,
    value: fractionToNumber(size),
    shouldBet: isPositiveFraction(size),
    detail:
      'conservative lower bound ' + String(low.numerator) + '/' + String(low.denominator) +
      ' x multiplier ' + String(mult.numerator) + '/' + String(mult.denominator) +
      ' = ' + String(size.numerator) + '/' + String(size.denominator),
  };
}

/**
 * Robust Kelly from a probability lower bound: take the full Kelly fraction at
 * p = low, then scale by a fractional multiplier in [0, 1] and clamp at zero.
 */
export function robustKellyFromRange(
  lowNum: bigint,
  lowDen: bigint,
  bNum: bigint,
  bDen: bigint,
  multiplier: RationalInput = 1n,
): FractionalKellyFromRange {
  const low = exactFraction(lowNum, lowDen);
  const odds = exactFraction(bNum, bDen);
  const full = kellyCore(low, odds);
  return fractionalKellyFromRange(full.numerator, full.denominator, multiplier);
}

/** The proven posterior range mapped through Kelly. */
export interface KellyRangeReport {
  /** 1/(N + K): the floor endpoint of the proven posterior range. */
  readonly lowProbability: ExactFraction;
  /** (N + 1)/(N + K): the cap endpoint. */
  readonly highProbability: ExactFraction;
  readonly lowFraction: ExactFraction;
  readonly highFraction: ExactFraction;
  /** max(lowFraction, 0): the ambiguity-averse stake. */
  readonly conservativeFraction: ExactFraction;
  /** max(highFraction, 0): the optimistic stake. */
  readonly aggressiveFraction: ExactFraction;
  readonly overbetIfAggressive: boolean;
  readonly detail: string;
}

/**
 * Map the proven collapse range for one outcome,
 * [1/(N+K), (N+1)/(N+K)] with budget N, through Kelly at net odds b.
 */
export function kellyRangeFromPosterior(
  counts: Counts,
  b: OddsInput,
  win = 0,
): KellyRangeReport {
  const countsBig = coerceCounts(counts);
  const k = countsBig.length;
  if (k === 0) throw new RangeError('kellyRangeFromPosterior requires at least one outcome');
  if (!Number.isInteger(win) || win < 0 || win >= k) {
    throw new RangeError('win = ' + String(win) + ' is outside [0, ' + String(k) + ')');
  }
  let total = 0n;
  for (const count of countsBig) total += count;
  const denominator = total + BigInt(k);
  const odds = toFraction(b, 'b');
  const lowProbability = exactFraction(1n, denominator);
  const highProbability = exactFraction(total + 1n, denominator);
  const lowFraction = kellyCore(lowProbability, odds);
  const highFraction = kellyCore(highProbability, odds);
  const conservativeFraction = isPositiveFraction(lowFraction) ? lowFraction : exactFraction(0n, 1n);
  const aggressiveFraction = isPositiveFraction(highFraction) ? highFraction : exactFraction(0n, 1n);
  const overbetIfAggressive = compareFractions(aggressiveFraction, conservativeFraction) > 0;
  return {
    lowProbability,
    highProbability,
    lowFraction,
    highFraction,
    conservativeFraction,
    aggressiveFraction,
    overbetIfAggressive,
    detail:
      'proven p range [1/' + String(denominator) + ', ' + String(highProbability.numerator) +
      '/' + String(highProbability.denominator) + ']; conservative f = ' +
      String(conservativeFraction.numerator) + '/' + String(conservativeFraction.denominator) +
      ', aggressive f = ' + String(aggressiveFraction.numerator) + '/' + String(aggressiveFraction.denominator),
  };
}

/** The small-sample overconfidence dual: MLE overbets add-one. */
export interface KellyAdversarialDual {
  readonly fires: boolean;
  readonly win: number;
  /** null when the MLE is undefined at N = 0. */
  readonly mle: KellyDecision | null;
  readonly addOne: KellyDecision;
  /** mle.fraction / addOne.fraction when both are positive, else null. */
  readonly overbetRatio: ExactFraction | null;
  readonly mleOverbets: boolean;
  readonly detail: string;
}

/**
 * Concrete failure mode of the MLE on a small sample: it plugs in an
 * overconfident p and overbets relative to the add-one posterior. The canonical
 * case is one win in one trial with the second outcome unobserved.
 */
export function kellyAdversarialDual(
  counts: Counts,
  b: OddsInput,
  win = 0,
): KellyAdversarialDual {
  const odds = toFraction(b, 'b');
  const addOneState = winPosterior(counts, win, true);
  const addOneFraction = kellyCore(addOneState.probability, odds);
  const addOne: KellyDecision = {
    probability: addOneState.probability,
    odds,
    fraction: addOneFraction,
    value: fractionToNumber(addOneFraction),
    recommendedFraction: isPositiveFraction(addOneFraction) ? addOneFraction : exactFraction(0n, 1n),
    shouldBet: isPositiveFraction(addOneFraction),
    detail: 'add-one',
  };

  let total = 0n;
  for (const count of addOneState.countsBig) total += count;
  if (total === 0n) {
    return {
      fires: false,
      win,
      mle: null,
      addOne,
      overbetRatio: null,
      mleOverbets: false,
      detail: 'N = 0: the MLE is undefined, so the overbet dual cannot run',
    };
  }

  const mleState = winPosterior(counts, win, false);
  const mleFraction = kellyCore(mleState.probability, odds);
  const mle: KellyDecision = {
    probability: mleState.probability,
    odds,
    fraction: mleFraction,
    value: fractionToNumber(mleFraction),
    recommendedFraction: isPositiveFraction(mleFraction) ? mleFraction : exactFraction(0n, 1n),
    shouldBet: isPositiveFraction(mleFraction),
    detail: 'MLE',
  };

  const bothPositive = isPositiveFraction(mleFraction) && isPositiveFraction(addOneFraction);
  const overbetRatio = bothPositive ? divFractions(mleFraction, addOneFraction) : null;
  const mleOverbets = compareFractions(mleFraction, addOneFraction) > 0;
  return {
    fires: mleOverbets,
    win,
    mle,
    addOne,
    overbetRatio,
    mleOverbets,
    detail:
      'MLE f* = ' + String(mleFraction.numerator) + '/' + String(mleFraction.denominator) +
      ' vs add-one f* = ' + String(addOneFraction.numerator) + '/' + String(addOneFraction.denominator) +
      (mleOverbets
        ? '; the MLE overbets' + (overbetRatio ? ' by ' + String(overbetRatio.numerator) + '/' + String(overbetRatio.denominator) : '')
        : '; the MLE does not overbet here'),
  };
}
