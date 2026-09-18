/**
 * newsvendor.ts -- the critical fractile and a discrete order quantity.
 *
 * The newsvendor model sets an order quantity q against uncertain demand D with
 * underage cost cu (per unit short) and overage cost co (per unit left over).
 * The optimal service level is the critical fractile
 *
 *   gamma = cu / (cu + co),
 *
 * and q* is the gamma-quantile of the demand distribution. This module reads
 * that quantile off the add-one (Laplace/Buleyean) posterior over discrete
 * demand categories, ranking categories by demand and accumulating exact mass.
 * Every mass and every cost is an exact rational.
 *
 * Robust variant: robustNewsvendorOrder wraps the cumulative mass in the PROVEN
 * collapse range [r/W, (R+1) r/W] for the first r rungs, where R is the
 * observation budget and W = N + K. That band is sound but deliberately wide;
 * it is a worst-case guarantee, not a tight estimate, and the module says so.
 *
 * Adversarial dual: ordering the mean is suboptimal when cu != co.
 * newsvendorMeanDual computes the exact expected cost of the critical-fractile
 * order and of the nearest-to-mean order and reports when the mean order loses.
 *
 * Nothing here is financial advice and no profit is claimed. The costs and the
 * demand categories are the caller's model; the arithmetic is exact.
 */

import { coerceCounts, type Counts } from '../void-crdt.js';
import {
  addFractions,
  compareFractions,
  exactFraction,
  fractionToNumber,
  mulFractions,
  subFractions,
  toFraction,
  type ExactFraction,
  type RationalInput,
} from './rational.js';

/** One demand level per state: a safe integer or an exact bigint. */
export type DemandVector = readonly (number | bigint)[];

/** Coerce demand levels to exact BigInt, rejecting unsafe doubles. */
export function coerceDemands(demands: DemandVector, where = 'demands'): bigint[] {
  const out: bigint[] = [];
  for (let i = 0; i < demands.length; i++) {
    const demand = demands[i]!;
    if (typeof demand === 'bigint') {
      out.push(demand);
      continue;
    }
    if (!Number.isSafeInteger(demand)) {
      throw new RangeError(
        where + '[' + i + '] = ' + String(demand) + ' is not a safe integer; use a bigint for exact extreme demand',
      );
    }
    out.push(BigInt(demand));
  }
  return out;
}

/** The exact critical fractile cu / (cu + co). */
export function criticalFractile(
  cuNum: bigint,
  cuDen: bigint,
  coNum: bigint,
  coDen: bigint,
): ExactFraction {
  const cu = exactFraction(cuNum, cuDen);
  const co = exactFraction(coNum, coDen);
  if (cu.numerator < 0n || co.numerator < 0n) {
    throw new RangeError('underage and overage costs must be non-negative');
  }
  const denominator = addFractions(cu, co);
  if (denominator.numerator === 0n) {
    throw new RangeError('cu + co must be strictly positive');
  }
  return exactFraction(cu.numerator * denominator.denominator, denominator.numerator * cu.denominator);
}

interface Rung {
  readonly category: number;
  readonly demand: bigint;
  readonly count: bigint;
  readonly numerator: bigint;
}

function sortedRungs(counts: Counts, demands: DemandVector): {
  rungs: Rung[];
  total: bigint;
  denominator: bigint;
  k: number;
} {
  const countsBig = coerceCounts(counts);
  const k = countsBig.length;
  if (k === 0) {
    throw new RangeError('newsvendor requires at least one demand category');
  }
  const demandBig = coerceDemands(demands);
  if (demandBig.length !== k) {
    throw new RangeError(
      'demands length ' + demandBig.length + ' must equal counts length ' + k,
    );
  }
  let total = 0n;
  for (const count of countsBig) total += count;
  const rungs: Rung[] = [];
  for (let i = 0; i < k; i++) {
    rungs.push({ category: i, demand: demandBig[i]!, count: countsBig[i]!, numerator: countsBig[i]! + 1n });
  }
  rungs.sort((left, right) => {
    if (left.demand !== right.demand) return left.demand < right.demand ? -1 : 1;
    return left.category - right.category;
  });
  return { rungs, total, denominator: total + BigInt(k), k };
}

/** One rung of the accumulated discrete quantile. */
export interface NewsvendorRung {
  readonly rank: number;
  readonly category: number;
  readonly demand: bigint;
  readonly count: bigint;
  readonly numerator: bigint;
  readonly cumulativeNumerator: bigint;
  readonly cumulativeMass: ExactFraction;
}

/** The discrete order quantity at the critical fractile. */
export interface NewsvendorOrder {
  readonly criticalFractile: ExactFraction;
  readonly criticalValue: number;
  readonly total: bigint;
  readonly posteriorDenominator: bigint;
  readonly orderDemand: bigint;
  readonly orderDemandNumber: number;
  readonly achievedMass: ExactFraction;
  readonly achievedValue: number;
  readonly reached: boolean;
  readonly rungs: readonly NewsvendorRung[];
  readonly detail: string;
}

/**
 * The order quantity is the first demand level whose cumulative add-one
 * posterior mass reaches the critical fractile cu/(cu+co). Reaching is decided
 * by exact integer cross-multiplication, never a float.
 */
export function newsvendorOrder(
  counts: Counts,
  demands: DemandVector,
  cu: RationalInput,
  co: RationalInput,
): NewsvendorOrder {
  const underage = toFraction(cu, 'cu');
  const overage = toFraction(co, 'co');
  if (underage.numerator < 0n || overage.numerator < 0n) {
    throw new RangeError('underage and overage costs must be non-negative');
  }
  const gamma = criticalFractile(underage.numerator, underage.denominator, overage.numerator, overage.denominator);
  const { rungs, total, denominator } = sortedRungs(counts, demands);

  let cumulative = 0n;
  let chosen: Rung | null = null;
  let chosenRank = -1;
  const accumulated: NewsvendorRung[] = [];
  for (let rank = 0; rank < rungs.length; rank++) {
    const rung = rungs[rank]!;
    cumulative += rung.numerator;
    accumulated.push({
      rank,
      category: rung.category,
      demand: rung.demand,
      count: rung.count,
      numerator: rung.numerator,
      cumulativeNumerator: cumulative,
      cumulativeMass: exactFraction(cumulative, denominator),
    });
    // cumulative/denominator >= gamma.numerator/gamma.denominator
    if (chosen === null && cumulative * gamma.denominator >= gamma.numerator * denominator) {
      chosen = rung;
      chosenRank = rank;
    }
  }

  const last = rungs[rungs.length - 1]!;
  const chosenRung = chosen ?? last;
  const achievedNumerator = accumulated[(chosen !== null ? chosenRank : accumulated.length - 1)]!.cumulativeNumerator;
  const achieved = exactFraction(achievedNumerator, denominator);
  return {
    criticalFractile: gamma,
    criticalValue: fractionToNumber(gamma),
    total,
    posteriorDenominator: denominator,
    orderDemand: chosenRung.demand,
    orderDemandNumber: fractionToNumber(exactFraction(chosenRung.demand, 1n)),
    achievedMass: achieved,
    achievedValue: fractionToNumber(achieved),
    reached: chosen !== null,
    rungs: accumulated,
    detail:
      'gamma = ' + String(gamma.numerator) + '/' + String(gamma.denominator) +
      '; order ' + String(chosenRung.demand) +
      (chosen !== null ? ' at cumulative mass ' + String(achieved.numerator) + '/' + String(achieved.denominator) : ' (unreached; returning the largest demand)'),
  };
}

/** The expected newsvendor cost of one order quantity, exact. */
export interface NewsvendorCost {
  readonly order: bigint;
  readonly cost: ExactFraction;
  readonly value: number;
  readonly underageCost: ExactFraction;
  readonly overageCost: ExactFraction;
  readonly detail: string;
}

/**
 * Exact expected cost of ordering q:
 *
 *   co * E[(q - D)+] + cu * E[(D - q)+]
 *
 * over the add-one posterior on the discrete demand categories.
 */
export function newsvendorCost(
  counts: Counts,
  demands: DemandVector,
  cu: RationalInput,
  co: RationalInput,
  order: number | bigint,
): NewsvendorCost {
  const underage = toFraction(cu, 'cu');
  const overage = toFraction(co, 'co');
  const q = coerceDemands([order], 'order')[0]!;
  const { rungs, denominator } = sortedRungs(counts, demands);

  let underageTotal = exactFraction(0n, 1n);
  let overageTotal = exactFraction(0n, 1n);
  for (const rung of rungs) {
    const probability = exactFraction(rung.numerator, denominator);
    if (rung.demand > q) {
      const short = rung.demand - q;
      underageTotal = addFractions(underageTotal, mulFractions(probability, mulFractions(underage, short)));
    } else if (q > rung.demand) {
      const excess = q - rung.demand;
      overageTotal = addFractions(overageTotal, mulFractions(probability, mulFractions(overage, excess)));
    }
  }
  const cost = addFractions(underageTotal, overageTotal);
  return {
    order: q,
    cost,
    value: fractionToNumber(cost),
    underageCost: underageTotal,
    overageCost: overageTotal,
    detail:
      'order ' + String(q) + ': cu E[short] = ' + String(underageTotal.numerator) + '/' + String(underageTotal.denominator) +
      ' + co E[excess] = ' + String(overageTotal.numerator) + '/' + String(overageTotal.denominator) +
      ' = ' + String(cost.numerator) + '/' + String(cost.denominator),
  };
}

/** The exact posterior mean demand. */
export function posteriorMeanDemand(counts: Counts, demands: DemandVector): ExactFraction {
  const { rungs, denominator } = sortedRungs(counts, demands);
  let mean = exactFraction(0n, 1n);
  for (const rung of rungs) {
    mean = addFractions(mean, mulFractions(exactFraction(rung.numerator, denominator), rung.demand));
  }
  return mean;
}

/** The demand level nearest the posterior mean, ties resolved to the lower. */
export function meanOrderDemand(counts: Counts, demands: DemandVector): bigint {
  const mean = posteriorMeanDemand(counts, demands);
  const demandBig = coerceDemands(demands);
  const levels = [...new Set(demandBig.map((level) => level.toString()))]
    .map((text) => BigInt(text))
    .sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
  let best = levels[0]!;
  let bestDistance: ExactFraction | null = null;
  for (const level of levels) {
    const distance = subFractions(exactFraction(level, 1n), mean);
    const absolute: ExactFraction = distance.numerator < 0n
      ? { numerator: -distance.numerator, denominator: distance.denominator }
      : distance;
    if (bestDistance === null || compareFractions(absolute, bestDistance) < 0) {
      best = level;
      bestDistance = absolute;
    }
  }
  return best;
}

/** The adversarial dual: the nearest-to-mean order can cost more than q*. */
export interface NewsvendorMeanDual {
  readonly fires: boolean;
  readonly criticalFractile: ExactFraction;
  readonly criticalOrder: bigint;
  readonly meanOrder: bigint;
  readonly meanDemand: ExactFraction;
  readonly criticalCost: ExactFraction;
  readonly meanCost: ExactFraction;
  readonly costPenalty: ExactFraction;
  readonly meanIsSuboptimal: boolean;
  readonly costsDiffer: boolean;
  readonly detail: string;
}

/**
 * Compute the exact expected cost of the critical-fractile order and of the
 * nearest-to-mean order. When cu != co and the mean order loses, the dual
 * fires: ordering the mean is not the same as ordering the quantile.
 */
export function newsvendorMeanDual(
  counts: Counts,
  demands: DemandVector,
  cu: RationalInput,
  co: RationalInput,
): NewsvendorMeanDual {
  const order = newsvendorOrder(counts, demands, cu, co);
  const meanOrder = meanOrderDemand(counts, demands);
  const meanDemand = posteriorMeanDemand(counts, demands);
  const criticalCost = newsvendorCost(counts, demands, cu, co, order.orderDemand).cost;
  const meanCost = newsvendorCost(counts, demands, cu, co, meanOrder).cost;
  const comparison = compareFractions(criticalCost, meanCost);
  const costsDiffer = comparison !== 0;
  const meanIsSuboptimal = comparison < 0;
  const cuF = toFraction(cu, 'cu');
  const coF = toFraction(co, 'co');
  const cuEqualsCo = compareFractions(cuF, coF) === 0;
  return {
    fires: !cuEqualsCo && meanIsSuboptimal,
    criticalFractile: order.criticalFractile,
    criticalOrder: order.orderDemand,
    meanOrder,
    meanDemand,
    criticalCost,
    meanCost,
    costPenalty: meanIsSuboptimal ? subFractions(meanCost, criticalCost) : exactFraction(0n, 1n),
    meanIsSuboptimal,
    costsDiffer,
    detail:
      'cu ' + (cuEqualsCo ? '=' : '!=') + ' co; q* = ' + String(order.orderDemand) +
      ' (cost ' + String(criticalCost.numerator) + '/' + String(criticalCost.denominator) + '), ' +
      'mean order ' + String(meanOrder) + ' (cost ' + String(meanCost.numerator) + '/' + String(meanCost.denominator) + '); ' +
      (meanIsSuboptimal ? 'the mean order is suboptimal' : 'the mean order is not worse here'),
  };
}

/** One endpoint of the robust (sound-band) order range. */
export interface RobustOrderEndpoint {
  /** True when the endpoint reached the critical fractile even at its band edge. */
  readonly reached: boolean;
  readonly orderDemand: bigint;
  readonly bandMass: ExactFraction;
  readonly bandValue: number;
  readonly rank: number;
}

/** The robust order range built from the proven collapse band. */
export interface RobustNewsvendorOrder {
  readonly criticalFractile: ExactFraction;
  readonly budget: bigint;
  readonly total: bigint;
  readonly posteriorDenominator: bigint;
  readonly point: bigint;
  /** Ambiguity-averse: smallest q whose LOWER band edge reaches gamma. */
  readonly guaranteed: RobustOrderEndpoint;
  /** Best-case: smallest q whose UPPER band edge reaches gamma. */
  readonly optimistic: RobustOrderEndpoint;
  /** True when point lies in [optimistic, guaranteed]. */
  readonly sound: boolean;
  readonly detail: string;
}

/**
 * Robust order quantity from the proven collapse band. For the first r rungs the
 * cumulative weight lies in [r, (R + 1) r] with R the observation budget, so the
 * cumulative mass lies in [r/W, (R + 1) r/W], W = N + K. The guaranteed endpoint
 * is the smallest demand whose lower edge reaches gamma; the optimistic endpoint
 * uses the upper edge. The band is sound but wide by construction: it must hold
 * for any admissible weighting, so it is a worst-case cover, not an estimate.
 */
export function robustNewsvendorOrder(
  counts: Counts,
  demands: DemandVector,
  cu: RationalInput,
  co: RationalInput,
  budget?: number | bigint,
): RobustNewsvendorOrder {
  const underage = toFraction(cu, 'cu');
  const overage = toFraction(co, 'co');
  const gamma = criticalFractile(underage.numerator, underage.denominator, overage.numerator, overage.denominator);
  const { rungs, total, denominator } = sortedRungs(counts, demands);

  let maxCount = 0n;
  for (const rung of rungs) if (rung.count > maxCount) maxCount = rung.count;
  const R = budget === undefined ? maxCount : coerceDemands([budget], 'budget')[0]!;
  if (R < 0n) throw new RangeError('budget must be non-negative');

  const point = newsvendorOrder(counts, demands, cu, co).orderDemand;

  let guaranteed: RobustOrderEndpoint | null = null;
  let optimistic: RobustOrderEndpoint | null = null;
  for (let rank = 0; rank < rungs.length; rank++) {
    const rung = rungs[rank]!;
    const width = BigInt(rank + 1);
    // lower edge r/W reaches gamma: r * gamma.denominator >= gamma.numerator * W
    if (guaranteed === null && width * gamma.denominator >= gamma.numerator * denominator) {
      guaranteed = {
        reached: true,
        orderDemand: rung.demand,
        bandMass: exactFraction(width, denominator),
        bandValue: fractionToNumber(exactFraction(width, denominator)),
        rank,
      };
    }
    // upper edge (R+1) r / W reaches gamma
    if (optimistic === null && (R + 1n) * width * gamma.denominator >= gamma.numerator * denominator) {
      optimistic = {
        reached: true,
        orderDemand: rung.demand,
        bandMass: exactFraction((R + 1n) * width, denominator),
        bandValue: fractionToNumber(exactFraction((R + 1n) * width, denominator)),
        rank,
      };
    }
  }

  const last = rungs[rungs.length - 1]!;
  const guaranteedEndpoint: RobustOrderEndpoint = guaranteed ?? {
    reached: false,
    orderDemand: last.demand,
    bandMass: exactFraction(BigInt(rungs.length), denominator),
    bandValue: fractionToNumber(exactFraction(BigInt(rungs.length), denominator)),
    rank: rungs.length - 1,
  };
  const optimisticEndpoint: RobustOrderEndpoint = optimistic ?? {
    reached: false,
    orderDemand: last.demand,
    bandMass: exactFraction((R + 1n) * BigInt(rungs.length), denominator),
    bandValue: fractionToNumber(exactFraction((R + 1n) * BigInt(rungs.length), denominator)),
    rank: rungs.length - 1,
  };

  const lowDemand = optimisticEndpoint.orderDemand < guaranteedEndpoint.orderDemand
    ? optimisticEndpoint.orderDemand
    : guaranteedEndpoint.orderDemand;
  const highDemand = optimisticEndpoint.orderDemand < guaranteedEndpoint.orderDemand
    ? guaranteedEndpoint.orderDemand
    : optimisticEndpoint.orderDemand;
  const sound = point >= lowDemand && point <= highDemand;

  return {
    criticalFractile: gamma,
    budget: R,
    total,
    posteriorDenominator: denominator,
    point,
    guaranteed: guaranteedEndpoint,
    optimistic: optimisticEndpoint,
    sound,
    detail:
      'sound band order in [' + String(lowDemand) + ', ' + String(highDemand) +
      '], point ' + String(point) + '; ' + (sound ? 'point inside the sound range' : 'point outside the sound range'),
  };
}
