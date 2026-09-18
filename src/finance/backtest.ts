/**
 * backtest.ts -- a deterministic backtest harness for the bet-sizing primitives.
 *
 * This module attacks the FINANCE.md falsification item 3: "if add-one Kelly
 * underperforms on a backtest because the prior is too conservative, the
 * regularization costs more than it saves." It builds a reproducible synthetic
 * market, runs five sizing rules over many seeded paths, and reports both
 * directions of the adversarial dual instead of declaring one universal winner.
 *
 * Design:
 *
 *  - The market is a biased binary bet: win with true probability p* and get
 *    net odds b, lose the stake otherwise. A regime-switching variant changes
 *    p* partway through the run, so one global estimate is wrong for part of it.
 *    Outcomes come from a seeded xorshift32 stream and are decided by an exact
 *    integer comparison (draw * pDen < pNum * 2^32), so the generator is
 *    deterministic and free of float-branch wobble.
 *  - Sizing decisions are exact rationals from kelly.ts. Wealth, log-wealth and
 *    drawdown are display floats; no truth claim is made on those floats.
 *  - Five strategies: (a) MLE Kelly on wins/trials, (b) add-one Kelly on the
 *    Buleyean posterior (wins+1)/(trials+2), (c) conservative range Kelly at
 *    the PROVEN lower endpoint from kelly.ts, (d) a fixed fraction, and (e) the
 *    no-bet baseline.
 *  - Metrics per strategy: mean and median terminal log-wealth, ruin
 *    probability (wealth ever at or below a floor), mean and median maximum
 *    drawdown, and the fraction of paths that ever overbet the true-Kelly
 *    fraction for the regime in force.
 *
 * Honest dual: a large sample with a small true edge shows add-one UNDERBETTING
 * the true-Kelly fraction (a real but second-order growth cost), while a small
 * sample with an overconfident MLE shows add-one SAVING CAPITAL (the MLE goes
 * all-in after an early all-win record and ruins). Neither rule dominates
 * everywhere; backtestAdversarialDual reports both and says so.
 *
 * Nothing here is financial advice and no profitability is claimed. A backtest
 * over a synthetic generator is an arithmetic demonstration about a stated
 * model, not evidence about any real market. The generator's p* and b are the
 * caller's assumptions; the prior's cost and the MLE's overconfidence are
 * properties of those assumptions, not forecasts.
 */

import {
  addFractions,
  compareFractions,
  exactFraction,
  fractionToNumber,
  fractionToString,
  isPositiveFraction,
  mulFractions,
  subFractions,
  toFraction,
  type ExactFraction,
  type RationalInput,
} from './rational.js';
import {
  buleyeanKelly,
  kellyFraction,
  kellyRangeFromPosterior,
  mleKelly,
} from './kelly.js';

/** 2^32 as an exact integer, the resolution of the xorshift32 stream. */
const TWO_POW_32 = 1n << 32n;

// ---------------------------------------------------------------------------
// Deterministic market generator
// ---------------------------------------------------------------------------

/**
 * splitmix32 finalizer. Raw xorshift32 maps a small seed s to a first output
 * proportional to s, so the seeds 1, 2, 3, ... produce linearly correlated
 * streams (seed 1 starts the market with the same first draw pattern as every
 * other small seed). Mixing the seed first removes that structure while keeping
 * the stream a pure function of the seed.
 */
function mixSeed32(seed: number): number {
  let z = seed >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x7feb352d) >>> 0;
  z = Math.imul(z ^ (z >>> 15), 0x846ca68b) >>> 0;
  z = (z ^ (z >>> 16)) >>> 0;
  return z;
}

/**
 * A deterministic xorshift32 generator. The seed is passed through a splitmix32
 * finalizer first, then the raw uint32 state (0 .. 2^32 - 1) is returned on
 * each call. The all-zero state is a fixed point, so seed 0 is rejected rather
 * than silently producing an all-loss stream.
 */
export function xorshift32(seed: number): () => number {
  const normalized = seed >>> 0;
  if (normalized === 0) {
    throw new RangeError('xorshift32 requires a non-zero seed (0 is a fixed point)');
  }
  const mixed = mixSeed32(normalized);
  let state = (mixed === 0 ? 0x9e3779b9 : mixed) >>> 0;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state;
  };
}

/** One constant-p* stretch of a market, starting at fromTrial (inclusive). */
export interface RegimeStep {
  readonly fromTrial: number;
  readonly pWin: ExactFraction;
}

/** A binary bet market: net odds b, trials rounds, a true p* per regime, a ruin floor. */
export interface MarketSpec {
  readonly payout: ExactFraction;
  readonly trials: number;
  /** Regimes ordered by fromTrial; the first MUST start at trial 0. */
  readonly regimes: readonly RegimeStep[];
  /** A path is ruined when wealth reaches this floor; the floor defaults to 0.01. */
  readonly floorWealth: number;
}

function validateMarket(spec: MarketSpec): void {
  if (!Number.isInteger(spec.trials) || spec.trials <= 0) {
    throw new RangeError('trials must be a positive integer');
  }
  if (spec.payout.numerator <= 0n) {
    throw new RangeError('payout b must be strictly positive');
  }
  if (typeof spec.floorWealth !== 'number' || !(spec.floorWealth > 0)) {
    throw new RangeError('floorWealth must be a positive number');
  }
  if (spec.regimes.length === 0) {
    throw new RangeError('at least one regime is required');
  }
  if (spec.regimes[0]!.fromTrial !== 0) {
    throw new RangeError('the first regime must start at trial 0');
  }
  let previous = -1;
  for (const regime of spec.regimes) {
    if (!Number.isInteger(regime.fromTrial) || regime.fromTrial < 0 || regime.fromTrial >= spec.trials) {
      throw new RangeError('regime fromTrial must lie in [0, trials)');
    }
    if (regime.fromTrial <= previous) {
      throw new RangeError('regime fromTrial values must be strictly increasing');
    }
    previous = regime.fromTrial;
    if (regime.pWin.numerator < 0n || compareFractions(regime.pWin, 1n) > 0) {
      throw new RangeError('regime pWin must lie in [0, 1]');
    }
  }
}

/** One constant-p* market. p* is an exact rational; the floor defaults to 0.01. */
export function staticMarket(
  pWin: RationalInput,
  payout: RationalInput,
  trials: number,
  floorWealth = 0.01,
): MarketSpec {
  const spec: MarketSpec = {
    payout: toFraction(payout, 'payout'),
    trials,
    regimes: [{ fromTrial: 0, pWin: toFraction(pWin, 'pWin') }],
    floorWealth,
  };
  validateMarket(spec);
  return spec;
}

/**
 * A two-regime market. pBefore holds on [0, switchAt) and pAfter on
 * [switchAt, trials), so a single global estimate is wrong for the second half.
 */
export function switchingMarket(
  pBefore: RationalInput,
  pAfter: RationalInput,
  switchAt: number,
  payout: RationalInput,
  trials: number,
  floorWealth = 0.01,
): MarketSpec {
  const spec: MarketSpec = {
    payout: toFraction(payout, 'payout'),
    trials,
    regimes: [
      { fromTrial: 0, pWin: toFraction(pBefore, 'pBefore') },
      { fromTrial: switchAt, pWin: toFraction(pAfter, 'pAfter') },
    ],
    floorWealth,
  };
  validateMarket(spec);
  return spec;
}

/**
 * The true win probability in force at a trial, from the regime whose start is
 * the latest at or before that trial. Assumes a validated MarketSpec; the hot
 * simulation loop calls this once per round, so validation is not repeated here.
 */
export function trueProbabilityAt(spec: MarketSpec, trial: number): ExactFraction {
  if (!Number.isInteger(trial) || trial < 0 || trial >= spec.trials) {
    throw new RangeError('trial ' + String(trial) + ' must lie in [0, ' + String(spec.trials) + ')');
  }
  let found = spec.regimes[0]!;
  for (const regime of spec.regimes) {
    if (regime.fromTrial <= trial) found = regime;
  }
  return found.pWin;
}

/** Exact win/loss for one draw: win iff draw / 2^32 < p. */
function drawWin(next: () => number, p: ExactFraction): boolean {
  const draw = next();
  return BigInt(draw) * p.denominator < p.numerator * TWO_POW_32;
}

/** The full outcome sequence for one seed (true = win). Regime-aware. */
export function generateOutcomes(spec: MarketSpec, seed: number): boolean[] {
  validateMarket(spec);
  const next = xorshift32(seed);
  const outcomes: boolean[] = [];
  for (let trial = 0; trial < spec.trials; trial++) {
    outcomes.push(drawWin(next, trueProbabilityAt(spec, trial)));
  }
  return outcomes;
}

// ---------------------------------------------------------------------------
// Exact sizing helpers (the decisions the strategies are judged on)
// ---------------------------------------------------------------------------

/** The exact full-Kelly fraction (b p - q)/b for a stated p* and net odds b. */
export function trueKellyFraction(pWin: RationalInput, payout: RationalInput): ExactFraction {
  const p = toFraction(pWin, 'pWin');
  const b = toFraction(payout, 'payout');
  return kellyFraction(p.numerator, p.denominator, b.numerator, b.denominator);
}

/** max(f*, 0): the true growth-optimal stake for a stated p* and b. */
export function recommendedTrueKelly(pWin: RationalInput, payout: RationalInput): ExactFraction {
  const full = trueKellyFraction(pWin, payout);
  return isPositiveFraction(full) ? full : exactFraction(0n, 1n);
}

/**
 * Expected log-growth per trial for a stated true p*, net odds b, and a fixed
 * stake fraction f:
 *
 *   g = p ln(1 + f b) + (1 - p) ln(1 - f).
 *
 * The stake factor is assembled exactly (1 + f b and 1 - f are reduced
 * rationals); only the logarithms are display floats. g is returned as
 * -Infinity when a stake would bankrupt the account on a loss (f >= 1).
 */
export function expectedLogGrowth(
  pWin: RationalInput,
  payout: RationalInput,
  fraction: RationalInput,
): number {
  const p = toFraction(pWin, 'pWin');
  const b = toFraction(payout, 'payout');
  const f = toFraction(fraction, 'fraction');
  const winFactor = addFractions(1n, mulFractions(f, b));
  const lossFactor = subFractions(1n, f);
  if (winFactor.numerator <= 0n || lossFactor.numerator <= 0n) {
    return Number.NEGATIVE_INFINITY;
  }
  const pn = fractionToNumber(p);
  return pn * Math.log(fractionToNumber(winFactor)) + (1 - pn) * Math.log(fractionToNumber(lossFactor));
}

/**
 * The growth-optimal per-trial rate at a stated p* and b: g evaluated at the
 * (clamped) true-Kelly fraction. This is the upper envelope all sizing rules
 * are compared against on a known p*.
 */
export function kellyGrowthRate(pWin: RationalInput, payout: RationalInput): number {
  const stake = recommendedTrueKelly(pWin, payout);
  if (!isPositiveFraction(stake)) return 0;
  return expectedLogGrowth(pWin, payout, stake);
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/** The observed history a strategy may condition on before the next trial. */
export interface BetContext {
  readonly wins: number;
  readonly losses: number;
  readonly trials: number;
  readonly payout: ExactFraction;
}

/** A sizing rule: an exact stake fraction in [0, 1] as a function of history. */
export interface BetStrategy {
  readonly id: string;
  readonly label: string;
  readonly decision: (context: BetContext) => ExactFraction;
}

/** (a) MLE Kelly: plug in wins/trials; stake 0 when the MLE is undefined (N = 0). */
export function mleKellyStrategy(): BetStrategy {
  return {
    id: 'mle-kelly',
    label: 'MLE Kelly (wins/trials)',
    decision: ({ wins, losses, trials, payout }) => {
      if (trials === 0) return exactFraction(0n, 1n);
      return mleKelly([wins, losses], payout, 0).recommendedFraction;
    },
  };
}

/** (b) Add-one Kelly: the Buleyean posterior (wins+1)/(trials+2). */
export function addOneKellyStrategy(): BetStrategy {
  return {
    id: 'add-one-kelly',
    label: 'Add-one Kelly ((wins+1)/(trials+2))',
    decision: ({ wins, losses, payout }) => buleyeanKelly([wins, losses], payout, 0).recommendedFraction,
  };
}

/**
 * (c) Conservative range Kelly: size at the PROVEN lower endpoint of the
 * collapse range from kelly.ts, then scale by a multiplier in [0, 1]. The lower
 * endpoint is 1/(N + K) mapped through Kelly, so this refuses any bet whose edge
 * the proven range does not cover. With a multiplier of 1 it equals
 * robustKellyFromRange(1, N + 2, b, 1).
 */
export function conservativeRangeKellyStrategy(multiplier: RationalInput = 1n): BetStrategy {
  const mult = toFraction(multiplier, 'multiplier');
  if (mult.numerator < 0n || compareFractions(mult, 1n) > 0) {
    throw new RangeError('multiplier must lie in [0, 1]');
  }
  return {
    id: 'conservative-range-kelly',
    label: 'Conservative range Kelly (proven lower endpoint)',
    decision: ({ wins, losses, payout }) => {
      const range = kellyRangeFromPosterior([wins, losses], payout, 0);
      const scaled = mulFractions(mult, range.conservativeFraction);
      return isPositiveFraction(scaled) ? scaled : exactFraction(0n, 1n);
    },
  };
}

/** (d) A constant stake fraction. */
export function fixedFractionStrategy(fraction: RationalInput): BetStrategy {
  const f = toFraction(fraction, 'fraction');
  if (f.numerator < 0n || compareFractions(f, 1n) > 0) {
    throw new RangeError('fixed fraction must lie in [0, 1]');
  }
  return {
    id: 'fixed-' + fractionToString(f),
    label: 'Fixed fraction ' + fractionToString(f),
    decision: () => f,
  };
}

/** (e) The no-bet baseline: stake 0 forever, so log-wealth stays 0. */
export function noBetStrategy(): BetStrategy {
  return {
    id: 'no-bet',
    label: 'No bet (baseline)',
    decision: () => exactFraction(0n, 1n),
  };
}

/** An oracle that peeks at the true p* of the regime in force. Reference only. */
export function oracleKellyStrategy(spec: MarketSpec): BetStrategy {
  validateMarket(spec);
  return {
    id: 'oracle-kelly',
    label: 'Oracle true Kelly (reference; peeks at p*)',
    decision: ({ trials }) => {
      const trial = Math.min(trials, spec.trials - 1);
      return recommendedTrueKelly(trueProbabilityAt(spec, trial), spec.payout);
    },
  };
}

/** The five required strategies, in the (a)..(e) order of the task. */
export function defaultStrategies(
  fixedFraction: RationalInput = { numerator: 1n, denominator: 4n },
): readonly BetStrategy[] {
  return [
    mleKellyStrategy(),
    addOneKellyStrategy(),
    conservativeRangeKellyStrategy(),
    fixedFractionStrategy(fixedFraction),
    noBetStrategy(),
  ];
}

// ---------------------------------------------------------------------------
// Bankroll simulation
// ---------------------------------------------------------------------------

function clampStake(fraction: ExactFraction): ExactFraction {
  if (fraction.numerator <= 0n) return exactFraction(0n, 1n);
  if (compareFractions(fraction, 1n) > 0) return exactFraction(1n, 1n);
  return fraction;
}

/** One simulated bankroll path. All wealth numbers are display floats. */
export interface PathResult {
  readonly seed: number;
  readonly strategyId: string;
  readonly terminalWealth: number;
  readonly terminalLogWealth: number;
  readonly ruined: boolean;
  readonly maxDrawdown: number;
  readonly overbetEver: boolean;
  readonly overbetSteps: number;
  readonly trialsSettled: number;
}

function settlePath(
  spec: MarketSpec,
  strategy: BetStrategy,
  outcomes: readonly boolean[],
  seed: number,
  initialWealth: number,
): PathResult {
  if (!(initialWealth > 0) || !Number.isFinite(initialWealth)) {
    throw new RangeError('initialWealth must be finite and > 0');
  }
  if (!(spec.floorWealth < initialWealth)) {
    throw new RangeError('floorWealth must be below initialWealth');
  }
  let wealth = initialWealth;
  let peak = wealth;
  let maxDrawdown = 0;
  let wins = 0;
  let losses = 0;
  let ruined = false;
  let overbetEver = false;
  let overbetSteps = 0;
  let settled = 0;

  const rounds = Math.min(outcomes.length, spec.trials);
  for (let trial = 0; trial < rounds; trial++) {
    const p = trueProbabilityAt(spec, trial);
    const stake = clampStake(
      strategy.decision({ wins, losses, trials: wins + losses, payout: spec.payout }),
    );
    const trueStake = recommendedTrueKelly(p, spec.payout);
    if (compareFractions(stake, trueStake) > 0) {
      overbetEver = true;
      overbetSteps++;
    }

    const won = outcomes[trial]!;
    if (won) {
      wealth *= fractionToNumber(addFractions(1n, mulFractions(stake, spec.payout)));
      wins++;
    } else {
      wealth *= fractionToNumber(subFractions(1n, stake));
      losses++;
    }
    settled++;

    if (wealth <= spec.floorWealth) {
      wealth = spec.floorWealth;
      ruined = true;
      const drawdown = peak > 0 ? (peak - wealth) / peak : 0;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
      break;
    }
    if (wealth > peak) peak = wealth;
    const drawdown = peak > 0 ? (peak - wealth) / peak : 0;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  return {
    seed,
    strategyId: strategy.id,
    terminalWealth: wealth,
    terminalLogWealth: wealth > 0 ? Math.log(wealth) : Number.NEGATIVE_INFINITY,
    ruined,
    maxDrawdown,
    overbetEver,
    overbetSteps,
    trialsSettled: settled,
  };
}

/** Simulate one seeded path: outcomes from xorshift32, then settle. */
export function runPath(
  spec: MarketSpec,
  strategy: BetStrategy,
  seed: number,
  initialWealth = 1,
): PathResult {
  validateMarket(spec);
  return settlePath(spec, strategy, generateOutcomes(spec, seed), seed, initialWealth);
}

/**
 * Settle a caller-supplied deterministic outcome sequence (no RNG). Useful for
 * exact, hand-built histories; the seed is carried for identification only.
 */
export function runOutcomeSequence(
  spec: MarketSpec,
  strategy: BetStrategy,
  outcomes: readonly boolean[],
  seed = 1,
  initialWealth = 1,
): PathResult {
  validateMarket(spec);
  if (outcomes.length > spec.trials) {
    throw new RangeError('outcome sequence is longer than the market horizon');
  }
  return settlePath(spec, strategy, outcomes, seed, initialWealth);
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/** Aggregated metrics for one strategy over a set of paths. */
export interface StrategyReport {
  readonly strategyId: string;
  readonly strategyLabel: string;
  readonly paths: number;
  readonly meanTerminalLogWealth: number;
  readonly medianTerminalLogWealth: number;
  readonly ruinProbability: number;
  readonly overbetPathFraction: number;
  /** Mean number of rounds per path whose stake exceeded the true-Kelly stake. */
  readonly meanOverbetSteps: number;
  readonly meanMaxDrawdown: number;
  readonly medianMaxDrawdown: number;
}

/** The whole backtest: market summary plus one StrategyReport per strategy. */
export interface BacktestReport {
  readonly trials: number;
  readonly seeds: number;
  readonly payout: ExactFraction;
  readonly regimes: readonly RegimeStep[];
  readonly initialWealth: number;
  readonly floorWealth: number;
  readonly trueKellyAtStart: ExactFraction;
  readonly reports: readonly StrategyReport[];
}

function meanOf(values: readonly number[]): number {
  if (values.length === 0) throw new RangeError('mean of an empty sample');
  let sum = 0;
  for (const value of values) sum += value;
  return sum / values.length;
}

function medianOf(values: readonly number[]): number {
  if (values.length === 0) throw new RangeError('median of an empty sample');
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : 0.5 * (sorted[middle - 1]! + sorted[middle]!);
}

/** Reduce a strategy's paths to the per-strategy metrics. */
export function summarizeStrategy(strategy: BetStrategy, paths: readonly PathResult[]): StrategyReport {
  if (paths.length === 0) throw new RangeError('at least one path is required');
  const logs = paths.map((path) => path.terminalLogWealth);
  const drawdowns = paths.map((path) => path.maxDrawdown);
  const overbetSteps = paths.map((path) => path.overbetSteps);
  let ruinCount = 0;
  let overbetCount = 0;
  for (const path of paths) {
    if (path.ruined) ruinCount++;
    if (path.overbetEver) overbetCount++;
  }
  return {
    strategyId: strategy.id,
    strategyLabel: strategy.label,
    paths: paths.length,
    meanTerminalLogWealth: meanOf(logs),
    medianTerminalLogWealth: medianOf(logs),
    ruinProbability: ruinCount / paths.length,
    overbetPathFraction: overbetCount / paths.length,
    meanOverbetSteps: meanOf(overbetSteps),
    meanMaxDrawdown: meanOf(drawdowns),
    medianMaxDrawdown: medianOf(drawdowns),
  };
}

/** A deterministic seed list starting at seed0 (never includes the xorshift 0). */
export function seedRange(count: number, seed0 = 1): number[] {
  if (!Number.isInteger(count) || count <= 0) {
    throw new RangeError('count must be a positive integer');
  }
  const seeds: number[] = [];
  for (let i = 0; i < count; i++) {
    let seed = (seed0 + i) >>> 0;
    if (seed === 0) seed = 1;
    seeds.push(seed);
  }
  return seeds;
}

/** Run every strategy over every seed and aggregate. Same seed => same paths. */
export function runBacktest(
  spec: MarketSpec,
  strategies: readonly BetStrategy[],
  seeds: readonly number[],
  initialWealth = 1,
): BacktestReport {
  validateMarket(spec);
  if (strategies.length === 0) throw new RangeError('at least one strategy is required');
  if (seeds.length === 0) throw new RangeError('at least one seed is required');
  const reports: StrategyReport[] = [];
  for (const strategy of strategies) {
    const paths: PathResult[] = [];
    for (const seed of seeds) {
      paths.push(runPath(spec, strategy, seed, initialWealth));
    }
    reports.push(summarizeStrategy(strategy, paths));
  }
  return {
    trials: spec.trials,
    seeds: seeds.length,
    payout: spec.payout,
    regimes: spec.regimes,
    initialWealth,
    floorWealth: spec.floorWealth,
    trueKellyAtStart: trueKellyFraction(trueProbabilityAt(spec, 0), spec.payout),
    reports,
  };
}

/** Rebuild the strategy descriptor for a report id, for tests and callers. */
export function strategyById(id: string, strategies: readonly BetStrategy[]): BetStrategy {
  for (const strategy of strategies) {
    if (strategy.id === id) return strategy;
  }
  throw new RangeError('no strategy with id ' + id);
}

// ---------------------------------------------------------------------------
// Adversarial dual: undershoot growth vs. save capital
// ---------------------------------------------------------------------------

/** The large-sample, small-edge case where add-one underbets true Kelly. */
export interface GrowthUndershootReport {
  readonly kind: 'undershoot';
  readonly pStar: ExactFraction;
  readonly payout: ExactFraction;
  readonly trials: number;
  readonly wins: number;
  readonly mleFraction: ExactFraction;
  readonly addOneFraction: ExactFraction;
  readonly trueKellyFraction: ExactFraction;
  readonly mleExpectedLogGrowth: number;
  readonly addOneExpectedLogGrowth: number;
  /** mleGrowth - addOneGrowth; strictly positive means add-one undershoots. */
  readonly growthGap: number;
  readonly growthGapOverHorizon: number;
  readonly addOneUndershoots: boolean;
  readonly detail: string;
}

export function growthUndershootCase(): GrowthUndershootReport {
  const pStar = exactFraction(51n, 100n);
  const payout = exactFraction(1n, 1n);
  const trials = 4000;
  const wins = 2040; // exactly p* * trials, so the MLE realized rate is p*
  const losses = trials - wins;
  const mle = mleKelly([wins, losses], payout, 0);
  const addOne = buleyeanKelly([wins, losses], payout, 0);
  const trueKelly = trueKellyFraction(pStar, payout);
  const mleGrowth = expectedLogGrowth(pStar, payout, mle.recommendedFraction);
  const addOneGrowth = expectedLogGrowth(pStar, payout, addOne.recommendedFraction);
  const growthGap = mleGrowth - addOneGrowth;
  return {
    kind: 'undershoot',
    pStar,
    payout,
    trials,
    wins,
    mleFraction: mle.recommendedFraction,
    addOneFraction: addOne.recommendedFraction,
    trueKellyFraction: trueKelly,
    mleExpectedLogGrowth: mleGrowth,
    addOneExpectedLogGrowth: addOneGrowth,
    growthGap,
    growthGapOverHorizon: growthGap * trials,
    addOneUndershoots: growthGap > 0,
    detail:
      'small true edge p*=' + fractionToString(pStar) + ' at b=' + fractionToString(payout) +
      ', large sample N=' + String(trials) + ': the add-one posterior is below p*, so add-one sizes ' +
      fractionToString(addOne.recommendedFraction) + ' where true Kelly is ' +
      fractionToString(trueKelly) + ' (the MLE, told the realized rate, sizes exactly true Kelly). ' +
      'Per-trial expected log-growth is lower by ' + growthGap.toExponential(3) + ' nats (' +
      (growthGap * trials).toExponential(3) + ' over the horizon). The prior is too conservative ' +
      'here, but its cost is second-order and negligible once the sample is large. This is a ' +
      'fixed-history expected-growth comparison; on a real path the MLE all-in ruins early instead.',
  };
}

/** The small-sample case where the MLE ruins and add-one saves capital. */
export interface CapitalSavedReport {
  readonly kind: 'saves-capital';
  readonly pStar: ExactFraction;
  readonly payout: ExactFraction;
  readonly trials: number;
  readonly floorWealth: number;
  readonly seeds: number;
  readonly mleRuinProbability: number;
  readonly addOneRuinProbability: number;
  readonly mleMeanTerminalLogWealth: number;
  readonly addOneMeanTerminalLogWealth: number;
  readonly mleMedianTerminalLogWealth: number;
  readonly addOneMedianTerminalLogWealth: number;
  readonly capitalSaved: boolean;
  readonly detail: string;
}

export function capitalSavedCase(seeds: readonly number[] = seedRange(512)): CapitalSavedReport {
  const pStar = exactFraction(11n, 20n);
  const payout = exactFraction(1n, 1n);
  const trials = 60;
  const floorWealth = 0.01;
  const spec = staticMarket(pStar, payout, trials, floorWealth);
  const report = runBacktest(spec, [mleKellyStrategy(), addOneKellyStrategy()], seeds);
  const mle = strategyReportById(report, 'mle-kelly');
  const addOne = strategyReportById(report, 'add-one-kelly');
  return {
    kind: 'saves-capital',
    pStar,
    payout,
    trials,
    floorWealth,
    seeds: seeds.length,
    mleRuinProbability: mle.ruinProbability,
    addOneRuinProbability: addOne.ruinProbability,
    mleMeanTerminalLogWealth: mle.meanTerminalLogWealth,
    addOneMeanTerminalLogWealth: addOne.meanTerminalLogWealth,
    mleMedianTerminalLogWealth: mle.medianTerminalLogWealth,
    addOneMedianTerminalLogWealth: addOne.medianTerminalLogWealth,
    capitalSaved: addOne.ruinProbability < mle.ruinProbability,
    detail:
      'small sample p*=' + fractionToString(pStar) + ' at b=' + fractionToString(payout) +
      ', N=' + String(trials) + ', ' + String(seeds.length) + ' seeds: an early all-win record makes ' +
      'the MLE all-in, so it ruins on ' + (mle.ruinProbability * 100).toFixed(1) + '% of paths, while ' +
      'add-one ruins on ' + (addOne.ruinProbability * 100).toFixed(1) + '% and lifts the median ' +
      'terminal log-wealth from ' + mle.medianTerminalLogWealth.toFixed(3) + ' to ' +
      addOne.medianTerminalLogWealth.toFixed(3) + '. The prior saves capital here. Saving capital ' +
      'is not the same as making money: with an edge this small the surviving add-one paths can ' +
      'still finish below their start, because the add-one posterior briefly overbets early.',
  };
}

function strategyReportById(report: BacktestReport, id: string): StrategyReport {
  for (const entry of report.reports) {
    if (entry.strategyId === id) return entry;
  }
  throw new RangeError('report has no strategy ' + id);
}

/** The honest dual: both directions, plus a verdict that names no universal winner. */
export interface BacktestAdversarialDual {
  readonly undershoot: GrowthUndershootReport;
  readonly savesCapital: CapitalSavedReport;
  readonly verdict: string;
}

export function backtestAdversarialDual(): BacktestAdversarialDual {
  const undershoot = growthUndershootCase();
  const savesCapital = capitalSavedCase();
  const verdict = [
    'Honest verdict: neither rule wins everywhere.',
    '(1) Undershoot: ' + undershoot.detail,
    '(2) Saves capital: ' + savesCapital.detail,
    'Which dominates where: on a fixed, accurately estimated large-sample history, add-one',
    'underbets the true-Kelly fraction -- the prior is too conservative there -- but the growth',
    'cost is second-order (about ' + undershoot.growthGap.toExponential(3) + ' nats per trial). On an actual',
    'small-sample path, the MLE all-in after an early all-win record drives frequent ruin while',
    'add-one keeps the stake finite and saves the capital. Saving capital is not the same as',
    'making money: with a very small edge add-one can still finish below its start, because its',
    'early posterior briefly overbets. Add-one is not a universal winner and no profitability is claimed.',
  ].join(' ');
  return { undershoot, savesCapital, verdict };
}
