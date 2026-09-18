import { describe, expect, it } from 'bun:test';
import {
  xorshift32,
  staticMarket,
  switchingMarket,
  trueProbabilityAt,
  generateOutcomes,
  trueKellyFraction,
  recommendedTrueKelly,
  expectedLogGrowth,
  kellyGrowthRate,
  mleKellyStrategy,
  addOneKellyStrategy,
  conservativeRangeKellyStrategy,
  fixedFractionStrategy,
  noBetStrategy,
  defaultStrategies,
  oracleKellyStrategy,
  runPath,
  runOutcomeSequence,
  runBacktest,
  seedRange,
  summarizeStrategy,
  growthUndershootCase,
  capitalSavedCase,
  backtestAdversarialDual,
  type BacktestReport,
  type PathResult,
} from './backtest';
import { compareFractions, exactFraction, type ExactFraction } from './rational';
import { kellyRangeFromPosterior, robustKellyFromRange } from './kelly';

function countWins(outcomes: readonly boolean[]): number {
  let wins = 0;
  for (const outcome of outcomes) if (outcome) wins++;
  return wins;
}

/**
 * The gnode test bridge JSON-serializes its payload, so an assertion failure
 * whose reported value carries a BigInt crashes the harness and hides the real
 * diff. Compare exact fractions through compareFractions (a primitive) instead.
 */
function expectFractionEqual(left: ExactFraction, right: ExactFraction): void {
  expect(compareFractions(left, right)).toBe(0);
}

/** BigInt-free signature of a whole report, for deterministic reproducibility. */
function reportSignature(report: BacktestReport): string {
  return JSON.stringify({
    trials: report.trials,
    seeds: report.seeds,
    payout: report.payout.numerator + '/' + report.payout.denominator,
    regimes: report.regimes.map((regime) => [regime.fromTrial, regime.pWin.numerator + '/' + regime.pWin.denominator]),
    initialWealth: report.initialWealth,
    floorWealth: report.floorWealth,
    trueKellyAtStart: report.trueKellyAtStart.numerator + '/' + report.trueKellyAtStart.denominator,
    reports: report.reports,
  });
}

describe('backtest -- deterministic market generator', () => {
  it('reproduces the xorshift32 stream exactly and rejects the zero seed', () => {
    const first = xorshift32(0x12345678);
    const second = xorshift32(0x12345678);
    for (let i = 0; i < 32; i++) {
      expect(first()).toBe(second());
    }
    expect(() => xorshift32(0)).toThrow(RangeError);
    const draws = Array.from({ length: 64 }, () => xorshift32(7)());
    expect(draws.every((value) => Number.isInteger(value) && value >= 0 && value < 2 ** 32)).toBe(true);
  });

  it('decorrelates nearby seeds so the first draw is not a function of the seed rank', () => {
    // Raw xorshift32 maps seed s to a first output proportional to s, so seeds
    // 1..64 would all land on the same side of a p = 1/2 threshold.
    const firstDraws = seedRange(64).map((seed) => xorshift32(seed)());
    const belowHalf = firstDraws.filter((draw) => draw < 2 ** 31).length;
    expect(belowHalf).toBeGreaterThan(16);
    expect(belowHalf).toBeLessThan(48);
  });

  it('matches the true p* within sampling error on a static market', () => {
    const spec = staticMarket(exactFraction(51n, 100n), 1n, 40000, 0.01);
    const outcomes = generateOutcomes(spec, 0xa11ce);
    const frequency = countWins(outcomes) / spec.trials;
    expect(Math.abs(frequency - 0.51)).toBeLessThan(0.02);
    expect(outcomes.length).toBe(40000);
    expect(generateOutcomes(spec, 0xa11ce)).toEqual(outcomes);
    expect(generateOutcomes(spec, 0xb0b)).not.toEqual(outcomes);
  });

  it('switches the regime partway and reads each regime back exactly', () => {
    const spec = switchingMarket(exactFraction(7n, 10n), exactFraction(2n, 5n), 4000, 1n, 8000, 0.01);
    expectFractionEqual(trueProbabilityAt(spec, 0), exactFraction(7n, 10n));
    expectFractionEqual(trueProbabilityAt(spec, 3999), exactFraction(7n, 10n));
    expectFractionEqual(trueProbabilityAt(spec, 4000), exactFraction(2n, 5n));
    expectFractionEqual(trueProbabilityAt(spec, 7999), exactFraction(2n, 5n));

    const outcomes = generateOutcomes(spec, 0x5eed);
    const before = countWins(outcomes.slice(0, 4000)) / 4000;
    const after = countWins(outcomes.slice(4000)) / 4000;
    expect(Math.abs(before - 0.7)).toBeLessThan(0.04);
    expect(Math.abs(after - 0.4)).toBeLessThan(0.04);
  });

  it('rejects malformed markets and trials outside the horizon', () => {
    expect(() => staticMarket(1.5, 1n, 10)).toThrow(RangeError);
    expect(() => staticMarket(0.5, 0n, 10)).toThrow(RangeError);
    expect(() => staticMarket(0.5, 1n, 0)).toThrow(RangeError);
    expect(() => switchingMarket(0.5, 0.4, 0, 1n, 10)).toThrow(RangeError);
    expect(() => switchingMarket(0.5, 0.4, 10, 1n, 10)).toThrow(RangeError);
    const spec = staticMarket(exactFraction(1n, 2n), 1n, 10);
    expect(() => trueProbabilityAt(spec, -1)).toThrow(RangeError);
    expect(() => trueProbabilityAt(spec, 10)).toThrow(RangeError);
    expect(() => runPath(spec, noBetStrategy(), 1, 0)).toThrow(RangeError);
  });
});

describe('backtest -- bankroll simulators', () => {
  it('holds the no-bet baseline at zero log-wealth with no drawdown', () => {
    const spec = staticMarket(exactFraction(11n, 20n), 1n, 50, 0.01);
    const path = runPath(spec, noBetStrategy(), 42);
    expect(path.terminalWealth).toBeCloseTo(1, 12);
    expect(path.terminalLogWealth).toBe(0);
    expect(path.ruined).toBe(false);
    expect(path.maxDrawdown).toBe(0);
    expect(path.overbetEver).toBe(false);
    expect(path.trialsSettled).toBe(50);
  });

  it('settles a fixed fraction on a deterministic sequence exactly', () => {
    const spec = staticMarket(exactFraction(1n, 2n), 1n, 2, 0.01);
    const path = runOutcomeSequence(spec, fixedFractionStrategy(exactFraction(1n, 2n)), [true, false]);
    // 1 -> *1.5 -> *0.5 = 0.75; peak 1.5; true Kelly at p=1/2 is 0, so both stakes overbet.
    expect(path.terminalWealth).toBeCloseTo(0.75, 12);
    expect(path.terminalLogWealth).toBeCloseTo(Math.log(0.75), 12);
    expect(path.maxDrawdown).toBeCloseTo(0.5, 12);
    expect(path.overbetEver).toBe(true);
    expect(path.overbetSteps).toBe(2);
    expect(path.ruined).toBe(false);
  });

  it('ruins the MLE after an early all-win record overbets the true Kelly', () => {
    const spec = staticMarket(exactFraction(51n, 100n), 1n, 2, 0.01);
    const path = runOutcomeSequence(spec, mleKellyStrategy(), [true, false]);
    // Trial 1 uses p=1 -> f=1, then a loss takes wealth to the floor.
    expect(path.overbetEver).toBe(true);
    expect(path.overbetSteps).toBe(1);
    expect(path.ruined).toBe(true);
    expect(path.terminalWealth).toBeCloseTo(0.01, 12);
    expect(path.terminalLogWealth).toBeCloseTo(Math.log(0.01), 12);
  });

  it('never overbets the true Kelly when it is handed the true p*', () => {
    const spec = staticMarket(exactFraction(11n, 20n), 1n, 200, 0.01);
    const oracle = oracleKellyStrategy(spec);
    for (const seed of seedRange(50)) {
      const path = runPath(spec, oracle, seed);
      expect(path.overbetEver).toBe(false);
      expect(path.overbetSteps).toBe(0);
    }
  });

  it('keeps drawdown, ruin and log-wealth consistent on every path', () => {
    const spec = switchingMarket(exactFraction(3n, 5n), exactFraction(9n, 20n), 30, 1n, 60, 0.01);
    for (const strategy of defaultStrategies()) {
      for (const seed of seedRange(64)) {
        const path = runPath(spec, strategy, seed);
        expect(path.maxDrawdown).toBeGreaterThanOrEqual(0);
        expect(path.maxDrawdown).toBeLessThanOrEqual(1);
        expect(Number.isFinite(path.terminalLogWealth)).toBe(true);
        if (path.ruined) {
          expect(path.terminalWealth).toBeLessThanOrEqual(spec.floorWealth);
          expect(path.trialsSettled).toBeLessThanOrEqual(spec.trials);
        }
      }
    }
  });

  it('aggregates a hand-built set of paths', () => {
    const paths: PathResult[] = [
      { seed: 1, strategyId: 'x', terminalWealth: 2, terminalLogWealth: Math.log(2), ruined: false, maxDrawdown: 0.1, overbetEver: true, overbetSteps: 1, trialsSettled: 5 },
      { seed: 2, strategyId: 'x', terminalWealth: 0.01, terminalLogWealth: Math.log(0.01), ruined: true, maxDrawdown: 0.99, overbetEver: false, overbetSteps: 0, trialsSettled: 5 },
      { seed: 3, strategyId: 'x', terminalWealth: 1, terminalLogWealth: 0, ruined: false, maxDrawdown: 0, overbetEver: true, overbetSteps: 3, trialsSettled: 5 },
    ];
    const report = summarizeStrategy(fixedFractionStrategy(exactFraction(1n, 4n)), paths);
    expect(report.paths).toBe(3);
    expect(report.medianTerminalLogWealth).toBe(0);
    expect(report.meanTerminalLogWealth).toBeCloseTo((Math.log(2) + Math.log(0.01) + 0) / 3, 12);
    expect(report.ruinProbability).toBeCloseTo(1 / 3, 12);
    expect(report.overbetPathFraction).toBeCloseTo(2 / 3, 12);
    expect(report.meanMaxDrawdown).toBeCloseTo((0.1 + 0.99 + 0) / 3, 12);
    expect(report.medianMaxDrawdown).toBeCloseTo(0.1, 12);
  });
});

describe('backtest -- adversarial dual', () => {
  it('undershoots growth on a large sample with a small true edge', () => {
    const report = growthUndershootCase();
    expect(report.addOneUndershoots).toBe(true);
    expect(report.growthGap).toBeGreaterThan(0);
    expect(report.growthGapOverHorizon).toBeGreaterThan(0);
    // Add-one is below p*, MLE (told the realized rate) is exactly true Kelly.
    expect(compareFractions(report.addOneFraction, report.trueKellyFraction)).toBeLessThan(0);
    expect(compareFractions(report.mleFraction, report.trueKellyFraction)).toBe(0);
    expect(report.addOneExpectedLogGrowth).toBeLessThan(report.mleExpectedLogGrowth);
  });

  it('saves capital on a small sample where the MLE overbets into ruin', () => {
    const report = capitalSavedCase();
    expect(report.capitalSaved).toBe(true);
    expect(report.mleRuinProbability).toBeGreaterThan(report.addOneRuinProbability);
    expect(report.mleRuinProbability).toBeGreaterThan(0.3);
    expect(report.addOneRuinProbability).toBeLessThan(0.05);
    expect(report.addOneMedianTerminalLogWealth).toBeGreaterThan(report.mleMedianTerminalLogWealth);
  });

  it('states both directions without claiming a universal winner', () => {
    const dual = backtestAdversarialDual();
    expect(dual.undershoot.kind).toBe('undershoot');
    expect(dual.savesCapital.kind).toBe('saves-capital');
    expect(dual.verdict.length).toBeGreaterThan(0);
    expect(dual.verdict).toContain('not a universal winner');
    expect(dual.verdict).toContain('saves the capital');
  });
});

describe('backtest -- provable properties', () => {
  it('orders ruin probability on the overbetting case', () => {
    const spec = staticMarket(exactFraction(11n, 20n), 1n, 60, 0.01);
    const seeds = seedRange(256);
    const report = runBacktest(spec, [mleKellyStrategy(), addOneKellyStrategy()], seeds);
    const mle = report.reports[0]!;
    const addOne = report.reports[1]!;
    expect(mle.strategyId).toBe('mle-kelly');
    expect(addOne.strategyId).toBe('add-one-kelly');
    expect(mle.ruinProbability).toBeGreaterThanOrEqual(addOne.ruinProbability);
    expect(mle.ruinProbability).toBeGreaterThan(0.3);
    expect(addOne.ruinProbability).toBeLessThan(0.05);
    // Nearly every path overbets at some point; add-one's per-step overbet is
    // bounded but it survives longer, so absolute overbet counts are not a
    // magnitude comparison. The proven ordering here is ruin.
    expect(mle.overbetPathFraction).toBeGreaterThanOrEqual(addOne.overbetPathFraction);
    expect(mle.meanOverbetSteps).toBeGreaterThan(0);
    expect(addOne.meanOverbetSteps).toBeGreaterThan(0);
  });

  it('never lets conservative range Kelly exceed true Kelly when the lower bound is below p*', () => {
    for (let trials = 1; trials <= 40; trials++) {
      for (const b of [1n, 2n, 3n, 5n]) {
        for (const multiplier of [1n, exactFraction(1n, 2n)]) {
          // p* is strictly above the proven lower endpoint 1/(N+2).
          const above = exactFraction(1n, BigInt(trials + 1));
          const strategy = conservativeRangeKellyStrategy(multiplier);
          const stake = strategy.decision({
            wins: 0,
            losses: trials,
            trials,
            payout: exactFraction(b, 1n),
          });
          const trueStake = recommendedTrueKelly(above, exactFraction(b, 1n));
          expect(compareFractions(stake, trueStake)).toBeLessThanOrEqual(0);
        }
      }
    }
  });

  it('matches the proven collapse-range lower endpoint exactly', () => {
    for (let trials = 0; trials <= 20; trials++) {
      for (const b of [exactFraction(1n, 1n), exactFraction(2n, 1n), exactFraction(3n, 2n)]) {
        const stake = conservativeRangeKellyStrategy().decision({
          wins: 0,
          losses: trials,
          trials,
          payout: b,
        });
        const range = kellyRangeFromPosterior([0, trials], b, 0);
        expectFractionEqual(stake, range.conservativeFraction);
        const robust = robustKellyFromRange(1n, BigInt(trials + 2), b.numerator, b.denominator, 1n);
        expectFractionEqual(stake, robust.size);
      }
    }
  });

  it('is monotone increasing in true p* for the growth-optimal stake', () => {
    for (const b of [1n, 2n, 3n]) {
      let previous = kellyGrowthRate(exactFraction(1n, 2n), exactFraction(b, 1n));
      for (let step = 1; step <= 49; step++) {
        const p = exactFraction(BigInt(50 + step), 100n);
        const current = kellyGrowthRate(p, exactFraction(b, 1n));
        expect(current).toBeGreaterThanOrEqual(previous - 1e-12);
        previous = current;
      }
      expect(previous).toBeGreaterThan(kellyGrowthRate(exactFraction(1n, 2n), exactFraction(b, 1n)) + 1e-6);
    }
  });

  it('is monotone increasing in true p* for a fixed stake', () => {
    const stake = exactFraction(1n, 5n);
    let previous = expectedLogGrowth(exactFraction(1n, 2n), 1n, stake);
    for (let step = 1; step <= 49; step++) {
      const p = exactFraction(BigInt(50 + step), 100n);
      const current = expectedLogGrowth(p, 1n, stake);
      expect(current).toBeGreaterThan(previous);
      previous = current;
    }
  });

  it('keeps the sizing decisions exact rationals', () => {
    expectFractionEqual(trueKellyFraction(exactFraction(51n, 100n), 1n), exactFraction(1n, 50n));
    expectFractionEqual(recommendedTrueKelly(exactFraction(1n, 4n), 1n), exactFraction(0n, 1n));

    const context = { wins: 1, losses: 0, trials: 1, payout: exactFraction(1n, 1n) };
    expectFractionEqual(addOneKellyStrategy().decision(context), exactFraction(1n, 3n));
    expectFractionEqual(mleKellyStrategy().decision(context), exactFraction(1n, 1n));

    // Add-one at (wins=3, losses=2) is kellyFor((3+1)/(5+2)) = kellyFor(4/7) = 1/7.
    const countsContext = { wins: 3, losses: 2, trials: 5, payout: exactFraction(1n, 1n) };
    const addOne = addOneKellyStrategy().decision(countsContext);
    expectFractionEqual(addOne, exactFraction(1n, 7n));
    expectFractionEqual(addOne, trueKellyFraction(exactFraction(4n, 7n), 1n));
  });

  it('reproduces the same output for the same seed and input', () => {
    const spec = switchingMarket(exactFraction(3n, 5n), exactFraction(9n, 20n), 25, 1n, 50, 0.01);
    const strategies = defaultStrategies();
    const seeds = seedRange(48);
    const first = runBacktest(spec, strategies, seeds);
    const second = runBacktest(spec, strategies, seeds);
    expect(reportSignature(first)).toBe(reportSignature(second));
    expect(first.reports.length).toBe(5);
    expect(first.reports.map((entry) => entry.strategyId)).toEqual([
      'mle-kelly',
      'add-one-kelly',
      'conservative-range-kelly',
      'fixed-1/4',
      'no-bet',
    ]);

    const pathA = runPath(spec, addOneKellyStrategy(), 123);
    const pathB = runPath(spec, addOneKellyStrategy(), 123);
    expect(pathA).toEqual(pathB);
    expect(generateOutcomes(spec, 123)).toEqual(generateOutcomes(spec, 123));
    expect(seedRange(5, 10)).toEqual([10, 11, 12, 13, 14]);
  });
});

describe('backtest -- regime switching', () => {
  it('makes a single global estimate wrong for part of the run', () => {
    const spec = switchingMarket(exactFraction(3n, 5n), exactFraction(9n, 20n), 50, 1n, 100, 0.01);
    // Before the switch the edge is real; after it the true Kelly is zero.
    expect(compareFractions(trueKellyFraction(trueProbabilityAt(spec, 49), 1n), 0n)).toBeGreaterThan(0);
    expectFractionEqual(recommendedTrueKelly(trueProbabilityAt(spec, 50), 1n), exactFraction(0n, 1n));
    // A strategy sized on the pooled pre-switch record (30W/20L) still bets 1/5.
    const pooled = mleKellyStrategy().decision({
      wins: 30,
      losses: 20,
      trials: 50,
      payout: exactFraction(1n, 1n),
    });
    expectFractionEqual(pooled, exactFraction(1n, 5n));
    expect(
      compareFractions(pooled, recommendedTrueKelly(trueProbabilityAt(spec, 50), 1n)),
    ).toBeGreaterThan(0);
  });

  it('runs every strategy through a switching market deterministically', () => {
    const spec = switchingMarket(exactFraction(7n, 10n), exactFraction(1n, 2n), 40, 1n, 80, 0.01);
    const report = runBacktest(spec, defaultStrategies(), seedRange(64));
    expect(report.trials).toBe(80);
    expect(report.seeds).toBe(64);
    expect(report.regimes.length).toBe(2);
    expectFractionEqual(report.trueKellyAtStart, exactFraction(2n, 5n));
    // After the switch p* = 1/2 at even odds: no edge, so the true Kelly is 0.
    expectFractionEqual(recommendedTrueKelly(trueProbabilityAt(spec, 40), 1n), exactFraction(0n, 1n));
  });
});
