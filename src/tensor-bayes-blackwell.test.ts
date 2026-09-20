import { describe, expect, it } from 'bun:test';
import {
  bayesRisk,
  enumerateRules,
  isBayesOptimal,
  garble,
  pulledBackRule,
  riskGarbleIdentityHolds,
  optimalRiskLeGarbled,
  garbleRowMassHolds,
  rowSumsTo,
  rgarble,
  rgarbleRowMassHolds,
  optimalRiskLeRgarbled,
  perfectExp,
  zeroOneLoss,
  mergeSignals,
  mergedExp,
  massBlindRiskSees,
  doubledKernel,
  rgarblingStrictlyHurts,
  perfectExp2,
  noiseExp,
  massDoesNotDetermineRisk,
} from './tensor-bayes-blackwell';

describe('blackwell -- experiments, rules and Bayes risk', () => {
  it('the identity rule on the perfect experiment has risk 0', () => {
    expect(bayesRisk(perfectExp, zeroOneLoss, [0, 1])).toBe(0);
  });

  it('the identity rule is Bayes-optimal on the perfect experiment', () => {
    expect(isBayesOptimal(perfectExp, zeroOneLoss, [0, 1])).toBe(true);
  });

  it('anti-vacuity: the constant rule is NOT optimal on the perfect experiment', () => {
    expect(isBayesOptimal(perfectExp, zeroOneLoss, [0, 0])).toBe(false);
  });

  it('enumerateRules produces d^m rules', () => {
    expect(enumerateRules(2, 2).length).toBe(4);
    expect(enumerateRules(3, 2).length).toBe(8);
  });
});

describe('blackwell -- risk_garble, the easy direction', () => {
  it('risk_garble: every rule on the garbled experiment equals the pulled-back rule on the original', () => {
    for (const deltaPrime of enumerateRules(1, 2)) {
      expect(riskGarbleIdentityHolds(perfectExp, zeroOneLoss, mergeSignals, 1, deltaPrime)).toBe(true);
    }
  });

  it('pulledBackRule composes f then delta-prime', () => {
    expect(pulledBackRule(mergeSignals, [1])).toEqual([1, 1]);
  });

  it('optimal_risk_le_garbled: garbling cannot help, for every garbled rule', () => {
    for (const deltaPrime of enumerateRules(1, 2)) {
      expect(optimalRiskLeGarbled(perfectExp, zeroOneLoss, mergeSignals, 1, [0, 1], deltaPrime)).toBe(true);
    }
  });

  it('garble_row_mass: deterministic garbling conserves mass on every row', () => {
    expect(garbleRowMassHolds(perfectExp, mergeSignals, 1, 0)).toBe(true);
    expect(garbleRowMassHolds(perfectExp, mergeSignals, 1, 1)).toBe(true);
  });
});

describe('blackwell -- THE HEADLINE: mass is blind, risk sees', () => {
  it('mass_blind_risk_sees: zero risk before, mass conserved by the merge, risk 1 after -- for every rule', () => {
    const result = massBlindRiskSees();
    expect(result.identityRiskOnPerfect).toBe(0);
    expect(result.massConservedOnEveryRow).toBe(true);
    expect(result.everyMergedRuleRisk.length).toBeGreaterThan(0);
    for (const risk of result.everyMergedRuleRisk) expect(risk).toBe(1);
  });

  it('garbling_strictly_hurts: the optimal original risk is strictly below every merged rule risk', () => {
    const merged = mergedExp();
    const optimalOriginal = bayesRisk(perfectExp, zeroOneLoss, [0, 1]);
    for (const rule of enumerateRules(1, 2)) {
      expect(optimalOriginal).toBeLessThan(bayesRisk(merged, zeroOneLoss, rule));
    }
  });
});

describe('blackwell -- randomized garbling, cleared of denominators', () => {
  it('doubledKernel is a valid RowSumsTo _ 2 kernel', () => {
    expect(rowSumsTo(doubledKernel, 2)).toBe(true);
    expect(rowSumsTo(doubledKernel, 1)).toBe(false);
  });

  it('rgarble_row_mass: randomized garbling scales mass by the common denominator', () => {
    expect(rgarbleRowMassHolds(perfectExp, doubledKernel, 2, 0)).toBe(true);
    expect(rgarbleRowMassHolds(perfectExp, doubledKernel, 2, 1)).toBe(true);
  });

  it('optimal_risk_le_rgarbled: the scaled cannot-help inequality holds for every rule', () => {
    for (const deltaPrime of enumerateRules(1, 2)) {
      expect(optimalRiskLeRgarbled(perfectExp, zeroOneLoss, doubledKernel, 2, [0, 1], deltaPrime)).toBe(true);
    }
  });

  it('rgarbling_strictly_hurts: the scaled bound can be far from tight (0 vs 2)', () => {
    const { scaledOptimum, everyRuleRisk } = rgarblingStrictlyHurts();
    expect(scaledOptimum).toBe(0);
    for (const risk of everyRuleRisk) expect(risk).toBe(2);
  });
});

describe('blackwell -- adversarial dual: mass does not determine risk', () => {
  it('two experiments with the same total mass 4 have best risks 0 and 2', () => {
    const result = massDoesNotDetermineRisk();
    expect(result.totalMassEqual).toBe(true);
    expect(result.perfect2BestRisk).toBe(0);
    expect(result.noiseWorstOverAllRules).toBe(2);
  });

  it('perfectExp2 and noiseExp really do carry equal mass', () => {
    const mass = (E: typeof perfectExp2) => E.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);
    expect(mass(perfectExp2)).toBe(4);
    expect(mass(noiseExp)).toBe(4);
  });
});

describe('blackwell -- the unit-row property is load-bearing', () => {
  it('a garbling that merges signal 0 and 1 into signal 0 has rgarble reproduce garble at d=1', () => {
    const detAsRandom: readonly (readonly number[])[] = [
      [1],
      [1],
    ];
    expect(rowSumsTo(detAsRandom, 1)).toBe(true);
    const rg = rgarble(perfectExp, detAsRandom);
    const g = garble(perfectExp, mergeSignals, 1);
    expect(rg).toEqual(g);
  });
});
