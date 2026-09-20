/**
 * tensor-bayes-blackwell.ts -- garbling cannot help, and mass cannot see why.
 *
 * Mirrors `Gnosis.TensorBayesBlackwell` and `Gnosis.TensorBayesBlackwellRatio`.
 * An experiment is a joint Nat mass table `P[state][signal]` (prior x
 * likelihood, cleared of denominators). A decision rule `delta` maps each
 * signal to an action; the Bayes risk of `delta` under a Nat loss table `L`
 * (`L[action][state]`) is
 *
 *   risk(P, L, delta) = sum_signal sum_state P[state][signal] * L[delta[signal]][state]
 *
 * `risk_garble` proves that garbling (merging signals) never lowers the best
 * achievable risk -- the pulled-back rule on the original experiment matches
 * every rule on the garbling EXACTLY. The headline dual, `mass_blind_risk_sees`,
 * is the point of the whole module: on the SAME merge, the mass ledger (the
 * corpus `loss` account) reads exactly 0 -- mass is perfectly conserved --
 * while the best achievable risk jumps from 0 to 1. Mass conservation cannot
 * see information loss.
 */

/** `P[state][signal]`, Nat masses. */
export type Experiment = ReadonlyArray<ReadonlyArray<number>>;
/** `L[action][state]`, Nat losses. */
export type LossTable = ReadonlyArray<ReadonlyArray<number>>;
/** `delta[signal] = action`. */
export type Rule = ReadonlyArray<number>;

/** `bayesRisk`: `sum_signal sum_state P[state][signal] * L[delta[signal]][state]`. */
export function bayesRisk(P: Experiment, L: LossTable, delta: Rule): number {
  const signals = delta.length;
  const states = P.length;
  let risk = 0;
  for (let x = 0; x < signals; x++) {
    for (let st = 0; st < states; st++) {
      risk += P[st]![x]! * L[delta[x]!]![st]!;
    }
  }
  return risk;
}

/** Every rule `Fin signals -> Fin actions`, for small finite enumeration
 *  (mirrors the concrete `Fin 1`/`Fin 2` witnesses the Lean uses). */
export function enumerateRules(signals: number, actions: number): Rule[] {
  let rules: number[][] = [[]];
  for (let s = 0; s < signals; s++) {
    const next: number[][] = [];
    for (const r of rules) for (let a = 0; a < actions; a++) next.push([...r, a]);
    rules = next;
  }
  return rules;
}

/** `IsBayesOptimal`: no other rule has strictly smaller risk, checked over
 *  the full finite rule space (only tractable for small signals/actions). */
export function isBayesOptimal(P: Experiment, L: LossTable, delta: Rule): boolean {
  const signals = delta.length;
  const actions = L.length;
  const risk = bayesRisk(P, L, delta);
  return enumerateRules(signals, actions).every((rule) => risk <= bayesRisk(P, L, rule));
}

/** `garble`: deterministic garbling, `f[signal] = mergedSignal`. */
export function garble(P: Experiment, f: readonly number[], signalsPrime: number): number[][] {
  return P.map((row) => {
    const out = new Array(signalsPrime).fill(0);
    for (let x = 0; x < row.length; x++) out[f[x]!] += row[x]!;
    return out;
  });
}

/** `garble_eq_compose`/pulled-back rule: `delta' . f`. */
export function pulledBackRule(f: readonly number[], deltaPrime: Rule): Rule {
  return f.map((y) => deltaPrime[y]!);
}

/**
 * `risk_garble`, the easy Blackwell direction's engine. Every rule `delta'`
 * on the garbled experiment has EXACTLY the risk of the pulled-back rule on
 * the original -- a genuine identity, not an inequality.
 */
export function riskGarbleIdentityHolds(
  P: Experiment,
  L: LossTable,
  f: readonly number[],
  signalsPrime: number,
  deltaPrime: Rule,
): boolean {
  const garbled = garble(P, f, signalsPrime);
  return bayesRisk(garbled, L, deltaPrime) === bayesRisk(P, L, pulledBackRule(f, deltaPrime));
}

/**
 * `optimal_risk_le_garbled`: garbling cannot help. A Bayes-optimal rule on
 * the original experiment does at least as well as every rule on the
 * garbling.
 */
export function optimalRiskLeGarbled(
  P: Experiment,
  L: LossTable,
  f: readonly number[],
  signalsPrime: number,
  deltaStar: Rule,
  deltaPrime: Rule,
): boolean {
  const garbled = garble(P, f, signalsPrime);
  return bayesRisk(P, L, deltaStar) <= bayesRisk(garbled, L, deltaPrime);
}

/** `garble_row_mass`: mass conservation under deterministic garbling, one row. */
export function garbleRowMassHolds(P: Experiment, f: readonly number[], signalsPrime: number, state: number): boolean {
  const garbled = garble(P, f, signalsPrime);
  const originalMass = P[state]!.reduce((a, b) => a + b, 0);
  const garbledMass = garbled[state]!.reduce((a, b) => a + b, 0);
  return originalMass === garbledMass;
}

// ═══════════════════════════════════════════════════════════════════════════
// Randomized garbling, cleared of denominators (TensorBayesBlackwellRatio)
// ═══════════════════════════════════════════════════════════════════════════

/** `G[signal][signalPrime]`, Nat weights. */
export type RandomKernel = ReadonlyArray<ReadonlyArray<number>>;

/** `RowSumsTo`: every row of `G` sums to the common denominator `d`. */
export function rowSumsTo(G: RandomKernel, d: number): boolean {
  return G.every((row) => row.reduce((a, b) => a + b, 0) === d);
}

/** `rgarble`: sequential composition `compose(P, G)`. */
export function rgarble(P: Experiment, G: RandomKernel): number[][] {
  const states = P.length;
  const signals = G.length;
  const signalsPrime = G[0]?.length ?? 0;
  const out: number[][] = Array.from({ length: states }, () => new Array(signalsPrime).fill(0));
  for (let st = 0; st < states; st++) {
    for (let x = 0; x < signals; x++) {
      for (let y = 0; y < signalsPrime; y++) {
        out[st]![y] += P[st]![x]! * G[x]![y]!;
      }
    }
  }
  return out;
}

/** `rgarble_row_mass`: the row mass scales by the common denominator `d`. */
export function rgarbleRowMassHolds(P: Experiment, G: RandomKernel, d: number, state: number): boolean {
  const garbled = rgarble(P, G);
  const originalMass = P[state]!.reduce((a, b) => a + b, 0);
  const garbledMass = garbled[state]!.reduce((a, b) => a + b, 0);
  return garbledMass === d * originalMass;
}

/**
 * `optimal_risk_le_rgarbled`: randomized garbling cannot help either, scaled
 * by the common row-denominator `d` -- no division, only a multiplication on
 * the optimal side: `d * risk(P,L,deltaStar) <= risk(rgarble(P,G), L, deltaPrime)`.
 */
export function optimalRiskLeRgarbled(
  P: Experiment,
  L: LossTable,
  G: RandomKernel,
  d: number,
  deltaStar: Rule,
  deltaPrime: Rule,
): boolean {
  const garbled = rgarble(P, G);
  return d * bayesRisk(P, L, deltaStar) <= bayesRisk(garbled, L, deltaPrime);
}

// ═══════════════════════════════════════════════════════════════════════════
// The headline concrete example: mass is blind, risk sees
// ═══════════════════════════════════════════════════════════════════════════

/** The perfectly informative 2-state, 2-signal experiment: signal = state. */
export const perfectExp: Experiment = [
  [1, 0],
  [0, 1],
];

/** 0-1 loss: an action costs 1 unless it names the true state. */
export const zeroOneLoss: LossTable = [
  [0, 1],
  [1, 0],
];

/** The garbling that merges both signals into one. */
export const mergeSignals: readonly number[] = [0, 0];

/** The merged (fully uninformative) experiment. */
export function mergedExp(): number[][] {
  return garble(perfectExp, mergeSignals, 1);
}

/**
 * `mass_blind_risk_sees`, THE HEADLINE. On this merge:
 * - the identity rule on the perfect experiment has risk exactly 0;
 * - the mass account (row sums) is UNCHANGED by the merge, i.e. the
 *   deterministic channel loses zero mass on every state row;
 * - yet EVERY rule on the merged experiment has risk exactly 1.
 * Mass conservation cannot see the information loss.
 */
export function massBlindRiskSees(): {
  identityRiskOnPerfect: number;
  massConservedOnEveryRow: boolean;
  everyMergedRuleRisk: number[];
} {
  const identityRiskOnPerfect = bayesRisk(perfectExp, zeroOneLoss, [0, 1]);
  const massConservedOnEveryRow =
    garbleRowMassHolds(perfectExp, mergeSignals, 1, 0) && garbleRowMassHolds(perfectExp, mergeSignals, 1, 1);
  const merged = mergedExp();
  const everyMergedRuleRisk = enumerateRules(1, 2).map((rule) => bayesRisk(merged, zeroOneLoss, rule));
  return { identityRiskOnPerfect, massConservedOnEveryRow, everyMergedRuleRisk };
}

/**
 * `doubledKernel`: a merge kernel whose rows sum to 2, not 1 -- a valid
 * `RowSumsTo _ 2` randomized garbling reusing `mergeSignals`.
 */
export const doubledKernel: RandomKernel = [
  [2],
  [2],
];

/**
 * `rgarbling_strictly_hurts`: the scaled bound can be far from tight. On the
 * perfect experiment the scaled optimum is `2 * 0 = 0`, while EVERY rule on
 * the `doubledKernel` randomized garbling has risk exactly 2.
 */
export function rgarblingStrictlyHurts(): { scaledOptimum: number; everyRuleRisk: number[] } {
  const scaledOptimum = 2 * bayesRisk(perfectExp, zeroOneLoss, [0, 1]);
  const garbled = rgarble(perfectExp, doubledKernel);
  const everyRuleRisk = enumerateRules(1, 2).map((rule) => bayesRisk(garbled, zeroOneLoss, rule));
  return { scaledOptimum, everyRuleRisk };
}

/**
 * `mass_does_not_determine_risk`: two experiments with equal total mass 4,
 * whose best achievable risks are 0 and 2 respectively. Mass conservation
 * does not capture information.
 */
export const perfectExp2: Experiment = [
  [2, 0],
  [0, 2],
];

export const noiseExp: Experiment = [
  [1, 1],
  [1, 1],
];

export function massDoesNotDetermineRisk(): {
  totalMassEqual: boolean;
  perfect2BestRisk: number;
  noiseWorstOverAllRules: number;
} {
  const totalMass = (E: Experiment) => E.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);
  const totalMassEqual = totalMass(perfectExp2) === totalMass(noiseExp);
  const perfect2BestRisk = bayesRisk(perfectExp2, zeroOneLoss, [0, 1]);
  const noiseRisks = enumerateRules(2, 2).map((rule) => bayesRisk(noiseExp, zeroOneLoss, rule));
  return { totalMassEqual, perfect2BestRisk, noiseWorstOverAllRules: Math.min(...noiseRisks) };
}
