/** Finite observed-prefix utility selection. Abduction is retained as provenance,
 * never converted to observations. All inputs describe one bounded trace census. */
import { compileFiniteDistribution, eventMass, probabilityRatio } from '@a0n/aeon-logic/browser';

export interface JourneyTraceMass {
  readonly id: string;
  readonly contextKey: string;
  readonly steps: readonly string[];
  readonly count: number;
  readonly source: 'observed';
}
export interface JourneyResourceCost {
  readonly id: string;
  readonly bytes: number;
  readonly parseMs: number;
  readonly buildMs: number;
}
export interface JourneySelectionCandidate {
  readonly id: string;
  /** Continuation AFTER the input prefix. */
  readonly steps: readonly string[];
  /** Incremental visible wait saved at each corresponding step, not cumulative. */
  readonly savedWaitMs: readonly number[];
  readonly resources: readonly JourneyResourceCost[];
}
export interface JourneyAbductionHint {
  readonly candidateId: string;
  readonly source: 'abduction';
  readonly weight: number;
  readonly evidenceIds: readonly string[];
}
export interface JourneySelectionBudget {
  readonly maxCandidates: number;
  readonly maxBytes: number;
  readonly maxParseMs: number;
  readonly maxBuildMs: number;
}
export interface JourneyPrefixWitness {
  readonly steps: readonly string[];
  readonly observedMass: number;
  readonly probability: { readonly numerator: number; readonly denominator: number } | null;
  readonly savedWaitMs: number;
}
export interface JourneyUtilityWitness {
  readonly candidateId: string;
  readonly observedContextMass: number;
  readonly prefixes: readonly JourneyPrefixWitness[];
  readonly expectedSavedWaitMs: number;
  readonly marginalSavedWaitMs: number;
  readonly marginalCost: Omit<JourneyResourceCost, 'id'>;
  readonly utilityMs: number;
  readonly decision: 'selected' | 'unobserved' | 'budget' | 'no-benefit';
  readonly hints: readonly JourneyAbductionHint[];
}
export interface JourneySelectionInput {
  readonly contextKey: string;
  readonly prefix: readonly string[];
  readonly traces: readonly JourneyTraceMass[];
  readonly candidates: readonly JourneySelectionCandidate[];
  readonly budget: JourneySelectionBudget;
  readonly hints?: readonly JourneyAbductionHint[];
  readonly residentResourceIds?: readonly string[];
  /** Explicit conversion of speculative resource costs to the latency objective.
   * Defaults: parsing costs 1 ms/ms; transfer/build are governed by hard budgets. */
  readonly costs?: { readonly transferMsPerByte?: number; readonly parseMsWeight?: number; readonly buildMsWeight?: number };
}
export interface JourneySelectionResult {
  readonly observedContextMass: number;
  readonly witnesses: readonly JourneyUtilityWitness[];
  readonly selected: readonly JourneyUtilityWitness[];
  readonly remainingBudget: JourneySelectionBudget;
}

function nonnegative(value: number, label: string, integer = false): void {
  if (!Number.isFinite(value) || value < 0 || (integer && !Number.isSafeInteger(value))) {
    throw new Error(`${label} must be a nonnegative ${integer ? 'safe integer' : 'finite number'}`);
  }
}
function name(value: string, label: string): void {
  if (value.trim().length === 0) throw new Error(`${label} must be nonempty`);
}
function startsWith(sequence: readonly string[], prefix: readonly string[]): boolean {
  return prefix.length <= sequence.length && prefix.every((step, i) => sequence[i] === step);
}
function uniqueIds(values: readonly { readonly id: string }[], label: string): void {
  const ids = new Set<string>();
  for (const value of values) {
    name(value.id, label);
    if (ids.has(value.id)) throw new Error(`duplicate ${label}: ${value.id}`);
    ids.add(value.id);
  }
}

/** Greedy marginal selection over observed ordered-prefix events. Shared resources
 * are charged once; overlapping prefixes earn only additional saved wait. */
export function selectJourneyCandidates(input: JourneySelectionInput): JourneySelectionResult {
  name(input.contextKey, 'contextKey');
  uniqueIds(input.traces, 'trace id');
  uniqueIds(input.candidates, 'candidate id');
  const { budget } = input;
  nonnegative(budget.maxCandidates, 'maxCandidates', true);
  nonnegative(budget.maxBytes, 'maxBytes', true);
  nonnegative(budget.maxParseMs, 'maxParseMs');
  nonnegative(budget.maxBuildMs, 'maxBuildMs');
  input.prefix.forEach(step => name(step, 'prefix step'));
  for (const trace of input.traces) {
    if (trace.source !== 'observed') throw new Error('trace mass must be observed');
    name(trace.contextKey, 'trace contextKey');
    nonnegative(trace.count, 'trace count', true);
    trace.steps.forEach(step => name(step, 'trace step'));
  }
  const resources = new Map<string, JourneyResourceCost>();
  for (const candidate of input.candidates) {
    if (candidate.steps.length === 0 || candidate.steps.length !== candidate.savedWaitMs.length) {
      throw new Error('candidate requires one savedWaitMs for every continuation step');
    }
    candidate.steps.forEach(step => name(step, 'candidate step'));
    candidate.savedWaitMs.forEach(value => nonnegative(value, 'savedWaitMs'));
    uniqueIds(candidate.resources, 'resource id');
    for (const resource of candidate.resources) {
      nonnegative(resource.bytes, 'resource bytes', true);
      nonnegative(resource.parseMs, 'resource parseMs');
      nonnegative(resource.buildMs, 'resource buildMs');
      const previous = resources.get(resource.id);
      if (previous && (previous.bytes !== resource.bytes || previous.parseMs !== resource.parseMs || previous.buildMs !== resource.buildMs)) {
        throw new Error(`inconsistent costs for shared resource ${resource.id}`);
      }
      resources.set(resource.id, resource);
    }
  }
  const candidateIds = new Set(input.candidates.map(candidate => candidate.id));
  for (const hint of input.hints ?? []) {
    if (hint.source !== 'abduction' || !candidateIds.has(hint.candidateId)) throw new Error('abduction hint requires a known candidate');
    nonnegative(hint.weight, 'hint weight');
    hint.evidenceIds.forEach(id => name(id, 'hint evidence id'));
  }
  const transferWeight = input.costs?.transferMsPerByte ?? 0;
  const parseWeight = input.costs?.parseMsWeight ?? 1;
  const buildWeight = input.costs?.buildMsWeight ?? 0;
  [transferWeight, parseWeight, buildWeight].forEach(value => nonnegative(value, 'cost weight'));
  const observed = input.traces.filter(trace => trace.contextKey === input.contextKey && startsWith(trace.steps, input.prefix) && trace.count > 0);
  const distribution = observed.length ? compileFiniteDistribution(observed.map(trace => trace.count)) : null;
  const total = distribution?.totalMass ?? 0;
  const resident = new Set(input.residentResourceIds ?? []);
  const covered = new Map<string, number>();
  const remaining = { ...budget };
  const selected: JourneyUtilityWitness[] = [];
  const pending = [...input.candidates];
  const witnesses: JourneyUtilityWitness[] = [];
  function assess(candidate: JourneySelectionCandidate): JourneyUtilityWitness {
    const prefixes = candidate.steps.map((_, index): JourneyPrefixWitness => {
      const steps = [...input.prefix, ...candidate.steps.slice(0, index + 1)];
      const mass = distribution ? eventMass(distribution, observed.map(trace => startsWith(trace.steps, steps))) : 0;
      return { steps, observedMass: mass, probability: distribution ? probabilityRatio(mass, total) : null, savedWaitMs: candidate.savedWaitMs[index] ?? 0 };
    });
    let expected = 0;
    let marginal = 0;
    for (const prefix of prefixes) {
      const p = prefix.probability ? prefix.probability.numerator / prefix.probability.denominator : 0;
      expected += p * prefix.savedWaitMs;
      marginal += p * Math.max(0, prefix.savedWaitMs - (covered.get(JSON.stringify(prefix.steps)) ?? 0));
    }
    const cost = { bytes: 0, parseMs: 0, buildMs: 0 };
    for (const resource of candidate.resources) {
      if (resident.has(resource.id)) continue;
      cost.bytes += resource.bytes;
      cost.parseMs += resource.parseMs;
      cost.buildMs += resource.buildMs;
    }
    nonnegative(cost.bytes, 'summed bytes', true);
    nonnegative(cost.parseMs, 'summed parseMs');
    nonnegative(cost.buildMs, 'summed buildMs');
    const utility = marginal - cost.bytes * transferWeight - cost.parseMs * parseWeight - cost.buildMs * buildWeight;
    if (!Number.isFinite(expected) || !Number.isFinite(utility)) throw new Error('journey utility overflow');
    const decision = total === 0 ? 'unobserved' : utility <= 0 ? 'no-benefit' : remaining.maxCandidates === 0 || cost.bytes > remaining.maxBytes || cost.parseMs > remaining.maxParseMs || cost.buildMs > remaining.maxBuildMs ? 'budget' : 'selected';
    return { candidateId: candidate.id, observedContextMass: total, prefixes, expectedSavedWaitMs: expected, marginalSavedWaitMs: marginal, marginalCost: cost, utilityMs: utility, decision, hints: (input.hints ?? []).filter(hint => hint.candidateId === candidate.id) };
  }
  while (pending.length > 0) {
    const ranked = pending.map(candidate => ({ candidate, witness: assess(candidate) })).sort((a, b) => b.witness.utilityMs - a.witness.utilityMs || a.candidate.id.localeCompare(b.candidate.id));
    const next = ranked.find(entry => entry.witness.decision === 'selected');
    if (!next) {
      witnesses.push(...ranked.map(entry => entry.witness));
      break;
    }
    const { candidate, witness } = next;
    witnesses.push(witness);
    selected.push(witness);
    pending.splice(pending.indexOf(candidate), 1);
    remaining.maxCandidates--;
    remaining.maxBytes -= witness.marginalCost.bytes;
    remaining.maxParseMs -= witness.marginalCost.parseMs;
    remaining.maxBuildMs -= witness.marginalCost.buildMs;
    for (const resource of candidate.resources) resident.add(resource.id);
    for (const prefix of witness.prefixes) {
      const key = JSON.stringify(prefix.steps);
      covered.set(key, Math.max(covered.get(key) ?? 0, prefix.savedWaitMs));
    }
  }
  return { observedContextMass: total, witnesses, selected, remainingBudget: remaining };
}
