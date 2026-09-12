import { buildAbduction, normalizeKey } from './abduction';

export interface MemoryPathCandidate {
  id: string;
  strength: number;
  /** Heuristic likelihoods in [0, 1], not empirical confidence. */
  signals: Record<string, number>;
}

export interface MemoryAbductionInput {
  candidates: MemoryPathCandidate[];
  observed: string[];
  probeFeatures?: string[];
}

export interface MemoryAbductionResult {
  ranked: Array<{ id: string; weight: number; rejection: number }>;
  probes: Array<{ id: string; feature: string }>;
  /** Equal likelihood vectors on observed evidence; priors may still differ. */
  indistinguishable: string[][];
}

function featuresOf(values: string[]): string[] {
  const result: string[] = [];
  for (const value of values) {
    if (typeof value !== 'string' || !normalizeKey(value)) {
      throw new Error('memory abduction: feature names must be nonempty strings');
    }
    const key = normalizeKey(value);
    if (!result.includes(key)) result.push(key);
  }
  return result;
}

/** Pure, bounded adapter. The caller owns relevance admission and local probes. */
export function rankMemoryPaths(input: MemoryAbductionInput): MemoryAbductionResult {
  if (!Array.isArray(input.candidates) || input.candidates.length > 64) {
    throw new Error('memory abduction: at most 64 candidates are admitted');
  }
  const observed = featuresOf(input.observed);
  const requestedProbes = featuresOf(input.probeFeatures ?? []).filter(f => !observed.includes(f));
  if (observed.length + requestedProbes.length > 128) {
    throw new Error('memory abduction: at most 128 features are admitted');
  }
  const ids = new Set<string>();
  let maxStrength = 0;
  const rows = input.candidates.map(candidate => {
    if (typeof candidate.id !== 'string' || !candidate.id.trim() || ids.has(candidate.id)) {
      throw new Error('memory abduction: candidate IDs must be unique nonempty strings');
    }
    ids.add(candidate.id);
    if (!Number.isFinite(candidate.strength) || candidate.strength <= 0) {
      throw new Error('memory abduction: strengths must be finite and positive');
    }
    maxStrength = Math.max(maxStrength, candidate.strength);
    const row: Record<string, number> = {};
    for (const name of Object.keys(candidate.signals)) {
      const value = candidate.signals[name]!;
      const key = normalizeKey(name);
      if (!key || row['$' + key] !== undefined || !Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error('memory abduction: signals must have distinct names and finite values in [0, 1]');
      }
      row['$' + key] = value;
    }
    return row;
  });
  const indistinguishable: string[][] = [];
  const groups = new Map<string, string[]>();
  for (let i = 0; i < rows.length; i++) {
    const signature = JSON.stringify(observed.map(f => rows[i]?.['$' + f] ?? 0.5));
    const group = groups.get(signature) ?? [];
    groups.set(signature, group.concat([input.candidates[i]!.id]));
  }
  for (const group of groups.values()) if (group.length > 1) indistinguishable.push(group);
  if (rows.length === 0 || observed.length === 0) return { ranked: [], probes: [], indistinguishable };

  // Offer only probes whose likelihoods can distinguish at least two paths.
  const discriminating: string[] = [];
  for (const feature of requestedProbes) {
    const first = rows[0]?.['$' + feature] ?? 0.5;
    for (let i = 1; i < rows.length; i++) {
      const other = rows[i]?.['$' + feature] ?? 0.5;
      if (other !== first) {
        discriminating.push(feature);
        break;
      }
    }
  }
  const features = [...observed, ...discriminating];
  const matrix = new Float64Array(rows.length * features.length);
  for (let i = 0; i < rows.length; i++) {
    for (let f = 0; f < features.length; f++) matrix[i * features.length + f] = rows[i]?.['$' + features[f]!] ?? 0.5;
  }
  // Scale before summing to avoid overflow for large, otherwise valid strengths.
  const scaled = input.candidates.map(candidate => candidate.strength / maxStrength);
  const total = scaled.reduce((sum, value) => sum + value, 0);
  const engine = buildAbduction({
    hypotheses: input.candidates.map((candidate, id) => ({ id, name: candidate.id, prior: scaled[id]! / total })),
    features,
    pFeatureGivenHypothesis: matrix,
  }, { hardWallHi: 1, hardWallLo: 0, probeK: 3 });
  const posterior = engine.abduce({ present: observed });
  return {
    ranked: posterior.survivors.map(item => ({ id: item.name, weight: item.probability, rejection: item.rejection })),
    probes: posterior.nextProbes.slice(0, 3).map(probe => ({ id: input.candidates[probe.hypothesisId]!.id, feature: probe.feature })),
    indistinguishable,
  };
}

/** JSON entrypoint for Moonshine's embedded Gnosis evaluator. */
export function app(input: MemoryAbductionInput): MemoryAbductionResult {
  return rankMemoryPaths(input);
}
