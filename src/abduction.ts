// Abduction — inference to the surviving explanation, for sets of ANY things.
//
// The universal-prediction machine's abductive engine. Given a finite set of
// HYPOTHESES (root causes), each with a prior and a per-FEATURE likelihood
// P(feature | hypothesis), and an OBSERVATION (features present / absent), it
// holds a strictly-positive, NEVER-COLLAPSING posterior over hypotheses, mapped
// by what the evidence rules OUT. It reports the survivors, what was eliminated,
// the next most-discriminating probe, co-occurring hypotheses (multi-root-cause),
// a nested-Monte-Carlo robustness map, and — when a hypothesis crosses a
// threshold — its MITIGATION PROTOCOL (runbook / repair / freeze+2FA /
// remediation / treatment-in-the-dataset).
//
// Same architecture across domains: a binary feature matrix → probability of a
// root cause → trigger a protocol. Instances: medical symptom→condition
// (@a0n/differential), distributed-systems incident triage (alerts→failure→
// runbook), OBD-II vehicle diagnostics (fault codes→fault→repair), financial
// fraud (anomalies→intent→freeze), code/security scanning (findings→defect
// class→remediation).
//
// THE GOD FORMULA is the anti-collapse mechanism: the posterior is the Buleyean
// complement distribution w = R - min(v, R) + 1 (@a0n/buleyean-kernel), whose +1
// sliver guarantees every surviving hypothesis keeps weight >= 1 — collapse is
// structurally impossible (gnosis-math buleyean_positivity / godWeight_pos). The
// only permitted collapse is a hard falsification wall (a near-deterministic
// implication the observation violates). Lean witness for the medical instance:
// gnosis-math/Gnosis/DiseaseAsStandingWaveFactorization.lean.

import { buleyeanWeight } from '@a0n/buleyean-kernel';

// ───────────────────────── types ─────────────────────────

/** A mitigation protocol attached to a hypothesis (runbook, repair, 2FA, …). */
export interface Protocol {
  readonly id: string;
  readonly label: string;
  readonly action?: string;
  readonly meta?: Record<string, unknown>;
}

/** A candidate root cause / explanation. */
export interface Hypothesis {
  readonly id: number;
  readonly name: string;
  /** Base-rate prior P(hypothesis). */
  readonly prior: number;
  /** Optional mitigation protocol surfaced when this hypothesis crosses threshold. */
  readonly protocol?: Protocol;
  /** Arbitrary domain metadata (ICD codes, runbook URL, severity, …). */
  readonly meta?: Record<string, unknown>;
}

/** The hypothesis space: the discrete carrier the differential lives on. */
export interface HypothesisSpace {
  readonly hypotheses: readonly Hypothesis[];
  readonly features: readonly string[];
  readonly size: number;
  /** P(feature present | hypothesis), row-major [hypothesis*nFeatures + feature]. */
  readonly pFeatureGivenHypothesis: Float64Array;
  hypothesisOf(id: number): Hypothesis;
  featureIndex(name: string): number; // -1 if unknown
  pOf(hypothesisId: number, featureIdx: number): number;
}

export type Polarity = 'present' | 'absent';

/** One feature observation's rejection field over the hypothesis space. */
export interface FeatureChannelVerdict {
  readonly feature: string;
  readonly polarity: Polarity;
  readonly weight: number;
  readonly rejection: Float64Array; // length = space.size
  readonly hardWall: boolean[]; // length = space.size; true = hypothesis ruled out
  readonly summary: string;
  readonly active: boolean;
}

/** The observations. Absent features carry information too. */
export interface Observation {
  readonly present: readonly string[];
  readonly absent?: readonly string[];
  readonly ctx?: Record<string, unknown>;
}

export interface Survivor {
  readonly id: number;
  readonly name: string;
  readonly probability: number;
  readonly rejection: number;
  readonly protocol?: Protocol;
}

export interface Eliminated {
  readonly id: number;
  readonly name: string;
  readonly ruledOutBy: { feature: string; polarity: Polarity };
}

export interface NextProbe {
  readonly hypothesisId: number;
  readonly feature: string;
  readonly rulingPolarity: Polarity;
  readonly leverage: number;
  readonly wouldBeHardWall: boolean;
}

export interface Factor {
  readonly hypothesisId: number;
  readonly name: string;
  readonly amplitude: number;
}

/** A fired mitigation protocol: a hypothesis above threshold that carries one. */
export interface ProtocolTrigger {
  readonly hypothesisId: number;
  readonly name: string;
  readonly probability: number;
  readonly protocol: Protocol;
}

export interface AbductivePosterior {
  readonly method: string;
  readonly survivors: Survivor[];
  readonly eliminated: Eliminated[];
  readonly nextProbes: NextProbe[];
  readonly factors: Factor[];
  readonly protocols: ProtocolTrigger[];
  readonly void: {
    readonly entropy: number;
    readonly effectiveSurvivors: number;
    readonly concentration: number;
    readonly survivorCount: number;
    readonly totalHypotheses: number;
  };
  readonly channels: Array<{ feature: string; polarity: Polarity; weight: number; summary: string }>;
}

export interface AbduceOptions {
  /** Sharpening exponent on the god-weight; 1 = pure sliver (max dilation). */
  lambda?: number;
  /** Denied feature with P >= this hard-falsifies a hypothesis. <1 to enable. */
  hardWallHi?: number;
  /** Present feature with P <= this hard-falsifies a hypothesis. >0 to enable. */
  hardWallLo?: number;
  /** How many survivors to compute next-probes for. */
  probeK?: number;
  /** Survivors with probability >= this and a protocol fire a ProtocolTrigger. */
  protocolThreshold?: number;
}

const DEFAULTS = {
  lambda: 1.0,
  hardWallHi: 0.985,
  hardWallLo: 0.0,
  probeK: 5,
  protocolThreshold: 0.8,
};

const EPS = 1e-6;

// ───────────────────────── space construction ─────────────────────────

/** Build a hypothesis space from a likelihood matrix. */
export function buildHypothesisSpace(input: {
  hypotheses: readonly Hypothesis[];
  features: readonly string[];
  /** Row-major P(feature present | hypothesis), length hypotheses*features. */
  pFeatureGivenHypothesis: Float64Array;
}): HypothesisSpace {
  const { hypotheses, features } = input;
  const nF = features.length;
  const p = input.pFeatureGivenHypothesis;
  const featureIdx = new Map<string, number>();
  features.forEach((f, i) => featureIdx.set(normalizeKey(f), i));
  return {
    hypotheses,
    features,
    size: hypotheses.length,
    pFeatureGivenHypothesis: p,
    hypothesisOf(id) {
      const h = hypotheses[id];
      if (!h) throw new Error(`abduction: no hypothesis ${id}`);
      return h;
    },
    featureIndex(name) {
      const i = featureIdx.get(normalizeKey(name));
      return i === undefined ? -1 : i;
    },
    pOf(hid, fIdx) {
      return p[hid * nF + fIdx] ?? 0.5;
    },
  };
}

/**
 * Handles the maybe normalize Key workflow.
 */
export function normalizeKey(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, ' ');
}

// ───────────────────────── channels ─────────────────────────

/**
 * Handles the maybe feature Channel workflow.
 */
export function featureChannel(
  space: HypothesisSpace,
  feature: string,
  polarity: Polarity,
  hardWallHi: number,
  hardWallLo: number,
  weight = 1,
): FeatureChannelVerdict | null {
  const f = space.featureIndex(feature);
  if (f < 0) return null; // unknown feature contributes nothing, never invents
  const n = space.size;
  const rejection = new Float64Array(n);
  const hardWall: boolean[] = new Array(n).fill(false);
  let wallCount = 0;
  for (let h = 0; h < n; h++) {
    const p = clamp(space.pOf(h, f), EPS, 1 - EPS);
    if (polarity === 'present') {
      rejection[h] = -Math.log(p);
      if (hardWallLo > 0 && p <= hardWallLo) { hardWall[h] = true; wallCount++; }
    } else {
      rejection[h] = -Math.log(1 - p);
      if (hardWallHi < 1 && p >= hardWallHi) { hardWall[h] = true; wallCount++; }
    }
  }
  return {
    feature,
    polarity,
    weight,
    rejection,
    hardWall,
    active: true,
    summary:
      polarity === 'present'
        ? `present: rules out hypotheses where near-absent (${wallCount} wall${wallCount === 1 ? '' : 's'})`
        : `absent: rules out hypotheses where near-universal (${wallCount} wall${wallCount === 1 ? '' : 's'})`,
  };
}

/**
 * Handles the maybe prior Channel workflow.
 */
export function priorChannel(space: HypothesisSpace): FeatureChannelVerdict {
  const n = space.size;
  const rejection = new Float64Array(n);
  for (let h = 0; h < n; h++) rejection[h] = -Math.log(clamp(space.hypothesisOf(h).prior, EPS, 1));
  return {
    feature: '(prior)',
    polarity: 'present',
    weight: 1,
    rejection,
    hardWall: new Array(n).fill(false),
    active: true,
    summary: 'base-rate prior over hypotheses',
  };
}

/**
 * Builds the Channels.
 */
export function buildChannels(
  space: HypothesisSpace,
  present: readonly string[],
  absent: readonly string[],
  hardWallHi: number,
  hardWallLo: number,
): FeatureChannelVerdict[] {
  const out: FeatureChannelVerdict[] = [priorChannel(space)];
  for (const f of present) {
    const v = featureChannel(space, f, 'present', hardWallHi, hardWallLo);
    if (v) out.push(v);
  }
  for (const f of absent) {
    const v = featureChannel(space, f, 'absent', hardWallHi, hardWallLo);
    if (v) out.push(v);
  }
  return out;
}

// ───────────────────────── fusion (God Formula) ─────────────────────────

export interface FuseResult {
  readonly posterior: Float64Array;
  readonly rejection: Float64Array;
  readonly walled: boolean[];
}

/**
 * Handles the maybe flatten Channels workflow.
 */
export function flattenChannels(
  verdicts: readonly FeatureChannelVerdict[],
  n: number,
): { active: FeatureChannelVerdict[]; rejectionFlat: Float64Array; weights: Float64Array; hardwallFlat: Uint8Array } {
  const active = verdicts.filter((v) => v.active && v.weight > 0);
  const nCh = active.length;
  const rejectionFlat = new Float64Array(nCh * n);
  const weights = new Float64Array(nCh);
  const hardwallFlat = new Uint8Array(nCh * n);
  for (let ch = 0; ch < nCh; ch++) {
    const v = active[ch]!;
    weights[ch] = v.weight;
    const base = ch * n;
    for (let h = 0; h < n; h++) {
      rejectionFlat[base + h] = v.rejection[h]!;
      hardwallFlat[base + h] = v.hardWall[h] ? 1 : 0;
    }
  }
  return { active, rejectionFlat, weights, hardwallFlat };
}

/**
 * Handles the maybe fuse workflow.
 */
export function fuse(
  space: HypothesisSpace,
  verdicts: readonly FeatureChannelVerdict[],
  lambda = DEFAULTS.lambda,
): FuseResult {
  const n = space.size;
  let posterior: Float64Array;
  let rejection: Float64Array;

  const wasm = getAbductionWasm();
  if (wasm) {
    const { active, rejectionFlat, weights, hardwallFlat } = flattenChannels(verdicts, n);
    const out = wasm.fuse_god_formula(rejectionFlat, weights, hardwallFlat, n, active.length, lambda);
    posterior = out.slice(0, n);
    rejection = out.slice(n, 2 * n);
  } else {
    ({ posterior, rejection } = fuseTs(verdicts, n, lambda));
  }
  const walled: boolean[] = new Array(n);
  for (let h = 0; h < n; h++) walled[h] = posterior[h] === 0;
  return { posterior, rejection, walled };
}

function fuseTs(
  verdicts: readonly FeatureChannelVerdict[],
  n: number,
  lambda: number,
): { posterior: Float64Array; rejection: Float64Array } {
  const rejection = new Float64Array(n);
  const walled: boolean[] = new Array(n).fill(false);
  for (const v of verdicts) {
    if (!v.active || v.weight <= 0) continue;
    for (let h = 0; h < n; h++) {
      rejection[h] = rejection[h]! + v.weight * v.rejection[h]!;
      if (v.hardWall[h]) walled[h] = true;
    }
  }
  let R = 0;
  for (let h = 0; h < n; h++) if (!walled[h] && rejection[h]! > R) R = rejection[h]!;
  const posterior = new Float64Array(n);
  let sum = 0;
  for (let h = 0; h < n; h++) {
    if (walled[h]) continue;
    let w = buleyeanWeight(R, rejection[h]!);
    if (lambda !== 1) w = Math.pow(w, lambda);
    posterior[h] = w;
    sum += w;
  }
  if (sum > 0) for (let h = 0; h < n; h++) posterior[h] = posterior[h]! / sum;
  return { posterior, rejection };
}

// ───────────────────────── next probe (the Sherlock question) ─────────────────────────

/**
 * Handles the maybe next Probes workflow.
 */
export function nextProbes(
  space: HypothesisSpace,
  survivors: readonly Survivor[],
  observed: readonly string[],
  hardWallHi: number,
  hardWallLo: number,
  k = DEFAULTS.probeK,
): NextProbe[] {
  const seen = new Set(observed.map(normalizeKey));
  const out: NextProbe[] = [];
  for (const surv of survivors.slice(0, k)) {
    let best: NextProbe | null = null;
    let bestRank = -1;
    for (let f = 0; f < space.features.length; f++) {
      const name = space.features[f]!;
      if (seen.has(normalizeKey(name))) continue;
      const p = space.pOf(surv.id, f);
      // The hallmark question: a feature this hypothesis near-always shows. If it
      // is ABSENT, the hypothesis drops sharply (hard-walled when >= hardWallHi).
      if (p < 0.5) continue;
      const wouldBeHardWall = hardWallHi < 1 && p >= hardWallHi;
      const leverage = p;
      const rank = (wouldBeHardWall ? 1 : 0) * 10 + leverage;
      if (rank > bestRank) {
        bestRank = rank;
        best = { hypothesisId: surv.id, feature: name, rulingPolarity: 'absent', leverage, wouldBeHardWall };
      }
    }
    if (!best) {
      let lowP = Infinity;
      let lowName = '';
      for (let f = 0; f < space.features.length; f++) {
        const name = space.features[f]!;
        if (seen.has(normalizeKey(name))) continue;
        const p = space.pOf(surv.id, f);
        if (p < lowP) { lowP = p; lowName = name; }
      }
      if (lowName) {
        best = {
          hypothesisId: surv.id,
          feature: lowName,
          rulingPolarity: 'present',
          leverage: 1 - lowP,
          wouldBeHardWall: hardWallLo > 0 && lowP <= hardWallLo,
        };
      }
    }
    if (best) out.push(best);
  }
  return out;
}

// ───────────────────────── factorization (multi-root-cause) ─────────────────────────

const COVER_P = 0.5;
const MARGINAL_MIN = 0.15;
const MAX_FACTORS = 4;

/**
 * Handles the maybe factorize workflow.
 */
export function factorize(
  space: HypothesisSpace,
  survivors: readonly Survivor[],
  present: readonly string[],
): Factor[] {
  const presentIdx: number[] = [];
  for (const f of present) {
    const i = space.featureIndex(normalizeKey(f));
    if (i >= 0) presentIdx.push(i);
  }
  if (presentIdx.length === 0 || survivors.length === 0) return [];
  const uncovered = new Set(presentIdx);
  const used = new Set<number>();
  const factors: Factor[] = [];
  const total = presentIdx.length;
  while (uncovered.size > 0 && factors.length < MAX_FACTORS) {
    let bestSurv: Survivor | null = null;
    let bestScore = 0;
    let bestCovers: number[] = [];
    for (const surv of survivors) {
      if (used.has(surv.id)) continue;
      let mass = 0;
      const covers: number[] = [];
      for (const f of uncovered) {
        const p = space.pOf(surv.id, f);
        mass += p;
        if (p >= COVER_P) covers.push(f);
      }
      const score = mass * (0.5 + surv.probability);
      if (score > bestScore) { bestScore = score; bestSurv = surv; bestCovers = covers; }
    }
    if (!bestSurv) break;
    const marginal = bestCovers.length / total;
    if (factors.length > 0 && marginal < MARGINAL_MIN) break;
    factors.push({ hypothesisId: bestSurv.id, name: bestSurv.name, amplitude: clamp01((bestCovers.length || 1) / total) });
    used.add(bestSurv.id);
    if (bestCovers.length === 0) break;
    for (const f of bestCovers) uncovered.delete(f);
  }
  return factors;
}

// ───────────────────────── the engine ─────────────────────────

export class Abduction {
  readonly space: HypothesisSpace;
  private readonly opts: Required<AbduceOptions>;
  constructor(space: HypothesisSpace, opts: AbduceOptions = {}) {
    this.space = space;
    this.opts = { ...DEFAULTS, ...stripUndefined(opts) };
  }

  abduce(obs: Observation, override: AbduceOptions = {}): AbductivePosterior {
    const o = { ...this.opts, ...stripUndefined(override) };
    const present = obs.present ?? [];
    const absent = obs.absent ?? [];
    const channels = buildChannels(this.space, present, absent, o.hardWallHi, o.hardWallLo);
    const fused = fuse(this.space, channels, o.lambda);
    const { survivors, eliminated } = buildSurvivorsEliminated(this.space, channels, fused);
    const observed = [...present, ...absent];
    const protocols: ProtocolTrigger[] = [];
    for (const s of survivors) {
      if (s.probability >= o.protocolThreshold && s.protocol) {
        protocols.push({ hypothesisId: s.id, name: s.name, probability: s.probability, protocol: s.protocol });
      }
    }
    return {
      method: 'abduction_buleyean_god_formula_v1',
      survivors,
      eliminated,
      nextProbes: nextProbes(this.space, survivors, observed, o.hardWallHi, o.hardWallLo, o.probeK),
      factors: factorize(this.space, survivors, present),
      protocols,
      void: voidMetrics(survivors, this.space.size),
      channels: channels
        .filter((c) => c.feature !== '(prior)')
        .map((c) => ({ feature: c.feature, polarity: c.polarity, weight: c.weight, summary: c.summary })),
    };
  }

  mapVoid(obs: Observation, opts: Partial<VoidMapOptions> = {}): VoidMap {
    return mapVoid(this.space, obs.present ?? [], obs.absent ?? [], {
      hardWallHi: this.opts.hardWallHi,
      hardWallLo: this.opts.hardWallLo,
      lambda: this.opts.lambda,
      ...stripUndefined(opts),
    });
  }
}

/**
 * Builds the Abduction.
 */
export function buildAbduction(
  input: Parameters<typeof buildHypothesisSpace>[0],
  opts?: AbduceOptions,
): Abduction {
  return new Abduction(buildHypothesisSpace(input), opts);
}

function buildSurvivorsEliminated(
  space: HypothesisSpace,
  verdicts: readonly FeatureChannelVerdict[],
  fused: FuseResult,
): { survivors: Survivor[]; eliminated: Eliminated[] } {
  const survivors: Survivor[] = [];
  const eliminated: Eliminated[] = [];
  for (let h = 0; h < space.size; h++) {
    const hyp = space.hypothesisOf(h);
    if (fused.walled[h]) {
      let by: { feature: string; polarity: Polarity } = { feature: '(unknown)', polarity: 'absent' };
      for (const v of verdicts) {
        if (v.active && v.weight > 0 && v.hardWall[h]) { by = { feature: v.feature, polarity: v.polarity }; break; }
      }
      eliminated.push({ id: h, name: hyp.name, ruledOutBy: by });
    } else {
      survivors.push({
        id: h,
        name: hyp.name,
        probability: fused.posterior[h]!,
        rejection: fused.rejection[h]!,
        ...(hyp.protocol ? { protocol: hyp.protocol } : {}),
      });
    }
  }
  survivors.sort((a, b) => b.probability - a.probability);
  return { survivors, eliminated };
}

/**
 * Handles the maybe void Metrics workflow.
 */
export function voidMetrics(survivors: Survivor[], totalHypotheses: number): AbductivePosterior['void'] {
  let entropy = 0;
  for (const s of survivors) if (s.probability > 0) entropy -= s.probability * Math.log(s.probability);
  const maxEntropy = survivors.length > 1 ? Math.log(survivors.length) : 1;
  const concentration = maxEntropy > 0 ? 1 - entropy / maxEntropy : 1;
  return {
    entropy,
    effectiveSurvivors: Math.exp(entropy),
    concentration,
    survivorCount: survivors.length,
    totalHypotheses,
  };
}

// ───────────────────────── nested Monte Carlo (map the void) ─────────────────────────

export interface VoidCell {
  readonly id: number;
  readonly name: string;
  readonly mass: number;
  readonly rankStability: number;
  readonly survival: number;
}
export interface VoidMap {
  readonly method: string;
  readonly cells: VoidCell[];
  readonly meanEntropy: number;
  readonly outerSamples: number;
  readonly innerSamples: number;
}
export interface VoidMapOptions {
  outerSamples?: number;
  innerSamples?: number;
  jitter?: number;
  seed?: number;
  lambda?: number;
  hardWallHi: number;
  hardWallLo: number;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Handles the maybe map Void workflow.
 */
export function mapVoid(
  space: HypothesisSpace,
  present: readonly string[],
  absent: readonly string[],
  opts: VoidMapOptions,
): VoidMap {
  const outerSamples = opts.outerSamples ?? 256;
  const innerSamples = opts.innerSamples ?? 64;
  const jitter = opts.jitter ?? 0.35;
  const seed = opts.seed ?? 0x9e3779b9;
  const lambda = opts.lambda ?? 1.0;
  const n = space.size;
  const base = buildChannels(space, present, absent, opts.hardWallHi, opts.hardWallLo);

  const wasm = getAbductionWasm();
  if (wasm) {
    const { active, rejectionFlat, weights, hardwallFlat } = flattenChannels(base, n);
    const perturbable = new Uint8Array(active.length);
    for (let i = 0; i < active.length; i++) perturbable[i] = active[i]!.feature === '(prior)' ? 0 : 1;
    const out = wasm.map_void(
      rejectionFlat, weights, hardwallFlat, perturbable,
      n, active.length, outerSamples, innerSamples, jitter, lambda, seed >>> 0,
    );
    const totalDraws = outerSamples * innerSamples;
    const cells: VoidCell[] = [];
    for (let h = 0; h < n; h++) {
      const mass = out[h]!; const top = out[n + h]!; const surv = out[2 * n + h]!;
      if (mass === 0 && surv === 0) continue;
      cells.push({
        id: h, name: space.hypothesisOf(h).name,
        mass: totalDraws > 0 ? mass / totalDraws : 0,
        rankStability: top / outerSamples, survival: surv / outerSamples,
      });
    }
    cells.sort((a, b) => b.mass - a.mass);
    return { method: 'abduction_void_map_nested_mc_wasm_v1', cells, meanEntropy: out[3 * n]! / outerSamples, outerSamples, innerSamples };
  }

  const rng = mulberry32(seed);
  const massTally = new Float64Array(n);
  const topTally = new Float64Array(n);
  const survivalTally = new Float64Array(n);
  let entropySum = 0;
  let totalDraws = 0;
  for (let o = 0; o < outerSamples; o++) {
    const perturbed: FeatureChannelVerdict[] = base.map((v) =>
      v.feature === '(prior)' ? v : { ...v, weight: Math.max(0, 1 - jitter * rng()) },
    );
    const fused = fuse(space, perturbed, lambda);
    const { survivors } = buildSurvivorsEliminated(space, perturbed, fused);
    for (const s of survivors) survivalTally[s.id] = survivalTally[s.id]! + 1;
    for (const s of survivors.slice(0, 3)) topTally[s.id] = topTally[s.id]! + 1;
    let H = 0;
    for (const s of survivors) if (s.probability > 0) H -= s.probability * Math.log(s.probability);
    entropySum += H;
    for (let i = 0; i < innerSamples; i++) {
      let r = rng();
      let pick = survivors[0]?.id ?? 0;
      for (const s of survivors) { r -= s.probability; if (r <= 0) { pick = s.id; break; } }
      massTally[pick] = massTally[pick]! + 1;
      totalDraws += 1;
    }
  }
  const cells: VoidCell[] = [];
  for (let h = 0; h < n; h++) {
    if (massTally[h] === 0 && survivalTally[h] === 0) continue;
    cells.push({
      id: h, name: space.hypothesisOf(h).name,
      mass: totalDraws > 0 ? massTally[h]! / totalDraws : 0,
      rankStability: topTally[h]! / outerSamples, survival: survivalTally[h]! / outerSamples,
    });
  }
  cells.sort((a, b) => b.mass - a.mass);
  return { method: 'abduction_void_map_nested_mc_v1', cells, meanEntropy: entropySum / outerSamples, outerSamples, innerSamples };
}

// ───────────────────────── WASM kernel (injected by the consumer) ─────────────────────────

export interface AbductionWasm {
  fuse_god_formula(
    rejection_flat: Float64Array, weights: Float64Array, hardwall_flat: Uint8Array,
    n: number, n_channels: number, lambda: number,
  ): Float64Array;
  map_void(
    base_rejection_flat: Float64Array, base_weights: Float64Array, base_hardwall_flat: Uint8Array,
    perturbable: Uint8Array, n: number, n_channels: number, outer: number, inner: number,
    jitter: number, lambda: number, seed: number,
  ): Float64Array;
}

let wasmKernel: AbductionWasm | null = null;

/**
 * Register a WASM fusion kernel (the gnosis-chaos-wasm-style Rust kernel). The
 * consumer owns the .wasm artifact and inits it, then hands the live exports
 * here. The math is bit-for-bit identical to the pure-TS path (same God Formula,
 * same mulberry32), so results are independent of which path runs.
 */
export function setAbductionWasm(kernel: AbductionWasm | null): void {
  wasmKernel = kernel;
}
/**
 * Handles the maybe get Abduction Wasm workflow.
 */
export function getAbductionWasm(): AbductionWasm | null {
  return wasmKernel;
}
/**
 * Returns whether is Abduction Wasm Ready is true.
 */
export function isAbductionWasmReady(): boolean {
  return wasmKernel !== null;
}

// ───────────────────────── helpers ─────────────────────────

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
function stripUndefined<T extends object>(o: T): Partial<T> {
  const out: Partial<T> = {};
  for (const k in o) if (o[k] !== undefined) out[k] = o[k];
  return out;
}
