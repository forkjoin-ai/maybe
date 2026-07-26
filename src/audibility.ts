// Audibility — can this evidence separate these explanations, even in principle?
//
// `abduction.ts` infers the surviving explanation FROM an observation, and the god formula
// guarantees no surviving hypothesis is ever crushed to zero by evidence (never-collapse).
// This module asks the prior question, about the evidence rather than the inference:
//
//   Do two hypotheses have the same observable signature?
//
// If they do, no posterior — however careful — will separate them, and `nextProbe` will hunt
// forever without discriminating. The collapse is forced by the SURFACE, not chosen by the
// reasoner. Never-collapse protects against over-confident inference; audibility protects
// against un-informative observation. They are complements, and only one of them is about
// the engine.
//
//   Silence is not the absence of a verdict.
//   It is a verdict that never reached the observer.
//
// Lean witness: `gnosis-math/Gnosis/AudibleRejection.lean` (Init-only, ZERO axioms) proves
// the shape — `Audible` / `FullyAudible` over a three-cell verdict, that a two-valued
// observation cannot be fully audible over three cells (`boolCollapse_not_fullyAudible`),
// and the soundness asymmetry (`unsound_silence_admits_false_accept`): a mirror that ACCEPTS
// where the oracle DECLINES is unrecoverable, while the reverse only costs a recomputation.
//
// Why this lives next to abduction rather than in a diagnostics package: it is a property of
// the hypothesis space itself, so every instance inherits it — medical symptom→condition
// (@a0n/differential), incident triage, OBD-II, fraud, code scanning. An indistinguishable
// pair in any of those is a differential that can never be closed by observation, and saying
// so up front beats discovering it after a thousand probes.

import type { HypothesisSpace, Hypothesis } from './abduction.js';

/**
 * How close two per-feature likelihoods must be to count as indistinguishable.
 *
 * Not zero: likelihoods are floats and are usually elicited or fitted, so exact equality
 * would under-report real collisions. Deliberately tight — a pair separated only by 1e-6 is
 * separable in arithmetic and not in practice.
 */
export const DEFAULT_AUDIBILITY_EPSILON = 1e-6;

/** Two hypotheses that no observation over the current feature set can tell apart. */
export interface IndistinguishablePair {
  readonly a: Hypothesis;
  readonly b: Hypothesis;
  /** Largest per-feature likelihood gap found; <= epsilon by construction. */
  readonly maxGap: number;
  /** The feature that came closest to separating them, if any feature differs at all. */
  readonly closestFeature: string | null;
}

/** A feature that discriminates nothing: identical likelihood across every hypothesis. */
export interface InertFeature {
  readonly feature: string;
  readonly spread: number;
}

export interface AudibilityReport {
  /** True when every pair of hypotheses differs somewhere by more than epsilon. */
  readonly audible: boolean;
  /** Pairs no probe can ever separate. Empty iff `audible`. */
  readonly indistinguishable: readonly IndistinguishablePair[];
  /**
   * Features whose likelihood is flat across the whole space. Observing them costs a probe
   * and buys nothing — worth surfacing separately because they are cheap to remove, whereas
   * an indistinguishable pair needs a NEW feature, not a deleted one.
   */
  readonly inertFeatures: readonly InertFeature[];
  /**
   * The smallest separation anywhere in the space. A tiny value that still exceeds epsilon is
   * a warning rather than a failure: the space is technically audible and practically fragile.
   */
  readonly minSeparation: number;
  readonly epsilon: number;
}

/**
 * Decide whether a hypothesis space can be discriminated by its own features.
 *
 * O(n² · f). Intended for design time and for a start-up assertion, not per-observation.
 */
export function auditAudibility(
  space: HypothesisSpace,
  epsilon: number = DEFAULT_AUDIBILITY_EPSILON
): AudibilityReport {
  const n = space.size;
  const features = space.features;
  const nFeatures = features.length;

  const indistinguishable: IndistinguishablePair[] = [];
  let minSeparation = Number.POSITIVE_INFINITY;

  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      let maxGap = 0;
      let closestFeature: string | null = null;

      for (let f = 0; f < nFeatures; f += 1) {
        const gap = Math.abs(
          space.pFeatureGivenHypothesis[i * nFeatures + f]! -
            space.pFeatureGivenHypothesis[j * nFeatures + f]!
        );
        if (gap > maxGap) {
          maxGap = gap;
          closestFeature = features[f]!;
        }
      }

      if (maxGap < minSeparation) minSeparation = maxGap;

      if (maxGap <= epsilon) {
        indistinguishable.push({
          a: space.hypothesisOf(space.hypotheses[i]!.id),
          b: space.hypothesisOf(space.hypotheses[j]!.id),
          maxGap,
          closestFeature,
        });
      }
    }
  }

  const inertFeatures: InertFeature[] = [];
  for (let f = 0; f < nFeatures; f += 1) {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i += 1) {
      const p = space.pFeatureGivenHypothesis[i * nFeatures + f]!;
      if (p < lo) lo = p;
      if (p > hi) hi = p;
    }
    const spread = n === 0 ? 0 : hi - lo;
    if (spread <= epsilon) {
      inertFeatures.push({ feature: features[f]!, spread });
    }
  }

  return {
    audible: indistinguishable.length === 0,
    indistinguishable,
    inertFeatures,
    minSeparation: Number.isFinite(minSeparation) ? minSeparation : 0,
    epsilon,
  };
}

/**
 * Throw unless the space is audible.
 *
 * For callers who would rather fail at construction than ship a differential that cannot be
 * closed. The message names the colliding pairs, because "this space is inaudible" without
 * saying which hypotheses collide is itself a silent failure.
 */
export function assertAudible(
  space: HypothesisSpace,
  epsilon: number = DEFAULT_AUDIBILITY_EPSILON
): void {
  const report = auditAudibility(space, epsilon);
  if (report.audible) return;
  const pairs = report.indistinguishable
    .map((p) => `${p.a.name} / ${p.b.name} (max gap ${p.maxGap.toExponential(2)})`)
    .join('; ');
  throw new Error(
    `hypothesis space is inaudible: no observation over these features can separate ` +
      `${report.indistinguishable.length} pair(s) — ${pairs}. ` +
      `Add a discriminating feature; removing inert ones will not help.`
  );
}

/**
 * Whether a specific pair is separable, and by which feature.
 *
 * `separator` is the feature that best distinguishes them — the probe worth spending next.
 * When `separable` is false there is no such probe, and that is the answer rather than a
 * failure to find one.
 */
export function separability(
  space: HypothesisSpace,
  hypothesisIdA: number,
  hypothesisIdB: number,
  epsilon: number = DEFAULT_AUDIBILITY_EPSILON
): { separable: boolean; separator: string | null; gap: number } {
  const features = space.features;
  const nFeatures = features.length;
  const indexOf = (id: number): number =>
    space.hypotheses.findIndex((h) => h.id === id);

  const i = indexOf(hypothesisIdA);
  const j = indexOf(hypothesisIdB);
  if (i < 0 || j < 0) {
    throw new Error(
      `separability: unknown hypothesis id ${i < 0 ? hypothesisIdA : hypothesisIdB}`
    );
  }

  let gap = 0;
  let separator: string | null = null;
  for (let f = 0; f < nFeatures; f += 1) {
    const d = Math.abs(
      space.pFeatureGivenHypothesis[i * nFeatures + f]! -
        space.pFeatureGivenHypothesis[j * nFeatures + f]!
    );
    if (d > gap) {
      gap = d;
      separator = features[f]!;
    }
  }

  const separable = gap > epsilon;
  return { separable, separator: separable ? separator : null, gap };
}
