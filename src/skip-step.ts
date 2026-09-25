/**
 * skip-step.ts -- Predictability of the Bule cost algebra, for runtime skip-step.
 *
 * Given a carrier (waste, opportunity, diversity) and a PARTIAL observation of
 * it, when is a downstream step's output already determined? This mirrors
 * Gnosis.Predictability (open-source/gnosis-math) and is meant for the polyglot
 * interpreter: a FOLD step whose output is fixed by the conserved charges
 * already on the stack can be SKIPPED -- take the predicted value instead of
 * re-running the step.
 *
 * The boundary:
 *   - future (act) from waste + residue : determined (skip)
 *   - future from waste alone           : blind (cannot skip)
 *   - past from the future alone        : free (cannot skip; the inversion orbit)
 *   - past from future + opportunity    : determined (skip)
 *
 * A partial predicts a target exactly when it refines the target's fibres; the
 * separating collision is the witness that it does not. This is the finite
 * data-processing boundary: what the channel destroys is what it cannot predict.
 */

export interface BuleCarrier {
  waste: number;
  opportunity: number;
  diversity: number;
}

export type PredictabilityClass = 'determined' | 'free' | 'blind';

/** The invisible residue: opportunity plus diversity. */
export function residue(b: BuleCarrier): number {
  return b.opportunity + b.diversity;
}

/** The two conserved charges (waste, residue). */
export function charges(b: BuleCarrier): readonly [number, number] {
  return [b.waste, residue(b)];
}

/** The causal map: the future of a carrier (opportunity spent into diversity). */
export function act(b: BuleCarrier): BuleCarrier {
  return { waste: b.waste, opportunity: 0, diversity: residue(b) };
}

export interface SkipDecision {
  class: PredictabilityClass;
  /** True when the target is fully determined and may be read from value. */
  skip: boolean;
  /** The recovered value when skip is true. */
  value?: BuleCarrier;
  reason: string;
}

/**
 * Decide whether the future (act) can be skipped from a partial observation.
 * It is determined exactly when both waste and residue are known.
 */
export function skipFuture(known: Partial<BuleCarrier>): SkipDecision {
  const hasWaste = known.waste !== undefined;
  const hasResidue = known.opportunity !== undefined && known.diversity !== undefined;
  if (hasWaste && hasResidue) {
    return {
      class: 'determined',
      skip: true,
      value: { waste: known.waste as number, opportunity: 0, diversity: (known.opportunity as number) + (known.diversity as number) },
      reason: 'waste and residue determine act',
    };
  }
  if (!hasWaste && !hasResidue) {
    return { class: 'blind', skip: false, reason: 'no conserved charge known' };
  }
  return { class: 'free', skip: false, reason: 'act needs both waste and residue' };
}

/**
 * Decide whether the past can be skipped from an observed future plus one gauge
 * coordinate. Without the gauge it is free (the inversion orbit); with it the
 * past is recovered exactly.
 */
export function skipPast(future: BuleCarrier, opportunity?: number): SkipDecision {
  if (opportunity === undefined) {
    return {
      class: 'free',
      skip: false,
      reason: 'the reversible phase is free without a gauge coordinate',
    };
  }
  if (opportunity > future.diversity) {
    return {
      class: 'blind',
      skip: false,
      reason: 'opportunity exceeds the terminal diversity: no such past',
    };
  }
  return {
    class: 'determined',
    skip: true,
    value: { waste: future.waste, opportunity, diversity: future.diversity - opportunity },
    reason: 'future plus opportunity recovers the past',
  };
}

/**
 * The generalized gap witness: find a collision of obs that target separates.
 * A non-null result proves no predictor can exist, so the step may not be
 * skipped from that observation. O(|sample|^2); call at compile/bake time.
 */
export function separatingCollision<A, B, C>(
  obs: (a: A) => B,
  target: (a: A) => C,
  sample: readonly A[],
): { left: A; right: A } | null {
  for (let i = 0; i < sample.length; i++) {
    for (let j = i + 1; j < sample.length; j++) {
      if (obs(sample[i]) === obs(sample[j]) && target(sample[i]) !== target(sample[j])) {
        return { left: sample[i], right: sample[j] };
      }
    }
  }
  return null;
}
