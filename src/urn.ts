/**
 * urn.ts -- the classic urn read through the Buleyean lens.
 *
 * A finite urn holds counts[i] observations in bucket i, with N = sum counts
 * draws in total. Counting every draw outside bucket i as a rejection of i
 * gives v_i = N - counts[i]. The clamped God Formula
 *
 *   w_i = N - min(v_i, N) + 1
 *
 * then collapses exactly onto counts[i] + 1, which is the Laplace numerator.
 * Normalizing those weights therefore reproduces Laplace's rule of succession
 * with no separate smoothing step: the +1 that keeps every weight alive is the
 * same +1 that smooths the count.
 *
 * The integer backbone of that equality is checked exactly. The posterior and
 * frequency helpers return display floats only and make no truth claim; the
 * exact claim lives in assertUrnLaplaceIdentity.
 */

/** A finite multiset of observations: one non-negative count per bucket. */
export interface Urn {
  readonly counts: readonly number[];
}

/** N = sum_i counts[i], the number of draws in the urn. */
export function urnTotal(counts: readonly number[]): number {
  let total = 0;
  for (const count of counts) total += count;
  return total;
}

/** v_i = N - counts[i]: the draws that rejected bucket i. */
export function urnRejections(counts: readonly number[]): number[] {
  const total = urnTotal(counts);
  return counts.map((count) => total - count);
}

/**
 * The clamped God Formula N - min(v_i, N) + 1 on urn counts.
 * Because v_i = N - counts[i] and counts[i] >= 0, this is counts[i] + 1.
 */
export function buleyeanWeightFromUrn(counts: readonly number[], i: number): number {
  const total = urnTotal(counts);
  const rejections = total - counts[i]!;
  return total - Math.min(rejections, total) + 1;
}

/** The Laplace posterior (counts[i] + 1) / (N + K) as display floats. */
export function laplacePosterior(counts: readonly number[]): number[] {
  const total = urnTotal(counts);
  const buckets = counts.length;
  return counts.map((count) => (count + 1) / (total + buckets));
}

/** The normalized Buleyean weights w_i / sum_j w_j as display floats. */
export function buleyeanPosteriorFromUrn(counts: readonly number[]): number[] {
  const weights = counts.map((count) => count + 1);
  let sum = 0;
  for (const weight of weights) sum += weight;
  return weights.map((weight) => weight / sum);
}

/** Empirical frequency counts[i] / N; null for every bucket when N = 0. */
export function empiricalFrequency(counts: readonly number[]): Array<number | null> {
  const total = urnTotal(counts);
  return counts.map((count) => (total === 0 ? null : count / total));
}

/** The exact integer reading of the urn/Laplace identity. */
export interface UrnLaplaceIdentity {
  readonly ok: boolean;
  readonly detail: string;
}

/**
 * Check the integer backbone of the urn/Laplace identity with no
 * floating-point truth claim:
 *
 *   sum_i (counts[i] + 1) === N + K   and   w_i === counts[i] + 1 for every i.
 *
 * Passing means the Buleyean normalization and the Laplace posterior share the
 * exact numerator counts[i] + 1 and the exact denominator N + K.
 */
export function assertUrnLaplaceIdentity(counts: readonly number[]): UrnLaplaceIdentity {
  const total = urnTotal(counts);
  const buckets = counts.length;
  let weightSum = 0;

  for (let i = 0; i < buckets; i++) {
    const weight = buleyeanWeightFromUrn(counts, i);
    const expected = counts[i]! + 1;
    if (weight !== expected) {
      return {
        ok: false,
        detail: `weight w_${i} = ${weight} !== counts[${i}] + 1 = ${expected}`,
      };
    }
    weightSum += weight;
  }

  if (weightSum !== total + buckets) {
    return {
      ok: false,
      detail: `sum_i (counts[i] + 1) = ${weightSum} !== N + K = ${total + buckets}`,
    };
  }

  return {
    ok: true,
    detail: `w_i = counts[i] + 1 for ${buckets} buckets; sum_i (counts[i] + 1) = ${weightSum} = N + K = ${total + buckets}`,
  };
}
