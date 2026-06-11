/**
 * Charisma / popularity / attention ranking on top of the Buleyean sliver.
 *
 * Takes any finite set of items and two scalar fields -- `advance` (the
 * charisma-like gain scalar, analog of the Buleyean "rounds observed") and
 * `resist` (the opposition / cost scalar, analog of "rejections") -- and
 * produces a total order by Nat-floored reception with a Buleyean upper
 * bound and an optional saturation cap.
 *
 * ## Correspondence to the seven laws
 *
 * Every ranked record satisfies the Buleyean sandwich
 * `pessimistic ≤ actual ≤ buleyeanPredict ≤ optimistic`. In particular:
 *
 * * **Law 1 (Impossibility of zero):** `buleyeanPredict ≥ baseline` always.
 *   The predictor can never fall below baseline, even under total
 *   resistance -- the +1 sliver guarantees the "always one seat" floor.
 * * **Law 2 (Strict ordering):** higher `advance` at equal `resist` always
 *   outranks; lower `resist` at equal `advance` always outranks. This is
 *   the Buleyean monotonicity axiom applied to popularity.
 * * **Law 3 (Universal sandwich):** `popularity ∈ [pessimistic, optimistic]`
 *   and `buleyeanPredict ∈ [baseline, baseline + advance]`.
 *
 * ## Runtime mirror
 *
 * Mirror of the Init-only Lean stack at
 * `open-source/gnosis/lean/Lean/ForkRaceFoldTheorems/Charisma*.lean`:
 *
 * * `charismaPopularity` ↔ `CharismaRanking.popularity`
 * * `buleyeanPredict`    ↔ `CharismaSandwich.buleyeanPredict`
 * * `saturatedPopularity` ↔ `CharismaRanking.saturatedPopularity`
 * * `gain` / `loss`       ↔ `CharismaProspectTheory.gain` / `loss`
 * * `prospectWeight`      ↔ `CharismaProspectTheory.prospectWeight`
 *
 * ## Use cases
 *
 * * **LLM model ranking.** advance = quality score; resist = $/token.
 *   Rank by `popularity`; celebrity plateau flags models past saturation.
 * * **Mesh attention verification.** advance = diagonal mass; resist =
 *   unobserved-leakage. Rank mesh columns by absorbed attention.
 * * **Social attention / popularity.** advance = signal strength; resist =
 *   audience resistance; baseline = bare-attention orienting reflex.
 *
 * The combinator is substrate-neutral. No normalization, no probability
 * baggage, no units. Works over any finite scalar (Number / BigInt encoded
 * as Number / rationals encoded as Number).
 *
 * Pure TS. No Node, no fs, no DOM. Safe in Cloudflare Workers, Bun, Deno,
 * browsers, Node.
 */

import { buleyeanWeight as _buleyeanWeight } from '@a0n/buleyean-kernel';

/** Input to the ranker: items, two scalar fields, optional baseline and cap. */
export interface CharismaRankInput<X> {
  readonly items: readonly X[];
  readonly advance: (x: X) => number;
  readonly resist: (x: X) => number;
  /** Reference point (Prospect Theory status-quo). Default 0. */
  readonly baseline?: number;
  /** Saturation cap (celebrity plateau threshold). Default `Infinity`. */
  readonly cap?: number;
  /** Secondary sort key on ties; default falls back to `advance`. */
  readonly tieBreaker?: (x: X) => number;
}

/** A scored + ranked item with the full sandwich exposed. */
export interface CharismaRanked<X> {
  readonly item: X;
  /** Zero-indexed rank; 0 = top. */
  readonly rank: number;
  /** Nat-floored actual reception. */
  readonly popularity: number;
  /** Buleyean predictor; always ≥ baseline (Law 1). */
  readonly buleyeanPredict: number;
  /** `min(popularity, cap)` -- the "visible" ranking value. */
  readonly saturated: number;
  readonly gain: number;
  readonly loss: number;
  readonly saturatedAtCap: boolean;
  readonly silenced: boolean;
  readonly advance: number;
  readonly resist: number;
}

/** Nat-floored reception. Lean: `CharismaRanking.popularity`. */
export function charismaPopularity(
  baseline: number,
  advance: number,
  resist: number
): number {
  return Math.max(0, baseline + advance - resist);
}

/**
 * Buleyean predictive measure = `baseline + advance - min(resist, advance)`.
 * Equivalently: `baseline - 1 + buleyeanWeight(advance, resist)`.
 * Always ≥ baseline (Law 1).
 *
 * The direct form is used to match the Lean SSOT at
 * `CharismaSandwich.buleyeanPredict` verbatim and to avoid the `+1 − 1`
 * float-precision drift on non-integer inputs (e.g., kernel entries).
 */
export function buleyeanPredict(
  baseline: number,
  advance: number,
  resist: number
): number {
  return baseline + advance - Math.min(resist, advance);
}

/** Saturated popularity. Lean: `CharismaRanking.saturatedPopularity`. */
export function saturatedPopularity(
  baseline: number,
  advance: number,
  resist: number,
  cap: number
): number {
  return Math.min(charismaPopularity(baseline, advance, resist), cap);
}

/** Prospect Theory gain relative to reference. */
export function gain(reference: number, outcome: number): number {
  return Math.max(0, outcome - reference);
}

/** Prospect Theory loss relative to reference. */
export function loss(reference: number, outcome: number): number {
  return Math.max(0, reference - outcome);
}

/**
 * Buleyean / prospect probability weight `w(R, v) = R - min(v, R) + 1`.
 * Direct re-export of `@a0n/buleyean-kernel::buleyeanWeight` in the
 * `(rounds, rejections)` argument order. Equals `CharismaProspectTheory.prospectWeight`.
 */
export function prospectWeight(rounds: number, rejections: number): number {
  return _buleyeanWeight(rounds, rejections);
}

/**
 * Rank items by charisma-style popularity.
 *
 * Output is sorted descending by `popularity`; ties fall back to
 * `tieBreaker` (default `advance`). Every returned record satisfies the
 * Buleyean sandwich.
 */
export function charismaRank<X>(
  input: CharismaRankInput<X>
): CharismaRanked<X>[] {
  const {
    items,
    advance: advanceFn,
    resist: resistFn,
    baseline = 0,
    cap = Number.POSITIVE_INFINITY,
    tieBreaker,
  } = input;

  const tieFn = tieBreaker ?? advanceFn;

  const scored: CharismaRanked<X>[] = items.map((item) => {
    const a = advanceFn(item);
    const r = resistFn(item);
    const outcome = baseline + a - r;
    const pop = Math.max(0, outcome);
    const pred = baseline + a - Math.min(r, a);
    const sat = Math.min(pop, cap);
    return {
      item,
      rank: 0,
      popularity: pop,
      buleyeanPredict: pred,
      saturated: sat,
      gain: Math.max(0, outcome - baseline),
      loss: Math.max(0, baseline - outcome),
      saturatedAtCap: pop >= cap,
      silenced: pop === 0 && r > 0,
      advance: a,
      resist: r,
    };
  });

  scored.sort((lhs, rhs) => {
    if (rhs.popularity !== lhs.popularity)
      return rhs.popularity - lhs.popularity;
    return tieFn(rhs.item) - tieFn(lhs.item);
  });

  return scored.map((entry, idx) => ({ ...entry, rank: idx }));
}

/** Sandwich-soundness witness for a single `(baseline, advance, resist)` triple. */
export interface SandwichWitness {
  readonly pessimistic: number;
  readonly actual: number;
  readonly predict: number;
  readonly optimistic: number;
  readonly chainHolds: boolean;
  readonly predictAboveBaseline: boolean;
  readonly exactOnCoveredRegime: boolean;
}

/** Produce the sandwich witness for a `(baseline, advance, resist)` triple. */
export function sandwichWitness(
  baseline: number,
  advance: number,
  resist: number
): SandwichWitness {
  const pessimistic = Math.max(0, baseline - resist);
  const actual = Math.max(0, baseline + advance - resist);
  const predict = baseline - 1 + _buleyeanWeight(advance, resist);
  const optimistic = baseline + advance;
  return {
    pessimistic,
    actual,
    predict,
    optimistic,
    chainHolds:
      pessimistic <= actual && actual <= predict && predict <= optimistic,
    predictAboveBaseline: predict >= baseline,
    exactOnCoveredRegime: resist <= advance ? actual === predict : true,
  };
}

/** True when two or more items saturate at the cap (rank lost among top tier). */
export function celebrityPlateau<X>(
  ranked: readonly CharismaRanked<X>[]
): boolean {
  let count = 0;
  for (const r of ranked) if (r.saturatedAtCap) count++;
  return count >= 2;
}
