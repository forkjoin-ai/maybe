/**
 * Charisma toy model: unilateral mechanism (Source, Audience, Resistance,
 * Capacity, cascade bottleneck) in pure Nat arithmetic.
 *
 * Runtime mirror of `open-source/gnosis/lean/Lean/ForkRaceFoldTheorems/CharismaToyModel.lean`.
 *
 * ## Correspondence
 *
 * | TS export                          | Lean definition / theorem                       |
 * |------------------------------------|-------------------------------------------------|
 * | `BASELINE_UPTAKE`                  | `CharismaToyModel.baselineUptake`               |
 * | `perReceiverUptake`                | `CharismaToyModel.perReceiverUptake`            |
 * | `totalReception`                   | `CharismaToyModel.totalReception`               |
 * | `netPerReceiverUptake`             | `CharismaToyModel.netPerReceiverUptake`         |
 * | `saturatedPerReceiverUptake`       | `CharismaToyModel.saturatedPerReceiverUptake`   |
 * | `cascadedCharisma`                 | `CharismaToyModel.cascadedCharisma`             |
 *
 * Theorems (Lean owns correctness; TS ships executable witnesses):
 * `receptionMonotoneInCharisma`, `receptionMonotoneInAudience`,
 * `zeroCharismaReducesToBaseline`, `positiveCharismaStrictlyBeatsBaseline`,
 * `resistanceMonotone`, `charismaResistanceCollision`,
 * `totalResistanceSilences`, `saturationUpperBound`,
 * `saturationDominatedByRaw`, `saturationHonorsBaseline`,
 * `cascadeBoundedBySource`, `cascadeBoundedByIntermediary`,
 * `cascadeSaturatesAtWeakest`.
 *
 * Pure TS. No Node, no fs, no DOM.
 */

/** Lean: `baselineUptake = 1`. */
export const BASELINE_UPTAKE = 1;

/** Lean: `Source`. */
export interface Source {
  readonly charisma: number;
}

/** Lean: `Audience` (size must be positive). */
export interface Audience {
  readonly size: number;
}

/** Lean: `Resistance`. */
export interface Resistance {
  readonly coefficient: number;
}

/** Lean: `Capacity` (cap ≥ baseline). */
export interface Capacity {
  readonly cap: number;
}

/** Lean: `perReceiverUptake s = baselineUptake + s.charisma`. */
export function perReceiverUptake(s: Source): number {
  return BASELINE_UPTAKE + Math.max(0, s.charisma);
}

/** Lean: `totalReception s a = a.size * perReceiverUptake s`. */
export function totalReception(s: Source, a: Audience): number {
  return Math.max(0, a.size) * perReceiverUptake(s);
}

/** Lean: `netPerReceiverUptake s r = perReceiverUptake s - r.coefficient` (Nat.sub). */
export function netPerReceiverUptake(s: Source, r: Resistance): number {
  return Math.max(0, perReceiverUptake(s) - Math.max(0, r.coefficient));
}

/** Lean: `saturatedPerReceiverUptake s c = min (perReceiverUptake s) c.cap`. */
export function saturatedPerReceiverUptake(s: Source, c: Capacity): number {
  return Math.min(perReceiverUptake(s), Math.max(BASELINE_UPTAKE, c.cap));
}

/** Lean: `cascadedCharisma s i = min s.charisma i.charisma`. */
export function cascadedCharisma(s: Source, i: Source): number {
  return Math.min(Math.max(0, s.charisma), Math.max(0, i.charisma));
}

/**
 * Total cascade reception through an intermediary: uses the bottleneck
 * charisma as the effective source charisma. Convenience wrapper; Lean
 * uses `cascadedCharisma` directly in its proofs.
 */
export function cascadeReception(s: Source, i: Source, a: Audience): number {
  return totalReception({ charisma: cascadedCharisma(s, i) }, a);
}
