/**
 * Charisma Skyrms signaling game: strategic layer on top of the
 * `CharismaToyModel` unilateral mechanism. Pure-Nat arithmetic.
 *
 * Runtime mirror of `open-source/gnosis/lean/Lean/ForkRaceFoldTheorems/CharismaSkyrmsGame.lean`.
 *
 * ## Correspondence
 *
 * | TS export                  | Lean definition / theorem                    |
 * |----------------------------|----------------------------------------------|
 * | `natDist`                  | `CharismaSkyrmsGame.natDist`                 |
 * | `reception`                | `CharismaSkyrmsGame.reception`               |
 * | `senderRegret`             | `CharismaSkyrmsGame.senderRegret`            |
 * | `receiverRegret`           | `CharismaSkyrmsGame.receiverRegret`          |
 * | `isEquilibrium`            | `CharismaSkyrmsGame.isEquilibrium` (bounded) |
 *
 * Theorems (Lean owns correctness): `natDistSelf`, `natDistComm`,
 * `underPushPlateau`, `overPushStrict`, `senderCostFloor`,
 * `receiverCostFloor`, `alignedPlateauEquilibrium`,
 * `alignedZeroEquilibrium`, `overPushNotEquilibrium`.
 *
 * Note: Lean `isEquilibrium` quantifies over all Nat deviations (unbounded).
 * TS `isEquilibrium` takes an explicit finite search radius `maxDeviation`
 * because JS cannot enumerate all Nat. Grid check, not a proof.
 *
 * Pure TS. No Node, no fs, no DOM.
 */

/** Lean: `Game`. */
export interface Game {
  readonly baseline: number;
  readonly target: number;
  readonly bias: number;
}

/** Lean: `natDist a b = (a - b) + (b - a)` over Nat. */
export function natDist(a: number, b: number): number {
  const an = Math.max(0, a);
  const bn = Math.max(0, b);
  return Math.abs(an - bn);
}

/** Lean: `reception g κ ρ = (baseline + κ) - ρ` (Nat.sub). */
export function reception(g: Game, kappa: number, rho: number): number {
  return Math.max(
    0,
    Math.max(0, g.baseline) + Math.max(0, kappa) - Math.max(0, rho)
  );
}

/** Lean: `senderRegret g κ ρ = natDist (reception g κ ρ) g.bias + κ`. */
export function senderRegret(g: Game, kappa: number, rho: number): number {
  return natDist(reception(g, kappa, rho), g.bias) + Math.max(0, kappa);
}

/** Lean: `receiverRegret g κ ρ = natDist (reception g κ ρ) g.target + ρ`. */
export function receiverRegret(g: Game, kappa: number, rho: number): number {
  return natDist(reception(g, kappa, rho), g.target) + Math.max(0, rho);
}

/**
 * Bounded-grid executable witness for Lean `isEquilibrium g κ ρ`.
 *
 * Lean quantifies over all `κ', ρ' : Nat`. Here we can only sweep a
 * finite window `[0, maxDeviation]`. Returns `true` iff no deviation in
 * that window strictly reduces either party's regret.
 */
export function isEquilibrium(
  g: Game,
  kappa: number,
  rho: number,
  maxDeviation: number
): boolean {
  const sr = senderRegret(g, kappa, rho);
  const rr = receiverRegret(g, kappa, rho);
  const hi = Math.max(0, Math.floor(maxDeviation));
  for (let kp = 0; kp <= hi; kp++) {
    if (senderRegret(g, kp, rho) < sr) return false;
  }
  for (let rp = 0; rp <= hi; rp++) {
    if (receiverRegret(g, kappa, rp) < rr) return false;
  }
  return true;
}

/**
 * Plateau value: sender regret when under-pushing with `ρ = 0`.
 * Lean `underPushPlateau`: this equals `bias - baseline`.
 */
export function underPushPlateauValue(g: Game): number {
  return Math.max(0, g.bias - g.baseline);
}
