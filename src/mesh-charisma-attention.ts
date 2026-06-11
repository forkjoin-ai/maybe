/**
 * Mesh Charisma Attention: Init-only Nat sibling of the Mathlib-backed
 * `MeshAttention*` stack. Gives every mesh column a `charisma` advantage
 * and a `resistance` mask, then recovers the Buleyean minorization floor,
 * saturation collapse, ranking preorder, and silence/cap endpoints.
 *
 * Runtime mirror of `open-source/gnosis/lean/Lean/ForkRaceFoldTheorems/MeshCharismaAttention.lean`.
 *
 * ## Correspondence
 *
 * | TS export              | Lean definition / theorem                   |
 * |------------------------|---------------------------------------------|
 * | `Mesh`                 | `MeshCharismaAttention.Mesh`                |
 * | `absorbed`             | `MeshCharismaAttention.absorbed`            |
 * | `buleyeanMass`         | `MeshCharismaAttention.buleyeanMass`        |
 * | `saturatedMass`        | `MeshCharismaAttention.saturatedMass`       |
 * | `outranks`             | `MeshCharismaAttention.outranks`            |
 * | `prospectWeightColumn` | `MeshCharismaAttention.prospectWeight`      |
 *
 * Theorems (Lean owns correctness): `buleyeanFloorUniform`,
 * `absorbedBelowBuleyean`, `columnReachesSilence`, `columnBoundedByCap`,
 * `celebrityPlateau`, `celebrityDominates`, `rankReflexive`,
 * `rankTransitive`, `rankTotal`, `prospectWeightAboveOne`.
 *
 * Uses `@a0n/buleyean-kernel::buleyeanWeight` for the sliver formula.
 *
 * Pure TS. No Node, no fs, no DOM.
 */

import { buleyeanWeight as _buleyeanWeight } from '@a0n/buleyean-kernel';

/** Lean: `Mesh n`. Columns indexed 0..n-1. */
export interface Mesh {
  readonly n: number;
  readonly baseline: number;
  readonly charisma: readonly number[];
  readonly resistance: readonly number[];
}

/** Accessor for column-j charisma (defaults to 0 if out of range). */
function chi(m: Mesh, j: number): number {
  return Math.max(0, m.charisma[j] ?? 0);
}

/** Accessor for column-j resistance (defaults to 0 if out of range). */
function rho(m: Mesh, j: number): number {
  return Math.max(0, m.resistance[j] ?? 0);
}

/** Lean: `absorbed m j = (baseline + charisma j) - resistance j` (Nat.sub). */
export function absorbed(m: Mesh, j: number): number {
  return Math.max(0, Math.max(0, m.baseline) + chi(m, j) - rho(m, j));
}

/**
 * Lean: `buleyeanMass m j = baseline + charisma j - min (resistance j) (charisma j)`.
 * Equivalently `baseline - 1 + buleyeanWeight(charisma j, resistance j)`.
 */
export function buleyeanMass(m: Mesh, j: number): number {
  return Math.max(0, m.baseline) - 1 + _buleyeanWeight(chi(m, j), rho(m, j));
}

/** Lean: `saturatedMass m cap j = min (absorbed m j) cap`. */
export function saturatedMass(m: Mesh, cap: number, j: number): number {
  return Math.min(absorbed(m, j), Math.max(0, cap));
}

/** Lean: `outranks m j1 j2 ↔ absorbed m j2 ≤ absorbed m j1`. */
export function outranks(m: Mesh, j1: number, j2: number): boolean {
  return absorbed(m, j2) <= absorbed(m, j1);
}

/**
 * Lean: `prospectWeight m j = charisma j - min (resistance j) (charisma j) + 1`.
 * Identical to `buleyeanWeight(charisma j, resistance j)`.
 */
export function prospectWeightColumn(m: Mesh, j: number): number {
  return _buleyeanWeight(chi(m, j), rho(m, j));
}

/**
 * Lean `columnReachesSilence` constructive witness: a resistance field
 * `ρ'(k) = baseline + charisma k` silences column `j`. Returned as a
 * fresh Mesh with that resistance applied uniformly.
 */
export function silencingResistance(m: Mesh): Mesh {
  const b = Math.max(0, m.baseline);
  const silenced = Array.from({ length: m.n }, (_, k) => b + chi(m, k));
  return { ...m, resistance: silenced };
}
