/**
 * retrocausal.ts -- Law 7 (chain termination) as backward constraint
 *
 * Terminal states are where the God Formula reaches its floor:
 * w_i = 1 (v_i = R, maximum rejection). These terminal states
 * propagate backward through the void boundary via Law 5 (conservation):
 * the void mass at the terminal redistributes to ancestors.
 *
 * Law 1 (impossibility of zero) guarantees that even catastrophic
 * paths retain the sliver -- they are never impossible, only
 * maximally unlikely. Law 7 (chain termination) guarantees the
 * backward propagation converges.
 *
 * This is the deepest layer (timescale: lifetime). It constrains all
 * other layers via upwardConstraint() in the BoundaryStack.
 *
 * Mechanized in RetrocausalBound.lean, GodFormula.lean (zero sorry):
 *   theorem retrocausal_propagation:
 *     terminal_void(s) > 0 → ∀ t < T, ancestor_void(s, t) > 0
 */

import type { VoidBoundary, TimescaleBoundary } from '@a0n/gnosis/src/void';
import { updateVoidBoundary } from '@a0n/gnosis/src/void';

// ============================================================================
// Terminal State Encoding
// ============================================================================

/**
 * A terminal state that must be avoided.
 * The severity determines how much void propagates backward.
 */
export interface TerminalState {
  /** Which dimension this terminal state corresponds to */
  dimensionIdx: number;
  /** Severity of the terminal state (higher = more void propagation) */
  severity: number;
  /** Human-readable description */
  description: string;
}

/**
 * Encode terminal states into the retrocausal void boundary.
 * Terminal states get massive initial void -- they are what
 * must not happen, so they start deep in the void.
 */
export function encodeTerminalStates(
  boundary: VoidBoundary,
  terminals: TerminalState[]
): void {
  for (const terminal of terminals) {
    if (
      terminal.dimensionIdx >= 0 &&
      terminal.dimensionIdx < boundary.counts.length
    ) {
      updateVoidBoundary(boundary, terminal.dimensionIdx, terminal.severity);
    }
  }
}

// ============================================================================
// Backward Propagation
// ============================================================================

/**
 * Propagate terminal void backward through an adjacency structure.
 * Dimensions adjacent to high-void dimensions inherit some void.
 *
 * @param boundary The void boundary
 * @param adjacency Adjacency list: adjacency[i] = indices reachable from i
 * @param propagationFactor How much void transfers per hop (0-1)
 * @param maxHops Maximum propagation depth
 */
export function propagateBackward(
  boundary: VoidBoundary,
  adjacency: number[][],
  propagationFactor: number = 0.5,
  maxHops: number = 3
): void {
  const n = boundary.counts.length;
  // Build reverse adjacency: who can reach dimension i?
  const reverseAdj: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < adjacency.length; i++) {
    for (const target of adjacency[i]) {
      if (target >= 0 && target < n) {
        reverseAdj[target].push(i);
      }
    }
  }

  // Cannon rotation: within each hop, all dimension propagations are
  // independent (different source counts, different ancestor sets).
  // The additions array accumulation is commutative + associative.
  // Under gnode --strategy cannon, dimensions distribute across lanes.
  // Hops remain sequential (each hop reads previous hop's boundary).
  for (let hop = 0; hop < maxHops; hop++) {
    const additions = new Array(n).fill(0);
    const hopDecay = Math.pow(0.5, hop);

    // FORK: all dimensions compute independently per hop
    for (let i = 0; i < n; i++) {
      if (boundary.counts[i] > 0) {
        const propagated = boundary.counts[i] * propagationFactor * hopDecay;
        for (const ancestor of reverseAdj[i]) {
          additions[ancestor] += propagated;
        }
      }
    }

    // FOLD: apply accumulated additions to boundary
    for (let i = 0; i < n; i++) {
      if (additions[i] > 0) {
        updateVoidBoundary(boundary, i, additions[i]);
      }
    }
  }
}

/**
 * Compute the retrocausal bound: the minimum void that must exist
 * at any ancestor of a terminal state.
 *
 * If terminal_void(s) > 0, then for all ancestors a of s:
 *   void(a) >= terminal_void(s) * propagationFactor^distance(a, s)
 */
export function retrocausalBound(
  terminalVoid: number,
  distance: number,
  propagationFactor: number = 0.5
): number {
  return terminalVoid * Math.pow(propagationFactor, distance);
}

/**
 * Check that the retrocausal bound holds for a boundary.
 * Returns true if all ancestors have sufficient void.
 */
export function verifyRetrocausalBound(
  boundary: VoidBoundary,
  terminals: TerminalState[],
  adjacency: number[][],
  propagationFactor: number = 0.5
): boolean {
  const n = boundary.counts.length;
  // Build reverse adjacency
  const reverseAdj: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < adjacency.length; i++) {
    for (const target of adjacency[i]) {
      if (target >= 0 && target < n) {
        reverseAdj[target].push(i);
      }
    }
  }

  // BFS from each terminal
  for (const terminal of terminals) {
    const visited = new Set<number>();
    const queue: { idx: number; dist: number }[] = [
      { idx: terminal.dimensionIdx, dist: 0 },
    ];
    while (queue.length > 0) {
      const { idx, dist } = queue.shift()!;
      if (visited.has(idx)) continue;
      visited.add(idx);

      if (dist > 0) {
        const minVoid = retrocausalBound(
          terminal.severity,
          dist,
          propagationFactor
        );
        if (boundary.counts[idx] < minVoid * 0.5) {
          // Allow 50% tolerance for floating point and decay
          return false;
        }
      }

      for (const ancestor of reverseAdj[idx]) {
        if (!visited.has(ancestor)) {
          queue.push({ idx: ancestor, dist: dist + 1 });
        }
      }
    }
  }
  return true;
}

/**
 * Initialize a retrocausal TimescaleBoundary with terminal states
 * and backward propagation.
 */
export function initRetrocausalLayer(
  layer: TimescaleBoundary,
  terminals: TerminalState[],
  adjacency?: number[][],
  propagationFactor?: number
): void {
  encodeTerminalStates(layer.boundary, terminals);
  if (adjacency) {
    propagateBackward(layer.boundary, adjacency, propagationFactor);
  }
}
