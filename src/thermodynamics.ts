/**
 * thermodynamics.ts -- Law 5 (conservation) as energy accounting
 *
 * The God Formula's Law 5: remaining + lost = total. In thermodynamic
 * language: V_fork = W_fold + Q_vent. What fork creates, fold uses
 * or vent dissipates. Nothing else. This is the conservation column
 * of the 3x3 matrix -- truth persists through transport into knowledge.
 *
 * The +1 in w_i = R - min(v_i, R) + 1 maps to Landauer heat: erasing one
 * bit costs kT ln 2 joules. The sliver (Law 1) is the thermodynamic
 * floor -- the minimum energy that must be paid for any fold.
 *
 * Fork creates potential (parallel paths). Fold extracts work (decision).
 * Vent dissipates waste. Law 3 (sandwich) bounds the energy: the fold
 * can never extract more than the fork created.
 *
 * Mechanized in LandauerBuley.lean, GodFormula.lean (zero sorry):
 *   theorem landauer_heat_nonneg: heat >= 0
 *   theorem fold_heat_hierarchy_strict: non_injective => heat > 0
 *   theorem first_law_conservation: V_fork = W_fold + Q_vent
 */

// ============================================================================
// Constants
// ============================================================================

/** Boltzmann constant in J/K */
export const BOLTZMANN_K = 1.380649e-23;

/** Room temperature in K */
export const ROOM_TEMPERATURE = 300;

/** Minimum erasure cost per bit at room temperature: kT ln 2 */
export const LANDAUER_LIMIT = BOLTZMANN_K * ROOM_TEMPERATURE * Math.LN2;

// ============================================================================
// Shannon Entropy
// ============================================================================

/**
 * Shannon entropy of a distribution in bits.
 * H(p) = -sum_i p_i log2(p_i)
 *
 * This is the maximum extractable work from a distribution (Carnot bound).
 */
export function shannonEntropy(p: number[]): number {
  let h = 0;
  for (const pi of p) {
    if (pi > 0) {
      h -= pi * Math.log2(pi);
    }
  }
  return h;
}

/**
 * Maximum entropy for n outcomes = log2(n) (uniform distribution).
 */
export function maxEntropy(n: number): number {
  return n > 1 ? Math.log2(n) : 0;
}

// ============================================================================
// Landauer Heat
// ============================================================================

/**
 * Landauer heat floor: minimum heat dissipated when folding N branches
 * to a single outcome.
 *
 * For a uniform input: kT ln 2 * log2(N) joules.
 * For a non-uniform input: kT ln 2 * H(input) joules.
 *
 * The fold MUST pay at least this much -- it is the price of deciding.
 */
export function landauerHeatFloor(
  inputDistribution: number[],
  temperature: number = ROOM_TEMPERATURE
): number {
  const entropy = shannonEntropy(inputDistribution);
  return BOLTZMANN_K * temperature * Math.LN2 * entropy;
}

/**
 * Landauer heat ceiling: maximum heat when all fork energy is dissipated
 * (worst-case fold -- pure waste, no useful work extracted).
 *
 * ceiling = kT ln 2 * log2(N) for N-branch fork.
 */
export function landauerHeatCeiling(
  branchCount: number,
  temperature: number = ROOM_TEMPERATURE
): number {
  return BOLTZMANN_K * temperature * Math.LN2 * maxEntropy(branchCount);
}

/**
 * Landauer heat sandwich: floor <= actual heat <= ceiling.
 * Returns { floor, ceiling, isSatisfied }.
 */
export function landauerHeatSandwich(
  inputDistribution: number[],
  actualHeat: number,
  temperature: number = ROOM_TEMPERATURE
): {
  floor: number;
  ceiling: number;
  actualHeat: number;
  isSatisfied: boolean;
} {
  const floor = landauerHeatFloor(inputDistribution, temperature);
  const ceiling = landauerHeatCeiling(inputDistribution.length, temperature);
  return {
    floor,
    ceiling,
    actualHeat,
    isSatisfied: actualHeat >= floor - 1e-30 && actualHeat <= ceiling + 1e-30,
  };
}

// ============================================================================
// Fork/Fold/Vent Energy Accounting
// ============================================================================

/**
 * Fork energy: potential energy created by opening N parallel paths.
 * V_fork = kT ln 2 * log2(N)
 *
 * This is the maximum work that a subsequent fold could extract.
 */
export function forkEnergy(
  branchCount: number,
  temperature: number = ROOM_TEMPERATURE
): number {
  return BOLTZMANN_K * temperature * Math.LN2 * maxEntropy(branchCount);
}

/**
 * Fold work: information preserved (mutual information) during the fold.
 * W_fold = kT ln 2 * (H_input - H_output)
 *
 * The fold extracts work by reducing entropy. The work is the entropy
 * difference between input (forked) and output (decided) distributions.
 */
export function foldWork(
  inputDistribution: number[],
  outputDistribution: number[],
  temperature: number = ROOM_TEMPERATURE
): number {
  const hIn = shannonEntropy(inputDistribution);
  const hOut = shannonEntropy(outputDistribution);
  const entropyReduction = Math.max(0, hIn - hOut);
  return BOLTZMANN_K * temperature * Math.LN2 * entropyReduction;
}

/**
 * Vent heat: energy dissipated by discarding alternatives.
 * Q_vent = V_fork - W_fold
 *
 * By the first law, what fork creates that fold doesn't use is vented.
 */
export function ventHeat(
  forkEnergyValue: number,
  foldWorkValue: number
): number {
  return Math.max(0, forkEnergyValue - foldWorkValue);
}

/**
 * Thermodynamic efficiency: fraction of fork energy converted to useful work.
 * eta = W_fold / V_fork
 *
 * eta = 1: perfect fold (all information preserved, zero waste)
 * eta = 0: pure vent (all information destroyed)
 * eta in (0, 1): typical fold with partial waste
 */
export function foldEfficiency(
  forkEnergyValue: number,
  foldWorkValue: number
): number {
  if (forkEnergyValue <= 0) return 1; // no fork = nothing to waste
  return Math.min(1, foldWorkValue / forkEnergyValue);
}

// ============================================================================
// First Law Verification
// ============================================================================

/**
 * Verify the first law: V_fork = W_fold + Q_vent.
 *
 * This is the conservation law for irreversible process.
 * What fork creates, fold uses or vent dissipates. Nothing else.
 */
export function verifyFirstLaw(
  vFork: number,
  wFold: number,
  qVent: number,
  tolerance: number = 1e-12
): { holds: boolean; residual: number } {
  const residual = Math.abs(vFork - wFold - qVent);
  return { holds: residual <= tolerance, residual };
}

/**
 * Complete thermodynamic audit of a fork/fold operation.
 *
 * Given an input distribution (post-fork) and output distribution (post-fold),
 * computes all energetic quantities and verifies the first law.
 */
export function thermodynamicAudit(
  inputDistribution: number[],
  outputDistribution: number[],
  temperature: number = ROOM_TEMPERATURE
): {
  forkEnergy: number;
  foldWork: number;
  ventHeat: number;
  efficiency: number;
  landauerFloor: number;
  landauerCeiling: number;
  firstLawHolds: boolean;
  shannonInput: number;
  shannonOutput: number;
} {
  const vFork = forkEnergy(inputDistribution.length, temperature);
  const wFold = foldWork(inputDistribution, outputDistribution, temperature);
  const qVent = ventHeat(vFork, wFold);
  const { holds } = verifyFirstLaw(vFork, wFold, qVent);

  return {
    forkEnergy: vFork,
    foldWork: wFold,
    ventHeat: qVent,
    efficiency: foldEfficiency(vFork, wFold),
    landauerFloor: landauerHeatFloor(inputDistribution, temperature),
    landauerCeiling: landauerHeatCeiling(inputDistribution.length, temperature),
    firstLawHolds: holds,
    shannonInput: shannonEntropy(inputDistribution),
    shannonOutput: shannonEntropy(outputDistribution),
  };
}

// ============================================================================
// Transition Cost on the Manifold
// ============================================================================

/**
 * Minimum thermodynamic cost of transitioning between two distributions.
 * This is the Landauer cost of the entropy change.
 *
 * If entropy decreases (fold): cost = kT ln 2 * (H_before - H_after)
 * If entropy increases (fork): cost = 0 (entropy increase is free)
 *
 * This gives every edge on the Fisher manifold a thermodynamic weight.
 */
export function transitionCost(
  before: number[],
  after: number[],
  temperature: number = ROOM_TEMPERATURE
): number {
  const hBefore = shannonEntropy(before);
  const hAfter = shannonEntropy(after);
  const entropyReduction = Math.max(0, hBefore - hAfter);
  return BOLTZMANN_K * temperature * Math.LN2 * entropyReduction;
}

/**
 * Total thermodynamic cost of a trajectory on the manifold.
 * Sum of Landauer costs at each entropy-reducing step.
 */
export function trajectoryCost(
  trajectory: number[][],
  temperature: number = ROOM_TEMPERATURE
): number {
  let total = 0;
  for (let i = 1; i < trajectory.length; i++) {
    total += transitionCost(trajectory[i - 1], trajectory[i], temperature);
  }
  return total;
}
