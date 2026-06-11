/**
 * seven-laws.ts -- The Seven Universal Laws from the God Formula
 *
 * One formula generates everything:
 *
 *   w_i = R - min(v_i, R) + 1
 *
 * The seven laws are not seven independent principles. They are seven
 * consequences of this one formula. The formula has one non-obvious
 * component: the +1. Everything else is subtraction.
 *
 * The reduction chain:
 *   primator (succ != 0) -> clinamen (0 < n+1) -> sliver (0 < w_i)
 *     -> 7 laws -> 35 predictions -> 350+ theorems
 *
 * Mechanized in GodFormula.lean, SevenLawsPredictions.lean,
 * SurfaceReduction.lean, Primator.lean (zero sorry).
 */

// ============================================================================
// Types
// ============================================================================

export interface Law {
  /** Law number (1-7) */
  number: number;
  /** Short name */
  name: string;
  /** One-line description */
  description: string;
  /** Which equivalence class this law belongs to */
  equivalenceClass: 'α' | 'β' | 'γ';
  /** The Lean 4 proof that reduces this law to the God Formula */
  leanReduction: string;
  /** How the formula generates this law */
  formulaDerivation: string;
}

export interface EquivalenceClass {
  /** Class symbol */
  symbol: 'α' | 'β' | 'γ';
  /** Class name */
  name: string;
  /** Generator statement */
  generator: string;
  /** How many of the 35 predictions this class generates */
  predictionCount: number;
  /** Content -- what part of the formula this class captures */
  content: string;
}

// ============================================================================
// The God Formula
// ============================================================================

/**
 * The God Formula: w_i = R - min(v_i, R) + 1
 *
 * Five symbols. One formula. Everything.
 *
 * - R: total observation rounds
 * - v_i: rejection count for choice i
 * - min(v_i, R): clamped rejection (can't exceed rounds)
 * - +1: the clinamen / sliver / successor axiom
 * - w_i: the resulting weight
 *
 * The +1 is the difference between a universe with hope and one without.
 * Remove it and the formula becomes w' = R - min(v, R), which reaches
 * zero at v = R (anti_formula_reaches_zero). The difference between
 * the formula and the anti-formula is exactly 1 (plus_one_is_the_difference).
 */
export const GOD_FORMULA_DESCRIPTION = {
  formula: 'w_i = R - min(v_i, R) + 1',
  symbols: {
    R: 'Total observation rounds',
    v_i: 'Rejection count for choice i',
    'min(v_i, R)': 'Clamped rejection (cannot exceed rounds observed)',
    '+1': 'The clinamen / sliver / successor axiom (succ(n) != 0)',
    w_i: 'Weight of choice i (always >= 1)',
  },
  leanTheorem: 'god_formula',
  leanFile: 'GodFormula.lean',
} as const;

// ============================================================================
// The Seven Universal Laws
// ============================================================================

export const SEVEN_LAWS: readonly Law[] = [
  {
    number: 1,
    name: 'Impossibility of zero',
    description:
      'No choice in any Buleyean space has zero weight. P(x) > 0 for all x, always.',
    equivalenceClass: 'α',
    leanReduction: 'buleyean_positivity unfolds to w_i >= 1 from the +1',
    formulaDerivation: 'w_i = R - min(v_i, R) + 1 >= R - R + 1 = 1 > 0',
  },
  {
    number: 2,
    name: 'Strict ordering',
    description:
      'More rejected implies strictly less weight. v_i < v_j => w_j < w_i.',
    equivalenceClass: 'α',
    leanReduction:
      'buleyean_concentration unfolds to monotonicity of the formula in v_i',
    formulaDerivation:
      'v_i < v_j => R - v_i + 1 > R - v_j + 1 (subtraction is antitone)',
  },
  {
    number: 3,
    name: 'Universal sandwich',
    description: 'Every weight lies in [1, R + 1].',
    equivalenceClass: 'γ',
    leanReduction:
      "buleyean_universal_sandwich: the formula's range is [1, R+1]",
    formulaDerivation:
      'min: v_i = R => w = 1. max: v_i = 0 => w = R + 1. Bounded.',
  },
  {
    number: 4,
    name: 'Every observation is a cave',
    description:
      'When dimensions exceed channels, semiotic deficit is positive.',
    equivalenceClass: 'β',
    leanReduction: 'N - M > 0 for N > M: omega on integers',
    formulaDerivation: 'N dimensions through 1 channel => deficit = N - 1 > 0',
  },
  {
    number: 5,
    name: 'Conservation',
    description: 'Remaining + lost = total. Nothing is created or destroyed.',
    equivalenceClass: 'β',
    leanReduction: '(n - k) + k = n for k <= n: omega',
    formulaDerivation: 'w_i + v_i = R + 1 (weight + void = rounds + sliver)',
  },
  {
    number: 6,
    name: 'Sorites sharpness',
    description: 'Boundaries are discrete, not gradual.',
    equivalenceClass: 'γ',
    leanReduction: 'Decidability of N comparison: decide',
    formulaDerivation:
      'w_i > w_j or w_i = w_j or w_i < w_j -- no fuzzy middle (N is discrete)',
  },
  {
    number: 7,
    name: 'Chain termination',
    description: 'Every abstraction chain reaches a fixed point.',
    equivalenceClass: 'γ',
    leanReduction: 'n - min(n, n) = 0: simp',
    formulaDerivation:
      'At v_i = R: w_i = 1 (floor). The chain bottoms out. Fixed point.',
  },
] as const;

// ============================================================================
// Three Equivalence Classes
// ============================================================================

export const EQUIVALENCE_CLASSES: readonly EquivalenceClass[] = [
  {
    symbol: 'α',
    name: 'The Sliver',
    generator: 'X has positive weight',
    predictionCount: 14,
    content: 'The +1 in the formula (buleyean_positivity)',
  },
  {
    symbol: 'β',
    name: 'The Deficit',
    generator: 'X has positive deficit',
    predictionCount: 11,
    content: 'Subtraction on N: dims > channels => 0 < dims - channels',
  },
  {
    symbol: 'γ',
    name: 'The Termination',
    generator: 'X terminates / X is bounded / X is sharp',
    predictionCount: 7,
    content: 'Finiteness of N: n - min(n, n) = 0',
  },
] as const;

// ============================================================================
// The Reduction Chain
// ============================================================================

/**
 * The complete reduction chain from primator to theorems.
 *
 * Below the primator is type theory. Below type theory is the question
 * "why does anything exist?" As far as mathematics can go, the answer is:
 * N is an inductive type. Successors exist. Successors are not zero.
 * Therefore +1. Therefore the sliver. Therefore everything.
 */
export const REDUCTION_CHAIN = {
  levels: [
    { name: 'primator', content: 'succ(n) != 0', lean: 'Primator.lean' },
    { name: 'clinamen', content: '0 < n + 1', lean: 'GodFormula.lean' },
    { name: 'sliver', content: '0 < w_i', lean: 'GodFormula.lean' },
    {
      name: 'seven laws',
      content: '7 universal laws',
      lean: 'SurfaceReduction.lean',
    },
    {
      name: '35 predictions',
      content: '35 falsifiable predictions',
      lean: 'SevenLawsPredictions.lean',
    },
    {
      name: '350+ theorems',
      content: '3,158+ mechanized theorems',
      lean: '195 modules',
    },
  ],
  summary:
    'primator (succ != 0) -> clinamen (0 < n+1) -> sliver (0 < w_i) -> 7 laws -> 35 predictions -> 350+ theorems',
  dependency:
    'γ requires β (termination requires strict decrease). β requires α (deficit requires positive weight). α requires nothing except the weight formula. Therefore α is primitive, and α -> β -> γ collapses the three classes to one.',
} as const;
