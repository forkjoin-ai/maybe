/**
 * arbitrage.ts -- the consensus law as a finite no-arbitrage checker.
 *
 * The TensorBayes consensus law states that one joint mass must be readable from
 * every ordered pair of faces:
 *
 *   cond[i][j] * M[j]  =  cond[j][i] * M[i].
 *
 * Read the left side as the joint mass of (i, j) and the right as the joint mass
 * of (j, i). The law is exactly the statement that the directed joint matrix
 * A[i][j] = cond[i][j] * M[j] is symmetric. In finance terms, cond[i][j] is the
 * price of state j quoted on venue i and M is the venue/state mass; the law is
 * the no-arbitrage consistency condition, and the antisymmetric part
 * A[i][j] - A[j][i] is the residual an arbitrageur can trade.
 *
 * findsArbitrage returns every violating ordered pair and a single witness. All
 * arithmetic is exact rational; a float would hide the very inconsistency the
 * checker exists to detect.
 *
 * Adversarial dual: a direction-blind checker that compares cond[i][j] with
 * cond[j][i] and drops the mass M misses a genuine inconsistency whenever the
 * masses differ. arbitrageAdversarialDual names that failure mode.
 *
 * Nothing here is financial advice and nothing is claimed about market prices.
 */

import {
  compareFractions,
  exactFraction,
  fractionToNumber,
  mulFractions,
  subFractions,
  toFraction,
  type ExactFraction,
  type RationalInput,
} from './rational.js';

/** One matrix entry: a number, bigint, or exact fraction. */
export type MatrixEntry = RationalInput;

/** A ragged-averse matrix; every row must have the same length. */
export type RationalMatrix = readonly (readonly MatrixEntry[])[];

/** One mass per face. */
export type MassVector = readonly RationalInput[];

/** Coerce a matrix to exact fractions, enforcing a non-empty square shape. */
export function asRationalMatrix(rows: RationalMatrix, where = 'cond'): ExactFraction[][] {
  if (rows.length === 0) {
    throw new RangeError(where + ' must have at least one row');
  }
  const width = rows[0]!.length;
  if (width === 0) {
    throw new RangeError(where + ' rows must have at least one column');
  }
  if (rows.length !== width) {
    throw new RangeError(where + ' must be square; got ' + String(rows.length) + 'x' + String(width));
  }
  return rows.map((row, i) => {
    if (row.length !== width) {
      throw new RangeError(
        where + ' row ' + String(i) + ' has length ' + String(row.length) + ', expected ' + String(width),
      );
    }
    return row.map((entry, j) => toFraction(entry, where + '[' + String(i) + '][' + String(j) + ']'));
  });
}

/** Coerce a mass vector to exact fractions of a required length. */
export function asMassVector(masses: MassVector, length: number, where = 'masses'): ExactFraction[] {
  if (masses.length !== length) {
    throw new RangeError(
      where + ' length ' + String(masses.length) + ' must equal the matrix dimension ' + String(length),
    );
  }
  return masses.map((mass, i) => toFraction(mass, where + '[' + String(i) + ']'));
}

/** One ordered-pair residual: the joint mass read both ways. */
export interface ArbitrageWitness {
  readonly i: number;
  readonly j: number;
  /** cond[i][j] * M[j]. */
  readonly left: ExactFraction;
  /** cond[j][i] * M[i]. */
  readonly right: ExactFraction;
  /** left - right; non-zero is the arbitrage. */
  readonly residual: ExactFraction;
  /** Display-only ratio of the residual magnitude to the smaller side. */
  readonly gap: number;
}

/** The full consistency report. */
export interface ArbitrageReport {
  readonly arbitrage: boolean;
  readonly violations: readonly ArbitrageWitness[];
  readonly witness: ArbitrageWitness | null;
  readonly checkedPairs: number;
  readonly detail: string;
}

function witnessFor(
  matrix: readonly (readonly ExactFraction[])[],
  masses: readonly ExactFraction[],
  i: number,
  j: number,
): ArbitrageWitness {
  const left = mulFractions(matrix[i]![j]!, masses[j]!);
  const right = mulFractions(matrix[j]![i]!, masses[i]!);
  const residual = subFractions(left, right);
  const larger = compareFractions(left, right) >= 0 ? left : right;
  const absResidual = residual.numerator < 0n
    ? exactFraction(-residual.numerator, residual.denominator)
    : residual;
  const absLarger = larger.numerator < 0n
    ? exactFraction(-larger.numerator, larger.denominator)
    : larger;
  const gap = compareFractions(absLarger, 0n) === 0
    ? 0
    : fractionToNumber(exactFraction(
        absResidual.numerator * absLarger.denominator,
        absResidual.denominator * absLarger.numerator,
      ));
  return { i, j, left, right, residual, gap };
}

/** The consensus residual cond[i][j] M[j] - cond[j][i] M[i], exact. */
export function consensusResidual(
  cond: RationalMatrix,
  masses: MassVector,
  i: number,
  j: number,
): ExactFraction {
  const matrix = asRationalMatrix(cond);
  const mass = asMassVector(masses, matrix.length);
  if (!Number.isInteger(i) || !Number.isInteger(j) || i < 0 || j < 0 || i >= matrix.length || j >= matrix.length) {
    throw new RangeError('pair (' + String(i) + ', ' + String(j) + ') is outside the matrix');
  }
  return subFractions(mulFractions(matrix[i]![j]!, mass[j]!), mulFractions(matrix[j]![i]!, mass[i]!));
}

/**
 * Find every ordered pair whose consensus residual is non-zero. The first
 * violation (lowest i, then lowest j) is returned as the witness.
 */
export function findsArbitrage(cond: RationalMatrix, masses: MassVector): ArbitrageReport {
  const matrix = asRationalMatrix(cond);
  const mass = asMassVector(masses, matrix.length);
  const n = matrix.length;
  const violations: ArbitrageWitness[] = [];
  let checkedPairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      checkedPairs += 1;
      const witness = witnessFor(matrix, mass, i, j);
      if (witness.residual.numerator !== 0n) violations.push(witness);
    }
  }
  const witness = violations.length > 0 ? violations[0]! : null;
  return {
    arbitrage: violations.length > 0,
    violations,
    witness,
    checkedPairs,
    detail: witness === null
      ? 'consensus holds on all ' + String(checkedPairs) + ' pairs; no arbitrage'
      : 'arbitrage at pair (' + String(witness.i) + ', ' + String(witness.j) + '): ' +
        String(witness.left.numerator) + '/' + String(witness.left.denominator) +
        ' != ' + String(witness.right.numerator) + '/' + String(witness.right.denominator) +
        ' (residual ' + String(witness.residual.numerator) + '/' + String(witness.residual.denominator) + ')',
  };
}

/** The two-venue check: cond = [[1, aToB], [bToA, 1]]. */
export function twoVenueArbitrage(
  aToB: MatrixEntry,
  bToA: MatrixEntry,
  massA: RationalInput,
  massB: RationalInput,
): ArbitrageReport {
  const cond: RationalMatrix = [
    [1n, aToB],
    [bToA, 1n],
  ];
  return findsArbitrage(cond, [massA, massB]);
}

/** The directed joint matrix A[i][j] = cond[i][j] * M[j], exact. */
export function directedJointMatrix(
  cond: RationalMatrix,
  masses: MassVector,
): ExactFraction[][] {
  const matrix = asRationalMatrix(cond);
  const mass = asMassVector(masses, matrix.length);
  return matrix.map((row) =>
    row.map((entry, j) => mulFractions(entry, mass[j]!)),
  );
}

/** The antisymmetric residual matrix A[i][j] - A[j][i], exact. */
export function antisymmetricResidualMatrix(
  cond: RationalMatrix,
  masses: MassVector,
): ExactFraction[][] {
  const joint = directedJointMatrix(cond, masses);
  const n = joint.length;
  const out: ExactFraction[][] = [];
  for (let i = 0; i < n; i++) {
    const row: ExactFraction[] = [];
    for (let j = 0; j < n; j++) row.push(subFractions(joint[i]![j]!, joint[j]![i]!));
    out.push(row);
  }
  return out;
}

/** True when the directed joint matrix is symmetric. */
export function directedJointIsSymmetric(cond: RationalMatrix, masses: MassVector): boolean {
  const joint = directedJointMatrix(cond, masses);
  for (let i = 0; i < joint.length; i++) {
    for (let j = 0; j < joint.length; j++) {
      if (compareFractions(joint[i]![j]!, joint[j]![i]!) !== 0) return false;
    }
  }
  return true;
}

/** The direction-blind check: compare cond entries, ignore the masses. */
export function directionBlindFindsArbitrage(cond: RationalMatrix): boolean {
  const matrix = asRationalMatrix(cond);
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      if (compareFractions(matrix[i]![j]!, matrix[j]![i]!) !== 0) return true;
    }
  }
  return false;
}

/** The adversarial dual: mass-blind checking misses real inconsistencies. */
export interface DirectionBlindDual {
  readonly lawFindsArbitrage: boolean;
  readonly directionBlindFinds: boolean;
  /** True when the full law finds an inconsistency the blind check misses. */
  readonly fires: boolean;
  readonly witness: ArbitrageWitness | null;
  readonly detail: string;
}

/**
 * Run both checkers. The blind checker compares cond[i][j] with cond[j][i] and
 * drops M. Whenever the masses are unequal it can declare consistency on a
 * matrix the full law rejects, so the dual fires and the witness is returned.
 */
export function arbitrageAdversarialDual(
  cond: RationalMatrix,
  masses: MassVector,
): DirectionBlindDual {
  const report = findsArbitrage(cond, masses);
  const blind = directionBlindFindsArbitrage(cond);
  const fires = report.arbitrage && !blind;
  return {
    lawFindsArbitrage: report.arbitrage,
    directionBlindFinds: blind,
    fires,
    witness: report.witness,
    detail: fires
      ? 'mass-weighted law rejects pair (' + String(report.witness!.i) + ', ' + String(report.witness!.j) +
        ') while the direction-blind check sees symmetry; the masses are the missing axis'
      : 'the direction-blind and mass-weighted checks agree here',
  };
}
