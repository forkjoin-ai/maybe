/**
 * Gauss-map die -- when does a convex polyhedron realize a probability vector?
 *
 * A die tumbling with uniformly random orientation lands on facet F with
 * probability equal to the solid angle of F's normal cone (its Gauss image),
 * divided by 4*pi. So a probability vector is a solid-angle sequence.
 *
 * Reduction: the landing probabilities are the areas of the spherical Voronoi
 * cells of the facet normals. Realizing an arbitrary vector is therefore an
 * area-constrained spherical Voronoi problem.
 *
 * This module solves the exactly solvable sub-family: put every normal on a
 * COMMON LATITUDE. The bisector of two same-latitude points is a meridian, so
 * each Voronoi cell is a full lune whose area is g_{i-1} + g_i, where g_i is the
 * longitude gap between neighbours. Requiring area_i = 4*pi*p_i gives the cyclic
 * linear system
 *
 *     g_{i-1} + g_i = 4*pi*p_i,   sum g = 2*pi.
 *
 * It has a unique solution for odd n, and a solution for even n only when the
 * alternating condition p_0 + p_2 + ... = p_1 + p_3 + ... holds. That is a real
 * obstruction for the equatorial family: an even-outcome die cannot always be
 * built by putting every facet normal on one circle.
 *
 * General n is not settled here. Surjectivity of the spherical-Voronoi area map
 * is the semi-discrete optimal-transport result of Aurenhammer-Hoffmann-Aronov
 * and Kitagawa-Merigot-Thibert (CITED, not proved in this repo).
 */

export const TWO_PI = 2 * Math.PI;
export const FOUR_PI = 4 * Math.PI;

export interface EquatorialVoronoi {
  /** True when the cyclic gaps exist. */
  readonly solvable: boolean;
  /** c_0 - S for even n; must be 0. Zero for odd n by construction. */
  readonly alternatingResidual: number;
  readonly gaps: readonly number[] | null;
  readonly realizedAreas: readonly number[] | null;
  readonly maxAreaError: number | null;
  readonly gapSum: number | null;
}

/**
 * Solve the equatorial die for a probability vector.
 *
 * The alternating condition is p_0 + p_2 + ... = p_1 + p_3 + ... for even n;
 * for odd n there is always a unique equatorial realization.
 */
export function analyzeEquatorialVoronoi(probabilities: readonly number[]): EquatorialVoronoi {
  const n = probabilities.length;
  if (n < 2) throw new RangeError('need at least two outcomes');
  let total = 0;
  for (const p of probabilities) {
    if (!Number.isFinite(p) || p <= 0) throw new TypeError('probabilities must be finite and positive');
    total += p;
  }
  if (Math.abs(total - 1) > 1e-9) throw new RangeError('probabilities must sum to one');

  const c = probabilities.map((p) => FOUR_PI * p);
  let s = 0;
  for (let j = 1; j <= n - 1; j++) {
    s += ((n - 1 - j) % 2 === 0 ? 1 : -1) * c[j];
  }
  const residual = c[0] - s;
  const odd = n % 2 === 1;
  const solvable = odd ? true : Math.abs(residual) < 1e-9;

  const gaps = new Array<number>(n).fill(0);
  gaps[0] = odd ? (c[0] - s) / 2 : 0;
  for (let i = 1; i < n; i++) gaps[i] = c[i] - gaps[i - 1];

  if (!solvable) {
    return { solvable: false, alternatingResidual: residual, gaps: null, realizedAreas: null, maxAreaError: null, gapSum: null };
  }
  const realized = c.map((_, i) => gaps[(i - 1 + n) % n] + gaps[i]);
  const maxAreaError = Math.max(...c.map((ci, i) => Math.abs(ci - realized[i])));
  const gapSum = gaps.reduce((a, b) => a + b, 0);
  return { solvable: true, alternatingResidual: residual, gaps, realizedAreas: realized, maxAreaError, gapSum };
}

/** The alternating condition for even n: even-index mass equals odd-index mass. */
export function equatorialAlternatingCondition(probabilities: readonly number[]): boolean {
  const n = probabilities.length;
  if (n % 2 === 1) return true;
  let even = 0;
  let oddMass = 0;
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) even += probabilities[i];
    else oddMass += probabilities[i];
  }
  return Math.abs(even - oddMass) < 1e-9;
}
