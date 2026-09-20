/**
 * tensor-bayes-johnson.ts -- the Johnson scheme J(11,2) on the fifty-five axes.
 *
 * Mirrors `Gnosis.TensorBayesJohnson`. The 55 axes of the hyper-joint are the
 * edges of the complete graph `K_11` (walkers 0..10). Two DISTINCT axes are
 * ADJACENT when they share exactly one walker, DISJOINT when they share none.
 * This is a strongly regular graph with parameters `(55, 18, 9, 4)`: every
 * axis has 18 adjacent and 36 disjoint neighbours, adjacent axes share 9
 * common adjacent neighbours, disjoint axes share 4.
 *
 * Everything here is EXECUTABLE brute force at n=55: a 55x55 integer matrix
 * multiply is 166,375 operations, trivial at runtime. Nothing here proves
 * anything; it is the runnable check that the same arithmetic the Lean
 * kernel-checks also holds when actually computed.
 */

export const WALKERS = 11;
export const AXES = 55;

/** The 55 axes, in a fixed (arbitrary but consistent) lexicographic order. */
export const EDGES: ReadonlyArray<readonly [number, number]> = (() => {
  const edges: Array<[number, number]> = [];
  for (let i = 0; i < WALKERS; i++) {
    for (let j = i + 1; j < WALKERS; j++) edges.push([i, j]);
  }
  return edges;
})();

if (EDGES.length !== AXES) {
  throw new Error(`edge enumeration produced ${EDGES.length} axes, expected ${AXES}`);
}

/** `edgeAt`: the walker pair carried by axis `a`. */
export function edgeAt(a: number): readonly [number, number] {
  return EDGES[a]!;
}

const EDGE_INDEX = new Map<string, number>();
EDGES.forEach(([i, j], idx) => {
  EDGE_INDEX.set(`${i},${j}`, idx);
  EDGE_INDEX.set(`${j},${i}`, idx);
});

/** `edgeIndex`: the axis carrying walker pair `(i, j)`, `i !== j`. */
export function edgeIndex(i: number, j: number): number {
  const idx = EDGE_INDEX.get(`${i},${j}`);
  if (idx === undefined) throw new Error(`no axis for walker pair (${i}, ${j})`);
  return idx;
}

/** `shareCount`: the number of walkers axes `a` and `b` have in common. */
export function shareCount(a: number, b: number): number {
  const [p1, p2] = EDGES[a]!;
  const [q1, q2] = EDGES[b]!;
  let count = 0;
  if (p1 === q1 || p1 === q2) count++;
  if (p2 === q1 || p2 === q2) count++;
  return count;
}

/** `delta`: the identity matrix `I` of the scheme. */
export function delta(a: number, b: number): number {
  return a === b ? 1 : 0;
}

/** `adjM`: the adjacency matrix `A` (triangular graph `T(11)`). */
export function adjM(a: number, b: number): number {
  return shareCount(a, b) === 1 ? 1 : 0;
}

/** `disjM`: the disjointness matrix `D` (Kneser graph `K(11,2)`). */
export function disjM(a: number, b: number): number {
  return shareCount(a, b) === 0 ? 1 : 0;
}

export type Matrix = number[][];

function buildMatrix(entry: (a: number, b: number) => number): Matrix {
  const m: Matrix = Array.from({ length: AXES }, () => new Array(AXES).fill(0));
  for (let a = 0; a < AXES; a++) for (let b = 0; b < AXES; b++) m[a]![b] = entry(a, b);
  return m;
}

/** The identity, adjacency and disjointness matrices, built once. */
export const I: Matrix = buildMatrix(delta);
export const A: Matrix = buildMatrix(adjM);
export const D: Matrix = buildMatrix(disjM);
/** `J`: the all-ones 55x55 matrix. */
export const J: Matrix = buildMatrix(() => 1);

/** `johnson_trichotomy`: exactly one of same/adjacent/disjoint holds. */
export function johnsonTrichotomy(a: number, b: number): 'same' | 'adjacent' | 'disjoint' {
  const s = shareCount(a, b);
  if (s === 2) return 'same';
  if (s === 1) return 'adjacent';
  return 'disjoint';
}

/** `partition_matrix`: I + A + D = J at every entry. */
export function partitionHolds(): boolean {
  for (let a = 0; a < AXES; a++) {
    for (let b = 0; b < AXES; b++) {
      if (I[a]![b]! + A[a]![b]! + D[a]![b]! !== J[a]![b]!) return false;
    }
  }
  return true;
}

/** Row sum of a matrix at a fixed row. */
function rowSum(m: Matrix, a: number): number {
  let sum = 0;
  for (let b = 0; b < AXES; b++) sum += m[a]![b]!;
  return sum;
}

/** `adjRow_eq_18`: every axis has exactly 18 adjacent neighbours. */
export function adjacentDegree(a: number): number {
  return rowSum(A, a);
}

/** `disjRow_eq_36`: every axis has exactly 36 disjoint neighbours. */
export function disjointDegree(a: number): number {
  return rowSum(D, a);
}

/** Matrix product, plain 55x55 integer multiply. */
export function matMul(X: Matrix, Y: Matrix): Matrix {
  const out: Matrix = Array.from({ length: AXES }, () => new Array(AXES).fill(0));
  for (let a = 0; a < AXES; a++) {
    for (let b = 0; b < AXES; b++) {
      let acc = 0;
      for (let c = 0; c < AXES; c++) acc += X[a]![c]! * Y[c]![b]!;
      out[a]![b] = acc;
    }
  }
  return out;
}

function matAdd(X: Matrix, Y: Matrix): Matrix {
  return X.map((row, a) => row.map((v, b) => v + Y[a]![b]!));
}

function matScale(k: number, X: Matrix): Matrix {
  return X.map((row) => row.map((v) => k * v));
}

function matEqual(X: Matrix, Y: Matrix): boolean {
  for (let a = 0; a < AXES; a++) for (let b = 0; b < AXES; b++) if (X[a]![b] !== Y[a]![b]) return false;
  return true;
}

/** `A * A`, computed once. */
export const AA: Matrix = matMul(A, A);

/** `adjSq_general`: `A*A = 18*I + 9*A + 4*D`, checked at every entry. */
export function adjSqGeneralHolds(): boolean {
  const rhs = matAdd(matAdd(matScale(18, I), matScale(9, A)), matScale(4, D));
  return matEqual(AA, rhs);
}

/** `adjSq_reduced_int`: `A*A - 5*A - 14*I = 4*J` (equivalently `A*A = 14*I + 5*A + 4*J`). */
export function adjSqReducedHolds(): boolean {
  const rhs = matAdd(matAdd(matScale(14, I), matScale(5, A)), matScale(4, J));
  return matEqual(AA, rhs);
}

/** `commonAdjSum`: the un-normalized (a,b)-entry of `A*A`. */
export function commonAdjSum(a: number, b: number): number {
  return AA[a]![b]!;
}

/** `johnson_lambda_general`: adjacent axes have exactly 9 common adjacent neighbours. */
export function johnsonLambdaHolds(): boolean {
  for (let a = 0; a < AXES; a++) {
    for (let b = 0; b < AXES; b++) {
      if (shareCount(a, b) === 1 && commonAdjSum(a, b) !== 9) return false;
    }
  }
  return true;
}

/** `johnson_mu_general`: disjoint axes have exactly 4 common adjacent neighbours. */
export function johnsonMuHolds(): boolean {
  for (let a = 0; a < AXES; a++) {
    for (let b = 0; b < AXES; b++) {
      if (shareCount(a, b) === 0 && commonAdjSum(a, b) !== 4) return false;
    }
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// The spectrum witnesses
// ═══════════════════════════════════════════════════════════════════════════

/** `star(x)`: the 0/1 indicator vector of the 10 axes through walker `x`. */
export function star(x: number): number[] {
  return EDGES.map(([i, j]) => (i === x || j === x ? 1 : 0));
}

/** `A` applied to a length-55 integer vector. */
export function matVec(m: Matrix, v: readonly number[]): number[] {
  const out = new Array(AXES).fill(0);
  for (let a = 0; a < AXES; a++) {
    let acc = 0;
    for (let b = 0; b < AXES; b++) acc += m[a]![b]! * v[b]!;
    out[a] = acc;
  }
  return out;
}

function vecEqual(u: readonly number[], v: readonly number[]): boolean {
  return u.length === v.length && u.every((x, i) => x === v[i]);
}

const ONES: readonly number[] = new Array(AXES).fill(1);

/**
 * `star_contract`: `A * star(x) = 7 * star(x) + 2 * J` (J the all-ones
 * VECTOR here), for every walker `x`. Checked at every axis.
 */
export function starContractHolds(x: number): boolean {
  const sx = star(x);
  const lhs = matVec(A, sx);
  const rhs = sx.map((v, i) => 7 * v + 2 * ONES[i]!);
  return vecEqual(lhs, rhs);
}

/**
 * `starDiff_eigen7`: for any two walkers `x !== y`, `star(x) - star(y)` is a
 * 7-eigenvector of `A` at every axis (the `+2*J` terms cancel in the
 * difference).
 */
export function starDiffEigen7Holds(x: number, y: number): boolean {
  const diff = star(x).map((v, i) => v - star(y)[i]!);
  const lhs = matVec(A, diff);
  const rhs = diff.map((v) => 7 * v);
  return vecEqual(lhs, rhs);
}

/** `star_overlap`: <star(x), star(y)> = 10 if x = y, else 1. */
export function starOverlap(x: number, y: number): number {
  const sx = star(x);
  const sy = star(y);
  let acc = 0;
  for (let a = 0; a < AXES; a++) acc += sx[a]! * sy[a]!;
  return acc;
}

/** `starV(x) := star(x) - star(0)`, the eigenvalue-7 witness family. */
export function starV(x: number): number[] {
  const s0 = star(0);
  return star(x).map((v, i) => v - s0[i]!);
}

/** `pairStar`: the dot product of a length-55 vector against `star(y)`. */
export function pairStar(v: readonly number[], y: number): number {
  const sy = star(y);
  let acc = 0;
  for (let a = 0; a < AXES; a++) acc += v[a]! * sy[a]!;
  return acc;
}

/**
 * `pairStar_starV`: for `y != 0`, the family `starV(1..10)` pairs against
 * `star(y)` as `9` on the diagonal (`x = y`) and `0` off it -- the diagonal
 * pairing that witnesses independence of the 10-vector eigenvalue-7 family,
 * with no determinant and no rank theory.
 */
export function pairStarStarVHolds(x: number, y: number): boolean {
  if (y === 0) throw new Error('pairStar_starV requires y != 0');
  const expected = x === y ? 9 : 0;
  return pairStar(starV(x), y) === expected;
}

/**
 * `eigen7_multiplicity_ge_ten`: the 10x10 pairing matrix of `starV(1..10)`
 * against `star(1..10)` is EXACTLY `9*I10`. A diagonal, invertible matrix
 * proves the 10 vectors are linearly independent -- a LOWER bound of 10 on
 * the eigenvalue-7 multiplicity, by explicit witness, not a dimension
 * argument (which this corpus does not have; see Honest boundary).
 */
export function eigen7MultiplicityWitness(): { pairingIsNineI: boolean; allNonzero: boolean; allEigen7: boolean } {
  let pairingIsNineI = true;
  for (let x = 1; x <= 10; x++) {
    for (let y = 1; y <= 10; y++) {
      if (!pairStarStarVHolds(x, y)) pairingIsNineI = false;
    }
  }
  let allNonzero = true;
  for (let x = 1; x <= 10; x++) {
    if (starV(x).every((v) => v === 0)) allNonzero = false;
  }
  let allEigen7 = true;
  for (let x = 1; x <= 10; x++) {
    const v = starV(x);
    if (!vecEqual(matVec(A, v), v.map((c) => 7 * c))) allEigen7 = false;
  }
  return { pairingIsNineI, allNonzero, allEigen7 };
}

/**
 * `quadZ`: the ordered quadrilateral `e{p,q} + e{r,s} - e{p,r} - e{q,s}`.
 * Requires the Lean order guards `p<q`, `r<s`, `p<r`, `q<s` so all four edges
 * are well-formed and distinct.
 */
export function quadZ(p: number, q: number, r: number, s: number): number[] {
  if (!(p < q && r < s && p < r && q < s)) {
    throw new Error('quadZ requires p<q, r<s, p<r, q<s');
  }
  const v = new Array(AXES).fill(0);
  v[edgeIndex(p, q)]! += 1;
  v[edgeIndex(r, s)]! += 1;
  v[edgeIndex(p, r)]! -= 1;
  v[edgeIndex(q, s)]! -= 1;
  return v;
}

/**
 * `quad_eigenNeg2`: every ordered quadrilateral is a (-2)-eigenvector of `A`
 * at every axis.
 */
export function quadEigenNeg2Holds(p: number, q: number, r: number, s: number): boolean {
  const v = quadZ(p, q, r, s);
  const lhs = matVec(A, v);
  const rhs = v.map((c) => -2 * c);
  return vecEqual(lhs, rhs);
}

/** Every strictly increasing 4-subset of the 11 walkers, sorted -- each gives
 *  a well-formed quadrilateral quadZ(w0,w1,w2,w3) (C(11,4) = 330 of them). */
export function increasingQuadruples(): Array<readonly [number, number, number, number]> {
  const out: Array<[number, number, number, number]> = [];
  for (let w0 = 0; w0 < WALKERS; w0++) {
    for (let w1 = w0 + 1; w1 < WALKERS; w1++) {
      for (let w2 = w1 + 1; w2 < WALKERS; w2++) {
        for (let w3 = w2 + 1; w3 < WALKERS; w3++) out.push([w0, w1, w2, w3]);
      }
    }
  }
  return out;
}

/** `highAxis`: an axis whose smaller walker is >= 2 (both walkers >= 2, since
 *  the smaller is the min). There are exactly 36 of them (`highAxis_count`). */
export function isHighAxis(t: number): boolean {
  return edgeAt(t)[0] >= 2;
}

export function highAxisCount(): number {
  let count = 0;
  for (let t = 0; t < AXES; t++) if (isHighAxis(t)) count++;
  return count;
}

/** `quadFam(t)`: the quadrilateral on walkers `0, 1, i, j` where `{i,j} = edgeAt(t)`. */
export function quadFam(t: number): number[] {
  const [i, j] = edgeAt(t);
  return quadZ(0, 1, i, j);
}

/**
 * `eigenNeg2_multiplicity_ge_36`: 36 explicit, mutually independent
 * (-2)-eigenvectors, one per high axis. Independence is by DIAGONAL
 * EVALUATION: `quadFam(t')` evaluated at high axis `t` is `1` if `t = t'`
 * and `0` otherwise (`quadFam_eval`), so a vanishing combination has every
 * coefficient forced to zero by reading off its own private axis. A LOWER
 * bound of 36 (of the conjectured 44), by explicit witness.
 */
export function eigenNeg2MultiplicityWitness(): { count36: boolean; diagonalEval: boolean; allEigenNeg2: boolean } {
  const highAxes: number[] = [];
  for (let t = 0; t < AXES; t++) if (isHighAxis(t)) highAxes.push(t);
  const count36 = highAxes.length === 36;

  let diagonalEval = true;
  for (const tPrime of highAxes) {
    const fam = quadFam(tPrime);
    for (const t of highAxes) {
      const expected = t === tPrime ? 1 : 0;
      if (fam[t] !== expected) diagonalEval = false;
    }
  }

  let allEigenNeg2 = true;
  for (const t of highAxes) {
    const v = quadFam(t);
    if (!vecEqual(matVec(A, v), v.map((c) => -2 * c))) allEigenNeg2 = false;
  }

  return { count36, diagonalEval, allEigenNeg2 };
}

/**
 * `multiplicity_solves`: `(1, 10, 44)` solves the trace-zero / dimension-55
 * consistency equations for a hypothetical spectrum `{18, 7, -2}`. THIS IS A
 * CONSISTENCY CHECK, NOT A PROOF: it does not establish the spectrum is
 * exactly `{18, 7, -2}`, nor that these multiplicities are forced or unique.
 * The corpus proves only the LOWER bounds (10 and 36, both witnessed above).
 */
export function multiplicitySolves(): { traceZero: boolean; dimensionFifty5: boolean } {
  return {
    traceZero: 18 + 7 * 10 + -2 * 44 === 0,
    dimensionFifty5: 1 + 10 + 44 === 55,
  };
}

/** `allOnes_eigen18`: the all-ones vector is an 18-eigenvector of `A`. */
export function allOnesEigen18Holds(): boolean {
  const lhs = matVec(A, ONES);
  const rhs = ONES.map((v) => 18 * v);
  return vecEqual(lhs, rhs);
}

/** `johnson_partition`/counting identities: 55 axes, 495 adjacent + 990
 *  disjoint unordered pairs = 1485 total pairs. */
export function pleromaCounts(): { axes: number; adjacentPairs: number; disjointPairs: number; totalPairs: number } {
  let adjacentPairs = 0;
  let disjointPairs = 0;
  for (let a = 0; a < AXES; a++) {
    for (let b = a + 1; b < AXES; b++) {
      const s = shareCount(a, b);
      if (s === 1) adjacentPairs++;
      else if (s === 0) disjointPairs++;
    }
  }
  return {
    axes: AXES,
    adjacentPairs,
    disjointPairs,
    totalPairs: adjacentPairs + disjointPairs,
  };
}

/** `jointIA_not_decomposable`: `I + A` is symmetric but violates the
 *  Plücker relation `J_ij * J_kl = J_ik * J_jl` at a concrete witness, so it
 *  is NOT a rank-one joint -- the corpus's rank-one/star-block results
 *  therefore cut something non-vacuous. */
export function jointIA(a: number, b: number): number {
  return delta(a, b) + adjM(a, b);
}

export function jointIAViolatesPlucker(i: number, j: number, k: number, l: number): boolean {
  return jointIA(i, j) * jointIA(k, l) !== jointIA(i, k) * jointIA(j, l);
}
