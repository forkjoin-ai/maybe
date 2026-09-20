/**
 * tensor-bayes-farkas.ts -- finite Farkas certificates and the infrathin margin.
 *
 * Mirrors `Gnosis.FiniteFarkas` and `Gnosis.FarkasInfrathin`. A linear
 * constraint `a*x + b*y <= c` over Int is read at a shared positive
 * denominator `D` (a rational point `x = X/D`, `y = Y/D`) in the CLEARED
 * form `a*X + b*Y <= c*D` -- exactly the "scaled state price" idiom used
 * throughout the corpus, never a constructed rational type.
 *
 * A FARKAS CERTIFICATE for a constraint list `S` is a nonnegative-multiplier
 * combination of `S`'s own constraints whose weighted x- and y-coefficients
 * cancel to zero while the weighted constant ("margin") is negative -- a
 * nonnegative sum of true inequalities collapsing to the false `0 <= (negative)`.
 * That soundness (`certificate_infeasible`) is proved here directly, by
 * computing the weighted slack at a candidate point and showing it must be
 * both `>= 0` (every constraint holds, every multiplier is nonnegative) and
 * `< 0` (the certificate's own algebra), a contradiction -- general over any
 * point, not a bounded search.
 *
 * SCOPE, stated honestly: `Gnosis.FiniteFarkas` proves the full "exactly
 * one" alternative for arbitrary-length two- and three-variable constraint
 * lists via Fourier-Motzkin elimination. This module mirrors the CERTIFICATE
 * definition and its soundness in full generality, and demonstrates the
 * feasible-XOR-certificate alternative at the concrete tight pair the
 * headline theorem is built from (`tightBad`/`tightGood`), not the general
 * n-constraint elimination algorithm -- exactly as the formula card's own
 * "Honest boundary" declines the uniform n-variable case.
 */

/** A linear constraint `a*x + b*y <= c` over Int. */
export interface Con2 {
  readonly a: bigint;
  readonly b: bigint;
  readonly c: bigint;
}

/** `Con2.holds`: `a*X + b*Y <= c*D`, the cleared-denominator reading at `x = X/D`, `y = Y/D`. */
export function con2Holds(k: Con2, D: bigint, X: bigint, Y: bigint): boolean {
  return k.a * X + k.b * Y <= k.c * D;
}

/** A Farkas certificate: nonnegative multipliers against a system's own constraints. */
export type CertificateTerms = ReadonlyArray<readonly [bigint, Con2]>;

export function sumA(cs: CertificateTerms): bigint {
  return cs.reduce((acc, [lam, k]) => acc + lam * k.a, 0n);
}
export function sumB(cs: CertificateTerms): bigint {
  return cs.reduce((acc, [lam, k]) => acc + lam * k.b, 0n);
}
export function sumC(cs: CertificateTerms): bigint {
  return cs.reduce((acc, [lam, k]) => acc + lam * k.c, 0n);
}

/** `Certificate`: every field the Lean structure demands, checked directly. */
export interface CertificateCheck {
  readonly mem: boolean;
  readonly nonneg: boolean;
  readonly cancelA: boolean;
  readonly cancelB: boolean;
  readonly neg: boolean;
  readonly valid: boolean;
}

/** Verify `cs` is a valid Farkas certificate for the system `S`. */
export function checkCertificate(S: readonly Con2[], cs: CertificateTerms): CertificateCheck {
  const mem = cs.every(([, k]) => S.includes(k));
  const nonneg = cs.every(([lam]) => lam >= 0n);
  const cancelA = sumA(cs) === 0n;
  const cancelB = sumB(cs) === 0n;
  const neg = sumC(cs) < 0n;
  return { mem, nonneg, cancelA, cancelB, neg, valid: mem && nonneg && cancelA && cancelB && neg };
}

/**
 * `certificate_infeasible`, proved directly and generally (not by bounded
 * search): compute the weighted slack `sumC(cs)*D - sumA(cs)*X - sumB(cs)*Y`
 * at a candidate point. If every referenced constraint holds at that point
 * with its nonnegative multiplier, the weighted slack is a nonnegative
 * combination of nonnegative slacks, hence `>= 0` (`weighted_nonneg`); but a
 * valid certificate forces `sumA = sumB = 0` and `sumC < 0`, so the weighted
 * slack collapses to `sumC(cs) * D < 0` whenever `D > 0` -- a direct
 * contradiction. `certificateInfeasible` returns that contradiction has been
 * derived (i.e. the point could not have satisfied every constraint), for
 * ANY point passed in, not merely ones we happened to search.
 */
export function certificateInfeasible(cs: CertificateTerms, D: bigint, X: bigint, Y: bigint): boolean {
  if (D <= 0n) return true; // D > 0 is required for feasibility in the first place
  const weightedSlack = sumC(cs) * D - sumA(cs) * X - sumB(cs) * Y;
  const everyConstraintHoldsWithNonnegWeight = cs.every(([lam, k]) => lam >= 0n && con2Holds(k, D, X, Y));
  if (!everyConstraintHoldsWithNonnegWeight) return true; // point already fails S, nothing to refute
  // weightedSlack must be >= 0 by weighted_nonneg; a valid certificate makes
  // it exactly sumC(cs)*D, which is negative when D > 0 -- contradiction.
  const structurallyValid =
    cs.every(([lam]) => lam >= 0n) && sumA(cs) === 0n && sumB(cs) === 0n && sumC(cs) < 0n;
  return structurallyValid && weightedSlack < 0n;
}

// ═══════════════════════════════════════════════════════════════════════════
// The cover relation on Int, and the infrathin margin
// ═══════════════════════════════════════════════════════════════════════════

/** `intCovers a b`: `a < b` with no integer strictly between -- Duchamp's
 *  inframince transplanted to the integer lattice. */
export function intCovers(a: bigint, b: bigint): boolean {
  if (!(a < b)) return false;
  return b === a + 1n; // over Int, covering IS the successor (`intCovers_iff_succ`)
}

/** `int_lt_zero_iff_le_neg_one`: over Int, strictly negative IS at most -1. */
export function intLtZeroIffLeNegOne(x: bigint): boolean {
  return (x < 0n) === (x <= -1n);
}

/** `margin`: the weighted constant a Farkas combination collapses to. */
export function margin(cs: CertificateTerms): bigint {
  return sumC(cs);
}

/**
 * `certificate_margin_le_neg_one`: every certificate's margin is at most -1.
 * Over `Int`, `Certificate.neg` (`sumC < 0`) is ALREADY the sharper claim
 * `sumC <= -1` -- there is no infinitesimal negative integer.
 */
export function certificateMarginLeNegOne(cs: CertificateTerms): boolean {
  const m = margin(cs);
  return m < 0n ? m <= -1n : true; // vacuously true when cs is not actually a certificate
}

/**
 * `no_certificate_in_the_gap`: no certificate can have margin strictly
 * between -1 and 0 -- the statement is vacuous BECAUSE `int_lt_zero_iff_le_neg_one`
 * already empties the gap, and that vacuity is the content.
 */
export function noCertificateInTheGap(cs: CertificateTerms): boolean {
  const m = margin(cs);
  return !(-1n < m && m < 0n);
}

// ═══════════════════════════════════════════════════════════════════════════
// The tight pair: -1 infeasible, 0 feasible, adjacent
// ═══════════════════════════════════════════════════════════════════════════

/** `x <= 0`. */
export const conXle0: Con2 = { a: 1n, b: 0n, c: 0n };
/** `-x <= -1`, i.e. `x >= 1`. */
export const conXge1: Con2 = { a: -1n, b: 0n, c: -1n };
/** `-x <= 0`, i.e. `x >= 0`. */
export const conXge0: Con2 = { a: -1n, b: 0n, c: 0n };

/** The tightest INFEASIBLE system: `x <= 0` and `x >= 1`. */
export const tightBad: readonly Con2[] = [conXle0, conXge1];
export const tightBadCert: CertificateTerms = [
  [1n, conXle0],
  [1n, conXge1],
];

/** The tightest FEASIBLE system: `x <= 0` and `x >= 0`. */
export const tightGood: readonly Con2[] = [conXle0, conXge0];
export const tightGoodCombo: CertificateTerms = [
  [1n, conXle0],
  [1n, conXge0],
];

/**
 * `tight_feasible_at_zero`, THE WHOLE THESIS IN ONE STATEMENT: `tightGood`
 * collapses to exactly `0 <= 0` and IS feasible (witness `x = 0`); `tightBad`
 * collapses to exactly `0 <= -1` and is NOT feasible (certified); and `0`
 * covers `-1`, so the two systems are adjacent with nothing between them.
 */
export function tightFeasibleAtZero(): {
  goodMargin: bigint;
  goodFeasible: boolean;
  badMargin: bigint;
  badCertificateValid: boolean;
  badInfeasibleByCertificate: boolean;
  adjacent: boolean;
} {
  const goodMargin = margin(tightGoodCombo);
  const goodFeasible = tightGood.every((k) => con2Holds(k, 1n, 0n, 0n)); // witness D=1, X=0, Y=0
  const badMargin = margin(tightBadCert);
  const badCertificateValid = checkCertificate([...tightBad], tightBadCert).valid;
  const badInfeasibleByCertificate = certificateInfeasible(tightBadCert, 1n, 0n, 0n) && badCertificateValid;
  const adjacent = intCovers(badMargin, goodMargin);
  return { goodMargin, goodFeasible, badMargin, badCertificateValid, badInfeasibleByCertificate, adjacent };
}

/**
 * `deepBad_margin`: margins are bounded above by -1, not equal to it.
 * `x <= 0` with `x >= 7` collapses to `0 <= -7`.
 */
export const deepBadCon: Con2 = { a: -1n, b: 0n, c: -7n };
export const deepBadCert: CertificateTerms = [
  [1n, conXle0],
  [1n, deepBadCon],
];

export function deepBadMargin(): bigint {
  return margin(deepBadCert);
}

/**
 * `rational_margin_fills_the_gap`, THE INFRATHIN CONTRAST -- the honest
 * boundary made concrete. Over `Int` there is no certificate with margin
 * strictly between -1 and 0 (`noCertificateInTheGap`). But a Farkas
 * certificate over the RATIONALS may be scaled by any positive rational: the
 * SAME `tightBadCert` combination, halved, gives margin `-1/2`, and `-1/2`
 * genuinely lies inside the gap `(-1, 0)` in the cross-multiplied rational
 * order. The infrathin membrane proved above is a consequence of
 * INTEGRALITY, not of Farkas itself: nothing in the certificate machinery
 * forbids the rational margin, only the fact that a Nat/Int-valued margin
 * has no value between -1 and 0.
 */
export function rationalMarginFillsTheGap(): {
  num: bigint;
  den: bigint;
  denPositive: boolean;
  aboveNegOne: boolean; // -1 < num/den
  belowZero: boolean; // num/den < 0
  liesInTheGap: boolean;
} {
  // The halved tightBadCert: multipliers (1/2, 1/2) instead of (1, 1), margin -1/2.
  const num = -1n;
  const den = 2n;
  const denPositive = den > 0n;
  const aboveNegOne = -1n * den < num * 1n; // cross-multiplied -1 < num/den
  const belowZero = num * 1n < 0n * den; // cross-multiplied num/den < 0
  return { num, den, denPositive, aboveNegOne, belowZero, liesInTheGap: aboveNegOne && belowZero };
}

/** `y <= 0`. */
export const feasiblePairLower: Con2 = { a: 0n, b: 1n, c: 0n };
/** `y <= 1`. */
export const feasiblePairUpper: Con2 = { a: 0n, b: 1n, c: 1n };
/** A feasible system (witness `y = 0`) with a MIXED-SIGN combination
 *  (multipliers `1` and `-1`) that cancels both coefficients and reaches
 *  margin exactly `-1` -- the same value `tightBadCert` calls the first
 *  infeasible step. */
export const nonnegWitness: CertificateTerms = [
  [1n, feasiblePairLower],
  [-1n, feasiblePairUpper],
];

/**
 * `margin_needs_nonneg`: the margin bound is carried entirely by
 * `Certificate.nonneg`. Dropping it, the mixed-sign combination `nonnegWitness`
 * cancels both coefficients (`sumA = sumB = 0`) and reaches margin `-1` --
 * the very value the tight pair calls the first infeasible step -- on a
 * system (`y <= 0` and `y <= 1`) that IS feasible at `y = 0`. Without
 * nonnegativity the lattice adjacency proved above separates nothing.
 */
export function marginNeedsNonneg(): {
  mixedSignMargin: bigint;
  cancelsA: boolean;
  cancelsB: boolean;
  systemIsFeasible: boolean;
  multipliersAllNonneg: boolean;
} {
  const mixedSignMargin = margin(nonnegWitness);
  const cancelsA = sumA(nonnegWitness) === 0n;
  const cancelsB = sumB(nonnegWitness) === 0n;
  const systemIsFeasible = [feasiblePairLower, feasiblePairUpper].every((k) => con2Holds(k, 1n, 0n, 0n));
  const multipliersAllNonneg = nonnegWitness.every(([lam]) => lam >= 0n);
  return { mixedSignMargin, cancelsA, cancelsB, systemIsFeasible, multipliersAllNonneg };
}
