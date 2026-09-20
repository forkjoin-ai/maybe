/**
 * tensor-bayes-laplace-gap.ts -- the price of the never-collapse floor.
 *
 * Mirrors `Gnosis.TensorBayesLaplaceGap`. The exact gap between the Laplace
 * (add-one/Buleyean) estimate and the raw frequency is
 *
 *   (n+1)/(N+K) - n/N  =  (N - K*n) / (N*(N+K))
 *
 * Every statement below is the CROSS-MULTIPLIED Nat/BigInt form the Lean
 * states, never a floating-point division. `n` is a colour's count out of
 * `N` total draws over `K` colours.
 */

/** SHARP OVERESTIMATE. `laplace_over_sharp`: holds with NO guards at all.
 *  `(n+1)*N <= n*(N+K) + N`. */
export function laplaceOverSharp(n: bigint, N: bigint, K: bigint): boolean {
  return (n + 1n) * N <= n * (N + K) + N;
}

/** SHARP UNDERESTIMATE. `laplace_under_sharp`: needs only `n <= N`.
 *  `n*(N+K) + N <= (n+1)*N + K*N`. */
export function laplaceUnderSharp(n: bigint, N: bigint, K: bigint): boolean {
  return n * (N + K) + N <= (n + 1n) * N + K * N;
}

/** LOOSE OVERESTIMATE. `laplace_over_loose`: needs `K >= 1`.
 *  `(n+1)*N <= n*(N+K) + K*N`. */
export function laplaceOverLoose(n: bigint, N: bigint, K: bigint): boolean {
  return (n + 1n) * N <= n * (N + K) + K * N;
}

/** LOOSE UNDERESTIMATE. `laplace_under_loose`: needs `n <= N`.
 *  `n*(N+K) <= (n+1)*N + K*N`. */
export function laplaceUnderLoose(n: bigint, N: bigint, K: bigint): boolean {
  return n * (N + K) <= (n + 1n) * N + K * N;
}

/** THE GAP BOUND. `laplace_gap_bound`: both loose directions at once, needing
 *  `K >= 1` and `n <= N`. */
export function laplaceGapBound(n: bigint, N: bigint, K: bigint): { over: boolean; under: boolean } {
  return { over: laplaceOverLoose(n, N, K), under: laplaceUnderLoose(n, N, K) };
}

/** `unseen_over_exact`: the sharp overestimate is attained EXACTLY at an
 *  unseen colour `n = 0`: `(0+1)*N = 0*(N+K) + N`. */
export function unseenOverExact(N: bigint, K: bigint): boolean {
  return (0n + 1n) * N === 0n * (N + K) + N;
}

/** `monopoly_under_exact`: the sharp underestimate is attained EXACTLY at a
 *  monopoly colour `n = N`: `N*(N+K) + N = (N+1)*N + K*N`. */
export function monopolyUnderExact(N: bigint, K: bigint): boolean {
  return N * (N + K) + N === (N + 1n) * N + K * N;
}

/** `laplace_exact_iff`: `(n+1)*N = n*(N+K)` iff `N = K*n` -- the colour sits
 *  at the uniform share `1/K`. Returned as the two sides so a test can check
 *  the biconditional directly rather than trusting a single boolean AND. */
export function laplaceExactIff(n: bigint, N: bigint, K: bigint): { crossEqual: boolean; uniformShare: boolean } {
  return {
    crossEqual: (n + 1n) * N === n * (N + K),
    uniformShare: N === K * n,
  };
}

/** `laplace_gap_below_eps`: the loose gap `K/(N+K)` drops below any rational
 *  tolerance `p/q` once `N >= K*q` (needs `K >= 1`, `p >= 1`), cross-multiplied
 *  as `K*q < p*(N+K)`. */
export function laplaceGapBelowEps(K: bigint, N: bigint, p: bigint, q: bigint): boolean {
  return K * q < p * (N + K);
}

/** `laplace_price_vanishes`: for every rational tolerance `p/q` (`p >= 1`) and
 *  every `N >= K*q` with `N > 0`, both one-sided gaps are strictly below `p/q`,
 *  cross-multiplied by `N*(N+K)*q`. */
export function laplacePriceVanishes(
  n: bigint,
  N: bigint,
  K: bigint,
  p: bigint,
  q: bigint,
): { over: boolean; under: boolean } {
  return {
    over: (n + 1n) * N * q < n * (N + K) * q + p * (N + K) * N,
    under: n * (N + K) * q < (n + 1n) * N * q + p * (N + K) * N,
  };
}

/** The explicit threshold `N0 = K*q` at which the price vanishes below `p/q`. */
export function laplaceVanishingThreshold(K: bigint, q: bigint): bigint {
  return K * q;
}

/**
 * ADVERSARIAL DUAL, `sharpGapDual_blindspot`. The SYMMETRIC sharp claim
 * `|L - F| <= 1/(N+K)` (both directions bounded by `N`, not `K*N`) is FALSE:
 * `K = 3, N = 1, n = 1` is well-formed (`1 <= K`, `n <= N`, `0 < N`) yet the
 * frequency exceeds the Laplace estimate by more than `1/(N+K)`. Only the
 * asymmetric sharp bounds (`laplaceOverSharp` bounded by `N`, `laplaceUnderSharp`
 * bounded by `K*N`) are proved; this function checks the FALSE symmetric claim
 * so the test can assert it fails, matching the Lean blindspot exactly.
 */
export function sharpSymmetricClaimHolds(n: bigint, N: bigint, K: bigint): boolean {
  return n * (N + K) <= (n + 1n) * N + N && (n + 1n) * N <= n * (N + K) + N;
}

/**
 * ADVERSARIAL DUAL, `gap_fails_without_colours`. Without `K >= 1` the loose
 * overestimate can fail: `K = 0, N = 1, n = 0`.
 */
export function gapFailsWithoutColours(): boolean {
  return !laplaceOverLoose(0n, 1n, 0n);
}

/**
 * ADVERSARIAL DUAL, `gap_fails_without_count_bound`. Without `n <= N` the
 * loose underestimate can fail: `K = 1, N = 1, n = 5`.
 */
export function gapFailsWithoutCountBound(): boolean {
  return !laplaceUnderLoose(5n, 1n, 1n);
}

/**
 * `exact_vacuous_at_empty`. At `N = 0` the cross-multiplied exactness test
 * passes at `n = 0` for every `K`, though the frequency `0/0` does not exist.
 * The `N > 0` guard is what gives the cross-multiplied statements meaning.
 */
export function exactVacuousAtEmpty(K: bigint): boolean {
  return (0n + 1n) * 0n === 0n * (0n + K);
}

/**
 * `mle_unseen_collapses`. The add-zero (maximum-likelihood) numerator of an
 * unseen colour is 0 while the add-one (Laplace/Buleyean) weight is 1 -- the
 * MLE collapses an unseen colour to a floor of zero, the Buleyean floor never
 * does.
 */
export function mleUnseenCollapses(count: bigint): { mleNumerator: bigint; buleyeanWeight: bigint } {
  return { mleNumerator: count, buleyeanWeight: count + 1n };
}
