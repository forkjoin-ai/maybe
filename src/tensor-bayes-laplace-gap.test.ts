import { describe, expect, it } from 'bun:test';
import {
  laplaceOverSharp,
  laplaceUnderSharp,
  laplaceOverLoose,
  laplaceUnderLoose,
  laplaceGapBound,
  unseenOverExact,
  monopolyUnderExact,
  laplaceExactIff,
  laplaceGapBelowEps,
  laplacePriceVanishes,
  laplaceVanishingThreshold,
  sharpSymmetricClaimHolds,
  gapFailsWithoutColours,
  gapFailsWithoutCountBound,
  exactVacuousAtEmpty,
  mleUnseenCollapses,
} from './tensor-bayes-laplace-gap';

describe('laplace gap -- sharp bounds hold everywhere in range', () => {
  it('laplace_over_sharp holds with no guards for every n, N, K', () => {
    for (let N = 0n; N <= 12n; N++) {
      for (let K = 0n; K <= 6n; K++) {
        for (let n = 0n; n <= N + 3n; n++) {
          expect(laplaceOverSharp(n, N, K)).toBe(true);
        }
      }
    }
  });

  it('laplace_under_sharp holds whenever n <= N', () => {
    for (let N = 0n; N <= 12n; N++) {
      for (let K = 0n; K <= 6n; K++) {
        for (let n = 0n; n <= N; n++) {
          expect(laplaceUnderSharp(n, N, K)).toBe(true);
        }
      }
    }
  });

  it('unseen_over_exact: the sharp overestimate is reached exactly at n = 0', () => {
    for (let N = 0n; N <= 20n; N++) {
      for (let K = 0n; K <= 10n; K++) {
        expect(unseenOverExact(N, K)).toBe(true);
        // And the sharp inequality is tight there: equality, not slack.
        expect((0n + 1n) * N === 0n * (N + K) + N).toBe(true);
      }
    }
  });

  it('monopoly_under_exact: the sharp underestimate is reached exactly at n = N', () => {
    for (let N = 0n; N <= 20n; N++) {
      for (let K = 0n; K <= 10n; K++) {
        expect(monopolyUnderExact(N, K)).toBe(true);
      }
    }
  });
});

describe('laplace gap -- loose bounds under their guards', () => {
  it('laplace_over_loose holds for K >= 1', () => {
    for (let N = 0n; N <= 12n; N++) {
      for (let K = 1n; K <= 6n; K++) {
        for (let n = 0n; n <= N + 3n; n++) {
          expect(laplaceOverLoose(n, N, K)).toBe(true);
        }
      }
    }
  });

  it('laplace_under_loose holds for n <= N', () => {
    for (let N = 0n; N <= 12n; N++) {
      for (let K = 0n; K <= 6n; K++) {
        for (let n = 0n; n <= N; n++) {
          expect(laplaceUnderLoose(n, N, K)).toBe(true);
        }
      }
    }
  });

  it('laplace_gap_bound bundles both loose directions', () => {
    for (let N = 1n; N <= 10n; N++) {
      for (let K = 1n; K <= 5n; K++) {
        for (let n = 0n; n <= N; n++) {
          const { over, under } = laplaceGapBound(n, N, K);
          expect(over).toBe(true);
          expect(under).toBe(true);
        }
      }
    }
  });
});

describe('laplace gap -- exactness at the uniform share', () => {
  it('laplace_exact_iff: the cross-multiplied equality matches N = K*n exactly', () => {
    for (let N = 0n; N <= 20n; N++) {
      for (let K = 1n; K <= 6n; K++) {
        for (let n = 0n; n <= N + 2n; n++) {
          const { crossEqual, uniformShare } = laplaceExactIff(n, N, K);
          expect(crossEqual).toBe(uniformShare);
        }
      }
    }
  });

  it('exact_at_uniform_share: K=2, N=4, n=2 sits at the uniform share (3/6 = 2/4)', () => {
    const { crossEqual, uniformShare } = laplaceExactIff(2n, 4n, 2n);
    expect(crossEqual).toBe(true);
    expect(uniformShare).toBe(true);
  });

  it('inexact_off_share: an unseen colour (n=0, N=4, K=2) is off the uniform share', () => {
    const { crossEqual, uniformShare } = laplaceExactIff(0n, 4n, 2n);
    expect(crossEqual).toBe(false);
    expect(uniformShare).toBe(false);
  });

  it('exact_vacuous_at_empty: at N=0 the cross-multiplied test passes vacuously', () => {
    for (let K = 0n; K <= 10n; K++) {
      expect(exactVacuousAtEmpty(K)).toBe(true);
    }
  });
});

describe('laplace gap -- the price vanishes at the explicit threshold N0 = K*q', () => {
  it('laplace_gap_below_eps holds once N >= K*q, for K >= 1, p >= 1', () => {
    for (let K = 1n; K <= 5n; K++) {
      for (let q = 1n; q <= 5n; q++) {
        for (let p = 1n; p <= 4n; p++) {
          const N0 = laplaceVanishingThreshold(K, q);
          expect(N0).toBe(K * q);
          for (let extra = 0n; extra <= 5n; extra++) {
            expect(laplaceGapBelowEps(K, N0 + extra, p, q)).toBe(true);
          }
        }
      }
    }
  });

  it('laplace_price_vanishes holds at N = N0 = K*q, both directions strictly under p/q', () => {
    for (let K = 1n; K <= 4n; K++) {
      for (let q = 1n; q <= 4n; q++) {
        const N0 = laplaceVanishingThreshold(K, q);
        if (N0 === 0n) continue; // needs N > 0
        for (let p = 1n; p <= 3n; p++) {
          for (let n = 0n; n <= N0; n++) {
            const { over, under } = laplacePriceVanishes(n, N0, K, p, q);
            expect(over).toBe(true);
            expect(under).toBe(true);
          }
        }
      }
    }
  });

  it('exactness at the threshold q=n: N0 = K*q makes the gap vanish entirely (margin 0)', () => {
    for (let K = 1n; K <= 6n; K++) {
      for (let q = 0n; q <= 6n; q++) {
        const N0 = laplaceVanishingThreshold(K, q);
        const { crossEqual, uniformShare } = laplaceExactIff(q, N0, K);
        expect(crossEqual).toBe(true);
        expect(uniformShare).toBe(true);
      }
    }
  });
});

describe('laplace gap -- adversarial duals (honest failure modes)', () => {
  it('sharpGapDual_blindspot: the symmetric sharp claim is FALSE at K=3, N=1, n=1', () => {
    expect(sharpSymmetricClaimHolds(1n, 1n, 3n)).toBe(false);
  });

  it('the asymmetric sharp bounds still hold at that same witness', () => {
    expect(laplaceOverSharp(1n, 1n, 3n)).toBe(true);
    expect(laplaceUnderSharp(1n, 1n, 3n)).toBe(true);
  });

  it('gap_fails_without_colours: K=0 breaks the loose overestimate', () => {
    expect(gapFailsWithoutColours()).toBe(true);
  });

  it('gap_fails_without_count_bound: n > N breaks the loose underestimate', () => {
    expect(gapFailsWithoutCountBound()).toBe(true);
  });

  it('mle_unseen_collapses: MLE numerator 0 vs Buleyean weight 1 at an unseen colour', () => {
    const { mleNumerator, buleyeanWeight } = mleUnseenCollapses(0n);
    expect(mleNumerator).toBe(0n);
    expect(buleyeanWeight).toBe(1n);
  });
});
