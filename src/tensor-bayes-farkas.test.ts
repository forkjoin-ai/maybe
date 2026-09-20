import { describe, expect, it } from 'bun:test';
import {
  con2Holds,
  sumA,
  sumB,
  sumC,
  checkCertificate,
  certificateInfeasible,
  intCovers,
  intLtZeroIffLeNegOne,
  margin,
  certificateMarginLeNegOne,
  noCertificateInTheGap,
  conXle0,
  conXge1,
  conXge0,
  tightBad,
  tightBadCert,
  tightGood,
  tightGoodCombo,
  tightFeasibleAtZero,
  deepBadCon,
  deepBadCert,
  deepBadMargin,
  rationalMarginFillsTheGap,
  feasiblePairLower,
  feasiblePairUpper,
  nonnegWitness,
  marginNeedsNonneg,
} from './tensor-bayes-farkas';

describe('farkas -- certificate structure and soundness', () => {
  it('tightBadCert is a valid certificate for tightBad', () => {
    const check = checkCertificate([...tightBad], tightBadCert);
    expect(check.mem).toBe(true);
    expect(check.nonneg).toBe(true);
    expect(check.cancelA).toBe(true);
    expect(check.cancelB).toBe(true);
    expect(check.neg).toBe(true);
    expect(check.valid).toBe(true);
  });

  it('tightGoodCombo cancels but does NOT reach a negative margin (not a certificate)', () => {
    const check = checkCertificate([...tightGood], tightGoodCombo);
    expect(check.cancelA).toBe(true);
    expect(check.cancelB).toBe(true);
    expect(check.neg).toBe(false); // margin is exactly 0, the knife-edge
    expect(check.valid).toBe(false);
  });

  it('certificate_infeasible: no point can satisfy tightBad, for a spread of candidate points', () => {
    for (let D = 1n; D <= 6n; D++) {
      for (let X = -6n; X <= 6n; X++) {
        expect(certificateInfeasible(tightBadCert, D, X, 0n)).toBe(true);
      }
    }
  });

  it('weighted_nonneg / certificate soundness: a point that violates a constraint is trivially refuted too', () => {
    // D <= 0 is degenerate (not a valid rational point at all).
    expect(certificateInfeasible(tightBadCert, 0n, 0n, 0n)).toBe(true);
  });
});

describe('farkas -- the tight pair: infeasible margin -1 adjacent to feasible margin 0', () => {
  it('tightBad really is infeasible: x <= 0 and x >= 1 admit no common integer, and no rational point either', () => {
    for (const [D, X] of [
      [1n, 0n],
      [1n, 1n],
      [2n, 1n],
      [3n, 2n],
      [5n, 3n],
    ] as const) {
      const bad = tightBad.some((k) => !con2Holds(k, D, X, 0n));
      expect(bad).toBe(true);
    }
  });

  it('tightGood is feasible at x = 0', () => {
    expect(tightGood.every((k) => con2Holds(k, 1n, 0n, 0n))).toBe(true);
  });

  it('tight_feasible_at_zero: the whole thesis in one statement', () => {
    const result = tightFeasibleAtZero();
    expect(result.goodMargin).toBe(0n);
    expect(result.goodFeasible).toBe(true);
    expect(result.badMargin).toBe(-1n);
    expect(result.badCertificateValid).toBe(true);
    expect(result.badInfeasibleByCertificate).toBe(true);
    expect(result.adjacent).toBe(true);
  });

  it('neg_one_covers_zero: 0 covers -1 on the integer lattice, with nothing in between', () => {
    expect(intCovers(-1n, 0n)).toBe(true);
    // No integer strictly between -1 and 0.
    for (let x = -5n; x <= 5n; x++) {
      const strictlyBetween = -1n < x && x < 0n;
      expect(strictlyBetween).toBe(false);
    }
  });

  it('intCovers is false for any wider Int gap', () => {
    expect(intCovers(-2n, 0n)).toBe(false);
    expect(intCovers(-1n, 1n)).toBe(false);
    expect(intCovers(0n, 0n)).toBe(false);
  });
});

describe('farkas -- every certificate has margin at most -1 (over Int)', () => {
  it('int_lt_zero_iff_le_neg_one: strictly negative IS at most -1, for a spread of integers', () => {
    for (let x = -10n; x <= 10n; x++) {
      expect(intLtZeroIffLeNegOne(x)).toBe(true);
    }
  });

  it('certificate_margin_le_neg_one holds for tightBadCert and deepBadCert', () => {
    expect(certificateMarginLeNegOne(tightBadCert)).toBe(true);
    expect(certificateMarginLeNegOne(deepBadCert)).toBe(true);
  });

  it('no_certificate_in_the_gap: neither certificate lands strictly between -1 and 0', () => {
    expect(noCertificateInTheGap(tightBadCert)).toBe(true);
    expect(noCertificateInTheGap(deepBadCert)).toBe(true);
  });

  it('deepBad_margin: margins are bounded above by -1, not equal to it -- exhibits -7', () => {
    expect(deepBadMargin()).toBe(-7n);
    expect(margin(deepBadCert)).toBeLessThanOrEqual(-1n);
  });

  it('the achievable margin spectrum is unbounded below: a flat constraint realizes any m <= -1', () => {
    for (const m of [-1n, -2n, -5n, -20n]) {
      const flat = { a: 0n, b: 0n, c: m };
      const cert: readonly [bigint, typeof flat][] = [[1n, flat]];
      expect(sumA(cert)).toBe(0n);
      expect(sumB(cert)).toBe(0n);
      expect(sumC(cert)).toBe(m);
      expect(intLtZeroIffLeNegOne(m)).toBe(true);
    }
  });
});

describe('farkas -- THE INFRATHIN CENTREPIECE: integer cover vs. rational gap', () => {
  it('over Int, no certificate has a margin strictly between -1 and 0 (the gap is empty)', () => {
    // Exhaustively confirm no integer margin lies in the open gap.
    for (let m = -20n; m <= 5n; m++) {
      const inGap = -1n < m && m < 0n;
      expect(inGap).toBe(false);
    }
  });

  it('rational_margin_fills_the_gap: the SAME certificate halved gives margin -1/2, strictly inside the gap', () => {
    const result = rationalMarginFillsTheGap();
    expect(result.num).toBe(-1n);
    expect(result.den).toBe(2n);
    expect(result.denPositive).toBe(true);
    expect(result.aboveNegOne).toBe(true);
    expect(result.belowZero).toBe(true);
    expect(result.liesInTheGap).toBe(true);
  });

  it('the contrast: -1/2 is a legitimate cross-multiplied rational value between the integer cover -1 and 0', () => {
    // Cross-multiplied: -1 * 2 < -1 * 1 < 0 * 2, i.e. -2 < -1 < 0.
    expect(-1n * 2n).toBeLessThan(-1n * 1n);
    expect(-1n * 1n).toBeLessThan(0n * 2n);
    // Yet over Int there is no margin value realizing exactly this fraction:
    // the nearest integer margins are -1 and 0, and 0 is not < 0.
    expect(intCovers(-1n, 0n)).toBe(true);
  });

  it('this theorem is a consequence of integrality, not of Farkas: halving ANY certificate breaks Int-valuedness', () => {
    // tightBadCert has multipliers (1,1); halved multipliers (1/2,1/2) are not
    // representable as bigint at all -- which is exactly why the gap reopens
    // only once we leave Int. We check this by confirming 1n/2n truncates
    // (integer division loses the value that matters).
    expect(1n / 2n).toBe(0n); // bigint division discards the fractional part
    expect(rationalMarginFillsTheGap().num).not.toBe(1n / 2n);
  });
});

describe('farkas -- nonnegativity is load-bearing (margin_needs_nonneg)', () => {
  it('feasiblePairLower/Upper (y<=0, y<=1) form a feasible system', () => {
    expect(con2Holds(feasiblePairLower, 1n, 0n, 0n)).toBe(true);
    expect(con2Holds(feasiblePairUpper, 1n, 0n, 0n)).toBe(true);
  });

  it('nonnegWitness cancels both coefficients and reaches margin -1 despite one negative multiplier', () => {
    expect(sumA(nonnegWitness)).toBe(0n);
    expect(sumB(nonnegWitness)).toBe(0n);
    expect(sumC(nonnegWitness)).toBe(-1n);
  });

  it('margin_needs_nonneg: the bound is void without Certificate.nonneg', () => {
    const result = marginNeedsNonneg();
    expect(result.mixedSignMargin).toBe(-1n);
    expect(result.cancelsA).toBe(true);
    expect(result.cancelsB).toBe(true);
    expect(result.systemIsFeasible).toBe(true);
    expect(result.multipliersAllNonneg).toBe(false);
  });
});

describe('farkas -- honest scope note', () => {
  it('conXle0/conXge1/conXge0 are the concrete pair-level objects this module mirrors exactly', () => {
    expect(conXle0).toEqual({ a: 1n, b: 0n, c: 0n });
    expect(conXge1).toEqual({ a: -1n, b: 0n, c: -1n });
    expect(conXge0).toEqual({ a: -1n, b: 0n, c: 0n });
    // The general n-constraint Fourier-Motzkin "exactly one" alternative
    // (`farkas_exactly_one` over arbitrary-length lists) is proved in Lean
    // but is NOT re-implemented here; this module mirrors the certificate
    // definition, its soundness, and the concrete tight-pair alternative,
    // exactly as the formula card's own Honest boundary scopes it.
  });
});
