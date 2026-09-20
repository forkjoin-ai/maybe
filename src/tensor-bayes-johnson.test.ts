import { describe, expect, it } from 'bun:test';
import {
  AXES,
  WALKERS,
  EDGES,
  edgeAt,
  edgeIndex,
  shareCount,
  adjM,
  disjM,
  partitionHolds,
  adjacentDegree,
  disjointDegree,
  adjSqGeneralHolds,
  adjSqReducedHolds,
  johnsonLambdaHolds,
  johnsonMuHolds,
  commonAdjSum,
  starContractHolds,
  starDiffEigen7Holds,
  starOverlap,
  pairStarStarVHolds,
  eigen7MultiplicityWitness,
  quadEigenNeg2Holds,
  increasingQuadruples,
  highAxisCount,
  eigenNeg2MultiplicityWitness,
  multiplicitySolves,
  allOnesEigen18Holds,
  pleromaCounts,
  jointIA,
  jointIAViolatesPlucker,
} from './tensor-bayes-johnson';

describe('johnson scheme -- the 55 axes are the edges of K_11', () => {
  it('enumerates exactly 55 distinct axes covering C(11,2)', () => {
    expect(AXES).toBe(55);
    expect(EDGES.length).toBe(55);
    const seen = new Set(EDGES.map(([i, j]) => `${i},${j}`));
    expect(seen.size).toBe(55);
  });

  it('edgeAt / edgeIndex are mutual inverses', () => {
    for (let a = 0; a < AXES; a++) {
      const [i, j] = edgeAt(a);
      expect(edgeIndex(i, j)).toBe(a);
    }
  });

  it('johnson_trichotomy: shareCount is 2 iff same axis, else 1 (adjacent) or 0 (disjoint)', () => {
    for (let a = 0; a < AXES; a++) {
      expect(shareCount(a, a)).toBe(2);
      for (let b = 0; b < AXES; b++) {
        if (a === b) continue;
        expect([0, 1]).toContain(shareCount(a, b));
      }
    }
  });
});

describe('johnson scheme -- degrees and the partition matrix', () => {
  it('partition_matrix: I + A + D = J at every one of the 3025 entries', () => {
    expect(partitionHolds()).toBe(true);
  });

  it('adjRow_eq_18: every axis has exactly 18 adjacent neighbours', () => {
    for (let a = 0; a < AXES; a++) expect(adjacentDegree(a)).toBe(18);
  });

  it('disjRow_eq_36: every axis has exactly 36 disjoint neighbours', () => {
    for (let a = 0; a < AXES; a++) expect(disjointDegree(a)).toBe(36);
  });

  it('the Pleroma counting identities: 55 axes, 495 adjacent + 990 disjoint = 1485', () => {
    const counts = pleromaCounts();
    expect(counts.axes).toBe(55);
    expect(counts.adjacentPairs).toBe(495);
    expect(counts.disjointPairs).toBe(990);
    expect(counts.totalPairs).toBe(1485);
  });
});

describe('johnson scheme -- the SRG identity (55, 18, 9, 4)', () => {
  it('adjSq_general: A*A = 18*I + 9*A + 4*D at every entry', () => {
    expect(adjSqGeneralHolds()).toBe(true);
  });

  it('adjSq_reduced_int: the D-free reduced quadratic A*A = 14*I + 5*A + 4*J', () => {
    expect(adjSqReducedHolds()).toBe(true);
  });

  it('johnson_lambda_general: adjacent axes share exactly 9 common adjacent neighbours', () => {
    expect(johnsonLambdaHolds()).toBe(true);
  });

  it('johnson_mu_general: disjoint axes share exactly 4 common adjacent neighbours', () => {
    expect(johnsonMuHolds()).toBe(true);
  });

  it('commonAdj concrete witnesses: an adjacent pair gives 9, a disjoint pair gives 4', () => {
    const adjacentPair = [edgeIndex(0, 1), edgeIndex(0, 2)] as const; // share walker 0
    expect(shareCount(...adjacentPair)).toBe(1);
    expect(commonAdjSum(...adjacentPair)).toBe(9);

    const disjointPair = [edgeIndex(0, 1), edgeIndex(2, 3)] as const; // share nothing
    expect(shareCount(...disjointPair)).toBe(0);
    expect(commonAdjSum(...disjointPair)).toBe(4);
  });

  it('adversarial dual: the wrong SRG constant mu = 5 is refuted', () => {
    const disjointPair = [edgeIndex(0, 1), edgeIndex(2, 3)] as const;
    expect(commonAdjSum(...disjointPair)).not.toBe(5);
  });
});

describe('johnson scheme -- adversarial dual: I+A is not a decomposable (rank-one) joint', () => {
  it('jointIA is symmetric', () => {
    for (let a = 0; a < 10; a++) {
      for (let b = 0; b < 10; b++) {
        expect(jointIA(a, b)).toBe(jointIA(b, a));
      }
    }
  });

  it('jointIA violates the Plucker relation at a concrete four-axis witness', () => {
    const i = edgeIndex(0, 1);
    const j = edgeIndex(0, 2);
    const k = edgeIndex(2, 3);
    const l = edgeIndex(1, 2);
    expect(jointIAViolatesPlucker(i, j, k, l)).toBe(true);
  });
});

describe('johnson scheme -- spectrum, general (every one of the 55 axes)', () => {
  it('allOnes_eigen18: the all-ones vector is an 18-eigenvector', () => {
    expect(allOnesEigen18Holds()).toBe(true);
  });

  it('star_contract: A*star(x) = 7*star(x) + 2*J for every walker', () => {
    for (let x = 0; x < WALKERS; x++) expect(starContractHolds(x)).toBe(true);
  });

  it('starDiff_eigen7: every star difference is a 7-eigenvector, for every walker pair', () => {
    for (let x = 0; x < WALKERS; x++) {
      for (let y = 0; y < WALKERS; y++) {
        if (x === y) continue;
        expect(starDiffEigen7Holds(x, y)).toBe(true);
      }
    }
  });

  it('star_overlap: <star(x),star(y)> = 10 if x=y, else 1', () => {
    for (let x = 0; x < WALKERS; x++) {
      for (let y = 0; y < WALKERS; y++) {
        expect(starOverlap(x, y)).toBe(x === y ? 10 : 1);
      }
    }
  });

  it('quad_eigenNeg2: every ordered quadrilateral is a (-2)-eigenvector, for all 330 quadruples', () => {
    const quads = increasingQuadruples();
    expect(quads.length).toBe(330); // C(11,4)
    for (const [p, q, r, s] of quads) {
      expect(quadEigenNeg2Holds(p, q, r, s)).toBe(true);
    }
  });
});

describe('johnson scheme -- multiplicities are LOWER BOUNDS ONLY, by explicit witness', () => {
  it('pairStar_starV: the ten star differences pair against the stars as 9*I', () => {
    for (let x = 1; x <= 10; x++) {
      for (let y = 1; y <= 10; y++) {
        expect(pairStarStarVHolds(x, y)).toBe(true);
      }
    }
  });

  it('eigen7_multiplicity_ge_ten: diagonal 9*I pairing, nonzero, and eigenvalue-7', () => {
    const witness = eigen7MultiplicityWitness();
    expect(witness.pairingIsNineI).toBe(true);
    expect(witness.allNonzero).toBe(true);
    expect(witness.allEigen7).toBe(true);
  });

  it('eigenNeg2_multiplicity_ge_36: 36 high axes, diagonal evaluation, eigenvalue -2', () => {
    expect(highAxisCount()).toBe(36);
    const witness = eigenNeg2MultiplicityWitness();
    expect(witness.count36).toBe(true);
    expect(witness.diagonalEval).toBe(true);
    expect(witness.allEigenNeg2).toBe(true);
  });

  it('multiplicity_solves is a CONSISTENCY CHECK, not a proof of exact multiplicities', () => {
    const check = multiplicitySolves();
    expect(check.traceZero).toBe(true);
    expect(check.dimensionFifty5).toBe(true);
    // Honesty: this does not establish {18,7,-2} is the whole spectrum, nor
    // that (1,10,44) are the unique multiplicities -- only that they solve
    // the trace/dimension equations. The corpus proves only the LOWER
    // bounds asserted above (>= 10 and >= 36), never the matching uppers.
  });
});
