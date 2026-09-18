/**
 * exactness.mjs -- integer / BigInt bit-exactness check for the affine linear rule.
 *
 * The claim under test: the naive O(N^2) affine sum and the O(N) prefix-sum
 * (sufficient-statistic) computation agree BIT-EXACTLY, not merely to floating
 * tolerance. All scores, budgets and arithmetic here are BigInt, so equality is
 * literal integer equality.
 *
 *   voidSign in {+1,-1};  v_ij = voidSign * s_ij;  w_ij = R - beta*v_ij + 1
 *   denom_naive  = sum_j w_ij
 *   denom_stats  = N*(R+1) - beta * q_i . (sum_j r_j),          r_j = voidSign*k_j
 *   num_naive[g] = sum_j w_ij * V_jg
 *   num_stats[g] = (R+1)*(sum_j V_jg) - beta * q_i . (sum_j r_j V_jg)
 *
 * The same randomized loop also compares the HINGE sum 1 + max(R - beta*v, 0)
 * against the affine sufficient-statistic formula. The hinge clamp is piecewise
 * linear and therefore cannot factorise; mismatches are expected and counted.
 */

import { mulberry32 } from './attention.mjs';

export function bigintAffineExactness(opts) {
  const trials = opts.trials;
  const seed = opts.seed;
  const maxN = opts.maxN;
  const maxD = opts.maxD;
  const maxDV = opts.maxDV;
  const rng = mulberry32(seed);
  const ri = function (lo, hi) { return lo + Math.floor(rng() * (hi - lo + 1)); };

  let comparisons = 0;
  let mismatches = 0;
  let hingeMismatches = 0;
  let hingeComparisons = 0;
  let rows = 0;
  let example = null;

  for (let tr = 0; tr < trials; tr++) {
    const N = ri(2, maxN);
    const d = ri(1, maxD);
    const dv = ri(1, maxDV);
    const voidSign = rng() < 0.5 ? 1n : -1n;
    const beta = BigInt(ri(0, 4));
    const R = BigInt(ri(0, 10));

    const K = [];
    const V = [];
    for (let j = 0; j < N; j++) {
      const k = [];
      const v = [];
      for (let f = 0; f < d; f++) k.push(BigInt(ri(-9, 9)));
      for (let f = 0; f < dv; f++) v.push(BigInt(ri(-9, 9)));
      K.push(k);
      V.push(v);
    }
    const M = ri(1, 4);
    const Q = [];
    for (let i = 0; i < M; i++) {
      const q = [];
      for (let f = 0; f < d; f++) q.push(BigInt(ri(-9, 9)));
      Q.push(q);
    }

    // Sufficient statistics, built once, BigInt exact.
    const Sk = new Array(d).fill(0n);
    const Sv = new Array(dv).fill(0n);
    const Skv = [];
    for (let f = 0; f < d; f++) Skv.push(new Array(dv).fill(0n));
    for (let j = 0; j < N; j++) {
      for (let f = 0; f < d; f++) Sk[f] += voidSign * K[j][f];
      for (let f = 0; f < dv; f++) Sv[f] += V[j][f];
      for (let f = 0; f < d; f++) {
        for (let g = 0; g < dv; g++) Skv[f][g] += voidSign * K[j][f] * V[j][g];
      }
    }

    for (let i = 0; i < M; i++) {
      const q = Q[i];
      let denomNaive = 0n;
      let denomHinge = 0n;
      const numNaive = new Array(dv).fill(0n);
      const numHinge = new Array(dv).fill(0n);
      for (let j = 0; j < N; j++) {
        let s = 0n;
        for (let f = 0; f < d; f++) s += q[f] * K[j][f];
        const base = R - beta * (voidSign * s);
        const w = base + 1n;                              // affine, unclamped
        denomNaive += w;
        for (let g = 0; g < dv; g++) numNaive[g] += w * V[j][g];
        const wh = 1n + (base > 0n ? base : 0n);          // hinge, clamped
        denomHinge += wh;
        for (let g = 0; g < dv; g++) numHinge[g] += wh * V[j][g];
      }
      let qSk = 0n;
      for (let f = 0; f < d; f++) qSk += q[f] * Sk[f];
      const denomStats = BigInt(N) * (R + 1n) - beta * qSk;
      const numStats = new Array(dv).fill(0n);
      for (let g = 0; g < dv; g++) {
        let acc = 0n;
        for (let f = 0; f < d; f++) acc += q[f] * Skv[f][g];
        numStats[g] = (R + 1n) * Sv[g] - beta * acc;
      }

      rows++;
      comparisons++;
      if (denomNaive !== denomStats) mismatches++;
      for (let g = 0; g < dv; g++) {
        comparisons++;
        if (numNaive[g] !== numStats[g]) mismatches++;
      }
      hingeComparisons++;
      if (denomHinge !== denomStats) hingeMismatches++;

      if (!example) {
        example = {
          N, d, dv, beta: beta.toString(), R: R.toString(), voidSign: voidSign.toString(),
          denomNaive: denomNaive.toString(), denomStats: denomStats.toString(),
          numNaive0: numNaive[0].toString(), numStats0: numStats[0].toString(),
        };
      }
    }
  }
  return { trials, rows, comparisons, mismatches, hingeComparisons, hingeMismatches, example };
}

function main() {
  const r = bigintAffineExactness({ trials: 400, seed: 20240607, maxN: 40, maxD: 6, maxDV: 5 });
  console.log('trials=' + r.trials + ' rows=' + r.rows + ' comparisons=' + r.comparisons +
    ' mismatches=' + r.mismatches);
  console.log('hinge-vs-affine-stats comparisons=' + r.hingeComparisons +
    ' mismatches=' + r.hingeMismatches);
  console.log('example=' + JSON.stringify(r.example));
}

if (process.argv[1] && process.argv[1].endsWith('exactness.mjs')) {
  main();
}
