/**
 * run.mjs -- the runnable entry for the softmax vs Buleyean/Laplace affine
 * linear-attention experiment. Run with a real Node:
 *
 *   /opt/homebrew/bin/node run.mjs
 *
 * Dependency-free. Prints the full report; README.md embeds this exact output.
 */

import {
  makeRecallTask, softmaxAttention, affineAttentionNaive, affineAttentionLinear,
  recallMetrics, meanKL, maxAbsDiff, makeCounter, totalOps,
  negativeWeightFraction, clampedFraction, dot,
} from './attention.mjs';
import { bigintAffineExactness } from './exactness.mjs';

const CFG = { M: 32, N: 64, d: 8, dv: 8, keyNoise: 0.1, budgetC: 1.0, seeds: 8, baseSeed: 1 };

function line() { console.log('--------------------------------------------------------------------------'); }
function pad(s, n) { return String(s).padStart(n); }
function f(x, n) { return Number(x).toFixed(n === undefined ? 4 : n); }

function printCounter(label, c) {
  console.log(pad(label, 16) + ' exp=' + pad(c.exp, 9) + ' mul=' + pad(c.mul, 10) +
    ' add=' + pad(c.add, 11) + ' div=' + pad(c.div, 9) + ' total=' + pad(totalOps(c), 12));
}

// ---------------------------------------------------------------------------
// 1. Cost
// ---------------------------------------------------------------------------
function sectionCost() {
  console.log('1) COST COUNTER  (M=' + CFG.M + ' N=' + CFG.N + ' d=' + CFG.d + ' dv=' + CFG.dv +
    ' keyNoise=' + CFG.keyNoise + ')');
  const task = makeRecallTask(CFG);
  const cs = makeCounter();
  softmaxAttention(task, 1.0, cs);
  const ca = makeCounter();
  affineAttentionLinear(task, 1.0, -1, CFG.budgetC, ca);
  const cn = makeCounter();
  affineAttentionNaive(task, 1.0, -1, CFG.budgetC, false, cn);
  const ch = makeCounter();
  affineAttentionNaive(task, 1.0, -1, CFG.budgetC, true, ch);

  printCounter('softmax', cs);
  printCounter('affine-linear', ca);
  printCounter('affine-naive', cn);
  printCounter('hinge-naive', ch);
  console.log('ratio softmax/affine-linear total = ' + f(totalOps(cs) / totalOps(ca), 2) + 'x');
  console.log('affine-linear exp() calls === 0 : ' + (ca.exp === 0));
  const smBytes = CFG.M * CFG.N * 8;
  const affBytes = (CFG.d * CFG.d + CFG.d + CFG.dv) * 8;
  console.log('memory softmax score/weight matrix = ' + smBytes + ' bytes; affine stats (d*d + d + dv doubles) = ' +
    affBytes + ' bytes; ratio = ' + f(smBytes / affBytes, 1) + 'x');
  console.log('NOTE: a streaming softmax needs only O(N) memory but still N exp() calls per query.');
  line();
}

// ---------------------------------------------------------------------------
// 2. O(N) scaling
// ---------------------------------------------------------------------------
function sectionScaling() {
  console.log('2) O(N) SCALING  (M=N, d=4, dv=4, beta=1, voidSign=-1)');
  const Ns = [16, 32, 64, 128, 256, 512, 1024];
  console.log(pad('N', 6) + pad('softmaxOps', 14) + pad('affLinOps', 12) + pad('affNaiveOps', 14) +
    pad('smOps/N', 12) + pad('affLinOps/N', 13) + pad('affLinOps/N^2', 15));
  const rows = [];
  for (let k = 0; k < Ns.length; k++) {
    const N = Ns[k];
    const task = makeRecallTask({ M: N, N: N, d: 4, dv: 4, keyNoise: 0.1, seed: 7 });
    const cs = makeCounter();
    softmaxAttention(task, 1.0, cs, { materialize: false });
    const ca = makeCounter();
    affineAttentionLinear(task, 1.0, -1, 1.0, ca);
    const cn = makeCounter();
    affineAttentionNaive(task, 1.0, -1, 1.0, false, cn, { materialize: false });
    const so = totalOps(cs);
    const ao = totalOps(ca);
    const no = totalOps(cn);
    rows.push({ N, so, ao, no });
    console.log(pad(N, 6) + pad(so, 14) + pad(ao, 12) + pad(no, 14) +
      pad(f(so / N, 1), 12) + pad(f(ao / N, 1), 13) + pad(f(ao / (N * N), 3), 15));
  }
  const first = rows[0];
  const last = rows[rows.length - 1];
  const nRatio = last.N / first.N;
  console.log('N grew ' + f(nRatio, 1) + 'x:  softmax ops grew ' + f(last.so / first.so, 1) +
    'x (quadratic);  affine-linear ops grew ' + f(last.ao / first.ao, 1) + 'x (linear).');
  console.log('affine-linear ops/N is flat (O(N));  softmax ops/N^2 is flat (O(N^2)).');
  line();
}

// ---------------------------------------------------------------------------
// 3. BigInt exactness
// ---------------------------------------------------------------------------
function sectionExactness() {
  console.log('3) BIGINT BIT-EXACTNESS  (integer affine rule, randomized trials)');
  const r = bigintAffineExactness({ trials: 400, seed: 20240607, maxN: 40, maxD: 6, maxDV: 5 });
  console.log('trials=' + r.trials + ' rows=' + r.rows + ' field-comparisons=' + r.comparisons +
    ' mismatches=' + r.mismatches);
  console.log('HINGE vs affine sufficient-statistic formula: comparisons=' + r.hingeComparisons +
    ' mismatches=' + r.hingeMismatches + '  (clamp is not linear -> cannot factorise)');
  console.log('example: ' + JSON.stringify(r.example));
  line();
}

// ---------------------------------------------------------------------------
// 4. Sigma sweep
// ---------------------------------------------------------------------------
function sweepPoint(beta, voidSign, cfg) {
  const acc = { smV: 0, smI: 0, afV: 0, afI: 0, hiV: 0, hiI: 0, klA: 0, klH: 0, lin: 0 };
  for (let s = 0; s < cfg.seeds; s++) {
    const task = makeRecallTask({
      M: cfg.M, N: cfg.N, d: cfg.d, dv: cfg.dv, keyNoise: cfg.keyNoise, seed: cfg.baseSeed + s * 101,
    });
    const sm = softmaxAttention(task, beta);
    const af = affineAttentionNaive(task, beta, voidSign, cfg.budgetC, false);
    const hi = affineAttentionNaive(task, beta, voidSign, cfg.budgetC, true);
    const ln = affineAttentionLinear(task, beta, voidSign, cfg.budgetC);
    const mSm = recallMetrics(task, sm.weights, sm.outputs);
    const mAf = recallMetrics(task, af.weights, af.outputs);
    const mHi = recallMetrics(task, hi.weights, hi.outputs);
    acc.smV += mSm.value; acc.smI += mSm.index;
    acc.afV += mAf.value; acc.afI += mAf.index;
    acc.hiV += mHi.value; acc.hiI += mHi.index;
    acc.klA += meanKL(sm.weights, af.weights);
    acc.klH += meanKL(sm.weights, hi.weights);
    acc.lin = Math.max(acc.lin, maxAbsDiff(af.outputs, ln.outputs));
  }
  const n = cfg.seeds;
  return {
    smV: acc.smV / n, smI: acc.smI / n, afV: acc.afV / n, afI: acc.afI / n,
    hiV: acc.hiV / n, hiI: acc.hiI / n, klA: acc.klA / n, klH: acc.klH / n, lin: acc.lin,
  };
}

function sectionSweep() {
  console.log('4) SIGMA SWEEP  (beta = logit scale; M=' + CFG.M + ' N=' + CFG.N + ' d=' + CFG.d +
    ' dv=' + CFG.dv + ' keyNoise=' + CFG.keyNoise + '; averaged over ' + CFG.seeds + ' seeds)');
  console.log('   recall.value = top-1 retrieval (nearest value to attention output); recall.index = argmax-weight key');
  const conventions = [
    ['voidSign=-1  (v=-s, softmax-aligned / void-dual affine)', -1],
    ['voidSign=+1  (literal w=R-s+1 exactly as written in the brief)', 1],
  ];
  const betas = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8];
  for (let ci = 0; ci < conventions.length; ci++) {
    console.log('');
    console.log(conventions[ci][0]);
    console.log(pad('beta', 6) + pad('smVal', 8) + pad('smIdx', 8) + pad('afVal', 8) + pad('afIdx', 8) +
      pad('hiVal', 8) + pad('KLsm|af', 10) + pad('KLsm|hi', 10) + pad('match', 7) + pad('linErr', 10));
    for (let bi = 0; bi < betas.length; bi++) {
      const beta = betas[bi];
      const p = sweepPoint(beta, conventions[ci][1], CFG);
      const match = (p.afV >= p.smV - 0.02) && (p.klA <= 0.05);
      console.log(pad(f(beta, 2), 6) + pad(f(p.smV), 8) + pad(f(p.smI), 8) + pad(f(p.afV), 8) +
        pad(f(p.afI), 8) + pad(f(p.hiV), 8) + pad(f(p.klA), 10) + pad(f(p.klH), 10) +
        pad(match ? 'YES' : 'no', 7) + pad(p.lin.toExponential(1), 10));
    }
  }
  line();
}

// ---------------------------------------------------------------------------
// 5. Temperature / budget: affine is the first-order (linear) softmax
// ---------------------------------------------------------------------------
function sectionTemperature() {
  console.log('5) TEMPERATURE / LINEAR BUDGET  (R = beta*C; affine slope = inverse temperature beta)');
  console.log('   maxWeightDiff = max_j |softmax_j - affine_j| on one row;  diff/beta^2 ~ const means O(beta^2) error');
  const task = makeRecallTask({ M: 1, N: 16, d: 4, dv: 1, keyNoise: 0, seed: 99 });
  const betas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8];
  console.log(pad('beta', 7) + pad('R=beta*C', 10) + pad('maxWeightDiff', 15) +
    pad('diff/beta^2', 13) + pad('KL(sm|af)', 11) + pad('affNegFrac', 12));
  for (let i = 0; i < betas.length; i++) {
    const beta = betas[i];
    const sm = softmaxAttention(task, beta);
    const af = affineAttentionNaive(task, beta, -1, 1.0, false);
    const md = maxAbsDiff(sm.weights, af.weights);
    const kl = meanKL(sm.weights, af.weights);
    const neg = negativeWeightFraction(af.weights);
    console.log(pad(f(beta, 2), 7) + pad(f(beta * 1.0, 2), 10) + pad(md.toExponential(3), 15) +
      pad(f(md / (beta * beta), 4), 13) + pad(f(kl, 6), 11) + pad(f(neg, 4), 12));
  }
  line();
}

// ---------------------------------------------------------------------------
// 6. Fixed (beta-independent) budget: when the clamp actually fires
// ---------------------------------------------------------------------------
function sectionClamp() {
  console.log('6) FIXED BUDGET R=1 (budgetC=1/beta): clamp activation and validity');
  console.log('   affine is unclamped and may go negative; hinge clamps to 1 and stays a valid distribution');
  const betas = [0.1, 0.5, 1, 2, 4, 8];
  console.log(pad('beta', 6) + pad('affNegFrac', 12) + pad('hingeClampedFrac', 18) +
    pad('smVal', 8) + pad('afVal', 8) + pad('hiVal', 8));
  const cfg = { M: 32, N: 64, d: 8, dv: 8, keyNoise: 0.1, seeds: 4, baseSeed: 11 };
  for (let i = 0; i < betas.length; i++) {
    const beta = betas[i];
    let neg = 0, clamp = 0, smV = 0, afV = 0, hiV = 0;
    for (let s = 0; s < cfg.seeds; s++) {
      const task = makeRecallTask({ M: cfg.M, N: cfg.N, d: cfg.d, dv: cfg.dv, keyNoise: cfg.keyNoise, seed: cfg.baseSeed + s * 101 });
      const sm = softmaxAttention(task, beta);
      const af = affineAttentionNaive(task, beta, -1, 1 / beta, false);
      const hi = affineAttentionNaive(task, beta, -1, 1 / beta, true);
      neg += negativeWeightFraction(af.weights);
      clamp += clampedFraction(task, beta, -1, 1 / beta);
      smV += recallMetrics(task, sm.weights, sm.outputs).value;
      afV += recallMetrics(task, af.weights, af.outputs).value;
      hiV += recallMetrics(task, hi.weights, hi.outputs).value;
    }
    console.log(pad(f(beta, 2), 6) + pad(f(neg / cfg.seeds, 4), 12) +
      pad(f(clamp / cfg.seeds, 4), 18) + pad(f(smV / cfg.seeds), 8) +
      pad(f(afV / cfg.seeds), 8) + pad(f(hiV / cfg.seeds), 8));
  }
  line();
}

// ---------------------------------------------------------------------------
// 7. Concentration ceiling
// ---------------------------------------------------------------------------
function sectionCeiling() {
  console.log('7) CONCENTRATION CEILING (why the affine rule cannot retrieve at ANY beta)');
  const N = 64;
  const d = 8;
  const C = 1.0;
  const task = makeRecallTask({ M: 1, N: N, d: d, dv: 8, keyNoise: 0, seed: 5 });
  const s = [];
  for (let j = 0; j < N; j++) s.push(dot(task.Q[0], task.K[j], null));
  let sumS = 0;
  for (let j = 0; j < N; j++) sumS += s[j];
  const ceiling = (C + s[0]) / (N * C + sumS);
  console.log('   N=' + N + ' C=' + C + ' s_target=' + f(s[0]) + ' sum_s=' + f(sumS, 3) +
    ' analytic affine ceiling (C+s_t)/(N*C+sum_s) = ' + f(ceiling));
  console.log('   note: linear attention is O(N*d^2), not O(N*d): the second-moment statistic S_kv is d x d.');
  console.log(pad('beta', 6) + pad('softmaxTargetFrac', 20) + pad('affineTargetFrac', 19) + pad('ceiling', 10));
  const betas = [0.1, 0.5, 1, 2, 4, 8, 16, 64];
  for (let i = 0; i < betas.length; i++) {
    const beta = betas[i];
    const sm = softmaxAttention(task, beta);
    const af = affineAttentionNaive(task, beta, -1, C, false);
    console.log(pad(f(beta, 2), 6) + pad(f(sm.weights[0][0], 6), 20) +
      pad(f(af.weights[0][0], 6), 19) + pad(f(ceiling, 6), 10));
  }
  console.log('   affine target mass saturates at the O(1/N) ceiling; softmax target mass -> 1. No beta rescues it.');
  line();
}

console.log('==========================================================================');
console.log('softmax vs Buleyean/Laplace affine-or-hinge linear attention');
console.log('node ' + process.version);
console.log('==========================================================================');
sectionCost();
sectionScaling();
sectionExactness();
sectionSweep();
sectionTemperature();
sectionClamp();
sectionCeiling();
console.log('done.');
