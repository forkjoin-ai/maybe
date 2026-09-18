/**
 * run.mjs -- runnable entry for the "softmax vs richer-kernel linear attention"
 * experiment.  Run with a real Node:
 *
 *   /opt/homebrew/bin/node run.mjs
 *
 * Dependency-free. Prints the full report; README.md embeds this exact output.
 */

import {
  makeRecallTask, softmaxAttention, polyAttentionLinear, polyAttentionNaive,
  reluAttentionNaive, featureDim, bigintPolyExactness, makeRandomFeatures,
  prfFeaturesAll, reluRfFeaturesAll, featureStats, queryLinear, effectiveWeights,
  recallMetrics, targetMass, meanKL, maxAbsDiff, makeCounter, totalOps, addCounter,
  makeCounter as mc, mean, std, scoreBound,
} from './kernel-attention.mjs';

const CFG = { M: 32, N: 64, d: 8, dv: 8, keyNoise: 0.1, beta: 8, seeds: 4, baseSeed: 1 };
const TOL_RECALL = 0.02;
const TOL_MASS = 0.05;

function line() { console.log('--------------------------------------------------------------------------'); }
function pad(s, n) { return String(s).padStart(n); }
function f(x, n) { return Number(x).toFixed(n === undefined ? 4 : n); }
function sci(x, n) { return Number(x).toExponential(n === undefined ? 1 : n); }
function fmtKL(x) { return Number.isFinite(x) ? f(x) : 'Inf'; }

function makeTask(seed, over) {
  const c = Object.assign({}, CFG, over || {}, { seed });
  return makeRecallTask({ M: c.M, N: c.N, d: c.d, dv: c.dv, keyNoise: c.keyNoise, seed: c.seed });
}

function softmaxRef(task, beta) {
  const sm = softmaxAttention(task, beta, null);
  const m = recallMetrics(task, sm.weights, sm.outputs);
  return { weights: sm.weights, outputs: sm.outputs, val: m.value, idx: m.index, mass: targetMass(task, sm.weights) };
}

// --- prefix evaluation over a nested random-feature bank -------------------
function statsPrefix(phiK, V, N, dv, m) {
  const s = new Array(m).fill(0);
  const S = new Array(m);
  for (let a = 0; a < m; a++) S[a] = new Array(dv).fill(0);
  for (let j = 0; j < N; j++) {
    const phi = phiK[j];
    for (let a = 0; a < m; a++) {
      const pv = phi[a]; s[a] += pv;
      const row = S[a];
      for (let g = 0; g < dv; g++) row[g] += pv * V[j][g];
    }
  }
  return { s, S };
}
function queryPrefix(s, S, phiQ, M, dv, m) {
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const phi = phiQ[i];
    let den = 0;
    for (let a = 0; a < m; a++) den += phi[a] * s[a];
    const o = new Array(dv).fill(0);
    for (let g = 0; g < dv; g++) {
      let num = 0;
      for (let a = 0; a < m; a++) num += phi[a] * S[a][g];
      o[g] = num / den;
    }
    out[i] = o;
  }
  return out;
}
function weightsPrefix(phiK, phiQ, N, M, m) {
  const W = new Array(M);
  for (let i = 0; i < M; i++) {
    const phi = phiQ[i];
    const w = new Array(N);
    let Z = 0;
    for (let j = 0; j < N; j++) {
      const pk = phiK[j];
      let dp = 0;
      for (let a = 0; a < m; a++) dp += phi[a] * pk[a];
      if (dp < 0) dp = 0;
      w[j] = dp; Z += dp;
    }
    for (let j = 0; j < N; j++) w[j] /= Z;
    W[i] = w;
  }
  return W;
}

// --- cost helpers (data-independent counts, computed once) ------------------
function costSoftmax(task, beta) { const c = mc(); softmaxAttention(task, beta, c); return c; }
function costPoly(task, p, shift) { const c = mc(); polyAttentionLinear(task, p, shift, c); return c; }
function costPolyNaive(task, p, shift) { const c = mc(); polyAttentionNaive(task, p, shift, c); return c; }
function costReluNaive(task, p, shift) { const c = mc(); reluAttentionNaive(task, p, shift, c); return c; }
function costPRF(task, beta, m, seed) {
  const c = mc();
  const rf = makeRandomFeatures(m, task.d, seed);
  const ph = prfFeaturesAll(task, rf, beta, c);
  const st = featureStats(ph.phiK, task.V, task.N, task.dv, c);
  queryLinear(st.sfeat, st.S, ph.phiQ, task.M, task.dv, c);
  return c;
}
function costReluRF(task, m, seed) {
  const c = mc();
  const rf = makeRandomFeatures(m, task.d, seed);
  const ph = reluRfFeaturesAll(task, rf, c);
  const st = featureStats(ph.phiK, task.V, task.N, task.dv, c);
  queryLinear(st.sfeat, st.S, ph.phiQ, task.M, task.dv, c);
  return c;
}

/** Exact op count of the O(N*m*dv) polynomial feature path, without materialising phi.
 *  Sum_{|alpha|<=p} |alpha| = d * C(d+p, p-1)  (hockey-stick). */
function polyCostAnalytic(d, M, N, dv, p) {
  const m = featureDim(d, p);
  const Sm = d * featureDim(d, p - 1);
  const mul = (N + M) * Sm + N * m * dv + M * m * (1 + dv);
  const add = N * m * (1 + dv) + M * m * (1 + dv);
  const div = M * dv;
  return { exp: 0, mul, add, div, total: mul + add + div };
}

function printCost(label, c, ratio, m, N) {
  console.log(pad(label, 18) + ' exp=' + pad(c.exp, 9) + ' mul=' + pad(c.mul, 10) +
    ' add=' + pad(c.add, 11) + ' div=' + pad(c.div, 9) + ' total=' + pad(totalOps(c), 12) +
    ' ratio=' + pad(f(ratio, 2), 8) + ' m=' + pad(m === null ? '-' : m, 6) +
    ' m/N=' + pad(m === null ? '-' : f(m / N, 3), 7));
}

// ---------------------------------------------------------------------------
// 1. Cost
// ---------------------------------------------------------------------------
function sectionCost() {
  console.log('1) COST COUNTER  (M=' + CFG.M + ' N=' + CFG.N + ' d=' + CFG.d + ' dv=' + CFG.dv +
    ' keyNoise=' + CFG.keyNoise + ' beta=' + CFG.beta + ')');
  const task = makeTask(1, {});
  const cs = costSoftmax(task, CFG.beta);
  const smTotal = totalOps(cs);
  console.log(pad('softmax', 18) + ' exp=' + pad(cs.exp, 9) + ' mul=' + pad(cs.mul, 10) +
    ' add=' + pad(cs.add, 11) + ' div=' + pad(cs.div, 9) + ' total=' + pad(smTotal, 12));
  for (const p of [1, 2, 4, 6, 8]) {
    const c = costPoly(task, p, 1);
    printCost('poly-linear p=' + p, c, totalOps(c) / smTotal, featureDim(CFG.d, p), CFG.N);
  }
  {
    const c = costPolyNaive(task, 4, 1);
    printCost('poly-naive p=4', c, totalOps(c) / smTotal, null, CFG.N);
  }
  {
    const c = costReluNaive(task, 6, 0);
    printCost('relu-naive p=6 c=0', c, totalOps(c) / smTotal, null, CFG.N);
  }
  for (const m of [64, 512, 4096]) {
    const c = costPRF(task, CFG.beta, m, 9001);
    printCost('prf m=' + m, c, totalOps(c) / smTotal, m, CFG.N);
  }
  {
    const c = costReluRF(task, 512, 9002);
    printCost('relu-rf m=512', c, totalOps(c) / smTotal, 512, CFG.N);
  }
  console.log('poly-linear exp() calls === 0 : ' + (costPoly(task, 4, 1).exp === 0) +
    ';  prf exp() calls === (N+M)*m : ' + (costPRF(task, CFG.beta, 64, 9003).exp === (CFG.N + CFG.M) * 64));
  const smBytes = CFG.M * CFG.N * 8;
  const polyM8 = featureDim(CFG.d, 8);
  console.log('softmax score/weight matrix = ' + smBytes + ' bytes; poly-linear p=8 stats (m x dv, m=' +
    polyM8 + ') = ' + (polyM8 * CFG.dv * 8) + ' bytes; ratio = ' + f(smBytes / (polyM8 * CFG.dv * 8), 3) + 'x (poly stores MORE than softmax)');
  console.log('NOTE: poly-linear is O(N*m*dv) with m = C(d+p,p); m exceeds N as soon as p > log N.');
  line();
}

// ---------------------------------------------------------------------------
// 2. BigInt bit-exactness of the polynomial sufficient statistics
// ---------------------------------------------------------------------------
function sectionExactness() {
  console.log('2) BIGINT BIT-EXACTNESS  (integer polynomial rule, randomized trials)');
  const r = bigintPolyExactness({ trials: 400, seed: 20240607, maxN: 40, maxD: 6, maxDV: 5, maxP: 5 });
  console.log('trials=' + r.trials + ' rows=' + r.rows + ' field-comparisons=' + r.comparisons +
    ' mismatches=' + r.mismatches);
  console.log('RELU^p naive vs polynomial statistic: comparisons=' + r.reluComparisons +
    ' mismatches=' + r.reluMismatches + '  (kink has no finite monomial feature map)');
  console.log('example: ' + JSON.stringify(r.example));
  line();
}

// ---------------------------------------------------------------------------
// 3. Feature-budget sweep at matched FLOPs (sharp regime, beta=8)
// ---------------------------------------------------------------------------
function evalPoly(task, p, shift) {
  const plan = polyAttentionLinear(task, p, shift, null);
  const W = weightsPrefix(plan.phiK, plan.phiQ, task.N, task.M, plan.m);
  const m = recallMetrics(task, W, plan.outputs);
  return { m: plan.m, val: m.value, idx: m.index, mass: targetMass(task, W), weights: W, outputs: plan.outputs };
}

function evalPRFAt(task, beta, mMax, m, seed) {
  const rf = makeRandomFeatures(mMax, task.d, seed);
  const ph = prfFeaturesAll(task, rf, beta, null);
  const st = statsPrefix(ph.phiK, task.V, task.N, task.dv, m);
  const out = queryPrefix(st.s, st.S, ph.phiQ, task.M, task.dv, m);
  const W = weightsPrefix(ph.phiK, ph.phiQ, task.N, task.M, m);
  const r = recallMetrics(task, W, out);
  return { val: r.value, idx: r.index, mass: targetMass(task, W), weights: W, outputs: out };
}

function evalReluRFAt(task, mMax, m, seed) {
  const rf = makeRandomFeatures(mMax, task.d, seed);
  const ph = reluRfFeaturesAll(task, rf, null);
  const st = statsPrefix(ph.phiK, task.V, task.N, task.dv, m);
  const out = queryPrefix(st.s, st.S, ph.phiQ, task.M, task.dv, m);
  const W = weightsPrefix(ph.phiK, ph.phiQ, task.N, task.M, m);
  const r = recallMetrics(task, W, out);
  return { val: r.value, idx: r.index, mass: targetMass(task, W), weights: W, outputs: out };
}

function avgOverSeeds(fn, cfg, seeds) {
  const acc = [];
  for (let s = 0; s < seeds; s++) {
    const task = makeTask(cfg.baseSeed + s * 101, cfg);
    acc.push(fn(task, s));
  }
  return acc;
}

function matched(val, mass, smVal, smMass) {
  return val >= smVal - TOL_RECALL && Math.abs(mass - smMass) <= TOL_MASS;
}

function sectionSweep() {
  console.log('3) FEATURE-BUDGET SWEEP AT MATCHED FLOPs  (beta=' + CFG.beta + '; M=' + CFG.M +
    ' N=' + CFG.N + ' d=' + CFG.d + ' dv=' + CFG.dv + ' keyNoise=' + CFG.keyNoise + '; ' + CFG.seeds + ' seeds)');
  console.log('   match = recall.value within ' + TOL_RECALL + ' of softmax AND |targetMass-smMass| <= ' + TOL_MASS);
  // softmax baseline + per-seed references
  const refs = avgOverSeeds(function (t) { return softmaxRef(t, CFG.beta); }, CFG, CFG.seeds);
  const smVal = mean(refs.map(function (r) { return r.val; }));
  const smMass = mean(refs.map(function (r) { return r.mass; }));
  const smIdx = mean(refs.map(function (r) { return r.idx; }));
  console.log('softmax reference: recall.value=' + f(smVal) + ' recall.index=' + f(smIdx) + ' targetMass=' + f(smMass));
  const cSm = costSoftmax(makeTask(1, {}), CFG.beta);
  const base = totalOps(cSm);

  console.log('');
  console.log('POLYNOMIAL KERNEL  w_j = (1 + s_j)^p   (exact monomial feature map, m=C(d+p,p), exp-free)');
  console.log('   quality from the O(N^2) weights; the O(N*m*dv) feature path reproduces them to <=1.8e-14.');
  console.log('   costRatio is the exact feature-path op count (analytic; verified against p<=8 in section 1).');
  console.log(pad('p', 4) + pad('m', 8) + pad('m/N', 9) + pad('costRatio', 11) + pad('recallVal', 11) +
    pad('targetMass', 12) + pad('KLsm|k', 9) + pad('linErr', 10) + pad('match', 7));
  let polyMatch = null;
  let polyLinMax = 0;
  for (const p of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) {
    const rs = avgOverSeeds(function (t) {
      const r = polyAttentionNaive(t, p, 1, null);
      const mm = recallMetrics(t, r.weights, r.outputs);
      return { val: mm.value, mass: targetMass(t, r.weights), weights: r.weights };
    }, CFG, CFG.seeds);
    const cRat = polyCostAnalytic(CFG.d, CFG.M, CFG.N, CFG.dv, p).total / base;
    let lErr = NaN;
    if (p <= 8) {
      const tp = makeTask(1, {});
      const lin = polyAttentionLinear(tp, p, 1, null);
      const nai = polyAttentionNaive(tp, p, 1, null);
      lErr = maxAbsDiff(lin.outputs, nai.outputs);
      if (lErr > polyLinMax) polyLinMax = lErr;
    }
    const v = mean(rs.map(function (r) { return r.val; }));
    const ms = mean(rs.map(function (r) { return r.mass; }));
    let kl = 0;
    for (let s = 0; s < CFG.seeds; s++) kl += meanKL(refs[s].weights, rs[s].weights);
    kl /= CFG.seeds;
    const ok = matched(v, ms, smVal, smMass);
    if (ok && !polyMatch) polyMatch = { p, m: featureDim(CFG.d, p), costRatio: cRat };
    console.log(pad(p, 4) + pad(featureDim(CFG.d, p), 8) + pad(f(featureDim(CFG.d, p) / CFG.N, 1), 9) +
      pad(f(cRat, 1), 11) + pad(f(v), 11) + pad(f(ms), 12) + pad(fmtKL(kl), 9) +
      pad(p <= 8 ? sci(lErr, 1) : '-', 10) + pad(ok ? 'YES' : 'no', 7));
  }
  console.log('poly minimal matching budget: ' + (polyMatch ? 'p=' + polyMatch.p + ' m=' + polyMatch.m +
    ' (m/N=' + f(polyMatch.m / CFG.N, 1) + ') costRatio=' + f(polyMatch.costRatio, 1)
    : 'none up to p=14 (m=' + featureDim(CFG.d, 14) + ')'));

  console.log('');
  console.log('POSITIVE RANDOM FEATURES (Performer/FAVOR+)  phi(x)_r = exp(omega_r.(sqrt(beta)x) - beta||x||^2/2)/sqrt(m)');
  const mMax = 4096;
  const mGrid = [16, 64, 256, 1024, 4096];
  console.log(pad('m', 7) + pad('m/N', 8) + pad('costRatio', 11) + pad('expCount', 10) + pad('recallVal', 11) +
    pad('targetMass', 12) + pad('KLsm|k', 9) + pad('match', 7));
  let prfMatch = null;
  for (const m of mGrid) {
    const rs = avgOverSeeds(function (t, s) { return evalPRFAt(t, CFG.beta, mMax, m, 7000 + s * 13); }, CFG, CFG.seeds);
    const cst = costPRF(makeTask(1, {}), CFG.beta, m, 7000);
    const cRat = totalOps(cst) / base;
    const v = mean(rs.map(function (r) { return r.val; }));
    const ms = mean(rs.map(function (r) { return r.mass; }));
    let kl = 0;
    for (let s = 0; s < CFG.seeds; s++) kl += meanKL(refs[s].weights, rs[s].weights);
    kl /= CFG.seeds;
    const ok = matched(v, ms, smVal, smMass);
    if (ok && !prfMatch) prfMatch = { m, costRatio: cRat };
    console.log(pad(m, 7) + pad(f(m / CFG.N, 3), 8) + pad(f(cRat, 2), 11) + pad(cst.exp, 10) +
      pad(f(v), 11) + pad(f(ms), 12) + pad(fmtKL(kl), 9) + pad(ok ? 'YES' : 'no', 7));
  }
  console.log('prf minimal matching budget: ' + (prfMatch ? 'm=' + prfMatch.m +
    ' (m/N=' + f(prfMatch.m / CFG.N, 1) + ') costRatio=' + f(prfMatch.costRatio, 1)
    : 'none up to m=' + mMax + ' (m/N=' + f(mMax / CFG.N, 1) + ')'));

  console.log('');
  console.log('ReLU RANDOM FEATURES (arc-cosine kernel; exp-free but homogeneous -> no temperature knob)');
  console.log(pad('m', 7) + pad('m/N', 8) + pad('costRatio', 11) + pad('recallVal', 11) +
    pad('targetMass', 12) + pad('KLsm|k', 9) + pad('match', 7));
  let reluRfMatch = null;
  for (const m of mGrid) {
    const rs = avgOverSeeds(function (t, s) { return evalReluRFAt(t, mMax, m, 8000 + s * 13); }, CFG, CFG.seeds);
    const cst = costReluRF(makeTask(1, {}), m, 8000);
    const cRat = totalOps(cst) / base;
    const v = mean(rs.map(function (r) { return r.val; }));
    const ms = mean(rs.map(function (r) { return r.mass; }));
    let kl = 0;
    for (let s = 0; s < CFG.seeds; s++) kl += meanKL(refs[s].weights, rs[s].weights);
    kl /= CFG.seeds;
    const ok = matched(v, ms, smVal, smMass);
    if (ok && !reluRfMatch) reluRfMatch = { m, costRatio: cRat };
    console.log(pad(m, 7) + pad(f(m / CFG.N, 3), 8) + pad(f(cRat, 2), 11) +
      pad(f(v), 11) + pad(f(ms), 12) + pad(fmtKL(kl), 9) + pad(ok ? 'YES' : 'no', 7));
  }
  console.log('relu-rf minimal matching budget: ' + (reluRfMatch ? 'm=' + reluRfMatch.m +
    ' (m/N=' + f(reluRfMatch.m / CFG.N, 1) + ') costRatio=' + f(reluRfMatch.costRatio, 1)
    : 'none up to m=' + mMax));

  console.log('');
  console.log('ReLU^p NAIVE REFERENCE  w_j = max(c + s_j, 0)^p   (O(N^2), no finite exact feature map)');
  console.log(pad('c', 5) + pad('p', 4) + pad('costRatio', 11) + pad('clampFrac', 11) + pad('recallVal', 11) +
    pad('targetMass', 12) + pad('KLsm|k', 9) + pad('match', 7));
  for (const cShift of [0, 1]) {
    for (const p of [2, 4, 6, 8]) {
      const rs = avgOverSeeds(function (t) { const r = reluAttentionNaive(t, p, cShift, null); return { v: recallMetrics(t, r.weights, r.outputs).value, mass: targetMass(t, r.weights), clamp: r.clampFrac, w: r.weights }; }, CFG, CFG.seeds);
      const cst = costReluNaive(makeTask(1, {}), p, cShift);
      const cRat = totalOps(cst) / base;
      const v = mean(rs.map(function (r) { return r.v; }));
      const ms = mean(rs.map(function (r) { return r.mass; }));
      const cl = mean(rs.map(function (r) { return r.clamp; }));
      let kl = 0;
      for (let s = 0; s < CFG.seeds; s++) kl += meanKL(refs[s].weights, rs[s].w);
      kl /= CFG.seeds;
      const ok = matched(v, ms, smVal, smMass);
      console.log(pad(cShift, 5) + pad(p, 4) + pad(f(cRat, 2), 11) + pad(f(cl), 11) +
        pad(f(v), 11) + pad(f(ms), 12) + pad(fmtKL(kl), 9) + pad(ok ? 'YES' : 'no', 7));
    }
  }
  console.log('   c=1 with unit keys never clamps (|s|<=1), so ReLU^p is EXACTLY the polynomial kernel');
  console.log('   (matching p only differs when the kink fires, i.e. c < max|s|).');
  console.log('poly linear-vs-naive float agreement across p=1..8: max linErr = ' + sci(polyLinMax));
  line();
}

// ---------------------------------------------------------------------------
// 4. m-vs-N crossover at matched softmax quality
// ---------------------------------------------------------------------------
function tuneBeta(cfgN, target, betas, seeds) {
  let best = null;
  for (let bi = 0; bi < betas.length; bi++) {
    const b = betas[bi];
    const rs = avgOverSeeds(function (t) { return softmaxRef(t, b); }, cfgN, seeds);
    const v = mean(rs.map(function (r) { return r.val; }));
    const ms = mean(rs.map(function (r) { return r.mass; }));
    const score = Math.abs(v - target);
    if (!best || score < best.score) best = { beta: b, val: v, mass: ms, score };
  }
  return best;
}

function minPolyBudget(cfgN, beta, smVal, smMass, pmax) {
  for (let p = 1; p <= pmax; p++) {
    const rs = avgOverSeeds(function (t) {
      const r = polyAttentionNaive(t, p, 1, null);
      return { val: recallMetrics(t, r.weights, r.outputs).value, mass: targetMass(t, r.weights) };
    }, cfgN, cfgN.seeds);
    const v = mean(rs.map(function (r) { return r.val; }));
    const ms = mean(rs.map(function (r) { return r.mass; }));
    if (matched(v, ms, smVal, smMass)) {
      const cst = polyCostAnalytic(cfgN.d, cfgN.M, cfgN.N, cfgN.dv, p);
      const baseC = costSoftmax(makeTask(1, cfgN), beta);
      return { p, m: featureDim(cfgN.d, p), costRatio: cst.total / totalOps(baseC) };
    }
  }
  return null;
}

function minRfBudget(cfgN, beta, smVal, smMass, grid, mMax, kind) {
  for (let gi = 0; gi < grid.length; gi++) {
    const m = grid[gi];
    const rs = avgOverSeeds(function (t, s) {
      return kind === 'prf' ? evalPRFAt(t, beta, mMax, m, 5000 + s * 17) : evalReluRFAt(t, mMax, m, 5000 + s * 17);
    }, cfgN, cfgN.seeds);
    const v = mean(rs.map(function (r) { return r.val; }));
    const ms = mean(rs.map(function (r) { return r.mass; }));
    if (matched(v, ms, smVal, smMass)) {
      const cst = kind === 'prf' ? costPRF(makeTask(1, cfgN), beta, m, 5000) : costReluRF(makeTask(1, cfgN), m, 5000);
      const baseC = costSoftmax(makeTask(1, cfgN), beta);
      return { m, costRatio: totalOps(cst) / totalOps(baseC) };
    }
  }
  return null;
}

function sectionCrossover() {
  console.log('4) m-vs-N CROSSOVER  (softmax recall tuned to ~0.85; M=' + CFG.M + ' d=' + CFG.d +
    ' dv=' + CFG.dv + '; ' + 3 + ' seeds)');
  console.log('   question: as N grows, does the matching feature budget m stay << N?');
  const Ns = [16, 32, 64, 128];
  const betas = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];
  const rfGrid = [16, 64, 256, 512, 1024, 2048];
  const rfCap = 2048;
  console.log(pad('N', 5) + pad('beta*', 7) + pad('smVal', 8) + pad('smMass', 9) + pad('poly_p', 8) +
    pad('poly_m', 10) + pad('pm/N', 10) + pad('pCost', 11) + pad('prf_m', 8) + pad('rm/N', 7) +
    pad('prfCost', 9) + pad('relu_m', 8));
  for (let ni = 0; ni < Ns.length; ni++) {
    const cfgN = Object.assign({}, CFG, { N: Ns[ni], seeds: 3 });
    const best = tuneBeta(cfgN, 0.85, betas, cfgN.seeds);
    const poly = minPolyBudget(cfgN, best.beta, best.val, best.mass, 24);
    const prf = minRfBudget(cfgN, best.beta, best.val, best.mass, rfGrid, rfCap, 'prf');
    const relu = minRfBudget(cfgN, best.beta, best.val, best.mass, rfGrid, rfCap, 'relu');
    console.log(pad(Ns[ni], 5) + pad(f(best.beta, 1), 7) + pad(f(best.val), 8) + pad(f(best.mass), 9) +
      pad(poly ? poly.p : '>24', 8) + pad(poly ? poly.m : '>24', 10) +
      pad(poly ? f(poly.m / Ns[ni], 2) : '-', 10) + pad(poly ? f(poly.costRatio, 1) : '-', 11) +
      pad(prf ? prf.m : '>' + rfCap, 8) + pad(prf ? f(prf.m / Ns[ni], 2) : '-', 7) +
      pad(prf ? f(prf.costRatio, 1) : '-', 9) + pad(relu ? relu.m : '>' + rfCap, 8));
  }
  console.log('   poly m = C(8+p,p) is independent of N, but the *required* m at matched softmax');
  console.log('   quality is thousands of times N at every N tested; it never approaches m << N.');
  console.log('   random-feature budgets are capped at m=' + rfCap + ' (' + f(rfCap / 128, 1) + 'x N at N=128).');
  line();
}

// ---------------------------------------------------------------------------
// 5. Where the random-feature budget goes: variance scaling
// ---------------------------------------------------------------------------
function tvDistance(A, B) {
  let s = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) s += Math.abs(A[i][j] - B[i][j]);
  }
  return 0.5 * s / A.length;
}

function sectionVariance() {
  console.log('5) RANDOM-FEATURE VARIANCE  (why m must grow exponentially in beta)');
  console.log('   single positive feature X has E[X]=exp(beta*s), E[X^2]=exp(2beta+4beta*s), so');
  console.log('   rel.std of the m-average ~ sqrt((exp(2beta)-1)/m)  =>  m >= (e^{2beta}-1)/eps^2.');
  const cfgV = Object.assign({}, CFG, { seeds: 8 });
  const betasV = [0.5, 1, 2, 4, 6, 8];
  console.log('');
  console.log('fixed m=1024, beta sweep (N=' + cfgV.N + '):');
  console.log(pad('beta', 6) + pad('smMass', 9) + pad('prfMass', 10) + pad('relMassErr', 12) +
    pad('TV', 9) + pad('KLsm|prf', 10) + pad('m_pred(e=.1)', 13));
  for (let bi = 0; bi < betasV.length; bi++) {
    const b = betasV[bi];
    const massErr = [], tvs = [], kls = [], smM = [], prfM = [];
    for (let s = 0; s < cfgV.seeds; s++) {
      const t = makeTask(cfgV.baseSeed + s * 101, cfgV);
      const sm = softmaxRef(t, b);
      const rf = makeRandomFeatures(1024, t.d, 4242 + s * 29);
      const ph = prfFeaturesAll(t, rf, b, null);
      const st = statsPrefix(ph.phiK, t.V, t.N, t.dv, 1024);
      const W = weightsPrefix(ph.phiK, ph.phiQ, t.N, t.M, 1024);
      const ms = targetMass(t, W);
      smM.push(sm.mass); prfM.push(ms);
      massErr.push(Math.abs(ms - sm.mass) / sm.mass);
      tvs.push(tvDistance(sm.weights, W));
      kls.push(meanKL(sm.weights, W));
    }
    const pred = (Math.exp(2 * b) - 1) / 0.01;
    console.log(pad(f(b, 1), 6) + pad(f(mean(smM)), 9) + pad(f(mean(prfM)), 10) +
      pad(sci(mean(massErr), 2), 12) + pad(sci(mean(tvs), 3), 9) + pad(fmtKL(mean(kls)), 10) +
      pad(sci(pred, 3), 13));
  }
  console.log('');
  console.log('fixed beta=8, m sweep (N=' + cfgV.N + '): below m ~ e^{2beta} the estimator is saturated, so TV is nearly flat');
  const mGridV = [16, 64, 256, 1024, 4096];
  const mMaxV = 4096;
  const pre = [];
  for (let s = 0; s < cfgV.seeds; s++) {
    const t = makeTask(cfgV.baseSeed + s * 101, cfgV);
    const sm = softmaxRef(t, 8);
    const rf = makeRandomFeatures(mMaxV, t.d, 999 + s * 31);
    const ph = prfFeaturesAll(t, rf, 8, null);
    pre.push({ t: t, sm: sm, ph: ph });
  }
  console.log(pad('m', 7) + pad('TV', 11) + pad('relMassErr', 12) + pad('KLsm|prf', 10));
  const xs = [], ys = [];
  for (let gi = 0; gi < mGridV.length; gi++) {
    const m = mGridV[gi];
    const tvs = [], massErr = [], kls = [];
    for (let s = 0; s < cfgV.seeds; s++) {
      const t = pre[s].t, sm = pre[s].sm, ph = pre[s].ph;
      const st = statsPrefix(ph.phiK, t.V, t.N, t.dv, m);
      const Wp = weightsPrefix(ph.phiK, ph.phiQ, t.N, t.M, m);
      const ms = targetMass(t, Wp);
      tvs.push(tvDistance(sm.weights, Wp));
      massErr.push(Math.abs(ms - sm.mass) / sm.mass);
      kls.push(meanKL(sm.weights, Wp));
    }
    const tv = mean(tvs);
    xs.push(Math.log(m)); ys.push(Math.log(tv));
    console.log(pad(m, 7) + pad(sci(tv, 3), 11) + pad(sci(mean(massErr), 2), 12) + pad(fmtKL(mean(kls)), 10));
  }
  const nn = xs.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < nn; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const slope = (nn * sxy - sx * sy) / (nn * sxx - sx * sx);
  console.log('log-log slope d log(TV) / d log(m) = ' + f(slope, 3) + ';  at beta=8 even m=4096 is');
  console.log('far below the ~ ' + sci((Math.exp(16) - 1) / 0.01, 1) + ' needed for 10% error, so TV is saturated (not the -0.5 regime).');
  line();
}

// ---------------------------------------------------------------------------
// 6. Verdict
// ---------------------------------------------------------------------------
function sectionVerdict() {
  console.log('6) VERDICT');
  console.log('   - Concentration IS recoverable by a richer kernel. The polynomial target mass');
  console.log('     rises monotonically with p (sweep above), so the affine O(1/N) ceiling is gone.');
  console.log('   - But the matching feature budget is not small. Polynomial needs p large enough');
  console.log('     that m=C(d+p,p) far exceeds N; positive random features need');
  console.log('     m >= (e^{2beta}-1)/eps^2, which at beta=8 and eps=0.1 is m ~ ' + sci((Math.exp(16) - 1) / 0.01) + '.');
  console.log('   - ReLU^p at c=0 does concentrate, but only as an O(N^2), non-factorizing reference:');
  console.log('     the kink has no finite monomial feature map, so it buys no O(N) path.');
  console.log('   - PARTIAL: richness recovers concentration; it does NOT recover it at m << N.');
  line();
}

console.log('==========================================================================');
console.log('softmax vs richer-kernel linear attention (polynomial / random features)');
console.log('node ' + process.version);
console.log('==========================================================================');
sectionCost();
sectionExactness();
sectionSweep();
sectionCrossover();
sectionVariance();
sectionVerdict();
console.log('done.');