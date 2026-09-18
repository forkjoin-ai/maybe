/**
 * run.mjs -- main runnable entry for the Buleyean Hamming / rejection-count
 * attention experiment (@a0n/maybe).
 *
 * Sections
 *   1. score exactness demo            (popcount == explicit; ternary masks == dot)
 *   2. primitive-instruction cost model (bit-ops vs FMA, W=32 and W=64)
 *   3. BigInt exactness + consensus + floor
 *   4. quality sweep over R (budget) and bit-width at the sibling task (beta=8)
 *   5. concentration: the Hamming kernel's own saturation ceiling (Hamming margin)
 *   6. N sweep
 *   7. ambient-dimension / bit-width crossover at matched softmax quality
 *   8. JS micro-benchmark (explicitly non-representative)
 *   9. verdict
 *
 * Real Node (the repo node is a shim):
 *   cd open-source/maybe/experiments/buleyean-hamming-attention
 *   /opt/homebrew/bin/node run.mjs
 * Everything is seeded (mulberry32) and deterministic; no network, no filesystem,
 * no dependencies (not even Node builtins).
 */
import {
  makeRecallTask, keyRange,
  buildBinaryCodes, buildTernaryCodes, dequantKeys,
  rejectionMatrix, hammingPosterior,
  hammingSoftmaxPosterior, hammingSoftmaxOutputsFloat, hammingSoftmaxTableExactness,
  quantizeValues, softmaxGeneral,
  recallMetrics, recallValueOnly, targetMass, meanKL, marginStats,
  softmaxCost, hammingCost, hammingSoftmaxCost, softmaxOnCodesCost,
  bigintHammingExactness, popcountXor, popcountAnd,
} from './hamming-attention.mjs';

const SEEDS = [1, 102, 203, 304];
const HS_BETAS = [1, 2, 5, 10, 20, 50, 100, 200];
const smCache = new Map();

function padL(s, n) { s = String(s); while (s.length < n) s += ' '; return s; }
function fx(x, n) { if (x === Infinity) return 'Inf'; if (x === -Infinity) return '-Inf'; if (!isFinite(x)) return String(x); return (Math.abs(x) < 1e-4 && x !== 0) ? x.toExponential(2) : x.toFixed(n); }
function line(c, n) { let s = ''; for (let i = 0; i < n; i++) s += c; return s; }
function section(title) { console.log(''); console.log(line('-', 74)); console.log(title); console.log(line('-', 74)); }

// ---------------------------------------------------------------------------
// softmax reference: fixed beta, or tuned so recall is closest to a target
// ---------------------------------------------------------------------------
function pickSoftmax(Q, K, V, task, betas, mode) {
  if (mode && mode.beta !== undefined) {
    const sm = softmaxGeneral(Q, K, V, mode.beta, null);
    const m = recallMetrics(task, sm.weights, sm.outputs);
    return { beta: mode.beta, value: m.value, index: m.index, mass: targetMass(task, sm.weights), weights: sm.weights, score: 0 };
  }
  let best = null;
  for (const beta of betas) {
    const sm = softmaxGeneral(Q, K, V, beta, null);
    const m = recallMetrics(task, sm.weights, sm.outputs);
    const mass = targetMass(task, sm.weights);
    // targetMass mode holds the softmax target mass fixed (the d=8, beta=8
    // operating point) so the reference does not saturate as d grows; the
    // recall mode tunes recall, with a lower-mass tie-break.
    const score = mode.targetMass !== undefined
      ? Math.abs(mass - mode.targetMass)
      : Math.abs(m.value - mode.target);
    const tie = mode.targetMass !== undefined ? m.value : mass;
    if (!best || score < best.score || (score === best.score && tie > best.tie)) {
      best = { beta, value: m.value, index: m.index, mass, weights: sm.weights, score, tie };
    }
  }
  return best;
}

function softmaxCached(key, Q, K, V, task, betas, mode) {
  if (smCache.has(key)) return smCache.get(key);
  const r = pickSoftmax(Q, K, V, task, betas, mode);
  smCache.set(key, r);
  return r;
}

function rGrid(dEff) {
  const step = Math.max(1, Math.round(dEff / 40));
  const g = [];
  for (let R = 0; R <= dEff; R += step) g.push(R);
  if (g[g.length - 1] !== dEff) g.push(dEff);
  return g;
}

/**
 * Sweep the Buleyean budget R.
 *   ham*     : full method (integer weights x quantised small-integer V)
 *   hamFloat : kernel only (the SAME integer weights x the real float V)
 *               -- isolates the score/kernel ceiling from value quantisation.
 */
function hammingSweep(vv, grid, Vint, vscale, task, smWeights) {
  let best = null, maxMass = -1, minP = Infinity, floatBest = -1, indexBest = 0;
  const curve = [];
  for (const R of grid) {
    const hp = hammingPosterior(vv, R, Vint, vscale, true);
    const m = recallMetrics(task, hp.weights, hp.outputs);
    const mass = targetMass(task, hp.weights);
    const kl = smWeights ? meanKL(smWeights, hp.weights) : NaN;
    const hf = hammingPosterior(vv, R, task.V, 1, false);
    const fv = recallValueOnly(task, hf.outputs);
    if (hp.minProb < minP) minP = hp.minProb;
    if (mass > maxMass) maxMass = mass;
    if (fv > floatBest) floatBest = fv;
    if (m.index > indexBest) indexBest = m.index;
    curve.push({ R, mass, value: m.value, floatValue: fv });
    if (!best || m.value > best.value || (m.value === best.value && mass > best.mass)) {
      best = { R, value: m.value, index: m.index, mass, kl };
    }
  }
  return { best, maxMass, minP, floatBest, indexBest, curve };
}

/** Sweep beta for softmax normalization over the exact integer Hamming scores. */
function hsSweep(vv, sMax, sMin, Vint, vscale, task, smWeights, betas) {
  let best = null, maxMass = -1, floatBest = -1, minP = Infinity;
  for (const beta of betas) {
    const hp = hammingSoftmaxPosterior(vv, beta, sMax, sMin, Vint, vscale, true);
    const m = recallMetrics(task, hp.weights, hp.outputs);
    const mass = targetMass(task, hp.weights);
    const kl = smWeights ? meanKL(smWeights, hp.weights) : NaN;
    const fv = recallValueOnly(task, hammingSoftmaxOutputsFloat(vv, beta, sMax, sMin, task.V));
    if (hp.minProb < minP) minP = hp.minProb;
    if (mass > maxMass) maxMass = mass;
    if (fv > floatBest) floatBest = fv;
    if (!best || m.value > best.value || (m.value === best.value && mass > best.mass)) {
      best = { beta, value: m.value, index: m.index, mass, kl };
    }
  }
  return { best, maxMass, floatBest, minP };
}

function buildCodesFor(task, bits, kind) {
  const kmax = keyRange(task);
  if (kind === 'ternary') {
    return { Kcodes: buildTernaryCodes(task.K, kmax / 3), Qcodes: buildTernaryCodes(task.Q, kmax / 3), kind: 'ternary', kmax };
  }
  return { Kcodes: buildBinaryCodes(task.K, bits, kmax), Qcodes: buildBinaryCodes(task.Q, bits, kmax), kind: bits === 1 ? 'binary' : 'bit', kmax };
}

function evalOpts(opts, betas, mode, seeds) {
  seeds = seeds || SEEDS;
  mode = mode || { beta: 8 };
  const acc = {
    smValue: 0, smMass: 0, smBeta: 0, smIndex: 0,
    sqValue: 0, sqMass: 0,
    hamValue: 0, hamFloat: 0, hamMass: 0, hamMaxMass: 0, hamKL: 0, hamR: 0, hamIndex: 0,
    hsValue: 0, hsFloat: 0, hsMass: 0, hsMaxMass: 0, hsKL: 0, hsBeta: 0,
    ceil: 0, exactMax: 0, vt: 0, vmin: 0, delta: 0,
    minP: 1, dEff: 0, kmax: 0, n: 0,
  };
  for (const seed of seeds) {
    const task = makeRecallTask({ M: opts.M, N: opts.N, d: opts.d, dv: opts.dv, seed, keyNoise: opts.keyNoise });
    const c = buildCodesFor(task, opts.bits, opts.kind);
    const dEff = c.Kcodes[0].dEff;
    const vv = rejectionMatrix(c.Kcodes, c.Qcodes, c.kind);
    const vkind = c.kind === 'ternary' ? 'ternary' : (opts.bits === 1 ? 'binary' : 'bit');
    const vq = quantizeValues(task.V, vkind, opts.bits);
    const smb = softmaxCached('f|' + seed + '|' + opts.M + '|' + opts.N + '|' + opts.d + '|' + opts.dv + '|' + opts.keyNoise + '|' + JSON.stringify(mode),
      task.Q, task.K, task.V, task, betas, mode);
    const Qq = dequantKeys(c.Qcodes, c.kmax), Kq = dequantKeys(c.Kcodes, c.kmax);
    const sqb = softmaxCached('q|' + seed + '|' + opts.M + '|' + opts.N + '|' + opts.d + '|' + opts.dv + '|' + opts.bits + '|' + c.kind + '|' + JSON.stringify(mode),
      Qq, Kq, task.V, task, betas, mode);
    const hs = hammingSweep(vv, rGrid(dEff), vq.Vint, vq.vscale, task, smb.weights);
    const hss = hsSweep(vv, c.Kcodes[0].sMax, c.Kcodes[0].sMin, vq.Vint, vq.vscale, task, smb.weights, HS_BETAS);
    const ms = marginStats(vv, task.targets);
    acc.smValue += smb.value; acc.smMass += smb.mass; acc.smBeta += smb.beta; acc.smIndex += smb.index;
    acc.sqValue += sqb.value; acc.sqMass += sqb.mass;
    acc.hamValue += hs.best.value; acc.hamFloat += hs.floatBest; acc.hamMass += hs.best.mass;
    acc.hamMaxMass += hs.maxMass; acc.hamKL += hs.best.kl; acc.hamR += hs.best.R; acc.hamIndex += hs.indexBest;
    acc.hsValue += hss.best.value; acc.hsFloat += hss.floatBest; acc.hsMass += hss.best.mass;
    acc.hsMaxMass += hss.maxMass; acc.hsKL += hss.best.kl; acc.hsBeta += hss.best.beta;
    if (hss.minP < acc.minP) acc.minP = hss.minP;
    acc.ceil += ms.ceilingMean; acc.exactMax += ms.exactMaxMean;
    acc.vt += ms.vtMean; acc.vmin += ms.vminMean; acc.delta += ms.deltaMean;
    if (hs.minP < acc.minP) acc.minP = hs.minP;
    acc.dEff = dEff; acc.kmax = c.kmax; acc.n++;
  }
  const n = acc.n;
  for (const k of Object.keys(acc)) if (k !== 'dEff' && k !== 'kmax' && k !== 'n') acc[k] /= n;
  return acc;
}

// ---------------------------------------------------------------------------
// 1. score exactness demo
// ---------------------------------------------------------------------------
function packBitsFast(bits) {
  const words = new Array((bits.length + 31) >>> 5).fill(0);
  for (let f = 0; f < bits.length; f++) if (bits[f]) words[f >>> 5] = (words[f >>> 5] | (1 << (f & 31))) >>> 0;
  return words;
}

function scoreDemo() {
  section('1) SCORE EXACTNESS DEMO  (popcount == explicit; ternary masks == signed dot)');
  let cmp = 0, bad = 0;
  for (let t = 0; t < 2000; t++) {
    const d = 1 + (t % 37);
    const a = new Array(d), b = new Array(d);
    for (let f = 0; f < d; f++) { a[f] = (t * 7 + f * 13) % 3 === 0 ? 1 : 0; b[f] = (t * 5 + f * 11) % 4 < 2 ? 1 : 0; }
    let naive = 0;
    for (let f = 0; f < d; f++) if (a[f] !== b[f]) naive++;
    cmp++; if (popcountXor(packBitsFast(a), packBitsFast(b)) !== naive) bad++;
  }
  console.log('binary  : popcount(xor) == explicit Hamming distance over ' + cmp + ' random pairs; mismatches=' + bad);
  let tcmp = 0, tbad = 0, sample = null;
  for (let t = 0; t < 2000; t++) {
    const d = 1 + (t % 23);
    const a = new Array(d), b = new Array(d);
    for (let f = 0; f < d; f++) { a[f] = ((t * 7 + f) % 3) - 1; b[f] = ((t * 3 + f * 2) % 3) - 1; }
    let dot = 0;
    for (let f = 0; f < d; f++) dot += a[f] * b[f];
    let A = 0, B = 0;
    for (let f = 0; f < d; f++) { if (a[f] === 1 && b[f] === 1) A++; else if (a[f] === -1 && b[f] === -1) A++; else if (a[f] === 1 && b[f] === -1) B++; else if (a[f] === -1 && b[f] === 1) B++; }
    tcmp++; if (dot !== A - B) tbad++;
    if (!sample && d >= 4) sample = { d, dot, A, B };
  }
  console.log('ternary : A - B (four AND-popcounts) == exact ternary dot over ' + tcmp + ' random pairs; mismatches=' + tbad);
  console.log('ternary example (d=' + sample.d + '): dot=' + sample.dot + '  A=' + sample.A + '  B=' + sample.B);
  // IDENTITY (correct sign-vector form): for z_q, z_k in {-1,+1}^d the number of
  // sign disagreements is v = (d - z_q.z_k)/2, so the Hamming score is an AFFINE
  // function of the signed dot product: d - v = (d + z_q.z_k)/2.  Hence softmax
  // over the Hamming score IS softmax over the quantised dot product (temperature
  // x2) and loses no concentration.  NOTE: the literal {0,1}^d claim
  // "popcount(xor) = d - q.k" is FALSE -- the agreement count is
  // d - |q| - |k| + 2 q.k -- and is counted below to make that explicit.
  let icmp = 0, ibad = 0, ibad01 = 0;
  for (let t = 0; t < 2000; t++) {
    const d = 1 + (t % 37);
    const zq = new Array(d), zk = new Array(d);
    for (let f = 0; f < d; f++) { zq[f] = ((t * 7 + f * 13) % 2) ? 1 : -1; zk[f] = ((t * 5 + f * 11) % 2) ? 1 : -1; }
    let dot = 0; for (let f = 0; f < d; f++) dot += zq[f] * zk[f];
    let ham = 0; for (let f = 0; f < d; f++) if (zq[f] !== zk[f]) ham++;
    icmp++; if ((d + dot) / 2 !== d - ham) ibad++;
    let q01 = 0, ham01 = 0;
    for (let f = 0; f < d; f++) { const a = zq[f] > 0 ? 1 : 0, b = zk[f] > 0 ? 1 : 0; q01 += a * b; if (a !== b) ham01++; }
    if (d - ham01 !== q01) ibad01++;
  }
  console.log('IDENTITY (sign vectors): d - popcount(xor) == (d + z_q.z_k)/2 over ' + icmp + ' random pairs; mismatches=' + ibad);
  console.log('  => the Hamming score is an AFFINE function of the signed dot product; softmax over it is');
  console.log('     softmax over the quantised dot product (temperature x2) and loses no concentration.');
  console.log('  [literal {0,1} claim "popcount(xor) = d - q.k" is false: mismatches=' + ibad01 + '/' + icmp + ']');
}

// ---------------------------------------------------------------------------
// 2. cost model
// ---------------------------------------------------------------------------
function costSection() {
  section('2) COST MODEL (primitive instructions, not wall time)  M=32 N=64 d=8 dv=8');
  const M = 32, N = 64, d = 8, dv = 8;
  const sm = softmaxCost(M, N, d, dv);
  console.log('float softmax : score=' + padL(sm.scoreFma, 7) + ' FMA  value=' + padL(sm.valueFma, 7) +
    ' FMA  exp=' + padL(sm.exp, 6) + '  div=' + padL(sm.div, 6) + '  total=' + padL(sm.total, 8));
  console.log('Hamming       : score = M*N*ceil(dEff/W) bit-ops; value = M*N*dv integer multiply-adds; exp = 0');
  console.log('');
  console.log('   method        dEff  wW32  bitopsW32  totalW32  ratioW32   wW64  bitopsW64  totalW64  ratioW64');
  const rows = [
    ['binary  b=1', 8, 'binary'],
    ['bit     b=2', 16, 'bit'],
    ['bit     b=4', 32, 'bit'],
    ['ternary    ', 16, 'ternary'],
  ];
  for (const [label, dEff, kind] of rows) {
    const h32 = hammingCost(M, N, dEff, dv, 32, kind);
    const h64 = hammingCost(M, N, dEff, dv, 64, kind);
    console.log(padL(label, 14) + padL(dEff, 5) + padL(h32.words, 6) + padL(h32.bitops, 11) + padL(h32.total, 10) +
      padL(fx(sm.total / h32.total, 3) + 'x', 10) + padL(h64.words, 6) + padL(h64.bitops, 11) + padL(h64.total, 10) +
      padL(fx(sm.total / h64.total, 3) + 'x', 10));
  }
  console.log('');
  console.log('at d=8 the score term is 8 of ~18 softmax primitives per pair, so the total Hamming win is small;');
  console.log('the exp() count is removed entirely (softmax makes M*N=2048, Hamming makes 0).');
  console.log('');
  console.log('MATCHED-INFORMATION baseline (float softmax on the same dEff dequantised coordinates):');
  for (const [label, dEff, kind] of rows) {
    const smc = softmaxOnCodesCost(M, N, dEff, dv);
    const h32 = hammingCost(M, N, dEff, dv, 32, kind);
    const h64 = hammingCost(M, N, dEff, dv, 64, kind);
    console.log(padL(label, 14) + 'dEff=' + padL(dEff, 5) + '  softmaxOnCodes.total=' + padL(smc.total, 8) +
      '  hamming total W32=' + padL(h32.total, 7) + ' (score ratio ' + fx(smc.scoreFma / h32.scoreBitOps, 1) + 'x)' +
      '  W64 score ratio ' + fx(smc.scoreFma / h64.scoreBitOps, 1) + 'x');
  }
  console.log('');
  console.log('CAVEAT: a b-bit code stores b bits/coordinate vs 32 bits for a float. Matching the exact BIT budget');
  console.log('means dEff(float) = d*b, and then the score win is W/32 (1x at W=32, 2x at W=64); the larger');
  console.log('wins above come from the lower-precision representation, not from popcount as such.');
  console.log('');
  console.log('SOFTMAX normalization over the integer Hamming scores (precomputed exp table, one memory read per pair):');
  console.log('  float softmax: total=' + sm.total + ' (exp=' + sm.exp + '), score=' + sm.scoreFma + ' FMA');
  console.log('   method        dEff  bitopsW32  tableReads  expCalls  value  totalW32  ratioW32   totalW64  ratioW64');
  for (const [label, dEff, kind] of rows) {
    const range = kind === 'ternary' ? dEff : dEff; // score range sMax-sMin
    const h32 = hammingSoftmaxCost(M, N, dEff, dv, 32, kind, range, range + 1);
    const h64 = hammingSoftmaxCost(M, N, dEff, dv, 64, kind, range, range + 1);
    console.log(padL(label, 14) + padL(dEff, 5) + padL(h32.bitops, 11) + padL(h32.tableReads, 12) + padL(h32.expCalls, 10) +
      padL(h32.imad, 7) + padL(h32.total, 10) + padL(fx(sm.total / h32.total, 3) + 'x', 10) +
      padL(h64.total, 10) + padL(fx(sm.total / h64.total, 3) + 'x', 10));
  }
  console.log('  expCalls is a CONSTANT (sMax-sMin+1), independent of M and N; the A.V value matmul (M*N*dv) remains.');
}

// ---------------------------------------------------------------------------
// 3. BigInt exactness
// ---------------------------------------------------------------------------
function exactnessSection() {
  section('3) BIGINT INTEGER EXACTNESS + CONSENSUS + FLOOR  (randomized)');
  const r = bigintHammingExactness({ trials: 400, seed: 20250701, maxN: 40, maxD: 8, maxDV: 5 });
  console.log('trials=' + r.trials + ' rows=' + r.rows + ' comparisons=' + r.comparisons + ' mismatches=' + r.mismatches);
  console.log('  rejection counts + weights + posterior num/den: naive per-coordinate vs packed popcount, literal BigInt equality');
  console.log('ternary naive-dot vs A-B: comparisons=' + r.ternaryComparisons + ' mismatches=' + r.ternaryMismatches);
  console.log('multi-view fuse (componentwise weight addition) vs naive per-view sum: comparisons=' + r.fusedComparisons + ' mismatches=' + r.fusedMismatches);
  console.log('fusion ORDER independence (shuffled view order): comparisons=' + r.orderComparisons + ' mismatches=' + r.orderMismatches);
  console.log('floor: min integer weight over all trials = ' + r.minWeight + '  (w = R - min(v,R) + 1 >= 1, so every key keeps positive mass)');
  console.log('control: sum rejection counts then God Formula vs sum of weights: comparisons=' + r.fuseThenRuleComparisons +
    ' mismatches=' + r.fuseThenRuleMismatches + '  (the min-clamp is NOT linear; only the count(weight) vectors fuse exactly)');
  console.log('example=' + JSON.stringify(r.example));
  const te = hammingSoftmaxTableExactness({ trials: 400, seed: 20250702, maxN: 40, maxRange: 64 });
  console.log('softmax-table exp: table lookup === direct Math.exp, comparisons=' + te.comparisons + ' mismatches=' + te.mismatches +
    '; tableExpEvals=' + te.tableEvals + ' vs directExpEvals=' + te.directEvals + ' (constant table vs per-pair exp)');
  console.log('  example=' + JSON.stringify(te.example));
}

// ---------------------------------------------------------------------------
// 4. quality over R and bit-width at the sibling task
// ---------------------------------------------------------------------------
function qualitySection() {
  section('4) QUALITY SWEEP OVER R AND BIT-WIDTH  (M=32 N=64 d=8 dv=8 keyNoise=0.1; beta=8; 4 seeds)');
  console.log('smVal/smMass = float softmax at beta=8 (sibling sharp regime); sq* = softmax on the SAME dequantised codes');
  console.log('hamVal = full method (integer weights x quantised V); hamFloat = kernel only (integer weights x real float V)');
  console.log('match = hamVal within 0.02 of smVal AND |hamMass - smMass| <= 0.05; chance = 1/N = 0.0156');
  const betas = [8];
  const mode = { beta: 8 };
  const base = { M: 32, N: 64, d: 8, dv: 8, keyNoise: 0.1 };
  const defs = [
    ['binary b=1', { bits: 1, kind: 'binary' }],
    ['bit    b=2', { bits: 2, kind: 'bit' }],
    ['bit    b=4', { bits: 4, kind: 'bit' }],
    ['ternary   ', { bits: 2, kind: 'ternary' }],
  ];
  console.log('');
  console.log('  method      dEff  smVal  smMass  sqVal  hamVal  hamFloat  hamMass  hsVal  hsFloat  hsMass  hsKL  hsBeta  ceil   exactMax');
  for (const [label, o] of defs) {
    const a = evalOpts(Object.assign({}, base, o), betas, mode);
    console.log(padL(label, 11) + padL(a.dEff, 5) + padL(fx(a.smValue, 3), 7) + padL(fx(a.smMass, 3), 8) +
      padL(fx(a.sqValue, 3), 7) + padL(fx(a.hamValue, 3), 8) + padL(fx(a.hamFloat, 3), 10) + padL(fx(a.hamMass, 3), 8) +
      padL(fx(a.hsValue, 3), 7) + padL(fx(a.hsFloat, 3), 8) + padL(fx(a.hsMass, 3), 8) + padL(fx(a.hsKL, 3), 9) +
      padL(fx(a.hsBeta, 1), 8) + padL(fx(a.ceil, 3), 7) + padL(fx(a.exactMax, 3), 9));
  }
  console.log('');
  console.log('hsVal/hsFloat/hsMass = SOFTMAX normalization over the same integer Hamming scores (exp table);');
  console.log('sqVal = float softmax on the dequantised codes (same scores, cross-check).');
  console.log('');
  console.log('At d=8 the Hamming margin is under one bit of separation on average and the linear God Formula');
  console.log('saturates at (delta+1)/(delta+N) ~ O(1/N); hamFloat (the kernel alone, with real values) is at');
  console.log('chance too, so the failure is the score/kernel, not the value quantisation.');
  // explicit R curve for the strongest d=8 codec
  const bits4 = defs[2][1];
  const task = makeRecallTask({ M: 32, N: 64, d: 8, dv: 8, seed: SEEDS[0], keyNoise: 0.1 });
  const c = buildCodesFor(task, bits4.bits, bits4.kind);
  const vv = rejectionMatrix(c.Kcodes, c.Qcodes, c.kind);
  const vq = quantizeValues(task.V, 'bit', 4);
  const hs = hammingSweep(vv, [0, 1, 2, 4, 8, 12, 16, 24, 32], vq.Vint, vq.vscale, task, null);
  console.log('');
  console.log('R curve, ' + bits4.kind + ' b=4, seed ' + SEEDS[0] + ':  R : targetMass : hamVal : hamFloat');
  for (const p of hs.curve) {
    if ([0, 1, 2, 4, 8, 12, 16, 24, 32].indexOf(p.R) >= 0) {
      console.log('    ' + padL(p.R, 3) + ' : ' + padL(fx(p.mass, 4), 10) + ' : ' + padL(fx(p.value, 3), 6) + ' : ' + padL(fx(p.floatValue, 3), 6));
    }
  }
}

// ---------------------------------------------------------------------------
// 5. concentration / saturation ceiling
// ---------------------------------------------------------------------------
function ceilingSection() {
  section('5) CONCENTRATION: THE HAMMING KERNEL HAS ITS OWN SATURATION CEILING  (d=8 N=64; beta=8)');
  console.log('per row: v_t = target rejection, v_min = closest non-target rejection, delta = v_min - v_t');
  console.log('best target mass over ANY budget R is the exact max of w_t(R)/sum_j w_j(R); at R=v_min it is (delta+1)/(delta+N)');
  console.log('ceil uses mean delta; exactMax is the mean of the exact per-row maxima (breakpoint scan, no grid).');
  console.log('');
  console.log('  method      dEff   v_t    v_min   delta   ceil(mean delta)   exactMaxOverR   softmaxMass   achievedMax');
  const betas = [8]; const mode = { beta: 8 };
  const base = { M: 32, N: 64, d: 8, dv: 8, keyNoise: 0.1 };
  const defs = [
    ['binary b=1', { bits: 1, kind: 'binary' }],
    ['bit    b=2', { bits: 2, kind: 'bit' }],
    ['bit    b=4', { bits: 4, kind: 'bit' }],
    ['ternary   ', { bits: 2, kind: 'ternary' }],
  ];
  for (const [label, o] of defs) {
    const a = evalOpts(Object.assign({}, base, o), betas, mode);
    console.log(padL(label, 11) + padL(a.dEff, 5) + padL(fx(a.vt, 2), 7) + padL(fx(a.vmin, 2), 7) +
      padL(fx(a.delta, 2), 7) + padL(fx(a.ceil, 4), 18) + padL(fx(a.exactMax, 4), 16) + padL(fx(a.smMass, 4), 13) + padL(fx(a.hamMaxMass, 4), 13));
  }
  console.log('');
  console.log('The ceiling is a function of the integer MARGIN: target mass ~= (delta+1)/(delta+N).  Since the');
  console.log('margin grows like dEff/2 - O(sqrt(dEff)), concentration needs dEff >> N.  This is the Hamming');
  console.log('analogue of the affine O(1/N) ceiling: it is reached at finite R and no larger R beats it.');
}

// ---------------------------------------------------------------------------
// 6. N sweep
// ---------------------------------------------------------------------------
function nSection() {
  section('6) N SWEEP  (d=8 dv=8 keyNoise=0.1; 4 seeds; softmax tuned to recall ~0.85; hamming over R)');
  const betas = [2, 3, 4, 5, 6, 8]; const mode = { targetMass: 0.64 };
  const rows = [16, 32, 64, 128, 256];
  console.log('   N   method      dEff  smVal  smMass  hamFloat  hamVal  hamMass  hamMax  ceil   bestR');
  for (const N of rows) {
    for (const bits of [1, 4]) {
      const a = evalOpts({ M: 32, N, d: 8, dv: 8, keyNoise: 0.1, bits, kind: bits === 1 ? 'binary' : 'bit' }, betas, mode);
      console.log(padL(N, 5) + padL(bits === 1 ? 'binary b=1' : 'bit    b=4', 11) + padL(a.dEff, 5) +
        padL(fx(a.smValue, 3), 7) + padL(fx(a.smMass, 3), 8) + padL(fx(a.hamFloat, 3), 10) + padL(fx(a.hamValue, 3), 8) +
        padL(fx(a.hamMass, 3), 9) + padL(fx(a.hamMaxMass, 3), 8) + padL(fx(a.ceil, 3), 7) + padL(fx(a.hamR, 1), 7));
    }
  }
  console.log('');
  console.log('chance is 1/N.  At fixed dEff the margin does not grow with N, so the ceiling falls like 1/N.');
}

// ---------------------------------------------------------------------------
// 7. ambient-dimension / bit-width crossover
// ---------------------------------------------------------------------------
function crossoverSection() {
  section('7) AMBIENT-DIMENSION / BIT-WIDTH CROSSOVER AT MATCHED SOFTMAX QUALITY');
  console.log('M=32 N=64 dv=8; unit keys, keyNoise = 0.1*sqrt(8/d) so the total query-noise norm is held fixed;');
  console.log('softmax beta tuned to the sibling sharp regime (target mass 0.64, the d=8/beta=8 point); 8 seeds.');
  console.log('hamFloat = kernel only (integer weights x real V); hamVal = full method (x quantised V).');
  console.log('massMatch (one-sided) = hamFloat within 0.02 of smVal AND hamMass >= smMass - 0.05 (overshooting mass is not a failure);');
  console.log('recallMatch = hamFloat alone; chance = 1/N.');
  console.log('');
  const betas = [2, 3, 4, 5, 6, 8];
  const mode = { targetMass: 0.64 };
  const cseeds = [1, 102, 203, 304, 405, 506];
  const dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
  const bitsSet = [1, 2, 4];
  const table = [];
  console.log('    d  bits   dEff   smVal  smMass  sqVal  hamFloat  hamVal  hamMass  hamMax  kernel  recall  full   ceil   exactMax  ratioW32  ratioW64');
  for (const d of dims) {
    for (const bits of bitsSet) {
      const kind = bits === 1 ? 'binary' : 'bit';
      const a = evalOpts({ M: 32, N: 64, d, dv: 8, keyNoise: 0.1 * Math.sqrt(8 / d), bits, kind }, betas, mode, cseeds);
      const dEff = a.dEff;
      const h32 = hammingCost(32, 64, dEff, 8, 32, kind);
      const h64 = hammingCost(32, 64, dEff, 8, 64, kind);
      const sm = softmaxCost(32, 64, d, 8);
      const smInfo = softmaxOnCodesCost(32, 64, dEff, 8);
      const scoreRange = dEff;   // sMax - sMin of the integer Hamming score
      const hs32 = hammingSoftmaxCost(32, 64, dEff, 8, 32, kind, scoreRange, scoreRange + 1);
      const hs64 = hammingSoftmaxCost(32, 64, dEff, 8, 64, kind, scoreRange, scoreRange + 1);
      const kernelMatch = (a.hamFloat >= a.smValue - 0.02) && (a.hamMass >= a.smMass - 0.05);
      const recallMatch = (a.hamFloat >= a.smValue - 0.02);
      const fullMatch = (a.hamValue >= a.smValue - 0.02) && (a.hamMass >= a.smMass - 0.05);
      const hsMatch = (a.hsFloat >= a.smValue - 0.02) && (a.hsMass >= a.smMass - 0.05);
      const hsFullMatch = (a.hsValue >= a.smValue - 0.02) && (a.hsMass >= a.smMass - 0.05);
      const r32 = sm.total / h32.total, r64 = sm.total / h64.total;
      const hsr32 = sm.total / hs32.total, hsr64 = sm.total / hs64.total;
      const info32 = smInfo.scoreFma / h32.scoreBitOps, info64 = smInfo.scoreFma / h64.scoreBitOps;
      table.push({ d, bits, dEff, smVal: a.smValue, smMass: a.smMass, sqVal: a.sqValue,
        hamFloat: a.hamFloat, hamVal: a.hamValue, hamMass: a.hamMass, kernelMatch, recallMatch, fullMatch,
        hsVal: a.hsValue, hsFloat: a.hsFloat, hsMass: a.hsMass, hsMatch, hsFullMatch, hsBeta: a.hsBeta,
        hsExp: hs32.expCalls, hsr32: hsr32, hsr64: hsr64, r32: r32, r64: r64, info32, info64, ceil: a.ceil, exactMax: a.exactMax });
      console.log(padL(d, 5) + padL(bits, 5) + padL(dEff, 6) + padL(fx(a.smValue, 3), 7) + padL(fx(a.smMass, 3), 8) +
        padL(fx(a.sqValue, 3), 7) + padL(fx(a.hamFloat, 3), 10) + padL(fx(a.hamValue, 3), 8) + padL(fx(a.hamMass, 3), 9) +
        padL(fx(a.hamMaxMass, 3), 8) + padL(kernelMatch ? 'YES' : 'no', 8) + padL(recallMatch ? 'YES' : 'no', 8) +
        padL(fullMatch ? 'YES' : 'no', 6) + padL(fx(a.ceil, 3), 7) + padL(fx(a.exactMax, 3), 10) +
        padL(fx(r32, 3) + 'x', 10) + padL(fx(r64, 3) + 'x', 10));
    }
  }
  console.log('');
  console.log('7b) SOFTMAX normalization over the exact integer Hamming scores (exp = precomputed table lookup)');
  console.log('     expCalls = sMax-sMin+1 per BATCH (table built once, reused across all M rows); hsFloat = kernel x real V, hsVal = kernel x quantised V.');
  console.log('    d  bits   dEff   smVal  smMass  hsFloat  hsVal  hsMass  hsBeta  kernel  full  expCalls  hsRatioW32  hsRatioW64');
  for (const r of table) {
    console.log(padL(r.d, 5) + padL(r.bits, 5) + padL(r.dEff, 6) + padL(fx(r.smVal, 3), 7) + padL(fx(r.smMass, 3), 8) +
      padL(fx(r.hsFloat, 3), 10) + padL(fx(r.hsVal, 3), 7) + padL(fx(r.hsMass, 3), 8) + padL(fx(r.hsBeta, 1), 8) +
      padL(r.hsMatch ? 'YES' : 'no', 8) + padL(r.hsFullMatch ? 'YES' : 'no', 6) + padL(r.hsExp, 10) +
      padL(fx(r.hsr32, 3) + 'x', 12) + padL(fx(r.hsr64, 3) + 'x', 12));
  }
  console.log('');
  const km = table.filter(function (r) { return r.kernelMatch; }).sort(function (a, b) { return a.dEff - b.dEff; });
  const rm = table.filter(function (r) { return r.recallMatch; }).sort(function (a, b) { return a.dEff - b.dEff; });
  const fm = table.filter(function (r) { return r.fullMatch; }).sort(function (a, b) { return a.dEff - b.dEff; });
  if (km.length) {
    const m = km[0];
    console.log('FIRST KERNEL (mass+recall) MATCH: d=' + m.d + ' bits=' + m.bits + ' dEff=' + m.dEff +
      '  floatSoftmaxRatio W32=' + fx(m.r32, 3) + 'x W64=' + fx(m.r64, 3) + 'x' +
      '  matchedBitsScoreRatio W32=' + fx(m.info32, 2) + 'x W64=' + fx(m.info64, 2) + 'x');
    for (const b of bitsSet) {
      const mm = km.filter(function (r) { return r.bits === b; })[0];
      if (mm) console.log('  bits=' + b + ': first match d=' + mm.d + ' dEff=' + mm.dEff + ' ratioW32=' + fx(mm.r32, 3) + 'x ratioW64=' + fx(mm.r64, 3) + 'x');
    }
  } else console.log('NO KERNEL MATCH up to d=' + dims[dims.length - 1]);
  if (rm.length) {
    const m = rm[0];
    console.log('FIRST RECALL-ONLY MATCH (kernel): d=' + m.d + ' bits=' + m.bits + ' dEff=' + m.dEff +
      '  floatSoftmaxRatio W32=' + fx(m.r32, 3) + 'x W64=' + fx(m.r64, 3) + 'x');
  }
  if (fm.length) console.log('FIRST FULL-METHOD MATCH (quantised V): d=' + fm[0].d + ' bits=' + fm[0].bits + ' dEff=' + fm[0].dEff + ' ratioW32=' + fx(fm[0].r32, 3) + 'x');
  else console.log('NO FULL-METHOD MATCH up to d=' + dims[dims.length - 1]);
  for (const b of bitsSet) {
    const mm = fm.filter(function (r) { return r.bits === b; })[0];
    if (mm) console.log('  full method bits=' + b + ': first match d=' + mm.d + ' dEff=' + mm.dEff + ' ratioW32=' + fx(mm.r32, 3) + 'x ratioW64=' + fx(mm.r64, 3) + 'x');
    else console.log('  full method bits=' + b + ': never matches up to d=' + dims[dims.length - 1] + ' (weaker bits lose too much value precision / top-weight sharpness)');
  }
  const hm2 = table.filter(function (r) { return r.hsMatch; }).sort(function (a, b) { return a.dEff - b.dEff; });
  if (hm2.length) {
    const m = hm2[0];
    console.log('FIRST SOFTMAX-NORMALIZED HAMMING MATCH: d=' + m.d + ' bits=' + m.bits + ' dEff=' + m.dEff +
      '  floatSoftmaxRatio W32=' + fx(m.hsr32, 3) + 'x W64=' + fx(m.hsr64, 3) + 'x  expCalls/batch=' + m.hsExp);
    for (const b of bitsSet) {
      const mm = hm2.filter(function (r) { return r.bits === b; })[0];
      if (mm) console.log('  bits=' + b + ': first softmax-normalized match d=' + mm.d + ' dEff=' + mm.dEff + ' ratioW32=' + fx(mm.hsr32, 3) + 'x ratioW64=' + fx(mm.hsr64, 3) + 'x');
    }
  } else console.log('NO SOFTMAX-NORMALIZED HAMMING MATCH up to d=' + dims[dims.length - 1]);
  const hf2 = table.filter(function (r) { return r.hsFullMatch; }).sort(function (a, b) { return a.dEff - b.dEff; });
  for (const b of bitsSet) {
    const mm = hf2.filter(function (r) { return r.bits === b; })[0];
    if (mm) console.log('  softmax-normalized FULL method (quantised V) bits=' + b + ': first match d=' + mm.d + ' dEff=' + mm.dEff + ' ratioW32=' + fx(mm.hsr32, 3) + 'x ratioW64=' + fx(mm.hsr64, 3) + 'x');
    else console.log('  softmax-normalized FULL method (quantised V) bits=' + b + ': never matches up to d=' + dims[dims.length - 1] + ' (binary value quantisation caps recall at ~0.75)');
  }
  console.log('');
  console.log('MATCHED-QUALITY COST BREAKDOWN (first softmax-normalized match per bit-width, W=32):');
  console.log('  bits   d   dEff  | float softmax (scoreFMA/exp/valueFMA/total)  | affine Hamming (bitops/value/total, quality FAILS)  | softmax-table Hamming (bitops/reads/expCalls/value/total, ratio)');
  for (const b of bitsSet) {
    const mm = hm2.filter(function (r) { return r.bits === b; })[0];
    if (!mm) continue;
    const kd = b === 1 ? 'binary' : 'bit';
    const hA = hammingCost(32, 64, mm.dEff, 8, 32, kd);
    const hS = hammingSoftmaxCost(32, 64, mm.dEff, 8, 32, kd, mm.dEff, mm.dEff + 1);
    const smc = softmaxCost(32, 64, mm.d, 8);
    console.log('  b=' + b + '  d=' + padL(mm.d, 4) + ' dEff=' + padL(mm.dEff, 5) +
      ' | ' + padL(smc.scoreFma, 7) + '/' + padL(smc.exp, 5) + '/' + padL(smc.valueFma, 6) + '/' + padL(smc.total, 6) +
      ' | ' + padL(hA.bitops, 6) + '/' + padL(hA.imad, 6) + '/' + padL(hA.total, 6) + ' (' + fx(smc.total / hA.total, 2) + 'x)' +
      ' | ' + padL(hS.bitops, 6) + '/' + padL(hS.tableReads, 5) + '/' + padL(hS.expCalls, 5) + '/' + padL(hS.imad, 6) + '/' + padL(hS.total, 6) + ' (' + fx(smc.total / hS.total, 2) + 'x)');
  }
  console.log('');
  console.log('Matched-INFORMATION view: the float baseline on the SAME dEff dequantised coordinates costs');
  console.log('M*N*dEff score FMAs; hamming costs M*N*ceil(dEff/W) bit-ops, so the score term is ~W x cheaper');
  console.log('(W=32 -> ~32x, W=64 -> ~64x); the shared M*N*dv value term dilutes the total.');
}

// ---------------------------------------------------------------------------
// 8. JS micro-benchmark (non-representative)
// ---------------------------------------------------------------------------
function benchSection() {
  section('8) JS MICRO-BENCHMARK (NON-REPRESENTATIVE OF NATIVE HARDWARE)');
  const d = 64, iters = 300000;
  const a = new Array(d), b = new Array(d);
  for (let f = 0; f < d; f++) { a[f] = Math.sin(f) * 0.3; b[f] = Math.cos(f * 1.7) * 0.3; }
  const wa = packBitsFast(a.map(function (x) { return x >= 0 ? 1 : 0; }));
  const wb = packBitsFast(b.map(function (x) { return x >= 0 ? 1 : 0; }));
  let acc = 0;
  let t0 = Date.now();
  for (let it = 0; it < iters; it++) { let s = 0; for (let f = 0; f < d; f++) s += a[f] * b[f]; acc += s; }
  const tFloat = (Date.now() - t0) / iters;
  t0 = Date.now();
  for (let it = 0; it < iters; it++) acc += popcountXor(wa, wb);
  const tBit = (Date.now() - t0) / iters;
  console.log('d=64: float dot ~' + fx(tFloat * 1e6, 1) + ' ns/pair; JS popcount(xor) ~' + fx(tBit * 1e6, 1) + ' ns/pair; ratio ' + fx(tFloat / Math.max(tBit, 1e-9), 2) + 'x');
  console.log('acc=' + fx(acc, 3) + ' (sink)');
  console.log('V8 has no native popcount, the loop carries call/bounds overhead, and the ratio is unstable');
  console.log('across runs (observed 4.20x and 0.20x for the same code) -- it is NOT the native bit-op/FMA');
  console.log('ratio.  The section-2 primitive count is the representative model; this block only shows that');
  console.log('JS cannot express the primitive.');
}

// ---------------------------------------------------------------------------
// 9. verdict
// ---------------------------------------------------------------------------
function verdictSection() {
  section('9) VERDICT');
  return [
    '  IDENTITY.  For sign vectors z in {-1,+1}^d the Hamming rejection is v = (d - z_q.z_k)/2, so the',
    '  Hamming score d - v = (d + z_q.z_k)/2 is an AFFINE function of the signed dot product (section 1).',
    '  popcount therefore removes the QK^T score matmul while preserving the score up to an affine',
    '  (temperature) transform; concentration is decided by the NORMALIZATION, not by the score.',
    '  [The literal {0,1} form popcount(xor) = d - q.k is false; it is counted in section 1.]',
    '',
    '  (a) BULEYEAN / AFFINE normalization  w = R - min(v,R) + 1.  It is affine in the rejection',
    '      count, so it inherits the affine lane ceiling: the best target mass over any budget R is',
    '      (delta+1)/(delta+N) with delta the Hamming margin.  At d=8 the margin is under one bit and',
    '      the kernel-only recall stays at chance (0.023 vs softmax 0.852); the ceiling is 0.044.',
    '      CONFIRMED numerically at every tested bit-width (section 4-5).  This normalization FAILS',
    '      to retrieve at any usable budget.',
    '',
    '  (b) SOFTMAX normalization over the exact integer Hamming scores.  s = dEff - popcount(xor) is an',
    '      integer with only dEff+1 values, so exp(beta*s) is a PRECOMPUTED TABLE of dEff+1 entries --',
    '      the transcendental becomes one memory read per pair and the score matmul becomes popcount.',
    '      Because the normalization is the same exponential family, concentration is retained, and',
    '      this variant DOES match float softmax at matched quality:',
    '        - 4-bit: first match at dEff=32 bits (d=8) with a 1.5x W32 total ratio; the full method',
    '          (quantised V) also matches there (hsVal 0.927 vs smVal 0.849).',
    '        - binary: first kernel match at dEff=32 bits (d=32) with 3.5x; the FULL binary method',
    '          never matches up to d=2048 because 1-bit value quantisation caps recall near 0.75.',
    '        - the ratio grows with the ambient dimension (value-term dilution shrinks): binary at',
    '          d=256 is 14.0x W32 / 17.7x W64, at d=1024 is 24.0x / 38.3x, approaching W=32 / 2W=64.',
    '',
    '  HONEST BOUNDARY.  Popcount removes the QK^T matmul only.  The value aggregation A.V is still an',
    '  N x N (here M x N) by d_v dense product and is NOT removed; it is the shared M*N*d_v term in',
    '  every cost total above.  Removing it needs a linear-attention factorization, which the sibling',
    '  kernel experiment showed costs quality (m >> N for sharp softmax).  So this is a score-matmul',
    '  and transcendental win at matched quality, NOT a fully matmul-free attention.',
    '',
    '  HEADLINE.  PARTIAL, and routed entirely by the normalization.  Buleyean/affine normalization',
    '  NEVER matches (O(1/N) ceiling).  Softmax normalization over the same integer Hamming scores',
    '  matches at every bit-width, first at dEff~32 bits, with a total ratio that is small there',
    '  (1.5x) and grows toward W as d grows (binary: 14x at d=256, 24x at d=1024 at W=32; roughly',
    '  2x that at W=64).  Required bits: 4-bit keys/values support the full method from d=8; binary',
    '  keys need d>~32 and binary VALUES never match.  Popcount does NOT remove the AV matmul.',
  ].join('\n');
}

// ---------------------------------------------------------------------------
// optional task-parity check against the sibling generator
// ---------------------------------------------------------------------------
async function parityCheck() {
  try {
    const sib = await import('../softmax-vs-buleyean-attention/attention.mjs');
    let bad = 0, checked = 0;
    for (const seed of SEEDS) {
      const cfg = { M: 32, N: 64, d: 8, dv: 8, seed, keyNoise: 0.1 };
      const mine = makeRecallTask(cfg), theirs = sib.makeRecallTask(cfg);
      checked++;
      for (let i = 0; i < mine.Q.length; i++) for (let f = 0; f < mine.d; f++) if (mine.Q[i][f] !== theirs.Q[i][f]) bad++;
      for (let j = 0; j < mine.K.length; j++) for (let f = 0; f < mine.d; f++) if (mine.K[j][f] !== theirs.K[j][f]) bad++;
      for (let j = 0; j < mine.V.length; j++) for (let f = 0; f < mine.dv; f++) if (mine.V[j][f] !== theirs.V[j][f]) bad++;
    }
    return 'task parity vs ../softmax-vs-buleyean-attention/attention.mjs: checked=' + checked + ' seeds, element mismatches=' + bad;
  } catch (e) {
    return 'task parity check skipped (sibling module unavailable): ' + e.message;
  }
}

async function main() {
  const t0 = Date.now();
  console.log(line('=', 74));
  console.log('Buleyean Hamming / rejection-count attention vs float softmax attention');
  console.log('node ' + process.version);
  console.log(line('=', 74));
  console.log(await parityCheck());
  scoreDemo();
  costSection();
  exactnessSection();
  qualitySection();
  ceilingSection();
  nSection();
  crossoverSection();
  benchSection();
  console.log('');
  console.log(line('-', 74));
  console.log(verdictSection());
  console.log('');
  console.log(line('-', 74));
  console.log('done.  (' + ((Date.now() - t0) / 1000).toFixed(1) + ' s)');
}

main().catch(function (e) { console.error(e && e.stack ? e.stack : e); process.exit(1); });
