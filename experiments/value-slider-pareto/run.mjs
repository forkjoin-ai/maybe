/**
 * run.mjs -- runnable entry for the value-side quality/speed slider experiment.
 *
 *   /opt/homebrew/bin/node run.mjs
 *
 * Dependency-free (no imports except the sibling core module, no Node builtins).
 * Prints the full report; README.md embeds this exact output.
 */

import {
  makeTask, softmaxWeights, exactValue, aggregateChunked, aggregateTopK,
  aggregateLinear, orthonormalProjection, outputQError, recallValues, meanKL,
  meanTargetMass, paretoFrontier, kneeElbow, kneeNear, chunkedWeights,
  topkWeights, topkKeptSet, chunkedKeptSet, admissionSweep, integerBlockExactness,
  integerTopKSelectExactness, integerTopKSumExactness,
} from './value-slider.mjs';

const D = 64;
const DV = 32;
const M = 128;
const BETA = 8;
const NS = [64, 128, 256, 512];
const SEED = 0x5eed;

const VARIANTS = [
  { name: 'assoc', noise: 0.05, nNeedle: 0, decoys: 0, decoyNoise: 0 },
  { name: 'needle', noise: 0.15, nNeedle: 4, decoys: 5, decoyNoise: 0.16 },
];

function line() {
  console.log('--------------------------------------------------------------------------');
}
function pad(s, n) {
  return String(s).padStart(n);
}
function f(x, n) {
  return Number(x).toFixed(n === undefined ? 4 : n);
}
function pct(x) {
  return (100 * x).toFixed(2) + '%';
}
function powers(n) {
  const out = [];
  for (let b = 1; b <= n; b *= 2) out.push(b);
  return out;
}
function ranksOf(d) {
  const out = [];
  for (let r = 1; r <= d; r *= 2) out.push(r);
  if (out[out.length - 1] !== d) out.push(d);
  return out;
}

/**
 * Evaluate every regime on one task.  All regimes see the SAME exact float
 * softmax weights A; only the value aggregate changes.
 */
function evaluate(variantCfg, N) {
  const task = makeTask({
    variant: variantCfg.name, N, d: D, dv: DV, M, beta: BETA,
    seed: (SEED ^ (N * 2654435761)) >>> 0,
    noise: variantCfg.noise, nNeedle: variantCfg.nNeedle,
    decoys: variantCfg.decoys, decoyNoise: variantCfg.decoyNoise,
  });
  const A = softmaxWeights(task.queries, task.keys, BETA);
  const exactOut = exactValue(A, task.values);
  const exactRecall = recallValues(task.values, exactOut, task.targets);
  const exactMass = meanTargetMass(A, task.targets);
  const exactCost = M * N * DV;

  const points = [];
  const push = (label, mode, param, res) => {
    const kl = meanKL(A, res.weights);
    const qerr = outputQError(res.output, exactOut);
    const recall = recallValues(task.values, res.output, task.targets);
    const mass = meanTargetMass(res.weights, task.targets);
    const p = {
      label, mode, param, cost: res.cost, mac: res.mac, select: res.select || 0,
      kl, qerr, recall, mass, quality: 1 / (1 + kl), qualityQ: 1 / (1 + qerr),
    };
    points.push(p);
    return p;
  };

  push('EXACT B=N', 'EXACT', N, {
    output: exactOut, weights: A, cost: exactCost, mac: exactCost, select: 0,
  });

  const bs = powers(N);
  for (const B of bs) {
    if (B >= N) continue;
    push('CHUNKED B=' + B, 'CHUNKED', B, aggregateChunked(A, task.values, B));
  }
  for (const k of powers(N)) {
    push('TOPK k=' + k, 'TOPK', k, aggregateTopK(A, task.values, k));
  }
  for (const r of ranksOf(D)) {
    const cols = orthonormalProjection(D, r, 0xabc + r);
    push('LINEAR r=' + r, 'LINEAR', r, aggregateLinear(task.keys, task.values, task.queries, cols));
  }

  const byMode = {};
  for (const p of points) {
    if (!byMode[p.mode] || p.cost < byMode[p.mode].cost) byMode[p.mode] = p;
  }

  const frontier = paretoFrontier(points);
  const frontierQ = paretoFrontier(points.map((p) => Object.assign({}, p, { quality: p.qualityQ })));
  const exactPoint = points[0];
  const linearPts = points.filter((p) => p.mode === 'LINEAR').sort((a, b) => a.cost - b.cost);
  const linearFull = linearPts[linearPts.length - 1];

  return {
    task, A, variant: variantCfg.name, N, exactOut, exactRecall, exactMass,
    exactCost, points, byMode, frontier, frontierQ, exactPoint, linearFull,
  };
}

function row(ev, p) {
  console.log(
    '  ' + p.label.padEnd(18) +
    pad(p.cost, 10) + pad(pct(p.cost / ev.exactCost), 8) +
    pad(p.mac, 10) + pad(f(p.kl), 8) + pad(f(p.qerr), 8) +
    pad(f(p.recall, 3), 8) + pad(f(p.mass), 8),
  );
}

function printTask(ev) {
  console.log('');
  line();
  console.log('[' + ev.variant + ' N=' + ev.N + ']  d=' + D + ' dv=' + DV + ' M=' + M + ' beta=' + BETA);
  console.log('  exact endpoint : cost=' + ev.exactCost + ' mac=' + ev.exactCost +
    ' recall=' + f(ev.exactRecall, 3) + ' targetMass=' + f(ev.exactMass, 4));
  console.log('  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))');
  console.log('  ' + 'point'.padEnd(18) + pad('cost', 10) + pad('cost%', 8) + pad('mac', 10) +
    pad('KL', 8) + pad('qerr', 8) + pad('recall', 8) + pad('mass', 8));
  for (const p of ev.frontier) row(ev, p);

  const cheapQ = ev.points.filter((p) => p.mode !== 'EXACT' && p.cost < ev.exactCost && p.qerr > 1e-9).reduce((a, b) => Math.min(a, b.qerr), Infinity);
  const kneeTiny = kneeNear(ev.frontierQ, (p) => p.qerr, 0.02);
  const kneePlateau = kneeNear(ev.frontierQ, (p) => p.qerr, Math.max(cheapQ * 1.1, 1e-9));
  const elbowKL = kneeElbow(ev.frontier, (p) => p.kl);
  const elbowQ = kneeElbow(ev.frontierQ, (p) => p.qerr);
  const kdesc = (p) => (p ? p.label + ' cost=' + p.cost + ' (' + pct(p.cost / ev.exactCost) +
    ', removed ' + pct(1 - p.cost / ev.exactCost) + ') KL=' + f(p.kl) + ' qerr=' + f(p.qerr) +
    ' recall=' + f(p.recall, 3) : 'none');
  console.log('  KNEE tiny (qerr<=0.02) : ' + kdesc(kneeTiny));
  console.log('  KNEE plateau (qerr<=1.10x best cheap ' + f(cheapQ) + '): ' + kdesc(kneePlateau));
  console.log('  ELBOW (qerr)           : ' + kdesc(elbowQ));
  console.log('  ELBOW (KL)             : ' + kdesc(elbowKL));

  console.log('  CHEAPEST PER MODE + endpoints');
  console.log('  ' + 'point'.padEnd(18) + pad('cost', 10) + pad('cost%', 8) + pad('mac', 10) +
    pad('KL', 8) + pad('qerr', 8) + pad('recall', 8) + pad('mass', 8));
  const modes = ['EXACT', 'CHUNKED', 'TOPK', 'LINEAR'];
  for (const m of modes) {
    const p = ev.byMode[m];
    if (p) row(ev, p);
  }
  const lf = ev.linearFull;
  if (lf && lf !== ev.byMode.LINEAR) row(ev, lf);
}

function printSummary(evs) {
  console.log('');
  line();
  console.log('CROSS-N / CROSS-VARIANT KNEE SUMMARY');
  console.log('  ' + 'variant'.padEnd(9) + pad('N', 5) + pad('exactCost', 11) +
    pad('kneeTiny', 10) + pad('cost%', 8) + pad('qerr', 8) +
    pad('kneePlateau', 13) + pad('cost%', 8) + pad('qerr', 8) +
    pad('lin%', 8) + pad('linRecall', 10));
  for (const ev of evs) {
    const cheapQ = ev.points.filter((p) => p.mode !== 'EXACT' && p.cost < ev.exactCost && p.qerr > 1e-9).reduce((a, b) => Math.min(a, b.qerr), Infinity);
    const tiny = kneeNear(ev.frontierQ, (p) => p.qerr, 0.02);
    const plat = kneeNear(ev.frontierQ, (p) => p.qerr, Math.max(cheapQ * 1.1, 1e-9));
    const lbl = (p) => (p ? p.label.replace('CHUNKED B=', 'B=') : 'none');
    console.log('  ' + ev.variant.padEnd(9) + pad(ev.N, 5) + pad(ev.exactCost, 11) +
      pad(lbl(tiny), 10) + pad(tiny ? pct(tiny.cost / ev.exactCost) : '-', 8) +
      pad(tiny ? f(tiny.qerr) : '-', 8) +
      pad(lbl(plat), 13) + pad(plat ? pct(plat.cost / ev.exactCost) : '-', 8) +
      pad(plat ? f(plat.qerr) : '-', 8) +
      pad(pct(ev.linearFull.cost / ev.exactCost), 8) +
      pad(f(ev.linearFull.recall, 3), 10));
  }
  console.log('  (kneeTiny: cheapest frontier point with qerr <= 0.02; kneePlateau: cheapest within 1.10x');
  console.log('   of the best cheap qerr; "none" = the task admits no such point)');
}

// ===========================================================================
// TAU miss-not-lie admission over the value slider
// ===========================================================================

const BAND_TOL = 0.05;
const TAUS = [0, 0.005, 0.01, 0.02, 0.05];

function candidatesFor(ev) {
  const A = ev.A;
  const V = ev.task.values;
  const out = [];
  for (const B of [1, 4, 16]) {
    if (B >= ev.N) continue;
    const c = aggregateChunked(A, V, B);
    out.push({
      label: 'CHUNKED B=' + B,
      kept: (row) => chunkedKeptSet(row, B),
      cheapCost: c.cost,
      witnessWeights: topkWeights(A, 4),
      witnessCost: aggregateTopK(A, V, 4).cost,
    });
  }
  for (const k of [1, 4, 16]) {
    if (k > ev.N) continue;
    const c = aggregateTopK(A, V, k);
    out.push({
      label: 'TOPK k=' + k,
      kept: (row) => topkKeptSet(row, k),
      cheapCost: c.cost,
      witnessWeights: chunkedWeights(A, 1),
      witnessCost: aggregateChunked(A, V, 1).cost,
    });
  }
  return out;
}

function runAdmission(ev, cand) {
  return admissionSweep(ev.A, ev.task.targets, {
    exactCost: ev.exactCost,
    cheapCost: cand.cheapCost,
    witnessWeights: cand.witnessWeights,
    witnessCost: cand.witnessCost,
    bandTol: BAND_TOL,
    taus: TAUS,
    makeKept: cand.kept,
  });
}

function bestZeroLie(points) {
  let best = null;
  for (const p of points) {
    if (p.lieRate <= 1e-9 && (best === null || p.effectiveCost < best.effectiveCost)) best = p;
  }
  return best;
}

function printAdmission(evs) {
  console.log('');
  line();
  console.log('TAU MISS-NOT-LIE ADMISSION (bandTol=' + BAND_TOL + ' on the served target mass)');
  console.log('  SOUND  = sound kept/dropped bracket, deployable, zero lies by construction.');
  console.log('  PROXY  = tail mass trusted to +/-0.02 inside a witness-anchored +/-0.2 band, can lie.');
  console.log('  ORACLE = cheap-vs-exact agreement, zero-lie ceiling, NOT deployable.');
  console.log('  ' + 'task'.padEnd(14) + 'candidate'.padEnd(14) + 'policy'.padEnd(8) +
    pad('tau', 7) + pad('admit', 7) + pad('miss', 7) + pad('lie', 7) +
    pad('effCost', 10) + pad('speedup', 9) + pad('maxLie', 8));
  let proverAllZeroSound = true;
  let globalMaxProxyLie = 0;
  const taskBest = [];
  for (const ev of evs) {
    const cands = candidatesFor(ev);
    let gBest = null;
    for (const cand of cands) {
      const pts = runAdmission(ev, cand);
      let maxProxyLie = 0;
      for (const p of pts) if (p.policy === 'PROXY' && p.lieRate > maxProxyLie) maxProxyLie = p.lieRate;
      if (maxProxyLie > globalMaxProxyLie) globalMaxProxyLie = maxProxyLie;
      for (const policy of ['SOUND', 'PROXY', 'ORACLE']) {
        const sub = pts.filter((p) => p.policy === policy);
        const best = bestZeroLie(sub);
        if (policy === 'SOUND' && sub.some((p) => p.lieRate > 1e-9)) proverAllZeroSound = false;
        if (best && (gBest === null || best.effectiveCost < gBest.effectiveCost)) gBest = best;
        console.log('  ' + (ev.variant + ' N=' + ev.N).padEnd(14) + cand.label.padEnd(14) +
          policy.padEnd(8) + pad(best ? f(best.tau, 3) : '-', 7) +
          pad(best ? f(best.admitRate, 3) : '-', 7) + pad(best ? f(best.missRate, 3) : '-', 7) +
          pad(best ? f(best.lieRate, 3) : '-', 7) +
          pad(best ? Math.round(best.effectiveCost) : '-', 10) +
          pad(best ? f(best.speedup, 2) + 'x' : '-', 9) +
          pad(policy === 'PROXY' ? f(maxProxyLie, 3) : '-', 8));
      }
    }
    taskBest.push({ ev, best: gBest });
  }
  console.log('');
  console.log('BEST ZERO-LIE SPEEDUP PER TASK (any candidate, any policy):');
  for (const t of taskBest) {
    console.log('  ' + (t.ev.variant + ' N=' + t.ev.N).padEnd(14) +
      (t.best ? t.best.policy + ' ' + f(t.best.speedup, 2) + 'x (' +
        'admit=' + f(t.best.admitRate, 3) + ', lie=' + f(t.best.lieRate, 3) +
        ', effCost=' + Math.round(t.best.effectiveCost) + ' vs exact ' + t.ev.exactCost + ')' : 'none'));
  }
  return { proverAllZeroSound, taskBest, maxProxyLie: globalMaxProxyLie };
}

// ===========================================================================
// Integer exactness
// ===========================================================================

function printInteger(evs) {
  console.log('');
  line();
  console.log('INTEGER EXACTNESS (BigInt vs naive Number; magnitudes kept below 2^53)');
  const cats = {
    block: { trials: 0, rows: 0, fields: 0, mismatches: 0 },
    select: { trials: 0, rows: 0, fields: 0, mismatches: 0 },
    sum: { trials: 0, rows: 0, fields: 0, mismatches: 0 },
  };
  const MINT = 32;
  const DVINT = 8;
  for (let vi = 0; vi < evs.length; vi++) {
    const N = evs[vi].N;
    for (const s of [1, 2, 3, 4]) {
      const cfg = {
        N, M: MINT, dv: DVINT, maxWeight: 32, maxValue: 10,
        seed: (0x1234 ^ (N * 2654435761) ^ (s * 40503)) >>> 0,
      };
      const be = integerBlockExactness(Object.assign({}, cfg, { blockSize: 4 }));
      const se = integerTopKSelectExactness(Object.assign({}, cfg, { ks: [1, 4, 16, N] }));
      const te = integerTopKSumExactness(Object.assign({}, cfg, { topK: 4 }));
      cats.block.trials += 1; cats.block.rows += be.rows; cats.block.fields += be.fields; cats.block.mismatches += be.mismatches;
      cats.select.trials += 1; cats.select.rows += se.rows; cats.select.fields += se.fields; cats.select.mismatches += se.mismatches;
      cats.sum.trials += 1; cats.sum.rows += te.rows; cats.sum.fields += te.fields; cats.sum.mismatches += te.mismatches;
    }
  }
  console.log('  ' + 'test'.padEnd(34) + pad('trials', 8) + pad('rows', 9) +
    pad('fields', 10) + pad('mismatches', 12));
  const show = (name, key) => {
    const c = cats[key];
    console.log('  ' + name.padEnd(34) + pad(c.trials, 8) + pad(c.rows, 9) +
      pad(c.fields, 10) + pad(c.mismatches, 12));
  };
  show('chunked integer block sums (B=4, dv=8)', 'block');
  show('topK selection (k in {1,4,16,N})', 'select');
  show('topK k=4 integer weighted sums', 'sum');
  const tot = { trials: 0, rows: 0, fields: 0, mismatches: 0 };
  for (const k of Object.keys(cats)) {
    tot.trials += cats[k].trials; tot.rows += cats[k].rows;
    tot.fields += cats[k].fields; tot.mismatches += cats[k].mismatches;
  }
  console.log('  ' + 'TOTAL'.padEnd(34) + pad(tot.trials, 8) + pad(tot.rows, 9) +
    pad(tot.fields, 10) + pad(tot.mismatches, 12));
  return { cats, tot };
}

// ===========================================================================
// Verdict
// ===========================================================================

function printVerdict(evs, adm, ints) {
  const rng = (arr, get) => {
    const v = arr.map(get);
    return [Math.min.apply(null, v), Math.max.apply(null, v)];
  };
  const b1 = evs.map((ev) => ev.points.find((p) => p.label === 'CHUNKED B=1'));
  const tk1 = evs.map((ev) => ev.points.find((p) => p.label === 'TOPK k=1'));
  const lin = evs.map((ev) => ev.linearFull);
  const rem = rng(b1, (p) => 1 - p.cost / evs[b1.indexOf(p) < 0 ? 0 : 0].exactCost);
  // compute removed fraction directly
  const remFrac = evs.map((ev, i) => 1 - b1[i].cost / ev.exactCost);
  const qerrB1 = b1.map((p) => p.qerr);
  const klB1 = b1.map((p) => p.kl);
  const recB1 = b1.map((p) => p.recall);
  const linRec = lin.map((p) => p.recall);
  const linQerr = lin.map((p) => p.qerr);
  const tk1Qerr = tk1.map((p) => p.qerr);
  const tk1Rec = tk1.map((p) => p.recall);
  const bestSpeed = adm.taskBest.filter((t) => t.best).map((t) => t.best.speedup);
  const bestSpeedMin = bestSpeed.length ? Math.min.apply(null, bestSpeed) : 0;
  const bestSpeedMax = bestSpeed.length ? Math.max.apply(null, bestSpeed) : 0;

  console.log('');
  line();
  console.log('VERDICT');
  console.log('  exact endpoint        : B=N, cost = M*N*dv, recall=' +
    f(rng(evs, (ev) => ev.exactRecall)[0], 3) + '..' + f(rng(evs, (ev) => ev.exactRecall)[1], 3) +
    ', targetMass=' + f(rng(evs, (ev) => ev.exactMass)[0], 3) + '..' + f(rng(evs, (ev) => ev.exactMass)[1], 3));
  console.log('  CHEAP END (LINEAR r=d): recall=' + f(Math.min.apply(null, linRec), 3) + '..' +
    f(Math.max.apply(null, linRec), 3) + ' (loses hard retrieval), qerr=' +
    f(Math.min.apply(null, linQerr), 3) + '..' + f(Math.max.apply(null, linQerr), 3) +
    ', cost=' + f(rng(lin, (p) => p.cost)[0] / 1000, 1) + 'k..' + f(rng(lin, (p) => p.cost)[1] / 1000, 1) + 'k');
  console.log('  KNEE (CHUNKED B=1)    : removes ' + pct(Math.min.apply(null, remFrac)) + '..' +
    pct(Math.max.apply(null, remFrac)) + ' of the A.V cost at qerr=' +
    f(Math.min.apply(null, qerrB1)) + '..' + f(Math.max.apply(null, qerrB1)) +
    ', KL=' + f(Math.min.apply(null, klB1)) + '..' + f(Math.max.apply(null, klB1)) +
    ', recall=' + f(Math.min.apply(null, recB1), 3) + '..' + f(Math.max.apply(null, recB1), 3));
  console.log('  KNEE (TOPK k=1)       : qerr=' + f(Math.min.apply(null, tk1Qerr)) + '..' +
    f(Math.max.apply(null, tk1Qerr)) + ', recall=' + f(Math.min.apply(null, tk1Rec), 3) + '..' +
    f(Math.max.apply(null, tk1Rec), 3) + ' (hard recall kept, distribution destroyed)');
  const macRem = evs.map((ev, i) => 1 - b1[i].mac / ev.exactCost);
  const macSpd = evs.map((ev, i) => ev.exactCost / b1[i].mac);
  console.log('  VALUE MACs (CHUNKED B=1): removes ' + pct(Math.min.apply(null, macRem)) + '..' +
    pct(Math.max.apply(null, macRem)) + ' of the A.V multiply-adds (' +
    f(Math.min.apply(null, macSpd), 1) + 'x..' + f(Math.max.apply(null, macSpd), 1) +
    'x mac-only); the full-op speedup also charges the per-query argmax scan.');
  console.log('  SOUND admission       : zero-lie for every candidate/tau = ' + adm.proverAllZeroSound);
  console.log('  BEST zero-lie speedup : ' + f(bestSpeedMin, 2) + 'x..' + f(bestSpeedMax, 2) +
    'x over exact');
  console.log('  INTEGER EXACTNESS     : ' + ints.tot.fields + ' field comparisons, ' +
    ints.tot.mismatches + ' mismatches');
  console.log('');
  console.log('  Honest boundaries:');
  console.log('   * The CHEAP end is not free quality: LINEAR rank=d loses hard retrieval entirely');
  console.log('     (recall=' + f(Math.max.apply(null, linRec), 3) + ').  The knee is inside CHUNKED/TOPK, not linear attention.');
  console.log('   * CHUNKED B=1 is already at the quality plateau; larger B buys almost no quality');
  console.log('     and costs more, so the knee is the cheapest block.');
  console.log('   * The SOUND bracket guarantees the TARGET MASS is exact (target in the kept');
  console.log('     set), which preserves hard argmax retrieval.  It does NOT make the full value');
  console.log('     vector exact: the mean-tail still moves the soft readout (qerr up to');
  console.log('     ' + f(Math.max.apply(null, qerrB1)) + ').  A soft readout must use a tighter metric or a larger B.');
  console.log('   * Tiny (<=2%) readout loss is NOT available on the diffuse needle task: no cheap');
  console.log('     point reaches qerr <= 0.02 there (the best cheap qerr is ' + f(Math.min.apply(null, evs.filter((e) => e.variant === 'needle').map((e) => e.points.filter((p) => p.mode !== 'EXACT' && p.cost < e.exactCost && p.qerr > 1e-9).reduce((a, b) => Math.min(a, b.qerr), Infinity)))) + ' at ~53% cost).');
  console.log('   * The PROXY is a loose heuristic, not a sound interval: maximum observed lie rate');
  console.log('     ' + f(adm.maxProxyLie, 3) + ' (needle, TOPK k=1).  Where it cannot admit it is slower than SOUND,');
  console.log('     and where it admits aggressively it ships lies.  SOUND is the zero-lie policy that pays.');
}

// ===========================================================================
// Main
// ===========================================================================

const EV = [];
for (const v of VARIANTS) {
  for (const N of NS) EV.push(evaluate(v, N));
}

console.log('==========================================================================');
console.log('VALUE-SIDE QUALITY/SPEED SLIDER -- exact softmax fixed, A.V varied');
console.log('d=' + D + ' dv=' + DV + ' M=' + M + ' beta=' + BETA + ' seeds=deterministic');
console.log('variants: assoc (noise 0.05) and needle (noise 0.15, 4 needles, 5 decoys each)');
console.log('cost unit: one multiply-add OR one comparison; mac = pure value multiply-adds');
console.log('==========================================================================');

for (const ev of EV) printTask(ev);
printSummary(EV);
const adm = printAdmission(EV);
const ints = printInteger(EV);
printVerdict(EV, adm, ints);
