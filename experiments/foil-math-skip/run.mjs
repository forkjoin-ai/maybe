#!/usr/bin/env node
// FOIL math skip: precompute common math expressions into sealed tables and
// measure how much recomputation the cache-hit path removes under the tau
// miss-not-lie admission. Dependency-free, deterministic.
//
// The claim under test: if a workload concentrates on a bounded integer key
// domain, a sealed table turns a transcendental call into one memory read, and
// a SOUND admission never serves a value outside its declared error bound. The
// honest question is the served rate and the zero-lie speedup, not a headline.

const N = 2_000_000;
const MARGIN = 0.5;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 0x100000000;
  };
}

// Common math expressions on a bounded integer key domain (0..domain).
const EXPRESSIONS = [
  { name: 'exp', f: Math.exp, domain: 2048, step: 1e-3 },
  { name: 'sin', f: Math.sin, domain: 2048, step: 1e-3 },
  { name: 'log1p', f: Math.log1p, domain: 2048, step: 1e-3 },
  { name: 'sqrt', f: Math.sqrt, domain: 4096, step: 1e-3 },
  { name: 'sigmoid', f: (x) => 1 / (1 + Math.exp(-x)), domain: 1024, step: 1e-3 },
  { name: 'tanh', f: Math.tanh, domain: 1024, step: 1e-3 },
];

// A concentrated workload: 70% of keys fall in the hot decile near the mode,
// 30% spread uniformly. This is the common-expression regime the FOIL thesis
// is about: a few shapes, hit again and again.
function workload(domain, seed) {
  const r = rng(seed);
  const hot = Math.max(1, Math.floor(domain / 10));
  const keys = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    keys[i] = r() < 0.7 ? Math.floor(r() * hot) : Math.floor(r() * (domain + 1));
  }
  return keys;
}

function timePerCall(fn, keys, reps) {
  // Warm up.
  let acc = 0;
  for (let i = 0; i < 100_000; i++) acc += fn(keys[i % keys.length]);
  const start = process.hrtime.bigint();
  for (let r = 0; r < reps; r++) {
    for (let i = 0; i < keys.length; i++) acc += fn(keys[i]);
  }
  const ns = Number(process.hrtime.bigint() - start);
  return { nsPerCall: ns / (reps * keys.length), acc };
}

function timeArrayRead(table, keys, reps) {
  let acc = 0;
  for (let i = 0; i < 100_000; i++) acc += table[keys[i % keys.length]];
  const start = process.hrtime.bigint();
  for (let r = 0; r < reps; r++) {
    for (let i = 0; i < keys.length; i++) acc += table[keys[i]];
  }
  const ns = Number(process.hrtime.bigint() - start);
  return { nsPerRead: ns / (reps * keys.length), acc };
}

function report(expr, seed) {
  const { name, f, domain, step } = expr;
  const exact = new Float64Array(domain + 1);
  const quant = new Float64Array(domain + 1);
  for (let k = 0; k <= domain; k++) {
    const v = f(k);
    exact[k] = v;
    quant[k] = Math.round(v / step) * step;
  }
  const tau = step / 2;
  const keys = workload(domain, seed);

  const recompute = timePerCall(f, keys, 2);
  const read = timeArrayRead(exact, keys, 2);

  // EXACT tier: always serve, bit-for-bit.
  let served = 0, misses = 0, lies = 0, maxErr = 0;
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    served++;
    const servedValue = exact[k];
    if (servedValue !== f(k)) lies++;
    maxErr = Math.max(maxErr, Math.abs(servedValue - f(k)));
  }
  const exactSpeedup = recompute.nsPerCall / read.nsPerRead;

  // SOUND approximation tier with a PER-ENTRY bound: bound(k) = tau * max(1, |f(k)|)
  // dominates the absolute rounding error everywhere, so serving is never a lie.
  // A single global tau is false for exp over this domain (values overflow), and
  // the misses below are the honest cost of that correction.
  let soundServed = 0, soundMiss = 0, soundLies = 0;
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const bound = tau * Math.max(1, Math.abs(f(k)));
    if (2 * bound <= MARGIN) {
      soundServed++;
      if (Math.abs(quant[k] - f(k)) > bound + 1e-9 * Math.max(1, Math.abs(f(k)))) soundLies++;
    } else {
      soundMiss++;
    }
  }
  const soundMissRate = soundMiss / (soundServed + soundMiss);
  const soundSpeedup = recompute.nsPerCall / (read.nsPerRead + soundMissRate * recompute.nsPerCall);

  return {
    name,
    domain,
    tau,
    perEntry: true,
    recomputeNs: recompute.nsPerCall,
    lookupNs: read.nsPerRead,
    exactSpeedup,
    exactLies: lies,
    served,
    misses,
    soundServed,
    soundMiss,
    soundMissRate,
    soundSpeedup,
    soundLies,
  };
}

const seeds = [0x5eed, 0xbeef, 0xfeed];
const rows = [];
for (const expr of EXPRESSIONS) {
  for (const seed of seeds) rows.push(report(expr, seed));
}
const byName = new Map();
for (const row of rows) {
  const cur = byName.get(row.name) ?? { name: row.name, n: 0, exactSpeedup: 0, soundSpeedup: 0, soundMissRate: 0, soundLies: 0, exactLies: 0, recomputeNs: 0, lookupNs: 0, tau: row.tau, perEntry: row.perEntry };
  cur.n++;
  cur.exactSpeedup += row.exactSpeedup;
  cur.soundSpeedup += row.soundSpeedup;
  cur.soundMissRate += row.soundMissRate;
  cur.soundLies += row.soundLies;
  cur.exactLies += row.exactLies;
  cur.recomputeNs += row.recomputeNs;
  cur.lookupNs += row.lookupNs;
  byName.set(row.name, cur);
}
const summary = [...byName.values()].map((r) => ({
  name: r.name,
  tau: r.tau,
  perEntry: r.perEntry,
  recomputeNs: r.recomputeNs / r.n,
  lookupNs: r.lookupNs / r.n,
  exactSpeedup: r.exactSpeedup / r.n,
  soundSpeedup: r.soundSpeedup / r.n,
  soundMissRate: r.soundMissRate / r.n,
  exactLies: r.exactLies,
  soundLies: r.soundLies,
}));

if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ n: N, margin: MARGIN, summary }, null, 2));
} else {
  console.log('FOIL math skip | N =', N.toLocaleString(), 'keys/expression | margin =', MARGIN);
  console.log('');
  console.log('expr      tau     recompute   lookup    exact x   sound x   miss%   lies');
  for (const r of summary) {
    console.log(
      r.name.padEnd(9) +
      r.tau.toExponential(0).padEnd(8) +
      (r.recomputeNs.toFixed(1) + ' ns').padEnd(12) +
      (r.lookupNs.toFixed(2) + ' ns').padEnd(10) +
      (r.exactSpeedup.toFixed(2) + 'x').padEnd(10) +
      (r.soundSpeedup.toFixed(2) + 'x').padEnd(10) +
      (100 * r.soundMissRate).toFixed(1).padEnd(8) +
      (r.exactLies + r.soundLies)
    );
  }
  const minExact = Math.min(...summary.map((r) => r.exactSpeedup));
  const minSound = Math.min(...summary.map((r) => r.soundSpeedup));
  const totalLies = summary.reduce((s, r) => s + r.exactLies + r.soundLies, 0);
  console.log('');
  console.log('exact speedup range:', minExact.toFixed(2) + 'x .. ' + Math.max(...summary.map((r) => r.exactSpeedup)).toFixed(2) + 'x');
  console.log('zero-lie (SOUND) speedup range:', minSound.toFixed(2) + 'x .. ' + Math.max(...summary.map((r) => r.soundSpeedup)).toFixed(2) + 'x');
  console.log('SERVED VALUES OUTSIDE DECLARED BOUND (lies):', totalLies);
}
