/**
 * kernel-attention.mjs -- dependency-free core for the
 * "does a richer kernel recover softmax concentration at O(N) cost?" experiment.
 *
 * No imports at all (not even Node builtins). Pure ECMAScript + BigInt.
 *
 * The prior experiment (../softmax-vs-buleyean-attention) falsified the affine
 * Buleyean rule w_j = R + s_j + 1: it is exp()-free, exactly integer-checkable
 * and O(N), but it has a hard O(1/N) concentration ceiling. This module tests
 * whether a *richer* kernel -- one whose weight grows faster than a linear ramp
 * in the score -- can match softmax with a feature budget m small compared to N.
 *
 * Three families live here:
 *
 *   1. softmax           w_j = exp(beta * s_j)                      O(N^2) reference
 *   2. polynomial kernel w_j = (shift + s_j)^p                      exact finite
 *                        feature map phi(x) = all monomials of x up to degree p,
 *                        m = C(d + p, p), O(N*m*dv), zero transcendentals.
 *                        The ReLU^p variant max(shift+s_j,0)^p is computed naively
 *                        O(N^2): it has a kink and no finite exact monomial feature
 *                        map, so it is a reference, not an O(N) path.
 *   3. positive random features (Performer/FAVOR+)
 *                        exp(beta*q.k) ~ phi(q).phi(k),
 *                        phi(x)_r = exp(omega_r.(sqrt(beta) x) - beta||x||^2/2)/sqrt(m),
 *                        omega_r ~ N(0,I). Uses m exp() per token, not N per query.
 *                        A homogeneous ReLU random-feature variant
 *                        phi(x)_r = max(omega_r.x,0)/sqrt(m) (arc-cosine kernel,
 *                        exp-free, no temperature knob) is included for contrast.
 *
 * Everything is seeded (mulberry32) and deterministic.
 */

// ---------------------------------------------------------------------------
// Deterministic PRNG (data / feature generation only)
// ---------------------------------------------------------------------------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Cost counter and primitives
// ---------------------------------------------------------------------------

export function makeCounter() {
  return { exp: 0, mul: 0, add: 0, div: 0 };
}

export function totalOps(c) {
  return c.exp + c.mul + c.add + c.div;
}

export function addCounter(a, b) {
  a.exp += b.exp; a.mul += b.mul; a.add += b.add; a.div += b.div;
  return a;
}

/** Dot product with optional multiply/add counting. */
export function dot(a, b, c) {
  let acc = 0;
  for (let t = 0; t < a.length; t++) {
    acc += a[t] * b[t];
    if (c) { c.mul++; c.add++; }
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Associative-recall / induction task  (identical construction to the prior exp)
// ---------------------------------------------------------------------------

export function makeRecallTask(opts) {
  const M = opts.M;
  const N = opts.N;
  const d = opts.d;
  const dv = opts.dv;
  const seed = opts.seed;
  const keyNoise = opts.keyNoise === undefined ? 0 : opts.keyNoise;
  const valueScale = opts.valueScale === undefined ? 1 : opts.valueScale;
  const rng = mulberry32(seed);
  const K = [];
  const V = [];
  for (let j = 0; j < N; j++) {
    const k = new Array(d);
    let nrm = 0;
    for (let f = 0; f < d; f++) { k[f] = gaussian(rng); nrm += k[f] * k[f]; }
    nrm = Math.sqrt(nrm) || 1;
    for (let f = 0; f < d; f++) k[f] /= nrm;
    K.push(k);
    const v = new Array(dv);
    for (let f = 0; f < dv; f++) v[f] = gaussian(rng) * valueScale;
    V.push(v);
  }
  const targets = [];
  const Q = [];
  for (let i = 0; i < M; i++) {
    const t = i % N;
    targets.push(t);
    const q = new Array(d);
    for (let f = 0; f < d; f++) q[f] = K[t][f] + keyNoise * gaussian(rng);
    let nrm = 0;
    for (let f = 0; f < d; f++) nrm += q[f] * q[f];
    nrm = Math.sqrt(nrm) || 1;
    for (let f = 0; f < d; f++) q[f] /= nrm;
    Q.push(q);
  }
  return { Q, K, V, targets, M, N, d, dv };
}

// ---------------------------------------------------------------------------
// Softmax attention O(M*N*d): the reference
// ---------------------------------------------------------------------------

export function softmaxAttention(task, beta, c, opts) {
  const materialize = !opts || opts.materialize !== false;
  const Q = task.Q, K = task.K, V = task.V, M = task.M, N = task.N, dv = task.dv;
  const W = materialize ? new Array(M) : null;
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const s = new Array(N);
    let m = -Infinity;
    for (let j = 0; j < N; j++) {
      const x = dot(Q[i], K[j], c);
      s[j] = x;
      if (x > m) m = x;
    }
    let Z = 0;
    const w = new Array(N);
    for (let j = 0; j < N; j++) {
      const e = Math.exp(beta * (s[j] - m));
      if (c) c.exp++;
      w[j] = e;
      Z += e;
      if (c) c.add++;
    }
    for (let j = 0; j < N; j++) {
      w[j] = w[j] / Z;
      if (c) c.div++;
    }
    const o = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const wj = w[j];
      for (let f = 0; f < dv; f++) {
        o[f] += wj * V[j][f];
        if (c) { c.mul++; c.add++; }
      }
    }
    if (materialize) W[i] = w;
    out[i] = o;
  }
  return { weights: W, outputs: out };
}

// ---------------------------------------------------------------------------
// Polynomial-kernel feature map: all monomials of x up to degree p
//   (shift + q.k)^p = sum_{|alpha|<=p} p!/(alpha! (p-|alpha|)!) shift^(p-|alpha|)
//                     * prod_i (q_i k_i)^{alpha_i}
// so phi_alpha(x) = sqrt(coeff_alpha) prod_i x_i^{alpha_i} gives phi(q).phi(k)
// exactly, with m = C(d+p, p) features.  'shift' is the task's c; shift >= 1
// with unit keys guarantees shift + s >= 0, so all weights are non-negative.
// ---------------------------------------------------------------------------

export function binom(n, k) {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}

export function featureDim(d, p) {
  return binom(d + p, p);
}

function fact(n) {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

/** All exponent vectors alpha with |alpha| <= p in d variables (length C(d+p,p)). */
export function monomialExponents(d, p) {
  const out = [new Array(d).fill(0)];
  function rec(start, remaining, cur) {
    if (remaining === 0) { out.push(cur.slice()); return; }
    for (let i = start; i < d; i++) {
      cur[i]++;
      rec(i, remaining - 1, cur);
      cur[i]--;
    }
  }
  for (let l = 1; l <= p; l++) rec(0, l, new Array(d).fill(0));
  return out;
}

export function polyCoeffNumber(alpha, p, shift) {
  let l = 0;
  let denom = 1;
  for (let i = 0; i < alpha.length; i++) { l += alpha[i]; denom *= fact(alpha[i]); }
  return (fact(p) / denom / fact(p - l)) * Math.pow(shift, p - l);
}

/** phi(x) = sqrt(coeff_alpha) * prod x_i^{alpha_i}; counts |alpha| muls per monomial. */
export function polyFeatures(x, exps, sqrtCoeffs, ctr) {
  const m = exps.length;
  const out = new Array(m);
  for (let a = 0; a < m; a++) {
    const al = exps[a];
    let v = sqrtCoeffs[a];
    for (let i = 0; i < al.length; i++) {
      let e = al[i];
      while (e-- > 0) { v *= x[i]; if (ctr) ctr.mul++; }
    }
    out[a] = v;
  }
  return out;
}

export function polyFeaturePlan(d, p, shift) {
  const exps = monomialExponents(d, p);
  const coeffs = exps.map(function (a) { return polyCoeffNumber(a, p, shift); });
  const sqrtCoeffs = coeffs.map(function (c) { return Math.sqrt(Math.max(c, 0)); });
  return { exps, coeffs, sqrtCoeffs, m: exps.length };
}

/** Naive polynomial attention O(N^2): w_j = (shift + s_j)^p. */
export function polyAttentionNaive(task, p, shift, c, opts) {
  const materialize = !opts || opts.materialize !== false;
  const Q = task.Q, K = task.K, V = task.V, M = task.M, N = task.N, dv = task.dv;
  const W = materialize ? new Array(M) : null;
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const w = new Array(N);
    let Z = 0;
    for (let j = 0; j < N; j++) {
      const s = dot(Q[i], K[j], c);
      const x = Math.pow(shift + s, p);
      if (c) c.mul += p;
      w[j] = x; Z += x; if (c) c.add++;
    }
    for (let j = 0; j < N; j++) { w[j] /= Z; if (c) c.div++; }
    const o = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const wj = w[j];
      for (let f = 0; f < dv; f++) { o[f] += wj * V[j][f]; if (c) { c.mul++; c.add++; } }
    }
    if (materialize) W[i] = w;
    out[i] = o;
  }
  return { weights: W, outputs: out };
}

/** Naive ReLU^p attention O(N^2): w_j = max(shift + s_j, 0)^p. Returns clamp frac. */
export function reluAttentionNaive(task, p, shift, c, opts) {
  const materialize = !opts || opts.materialize !== false;
  const Q = task.Q, K = task.K, V = task.V, M = task.M, N = task.N, dv = task.dv;
  const W = materialize ? new Array(M) : null;
  const out = new Array(M);
  let clamped = 0;
  let all = 0;
  for (let i = 0; i < M; i++) {
    const w = new Array(N);
    let Z = 0;
    for (let j = 0; j < N; j++) {
      const s = dot(Q[i], K[j], c);
      const base = shift + s;
      const x = Math.pow(base > 0 ? base : 0, p);
      if (c) { c.mul += p + 1; c.add++; }
      if (base <= 0) clamped++;
      all++;
      w[j] = x; Z += x; if (c) c.add++;
    }
    for (let j = 0; j < N; j++) { w[j] /= Z; if (c) c.div++; }
    const o = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const wj = w[j];
      for (let f = 0; f < dv; f++) { o[f] += wj * V[j][f]; if (c) { c.mul++; c.add++; } }
    }
    if (materialize) W[i] = w;
    out[i] = o;
  }
  return { weights: W, outputs: out, clampFrac: all ? clamped / all : 0 };
}

// ---------------------------------------------------------------------------
// Generic feature-space linear attention: stats S (m x dv) and s (m)
//   num = phi(q)^T S,  den = phi(q)^T s,  out = num / den    O(N*m*dv + M*m*dv)
// ---------------------------------------------------------------------------

export function featureStats(phiK, V, N, dv, ctr) {
  const m = phiK[0].length;
  const sfeat = new Array(m).fill(0);
  const S = new Array(m);
  for (let a = 0; a < m; a++) S[a] = new Array(dv).fill(0);
  for (let j = 0; j < N; j++) {
    const phi = phiK[j];
    for (let a = 0; a < m; a++) {
      const pv = phi[a];
      sfeat[a] += pv; if (ctr) ctr.add++;
      const row = S[a];
      for (let g = 0; g < dv; g++) { row[g] += pv * V[j][g]; if (ctr) { ctr.mul++; ctr.add++; } }
    }
  }
  return { sfeat, S, m };
}

export function queryLinear(sfeat, S, phiQ, M, dv, ctr) {
  const m = sfeat.length;
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const phi = phiQ[i];
    let den = 0;
    for (let a = 0; a < m; a++) { den += phi[a] * sfeat[a]; if (ctr) { ctr.mul++; ctr.add++; } }
    const o = new Array(dv).fill(0);
    for (let g = 0; g < dv; g++) {
      let num = 0;
      for (let a = 0; a < m; a++) { num += phi[a] * S[a][g]; if (ctr) { ctr.mul++; ctr.add++; } }
      o[g] = num / den; if (ctr) ctr.div++;
    }
    out[i] = o;
  }
  return out;
}

export function effectiveWeights(phiK, phiQ, N, M, ctr) {
  const m = phiK[0].length;
  const W = new Array(M);
  for (let i = 0; i < M; i++) {
    const phi = phiQ[i];
    const w = new Array(N);
    let Z = 0;
    for (let j = 0; j < N; j++) {
      const phik = phiK[j];
      let dp = 0;
      for (let a = 0; a < m; a++) { dp += phi[a] * phik[a]; if (ctr) { ctr.mul++; ctr.add++; } }
      if (dp < 0) dp = 0;
      w[j] = dp; Z += dp; if (ctr) ctr.add++;
    }
    for (let j = 0; j < N; j++) { w[j] /= Z; if (ctr) ctr.div++; }
    W[i] = w;
  }
  return W;
}

// ---------------------------------------------------------------------------
// Polynomial linear attention (float), built from the exact monomial feature map
// ---------------------------------------------------------------------------

export function polyAttentionLinear(task, p, shift, ctr) {
  const plan = polyFeaturePlan(task.d, p, shift);
  const N = task.N, M = task.M, dv = task.dv;
  const phiK = new Array(N);
  const phiQ = new Array(M);
  for (let j = 0; j < N; j++) phiK[j] = polyFeatures(task.K[j], plan.exps, plan.sqrtCoeffs, ctr);
  for (let i = 0; i < M; i++) phiQ[i] = polyFeatures(task.Q[i], plan.exps, plan.sqrtCoeffs, ctr);
  const st = featureStats(phiK, task.V, N, dv, ctr);
  const outputs = queryLinear(st.sfeat, st.S, phiQ, M, dv, ctr);
  return { outputs, phiK, phiQ, m: plan.m };
}

// ---------------------------------------------------------------------------
// BigInt exact polynomial attention
//   coeff_alpha = p!/(alpha! (p-|alpha|)!) * shift^(p-|alpha|)  is an integer,
//   so T_alpha = sum_j prod_i k_ji^{alpha_i} V_j  and  t_alpha likewise are
//   integers, and num/den are integer sums.  No square roots, no floats.
// ---------------------------------------------------------------------------

export function polyLinearBigInt(plan, K, V, Q, N, M, dv) {
  const exps = plan.exps;
  const coeffs = plan.coeffs.map(function (c) { return BigInt(Math.round(c)); });
  const m = exps.length;
  const T = new Array(m);
  const t = new Array(m).fill(0n);
  for (let a = 0; a < m; a++) T[a] = new Array(dv).fill(0n);
  for (let j = 0; j < N; j++) {
    for (let a = 0; a < m; a++) {
      const al = exps[a];
      let mono = 1n;
      for (let i = 0; i < al.length; i++) { let e = al[i]; while (e-- > 0) mono *= K[j][i]; }
      t[a] += mono;
      for (let g = 0; g < dv; g++) T[a][g] += mono * V[j][g];
    }
  }
  const den = new Array(M).fill(0n);
  const num = new Array(M);
  for (let i = 0; i < M; i++) num[i] = new Array(dv).fill(0n);
  for (let i = 0; i < M; i++) {
    for (let a = 0; a < m; a++) {
      const al = exps[a];
      let mono = 1n;
      for (let f = 0; f < al.length; f++) { let e = al[f]; while (e-- > 0) mono *= Q[i][f]; }
      const coef = coeffs[a] * mono;
      den[i] += coef * t[a];
      for (let g = 0; g < dv; g++) num[i][g] += coef * T[a][g];
    }
  }
  return { den, num, m };
}

function bigPow(base, p) {
  let r = 1n;
  for (let i = 0; i < p; i++) r *= base;
  return r;
}

/**
 * Randomized integer exactness check. The naive O(N^2) sums
 *   denN[i] = sum_j (shift + s_ij)^p,  numN[i][g] = sum_j (shift + s_ij)^p V_jg
 * are compared, by literal BigInt equality, with the monomial sufficient-statistic
 * sums. The ReLU^p naive sum max(shift+s,0)^p is compared with the polynomial
 * statistic as a control: it differs exactly when the clamp fires, evidence that
 * the rectified kernel has no finite monomial feature map.
 */
export function bigintPolyExactness(opts) {
  const trials = opts.trials;
  const seed = opts.seed;
  const maxN = opts.maxN, maxD = opts.maxD, maxDV = opts.maxDV, maxP = opts.maxP;
  const rng = mulberry32(seed);
  const ri = function (lo, hi) { return lo + Math.floor(rng() * (hi - lo + 1)); };
  let rows = 0, comparisons = 0, mismatches = 0;
  let reluComparisons = 0, reluMismatches = 0;
  let example = null;
  for (let tr = 0; tr < trials; tr++) {
    const N = ri(2, maxN), d = ri(1, maxD), dv = ri(1, maxDV), p = ri(1, maxP);
    const shift = BigInt(ri(0, 3));
    const K = [], V = [], Q = [];
    for (let j = 0; j < N; j++) {
      const k = [], v = [];
      for (let f = 0; f < d; f++) k.push(BigInt(ri(-9, 9)));
      for (let f = 0; f < dv; f++) v.push(BigInt(ri(-9, 9)));
      K.push(k); V.push(v);
    }
    const M = ri(1, 3);
    for (let i = 0; i < M; i++) {
      const q = [];
      for (let f = 0; f < d; f++) q.push(BigInt(ri(-9, 9)));
      Q.push(q);
    }
    const plan = polyFeaturePlan(d, p, Number(shift));
    const approx = polyLinearBigInt(plan, K, V, Q, N, M, dv);
    for (let i = 0; i < M; i++) {
      let denN = 0n, denRelu = 0n;
      const numN = new Array(dv).fill(0n);
      for (let j = 0; j < N; j++) {
        let s = 0n;
        for (let f = 0; f < d; f++) s += Q[i][f] * K[j][f];
        const base = shift + s;
        const w = bigPow(base, p);
        denN += w;
        for (let g = 0; g < dv; g++) numN[g] += w * V[j][g];
        const wr = bigPow(base > 0n ? base : 0n, p);
        denRelu += wr;
      }
      rows++;
      comparisons++;
      if (denN !== approx.den[i]) mismatches++;
      for (let g = 0; g < dv; g++) {
        comparisons++;
        if (numN[g] !== approx.num[i][g]) mismatches++;
      }
      reluComparisons++;
      if (denRelu !== approx.den[i]) reluMismatches++;
      if (!example) {
        example = {
          N, d, dv, p, shift: shift.toString(),
          denN: denN.toString(), denStats: approx.den[i].toString(),
          numN0: numN[0].toString(), numStats0: approx.num[i][0].toString(),
        };
      }
    }
  }
  return { trials, rows, comparisons, mismatches, reluComparisons, reluMismatches, example };
}

// ---------------------------------------------------------------------------
// Positive random features (Performer / FAVOR+)
//   phi(x)_r = exp(omega_r . (sqrt(beta) x) - beta ||x||^2 / 2) / sqrt(m)
//   E[phi(q).phi(k)] = exp(beta q.k)   for omega_r ~ N(0, I)
// Uses m exp() per token (not N per query).  A homogeneous ReLU variant
//   phi(x)_r = max(omega_r . x, 0) / sqrt(m)
// gives the exp()-free arc-cosine kernel; it has no temperature knob because it
// is homogeneous (scaling x scales every feature equally, and normalization
// cancels it).
// ---------------------------------------------------------------------------

export function makeRandomFeatures(m, d, seed) {
  const rng = mulberry32(seed);
  const omega = new Array(m);
  for (let r = 0; r < m; r++) {
    const v = new Array(d);
    for (let f = 0; f < d; f++) v[f] = gaussian(rng);
    omega[r] = v;
  }
  return { omega, m, d };
}

export function prfFeatures(x, rf, beta, ctr) {
  const m = rf.m, d = rf.d;
  const scale = Math.sqrt(beta);
  const norm2 = beta * dot(x, x, ctr);
  const inv = 1 / Math.sqrt(m);
  const out = new Array(m);
  for (let r = 0; r < m; r++) {
    const w = rf.omega[r];
    let dp = 0;
    for (let f = 0; f < d; f++) { dp += w[f] * scale * x[f]; if (ctr) { ctr.mul++; ctr.add++; } }
    const e = Math.exp(dp - norm2 / 2);
    if (ctr) ctr.exp++;
    out[r] = e * inv;
    if (ctr) ctr.mul++;
  }
  return out;
}

export function reluRfFeatures(x, rf, ctr) {
  const m = rf.m, d = rf.d;
  const inv = 1 / Math.sqrt(m);
  const out = new Array(m);
  for (let r = 0; r < m; r++) {
    const w = rf.omega[r];
    let dp = 0;
    for (let f = 0; f < d; f++) { dp += w[f] * x[f]; if (ctr) { ctr.mul++; ctr.add++; } }
    out[r] = (dp > 0 ? dp : 0) * inv;
    if (ctr) ctr.mul++;
  }
  return out;
}

/** Build all token features for a random-feature bank (used by sweeps). */
export function prfFeaturesAll(task, rf, beta, ctr) {
  const N = task.N, M = task.M;
  const phiK = new Array(N);
  const phiQ = new Array(M);
  for (let j = 0; j < N; j++) phiK[j] = prfFeatures(task.K[j], rf, beta, ctr);
  for (let i = 0; i < M; i++) phiQ[i] = prfFeatures(task.Q[i], rf, beta, ctr);
  return { phiK, phiQ };
}

export function reluRfFeaturesAll(task, rf, ctr) {
  const N = task.N, M = task.M;
  const phiK = new Array(N);
  const phiQ = new Array(M);
  for (let j = 0; j < N; j++) phiK[j] = reluRfFeatures(task.K[j], rf, ctr);
  for (let i = 0; i < M; i++) phiQ[i] = reluRfFeatures(task.Q[i], rf, ctr);
  return { phiK, phiQ };
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export function recallMetrics(task, weights, outputs) {
  const V = task.V, targets = task.targets, M = task.M;
  let index = 0, value = 0;
  for (let i = 0; i < M; i++) {
    const w = weights[i];
    let am = -Infinity, ai = -1;
    for (let j = 0; j < w.length; j++) { if (w[j] > am) { am = w[j]; ai = j; } }
    if (ai === targets[i]) index++;
    const o = outputs[i];
    let bd = Infinity, bi = -1;
    for (let m = 0; m < V.length; m++) {
      let dist = 0;
      for (let f = 0; f < o.length; f++) { const diff = o[f] - V[m][f]; dist += diff * diff; }
      if (dist < bd) { bd = dist; bi = m; }
    }
    if (bi === targets[i]) value++;
  }
  return { index: index / M, value: value / M };
}

export function targetMass(task, weights) {
  let s = 0;
  for (let i = 0; i < task.M; i++) s += weights[i][task.targets[i]];
  return s / task.M;
}

export function meanKL(P, Q) {
  let total = 0;
  for (let i = 0; i < P.length; i++) {
    let row = 0;
    for (let j = 0; j < P[i].length; j++) {
      const p = P[i][j], q = Q[i][j];
      if (p > 0) {
        if (!(q > 0)) return Infinity;
        row += p * Math.log(p / q);
      }
    }
    total += row;
  }
  return total / P.length;
}

export function maxAbsDiff(A, B) {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    for (let f = 0; f < A[i].length; f++) {
      const d = Math.abs(A[i][f] - B[i][f]);
      if (d > m) m = d;
    }
  }
  return m;
}

export function mean(arr) {
  if (arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

export function std(arr) {
  if (arr.length < 2) return 0;
  const mu = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += (arr[i] - mu) * (arr[i] - mu);
  return Math.sqrt(s / (arr.length - 1));
}

/** The task's score bound: unit keys and unit queries imply |q.k| <= 1. */
export function scoreBound(task) {
  let mx = 0;
  for (let i = 0; i < task.M; i++) {
    for (let j = 0; j < task.N; j++) {
      const s = Math.abs(dot(task.Q[i], task.K[j], null));
      if (s > mx) mx = s;
    }
  }
  return mx;
}
