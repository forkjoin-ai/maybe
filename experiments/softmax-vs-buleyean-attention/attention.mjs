/**
 * attention.mjs -- dependency-free core for the
 * "softmax vs Buleyean/Laplace affine-or-hinge linear attention" experiment.
 *
 * No imports at all (not even Node builtins). Pure ECMAScript + BigInt.
 *
 * Conventions
 * -----------
 *   score             s_ij = q_i . k_j
 *   rejection / void  v_ij = voidSign * s_ij
 *   softmax weight    w_ij ~ exp(beta * s_ij) = exp(-beta * v_ij)
 *                     (beta is the logit scale / inverse temperature, the brief's sigma)
 *   affine weight     w_ij  = R_i - beta * v_ij + 1                 (unclamped Buleyean/Laplace)
 *   hinge weight      w_ij  = 1 + max(R_i - beta * v_ij, 0)         (clamped reference)
 *
 *   voidSign = -1 : v = -s, so the affine rule is INCREASING in the score.
 *                   This is the softmax-aligned ("void-dual") reading and the only
 *                   reading under which the affine rule can approximate softmax.
 *   voidSign = +1 : v = +s, i.e. the literal w = R - s + 1 written in the brief.
 *
 * With the budget R_i = beta * C (C a fixed score ceiling), the affine slope is beta
 * and R is a LINEAR inverse-temperature, exactly as the brief requests.
 *
 * The unclamped affine rule factorises (r_j = voidSign * k_j):
 *   sum_j v_ij      = q_i . (sum_j r_j)
 *   sum_j w_ij      = N * (R_i + 1) - beta * q_i . (sum_j r_j)
 *   sum_j w_ij V_jf = (R_i + 1) * (sum_j V_jf) - beta * q_i . (sum_j r_j V_jf)
 * so two sufficient statistics -- S_k = sum_j r_j and S_kv = sum_j r_j V_j^T -- plus
 * the value sum S_v answer every query in O(d * dv) without ever materialising N
 * weights. The hinge clamp is not linear and therefore does not factorise.
 */

// ---------------------------------------------------------------------------
// Deterministic PRNG (data generation only; NOT counted as attention cost)
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
// Associative-recall / induction task
// ---------------------------------------------------------------------------

/**
 * Build an associative-recall task. N random unit keys and N random values;
 * M queries, query i is unit key targets[i] plus keyNoise. targets[i] = i % N.
 * Unit keys/queries imply |score| <= 1, so budgetC = 1 bounds every |void|.
 */
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
      const e = Math.exp(beta * (s[j] - m));   // one transcendental per key
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
// Affine (unclamped) / hinge (clamped) attention computed naively, O(N^2)
// ---------------------------------------------------------------------------

/**
 * w_ij = R - beta*v_ij + 1        (affine)
 * w_ij = 1 + max(R - beta*v_ij, 0) (hinge)
 * with R = beta * budgetC and v = voidSign * s.
 */
export function affineAttentionNaive(task, beta, voidSign, budgetC, hinge, c, opts) {
  const materialize = !opts || opts.materialize !== false;
  const Q = task.Q, K = task.K, V = task.V, M = task.M, N = task.N, dv = task.dv;
  const R = beta * budgetC;
  const W = materialize ? new Array(M) : null;
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const s = new Array(N);
    for (let j = 0; j < N; j++) s[j] = dot(Q[i], K[j], c);
    let Z = 0;
    const w = new Array(N);
    for (let j = 0; j < N; j++) {
      const v = voidSign * s[j];
      const base = R - beta * v;
      let x;
      if (hinge) x = 1 + (base > 0 ? base : 0);
      else x = base + 1;
      if (c) { c.mul++; c.add++; c.add++; }   // beta*v, R - ., +1 / clamp
      w[j] = x;
      Z += x;
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
// Affine linear attention O(N*d^2 + M*d^2): sufficient statistics only
// ---------------------------------------------------------------------------

export function affineAttentionLinear(task, beta, voidSign, budgetC, c) {
  const Q = task.Q, K = task.K, V = task.V, M = task.M, N = task.N;
  const d = task.d, dv = task.dv;
  const R = beta * budgetC;

  const Sk = new Array(d).fill(0);          // sum_j r_j,  r_j = voidSign * k_j
  const Sv = new Array(dv).fill(0);         // sum_j V_j
  for (let j = 0; j < N; j++) {
    for (let f = 0; f < d; f++) { Sk[f] += voidSign * K[j][f]; if (c) { c.mul++; c.add++; } }
    for (let f = 0; f < dv; f++) { Sv[f] += V[j][f]; if (c) c.add++; }
  }
  const Skv = [];                            // sum_j r_j V_j^T
  for (let f = 0; f < d; f++) {
    const row = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const rjf = voidSign * K[j][f];
      for (let g = 0; g < dv; g++) { row[g] += rjf * V[j][g]; if (c) { c.mul++; c.add++; } }
    }
    Skv.push(row);
  }

  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const q = Q[i];
    const qSk = dot(q, Sk, c);                                     // q . sum_j r_j
    const denom = N * (R + 1) - beta * qSk;                        // sum_j w_ij
    if (c) { c.mul++; c.add++; c.add++; }
    const o = new Array(dv).fill(0);
    for (let g = 0; g < dv; g++) {
      let acc = 0;
      for (let f = 0; f < d; f++) { acc += q[f] * Skv[f][g]; if (c) { c.mul++; c.add++; } }
      const num = (R + 1) * Sv[g] - beta * acc;                    // sum_j w_ij V_jg
      if (c) { c.mul++; c.add++; }
      o[g] = num / denom;
      if (c) c.div++;
    }
    out[i] = o;
  }
  return { weights: null, outputs: out };
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export function recallMetrics(task, weights, outputs) {
  const V = task.V, targets = task.targets, M = task.M;
  let index = 0;
  let value = 0;
  for (let i = 0; i < M; i++) {
    const w = weights[i];
    let am = -Infinity;
    let ai = -1;
    for (let j = 0; j < w.length; j++) {
      if (w[j] > am) { am = w[j]; ai = j; }
    }
    if (ai === targets[i]) index++;
    const o = outputs[i];
    let bd = Infinity;
    let bi = -1;
    for (let m = 0; m < V.length; m++) {
      let dist = 0;
      for (let f = 0; f < o.length; f++) {
        const diff = o[f] - V[m][f];
        dist += diff * diff;
      }
      if (dist < bd) { bd = dist; bi = m; }
    }
    if (bi === targets[i]) value++;
  }
  return { index: index / M, value: value / M };
}

export function meanKL(P, Q) {
  let total = 0;
  for (let i = 0; i < P.length; i++) {
    let row = 0;
    for (let j = 0; j < P[i].length; j++) {
      const p = P[i][j];
      const q = Q[i][j];
      if (p > 0) {
        if (q <= 0) return NaN;
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

export function negativeWeightFraction(W) {
  let neg = 0;
  let all = 0;
  for (let i = 0; i < W.length; i++) {
    for (let j = 0; j < W[i].length; j++) { if (W[i][j] < 0) neg++; all++; }
  }
  return neg / all;
}

export function clampedFraction(task, beta, voidSign, budgetC) {
  const Q = task.Q, K = task.K, M = task.M, N = task.N;
  const R = beta * budgetC;
  let clamped = 0;
  let all = 0;
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const s = dot(Q[i], K[j], null);
      const base = R - beta * (voidSign * s);
      if (base < 0) clamped++;
      all++;
    }
  }
  return clamped / all;
}
