/**
 * hamming-attention.mjs -- dependency-free core for the
 * "rejection-count (popcount / Hamming) score instead of the float dot product"
 * experiment, for the @a0n/maybe package.
 *
 * No imports at all (not even Node builtins). Pure ECMAScript + BigInt.
 *
 * Setting
 * -------
 * The two sibling experiments showed the affine Buleyean rule
 *   w_j = R + s_j + 1          (s_j = q . k_j, exp-free, O(N))
 * is bit-exactly linearizable and O(N), but has a hard O(1/N) concentration
 * ceiling, and that richer *feature* kernels only remove the ceiling at a
 * feature budget m far larger than N.  This experiment tests the remaining
 * matmul-avoidance route: replace the dot product by a REJECTION COUNT.
 *
 *   1. Quantize Q, K to b-bit integer codes (b = 1 binary, 2, 4; plus an
 *      explicit ternary {-1,0,+1} codec).  The key/query component range is
 *      data-driven ([-kmax, kmax]) so the codec does not collapse when d grows
 *      while unit components shrink like 1/sqrt(d).  b-bit levels use a
 *      binary-reflected Gray code so adjacent levels differ in one bit.
 *   2. Score with popcount, exact integer arithmetic:
 *        binary/bit:   v_ij = popcount(xor(code(q_i), code(k_j)))   in [0, d*b]
 *                      s_ij = d*b - v_ij
 *        ternary:      A = popcount(Pq & Pk) + popcount(Mq & Mk)      (agreements)
 *                      B = popcount(Pq & Mk) + popcount(Mq & Pk)      (oppositions)
 *                      s_ij = A - B   (this is EXACTLY the integer ternary dot
 *                                      product; P/M are the +1 / -1 bitmasks)
 *                      v_ij = d - s_ij
 *      The Buleyean "God Formula" acts on the REJECTION count:
 *        w_ij = R - min(v_ij, R) + 1            (so w >= 1 always: the floor)
 *        P_ij = w_ij / sum_j w_ij
 *   3. Value aggregation: integer weighted counts.  Quantize V to small
 *      integers Vint and accumulate num_ig = sum_j w_ij * Vint_jg, den_i =
 *      sum_j w_ij, both exact integers; the float output is
 *      out_ig = vscale * num_ig / den_i (one division per output element).
 *      vscale is chosen so the dequantized values match the RMS of the real V.
 *   4. Cost is counted in primitive instructions (bit-ops / FMA), not wall time.
 *
 * The Hamming kernel is linear in the rejection count, so it inherits a
 * saturation ceiling too; the core of the experiment is characterising it:
 * per row, with target rejection v_t and closest non-target rejection
 *   v_min = min_{j != t} v_ij,   margin delta = v_min - v_t,
 * the best possible target mass over any budget R is the exact maximum of
 * w_t(R) / sum_j w_j(R) over the breakpoints R in {v_ij}.  When delta >= 0 the
 * optimum sits at R = v_min and equals (delta+1)/(delta+N): softmax-like
 * concentration needs a large Hamming MARGIN, which is how bit-width and the
 * ambient dimension d buy quality.
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
// Associative-recall / induction task (verbatim from
// ../softmax-vs-buleyean-attention/attention.mjs, so the numbers match).
// N random unit keys, N Gaussian values, M queries; query i is unit key
// targets[i] = i % N plus keyNoise, renormalised.  Unit keys/queries bound
// |q.k| <= 1 in the float reference.  keyNoise is per coordinate; callers that
// sweep d scale it by sqrt(8/d) to hold the total noise norm constant.
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
// Bit primitives (exact integer)
// ---------------------------------------------------------------------------
export function popcount32(x) {
  x = x >>> 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >>> 24);
}

export function popcountBig(x) {
  let c = 0;
  const MASK = 0xffffffffn;
  while (x > 0n) { c += popcount32(Number(x & MASK) >>> 0); x >>= 32n; }
  return c;
}

export function wordsForBits(nbits) { return (nbits + 31) >>> 5; }

/** Binary-reflected Gray code (adjacent integers differ in exactly one bit). */
export function grayCode(l) { return (l ^ (l >>> 1)) >>> 0; }

/** Pack integer codes LSB-first, bitsPer bits each, into 32-bit words. */
export function packLevels(codes, bitsPer) {
  const words = new Array(wordsForBits(codes.length * bitsPer)).fill(0);
  let bp = 0;
  for (let f = 0; f < codes.length; f++) {
    let v = codes[f];
    for (let t = 0; t < bitsPer; t++) {
      if ((v >>> t) & 1) {
        const wi = bp >>> 5;
        const bi = bp & 31;
        words[wi] = (words[wi] | (1 << bi)) >>> 0;
      }
      bp++;
    }
  }
  return words;
}

export function popcountXor(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += popcount32((a[k] ^ b[k]) >>> 0);
  return s;
}

export function popcountAnd(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += popcount32((a[k] & b[k]) >>> 0);
  return s;
}

// ---------------------------------------------------------------------------
// Quantisers
// ---------------------------------------------------------------------------

/** Data-driven symmetric key/query component range (unit vectors). */
export function keyRange(task) {
  let m = 0;
  for (const A of [task.K, task.Q]) {
    for (let j = 0; j < A.length; j++) {
      for (let f = 0; f < A[j].length; f++) {
        const a = Math.abs(A[j][f]);
        if (a > m) m = a;
      }
    }
  }
  return m || 1;
}

/**
 * b-bit quantisation of key/query vectors.  level l in [0, 2^b-1] by rounding
 * onto uniform bins over [-kmax, kmax]; code = gray(l).  dEff = d*b output bits.
 * bits = 1 is sign quantisation (code 0 for x<0, 1 for x>=0).
 */
export function buildBinaryCodes(vectors, bits, kmax) {
  const d = vectors[0].length;
  const L = (1 << bits) - 1;
  const out = new Array(vectors.length);
  for (let j = 0; j < vectors.length; j++) {
    const levels = new Array(d);
    const codes = new Array(d);
    for (let f = 0; f < d; f++) {
      let l;
      if (bits === 1) {
        l = vectors[j][f] >= 0 ? 1 : 0;
      } else {
        l = Math.round(((vectors[j][f] + kmax) / (2 * kmax)) * L);
        if (l < 0) l = 0; else if (l > L) l = L;
      }
      levels[f] = l;
      codes[f] = grayCode(l);
    }
    out[j] = { kind: bits === 1 ? 'binary' : 'bit', bits, d, levels, words: packLevels(codes, bits), dEff: d * bits, sMax: d * bits, sMin: 0 };
  }
  return out;
}

/**
 * Ternary {-1,0,+1} quantisation.  tau = kmax/3 (three equal bins over
 * [-kmax,kmax]).  Two d-bit masks: pos (level +1) and neg (level -1).
 * dEff (bit budget) = 2*d, but the score costs FOUR AND-popcounts, not two.
 */
export function buildTernaryCodes(vectors, tau) {
  const d = vectors[0].length;
  const out = new Array(vectors.length);
  for (let j = 0; j < vectors.length; j++) {
    const pos = new Array(d).fill(0);
    const neg = new Array(d).fill(0);
    const levels = new Array(d);
    for (let f = 0; f < d; f++) {
      const x = vectors[j][f];
      const l = x > tau ? 1 : (x < -tau ? -1 : 0);
      levels[f] = l;
      if (l === 1) pos[f] = 1; else if (l === -1) neg[f] = 1;
    }
    out[j] = { kind: 'ternary', bits: 2, d, levels, pos: packLevels(pos, 1), neg: packLevels(neg, 1), dEff: 2 * d, sMax: d, sMin: -d };
  }
  return out;
}

/** Reconstruct floating key/query vectors from b-bit codes (for the control). */
export function dequantKeys(codes, kmax) {
  const out = new Array(codes.length);
  for (let j = 0; j < codes.length; j++) {
    const c = codes[j];
    const L = (1 << c.bits) - 1;
    const v = new Array(c.d);
    let nrm = 0;
    for (let f = 0; f < c.d; f++) {
      let x;
      if (c.kind === 'ternary') x = c.levels[f];
      else x = kmax * (2 * (c.levels[f] / L) - 1);
      v[f] = x;
      nrm += x * x;
    }
    nrm = Math.sqrt(nrm) || 1;
    for (let f = 0; f < c.d; f++) v[f] /= nrm;
    out[j] = v;
  }
  return out;
}

/**
 * Exact integer rejection matrix v_ij.  Rows are the M queries, columns N keys.
 *   binary/bit : v = popcount(xor)          (score s = dEff - v)
 *   ternary    : v = d - (A - B)            (score s = A - B = ternary dot)
 */
export function rejectionMatrix(Kcodes, Qcodes, kind) {
  const M = Qcodes.length, N = Kcodes.length;
  const vv = new Array(M);
  for (let i = 0; i < M; i++) {
    const row = new Array(N);
    for (let j = 0; j < N; j++) {
      if (kind === 'ternary') {
        const q = Qcodes[i], k = Kcodes[j];
        const A = popcountAnd(q.pos, k.pos) + popcountAnd(q.neg, k.neg);
        const B = popcountAnd(q.pos, k.neg) + popcountAnd(q.neg, k.pos);
        row[j] = q.d - (A - B);
      } else {
        row[j] = popcountXor(Qcodes[i].words, Kcodes[j].words);
      }
    }
    vv[i] = row;
  }
  return vv;
}

/** The God Formula on the rejection count: w = R - min(v, R) + 1 >= 1. */
export function buleyeanWeight(v, R) { return R - (v < R ? v : R) + 1; }

/**
 * Integer Buleyean Hamming posterior.  Vint[j] are small integers (binary +-1,
 * ternary -1/0/+1, b-bit centered levels).  Returns exact integer numerators
 * and denominators plus the normalised float weights and dequantised output.
 */
export function hammingPosterior(vv, R, Vint, vscale, materializeWeights) {
  const M = vv.length, N = vv[0].length, dv = Vint[0].length;
  const W = materializeWeights ? new Array(M) : null;
  const den = new Array(M);
  const num = new Array(M);
  const out = new Array(M);
  let minW = Infinity;
  let minP = Infinity;
  for (let i = 0; i < M; i++) {
    const w = new Array(N);
    let Z = 0;
    for (let j = 0; j < N; j++) {
      const x = buleyeanWeight(vv[i][j], R);
      if (x < minW) minW = x;
      w[j] = x; Z += x;
    }
    den[i] = Z;
    const n = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const wj = w[j], vj = Vint[j];
      for (let g = 0; g < dv; g++) n[g] += wj * vj[g];
    }
    num[i] = n;
    const o = new Array(dv);
    for (let g = 0; g < dv; g++) o[g] = (vscale * n[g]) / Z;
    out[i] = o;
    for (let j = 0; j < N; j++) {
      const p = w[j] / Z;
      if (p < minP) minP = p;
      if (materializeWeights) w[j] = p;
    }
    if (materializeWeights) W[i] = w;
  }
  return { weights: W, den, num, outputs: out, minWeight: minW, minProb: minP };
}

// ---------------------------------------------------------------------------
// Value quantisation to small integers, with RMS-matched dequantisation.
//   binary : Vint in {-1,+1}
//   ternary: Vint in {-1,0,+1}          (dead zone at +-0.5*std of V)
//   b-bit  : Vint = 2*level-(2^b-1)
// vscale = rms(V) / sqrt(mean(Vint^2)) so the dequantised values match the
// second moment of the real values (zero mean by construction).
// ---------------------------------------------------------------------------
export function quantizeValues(V, kind, bits) {
  const N = V.length, dv = V[0].length;
  let s2 = 0, cnt = 0, vmax = 0;
  for (let j = 0; j < N; j++) for (let g = 0; g < dv; g++) { s2 += V[j][g] * V[j][g]; cnt++; const a = Math.abs(V[j][g]); if (a > vmax) vmax = a; }
  const rms = Math.sqrt(s2 / cnt) || 1;
  const Vint = new Array(N);
  let m2 = 0;
  if (kind === 'binary' || kind === 'ternary') {
    const tau = kind === 'ternary' ? 0.5 * rms : 0;
    for (let j = 0; j < N; j++) {
      const row = new Array(dv);
      for (let g = 0; g < dv; g++) {
        const x = V[j][g];
        const l = kind === 'binary' ? (x >= 0 ? 1 : -1) : (x > tau ? 1 : (x < -tau ? -1 : 0));
        row[g] = l; m2 += l * l;
      }
      Vint[j] = row;
    }
  } else {
    const L = (1 << bits) - 1;
    for (let j = 0; j < N; j++) {
      const row = new Array(dv);
      for (let g = 0; g < dv; g++) {
        let l = Math.round(((V[j][g] + vmax) / (2 * vmax)) * L);
        if (l < 0) l = 0; else if (l > L) l = L;
        const vi = 2 * l - L;
        row[g] = vi; m2 += vi * vi;
      }
      Vint[j] = row;
    }
  }
  const vscale = rms / Math.sqrt(m2 / cnt);
  return { Vint, vscale, rms, vmax, kind, bits };
}

// ---------------------------------------------------------------------------
// Float softmax reference (generic Q, K, V; also used on dequantised codes)
// ---------------------------------------------------------------------------
export function makeCounter() { return { exp: 0, mul: 0, add: 0, div: 0, fma: 0, bitops: 0 }; }

export function softmaxGeneral(Q, K, V, beta, c) {
  const M = Q.length, N = K.length, dv = V[0].length;
  const W = new Array(M), out = new Array(M);
  for (let i = 0; i < M; i++) {
    const s = new Array(N);
    let m = -Infinity;
    for (let j = 0; j < N; j++) {
      let a = 0;
      const q = Q[i], k = K[j];
      for (let f = 0; f < q.length; f++) { a += q[f] * k[f]; if (c) { c.mul++; c.add++; c.fma++; } }
      s[j] = a;
      if (a > m) m = a;
    }
    let Z = 0;
    const w = new Array(N);
    for (let j = 0; j < N; j++) {
      const e = Math.exp(beta * (s[j] - m));
      if (c) c.exp++;
      w[j] = e; Z += e;
      if (c) c.add++;
    }
    for (let j = 0; j < N; j++) { w[j] /= Z; if (c) c.div++; }
    const o = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const wj = w[j], vj = V[j];
      for (let g = 0; g < dv; g++) { o[g] += wj * vj[g]; if (c) { c.mul++; c.add++; c.fma++; } }
    }
    W[i] = w; out[i] = o;
  }
  return { weights: W, outputs: out };
}

export function softmaxFloat(task, beta, c) {
  return softmaxGeneral(task.Q, task.K, task.V, beta, c);
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

export function recallValueOnly(task, outputs) {
  const V = task.V, targets = task.targets, M = task.M;
  let value = 0;
  for (let i = 0; i < M; i++) {
    const o = outputs[i];
    let bd = Infinity, bi = -1;
    for (let m = 0; m < V.length; m++) {
      let dist = 0;
      for (let f = 0; f < o.length; f++) { const diff = o[f] - V[m][f]; dist += diff * diff; }
      if (dist < bd) { bd = dist; bi = m; }
    }
    if (bi === targets[i]) value++;
  }
  return value / M;
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

export function mean(arr) { if (!arr.length) return 0; let s = 0; for (const x of arr) s += x; return s / arr.length; }
export function std(arr) {
  if (arr.length < 2) return 0;
  const mu = mean(arr);
  let s = 0;
  for (const x of arr) s += (x - mu) * (x - mu);
  return Math.sqrt(s / (arr.length - 1));
}
export function quantile(arr, q) {
  if (!arr.length) return 0;
  const a = arr.slice().sort(function (x, y) { return x - y; });
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return a[lo];
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

// ---------------------------------------------------------------------------
// Concentration: exact achievable max target mass over the budget R.
// w_j(R) = 1 + max(R - v_j, 0) is piecewise linear with breakpoints at v_j;
// w_t(R)/sum_j w_j(R) is monotone between breakpoints, so its maximum is at a
// breakpoint.  Scanning the breakpoints is exact, no grid.
// ---------------------------------------------------------------------------
export function maxTargetMassOverR(vRow, target) {
  const cand = [];
  const seen = {};
  for (let j = 0; j < vRow.length; j++) {
    const v = vRow[j];
    if (!seen[v]) { seen[v] = 1; cand.push(v); }
  }
  cand.sort(function (a, b) { return a - b; });
  let best = -1, bestR = 0;
  for (let k = 0; k < cand.length; k++) {
    const R = cand[k];
    let wt = buleyeanWeight(vRow[target], R), Z = 0;
    for (let j = 0; j < vRow.length; j++) Z += buleyeanWeight(vRow[j], R);
    const mass = wt / Z;
    if (mass > best) { best = mass; bestR = R; }
  }
  return { mass: best, bestR };
}

/**
 * Per-row Hamming margin and saturation ceiling.
 *   v_t   = target rejection;  v_min = closest non-target rejection
 *   delta = v_min - v_t
 *   analytic ceiling (when delta >= 0) = (delta+1)/(delta+N)
 *   exactMax = exact max target mass over R (breakpoint scan)
 */
export function marginStats(vv, targets) {
  const M = vv.length, N = vv[0].length;
  const deltas = [];
  let exactMaxSum = 0, ceilingSum = 0, bestRSum = 0, vtSum = 0, vminSum = 0;
  for (let i = 0; i < M; i++) {
    const row = vv[i], t = targets[i];
    let vmin = Infinity;
    for (let j = 0; j < N; j++) if (j !== t && row[j] < vmin) vmin = row[j];
    const vt = row[t];
    const delta = vmin - vt;
    deltas.push(delta);
    const ceiling = delta >= 0 ? (delta + 1) / (delta + N) : 1 / N;
    ceilingSum += ceiling;
    const em = maxTargetMassOverR(row, t);
    exactMaxSum += em.mass;
    bestRSum += em.bestR;
    vtSum += vt; vminSum += vmin;
  }
  return {
    M, N,
    deltaMean: mean(deltas),
    deltaMedian: quantile(deltas, 0.5),
    deltaMin: Math.min.apply(null, deltas),
    deltaMax: Math.max.apply(null, deltas),
    ceilingMean: ceilingSum / M,
    exactMaxMean: exactMaxSum / M,
    bestRMean: bestRSum / M,
    vtMean: vtSum / M,
    vminMean: vminSum / M,
  };
}

// ---------------------------------------------------------------------------
// Cost model: primitive instructions, not wall time.
//   bit-op = one XOR+POPCOUNT fused primitive covering W bits
//   fma    = one floating multiply-add
// Softmax float:  M*N d FMA (scores) + M*N exp + M*N dv FMA (values) + M*N div.
// Hamming:        M*N ceil(dEff/W) bit-ops (scores) + M*N dv integer multiply-
//                 adds (values) + M*N div; zero exp, zero float score matmul.
// Ternary scores cost 4 AND-popcounts per word (2 agreements + 2 oppositions).
// ---------------------------------------------------------------------------
export function softmaxCost(M, N, d, dv) {
  const fma = M * N * (d + dv);
  const exp = M * N;
  const div = M * N;
  return { fma, exp, div, total: fma + exp + div, scoreFma: M * N * d, valueFma: M * N * dv };
}

export function hammingCost(M, N, dEff, dv, W, kind) {
  const words = Math.ceil(dEff / W);
  const groups = kind === 'ternary' ? 4 : 1;
  const bitops = M * N * words * groups;
  const imad = M * N * dv;
  const div = M * N;
  return { bitops, imad, div, total: bitops + imad + div, words, groups, scoreBitOps: bitops };
}

/**
 * Cost of Hamming scores with SOFTMAX normalization and a precomputed exp table.
 *   score    : M*N*ceil(dEff/W) bit-ops (popcount)
 *   max      : M*N integer compares (row max for the shifted table)
 *   exp      : (sMax-sMin+1) Math.exp calls TOTAL, then M*N table READS (memory)
 *   value    : M*N*dv integer multiply-adds (the A.V matmul is NOT removed)
 *   div      : M*N
 * The exp() count is a constant in M,N -- it is no longer per pair.
 */
export function hammingSoftmaxCost(M, N, dEff, dv, W, kind, scoreRange, tableEntries) {
  const words = Math.ceil(dEff / W);
  const groups = kind === 'ternary' ? 4 : 1;
  const bitops = M * N * words * groups;
  const tableReads = M * N;
  const maxReduce = M * N;
  const imad = M * N * dv;
  const div = M * N;
  const expCalls = tableEntries === undefined ? (scoreRange === undefined ? 2 : scoreRange + 1) : tableEntries;
  return {
    bitops, tableReads, maxReduce, imad, div, expCalls,
    total: bitops + tableReads + maxReduce + imad + div,
    words, groups, scoreFma: 0,
  };
}

/** Float softmax on the SAME dEff dequantised bits (matched-information baseline). */
export function softmaxOnCodesCost(M, N, dEff, dv) {
  const fma = M * N * (dEff + dv);
  const exp = M * N, div = M * N;
  return { fma, exp, div, total: fma + exp + div, scoreFma: M * N * dEff, valueFma: M * N * dv };
}

// ---------------------------------------------------------------------------
// SOFTMAX normalization over the exact integer Hamming scores.
//
// The Buleyean "God Formula" w = R - min(v,R) + 1 is AFFINE in the rejection
// count, so it inherits the sibling lane's O(1/N) concentration ceiling.  The
// alternative tested here keeps the popcount score but uses SOFTMAX
// normalization:
//
//   binary/bit:  score s_ij = sMax - v_ij = dEff - popcount(xor)   in [0, dEff]
//   ternary:     score s_ij = A - B = d - v_ij                     in [-d, d]
//
// s_ij is an INTEGER with only (sMax - sMin + 1) distinct values, so
// exp(beta * s_ij) needs at most that many evaluations -- a PRECOMPUTED TABLE
// of size (sMax - sMin + 1), indexed by (rowMax - s_ij).  The transcendental
// becomes one memory read per (i,j); the score matmul is still replaced by
// popcount.  This retains softmax concentration because the normalization is
// the same exponential family (the score is a monotone integer function of the
// Hamming distance).
//
// For SIGN vectors z in {-1,+1}^d the rejection is v = (d - z_q.z_k)/2, so the
// Hamming score d - v is an affine function of the signed dot product and this
// table-softmax is softmax on the quantised keys up to a temperature.  (The
// literal {0,1} form "popcount(xor) = d - q.k" is false.)  For b>1 it is softmax
// on the Gray-code bit metric.
// ---------------------------------------------------------------------------
export function hammingSoftmaxPosterior(vv, beta, sMax, sMin, Vint, vscale, materializeWeights) {
  const M = vv.length, N = vv[0].length, dv = Vint[0].length;
  const range = sMax - sMin;
  const tab = new Array(range + 1);
  for (let k = 0; k <= range; k++) tab[k] = Math.exp(-beta * k);
  const W = materializeWeights ? new Array(M) : null;
  const den = new Array(M), num = new Array(M), out = new Array(M);
  let minP = Infinity;
  for (let i = 0; i < M; i++) {
    const s = new Array(N);
    let smax = -Infinity;
    for (let j = 0; j < N; j++) { const sj = sMax - vv[i][j]; s[j] = sj; if (sj > smax) smax = sj; }
    let Z = 0;
    const w = new Array(N);
    for (let j = 0; j < N; j++) { const x = tab[smax - s[j]]; w[j] = x; Z += x; }
    den[i] = Z;
    const n = new Array(dv).fill(0);
    for (let j = 0; j < N; j++) { const wj = w[j], vj = Vint[j]; for (let g = 0; g < dv; g++) n[g] += wj * vj[g]; }
    num[i] = n;
    const o = new Array(dv);
    for (let g = 0; g < dv; g++) o[g] = (vscale * n[g]) / Z;
    out[i] = o;
    for (let j = 0; j < N; j++) { const p = w[j] / Z; if (p < minP) minP = p; if (materializeWeights) w[j] = p; }
    if (materializeWeights) W[i] = w;
  }
  return { weights: W, den, num, outputs: out, minProb: minP, tableLen: range + 1 };
}

/** Float-V control for the table-softmax path (weights x real values). */
export function hammingSoftmaxOutputsFloat(vv, beta, sMax, sMin, Vfloat) {
  return hammingSoftmaxPosterior(vv, beta, sMax, sMin, Vfloat, 1, false).outputs;
}

/** Does the table lookup reproduce direct Math.exp bit-for-bit, and how many exp() calls does it save? */
export function hammingSoftmaxTableExactness(opts) {
  const rng = mulberry32(opts.seed);
  const ri = function (lo, hi) { return lo + Math.floor(rng() * (hi - lo + 1)); };
  let comparisons = 0, mismatches = 0, tableEvals = 0, directEvals = 0, example = null;
  for (let tr = 0; tr < opts.trials; tr++) {
    const N = ri(2, opts.maxN);
    const M = ri(1, 4);
    const range = ri(1, opts.maxRange);
    const beta = ri(1, 6);
    // The table is built ONCE per (beta, range) and reused across all M rows;
    // direct softmax evaluates Math.exp once per (i,j) pair.
    const tab = new Array(range + 1);
    for (let k = 0; k <= range; k++) { tab[k] = Math.exp(-beta * k); tableEvals++; }
    for (let i = 0; i < M; i++) {
      const row = new Array(N);
      for (let j = 0; j < N; j++) row[j] = ri(0, range);
      let smax = -Infinity;
      for (let j = 0; j < N; j++) if (row[j] > smax) smax = row[j];
      for (let j = 0; j < N; j++) {
        const s = row[j];
        const viaTable = tab[smax - s];
        const direct = Math.exp(beta * (s - smax));
        comparisons++; directEvals++;
        if (viaTable !== direct) mismatches++;
        if (!example) example = { beta, range, s, smax, viaTable, direct };
      }
    }
  }
  return { trials: opts.trials, comparisons, mismatches, tableEvals, directEvals, example };
}

// ---------------------------------------------------------------------------
// BigInt exactness: popcount scores + integer weights + posteriors, compared
// bit-for-bit with the naive per-coordinate O(N^2) computation, plus the
// multi-view consensus algebra.
// ---------------------------------------------------------------------------
function randLevel(rng, bits) {
  if (bits === 1) return rng() < 0.5 ? 1 : 0;
  return Math.floor(rng() * (1 << bits));
}

function packCodesToBigInt(codes, bitsPer) {
  let x = 0n;
  let bp = 0n;
  for (let f = 0; f < codes.length; f++) {
    x |= (BigInt(codes[f]) << bp);
    bp += BigInt(bitsPer);
  }
  return x;
}

/**
 * Randomized integer exactness.  Each trial builds random b-bit key/query
 * codes and random small-integer values; the naive per-coordinate path and the
 * popcount path must agree on every rejection count, every weight, and every
 * posterior numerator/denominator, as literal BigInt equality.
 *
 * Also checked:
 *   - ternary: naive signed dot product vs the four AND-popcount expression.
 *   - consensus: fuse k views by componentwise addition of integer WEIGHT
 *     vectors; equals the naive sum over views bit-for-bit and is order-
 *     independent.  The control "sum the rejection counts, then apply the God
 *     Formula" is NOT equal (the min clamp is not linear) -- mismatches counted.
 */
export function bigintHammingExactness(opts) {
  const trials = opts.trials;
  const rng = mulberry32(opts.seed);
  const ri = function (lo, hi) { return lo + Math.floor(rng() * (hi - lo + 1)); };
  const maxN = opts.maxN, maxD = opts.maxD, maxDV = opts.maxDV;
  let rows = 0, comparisons = 0, mismatches = 0;
  let ternaryComparisons = 0, ternaryMismatches = 0;
  let fusedComparisons = 0, fusedMismatches = 0;
  let orderComparisons = 0, orderMismatches = 0;
  let fuseThenRuleComparisons = 0, fuseThenRuleMismatches = 0;
  let minWeight = null;
  let example = null;

  for (let tr = 0; tr < trials; tr++) {
    const N = ri(2, maxN), d = ri(1, maxD), dv = ri(1, maxDV);
    const bits = [1, 2, 4][ri(0, 2)];
    const R = BigInt(ri(0, d * bits + 3));
    const Klev = [], Qlev = [];
    for (let j = 0; j < N; j++) { const a = []; for (let f = 0; f < d; f++) a.push(randLevel(rng, bits)); Klev.push(a); }
    const M = ri(1, 4);
    for (let i = 0; i < M; i++) { const a = []; for (let f = 0; f < d; f++) a.push(randLevel(rng, bits)); Qlev.push(a); }
    const V = [];
    for (let j = 0; j < N; j++) { const a = []; for (let g = 0; g < dv; g++) a.push(BigInt(ri(-4, 4))); V.push(a); }

    // codes = Gray(level), packed LSB-first; naive path compares raw levels bit
    // by bit (equivalent to popcount(Gray) because Gray is a bijection).
    const Kcodes = Klev.map(function (a) {
      const g = a.map(grayCode);
      return { codes: g, big: packCodesToBigInt(g, bits), levels: a };
    });
    const Qcodes = Qlev.map(function (a) {
      const g = a.map(grayCode);
      return { codes: g, big: packCodesToBigInt(g, bits), levels: a };
    });

    const denN = new Array(M).fill(0n);
    const numN = []; for (let i = 0; i < M; i++) numN.push(new Array(dv).fill(0n));
    const denP = new Array(M).fill(0n);
    const numP = []; for (let i = 0; i < M; i++) numP.push(new Array(dv).fill(0n));

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        // naive: explicit per-coordinate difference count on the packed bits
        let vNaive = 0;
        for (let f = 0; f < d; f++) {
          let x = Kcodes[j].codes[f] ^ Qcodes[i].codes[f];
          let t = 0;
          while (t < bits) { if ((x >>> t) & 1) vNaive++; t++; }
        }
        // popcount on the packed BigInt
        const vFast = popcountBig(Qcodes[i].big ^ Kcodes[j].big);
        const wN = R - (BigInt(vNaive) < R ? BigInt(vNaive) : R) + 1n;
        const wP = R - (BigInt(vFast) < R ? BigInt(vFast) : R) + 1n;
        comparisons += 2;
        if (vNaive !== vFast) mismatches++;
        if (wN !== wP) mismatches++;
        if (minWeight === null || wN < minWeight) minWeight = wN;
        denN[i] += wN; denP[i] += wP;
        for (let g = 0; g < dv; g++) { numN[i][g] += wN * V[j][g]; numP[i][g] += wP * V[j][g]; }
      }
      rows++;
      comparisons++;
      if (denN[i] !== denP[i]) mismatches++;
      for (let g = 0; g < dv; g++) { comparisons++; if (numN[i][g] !== numP[i][g]) mismatches++; }
      if (!example) {
        example = {
          N, d, dv, bits, R: R.toString(),
          denNaive: denN[i].toString(), denPopcount: denP[i].toString(),
          numNaive0: numN[i][0].toString(), numPopcount0: numP[i][0].toString(),
        };
      }
    }

    // --- ternary naive dot vs four AND-popcounts ---
    const tq = [], tk = [];
    for (let j = 0; j < N; j++) {
      const lv = []; for (let f = 0; f < d; f++) lv.push(ri(-1, 1));
      tk.push(lv);
    }
    for (let i = 0; i < M; i++) {
      const lv = []; for (let f = 0; f < d; f++) lv.push(ri(-1, 1));
      tq.push(lv);
    }
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sNaive = 0;
        for (let f = 0; f < d; f++) sNaive += tq[i][f] * tk[j][f];
        let A = 0, B = 0;
        for (let f = 0; f < d; f++) {
          const a = tq[i][f], b = tk[j][f];
          if (a === 1 && b === 1) A++; else if (a === -1 && b === -1) A++;
          else if (a === 1 && b === -1) B++; else if (a === -1 && b === 1) B++;
        }
        ternaryComparisons++;
        if (sNaive !== A - B) ternaryMismatches++;
      }
    }

    // --- multi-view consensus: componentwise addition of integer weights ---
    const kViews = ri(2, 4);
    const viewW = [];
    const viewRej = [];
    for (let v = 0; v < kViews; v++) {
      const wv = new Array(N);
      const rv = new Array(N);
      for (let j = 0; j < N; j++) {
        const vv = ri(0, d * bits);
        rv[j] = vv;
        wv[j] = R - (BigInt(vv) < R ? BigInt(vv) : R) + 1n;
      }
      viewW.push(wv); viewRej.push(rv);
    }
    // naive per-view accumulation
    const denRef = new Array(M).fill(0n);
    const numRef = []; for (let i = 0; i < M; i++) numRef.push(new Array(dv).fill(0n));
    for (let v = 0; v < kViews; v++) {
      for (let i = 0; i < M; i++) {
        const wv = viewW[v];
        for (let j = 0; j < N; j++) {
          denRef[i] += wv[j];
          for (let g = 0; g < dv; g++) numRef[i][g] += wv[j] * V[j][g];
        }
      }
    }
    // fused: add weight vectors first, then aggregate once
    const fused = new Array(N).fill(0n);
    for (let v = 0; v < kViews; v++) for (let j = 0; j < N; j++) fused[j] += viewW[v][j];
    const denF = new Array(M).fill(0n);
    const numF = []; for (let i = 0; i < M; i++) numF.push(new Array(dv).fill(0n));
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        denF[i] += fused[j];
        for (let g = 0; g < dv; g++) numF[i][g] += fused[j] * V[j][g];
      }
    }
    for (let i = 0; i < M; i++) {
      fusedComparisons++;
      if (denRef[i] !== denF[i]) fusedMismatches++;
      for (let g = 0; g < dv; g++) { fusedComparisons++; if (numRef[i][g] !== numF[i][g]) fusedMismatches++; }
    }
    // order independence: shuffle the view order, re-add
    for (let v = 0; v < kViews; v++) {
      const a = ri(0, kViews - 1), b = ri(0, kViews - 1);
      const tmp = viewW[a]; viewW[a] = viewW[b]; viewW[b] = tmp;
    }
    const fused2 = new Array(N).fill(0n);
    for (let v = 0; v < kViews; v++) for (let j = 0; j < N; j++) fused2[j] += viewW[v][j];
    for (let j = 0; j < N; j++) { orderComparisons++; if (fused[j] !== fused2[j]) orderMismatches++; }
    // control: sum rejection counts then apply the God Formula (not linear)
    const rejectSum = new Array(N).fill(0n);
    for (let v = 0; v < kViews; v++) for (let j = 0; j < N; j++) rejectSum[j] += BigInt(viewRej[v][j]);
    for (let j = 0; j < N; j++) {
      const wr = R - (rejectSum[j] < R ? rejectSum[j] : R) + 1n;
      fuseThenRuleComparisons++;
      if (wr !== fused[j]) fuseThenRuleMismatches++;
    }
  }
  return {
    trials, rows, comparisons, mismatches,
    ternaryComparisons, ternaryMismatches,
    fusedComparisons, fusedMismatches,
    orderComparisons, orderMismatches,
    fuseThenRuleComparisons, fuseThenRuleMismatches,
    minWeight: minWeight === null ? null : minWeight.toString(),
    example,
  };
}
