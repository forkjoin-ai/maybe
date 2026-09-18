/**
 * value-slider.mjs -- the VALUE-SIDE quality/speed slider for softmax attention.
 *
 * The score side is fixed: the exact float softmax weights A are computed once
 * and never changed.  Every regime below re-uses that SAME A and only changes
 * how the value aggregate A.V is discharged.  The question is where the
 * value-side knee is.
 *
 * Regimes
 *   EXACT    B=N          full A.V, O(M*N*dv).
 *   CHUNKED  blockSize B  keep the block around the argmax column EXACT and
 *                         replace the tail by its total mass times the global
 *                         mean value (the parent's value-side scheme).
 *   LINEAR   rank r       one global sufficient statistic
 *                         S = sum_j v_j [1, P^T k_j]^T with P a d x r
 *                         orthonormal projection.  r=d is the full affine
 *                         (soft) kernel; r<d is its rank-r truncation.
 *   TOPK     topK k       keep the k largest weights, renormalize, drop tail.
 *
 * Dependency-free: no imports at all, not even Node builtins.  Deterministic.
 *
 * Cost model (one scalar unit = one multiply-add OR one comparison):
 *   EXACT   cost = M*N*dv
 *   CHUNKED cost = N*dv + M*(N + (B+1)*dv)
 *   TOPK    cost = M*(N + k*dv)
 *   LINEAR  cost = N*dv + N*r*(d+dv) + M*r*(d+dv+1)
 * The bare value multiply-add count (mac) is reported alongside for every
 * point, so a reader can price the value matmul alone or the full operation.
 */

// ===========================================================================
// Deterministic RNG and vector helpers
// ===========================================================================

/** mulberry32: deterministic, seedable, uniform in [0,1). */
export function mulberry32(seed) {
  let a = (seed >>> 0) || 0x9e3779b9;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box-Muller from a uniform stream. */
export function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalizeInPlace(x) {
  let n = 0;
  for (let i = 0; i < x.length; i++) n += x[i] * x[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < x.length; i++) x[i] /= n;
  return x;
}

/** Unit vector in R^dim, seeded. */
export function unitVector(rng, dim) {
  const x = new Array(dim);
  for (let i = 0; i < dim; i++) x[i] = gaussian(rng);
  return normalizeInPlace(x);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// ===========================================================================
// The harder workload: long-context associative recall + needle-in-haystack
// ===========================================================================

/**
 * Long-context associative / induction recall with M multi-queries.
 *
 * assoc  : every key is a target; q_i = normalize(k_t + noise * gauss).
 *          This is the induction-head shape (key -> successor value).
 * needle : a few planted needle keys, each shadowed by several decoy keys that
 *          are noisy copies of the needle (and therefore compete for softmax
 *          mass).  Queries target the needles with larger noise.  The softmax
 *          head is diffuse, so the discarded tail carries real mass.
 */
export function makeTask(cfg) {
  const N = cfg.N;
  const d = cfg.d;
  const dv = cfg.dv;
  const M = cfg.M;
  const rng = mulberry32(cfg.seed);
  const keys = [];
  for (let j = 0; j < N; j++) keys.push(unitVector(rng, d));
  const values = [];
  for (let j = 0; j < N; j++) values.push(unitVector(rng, dv));

  const needleIdx = [];
  const decoyAt = [];
  const decoyOf = [];
  if (cfg.variant === 'needle') {
    const nNeedle = Math.min(cfg.nNeedle, N);
    for (let t = 0; t < nNeedle; t++) needlesPush(needleIdx, N, t, nNeedle);
    let cursor = 0;
    for (let t = 0; t < nNeedle; t++) {
      const base = keys[needleIdx[t]];
      let placed = 0;
      while (placed < cfg.decoys) {
        cursor = (cursor + 1) % N;
        if (needleIdx.indexOf(cursor) !== -1) continue;
        const copy = base.slice();
        for (let f = 0; f < d; f++) copy[f] += cfg.decoyNoise * gaussian(rng);
        keys[cursor] = normalizeInPlace(copy);
        decoyAt.push(cursor);
        decoyOf.push(needleIdx[t]);
        placed += 1;
      }
    }
  }

  const targets = new Array(M);
  const queries = new Array(M);
  const needles = cfg.variant === 'needle' ? needleIdx : null;
  for (let i = 0; i < M; i++) {
    let t;
    if (needles) t = needles[i % needles.length];
    else t = i % N;
    targets[i] = t;
    const q = keys[t].slice();
    for (let f = 0; f < d; f++) q[f] += cfg.noise * gaussian(rng);
    queries[i] = normalizeInPlace(q);
  }

  return {
    variant: cfg.variant,
    N,
    d,
    dv,
    M,
    beta: cfg.beta,
    seed: cfg.seed,
    keys,
    values,
    queries,
    targets,
    needleIdx,
    decoyAt,
    decoyOf,
  };
}

function needlesPush(arr, N, t, nNeedle) {
  arr.push(Math.floor(((t + 0.5) * N) / nNeedle));
}

/** Exact float softmax weights A (M x N), row-normalized. Reference. */
export function softmaxWeights(queries, keys, beta) {
  const M = queries.length;
  const N = keys.length;
  const A = new Array(M);
  for (let i = 0; i < M; i++) {
    const q = queries[i];
    const s = new Array(N);
    let mx = -Infinity;
    for (let j = 0; j < N; j++) {
      const sc = beta * dot(q, keys[j]);
      s[j] = sc;
      if (sc > mx) mx = sc;
    }
    let z = 0;
    for (let j = 0; j < N; j++) {
      s[j] = Math.exp(s[j] - mx);
      z += s[j];
    }
    for (let j = 0; j < N; j++) s[j] /= z;
    A[i] = s;
  }
  return A;
}

// ===========================================================================
// The regime implementations (all operate on the fixed A)
// ===========================================================================

export function exactValue(A, V) {
  const M = A.length;
  const dv = V[0].length;
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const row = A[i];
    const o = new Array(dv).fill(0);
    for (let j = 0; j < row.length; j++) {
      const a = row[j];
      if (a === 0) continue;
      const vj = V[j];
      for (let g = 0; g < dv; g++) o[g] += a * vj[g];
    }
    out[i] = o;
  }
  return out;
}

function columnMean(V) {
  const N = V.length;
  const dv = V[0].length;
  const m = new Array(dv).fill(0);
  for (let j = 0; j < N; j++) {
    const vj = V[j];
    for (let g = 0; g < dv; g++) m[g] += vj[g];
  }
  for (let g = 0; g < dv; g++) m[g] /= N;
  return m;
}

function argmax(row) {
  let best = -Infinity;
  let bi = 0;
  for (let j = 0; j < row.length; j++) {
    if (row[j] > best) {
      best = row[j];
      bi = j;
    }
  }
  return bi;
}

/**
 * CHUNKED B: the block of B consecutive columns containing the argmax is
 * discharged exactly; the complementary tail is replaced by (1 - blockMass)
 * times the global mean value.  B=1 is the cheapest, B=N the exact endpoint.
 */
export function chunkedWeights(A, B) {
  const N = A[0].length;
  const out = new Array(A.length);
  for (let i = 0; i < A.length; i++) {
    const row = A[i];
    const jstar = argmax(row);
    const b0 = Math.floor(jstar / B) * B;
    const b1 = Math.min(b0 + B, N);
    const w = new Array(N);
    let mL = 0;
    for (let j = b0; j < b1; j++) {
      w[j] = row[j];
      mL += row[j];
    }
    const tail = N - (b1 - b0);
    const avg = tail > 0 ? (1 - mL) / tail : 0;
    for (let j = 0; j < N; j++) if (j < b0 || j >= b1) w[j] = avg;
    out[i] = w;
  }
  return out;
}

export function aggregateChunked(A, V, B) {
  const M = A.length;
  const dv = V[0].length;
  const Vbar = columnMean(V);
  const out = new Array(M);
  let mac = V.length * dv;
  let select = 0;
  for (let i = 0; i < M; i++) {
    const row = A[i];
    const jstar = argmax(row);
    const b0 = Math.floor(jstar / B) * B;
    const b1 = Math.min(b0 + B, V.length);
    select += row.length;
    const o = new Array(dv).fill(0);
    let mL = 0;
    for (let j = b0; j < b1; j++) {
      const a = row[j];
      mL += a;
      const vj = V[j];
      for (let g = 0; g < dv; g++) o[g] += a * vj[g];
    }
    const tailMass = 1 - mL;
    for (let g = 0; g < dv; g++) o[g] += tailMass * Vbar[g];
    out[i] = o;
    mac += (b1 - b0) * dv + dv;
  }
  return {
    output: out,
    mac,
    select,
    cost: mac + M * V.length,
    weights: chunkedWeights(A, B),
  };
}

/** Stable top-k by (weight desc, index asc) -- the naive reference. */
export function topKNaive(row, k) {
  const idx = Array.from({ length: row.length }, (_, j) => j);
  idx.sort((a, b) => (row[b] - row[a]) || (a - b));
  return idx.slice(0, Math.min(k, row.length));
}

/** One-pass insertion top-k, same total order as topKNaive. */
export function topKScan(row, k) {
  const kept = [];
  for (let j = 0; j < row.length; j++) {
    const w = row[j];
    if (kept.length === k && w < row[kept[k - 1]]) continue;
    if (kept.length === k && w === row[kept[k - 1]] && j > kept[k - 1]) continue;
    let lo = 0;
    let hi = kept.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      const wm = row[kept[mid]];
      if (wm > w || (wm === w && kept[mid] < j)) lo = mid + 1;
      else hi = mid;
    }
    kept.splice(lo, 0, j);
    if (kept.length > k) kept.pop();
  }
  return kept;
}

export function topkWeights(A, k) {
  const N = A[0].length;
  const out = new Array(A.length);
  for (let i = 0; i < A.length; i++) {
    const row = A[i];
    const kept = topKNaive(row, k);
    let z = 0;
    for (let t = 0; t < kept.length; t++) z += row[kept[t]];
    const w = new Array(N).fill(0);
    for (let t = 0; t < kept.length; t++) w[kept[t]] = z > 0 ? row[kept[t]] / z : 0;
    out[i] = w;
  }
  return out;
}

export function aggregateTopK(A, V, k) {
  const M = A.length;
  const dv = V[0].length;
  const out = new Array(M);
  let mac = 0;
  let select = 0;
  for (let i = 0; i < M; i++) {
    const row = A[i];
    const kept = topKNaive(row, k);
    select += row.length;
    let z = 0;
    for (let t = 0; t < kept.length; t++) z += row[kept[t]];
    const o = new Array(dv).fill(0);
    for (let t = 0; t < kept.length; t++) {
      const w = z > 0 ? row[kept[t]] / z : 0;
      const vj = V[kept[t]];
      for (let g = 0; g < dv; g++) o[g] += w * vj[g];
    }
    out[i] = o;
    mac += kept.length * dv;
  }
  return { output: out, mac, select, cost: mac + select, weights: topkWeights(A, k) };
}

/**
 * Gram-Schmidt orthonormal columns (d x r) from a seeded Gaussian matrix.
 * For r = d the projection is orthogonal, so the r=d kernel is exactly the
 * full affine kernel 1 + q.k.
 */
export function orthonormalProjection(d, r, seed) {
  const rng = mulberry32(seed >>> 0);
  const cols = [];
  for (let c = 0; c < r; c++) {
    let v = new Array(d);
    for (let i = 0; i < d; i++) v[i] = gaussian(rng);
    for (let p = 0; p < cols.length; p++) {
      const u = cols[p];
      const s = dot(v, u);
      for (let i = 0; i < d; i++) v[i] -= s * u[i];
    }
    let n = Math.sqrt(dot(v, v));
    if (n < 1e-9) {
      v = new Array(d).fill(0);
      v[c % d] = 1;
      for (let p = 0; p < cols.length; p++) {
        const u = cols[p];
        const s = dot(v, u);
        for (let i = 0; i < d; i++) v[i] -= s * u[i];
      }
      n = Math.sqrt(dot(v, v)) || 1;
    }
    for (let i = 0; i < d; i++) v[i] /= n;
    cols.push(v);
  }
  return cols;
}

/** P^T x for P the d x r matrix whose columns are cols. */
function project(cols, x) {
  const out = new Array(cols.length);
  for (let c = 0; c < cols.length; c++) out[c] = dot(cols[c], x);
  return out;
}

/**
 * LINEAR rank r: numerator = Vsum + S_r (P^T q_i),
 *                 denom     = N + (P^T q_i) . sum_j (P^T k_j),
 * with S_r = sum_j v_j (P^T k_j)^T and kernel 1 + (P^T q).(P^T k) >= 0
 * because ||P^T x|| <= ||x|| = 1.  r = d is the full affine kernel.
 */
export function aggregateLinear(keys, values, queries, cols) {
  const N = keys.length;
  const dv = values[0].length;
  const r = cols.length;
  const pk = new Array(N);
  const zr = new Array(r).fill(0);
  for (let j = 0; j < N; j++) {
    const p = project(cols, keys[j]);
    pk[j] = p;
    for (let a = 0; a < r; a++) zr[a] += p[a];
  }
  const Sr = new Array(r);
  for (let a = 0; a < r; a++) Sr[a] = new Array(dv).fill(0);
  for (let j = 0; j < N; j++) {
    const p = pk[j];
    const vj = values[j];
    for (let a = 0; a < r; a++) {
      const pa = p[a];
      if (pa === 0) continue;
      const row = Sr[a];
      for (let g = 0; g < dv; g++) row[g] += pa * vj[g];
    }
  }
  const Vsum = new Array(dv).fill(0);
  for (let j = 0; j < N; j++) {
    const vj = values[j];
    for (let g = 0; g < dv; g++) Vsum[g] += vj[g];
  }

  const M = queries.length;
  const out = new Array(M);
  const weights = new Array(M);
  for (let i = 0; i < M; i++) {
    const pq = project(cols, queries[i]);
    const num = Vsum.slice();
    for (let a = 0; a < r; a++) {
      const pa = pq[a];
      if (pa === 0) continue;
      const row = Sr[a];
      for (let g = 0; g < dv; g++) num[g] += pa * row[g];
    }
    let den = N;
    for (let a = 0; a < r; a++) den += pq[a] * zr[a];
    const o = new Array(dv);
    for (let g = 0; g < dv; g++) o[g] = den !== 0 ? num[g] / den : Vsum[g] / N;
    out[i] = o;
    const w = new Array(N);
    for (let j = 0; j < N; j++) {
      let ker = 1;
      const p = pk[j];
      for (let a = 0; a < r; a++) ker += pq[a] * p[a];
      w[j] = den !== 0 ? ker / den : 1 / N;
    }
    weights[i] = w;
  }
  const cost = N * dv + N * r * (keys[0].length + dv) + M * r * (keys[0].length + dv + 1);
  return { output: out, mac: cost, select: 0, cost, weights };
}

// ===========================================================================
// Metrics
// ===========================================================================

export function outputQError(outputs, exact) {
  let s = 0;
  for (let i = 0; i < outputs.length; i++) {
    let num = 0;
    let den = 0;
    for (let g = 0; g < outputs[i].length; g++) {
      const d = outputs[i][g] - exact[i][g];
      num += d * d;
      den += exact[i][g] * exact[i][g];
    }
    s += Math.sqrt(num) / Math.max(Math.sqrt(den), 1e-12);
  }
  return outputs.length > 0 ? s / outputs.length : 0;
}

/** Fraction of queries whose nearest value (by L2) is the target value index. */
export function recallValues(V, outputs, targets) {
  const dv = V[0].length;
  let hits = 0;
  for (let i = 0; i < outputs.length; i++) {
    const o = outputs[i];
    let bd = Infinity;
    let bi = -1;
    for (let j = 0; j < V.length; j++) {
      let dist = 0;
      const vj = V[j];
      for (let g = 0; g < dv; g++) {
        const diff = o[g] - vj[g];
        dist += diff * diff;
      }
      if (dist < bd) {
        bd = dist;
        bi = j;
      }
    }
    if (bi === targets[i]) hits += 1;
  }
  return outputs.length > 0 ? hits / outputs.length : 0;
}

const KL_EPS = 1e-12;

export function meanKL(P, Q) {
  let total = 0;
  for (let i = 0; i < P.length; i++) {
    let row = 0;
    for (let j = 0; j < P[i].length; j++) {
      const p = P[i][j];
      if (p > 0) row += p * Math.log(p / Math.max(Q[i][j], KL_EPS));
    }
    total += row;
  }
  return P.length > 0 ? total / P.length : 0;
}

export function meanTargetMass(W, targets) {
  let s = 0;
  for (let i = 0; i < W.length; i++) s += W[i][targets[i]];
  return W.length > 0 ? s / W.length : 0;
}

// ===========================================================================
// Pareto frontier and knee
// ===========================================================================

/** Non-dominated points in (cost asc, quality desc), duplicates folded. */
export function paretoFrontier(points) {
  const kept = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    let dominated = false;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j];
      if (q.cost <= p.cost && q.quality >= p.quality && (q.cost < p.cost || q.quality > p.quality)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      const dup = kept.findIndex((r) => r.cost === p.cost && r.quality === p.quality);
      if (dup === -1) kept.push(p);
    }
  }
  return kept.sort((a, b) => a.cost - b.cost);
}

/**
 * Elbow by maximum perpendicular distance to the chord in
 * (log10 cost, quality-error) space -- the point where quality starts falling
 * faster than cost.  errOf is the lower-is-better quantity (qerr or kl).
 */
export function kneeElbow(frontier, errOf) {
  if (frontier.length < 3) return frontier.length > 0 ? frontier[0] : null;
  const xs = frontier.map((p) => Math.log10(Math.max(p.cost, 1)));
  const ys = frontier.map(errOf);
  const x0 = Math.min.apply(null, xs);
  const x1 = Math.max.apply(null, xs);
  const y0 = Math.min.apply(null, ys);
  const y1 = Math.max.apply(null, ys);
  const sx = x1 - x0 || 1;
  const sy = y1 - y0 || 1;
  const nx = xs.map((x) => (x - x0) / sx);
  const ny = ys.map((y) => (y - y0) / sy);
  const ax = nx[0];
  const ay = ny[0];
  const bx = nx[nx.length - 1];
  const by = ny[ny.length - 1];
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy) || 1;
  let best = 1;
  let bestD = -Infinity;
  for (let i = 1; i < frontier.length - 1; i++) {
    const d = Math.abs(dy * (nx[i] - ax) - dx * (ny[i] - ay)) / len;
    if (d > bestD) {
      bestD = d;
      best = i;
    }
  }
  return frontier[best];
}

/** Cheapest frontier point with err <= tol. */
export function kneeNear(frontier, errOf, tol) {
  const ok = frontier.filter((p) => errOf(p) <= tol);
  if (ok.length === 0) return null;
  return ok.reduce((a, b) => (a.cost <= b.cost ? a : b));
}

// ===========================================================================
// TAU miss-not-lie admission
// ===========================================================================

export function missNotLieAdmit(interval, band, tau) {
  if (!interval.sound) return 'miss';
  if (!Number.isFinite(interval.low) || !Number.isFinite(interval.high)) return 'miss';
  if (interval.low - tau >= band.low && interval.high + tau <= band.high) return 'admit';
  return 'miss';
}

/**
 * Sound bracket for the CHEAP effective target mass of one query.
 *
 * CHUNKED: if the target column sits in the exact argmax block its cheap mass
 * equals the exact mass (point).  Otherwise the exact mass is only known to lie
 * in [0, droppedMass].  Both are sound brackets, no distributional assumption.
 * LINEAR: the soft kernel is a different function of the query; there is no
 * bracket on the exact softmax mass, so the interval is marked unsound and the
 * policy must MISS.  That refusal is the honest boundary.
 */
export function cheapMassInterval(row, target, keptSet) {
  if (keptSet.has(target)) return { low: row[target], high: row[target], sound: true, kept: true };
  let dropped = 0;
  for (let j = 0; j < row.length; j++) if (!keptSet.has(j)) dropped += row[j];
  return { low: 0, high: dropped, sound: true, kept: false };
}

export function chunkedKeptSet(row, B) {
  const N = row.length;
  const jstar = argmax(row);
  const b0 = Math.floor(jstar / B) * B;
  const b1 = Math.min(b0 + B, N);
  const set = new Set();
  for (let j = b0; j < b1; j++) set.add(j);
  return set;
}

export function topkKeptSet(row, k) {
  return new Set(topKNaive(row, k));
}

/** The effective cheap target mass implied by a kept/dropped split. */
export function cheapMassFor(row, target, keptSet) {
  if (keptSet.has(target)) return row[target];
  let dropped = 0;
  for (let j = 0; j < row.length; j++) if (!keptSet.has(j)) dropped += row[j];
  const rest = row.length - keptSet.size;
  return rest > 0 ? dropped / rest : 0;
}

/**
 * Run SOUND / PROXY / ORACLE admission for one cheap candidate over the M
 * queries and one tau.  Returns admit/miss/lie rates and the effective cost.
 *
 *   SOUND  : sound bracket from the kept/dropped split (deployable).
 *   PROXY  : agreement between the candidate mass and an independent cheap
 *            witness mass (deployable, can lie -- the loose-proxy boundary).
 *   ORACLE : cheap-vs-exact agreement (not deployable; the zero-lie ceiling).
 *
 * band is [m_exact - tol, m_exact + tol]; a lie is an admitted query whose
 * served mass is outside that band.
 */
export function admissionSweep(A, targets, opts) {
  const M = A.length;
  const exactCost = opts.exactCost;
  const cheapCost = opts.cheapCost;
  const witness = opts.witnessWeights || null;
  const witnessCost = opts.witnessCost || 0;
  const tol = opts.bandTol;
  const taus = opts.taus;
  const makeKept = opts.makeKept;

  const out = [];
  for (let ti = 0; ti < taus.length; ti++) {
    const tau = taus[ti];
    const acc = {
      sound: { admit: 0, miss: 0, lie: 0 },
      proxy: { admit: 0, miss: 0, lie: 0 },
      oracle: { admit: 0, miss: 0, lie: 0 },
    };
    for (let i = 0; i < M; i++) {
      const row = A[i];
      const t = targets[i];
      const mExact = row[t];
      const kept = makeKept(row);
      const band = { low: mExact - tol, high: mExact + tol };

      const iv = cheapMassInterval(row, t, kept);
      const ds = missNotLieAdmit(iv, band, tau);
      if (ds === 'admit') {
        acc.sound.admit += 1;
        if (mExact < iv.low || mExact > iv.high) acc.sound.lie += 1;
      } else acc.sound.miss += 1;

      const mCheap = cheapMassFor(row, t, kept);
      const oracle = Math.abs(mCheap - mExact) + tau <= tol;
      if (oracle) {
        acc.oracle.admit += 1;
        if (Math.abs(mCheap - mExact) > tol) acc.oracle.lie += 1;
      } else acc.oracle.miss += 1;

      if (witness) {
        const proxyTol = opts.proxyTol === undefined ? 0.2 : opts.proxyTol;
        const proxyWidth = opts.proxyWidth === undefined ? 0.02 : opts.proxyWidth;
        const ivProxy = kept.has(t)
          ? { low: mCheap, high: mCheap }
          : { low: Math.max(0, mCheap - proxyWidth), high: mCheap + proxyWidth };
        const mW = witness[i][t];
        const bandProxy = { low: mW - proxyTol, high: mW + proxyTol };
        const ok = ivProxy.low - tau >= bandProxy.low && ivProxy.high + tau <= bandProxy.high;
        if (ok) {
          acc.proxy.admit += 1;
          if (Math.abs(mCheap - mExact) > tol) acc.proxy.lie += 1;
        } else acc.proxy.miss += 1;
      }
    }
    const mk = (policy, name, costValue) => {
      const admitRate = policy.admit / M;
      const missRate = policy.miss / M;
      const lieRate = policy.lie / M;
      const eff = admitRate * costValue + missRate * exactCost;
      return {
        policy: name,
        tau,
        cheapCost: costValue,
        exactCost,
        admitRate,
        missRate,
        lieRate,
        effectiveCost: eff,
        speedup: eff > 0 ? exactCost / eff : 0,
      };
    };
    out.push(mk(acc.sound, 'SOUND', cheapCost));
    out.push(mk(acc.oracle, 'ORACLE', cheapCost));
    if (witness) out.push(mk(acc.proxy, 'PROXY', cheapCost + witnessCost));
  }
  return out;
}

// ===========================================================================
// Integer exactness
// ===========================================================================

/** Integer-valued task: integer weights and integer values, for bit-exactness. */
export function integerTask(cfg) {
  const rng = mulberry32(cfg.seed);
  const M = cfg.M;
  const N = cfg.N;
  const dv = cfg.dv;
  const W = new Array(M);
  for (let i = 0; i < M; i++) {
    const row = new Array(N);
    for (let j = 0; j < N; j++) row[j] = Math.floor(rng() * cfg.maxWeight);
    W[i] = row;
  }
  const V = new Array(N);
  for (let j = 0; j < N; j++) {
    const v = new Array(dv);
    for (let g = 0; g < dv; g++) v[g] = Math.floor(rng() * cfg.maxValue);
    V[j] = v;
  }
  return { W, V, M, N, dv };
}

/** BigInt block sums vs Number block sums; exact when magnitudes stay below 2^53. */
export function integerBlockExactness(cfg) {
  const t = integerTask(cfg);
  const M = t.M;
  const N = t.N;
  const dv = t.dv;
  const B = cfg.blockSize || 1;
  let fields = 0;
  let mismatches = 0;
  let rows = 0;
  for (let i = 0; i < M; i++) {
    const big = new Array(dv).fill(0n);
    const num = new Array(dv).fill(0);
    const w = t.W[i];
    for (let j = 0; j < N; j++) {
      const vj = t.V[j];
      for (let g = 0; g < dv; g++) {
        big[g] += BigInt(w[j]) * BigInt(vj[g]);
        num[g] += w[j] * vj[g];
      }
      if ((j + 1) % B === 0 || j === N - 1) {
        for (let g = 0; g < dv; g++) {
          fields += 1;
          if (BigInt(num[g]) !== big[g]) mismatches += 1;
        }
      }
    }
    rows += 1;
  }
  return { rows, fields, mismatches, mode: 'block B=' + B, N, dv, M };
}

/** Naive top-k selection vs one-pass insertion selection. */
export function integerTopKSelectExactness(cfg) {
  const t = integerTask(cfg);
  let fields = 0;
  let mismatches = 0;
  let rows = 0;
  const ks = cfg.ks;
  for (let i = 0; i < t.M; i++) {
    for (let ki = 0; ki < ks.length; ki++) {
      const k = Math.min(ks[ki], t.N);
      const a = topKNaive(t.W[i], k);
      const b = topKScan(t.W[i], k);
      for (let p = 0; p < k; p++) {
        fields += 1;
        if (a[p] !== b[p]) mismatches += 1;
      }
      rows += 1;
    }
  }
  return { rows, fields, mismatches, mode: 'topK selection', N: t.N, M: t.M };
}

/** BigInt top-k weighted sums vs Number, plus the full integer matmul. */
export function integerTopKSumExactness(cfg) {
  const t = integerTask(cfg);
  const dv = t.dv;
  let fields = 0;
  let mismatches = 0;
  let rows = 0;
  const k = Math.min(cfg.topK, t.N);
  for (let i = 0; i < t.M; i++) {
    const kept = topKNaive(t.W[i], k);
    const big = new Array(dv).fill(0n);
    const num = new Array(dv).fill(0);
    for (let p = 0; p < kept.length; p++) {
      const j = kept[p];
      for (let g = 0; g < dv; g++) {
        big[g] += BigInt(t.W[i][j]) * BigInt(t.V[j][g]);
        num[g] += t.W[i][j] * t.V[j][g];
      }
    }
    for (let g = 0; g < dv; g++) {
      fields += 1;
      if (BigInt(num[g]) !== big[g]) mismatches += 1;
    }
    rows += 1;
  }
  return { rows, fields, mismatches, mode: 'topK k=' + k + ' sum', N: t.N, M: t.M, dv };
}
