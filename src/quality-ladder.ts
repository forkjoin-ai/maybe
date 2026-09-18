/**
 * quality-ladder.ts -- the precision ladder for table-lookup softmax.
 *
 * The table-lookup softmax is a PRECISION KNOB, not just a kernel. For
 * binary/ternary Q,K the score is an integer Hamming similarity in [0, d], so
 * exp(s) has only d+1 distinct inputs and can be a PRECOMPUTED TABLE: no
 * transcendental at run time and no float score matmul. That is exactly the
 * Hope Jar shape -- charge once, seal, discharge forever -- so the three tiers
 * below sit on the same budget axis the Rust tier selector already uses.
 *
 *   EXACT   float softmax reference         (the quality reference)
 *   TABLE   table-lookup softmax over the   (near-exact at integer scores;
 *           integer Hamming scores           no Math.exp, no score matmul)
 *   AFFINE  Buleyean w = R - min(v,R) + 1    (O(N), but a proven O(1/N)
 *                                              concentration ceiling)
 *
 * This file is a pure-TS executable spec. It touches no .lean file and no Rust
 * crate; the Rust integration sketch lives in
 * distributed-inference/HOPE_JAR_QUALITY_LADDER.md.
 *
 * Value-side precision axis. The value matmul A.V is the dominant cost
 * (O(N^2 d_v)) and, unlike the score side, it has a real continuum between
 * exact and linear:
 *
 *   EXACT     B = N               full A.V
 *   CHUNKED   B in [1, N]         exact softmax weights over the top-argmax
 *                                 block of B keys + a cheap global-mean tail
 *                                 (B=N exact, B=1 approaches the linear
 *                                 sufficient statistic)
 *   LINEAR    one statistic       S = sum_j phi(k_j) v_j^T, value = phi(q_i).S
 *                                 optionally low-rank to d x r (rank slider)
 *   TOPK      k largest weights   sparse, mixed knob
 *
 * The honest line: the early/mid slider keeps exact scores and exact softmax
 * weights and only approximates the value matmul, so it can preserve sharp
 * selection while cutting the dominant cost. The cheap END replaces the kernel
 * with a soft/affine one and loses quality. The slider is a supply of
 * quality/cost points, not a free lunch.
 */

// ===========================================================================
// Measured (buleyean-hamming-attention experiment, W = 32)
// ===========================================================================
//
// 1. Sign form, not the {0,1} dot. popcount(q XOR k) = d - q.k is FALSE for
//    {0,1}^d. For sign vectors z in {-1,+1}^d the rejection is
//    v = (d - z_q . z_k)/2, so the Hamming score d - v = (d + z_q . z_k)/2 is
//    AFFINE in the signed dot product. The integer table is exact either way;
//    the sign form is the one that reduces to softmax on the quantised keys
//    up to a temperature.
// 2. The table tier works. Softmax over exact integer Hamming scores with a
//    precomputed exp table reaches quality parity from dEff ~ 32 packed bits.
//    The table is (sMax - sMin + 1) entries rebuilt per BATCH (never per pair),
//    one memory read per (i,j), and reproduces Math.exp bit-for-bit
//    (measured 0 / 22314 mismatches).
// 3. Speedups vs float softmax at W = 32 (headline; see the equal-bit caveat):
//    4-bit d=8 1.5x; binary d=32 3.5x; binary d=256 14.0x; binary d=1024 24.0x;
//    binary d=2048 27.4x. At W = 64 these are ~1.47-1.75x lower.
// 4. Equal-bit-budget caveat. At MATCHED information bits the score win is only
//    W/32 (1x at W=32, 2x at W=64). The headline ratios compare b-bit codes
//    against 32-bit floats, not against an equal-bit float baseline.
// 5. A.V boundary. Popcount removes QK^T only; A.V is still M*N*d_v dense. A
//    >=5x total win (W=32) needs ambient d > ~64 at 4-bit or d > ~512 binary,
//    because the shared value term dominates at small d.
// 6. The AFFINE tier is not a concentration rule. The exact achievable max
//    target mass over any budget R is (delta+1)/(delta+N) with delta the
//    Hamming margin, which is chance at d=8; binary distance-to-match needs
//    dEff=512 bits (~20.9x ratio). The FULL method (integer weighted V)
//    matches at 4-bit from d=8 and 2-bit from d=32; binary 1-bit V never
//    matches (recall caps ~0.75).

/** Packed-bit density at which table softmax reaches quality parity. */
export const TABLE_QUALITY_PARITY_DEFF = 32;
/** The machine word the measured ratios use (W=32). */
export const TABLE_BITOP_W32 = 32;
/** Measured total speedup vs float softmax at W = 32, by code shape. */
export const TABLE_SPEEDUP_VS_FLOAT_W32 = {
  '4bit_d8': 1.5,
  binary_d32: 3.5,
  binary_d256: 14.0,
  binary_d1024: 24.0,
  binary_d2048: 27.4,
} as const;
/** At matched information bits the score win is only W/32 (1x at W=32). */
export function equalBitScoreWin(wordBits: number): number {
  return wordBits / 32;
}
/** Ambient d above which a >=5x total win is plausible (A.V still dense). */
export const AV_DOMINANCE_CROSSOVER = { fourBitD: 64, binaryD: 512 } as const;

// ===========================================================================
// Instrumentation: the only Math.exp call in this module is chargeExp, used
// once per table entry at charge time. The discharge paths never call it.
// ===========================================================================

let _mathExpCalls = 0;

/** The one and only Math.exp call site in the module (charge time only). */
function chargeExp(x: number): number {
  _mathExpCalls += 1;
  return Math.exp(x);
}

/** Total Math.exp calls made through the module's counted charge path. */
export function mathExpCallCount(): number {
  return _mathExpCalls;
}

// ===========================================================================
// Sealed exp table -- the precomputed precision knob
// ===========================================================================

/** The Fibonacci hash constant (2^64 / phi) used by the Hope Jar seed tables. */
export const FIBONACCI_HASH = 11400714819323198485n;

/** The Hope Jar box state; the table moves Empty -> Charging -> Sealed. */
export type BoxState = 'Empty' | 'Charging' | 'Sealed';

export interface SealedExpTable {
  /** The largest integer score the table covers; entries has d+1 slots. */
  readonly d: number;
  /** entries[s] === Math.exp(s) for every integer s in [0, d]. */
  readonly entries: Float64Array;
  /** entries.byteLength = 8 * (d + 1). */
  readonly bytes: number;
  readonly state: BoxState;
  /** A stable Fibonacci-hash identity for the sealed table. */
  readonly fibonacciHash: bigint;
  /** How many counted Math.exp calls were spent charging this table. */
  readonly expCalls: number;
  /** The table-local index read (same as tableScore(s, this)). */
  score(s: number): number;
}

/** The sealed-table description in the Hope Jar shape. */
export interface PrewarmPlan {
  readonly d: number;
  readonly entries: number;
  readonly bytes: number;
  readonly kilobytes: number;
  readonly fibonacciHash: bigint;
  readonly state: BoxState;
  /** Empty -> Charging -> Sealed. */
  readonly chargeSequence: readonly BoxState[];
  readonly sealed: boolean;
}

/** A stable 64-bit Fibonacci-hash identity for a table of d+1 entries. */
export function fibonacciHashId(d: number): bigint {
  const entries = BigInt(d + 1);
  return (entries * FIBONACCI_HASH + BigInt(d)) & 0xFFFFFFFFFFFFFFFFn;
}

const _tableCache = new Map<number, SealedExpTable>();

/**
 * Precompute exp(s) for every integer s in [0, d] and seal the result.
 *
 * The table is cached: the same d returns the SAME sealed object, so it is
 * literally charged once and discharged forever (the Hope Jar contract).
 */
export function expTable(d: number): SealedExpTable {
  if (!Number.isInteger(d) || d < 0) {
    throw new RangeError('expTable(d) needs a non-negative integer d; got ' + String(d));
  }
  const cached = _tableCache.get(d);
  if (cached) return cached;

  const entries = new Float64Array(d + 1);
  const before = _mathExpCalls;
  for (let s = 0; s <= d; s++) entries[s] = chargeExp(s);

  const table: SealedExpTable = {
    d,
    entries,
    bytes: entries.byteLength,
    state: 'Sealed',
    fibonacciHash: fibonacciHashId(d),
    expCalls: _mathExpCalls - before,
    score(s: number): number {
      return tableScore(s, table);
    },
  };
  _tableCache.set(d, table);
  return table;
}

/**
 * A pure index read: entries[s]. No Math.exp, no branch on transcendental.
 * Throws (not silently undefined) for a non-integer or out-of-range index.
 */
export function tableScore(s: number, table: SealedExpTable): number {
  if (!Number.isInteger(s) || s < 0 || s > table.d) {
    throw new RangeError(
      'tableScore: index ' + String(s) + ' outside the sealed table [0, ' + table.d + ']',
    );
  }
  return table.entries[s];
}

/** The sealed-table description in the Hope Jar shape (entries, bytes, hash). */
export function prewarmPlan(d: number): PrewarmPlan {
  const entries = d + 1;
  const bytes = 8 * entries;
  return {
    d,
    entries,
    bytes,
    kilobytes: bytes / 1024,
    fibonacciHash: fibonacciHashId(d),
    state: 'Sealed',
    chargeSequence: ['Empty', 'Charging', 'Sealed'],
    sealed: true,
  };
}

// ===========================================================================
// Score side: the three quality tiers
// ===========================================================================

export const QUALITY_TIER_ORDER = ['AFFINE', 'TABLE', 'EXACT'] as const;
export type QualityTier = (typeof QUALITY_TIER_ORDER)[number];

/** The measured quality each tier targets (documented, not a proof). */
export const AFFINE_TARGET = 0.5;
export const TABLE_TARGET = 0.99;
export const EXACT_TARGET = 1.0;
/** Normalized contention at or above which the load override downgrades. */
export const HIGH_LOAD = 0.85;

export function tierQuality(tier: QualityTier): number {
  switch (tier) {
    case 'EXACT':
      return EXACT_TARGET;
    case 'TABLE':
      return TABLE_TARGET;
    default:
      return AFFINE_TARGET;
  }
}

export function tierRank(tier: QualityTier): number {
  return QUALITY_TIER_ORDER.indexOf(tier);
}

/** Table footprint in bytes: one Float64 per integer score in [0, d]. */
export function tableFootprintBytes(d: number): number {
  return 8 * (d + 1);
}

/** Table footprint in kibibytes (binary KB, matching the Hope Jar budget unit). */
export function tableFootprintKb(d: number): number {
  return tableFootprintBytes(d) / 1024;
}

export interface TableSoftmaxResult {
  readonly weights: number[];
  /** Always 0 for a correct table path; the counter proves it. */
  readonly expCalls: number;
  readonly tableLookups: number;
}

/**
 * Softmax over integer Hamming scores using ONLY sealed-table reads.
 *
 * Every score is shifted UP by (table.d - max) so the largest score indexes
 * table.d and every other index is still in [0, d]. The weights are then
 * table[s + shift] / sum(...). No Math.exp, no float score matmul. This is
 * exact while exp(d) is finite (d <= 709); beyond that the table saturates to
 * Infinity exactly as Math.exp does and the caller must use EXACT.
 */
export function hammingSoftmax(
  scores: readonly number[],
  table: SealedExpTable,
): TableSoftmaxResult {
  const before = _mathExpCalls;
  const n = scores.length;
  if (n === 0) return { weights: [], expCalls: 0, tableLookups: 0 };
  if (!Number.isFinite(table.entries[table.d])) {
    throw new RangeError(
      'hammingSoftmax: exp(' + table.d + ') is not finite; the table path is exact only for d <= 709',
    );
  }

  let smax = -Infinity;
  for (let i = 0; i < n; i++) {
    const s = scores[i];
    if (!Number.isInteger(s) || s < 0 || s > table.d) {
      throw new RangeError('hammingSoftmax: score ' + String(s) + ' outside [0, ' + table.d + ']');
    }
    if (s > smax) smax = s;
  }
  const shift = table.d - smax;

  const weights = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = tableScore(scores[i] + shift, table);
    weights[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) weights[i] = weights[i] / sum;
  return { weights, expCalls: _mathExpCalls - before, tableLookups: n };
}

/** The EXACT float reference: max-shifted softmax with real Math.exp calls. */
export function exactSoftmax(scores: readonly number[]): number[] {
  const n = scores.length;
  if (n === 0) return [];
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (scores[i] > m) m = scores[i];
  const weights = new Array<number>(n);
  let z = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(scores[i] - m);
    weights[i] = e;
    z += e;
  }
  for (let i = 0; i < n; i++) weights[i] = weights[i] / z;
  return weights;
}

/**
 * The AFFINE tier: the Buleyean linear ramp w_j = R - min(v_j, R) + 1 with the
 * softmax-aligned void v_j = -beta * s_j (so the ramp is increasing in the
 * score). C is the score bound; beta = R / C. With C = max(scores) the family
 * saturates at the analytic ceiling below. Returns normalized weights.
 */
export function buleyeanAffine(scores: readonly number[], R: number, C?: number): number[] {
  const n = scores.length;
  if (n === 0) return [];
  let smax = 0;
  for (let i = 0; i < n; i++) if (scores[i] > smax) smax = scores[i];
  const bound = C !== undefined && C > 0 ? C : smax > 0 ? smax : 1;
  const beta = R / bound;

  const weights = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = -beta * scores[i];
    const w = R - Math.min(v, R) + 1;
    weights[i] = w;
    sum += w;
  }
  for (let i = 0; i < n; i++) weights[i] = weights[i] / sum;
  return weights;
}

/**
 * The analytic O(1/N) concentration ceiling of the affine family,
 *
 *   (C + s_max) / (N*C + sum_j s_j),
 *
 * which the normalized affine weight of ANY key is bounded by. It is
 * independent of the budget R, which is why no temperature rescues the affine
 * rule (see experiments/softmax-vs-buleyean-attention/README.md).
 */
export function concentrationCeiling(scores: readonly number[], C?: number): number {
  const n = scores.length;
  if (n === 0) return 1;
  let smax = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (scores[i] > smax) smax = scores[i];
    sum += scores[i];
  }
  const bound = C !== undefined && C > 0 ? C : smax > 0 ? smax : 1;
  return (bound + smax) / (n * bound + sum);
}

/**
 * The exact achievable max target mass of the affine/Hamming rule over ANY
 * budget R: (delta + 1) / (delta + N), where delta is the Hamming margin (the
 * gap between the target rejection count and the best rival). This is the
 * Hamming-margin form of the ceiling above; at delta = 0 it is 1/N (chance),
 * which is why the AFFINE tier cannot retrieve at d = 8.
 */
export function affineHammingCeiling(delta: number, N: number): number {
  if (N <= 0) return 0;
  return (Math.max(0, delta) + 1) / (Math.max(0, delta) + N);
}

// ===========================================================================
// Budget -> quality-tier selection
// ===========================================================================

export interface SelectTierInput {
  /** Hope Jar memory budget in KB (the same unit as HopeTier::from_budget). */
  readonly budgetKb: number;
  /** Normalized contention in [0, inf); >= HIGH_LOAD downgrades one tier. */
  readonly load?: number;
  /** Required quality in (0, 1]; raises the tier when the budget is too low. */
  readonly targetQuality?: number;
  /** The integer Hamming score range (0..d), which sizes the sealed table. */
  readonly d: number;
}

export interface TierSelection {
  readonly tier: QualityTier;
  /** The Hope Jar rung selected by the same budget boundaries (Lean ladder). */
  readonly hopeTier: HopeTier;
  /** The exact latent key capacity of that rung (Lean SSOT, bigint key count). */
  readonly capacity: bigint;
  readonly tableFootprintKb: number;
  readonly tableBytes: number;
  readonly budgetKb: number;
  readonly load: number;
  readonly targetQuality: number | null;
  readonly reason: string;
}

// ===========================================================================
// Hope Jar capacity axis -- the Lean ladder bridge (ledger A2, TS side)
// ===========================================================================
//
// The budget boundaries below are shared with the Hope Jar rung selector; this
// exposes the rung itself so the TS boundary can be checked against the Lean
// capacities. The SSOT is Gnosis/LatticeLadder.lean
// (`rung_capacity_is_group_order`):
//
//   E6    |W(E6)|       = 51,840
//   E7    |W(E7)|       = 2,903,040        = E6 * 56
//   E8    |W(E8)|       = 696,729,600      = E7 * 240
//   Leech 3 * |W(E8)|   = 2,090,188,800
//   Kaiju |Co0|         = 8,315,553,613,086,720,000
//                       = 24 * 196,560 * 1,762,725,888,000
//
// The capacity is a latent KEY COUNT, not the resident KB budget, and the Kaiju
// rung exceeds Number.MAX_SAFE_INTEGER (9,007,199,254,740,991), so it is carried
// as bigint. The quality tier (AFFINE/TABLE/EXACT) is a separate, deliberately
// different axis over the same boundaries -- this bridge makes the two axes
// share ONE rung map instead of duplicating the boundaries.

export const HOPE_TIER_ORDER = ['E6', 'E7', 'E8', 'Leech', 'Kaiju'] as const;
export type HopeTier = (typeof HOPE_TIER_ORDER)[number];

/** The exact rung capacities from Gnosis/LatticeLadder.lean (Lean SSOT). */
export const HOPE_TIER_CAPACITY: { readonly [T in HopeTier]: bigint } = {
  E6: 51_840n,
  E7: 2_903_040n,
  E8: 696_729_600n,
  Leech: 2_090_188_800n,
  Kaiju: 8_315_553_613_086_720_000n,
};

/**
 * The Hope Jar rung for a budget in KB. Same boundaries as
 * HopeTier::from_budget (Rust) and HopeJarE6/E7/Leech/Kaiju (Lean).
 */
export function hopeTierFromBudget(budgetKb: number): HopeTier {
  if (budgetKb <= 64) return 'E6';
  if (budgetKb <= 512) return 'E7';
  if (budgetKb <= 5_000) return 'E8';
  if (budgetKb <= 50_000) return 'Leech';
  return 'Kaiju';
}

/** The exact latent key capacity of a Hope Jar rung (Lean SSOT). */
export function hopeTierCapacity(tier: HopeTier): bigint {
  return HOPE_TIER_CAPACITY[tier];
}

/**
 * The default budget -> quality-tier map. It shares the rung boundaries with
 * hopeTierFromBudget (0..=64 E6, 65..=512 E7, 513..=5_000 E8,
 * 5_001..=50_000 Leech, >50_000 Kaiju) but reads them as a PRECISION budget:
 * the small E6/E7 jars discharge too little to charge a table, the standard E8
 * jar affords TABLE, the unconstrained Leech/Kaiju tiers afford the EXACT
 * float reference.
 */
function budgetTier(budgetKb: number): QualityTier {
  switch (hopeTierFromBudget(budgetKb)) {
    case 'E6':
    case 'E7':
      return 'AFFINE';
    case 'E8':
      return 'TABLE';
    default:
      return 'EXACT';
  }
}

function tierForQuality(q: number): QualityTier {
  if (q <= AFFINE_TARGET) return 'AFFINE';
  if (q <= TABLE_TARGET) return 'TABLE';
  return 'EXACT';
}

/**
 * Select a QUALITY tier from budget, load and targetQuality.
 *
 * Order of authority: the budget default, then a high-load downgrade, then a
 * targetQuality upgrade. The targetQuality floor wins last so a caller that
 * asks for EXACT quality is never silently served AFFINE.
 */
export function selectTier(input: SelectTierInput): TierSelection {
  const load = input.load ?? 0;
  const targetQuality = input.targetQuality ?? null;
  const bytes = tableFootprintBytes(input.d);
  const kb = bytes / 1024;

  let tier = budgetTier(input.budgetKb);
  let reason = 'budget ' + input.budgetKb + ' KB -> ' + tier;

  if (tier === 'TABLE' && kb > input.budgetKb) {
    tier = 'AFFINE';
    reason += '; table footprint ' + kb.toFixed(3) + ' KB exceeds budget -> AFFINE';
  }

  if (load >= HIGH_LOAD && tier !== 'AFFINE') {
    tier = QUALITY_TIER_ORDER[Math.max(0, tierRank(tier) - 1)];
    reason += '; load ' + load + ' >= ' + HIGH_LOAD + ' downgrade -> ' + tier;
  }

  if (targetQuality !== null) {
    const demanded = tierForQuality(targetQuality);
    if (tierRank(demanded) > tierRank(tier)) {
      tier = demanded;
      reason += '; targetQuality ' + targetQuality + ' raised -> ' + tier;
    }
  }

  const hopeTier = hopeTierFromBudget(input.budgetKb);
  return {
    tier,
    hopeTier,
    capacity: HOPE_TIER_CAPACITY[hopeTier],
    tableFootprintKb: kb,
    tableBytes: bytes,
    budgetKb: input.budgetKb,
    load,
    targetQuality,
    reason,
  };
}

// ===========================================================================
// Value-side precision axis
// ===========================================================================

export type ValueMode = 'EXACT' | 'CHUNKED' | 'LINEAR' | 'TOPK';

export interface ValueAggregateOptions {
  /** CHUNKED: local block size B in [1, N]. B >= N is exact. */
  readonly blockSize?: number;
  /** LINEAR: feature rank r (S truncated to d x r). */
  readonly rank?: number;
  /** TOPK: number of keys kept per query. */
  readonly topK?: number;
  /** Explicit mode; otherwise inferred from blockSize/rank/topK. */
  readonly mode?: ValueMode;
  /** LINEAR: query feature rows (integer phi(q)) for the sufficient statistic. */
  readonly queryFeatures?: readonly (readonly bigint[])[];
  /** LINEAR: key feature rows (integer phi(k)) for the sufficient statistic. */
  readonly keyFeatures?: readonly (readonly bigint[])[];
}

export interface ValueAggregateResult {
  readonly output: number[][];
  readonly mode: ValueMode;
  /** Counted floating/integer multiply-adds for the value-side work. */
  readonly multiplyAdds: number;
  readonly detail: string;
}

export interface LinearStatisticResult {
  readonly output: number[][];
  /** The exact integer form of output (BigInt), for bit-for-bit checks. */
  readonly outputExact: bigint[][];
  /** S[a][g] = sum_j phi_k(j)[a] * v(j)[g], built once, discharged forever. */
  readonly statistic: bigint[][];
  readonly featureDim: number;
  readonly rank: number;
  readonly multiplyAdds: number;
}

function exactValueAggregate(
  A: readonly (readonly number[])[],
  V: readonly (readonly number[])[],
): number[][] {
  const M = A.length;
  const dv = V.length > 0 ? V[0].length : 0;
  const out = new Array<number[]>(M);
  for (let i = 0; i < M; i++) {
    const row = A[i];
    const o = new Array<number>(dv).fill(0);
    for (let j = 0; j < row.length; j++) {
      const a = row[j];
      const vj = V[j];
      for (let g = 0; g < dv; g++) o[g] += a * vj[g];
    }
    out[i] = o;
  }
  return out;
}

/**
 * The fully linear sufficient-statistic value path:
 *
 *   S[a][g] = sum_j phi_k(j)[a] * v(j)[g]     (dF x d_v, built once)
 *   out[i]  = sum_a phi_q(i)[a] * S[a][g]      (phi_q(i) . S)
 *
 * With integer features and integer values the statistic and the output are
 * exact integers; outputExact is that literal BigInt value. rank r truncates
 * the feature dimension to dF x r (the rank slider). This is the cheap, soft
 * end of the slider: it removes the score matmul and the A.V matmul but only
 * matches sharp softmax at a feature budget m >> N (see
 * experiments/softmax-vs-kernel-attention/README.md).
 */
export function linearSufficientStatistic(
  queryFeatures: readonly (readonly bigint[])[],
  keyFeatures: readonly (readonly bigint[])[],
  values: readonly (readonly bigint[])[],
  rank?: number,
): LinearStatisticResult {
  const M = queryFeatures.length;
  const N = keyFeatures.length;
  const dF = keyFeatures.length > 0 ? keyFeatures[0].length : 0;
  const dv = values.length > 0 ? values[0].length : 0;
  const r = rank === undefined ? dF : Math.max(1, Math.min(Math.floor(rank), dF));

  const S: bigint[][] = [];
  for (let a = 0; a < r; a++) S.push(new Array<bigint>(dv).fill(0n));
  for (let j = 0; j < N; j++) {
    const kf = keyFeatures[j];
    const vj = values[j];
    for (let a = 0; a < r; a++) {
      const ka = kf[a];
      if (ka === 0n) continue;
      const Sa = S[a];
      for (let g = 0; g < dv; g++) Sa[g] += ka * vj[g];
    }
  }

  const outputExact = new Array<bigint[]>(M);
  const output = new Array<number[]>(M);
  for (let i = 0; i < M; i++) {
    const qf = queryFeatures[i];
    const row = new Array<bigint>(dv).fill(0n);
    for (let a = 0; a < r; a++) {
      const qa = qf[a];
      if (qa === 0n) continue;
      const Sa = S[a];
      for (let g = 0; g < dv; g++) row[g] += qa * Sa[g];
    }
    outputExact[i] = row;
    output[i] = row.map((x) => Number(x));
  }

  return {
    output,
    outputExact,
    statistic: S,
    featureDim: dF,
    rank: r,
    multiplyAdds: N * r * dv + M * r * dv,
  };
}

function inferValueMode(opts: ValueAggregateOptions): ValueMode {
  if (opts.mode) return opts.mode;
  if (opts.topK !== undefined) return 'TOPK';
  if (opts.blockSize !== undefined) return 'CHUNKED';
  if (opts.rank !== undefined) return 'LINEAR';
  return 'EXACT';
}

/**
 * A6 assessment (DISMISSED). The 128-block packing (`kaiju_blend_128_exact`,
 * `perfect_lane_packings`: head_dim 128 = 5*24 + 1*8) constrains the RUST
 * value table's quantizer lane layout -- Leech-24 and E8-8 lanes packed into the
 * 128-dim KV head. It is not this slider's knob. Here `blockSize B` is the
 * number of KEYS in the argmax block of the softmax weights, and this module
 * does no lane/bit packing at all (values are number/BigInt). The two are
 * different axes, so importing the packing would be a category error, not a
 * speedup. See KaijuCodebook.lean for the Rust-table packing.
 */
/**
 * Aggregate the values against a fixed attention matrix A. The same A is used
 * for every mode so the slider isolates the VALUE side.
 */
export function valueAggregate(
  A: readonly (readonly number[])[],
  V: readonly (readonly number[])[],
  opts: ValueAggregateOptions = {},
): ValueAggregateResult {
  const mode = inferValueMode(opts);
  const M = A.length;
  const N = V.length;
  const dv = V.length > 0 ? V[0].length : 0;

  if (mode === 'EXACT') {
    return {
      output: exactValueAggregate(A, V),
      mode,
      multiplyAdds: M * N * dv,
      detail: 'A.V full value matmul, O(M*N*d_v)',
    };
  }

  if (mode === 'CHUNKED') {
    const B = Math.max(1, Math.floor(opts.blockSize ?? N));
    if (B >= N) {
      return {
        output: exactValueAggregate(A, V),
        mode,
        multiplyAdds: M * N * dv,
        detail: 'B=' + B + ' >= N: exact A.V',
      };
    }
    const Vsum = new Array<number>(dv).fill(0);
    for (let j = 0; j < N; j++) {
      const vj = V[j];
      for (let g = 0; g < dv; g++) Vsum[g] += vj[g];
    }
    const Vbar = Vsum.map((x) => x / N);

    const out = new Array<number[]>(M);
    let cost = N * dv;
    for (let i = 0; i < M; i++) {
      const row = A[i];
      let jstar = 0;
      let best = -Infinity;
      for (let j = 0; j < N; j++) {
        if (row[j] > best) {
          best = row[j];
          jstar = j;
        }
      }
      const b0 = Math.floor(jstar / B) * B;
      const b1 = Math.min(b0 + B, N);
      const o = new Array<number>(dv).fill(0);
      let mL = 0;
      for (let j = b0; j < b1; j++) {
        const a = row[j];
        mL += a;
        const vj = V[j];
        for (let g = 0; g < dv; g++) o[g] += a * vj[g];
      }
      const mT = 1 - mL;
      for (let g = 0; g < dv; g++) o[g] += mT * Vbar[g];
      out[i] = o;
      cost += N + (b1 - b0) * dv + dv;
    }
    return {
      output: out,
      mode,
      multiplyAdds: cost,
      detail: 'B=' + B + ': exact softmax weights over the argmax block, global-mean tail',
    };
  }

  if (mode === 'TOPK') {
    let k = Math.max(1, Math.floor(opts.topK ?? N));
    k = Math.min(k, N);
    const out = new Array<number[]>(M);
    let cost = 0;
    for (let i = 0; i < M; i++) {
      const row = A[i];
      const order = Array.from({ length: N }, (_, j) => j).sort((a, b) => row[b] - row[a]);
      const kept = order.slice(0, k);
      let z = 0;
      for (let t = 0; t < kept.length; t++) z += row[kept[t]];
      const o = new Array<number>(dv).fill(0);
      for (let t = 0; t < kept.length; t++) {
        const j = kept[t];
        const w = row[j] / z;
        const vj = V[j];
        for (let g = 0; g < dv; g++) o[g] += w * vj[g];
      }
      out[i] = o;
      cost += N + k * dv;
    }
    return {
      output: out,
      mode,
      multiplyAdds: cost,
      detail: 'top-' + k + ' weights per query, renormalized',
    };
  }

  // LINEAR
  if (!opts.queryFeatures || !opts.keyFeatures) {
    throw new Error('valueAggregate LINEAR mode needs queryFeatures and keyFeatures');
  }
  const values = V.map((row) => row.map((x) => BigInt(Math.round(x))));
  const res = linearSufficientStatistic(opts.queryFeatures, opts.keyFeatures, values, opts.rank);
  return {
    output: res.output,
    mode: 'LINEAR',
    multiplyAdds: res.multiplyAdds,
    detail: 'S = sum_j phi(k_j) v_j^T (rank ' + res.rank + '), value = phi(q_i).S',
  };
}

// ===========================================================================
// Hamming task helpers (binary/ternary integer scores)
// ===========================================================================

/** Population count of a 32-bit integer. */
export function popcount32(x: number): number {
  let v = x - ((x >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/**
 * SCORE FORM (measured). popcount(q XOR k) = d - q.k is FALSE for {0,1}
 * vectors. For sign vectors z in {-1,+1}^d the rejection is
 * v = (d - z_q . z_k)/2, so the Hamming score d - v = (d + z_q . z_k)/2 is
 * affine in the signed dot product. The integer table is exact either way.
 */
/** Hamming distance on equal-length bit vectors. */
export function hammingDistance(a: readonly number[], b: readonly number[]): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

/** Integer Hamming similarity in [0, d]: d - Hamming distance. */
export function hammingSimilarity(a: readonly number[], b: readonly number[]): number {
  return a.length - hammingDistance(a, b);
}

/** Deterministic xorshift32 in [0, 1); zero seeds are remapped. */
export function xorshift32(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

// ===========================================================================
// Pareto sweep over the value-side slider
// ===========================================================================

export interface ParetoPoint {
  readonly label: string;
  readonly mode: ValueMode;
  readonly blockSize: number | null;
  readonly rank: number | null;
  readonly topK: number | null;
  readonly cost: number;
  /** 1 / (1 + KL(exact || method)); higher is better. */
  readonly quality: number;
  readonly recall: number;
  readonly kl: number;
  readonly targetMass: number;
}

export interface ParetoSweepOptions {
  readonly N?: number;
  readonly d?: number;
  readonly dv?: number;
  readonly M?: number;
  readonly seed?: number;
  readonly flips?: number;
  readonly blockSizes?: readonly number[];
  readonly topKs?: readonly number[];
  readonly ranks?: readonly number[];
}

export interface ParetoSweepReport {
  readonly points: readonly ParetoPoint[];
  readonly frontier: readonly ParetoPoint[];
  /** The B=N exact endpoint (sharp, full A.V). */
  readonly exact: ParetoPoint;
  /** The fully linear endpoint (cheap, soft). */
  readonly linear: ParetoPoint;
  /** The cheapest frontier point within 0.02 quality of exact. */
  readonly knee: ParetoPoint | null;
  readonly note: string;
}

const KL_EPS = 1e-12;

function meanKL(P: readonly (readonly number[])[], Q: readonly (readonly number[])[]): number {
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

function meanTargetMass(W: readonly (readonly number[])[], targets: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < W.length; i++) s += W[i][targets[i]];
  return W.length > 0 ? s / W.length : 0;
}

function recallValue(
  targets: readonly number[],
  V: readonly (readonly number[])[],
  outputs: readonly (readonly number[])[],
): number {
  const dv = V.length > 0 ? V[0].length : 0;
  let hits = 0;
  for (let i = 0; i < outputs.length; i++) {
    const o = outputs[i];
    let bd = Infinity;
    let bi = -1;
    for (let m = 0; m < V.length; m++) {
      let dist = 0;
      const vm = V[m];
      for (let g = 0; g < dv; g++) {
        const diff = o[g] - vm[g];
        dist += diff * diff;
      }
      if (dist < bd) {
        bd = dist;
        bi = m;
      }
    }
    if (bi === targets[i]) hits++;
  }
  return outputs.length > 0 ? hits / outputs.length : 0;
}

function doubling(n: number): number[] {
  const out: number[] = [];
  for (let b = 1; b <= n; b *= 2) out.push(b);
  if (out[out.length - 1] !== n) out.push(n);
  return out;
}

/** Effective weights for the CHUNKED approximation (for KL / target mass). */
function chunkedWeights(A: readonly (readonly number[])[], B: number): number[][] {
  const N = A[0].length;
  if (B >= N) return A.map((row) => row.slice());
  const out: number[][] = [];
  for (let i = 0; i < A.length; i++) {
    const row = A[i];
    let jstar = 0;
    let best = -Infinity;
    for (let j = 0; j < N; j++) {
      if (row[j] > best) {
        best = row[j];
        jstar = j;
      }
    }
    const b0 = Math.floor(jstar / B) * B;
    const b1 = Math.min(b0 + B, N);
    const w = new Array<number>(N).fill(0);
    let mL = 0;
    for (let j = b0; j < b1; j++) {
      w[j] = row[j];
      mL += row[j];
    }
    const tail = (1 - mL) / (N - (b1 - b0));
    for (let j = 0; j < N; j++) if (j < b0 || j >= b1) w[j] = tail;
    out.push(w);
  }
  return out;
}

/** Effective weights for the top-k approximation. */
function topkWeights(A: readonly (readonly number[])[], k: number): number[][] {
  const N = A[0].length;
  const out: number[][] = [];
  for (let i = 0; i < A.length; i++) {
    const row = A[i];
    const order = Array.from({ length: N }, (_, j) => j).sort((a, b) => row[b] - row[a]).slice(0, k);
    let z = 0;
    for (let t = 0; t < order.length; t++) z += row[order[t]];
    const w = new Array<number>(N).fill(0);
    for (let t = 0; t < order.length; t++) w[order[t]] = row[order[t]] / z;
    out.push(w);
  }
  return out;
}

/** Effective affine-kernel weights: w_j ~ 1 + q_i . k_j (soft). */
function affineKernelWeights(
  queries: readonly (readonly number[])[],
  keys: readonly (readonly number[])[],
): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < queries.length; i++) {
    const q = queries[i];
    const w = new Array<number>(keys.length);
    let z = 0;
    for (let j = 0; j < keys.length; j++) {
      const k = keys[j];
      let dot = 0;
      for (let f = 0; f < q.length; f++) dot += q[f] * k[f];
      const x = 1 + dot;
      w[j] = x;
      z += x;
    }
    for (let j = 0; j < keys.length; j++) w[j] = w[j] / z;
    out.push(w);
  }
  return out;
}

/**
 * Keep only the non-dominated points. A point q strictly dominates p when
 * q.cost <= p.cost and q.quality >= p.quality and at least one is strict.
 * Equal (cost, quality) duplicates are folded to one representative.
 */
export function paretoFrontier(points: readonly ParetoPoint[]): ParetoPoint[] {
  const kept: ParetoPoint[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    let dominated = false;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j];
      const betterCost = q.cost <= p.cost;
      const betterQuality = q.quality >= p.quality;
      const strict = q.cost < p.cost || q.quality > p.quality;
      if (betterCost && betterQuality && strict) {
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
 * Measure the value-side slider on a small associative-recall task and return
 * the Pareto frontier of (quality, cost).
 *
 * Task: binary keys in {0,1}^d, M queries each derived from a target key by
 * flipping flips bits, one-hot values in R^N, integer Hamming scores. Hard
 * selection is therefore checkable exactly, and the exact softmax weights are
 * discharged from a sealed exp table.
 */
export function paretoSweep(opts: ParetoSweepOptions = {}): ParetoSweepReport {
  const N = opts.N ?? 16;
  const d = opts.d ?? 8;
  const dv = opts.dv ?? N;
  const M = opts.M ?? 8;
  const rng = xorshift32(opts.seed ?? 0x5eed);
  const flips = opts.flips ?? Math.max(1, Math.floor(d / 4));

  const keys: number[][] = [];
  for (let j = 0; j < N; j++) {
    keys.push(Array.from({ length: d }, () => (rng() < 0.5 ? 1 : 0)));
  }
  const targets: number[] = [];
  const queries: number[][] = [];
  for (let i = 0; i < M; i++) {
    const t = i % N;
    const q = keys[t].slice();
    for (let f = 0; f < flips; f++) {
      const bit = Math.floor(rng() * d);
      q[bit] = q[bit] ^ 1;
    }
    targets.push(t);
    queries.push(q);
  }

  const scores = queries.map((q) => keys.map((k) => hammingSimilarity(q, k)));
  const table = expTable(d);
  const A = scores.map((row) => hammingSoftmax(row, table).weights);

  const V = Array.from({ length: N }, (_, j) =>
    Array.from({ length: dv }, (_, g) => (g === j ? 1 : 0)),
  );

  const exactOut = exactValueAggregate(A, V);
  const exactRecall = recallValue(targets, V, exactOut);
  const exactMass = meanTargetMass(A, targets);

  const phi = (x: readonly number[]): bigint[] => [1n, ...x.map((v) => BigInt(v))];
  const qF = queries.map(phi);
  const kF = keys.map(phi);
  const Vbig = V.map((row) => row.map((x) => BigInt(x)));

  const points: ParetoPoint[] = [];

  points.push({
    label: 'EXACT (B=N)',
    mode: 'EXACT',
    blockSize: N,
    rank: null,
    topK: null,
    cost: M * N * dv,
    quality: 1,
    recall: exactRecall,
    kl: 0,
    targetMass: exactMass,
  });

  const blockSizes = (opts.blockSizes ?? doubling(N)).filter((b) => b < N);
  for (let bi = 0; bi < blockSizes.length; bi++) {
    const B = blockSizes[bi];
    const res = valueAggregate(A, V, { blockSize: B });
    const w = chunkedWeights(A, B);
    const kl = meanKL(A, w);
    points.push({
      label: 'CHUNKED B=' + B,
      mode: 'CHUNKED',
      blockSize: B,
      rank: null,
      topK: null,
      cost: res.multiplyAdds,
      quality: 1 / (1 + kl),
      recall: recallValue(targets, V, res.output),
      kl,
      targetMass: meanTargetMass(w, targets),
    });
  }

  const topKs = opts.topKs ?? doubling(N);
  for (let ki = 0; ki < topKs.length; ki++) {
    const k = topKs[ki];
    const res = valueAggregate(A, V, { topK: k });
    const w = topkWeights(A, k);
    const kl = meanKL(A, w);
    points.push({
      label: 'TOPK k=' + k,
      mode: 'TOPK',
      blockSize: null,
      rank: null,
      topK: k,
      cost: res.multiplyAdds,
      quality: 1 / (1 + kl),
      recall: recallValue(targets, V, res.output),
      kl,
      targetMass: meanTargetMass(w, targets),
    });
  }

  const ranks = opts.ranks ?? [1, 2, 4, d + 1];
  const affineW = affineKernelWeights(queries, keys);
  for (let ri = 0; ri < ranks.length; ri++) {
    const r = ranks[ri];
    const res = linearSufficientStatistic(qF, kF, Vbig, r);
    const kl = meanKL(A, affineW);
    points.push({
      label: 'LINEAR rank=' + r,
      mode: 'LINEAR',
      blockSize: null,
      rank: r,
      topK: null,
      cost: res.multiplyAdds,
      quality: 1 / (1 + kl),
      recall: recallValue(targets, V, res.output),
      kl,
      targetMass: meanTargetMass(affineW, targets),
    });
  }

  const frontier = paretoFrontier(points);
  const linearPoints = points.filter((p) => p.mode === 'LINEAR').sort((a, b) => a.cost - b.cost);
  const linear = linearPoints.length > 0 ? linearPoints[0] : points[points.length - 1];
  const exact = points[0];
  const nearExact = frontier.filter((p) => p.quality >= exact.quality - 0.02);
  const knee =
    nearExact.length > 0
      ? nearExact.reduce((a, b) => (a.cost <= b.cost ? a : b))
      : frontier.length > 0
        ? frontier[0]
        : null;

  return {
    points,
    frontier,
    exact,
    linear,
    knee,
    note:
      'Quality is 1/(1+KL(exact||method)); cost is counted value-side multiply-adds. ' +
      'The exact endpoint (B=N) is sharp and full cost; the linear endpoint is cheap and soft. ' +
      'The early/mid CHUNKED points keep exact scores and exact softmax weights and only ' +
      'approximate the value matmul, so they hold sharp selection while cutting the dominant cost. ' +
      'A sharp softmax kernel cannot be recovered at m << N (see softmax-vs-kernel-attention).',
  };
}

// ===========================================================================
// TAU miss-not-lie admission -- the safety layer over both sliders
// ===========================================================================

/**
 * A sound proven range on a target mass: [width/W, (R+1)*width/W]. This is the
 * floor/ceiling bracket of the Buleyean arithmetic (every weight is at least
 * the floor 1 and at most the ceiling R+1), not a sampling confidence
 * interval. loose where the weights are unconstrained is what makes it a
 * robustness tool.
 */
export interface SoundRange {
  readonly low: number;
  readonly high: number;
  readonly sound: boolean;
  readonly width: number;
  readonly total: number;
  readonly budget: number;
}

export interface ToleranceBand {
  readonly low: number;
  readonly high: number;
}

/** Exactly two states. Refusal is a first-class output; there is no 'lie'. */
export type Admission = 'admit' | 'miss';

export interface MissNotLieInput {
  readonly interval: { readonly low: number; readonly high: number; readonly sound: boolean };
  readonly band: ToleranceBand;
  /**
   * The uncertainty pad added to the proven range before it is checked. A
   * small tau admits more (thin pad) but leaves less slack; a large tau
   * refuses more and gives a stronger no-lie guarantee.
   */
  readonly tau: number;
}

/**
 * Admit the cheap tier only when its answer is sound AND the whole proven
 * range, padded by tau, lies inside the tolerance band. Otherwise MISS
 * (abstain and recompute on the exact tier). Never a third state.
 */
export function missNotLieAdmit(input: MissNotLieInput): Admission {
  const interval = input.interval;
  const band = input.band;
  const tau = input.tau;
  if (!interval.sound) return 'miss';
  if (!Number.isFinite(interval.low) || !Number.isFinite(interval.high)) return 'miss';
  if (interval.low - tau >= band.low && interval.high + tau <= band.high) return 'admit';
  return 'miss';
}

/**
 * The proven collapse range of the Buleyean integer weights onto a target set:
 * low = width/W, high = (R+1)*width/W, sound = floor (w_i >= 1) and ceiling
 * (w_i <= R+1). budget R defaults to max(w) - 1, the tightest ceiling.
 */
export function collapseRange(
  weights: readonly number[],
  target: readonly number[],
  budget?: number,
): SoundRange {
  if (weights.length === 0) throw new RangeError('weights must be non-empty');
  if (target.length === 0) throw new RangeError('target must be non-empty');
  let total = 0;
  let maxWeight = 0;
  let floorOk = true;
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (!Number.isInteger(w) || w < 0) {
      throw new TypeError('weights must be non-negative integers');
    }
    total += w;
    if (w > maxWeight) maxWeight = w;
    if (w < 1) floorOk = false;
  }
  if (total <= 0) throw new RangeError('total weight must be positive');
  const r = budget === undefined ? Math.max(0, maxWeight - 1) : budget;
  if (!Number.isInteger(r) || r < 0) throw new TypeError('budget must be a non-negative integer');
  let width = 0;
  for (let t = 0; t < target.length; t++) {
    const idx = target[t];
    if (!Number.isInteger(idx) || idx < 0 || idx >= weights.length) {
      throw new RangeError('target index out of range');
    }
    width += 1;
  }
  const budgetOk = maxWeight <= r + 1;
  return {
    low: width / total,
    high: ((r + 1) * width) / total,
    sound: floorOk && budgetOk,
    width,
    total,
    budget: r,
  };
}

/**
 * The effective cost of an admission policy:
 *
 *   effectiveCost = admitRate * cheapCost + missRate * exactCost.
 *
 * With admitRate + missRate = 1 this is the per-query average cost. The
 * identity is exact; the test recomputes it from the rates.
 */
export function effectiveCost(
  admitRate: number,
  missRate: number,
  cheapCost: number,
  exactCost: number,
): number {
  return admitRate * cheapCost + missRate * exactCost;
}

function normalizeRows(rows: readonly (readonly number[])[]): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    let z = 0;
    for (let j = 0; j < row.length; j++) z += row[j];
    out.push(row.map((x) => (z !== 0 ? x / z : 1 / row.length)));
  }
  return out;
}

export type AdmissionMode = ValueMode | 'AFFINE';

export interface MissNotLiePoint {
  readonly label: string;
  readonly mode: AdmissionMode;
  readonly blockSize: number | null;
  readonly rank: number | null;
  readonly topK: number | null;
  readonly tau: number;
  /** Total (over M queries) cost of the cheap tier. */
  readonly cheapCost: number;
  /** Total (over M queries) cost of the exact tier. */
  readonly exactCost: number;
  readonly admitRate: number;
  readonly missRate: number;
  /** Served answers whose target mass is beyond the band. Must stay 0. */
  readonly lieRate: number;
  /** Per-query average: admitRate * cheapCost + missRate * exactCost. */
  readonly effectiveCost: number;
  readonly speedup: number;
}

export interface MissNotLieSweepOptions extends ParetoSweepOptions {
  /** Gate thresholds to sweep. Default [0, 0.005, 0.01, 0.02, 0.05, 0.1]. */
  readonly taus?: readonly number[];
  /** Half-width of the tolerance band around the exact target mass. */
  readonly bandTol?: number;
}

export interface MissNotLieSweepReport {
  readonly points: readonly MissNotLiePoint[];
  /** Frontier (b): non-dominated in (effectiveCost, lieRate), lieRate ~ 0. */
  readonly frontier: readonly MissNotLiePoint[];
  /** Frontier (a): the raw quality/cost frontier from paretoSweep. */
  readonly rawFrontier: readonly ParetoPoint[];
  /** Cheapest point with lieRate ~ 0 across every candidate and tau. */
  readonly best: MissNotLiePoint | null;
  readonly maxLieRate: number;
  readonly zeroLieSpeedup: number;
  readonly note: string;
}

function missNotLieFrontier(points: readonly MissNotLiePoint[]): MissNotLiePoint[] {
  const kept: MissNotLiePoint[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    let dominated = false;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j];
      const better =
        q.effectiveCost <= p.effectiveCost &&
        q.lieRate <= p.lieRate &&
        (q.effectiveCost < p.effectiveCost || q.lieRate < p.lieRate);
      if (better) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      const dup = kept.findIndex(
        (r) => r.effectiveCost === p.effectiveCost && r.lieRate === p.lieRate,
      );
      if (dup === -1) kept.push(p);
    }
  }
  return kept.sort((a, b) => a.effectiveCost - b.effectiveCost);
}

/**
 * Run the miss-not-lie admission over the value-side slider and the score-side
 * AFFINE tier, for a grid of tau. Every point reports admitRate, missRate,
 * lieRate and effectiveCost; frontier (b) is over (effectiveCost, lieRate)
 * with lieRate ~ 0.
 *
 * The value-side approximations are deterministic, so their target-mass
 * interval is a point [m, m] (sound by construction). The score-side AFFINE
 * tier uses the proven Buleyean collapse range; its range is loose, so it is
 * refused (miss) rather than allowed to lie.
 */
export function missNotLieSweep(opts: MissNotLieSweepOptions = {}): MissNotLieSweepReport {
  const N = opts.N ?? 16;
  const d = opts.d ?? 8;
  const dv = opts.dv ?? N;
  const M = opts.M ?? 8;
  const rng = xorshift32(opts.seed ?? 0x5eed);
  const flips = opts.flips ?? Math.max(1, Math.floor(d / 4));

  const keys: number[][] = [];
  for (let j = 0; j < N; j++) {
    keys.push(Array.from({ length: d }, () => (rng() < 0.5 ? 1 : 0)));
  }
  const targets: number[] = [];
  const queries: number[][] = [];
  for (let i = 0; i < M; i++) {
    const t = i % N;
    const q = keys[t].slice();
    for (let f = 0; f < flips; f++) {
      const bit = Math.floor(rng() * d);
      q[bit] = q[bit] ^ 1;
    }
    targets.push(t);
    queries.push(q);
  }

  const scores = queries.map((q) => keys.map((k) => hammingSimilarity(q, k)));
  const table = expTable(d);
  const A = scores.map((row) => hammingSoftmax(row, table).weights);
  const V = Array.from({ length: N }, (_, j) =>
    Array.from({ length: dv }, (_, g) => (g === j ? 1 : 0)),
  );
  const phi = (x: readonly number[]): bigint[] => [1n, ...x.map((v) => BigInt(v))];
  const qF = queries.map(phi);
  const kF = keys.map(phi);
  const Vbig = V.map((row) => row.map((x) => BigInt(x)));
  const affineW = affineKernelWeights(queries, keys);

  const exactCost = M * N * dv;
  const bandTol = opts.bandTol ?? 0.05;
  const taus = opts.taus ?? [0, 0.005, 0.01, 0.02, 0.05, 0.1];

  interface Job {
    label: string;
    mode: AdmissionMode;
    blockSize: number | null;
    rank: number | null;
    topK: number | null;
    cheapCost: number;
    weights: number[][];
    interval: 'point' | 'collapse';
    intWeights?: number[][];
  }

  const jobs: Job[] = [];
  const blockSizes = (opts.blockSizes ?? doubling(N)).filter((b) => b < N);
  for (let bi = 0; bi < blockSizes.length; bi++) {
    const B = blockSizes[bi];
    const res = valueAggregate(A, V, { blockSize: B });
    jobs.push({
      label: 'CHUNKED B=' + B,
      mode: 'CHUNKED',
      blockSize: B,
      rank: null,
      topK: null,
      cheapCost: res.multiplyAdds,
      weights: chunkedWeights(A, B),
      interval: 'point',
    });
  }
  const topKs = opts.topKs ?? doubling(N);
  for (let ki = 0; ki < topKs.length; ki++) {
    const k = topKs[ki];
    const res = valueAggregate(A, V, { topK: k });
    jobs.push({
      label: 'TOPK k=' + k,
      mode: 'TOPK',
      blockSize: null,
      rank: null,
      topK: k,
      cheapCost: res.multiplyAdds,
      weights: topkWeights(A, k),
      interval: 'point',
    });
  }
  const ranks = opts.ranks ?? [1, 2, 4, d + 1];
  for (let ri = 0; ri < ranks.length; ri++) {
    const r = ranks[ri];
    const res = linearSufficientStatistic(qF, kF, Vbig, r);
    jobs.push({
      label: 'LINEAR rank=' + r,
      mode: 'LINEAR',
      blockSize: null,
      rank: r,
      topK: null,
      cheapCost: res.multiplyAdds,
      weights: affineW,
      interval: 'point',
    });
  }

  // Score-side AFFINE: the literal Buleyean integer weights w = R - min(v,R) + 1
  // with rejection v = Hamming distance and budget R = d.
  const affineIntW: number[][] = [];
  for (let i = 0; i < M; i++) {
    const row = scores[i];
    const w = new Array<number>(N);
    for (let j = 0; j < N; j++) {
      const v = d - row[j];
      w[j] = d - Math.min(v, d) + 1;
    }
    affineIntW.push(w);
  }
  jobs.push({
    label: 'AFFINE (R=d score side)',
    mode: 'AFFINE',
    blockSize: null,
    rank: null,
    topK: null,
    cheapCost: M * N,
    weights: normalizeRows(affineIntW),
    interval: 'collapse',
    intWeights: affineIntW,
  });

  const points: MissNotLiePoint[] = [];
  for (let ji = 0; ji < jobs.length; ji++) {
    const job = jobs[ji];
    for (let ti = 0; ti < taus.length; ti++) {
      const tau = taus[ti];
      let admitted = 0;
      let missed = 0;
      let lies = 0;
      for (let i = 0; i < M; i++) {
        const t = targets[i];
        const mExact = A[i][t];
        const band: ToleranceBand = { low: mExact - bandTol, high: mExact + bandTol };
        let interval: { low: number; high: number; sound: boolean };
        if (job.interval === 'collapse' && job.intWeights) {
          interval = collapseRange(job.intWeights[i], [t], d);
        } else {
          const mCheap = job.weights[i][t];
          interval = { low: mCheap, high: mCheap, sound: true };
        }
        const decision = missNotLieAdmit({ interval, band, tau });
        if (decision === 'admit') {
          admitted += 1;
          const mCheap = job.weights[i][t];
          if (Math.abs(mCheap - mExact) > bandTol) lies += 1;
        } else {
          missed += 1;
        }
      }
      const admitRate = admitted / M;
      const missRate = missed / M;
      const lieRate = lies / M;
      const eff = effectiveCost(admitRate, missRate, job.cheapCost, exactCost);
      points.push({
        label: job.label,
        mode: job.mode,
        blockSize: job.blockSize,
        rank: job.rank,
        topK: job.topK,
        tau,
        cheapCost: job.cheapCost,
        exactCost,
        admitRate,
        missRate,
        lieRate,
        effectiveCost: eff,
        speedup: eff > 0 ? exactCost / eff : 0,
      });
    }
  }

  const zeroLie = points.filter((p) => p.lieRate <= 1e-9);
  const frontier = missNotLieFrontier(zeroLie);
  const best =
    zeroLie.length > 0
      ? zeroLie.reduce((a, b) => (a.effectiveCost <= b.effectiveCost ? a : b))
      : null;
  let maxLieRate = 0;
  for (let i = 0; i < points.length; i++) if (points[i].lieRate > maxLieRate) maxLieRate = points[i].lieRate;

  return {
    points,
    frontier,
    rawFrontier: paretoSweep(opts).frontier,
    best,
    maxLieRate,
    zeroLieSpeedup: best ? best.speedup : 0,
    note:
      'The admission converts a quality slider into a SAFE quality slider: a cheap answer is ' +
      'served only when its proven range, padded by tau, lies inside the tolerance band; otherwise ' +
      'the query MISSES and recomputes on the exact tier. You spend misses, not lies. A small tau ' +
      'admits more (lower effectiveCost) with a thinner pad; a large tau refuses more with a stronger ' +
      'no-lie guarantee. The cheap tier is not made accurate, only detectable and refusable.',
  };
}

