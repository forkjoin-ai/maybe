import { describe, expect, it } from 'bun:test';
import {
  expTable,
  tableScore,
  hammingSoftmax,
  exactSoftmax,
  buleyeanAffine,
  concentrationCeiling,
  mathExpCallCount,
  selectTier,
  prewarmPlan,
  tableFootprintBytes,
  fibonacciHashId,
  valueAggregate,
  linearSufficientStatistic,
  paretoFrontier,
  paretoSweep,
  hammingSimilarity,
  xorshift32,
  missNotLieAdmit,
  collapseRange,
  effectiveCost,
  missNotLieSweep,
  affineHammingCeiling,
  TABLE_QUALITY_PARITY_DEFF,
  TABLE_SPEEDUP_VS_FLOAT_W32,
  equalBitScoreWin,
  HOPE_TIER_ORDER,
  HOPE_TIER_CAPACITY,
  hopeTierFromBudget,
  hopeTierCapacity,
  type HopeTier,
  type ParetoPoint,
} from './quality-ladder';

/** Deterministic xorshift32 so a failing case is reproducible. */
function rng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state / 0x100000000;
  };
}

describe('quality ladder -- sealed exp table', () => {
  it('reproduces Math.exp bit-for-bit for every integer score', () => {
    for (const d of [0, 1, 8, 33, 64, 256]) {
      const table = expTable(d);
      expect(table.d).toBe(d);
      expect(table.entries.length).toBe(d + 1);
      for (let s = 0; s <= d; s++) {
        // Same double, not merely close.
        expect(table.entries[s]).toBe(Math.exp(s));
        expect(tableScore(s, table)).toBe(Math.exp(s));
        expect(table.score(s)).toBe(Math.exp(s));
      }
    }
  });

  it('is charged once and returned sealed (same object for the same d)', () => {
    const a = expTable(72);
    const b = expTable(72);
    expect(a).toBe(b);
    expect(a.state).toBe('Sealed');
    expect(a.expCalls).toBe(73);
    expect(a.bytes).toBe(tableFootprintBytes(72));
    expect(a.fibonacciHash).toBe(fibonacciHashId(72));
  });

  it('rejects out-of-range indices instead of returning undefined', () => {
    const table = expTable(4);
    expect(() => tableScore(-1, table)).toThrow();
    expect(() => tableScore(5, table)).toThrow();
    expect(() => tableScore(1.5, table)).toThrow();
  });
});

describe('quality ladder -- table softmax proves zero Math.exp at runtime', () => {
  it('makes zero counted Math.exp calls and zero native Math.exp calls', () => {
    const table = expTable(128);
    const scores = [17, 0, 128, 64, 3, 99, 128, 12];

    const countedBefore = mathExpCallCount();
    const result = hammingSoftmax(scores, table);
    expect(result.expCalls).toBe(0);
    expect(result.tableLookups).toBe(scores.length);
    expect(mathExpCallCount()).toBe(countedBefore);

    // Stronger: replace the global Math.exp for the duration of the call.
    const original = Math.exp;
    let nativeCalls = 0;
    (Math as unknown as { exp: (x: number) => number }).exp = (x: number) => {
      nativeCalls += 1;
      return original(x);
    };
    try {
      const patched = hammingSoftmax(scores, table);
      expect(patched.expCalls).toBe(0);
    } finally {
      (Math as unknown as { exp: (x: number) => number }).exp = original;
    }
    expect(nativeCalls).toBe(0);
  });

  it('matches the EXACT float softmax on integer Hamming scores', () => {
    const table = expTable(96);
    const rand = rng(0x7ab1e);
    for (let t = 0; t < 300; t++) {
      const n = 1 + Math.floor(rand() * 12);
      const scores = Array.from({ length: n }, () => Math.floor(rand() * 97));
      const fast = hammingSoftmax(scores, table).weights;
      const exact = exactSoftmax(scores);
      let sum = 0;
      for (let i = 0; i < n; i++) {
        expect(fast[i]).toBeCloseTo(exact[i], 12);
        sum += fast[i];
      }
      expect(sum).toBeCloseTo(1, 12);
    }
  });

  it('reads integer Hamming similarity in [0, d]', () => {
    const a = [1, 0, 1, 1];
    const b = [1, 1, 0, 1];
    expect(hammingSimilarity(a, b)).toBe(2);
    expect(hammingSimilarity(a, a)).toBe(4);
  });
});

describe('quality ladder -- AFFINE tier and its O(1/N) ceiling', () => {
  it('is the Buleyean ramp w = R - min(v,R) + 1 and normalizes with a floor of 1', () => {
    const scores = [1, 0, 2, 0, 3];
    const R = 4;
    const weights = buleyeanAffine(scores, R);
    let sum = 0;
    for (const w of weights) {
      expect(w).toBeGreaterThan(0);
      sum += w;
    }
    expect(sum).toBeCloseTo(1, 12);
    // The un-normalized numerator is exactly R + (R/C) s + 1 with C = max score.
    const C = 3;
    const beta = R / C;
    let raw = 0;
    for (const s of scores) raw += R - Math.min(-beta * s, R) + 1;
    for (let i = 0; i < scores.length; i++) {
      expect(weights[i] * raw).toBeCloseTo(R + beta * scores[i] + 1, 9);
    }
  });

  it('never exceeds the analytic cap (C + s_max) / (N*C + sum s)', () => {
    const scores = [1, 0, 2, 0, 3, 1, 0, 2];
    const C = 3;
    const cap = concentrationCeiling(scores, C);
    expect(cap).toBeCloseTo((C + 3) / (scores.length * C + 9), 12);
    for (const R of [0, 1, 2, 5, 10, 50, 500, 5000, 1e6]) {
      const weights = buleyeanAffine(scores, R, C);
      expect(Math.max(...weights)).toBeLessThanOrEqual(cap + 1e-12);
    }
    // The ceiling is O(1/N) and independent of the budget.
    expect(cap).toBeLessThan(0.5);
  });

  it('saturates toward the cap as R grows, so no temperature rescues it', () => {
    const scores = [1, 0, 2, 0, 3, 1, 0, 2];
    const cap = concentrationCeiling(scores, 3);
    const small = Math.max(...buleyeanAffine(scores, 1, 3));
    const large = Math.max(...buleyeanAffine(scores, 1e6, 3));
    expect(small).toBeLessThan(large);
    expect(large).toBeLessThanOrEqual(cap + 1e-12);
    expect(large).toBeGreaterThan(cap - 1e-3);
  });

  it('carries the Hamming-margin ceiling and the measured parity constants', () => {
    expect(affineHammingCeiling(0, 64)).toBeCloseTo(1 / 64, 12);
    expect(affineHammingCeiling(2, 64)).toBeCloseTo(3 / 66, 12);
    expect(TABLE_QUALITY_PARITY_DEFF).toBe(32);
    expect(equalBitScoreWin(32)).toBe(1);
    expect(equalBitScoreWin(64)).toBe(2);
    expect(TABLE_SPEEDUP_VS_FLOAT_W32.binary_d256).toBe(14);
  });
});

describe('quality ladder -- selectTier budget boundaries', () => {
  it('picks the expected tier across every HopeTier::from_budget range', () => {
    const d = 64;
    const cases: ReadonlyArray<readonly [number, 'AFFINE' | 'TABLE' | 'EXACT']> = [
      [0, 'AFFINE'],
      [64, 'AFFINE'],
      [65, 'AFFINE'],
      [512, 'AFFINE'],
      [513, 'TABLE'],
      [5_000, 'TABLE'],
      [5_001, 'EXACT'],
      [50_000, 'EXACT'],
      [50_001, 'EXACT'],
    ];
    for (const [budget, tier] of cases) {
      const sel = selectTier({ budgetKb: budget, d });
      expect(sel.tier).toBe(tier);
      expect(sel.reason).toContain(tier);
      expect(sel.tableBytes).toBe(tableFootprintBytes(d));
    }
  });

  it('downgrades under high load and refuses to charge a table that does not fit', () => {
    const loaded = selectTier({ budgetKb: 10_000, d: 64, load: 0.9 });
    expect(loaded.tier).toBe('TABLE');
    expect(loaded.reason).toContain('load');

    const tight = selectTier({ budgetKb: 513, d: 100_000 });
    expect(tight.tier).toBe('AFFINE');
    expect(tight.reason).toContain('exceeds budget');
  });

  it('raises the tier to meet targetQuality (the floor wins last)', () => {
    const table = selectTier({ budgetKb: 64, d: 64, targetQuality: 0.99 });
    expect(table.tier).toBe('TABLE');
    const exact = selectTier({ budgetKb: 64, d: 64, targetQuality: 0.995 });
    expect(exact.tier).toBe('EXACT');
    // Under load but with an explicit quality floor, the floor still holds.
    const floor = selectTier({ budgetKb: 10_000, d: 64, load: 0.95, targetQuality: 1.0 });
    expect(floor.tier).toBe('EXACT');
    expect(floor.reason).toContain('targetQuality');
  });
});

describe('quality ladder -- Hope Jar capacity bridge (ledger A2, TS side)', () => {
  it('matches the exact Lean rung capacities (LatticeLadder.rung_capacity_is_group_order)', () => {
    expect(HOPE_TIER_CAPACITY.E6).toBe(51_840n);
    expect(HOPE_TIER_CAPACITY.E7).toBe(2_903_040n);
    expect(HOPE_TIER_CAPACITY.E8).toBe(696_729_600n);
    expect(HOPE_TIER_CAPACITY.Leech).toBe(2_090_188_800n);
    // |Co0| exceeds Number.MAX_SAFE_INTEGER: 8,315,553,613,086,720,000.
    expect(HOPE_TIER_CAPACITY.Kaiju).toBe(8_315_553_613_086_720_000n);
    expect(HOPE_TIER_ORDER).toEqual(['E6', 'E7', 'E8', 'Leech', 'Kaiju']);
  });

  it('carries the Lean ladder ratios (56, 240, the Niemeier 3, the Co0 factorisation)', () => {
    expect(HOPE_TIER_CAPACITY.E7).toBe(HOPE_TIER_CAPACITY.E6 * 56n);
    expect(HOPE_TIER_CAPACITY.E8).toBe(HOPE_TIER_CAPACITY.E7 * 240n);
    expect(HOPE_TIER_CAPACITY.Leech).toBe(3n * HOPE_TIER_CAPACITY.E8);
    expect(HOPE_TIER_CAPACITY.Kaiju).toBe(24n * 196_560n * 1_762_725_888_000n);
    // Strictly increasing capacity across the ladder.
    let prev = 0n;
    for (const rung of HOPE_TIER_ORDER) {
      expect(HOPE_TIER_CAPACITY[rung]).toBeGreaterThan(prev);
      prev = HOPE_TIER_CAPACITY[rung];
    }
  });

  it('maps every budget boundary to the Rust HopeTier::from_budget rung', () => {
    const cases: ReadonlyArray<readonly [number, HopeTier]> = [
      [0, 'E6'],
      [64, 'E6'],
      [65, 'E7'],
      [512, 'E7'],
      [513, 'E8'],
      [5_000, 'E8'],
      [5_001, 'Leech'],
      [50_000, 'Leech'],
      [50_001, 'Kaiju'],
    ];
    for (const [budget, rung] of cases) {
      expect(hopeTierFromBudget(budget)).toBe(rung);
      expect(hopeTierCapacity(rung)).toBe(HOPE_TIER_CAPACITY[rung]);
      const sel = selectTier({ budgetKb: budget, d: 64 });
      expect(sel.hopeTier).toBe(rung);
      expect(sel.capacity).toBe(HOPE_TIER_CAPACITY[rung]);
    }
  });
});

describe('quality ladder -- prewarmPlan (Hope Jar shape)', () => {
  it('reports d+1 entries and 8*(d+1) bytes with the Empty->Charging->Sealed path', () => {
    for (const d of [0, 1, 64, 128, 1024]) {
      const plan = prewarmPlan(d);
      expect(plan.entries).toBe(d + 1);
      expect(plan.bytes).toBe(8 * (d + 1));
      expect(plan.bytes).toBe(tableFootprintBytes(d));
      expect(plan.kilobytes).toBe((8 * (d + 1)) / 1024);
      expect(plan.state).toBe('Sealed');
      expect(plan.sealed).toBe(true);
      expect(plan.chargeSequence).toEqual(['Empty', 'Charging', 'Sealed']);
      expect(plan.fibonacciHash).toBe(fibonacciHashId(d));
    }
  });
});

describe('quality ladder -- value side', () => {
  function attentionMatrix(M: number, N: number, rand: () => number): number[][] {
    const A: number[][] = [];
    for (let i = 0; i < M; i++) {
      const row = Array.from({ length: N }, () => rand());
      const z = row.reduce((a, b) => a + b, 0);
      A.push(row.map((x) => x / z));
    }
    return A;
  }

  function values(N: number, dv: number, rand: () => number): number[][] {
    return Array.from({ length: N }, () =>
      Array.from({ length: dv }, () => Math.floor(rand() * 7) - 3),
    );
  }

  it('B = N reproduces the exact value aggregate bit-for-bit', () => {
    const rand = rng(0x5eed);
    const M = 4, N = 16, dv = 8;
    const A = attentionMatrix(M, N, rand);
    const V = values(N, dv, rand);
    const exact = valueAggregate(A, V, { mode: 'EXACT' }).output;
    const chunked = valueAggregate(A, V, { blockSize: N }).output;
    for (let i = 0; i < M; i++) {
      for (let g = 0; g < dv; g++) expect(chunked[i][g]).toBe(exact[i][g]);
    }
  });

  it('decreasing B monotonically reduces the counted value cost', () => {
    const rand = rng(0xbee);
    const M = 8, N = 16, dv = 16;
    const A = attentionMatrix(M, N, rand);
    const V = values(N, dv, rand);
    const grid = [1, 2, 4, 8, 16];
    let prev = -Infinity;
    for (const B of grid) {
      const cost = valueAggregate(A, V, { blockSize: B }).multiplyAdds;
      expect(cost).toBeGreaterThanOrEqual(prev);
      prev = cost;
    }
    const b1 = valueAggregate(A, V, { blockSize: 1 }).multiplyAdds;
    const bN = valueAggregate(A, V, { blockSize: N }).multiplyAdds;
    expect(b1).toBeLessThan(bN);
  });

  it('the linear endpoint equals the sufficient-statistic path bit-for-bit', () => {
    const rand = rng(0x1eaf);
    for (let trial = 0; trial < 200; trial++) {
      const M = 1 + Math.floor(rand() * 4);
      const N = 2 + Math.floor(rand() * 8);
      const dF = 1 + Math.floor(rand() * 5);
      const dv = 1 + Math.floor(rand() * 4);
      const rank = 1 + Math.floor(rand() * dF);
      const qF: bigint[][] = Array.from({ length: M }, () =>
        Array.from({ length: dF }, () => BigInt(Math.floor(rand() * 5) - 2)),
      );
      const kF: bigint[][] = Array.from({ length: N }, () =>
        Array.from({ length: dF }, () => BigInt(Math.floor(rand() * 5) - 2)),
      );
      const V: bigint[][] = Array.from({ length: N }, () =>
        Array.from({ length: dv }, () => BigInt(Math.floor(rand() * 9) - 4)),
      );

      const res = linearSufficientStatistic(qF, kF, V, rank);
      expect(res.rank).toBe(rank);
      for (let i = 0; i < M; i++) {
        for (let g = 0; g < dv; g++) {
          let naive = 0n;
          for (let j = 0; j < N; j++) {
            let dot = 0n;
            for (let a = 0; a < rank; a++) dot += qF[i][a] * kF[j][a];
            naive += dot * V[j][g];
          }
          expect(res.outputExact[i][g]).toBe(naive);
        }
      }
      for (let a = 0; a < rank; a++) {
        for (let g = 0; g < dv; g++) {
          let s = 0n;
          for (let j = 0; j < N; j++) s += kF[j][a] * V[j][g];
          expect(res.statistic[a][g]).toBe(s);
        }
      }
    }
  });

  it('the valueAggregate LINEAR mode delegates to the same statistic', () => {
    const rand = rng(0xabcd);
    const M = 3, N = 6, dv = 4;
    const A = attentionMatrix(M, N, rand);
    const Vnum = values(N, dv, rand);
    const qF = Array.from({ length: M }, () => [1n, 2n, 3n]);
    const kF = Array.from({ length: N }, () => [1n, 0n, 1n]);
    const res = valueAggregate(A, Vnum, { mode: 'LINEAR', queryFeatures: qF, keyFeatures: kF });
    const direct = linearSufficientStatistic(qF, kF, Vnum.map((r) => r.map((x) => BigInt(x))));
    for (let i = 0; i < M; i++) {
      for (let g = 0; g < dv; g++) expect(res.output[i][g]).toBe(direct.output[i][g]);
    }
  });
});

describe('quality ladder -- Pareto sweep', () => {
  it('returns explicit exact and linear endpoints and a monotone frontier', () => {
    const report = paretoSweep({ N: 16, d: 8, M: 8, seed: 0x5eed });
    expect(report.exact.mode).toBe('EXACT');
    expect(report.exact.blockSize).toBe(16);
    expect(report.linear.mode).toBe('LINEAR');
    expect(report.frontier.length).toBeGreaterThan(0);

    // No frontier point is strictly dominated by another.
    for (const p of report.frontier) {
      for (const q of report.points) {
        const dominated =
          q.cost <= p.cost && q.quality >= p.quality && (q.cost < p.cost || q.quality > p.quality);
        expect(dominated).toBe(false);
      }
    }

    // Sorted by cost ascending, quality is non-decreasing (no cheap point is
    // better than a more expensive frontier point).
    for (let i = 1; i < report.frontier.length; i++) {
      expect(report.frontier[i].cost).toBeGreaterThanOrEqual(report.frontier[i - 1].cost);
      expect(report.frontier[i].quality).toBeGreaterThanOrEqual(report.frontier[i - 1].quality - 1e-12);
    }
    expect(report.note.length).toBeGreaterThan(0);
  });

  it('paretoFrontier drops dominated points and folds duplicates', () => {
    const mk = (label: string, cost: number, quality: number): ParetoPoint => ({
      label,
      mode: 'CHUNKED',
      blockSize: null,
      rank: null,
      topK: null,
      cost,
      quality,
      recall: 0,
      kl: 0,
      targetMass: 0,
    });
    const points = [mk('a', 10, 0.5), mk('b', 20, 0.9), mk('c', 30, 0.4), mk('d', 10, 0.5)];
    const front = paretoFrontier(points);
    expect(front.map((p) => p.label)).toEqual(['a', 'b']);
  });

  it('the early/mid slider keeps sharp selection at lower cost than exact', () => {
    const report = paretoSweep({ N: 16, d: 8, M: 8, seed: 0x5eed });
    const mid = report.points.find((p) => p.mode === 'CHUNKED' && p.blockSize === 8);
    expect(mid).toBeDefined();
    const m = mid as ParetoPoint;
    expect(m.cost).toBeLessThan(report.exact.cost);
    // Exact-score, exact-local-weight regime: recall matches the exact endpoint.
    expect(m.recall).toBeGreaterThanOrEqual(report.exact.recall);
    // The linear endpoint is the cheap, soft one.
    expect(report.linear.quality).toBeLessThanOrEqual(report.exact.quality);
  });
});

describe('quality ladder -- TAU miss-not-lie admission', () => {
  it('admit never serves an answer outside the band', () => {
    const rand = rng(0x0d0d);
    for (let t = 0; t < 3000; t++) {
      const a = rand();
      const b = rand();
      const band = { low: Math.min(a, b), high: Math.max(a, b) };
      const low = rand();
      const high = low + rand();
      const interval = { low, high, sound: rand() < 0.9 };
      const tau = rand() * 0.2;
      const decision = missNotLieAdmit({ interval, band, tau });
      if (decision === 'admit') {
        expect(interval.sound).toBe(true);
        expect(interval.low).toBeGreaterThanOrEqual(band.low);
        expect(interval.high).toBeLessThanOrEqual(band.high);
        expect(interval.low - tau).toBeGreaterThanOrEqual(band.low - 1e-12);
        expect(interval.high + tau).toBeLessThanOrEqual(band.high + 1e-12);
      }
    }
  });

  it('converts a perturbed cheap answer that would lie into a miss', () => {
    const band = { low: 0.4, high: 0.6 };
    const honest = { low: 0.42, high: 0.58, sound: true };
    expect(missNotLieAdmit({ interval: honest, band, tau: 0 })).toBe('admit');
    // Padding the range by tau pushes the low edge outside the band -> miss.
    expect(missNotLieAdmit({ interval: honest, band, tau: 0.05 })).toBe('miss');
    // A perturbed (would-lie) range is refused even at tau = 0.
    const liar = { low: 0.2, high: 0.55, sound: true };
    expect(missNotLieAdmit({ interval: liar, band, tau: 0 })).toBe('miss');
    // An unsound range is always a miss.
    const unsound = { low: 0.42, high: 0.58, sound: false };
    expect(missNotLieAdmit({ interval: unsound, band, tau: 0 })).toBe('miss');
  });

  it('the proven collapse range is [width/W, (R+1)*width/W]', () => {
    const weights = [3, 1, 4, 1, 5]; // W = 14, R defaults to max-1 = 4
    const range = collapseRange(weights, [4]);
    expect(range.width).toBe(1);
    expect(range.total).toBe(14);
    expect(range.budget).toBe(4);
    expect(range.low).toBeCloseTo(1 / 14, 12);
    expect(range.high).toBeCloseTo(5 / 14, 12);
    expect(range.sound).toBe(true);
    const wide = collapseRange(weights, [0, 2], 4);
    expect(wide.low).toBeCloseTo(2 / 14, 12);
    expect(wide.high).toBeCloseTo(10 / 14, 12);
  });

  it('the effective-cost identity holds exactly', () => {
    const rand = rng(0xefec);
    for (let t = 0; t < 2000; t++) {
      const admitRate = rand();
      const missRate = 1 - admitRate;
      const cheap = Math.floor(rand() * 10000);
      const exact = cheap + Math.floor(rand() * 10000);
      const eff = effectiveCost(admitRate, missRate, cheap, exact);
      expect(eff).toBe(admitRate * cheap + missRate * exact);
    }
  });

  it('is zero-lie across the sweep while the miss rate is positive, and tau moves the cost', () => {
    const report = missNotLieSweep({ N: 16, d: 8, M: 8, seed: 0x5eed, bandTol: 0.05 });
    expect(report.maxLieRate).toBe(0);
    expect(report.frontier.length).toBeGreaterThan(0);
    expect(report.frontier.every((p) => p.lieRate === 0)).toBe(true);
    expect(Math.max(...report.points.map((p) => p.missRate))).toBeGreaterThan(0);
    expect(report.rawFrontier.length).toBeGreaterThan(0);

    // effectiveCost re-derived exactly from the reported rates.
    for (const p of report.points) {
      expect(p.effectiveCost).toBe(
        effectiveCost(p.admitRate, p.missRate, p.cheapCost, p.exactCost),
      );
    }

    // tau is the knob along frontier (b): for a fixed candidate, admissions are
    // non-increasing in tau, so effectiveCost is non-decreasing in tau.
    const chunked = report.points.filter((p) => p.label === 'CHUNKED B=8');
    expect(chunked.length).toBeGreaterThan(1);
    for (let i = 1; i < chunked.length; i++) {
      expect(chunked[i].tau).toBeGreaterThan(chunked[i - 1].tau);
      expect(chunked[i].admitRate).toBeLessThanOrEqual(chunked[i - 1].admitRate);
      expect(chunked[i].effectiveCost).toBeGreaterThanOrEqual(chunked[i - 1].effectiveCost);
    }

    // The speedup at lieRate = 0 comes from the near-exact CHUNKED tier.
    expect(report.zeroLieSpeedup).toBeGreaterThan(1);
    expect(report.best).not.toBeNull();
    // The loose AFFINE collapse range is refused, not allowed to lie.
    const affine = report.points.filter((p) => p.mode === 'AFFINE');
    expect(affine.length).toBeGreaterThan(0);
    expect(affine.every((p) => p.lieRate === 0)).toBe(true);
  });
});
