import { describe, expect, it } from 'bun:test';
import { buildAbduction } from './abduction';
import { app, rankMemoryPaths, type MemoryAbductionInput } from './memory-abduction';

function fixture(): MemoryAbductionInput {
  return {
    candidates: [
      { id: 'path:old', strength: 3, signals: { query: 0.9, local: 0.8 } },
      { id: 'path:new', strength: 1, signals: { query: 0.2, local: 0.1 } },
    ],
    observed: ['query'],
    probeFeatures: ['local'],
  };
}

describe('memory abduction adapter', () => {
  it('matches the actual engine on a fixed likelihood matrix and prior', () => {
    const result = app(fixture());
    const reference = buildAbduction({
      hypotheses: [{ id: 0, name: 'path:old', prior: 0.75 }, { id: 1, name: 'path:new', prior: 0.25 }],
      features: ['query', 'local'],
      pFeatureGivenHypothesis: new Float64Array([0.9, 0.8, 0.2, 0.1]),
    }, { hardWallHi: 1, hardWallLo: 0, probeK: 3 }).abduce({ present: ['query'] });
    expect(result.ranked).toEqual(reference.survivors.map(s => ({ id: s.name, weight: s.probability, rejection: s.rejection })));
    expect(result.ranked.reduce((sum, item) => sum + item.weight, 0)).toBeCloseTo(1, 12);
    expect(result.probes).toEqual([{ id: 'path:old', feature: 'local' }, { id: 'path:new', feature: 'local' }]);
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);
  });

  it('does not force a match from priors when no evidence was observed', () => {
    expect(rankMemoryPaths({ ...fixture(), observed: [] }).ranked).toEqual([]);
    expect(rankMemoryPaths({ candidates: [], observed: ['query'] })).toEqual({ ranked: [], probes: [], indistinguishable: [] });
  });

  it('treats unspecified signals as unknown and identifies indistinguishable evidence', () => {
    const result = rankMemoryPaths({
      candidates: [{ id: 'a', strength: 1, signals: {} }, { id: 'b', strength: 1, signals: { query: 0.5 } }],
      observed: [' QUERY ', 'query'], probeFeatures: ['unknown'],
    });
    expect(result.ranked.map(s => s.weight)).toEqual([0.5, 0.5]);
    expect(result.indistinguishable).toEqual([['a', 'b']]);
    expect(result.probes).toEqual([]);
  });

  it('never converts heuristic endpoint signals into hard walls', () => {
    const result = rankMemoryPaths({ candidates: [
      { id: 'zero', strength: 1, signals: { query: 0 } },
      { id: 'one', strength: 1, signals: { query: 1 } },
    ], observed: ['query'] });
    expect(result.ranked.length).toBe(2);
    for (const item of result.ranked) {
      expect(item.weight > 0).toBe(true);
      expect(Number.isFinite(item.rejection)).toBe(true);
    }
  });

  it('does not propose probes for a singleton or identical known likelihoods', () => {
    const first = { id: 'a', strength: 1, signals: { query: 0.9, topic: 0.9 } };
    const second = { ...first, id: 'b' };
    const evidence = { observed: ['query'], probeFeatures: ['topic'] };
    expect(rankMemoryPaths({ ...evidence, candidates: [first] }).probes).toEqual([]);
    expect(rankMemoryPaths({ ...evidence, candidates: [first, second] }).probes).toEqual([]);
    const differing = { ...second, signals: { query: 0.9, topic: 0.5 } };
    expect(rankMemoryPaths({ ...evidence, candidates: [first, differing] }).probes.length > 0).toBe(true);
  });

  it('bounds probes and supports finite extreme strengths', () => {
    const result = rankMemoryPaths({ candidates: Array.from({ length: 64 }, (_, i) => ({
      id: String(i), strength: Number.MAX_VALUE, signals: { query: i / 64, local: i / 64 },
    })), observed: ['query'], probeFeatures: ['local'] });
    expect(result.probes.length).toBe(3);
    expect(result.ranked.reduce((sum, item) => sum + item.weight, 0)).toBeCloseTo(1, 12);
  });

  it('rejects invalid strengths, duplicate IDs, malformed signals and oversized requests', () => {
    for (const strength of [NaN, Infinity, 0, -1]) {
      expect(() => rankMemoryPaths({ candidates: [{ id: 'a', strength, signals: {} }], observed: [] })).toThrow();
    }
    for (const value of [NaN, Infinity, -0.1, 1.1]) {
      expect(() => rankMemoryPaths({ candidates: [{ id: 'a', strength: 1, signals: { query: value } }], observed: [] })).toThrow();
    }
    const a = fixture().candidates[0]!;
    expect(() => rankMemoryPaths({ candidates: [a, a], observed: [] })).toThrow();
    expect(() => rankMemoryPaths({ candidates: Array(65).fill(a), observed: [] })).toThrow();
    expect(() => rankMemoryPaths({ candidates: [], observed: Array.from({ length: 129 }, (_, i) => String(i)) })).toThrow();
  });
});
