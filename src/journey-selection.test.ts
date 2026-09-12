import { describe, expect, it } from 'bun:test';
import { selectJourneyCandidates, type JourneySelectionInput, type JourneySelectionCandidate } from './journey-selection';

const budget = { maxCandidates: 4, maxBytes: 1000, maxParseMs: 100, maxBuildMs: 100 };
function candidate(id: string, steps: string[], savedWaitMs: number[]): JourneySelectionCandidate {
  return { id, steps, savedWaitMs, resources: [] };
}
function input(): JourneySelectionInput {
  return {
    contextKey: 'editor', prefix: ['open'], budget,
    traces: [
      { id: 'a', contextKey: 'editor', steps: ['open', 'save', 'code'], count: 6, source: 'observed' },
      { id: 'b', contextKey: 'editor', steps: ['open', 'save', 'share'], count: 2, source: 'observed' },
      { id: 'c', contextKey: 'editor', steps: ['open', 'exit'], count: 2, source: 'observed' },
      { id: 'irrelevant', contextKey: 'home', steps: ['open', 'save', 'code'], count: 500, source: 'observed' },
    ],
    candidates: [candidate('save-code', ['save', 'code'], [100, 200])],
  };
}

describe('observed journey selection', () => {
  it('conditions on context and the ordered prefix and sums incremental saved wait', () => {
    const result = selectJourneyCandidates(input());
    expect(result.observedContextMass).toBe(10);
    expect(result.selected[0]?.prefixes.map(prefix => prefix.probability)).toEqual([{ numerator: 8, denominator: 10 }, { numerator: 6, denominator: 10 }]);
    expect(result.selected[0]?.expectedSavedWaitMs).toBe(200);
    const continued = selectJourneyCandidates({ ...input(), prefix: ['open', 'save'], candidates: [candidate('code', ['code'], [200])] });
    expect(continued.selected[0]?.prefixes[0]?.probability).toEqual({ numerator: 6, denominator: 8 });
    expect(continued.selected[0]?.expectedSavedWaitMs).toBe(150);
  });

  it('does not turn abduction weights or zero observations into empirical confidence', () => {
    const result = selectJourneyCandidates({ ...input(), traces: [], hints: [{ candidateId: 'save-code', weight: 0.99, source: 'abduction', evidenceIds: ['hypothesis-1'] }] });
    expect(result.selected).toHaveLength(0);
    expect(result.witnesses[0]?.decision).toBe('unobserved');
    expect(result.witnesses[0]?.prefixes[0]?.probability).toBeNull();
    expect(result.witnesses[0]?.hints[0]?.weight).toBe(0.99);
  });

  it('charges shared resource closure once and prevents duplicate prefix rewards', () => {
    const resource = { id: 'shared', bytes: 90, parseMs: 2, buildMs: 5 };
    const result = selectJourneyCandidates({ ...input(), budget: { ...budget, maxBytes: 90 }, candidates: [
      { ...candidate('code', ['save', 'code'], [100, 200]), resources: [resource] },
      { ...candidate('share', ['save', 'share'], [100, 100]), resources: [resource] },
      { ...candidate('duplicate', ['save', 'code'], [100, 200]), resources: [resource] },
    ] });
    expect(result.selected.map(item => item.candidateId)).toEqual(['code', 'share']);
    expect(result.selected[1]?.marginalSavedWaitMs).toBe(20);
    expect(result.selected[1]?.marginalCost.bytes).toBe(0);
    expect(result.witnesses.find(item => item.candidateId === 'duplicate')?.decision).toBe('no-benefit');
    expect(result.remainingBudget.maxBytes).toBe(0);
  });

  it('selects feasible positive-utility candidates within all budgets', () => {
    const result = selectJourneyCandidates({ ...input(), budget: { maxCandidates: 1, maxBytes: 50, maxParseMs: 3, maxBuildMs: 5 }, candidates: [
      { ...candidate('too-large', ['save'], [1000]), resources: [{ id: 'large', bytes: 51, parseMs: 1, buildMs: 1 }] },
      { ...candidate('fits', ['save'], [100]), resources: [{ id: 'small', bytes: 50, parseMs: 3, buildMs: 5 }] },
    ] });
    expect(result.selected.map(item => item.candidateId)).toEqual(['fits']);
    expect(result.remainingBudget).toEqual({ maxCandidates: 0, maxBytes: 0, maxParseMs: 0, maxBuildMs: 0 });
  });

  it('rejects invalid or inconsistent evidence and resource identities', () => {
    const base = input();
    expect(() => selectJourneyCandidates({ ...base, traces: [...base.traces, base.traces[0]!] })).toThrow('duplicate trace id');
    expect(() => selectJourneyCandidates({ ...base, traces: [{ id: 'bad', contextKey: 'editor', steps: ['open'], count: Number.MAX_SAFE_INTEGER + 1, source: 'observed' }] })).toThrow('safe integer');
    expect(() => selectJourneyCandidates({ ...base, candidates: [{ ...candidate('a', ['save'], [1]), resources: [{ id: 'x', bytes: 1, parseMs: 0, buildMs: 0 }] }, { ...candidate('b', ['save'], [2]), resources: [{ id: 'x', bytes: 2, parseMs: 0, buildMs: 0 }] }] })).toThrow('inconsistent costs');
    expect(() => selectJourneyCandidates({ ...base, candidates: [candidate('bad', ['save'], [])] })).toThrow('one savedWaitMs');
  });
});
