import { describe, expect, it } from 'bun:test';
import { analyzeEquatorialVoronoi, equatorialAlternatingCondition, FOUR_PI, TWO_PI } from './gauss-map-die';

describe('gauss-map die -- equatorial realization', () => {
  it('always realizes an odd number of outcomes', () => {
    for (const p of [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.3, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]]) {
      const r = analyzeEquatorialVoronoi(p);
      expect(r.solvable).toBe(true);
      expect(r.maxAreaError).not.toBeNull();
      expect(r.maxAreaError as number).toBeLessThan(1e-9);
      expect(r.gapSum as number).toBeCloseTo(TWO_PI, 9);
      expect(r.realizedAreas?.reduce((a, b) => a + b, 0)).toBeCloseTo(FOUR_PI, 9);
    }
  });

  it('realizes an even vector only when the alternating condition holds', () => {
    const solvable = [0.3, 0.2, 0.2, 0.3]; // 0.3 + 0.2 = 0.2 + 0.3
    const notSolvable = [0.4, 0.1, 0.4, 0.1]; // 0.8 != 0.2
    expect(equatorialAlternatingCondition(solvable)).toBe(true);
    expect(analyzeEquatorialVoronoi(solvable).solvable).toBe(true);
    expect(equatorialAlternatingCondition(notSolvable)).toBe(false);
    const bad = analyzeEquatorialVoronoi(notSolvable);
    expect(bad.solvable).toBe(false);
    expect(bad.gaps).toBeNull();
    expect(Math.abs(bad.alternatingResidual)).toBeGreaterThan(0.1);
  });

  it('rejects a vector that is not a probability vector', () => {
    expect(() => analyzeEquatorialVoronoi([0.5, 0.4])).toThrow();
    expect(() => analyzeEquatorialVoronoi([0.5, 0.5, 0])).toThrow();
    expect(() => analyzeEquatorialVoronoi([1])).toThrow();
  });

  it('the two-outcome equatorial solution is the degenerate pair of hemispheres', () => {
    const r = analyzeEquatorialVoronoi([0.5, 0.5]);
    expect(r.solvable).toBe(true);
    expect(r.realizedAreas).toEqual([TWO_PI, TWO_PI]);
  });
});
