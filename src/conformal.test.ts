import { describe, expect, it } from 'vitest';
import {
  conformalInterval,
  measureCalibration,
  type CalibrationObservation,
  type ConformalInterval,
} from './conformal';

describe('conformalInterval', () => {
  it('returns prediction with ~zero radius on a trivially perfect calibration set', () => {
    const calibrationSet: CalibrationObservation[] = Array.from(
      { length: 50 },
      (_, i) => ({
        predicted: i,
        actual: i,
      })
    );
    const interval = conformalInterval({
      calibrationSet,
      prediction: 10,
      confidence: 0.9,
    });
    expect(interval.source).toBe('MaybeConformal');
    expect(interval.predicted).toBe(10);
    expect(interval.lower).toBeCloseTo(10, 10);
    expect(interval.upper).toBeCloseTo(10, 10);
    expect(interval.calibrationSize).toBe(50);
    expect(interval.calibrationError).toBeCloseTo(0, 10);
  });

  it('produces a 95% interval at least as wide as the 80% interval for the same calibration', () => {
    // Irregular residuals
    const calibrationSet: CalibrationObservation[] = Array.from(
      { length: 100 },
      (_, i) => ({
        predicted: 0,
        actual: Math.sin(i * 0.7) * 3 + (i % 7) * 0.4,
      })
    );

    const narrow = conformalInterval({
      calibrationSet,
      prediction: 0,
      confidence: 0.8,
    });
    const wide = conformalInterval({
      calibrationSet,
      prediction: 0,
      confidence: 0.95,
    });

    const narrowWidth = narrow.upper - narrow.lower;
    const wideWidth = wide.upper - wide.lower;

    expect(wideWidth).toBeGreaterThanOrEqual(narrowWidth);
  });

  it('widens intervals when calibration residuals have higher variance', () => {
    const tight: CalibrationObservation[] = Array.from(
      { length: 100 },
      (_, i) => ({
        predicted: 0,
        actual: (i % 2 === 0 ? 1 : -1) * 0.1,
      })
    );
    const loose: CalibrationObservation[] = Array.from(
      { length: 100 },
      (_, i) => ({
        predicted: 0,
        actual: (i % 2 === 0 ? 1 : -1) * 5,
      })
    );

    const tightInterval = conformalInterval({
      calibrationSet: tight,
      prediction: 0,
      confidence: 0.9,
    });
    const looseInterval = conformalInterval({
      calibrationSet: loose,
      prediction: 0,
      confidence: 0.9,
    });

    const tightWidth = tightInterval.upper - tightInterval.lower;
    const looseWidth = looseInterval.upper - looseInterval.lower;

    expect(looseWidth).toBeGreaterThan(tightWidth);
    expect(looseInterval.calibrationError).toBeGreaterThan(
      tightInterval.calibrationError
    );
  });

  it('handles weighted observations by emphasizing high-weight residuals', () => {
    // Small residuals with weight 1, one huge residual with weight 1000.
    const base: CalibrationObservation[] = Array.from({ length: 20 }, () => ({
      predicted: 0,
      actual: 0.01,
      weight: 1,
    }));
    const unweighted: CalibrationObservation[] = [
      ...base,
      { predicted: 0, actual: 100, weight: 1 },
    ];
    const weighted: CalibrationObservation[] = [
      ...base,
      { predicted: 0, actual: 100, weight: 1000 },
    ];

    const noWeight = conformalInterval({
      calibrationSet: unweighted,
      prediction: 0,
      confidence: 0.9,
    });
    const withWeight = conformalInterval({
      calibrationSet: weighted,
      prediction: 0,
      confidence: 0.9,
    });

    // The weighted version puts most mass on the 100 residual, so the 90%
    // quantile snaps to that large value. The unweighted version only has
    // one outlier among many and keeps the radius tight.
    const noWeightWidth = noWeight.upper - noWeight.lower;
    const withWeightWidth = withWeight.upper - withWeight.lower;
    expect(withWeightWidth).toBeGreaterThan(noWeightWidth);
  });

  it('returns degenerate interval on empty calibration set', () => {
    const interval = conformalInterval({
      calibrationSet: [],
      prediction: 7,
      confidence: 0.9,
    });
    expect(interval.calibrationSize).toBe(0);
    expect(interval.lower).toBe(7);
    expect(interval.upper).toBe(7);
  });

  it('rejects confidence outside (0, 1)', () => {
    expect(() =>
      conformalInterval({ calibrationSet: [], prediction: 0, confidence: 0 })
    ).toThrow();
    expect(() =>
      conformalInterval({ calibrationSet: [], prediction: 0, confidence: 1 })
    ).toThrow();
    expect(() =>
      conformalInterval({ calibrationSet: [], prediction: 0, confidence: 1.5 })
    ).toThrow();
  });
});

describe('measureCalibration', () => {
  it('reports hitRate ~= confidence for a well-calibrated generator', () => {
    // Manufacture predictions whose intervals cover exactly 90% of outcomes.
    const confidence = 0.9;
    const n = 200;

    const predictions: ConformalInterval[] = [];
    const outcomes: number[] = [];

    for (let i = 0; i < n; i++) {
      const prediction = 0;
      const inside = i < Math.round(n * confidence); // first 180 are hits
      const radius = 1;
      predictions.push({
        source: 'MaybeConformal',
        predicted: prediction,
        confidence,
        lower: prediction - radius,
        upper: prediction + radius,
        calibrationSize: n,
        calibrationError: 0.5,
      });
      outcomes.push(inside ? 0.5 : 5); // 0.5 is inside [-1,1]; 5 is outside
    }

    const report = measureCalibration({
      predictions,
      outcomes,
      threshold: 0.05,
    });
    expect(report.source).toBe('MaybeConformalCalibration');
    expect(report.confidence).toBe(confidence);
    expect(report.empiricalHitRate).toBeCloseTo(confidence, 5);
    expect(report.calibrationGap).toBeLessThan(0.05);
    expect(report.isWellCalibrated).toBe(true);
    expect(report.sampleSize).toBe(n);
  });

  it('reports a large calibration gap for systematically biased predictions', () => {
    // Intervals all [lower=10, upper=12], but true outcomes cluster at 0.
    // Nominal 90% confidence, empirical 0%.
    const confidence = 0.9;
    const n = 100;
    const predictions: ConformalInterval[] = Array.from({ length: n }, () => ({
      source: 'MaybeConformal',
      predicted: 11,
      confidence,
      lower: 10,
      upper: 12,
      calibrationSize: n,
      calibrationError: 11,
    }));
    const outcomes = Array.from({ length: n }, () => 0);

    const report = measureCalibration({ predictions, outcomes });
    expect(report.empiricalHitRate).toBe(0);
    expect(report.calibrationGap).toBeCloseTo(0.9, 10);
    expect(report.isWellCalibrated).toBe(false);
  });

  it('throws when predictions and outcomes have mismatched lengths', () => {
    const pred: ConformalInterval = {
      source: 'MaybeConformal',
      predicted: 0,
      confidence: 0.9,
      lower: -1,
      upper: 1,
      calibrationSize: 10,
      calibrationError: 0.5,
    };
    expect(() =>
      measureCalibration({ predictions: [pred], outcomes: [0, 0] })
    ).toThrow();
  });

  it('throws when predictions have mixed confidence levels', () => {
    const mk = (c: number): ConformalInterval => ({
      source: 'MaybeConformal',
      predicted: 0,
      confidence: c,
      lower: -1,
      upper: 1,
      calibrationSize: 10,
      calibrationError: 0.5,
    });
    expect(() =>
      measureCalibration({
        predictions: [mk(0.9), mk(0.95)],
        outcomes: [0, 0],
      })
    ).toThrow();
  });

  it('returns a zero report for empty input', () => {
    const report = measureCalibration({ predictions: [], outcomes: [] });
    expect(report.sampleSize).toBe(0);
    expect(report.empiricalHitRate).toBe(0);
    expect(report.isWellCalibrated).toBe(false);
  });
});
