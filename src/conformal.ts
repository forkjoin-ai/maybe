/**
 * conformal.ts -- Calibrated Uncertainty via Split Conformal Prediction
 *
 * Given a calibration set of (predicted, actual) observations and a new
 * prediction, compute a prediction interval [lower, upper] such that the
 * true value falls inside with probability >= confidence (asymptotically,
 * under exchangeability).
 *
 * Algorithm (Vovk, Gammerman, Shafer, 2005):
 *   1. Compute residuals r_i = |actual_i - predicted_i| over the calibration set.
 *   2. Take the ceil((n+1)(1 - (1-c)))/n = (1-alpha) empirical quantile q.
 *      (Weighted quantile when sample weights are provided.)
 *   3. Interval: [prediction - q, prediction + q].
 *
 * This is a distribution-free wrapper around any point predictor. The only
 * assumption is that the calibration and test observations are exchangeable.
 *
 * Why this belongs next to buleyeanProbability: conformal prediction
 * converts a point estimate into a set of compatible worlds. The Buleyean
 * distribution tells us how much weight each world carries; conformal
 * prediction tells us which worlds survive the evidence. Both refuse the
 * false certainty of a single number -- the sliver never collapses.
 */

/**
 * A single calibration observation: a point prediction and its ground truth.
 * outcome is binary 0/1 for classification, real-valued for regression.
 */
export interface CalibrationObservation {
  readonly predicted: number; // model's predicted value / probability
  readonly actual: number; // observed ground truth
  readonly weight?: number; // optional sample weight (default 1)
}

export interface ConformalInterval {
  readonly source: 'MaybeConformal';
  readonly predicted: number;
  readonly confidence: number; // e.g., 0.9 for 90%
  readonly lower: number;
  readonly upper: number;
  readonly calibrationSize: number;
  readonly calibrationError: number; // observed mean abs error
}

// ---------------------------------------------------------------------------
// Internal: quantile helpers
// ---------------------------------------------------------------------------

/**
 * Weighted empirical quantile over an unsorted array of (value, weight) pairs.
 * Uses the normalized cumulative-weight definition: returns the smallest value
 * whose cumulative weight fraction is >= q. Ties break toward the upper value.
 */
function weightedQuantile(
  items: readonly { value: number; weight: number }[],
  q: number
): number {
  if (items.length === 0) return 0;
  const sorted = [...items].sort((a, b) => a.value - b.value);
  const totalWeight = sorted.reduce((s, x) => s + x.weight, 0);
  if (totalWeight <= 0) return sorted[sorted.length - 1]!.value;
  const target = q * totalWeight;
  let cum = 0;
  for (const { value, weight } of sorted) {
    cum += weight;
    if (cum >= target) return value;
  }
  return sorted[sorted.length - 1]!.value;
}

// ---------------------------------------------------------------------------
// conformalInterval
// ---------------------------------------------------------------------------

/**
 * Compute a split-conformal prediction interval.
 *
 * Uses the (n+1) finite-sample correction: we take the
 * ceil((n+1) * (1 - alpha)) / n -th residual as the radius, capped at 1.
 * When n is small, this inflates the interval to preserve coverage.
 */
export function conformalInterval(input: {
  readonly calibrationSet: readonly CalibrationObservation[];
  readonly prediction: number;
  readonly confidence: number;
}): ConformalInterval {
  const { calibrationSet, prediction, confidence } = input;

  if (!(confidence > 0 && confidence < 1)) {
    throw new Error(`confidence must be in (0, 1); got ${confidence}`);
  }

  const n = calibrationSet.length;
  if (n === 0) {
    // No calibration -- cannot claim anything; return a degenerate interval
    // centered on the prediction with zero radius. The sliver (+1) in the
    // Buleyean sense is the calibrationSize field itself: it is 0, which
    // is the honest report of "we have no evidence."
    return {
      source: 'MaybeConformal',
      predicted: prediction,
      confidence,
      lower: prediction,
      upper: prediction,
      calibrationSize: 0,
      calibrationError: 0,
    };
  }

  const residuals = calibrationSet.map((obs) => ({
    value: Math.abs(obs.actual - obs.predicted),
    weight: obs.weight ?? 1,
  }));

  // Finite-sample corrected quantile level: (1 - alpha) * (n+1) / n, capped at 1.
  // Standard split-conformal correction (Vovk et al).
  const alpha = 1 - confidence;
  const rawLevel = ((n + 1) * (1 - alpha)) / n;
  const quantileLevel = Math.min(1, rawLevel);

  const radius = weightedQuantile(residuals, quantileLevel);

  const totalWeight = residuals.reduce((s, r) => s + r.weight, 0) || n;
  const calibrationError =
    residuals.reduce((s, r) => s + r.value * r.weight, 0) / totalWeight;

  return {
    source: 'MaybeConformal',
    predicted: prediction,
    confidence,
    lower: prediction - radius,
    upper: prediction + radius,
    calibrationSize: n,
    calibrationError,
  };
}

// ---------------------------------------------------------------------------
// measureCalibration
// ---------------------------------------------------------------------------

export interface CalibrationReport {
  readonly source: 'MaybeConformalCalibration';
  readonly confidence: number;
  readonly empiricalHitRate: number;
  readonly sampleSize: number;
  readonly calibrationGap: number; // |hitRate - confidence|
  readonly isWellCalibrated: boolean; // gap < threshold
}

/**
 * Measure empirical calibration: fraction of outcomes that fell inside the
 * reported intervals. A well-calibrated model has empiricalHitRate ~= confidence.
 *
 * All predictions must share the same confidence level (this is a per-level
 * calibration check -- mixing 90% and 95% intervals in one report is a
 * category error).
 */
export function measureCalibration(input: {
  readonly predictions: readonly ConformalInterval[];
  readonly outcomes: readonly number[];
  readonly threshold?: number;
}): CalibrationReport {
  const { predictions, outcomes } = input;
  const threshold = input.threshold ?? 0.05;

  if (predictions.length !== outcomes.length) {
    throw new Error(
      `predictions and outcomes must have equal length; got ${predictions.length} and ${outcomes.length}`
    );
  }

  const n = predictions.length;

  if (n === 0) {
    return {
      source: 'MaybeConformalCalibration',
      confidence: 0,
      empiricalHitRate: 0,
      sampleSize: 0,
      calibrationGap: 0,
      isWellCalibrated: false,
    };
  }

  // All predictions should share a confidence level; use the first as nominal.
  const nominal = predictions[0]!.confidence;
  for (const p of predictions) {
    if (p.confidence !== nominal) {
      throw new Error(
        `measureCalibration requires all predictions at the same confidence level; saw ${nominal} and ${p.confidence}`
      );
    }
  }

  let hits = 0;
  for (let i = 0; i < n; i++) {
    const pred = predictions[i]!;
    const y = outcomes[i]!;
    if (y >= pred.lower && y <= pred.upper) hits++;
  }

  const empiricalHitRate = hits / n;
  const calibrationGap = Math.abs(empiricalHitRate - nominal);

  return {
    source: 'MaybeConformalCalibration',
    confidence: nominal,
    empiricalHitRate,
    sampleSize: n,
    calibrationGap,
    isWellCalibrated: calibrationGap < threshold,
  };
}
