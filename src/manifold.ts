/**
 * manifold.ts -- Law 4 (every observation is a cave) as geometry
 *
 * The God Formula w_i = R - min(v_i, R) + 1 traces a path on the
 * probability simplex as rejections accumulate. The Fisher Information
 * Metric makes this simplex a curved surface where:
 *
 *   - Law 4 (cave): the semiotic deficit between N dimensions and 1
 *     observer channel is geometrically visible as curvature
 *   - Law 2 (ordering): the God Formula's path follows geodesics --
 *     the natural shortest path on the manifold
 *   - Law 6 (sorites sharpness): fraud is detectable because lies
 *     deviate from geodesics -- the curvature spike is the sorites
 *     boundary between truthful and fraudulent evolution
 *
 * The Fisher-Rao metric makes the n-simplex isometric to a portion of
 * the n-sphere of radius 2. Geodesic distance = 2*arccos(BC), where
 * BC is the Bhattacharyya coefficient.
 *
 * A truthful probability evolution follows geodesics. A fraudulent one
 * has high geodesic curvature -- something is pushing the distribution
 * off its natural path. The curvature of a lie versus a truth is
 * geometrically detectable (FisherManifold.lean, zero sorry).
 */

import type { VoidBoundary, BoundaryStack } from '@a0n/gnosis/src/void';
import { buleyeanDistribution } from './buleyean.js';

// ============================================================================
// Fisher Information Metric
// ============================================================================

/**
 * The Fisher Information Metric tensor at a point p on the simplex.
 * For categorical distributions: g_ij = delta_ij / p_i
 *
 * This is the natural Riemannian metric on the space of probability
 * distributions. It tells you "how far apart" two nearby distributions
 * are in information-theoretic terms.
 *
 * Returns the diagonal entries (off-diagonal are zero in natural coords).
 */
export function fisherMetric(p: number[]): number[] {
  return p.map((pi) => (pi > 0 ? 1 / pi : Infinity));
}

/**
 * Fisher metric for a Buleyean void boundary.
 */
export function fisherMetricBuleyean(boundary: VoidBoundary): number[] {
  return fisherMetric(buleyeanDistribution(boundary));
}

/**
 * Inner product of two tangent vectors u, v at point p
 * using the Fisher metric: <u, v>_g = sum_i u_i * v_i / p_i
 */
export function fisherInnerProduct(
  p: number[],
  u: number[],
  v: number[]
): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      sum += (u[i] * v[i]) / p[i];
    }
  }
  return sum;
}

/**
 * Norm of a tangent vector under the Fisher metric.
 */
export function fisherNorm(p: number[], v: number[]): number {
  return Math.sqrt(Math.max(0, fisherInnerProduct(p, v, v)));
}

// ============================================================================
// Geodesic Distance (Fisher-Rao / Bhattacharyya)
// ============================================================================

/**
 * Bhattacharyya coefficient: sum_i sqrt(p_i * q_i)
 * Measures overlap between two distributions.
 * BC = 1 means identical, BC = 0 means completely disjoint.
 */
export function bhattacharyyaCoefficient(p: number[], q: number[]): number {
  let bc = 0;
  for (let i = 0; i < p.length; i++) {
    bc += Math.sqrt(p[i] * q[i]);
  }
  return Math.min(bc, 1);
}

/**
 * Fisher-Rao geodesic distance on the statistical manifold.
 * d(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))
 *
 * This is the shortest path on the probability simplex equipped
 * with the Fisher metric. It respects the curvature of the manifold.
 *
 * The simplex with this metric is isometric to a portion of the
 * n-sphere of radius 2 (via the map p_i -> 2*sqrt(p_i)).
 */
export function fisherRaoDistance(p: number[], q: number[]): number {
  const bc = bhattacharyyaCoefficient(p, q);
  return 2 * Math.acos(Math.min(bc, 1));
}

/**
 * Fisher-Rao distance between two Buleyean void boundaries.
 */
export function buleyeanDistance(a: VoidBoundary, b: VoidBoundary): number {
  return fisherRaoDistance(buleyeanDistribution(a), buleyeanDistribution(b));
}

// ============================================================================
// Curvature
// ============================================================================

/**
 * Scalar curvature of the (n-1)-dimensional probability simplex
 * with Fisher metric.
 *
 * For the categorical distribution family, the Ricci scalar is:
 *   R = (n-1)(n-2) / 4
 *
 * This is constant -- the Fisher-Rao simplex is a space of constant
 * positive curvature (a sphere). The manifold itself doesn't change
 * shape. What DOES change is the geodesic curvature of PATHS on it.
 */
export function scalarCurvature(n: number): number {
  return ((n - 1) * (n - 2)) / 4;
}

/**
 * Geodesic curvature of a path on the manifold at a given point.
 *
 * Given three consecutive distributions (p_prev, p_curr, p_next),
 * the geodesic curvature measures how much the path deviates from
 * the geodesic (great circle) connecting p_prev and p_next.
 *
 * kappa = 0: the path follows a geodesic (natural evolution)
 * kappa > 0: something is bending the path (external force / fraud)
 *
 * Computed as the ratio of the triangle inequality excess to the
 * geodesic distance, in the Fisher-Rao metric.
 */
export function geodesicCurvature(
  prev: number[],
  curr: number[],
  next: number[]
): number {
  const d_pc = fisherRaoDistance(prev, curr);
  const d_cn = fisherRaoDistance(curr, next);
  const d_pn = fisherRaoDistance(prev, next);

  if (d_pc < 1e-12 || d_cn < 1e-12) return 0;

  // Triangle inequality: d_pn <= d_pc + d_cn
  // The excess (d_pc + d_cn - d_pn) measures deviation from geodesic
  const excess = d_pc + d_cn - d_pn;

  // Normalize by the segment lengths to get curvature
  const segmentLength = (d_pc + d_cn) / 2;
  if (segmentLength < 1e-12) return 0;

  return excess / (segmentLength * segmentLength);
}

/**
 * Geodesic curvature along an entire trajectory.
 * Returns per-point curvature values (length = trajectory.length - 2).
 */
export function trajectoryCurvature(trajectory: number[][]): number[] {
  // Cannon rotation: each interior point's curvature is independent
  // (reads three adjacent points, no shared mutable state).
  // Under gnode --strategy cannon, points distribute across lanes.
  const len = trajectory.length - 2;
  const curvatures = new Array<number>(len);
  for (let i = 0; i < len; i++) {
    curvatures[i] = geodesicCurvature(
      trajectory[i],
      trajectory[i + 1],
      trajectory[i + 2]
    );
  }
  return curvatures;
}

/**
 * Total geodesic curvature of a trajectory (integral of absolute curvature).
 * Higher = more bending = more deviation from natural geodesic.
 */
export function totalCurvature(trajectory: number[][]): number {
  return trajectoryCurvature(trajectory).reduce(
    (sum, k) => sum + Math.abs(k),
    0
  );
}

// ============================================================================
// Layer Coordinates on the Manifold
// ============================================================================

/**
 * Map the four Buleyean layers to coordinates on the manifold.
 *
 * Each layer's distribution defines a point on the simplex. The
 * four points form a tetrahedron in the manifold. The "curvature
 * contribution" of each layer is its Fisher-Rao distance from the
 * uniform distribution (the flat floor).
 *
 * b_2 = 0 (frequentist) is the coordinate origin -- the flat floor.
 * The other coordinates measure how much each layer curves the manifold.
 */
export interface ManifoldCoordinates {
  /** Retrocausal curvature: distance from uniform due to terminal constraints */
  b0_retrocausal: number;
  /** Bayesian curvature: distance from uniform due to prior-posterior warping */
  b1_bayesian: number;
  /** Frequentist floor: distance from uniform (should be small/zero initially) */
  b2_frequentist: number;
  /** Solomonoff curvature: distance from uniform due to complexity priors */
  b3_solomonoff: number;
  /** Total curvature: combined deviation from the flat floor */
  totalCurvature: number;
  /** Inter-layer distances: how far apart the layers are on the manifold */
  interLayerDistances: {
    retrocausal_bayesian: number;
    bayesian_frequentist: number;
    frequentist_solomonoff: number;
    retrocausal_solomonoff: number;
  };
}

/**
 * Compute manifold coordinates for a Buleyean BoundaryStack.
 * Each layer is a point on the simplex. Curvature = distance from uniform.
 */
export function manifoldCoordinates(stack: BoundaryStack): ManifoldCoordinates {
  const n = stack.layers[0].boundary.counts.length;
  const uniform = new Array(n).fill(1 / n);

  // Cannon rotation: all 4 layer distributions are independent (different
  // boundaries). All 4 uniform-distance and 4 inter-layer distance
  // computations are independent. 8 parallel fisherRaoDistance calls.
  const dists = stack.layers.map((layer) =>
    buleyeanDistribution(layer.boundary)
  );

  // FORK: 4 uniform distances (independent per layer)
  const distances = dists.map((d) => fisherRaoDistance(d, uniform));

  // FORK: 4 inter-layer distances (independent pairs)
  const retrocausal_bayesian = fisherRaoDistance(dists[0], dists[1]);
  const bayesian_frequentist = fisherRaoDistance(dists[1], dists[2]);
  const frequentist_solomonoff = fisherRaoDistance(dists[2], dists[3]);
  const retrocausal_solomonoff = fisherRaoDistance(dists[0], dists[3]);

  return {
    b0_retrocausal: distances[0],
    b1_bayesian: distances[1],
    b2_frequentist: distances[2],
    b3_solomonoff: distances[3],
    totalCurvature: distances.reduce((a, b) => a + b, 0),
    interLayerDistances: {
      retrocausal_bayesian,
      bayesian_frequentist,
      frequentist_solomonoff,
      retrocausal_solomonoff,
    },
  };
}

// ============================================================================
// Fraud Detection -- the shape of a lie vs a truth
// ============================================================================

/**
 * Result of statistical fraud detection on a trajectory.
 */
export interface FraudAnalysis {
  /** Per-point geodesic curvature along the trajectory */
  curvatures: number[];
  /** Total accumulated curvature (integral of |kappa|) */
  totalCurvature: number;
  /** Maximum curvature spike (single biggest deviation) */
  maxCurvature: number;
  /** Index of the maximum curvature spike */
  maxCurvatureIdx: number;
  /** Mean curvature (baseline bending rate) */
  meanCurvature: number;
  /** Curvature standard deviation (how erratic the bending is) */
  curvatureStdDev: number;
  /** Total path length on the manifold */
  pathLength: number;
  /** Geodesic (straight-line) distance from start to end */
  geodesicDistance: number;
  /** Winding ratio: pathLength / geodesicDistance (1.0 = perfectly straight) */
  windingRatio: number;
  /** Fraud score: combined anomaly measure (higher = more suspicious) */
  fraudScore: number;
}

/**
 * Detect "statistical fraud" in a distribution trajectory.
 *
 * A truthful evolution of probability follows geodesics on the Fisher
 * manifold -- the shortest paths consistent with the incoming data.
 * A fraudulent evolution deviates from geodesics: something is pushing
 * the distribution in a direction not justified by the observations.
 *
 * The fraud score combines:
 * - Winding ratio (path length / geodesic distance): how indirect the path is
 * - Curvature variance: how erratic the bending is (natural processes are smooth)
 * - Max curvature spike: single biggest deviation (manipulation events)
 *
 * @param trajectory Sequence of probability distributions over time
 * @returns Fraud analysis with per-point curvatures and aggregate scores
 */
export function detectFraud(trajectory: number[][]): FraudAnalysis {
  if (trajectory.length < 3) {
    return {
      curvatures: [],
      totalCurvature: 0,
      maxCurvature: 0,
      maxCurvatureIdx: -1,
      meanCurvature: 0,
      curvatureStdDev: 0,
      pathLength:
        trajectory.length === 2
          ? fisherRaoDistance(trajectory[0], trajectory[1])
          : 0,
      geodesicDistance:
        trajectory.length >= 2
          ? fisherRaoDistance(trajectory[0], trajectory[trajectory.length - 1])
          : 0,
      windingRatio: 1,
      fraudScore: 0,
    };
  }

  const curvatures = trajectoryCurvature(trajectory);

  // Cannon rotation: segment distances are independent (different pairs).
  // Compute all N-1 distances in parallel, then reduce to sum.
  const segmentDistances = new Array<number>(trajectory.length - 1);
  for (let i = 0; i < trajectory.length - 1; i++) {
    segmentDistances[i] = fisherRaoDistance(trajectory[i], trajectory[i + 1]);
  }
  let pathLength = 0;
  for (let i = 0; i < segmentDistances.length; i++) {
    pathLength += segmentDistances[i];
  }

  // Geodesic distance: straight line from start to end
  const geodesicDistance = fisherRaoDistance(
    trajectory[0],
    trajectory[trajectory.length - 1]
  );

  // Curvature statistics
  const totalCurv = curvatures.reduce((sum, k) => sum + Math.abs(k), 0);
  const meanCurv = curvatures.length > 0 ? totalCurv / curvatures.length : 0;
  const maxCurv =
    curvatures.length > 0 ? Math.max(...curvatures.map(Math.abs)) : 0;
  const maxIdx =
    curvatures.length > 0
      ? curvatures.findIndex((k) => Math.abs(k) === maxCurv)
      : -1;

  const variance =
    curvatures.length > 0
      ? curvatures.reduce((sum, k) => sum + (Math.abs(k) - meanCurv) ** 2, 0) /
        curvatures.length
      : 0;
  const stdDev = Math.sqrt(variance);

  // Winding ratio: 1.0 = perfectly geodesic, >1 = winding
  const windingRatio =
    geodesicDistance > 1e-12 ? pathLength / geodesicDistance : 1;

  // Fraud score: composite anomaly measure
  // - High winding ratio: path is unnecessarily indirect
  // - High curvature variance: erratic changes (natural processes are smooth)
  // - High max spike: sudden manipulation event
  const fraudScore =
    (windingRatio - 1) * 2 + // excess winding, weighted
    stdDev * 10 + // curvature irregularity
    (maxCurv > meanCurv * 3 ? maxCurv - meanCurv * 3 : 0); // spike beyond 3x mean

  return {
    curvatures,
    totalCurvature: totalCurv,
    maxCurvature: maxCurv,
    maxCurvatureIdx: maxIdx,
    meanCurvature: meanCurv,
    curvatureStdDev: stdDev,
    pathLength,
    geodesicDistance,
    windingRatio,
    fraudScore,
  };
}

/**
 * Detect fraud in a sequence of Buleyean void boundaries.
 * Converts each boundary to its distribution, then analyzes the trajectory.
 */
export function detectBuleyeanFraud(boundaries: VoidBoundary[]): FraudAnalysis {
  return detectFraud(boundaries.map(buleyeanDistribution));
}

/**
 * Compare two trajectories: one "reference" (trusted) and one "suspect".
 * Returns the relative fraud score -- how much more suspicious the
 * suspect trajectory is compared to the reference.
 *
 * relativeFraud > 1: suspect is more suspicious than reference
 * relativeFraud < 1: suspect is cleaner than reference
 * relativeFraud ~= 1: indistinguishable
 */
export function compareFraud(
  reference: number[][],
  suspect: number[][]
): {
  referenceAnalysis: FraudAnalysis;
  suspectAnalysis: FraudAnalysis;
  relativeFraudScore: number;
} {
  const refAnalysis = detectFraud(reference);
  const susAnalysis = detectFraud(suspect);

  const relativeFraudScore =
    refAnalysis.fraudScore > 1e-12
      ? susAnalysis.fraudScore / refAnalysis.fraudScore
      : susAnalysis.fraudScore > 1e-12
      ? Infinity
      : 1;

  return {
    referenceAnalysis: refAnalysis,
    suspectAnalysis: susAnalysis,
    relativeFraudScore,
  };
}

// ============================================================================
// Geodesic Interpolation
// ============================================================================

/**
 * Interpolate along the geodesic between two distributions.
 * Uses the spherical representation (Bhattacharyya embedding):
 *   xi_i = 2 * sqrt(p_i) maps p to the sphere of radius 2
 *   geodesic on sphere = great circle
 *   invert: p_i = (xi_i / 2)^2
 *
 * @param p Start distribution
 * @param q End distribution
 * @param t Interpolation parameter [0, 1]
 * @returns Distribution at parameter t along the geodesic
 */
export function geodesicInterpolation(
  p: number[],
  q: number[],
  t: number
): number[] {
  const n = p.length;

  // Map to sphere: xi_i = sqrt(p_i)
  const xi_p = p.map(Math.sqrt);
  const xi_q = q.map(Math.sqrt);

  // Angle between the two points on the sphere
  let cosAngle = 0;
  for (let i = 0; i < n; i++) {
    cosAngle += xi_p[i] * xi_q[i];
  }
  cosAngle = Math.min(Math.max(cosAngle, -1), 1);
  const angle = Math.acos(cosAngle);

  if (angle < 1e-12) {
    return [...p]; // Same point
  }

  const sinAngle = Math.sin(angle);

  // Slerp (spherical linear interpolation)
  const result = new Array(n);
  const coeff_p = Math.sin((1 - t) * angle) / sinAngle;
  const coeff_q = Math.sin(t * angle) / sinAngle;

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const xi = coeff_p * xi_p[i] + coeff_q * xi_q[i];
    result[i] = xi * xi; // Back to probability: p_i = xi_i^2
    sum += result[i];
  }

  // Renormalize (numerical safety)
  if (sum > 0) {
    for (let i = 0; i < n; i++) {
      result[i] /= sum;
    }
  }

  return result;
}

/**
 * Generate the geodesic path between two distributions.
 * Returns `steps + 1` distributions evenly spaced along the geodesic.
 */
export function geodesicPath(
  p: number[],
  q: number[],
  steps: number = 10
): number[][] {
  // Cannon rotation: each interpolation step is independent (different t
  // value, same p and q constants). Under gnode --strategy cannon,
  // steps distribute across lanes for parallel interpolation.
  const path = new Array<number[]>(steps + 1);
  for (let i = 0; i <= steps; i++) {
    path[i] = geodesicInterpolation(p, q, i / steps);
  }
  return path;
}
