import { describe, expect, it } from 'bun:test';
import {
  fisherMetric,
  fisherMetricBuleyean,
  fisherInnerProduct,
  fisherNorm,
  bhattacharyyaCoefficient,
  fisherRaoDistance,
  buleyeanDistance,
  scalarCurvature,
  geodesicCurvature,
  trajectoryCurvature,
  totalCurvature,
  manifoldCoordinates,
  detectFraud,
  detectBuleyeanFraud,
  compareFraud,
  geodesicInterpolation,
  geodesicPath,
} from './manifold';
import { createVoidBoundary, updateVoidBoundary } from '@a0n/gnosis/src/void';
import { createBuleyeanStack } from './layers';
import { buleyeanDistribution } from './buleyean';

describe('fisherMetric', () => {
  it('returns 1/p_i for each dimension', () => {
    const g = fisherMetric([0.25, 0.25, 0.25, 0.25]);
    for (const gi of g) {
      expect(gi).toBe(4);
    }
  });

  it('higher metric where probability is lower', () => {
    const g = fisherMetric([0.1, 0.9]);
    expect(g[0]).toBeGreaterThan(g[1]); // 10 > 1.11
  });

  it('works on Buleyean distributions', () => {
    const boundary = createVoidBoundary(3);
    updateVoidBoundary(boundary, 0, 5);
    const g = fisherMetricBuleyean(boundary);
    expect(g).toHaveLength(3);
    // Most rejected dimension has lowest probability -> highest metric
    expect(g[0]).toBeGreaterThan(g[1]);
    expect(g[0]).toBeGreaterThan(g[2]);
  });
});

describe('fisherInnerProduct', () => {
  it('computes weighted inner product', () => {
    const p = [0.5, 0.5];
    const u = [0.1, -0.1]; // tangent vector (sums to 0)
    const v = [0.1, -0.1];
    const ip = fisherInnerProduct(p, u, v);
    // sum(u_i * v_i / p_i) = 0.01/0.5 + 0.01/0.5 = 0.04
    expect(ip).toBeCloseTo(0.04, 10);
  });
});

describe('fisherNorm', () => {
  it('computes norm of tangent vector', () => {
    const p = [0.5, 0.5];
    const v = [0.1, -0.1];
    expect(fisherNorm(p, v)).toBeCloseTo(0.2, 10);
  });
});

describe('bhattacharyyaCoefficient', () => {
  it('is 1 for identical distributions', () => {
    const p = [0.3, 0.7];
    expect(bhattacharyyaCoefficient(p, p)).toBeCloseTo(1, 10);
  });

  it('is less than 1 for different distributions', () => {
    const p = [0.9, 0.1];
    const q = [0.1, 0.9];
    expect(bhattacharyyaCoefficient(p, q)).toBeLessThan(1);
  });

  it('approaches 0 for disjoint distributions', () => {
    const p = [1, 0];
    const q = [0, 1];
    expect(bhattacharyyaCoefficient(p, q)).toBeCloseTo(0, 10);
  });
});

describe('fisherRaoDistance', () => {
  it('is 0 for identical distributions', () => {
    const p = [0.25, 0.25, 0.25, 0.25];
    expect(fisherRaoDistance(p, p)).toBeCloseTo(0, 10);
  });

  it('is positive for different distributions', () => {
    const p = [0.7, 0.3];
    const q = [0.3, 0.7];
    expect(fisherRaoDistance(p, q)).toBeGreaterThan(0);
  });

  it('satisfies triangle inequality', () => {
    const p = [0.6, 0.4];
    const q = [0.3, 0.7];
    const r = [0.5, 0.5];
    const d_pq = fisherRaoDistance(p, q);
    const d_qr = fisherRaoDistance(q, r);
    const d_pr = fisherRaoDistance(p, r);
    expect(d_pr).toBeLessThanOrEqual(d_pq + d_qr + 1e-10);
  });

  it('is symmetric', () => {
    const p = [0.8, 0.2];
    const q = [0.4, 0.6];
    expect(fisherRaoDistance(p, q)).toBeCloseTo(fisherRaoDistance(q, p), 10);
  });

  it('works on Buleyean boundaries', () => {
    const a = createVoidBoundary(3);
    const b = createVoidBoundary(3);
    updateVoidBoundary(b, 0, 10);
    expect(buleyeanDistance(a, b)).toBeGreaterThan(0);
  });
});

describe('scalarCurvature', () => {
  it('is 0 for 1-simplex (2 outcomes)', () => {
    // 1-dimensional manifold has no intrinsic curvature
    expect(scalarCurvature(2)).toBe(0);
  });

  it('is positive for n >= 3', () => {
    expect(scalarCurvature(3)).toBe(0.5); // (2*1)/4
    expect(scalarCurvature(4)).toBe(1.5); // (3*2)/4
    expect(scalarCurvature(5)).toBe(3.0); // (4*3)/4
  });
});

describe('geodesicCurvature', () => {
  it('is 0 for collinear points on a geodesic', () => {
    const p = [0.5, 0.5];
    const q = [0.5, 0.5]; // same point = zero curvature
    const r = [0.5, 0.5];
    expect(geodesicCurvature(p, q, r)).toBeCloseTo(0, 10);
  });

  it('is positive when path bends', () => {
    const p = [0.8, 0.1, 0.1];
    const q = [0.1, 0.8, 0.1]; // sharp turn
    const r = [0.1, 0.1, 0.8];
    expect(geodesicCurvature(p, q, r)).toBeGreaterThan(0);
  });

  it('is larger for sharper bends', () => {
    // Mild bend
    const p1 = [0.5, 0.3, 0.2];
    const q1 = [0.4, 0.35, 0.25];
    const r1 = [0.3, 0.4, 0.3];

    // Sharp bend
    const p2 = [0.8, 0.1, 0.1];
    const q2 = [0.1, 0.8, 0.1];
    const r2 = [0.8, 0.1, 0.1]; // reversal!

    expect(geodesicCurvature(p2, q2, r2)).toBeGreaterThan(
      geodesicCurvature(p1, q1, r1)
    );
  });
});

describe('trajectoryCurvature', () => {
  it('returns n-2 curvature values for n-point trajectory', () => {
    const traj = [
      [0.5, 0.5],
      [0.6, 0.4],
      [0.7, 0.3],
      [0.8, 0.2],
    ];
    const curv = trajectoryCurvature(traj);
    expect(curv).toHaveLength(2);
  });
});

describe('manifoldCoordinates', () => {
  it('empty stack has zero curvature everywhere', () => {
    const stack = createBuleyeanStack(4);
    const coords = manifoldCoordinates(stack);
    // All layers uniform -> all distances from uniform are 0
    expect(coords.b0_retrocausal).toBeCloseTo(0, 10);
    expect(coords.b1_bayesian).toBeCloseTo(0, 10);
    expect(coords.b2_frequentist).toBeCloseTo(0, 10);
    expect(coords.b3_solomonoff).toBeCloseTo(0, 10);
    expect(coords.totalCurvature).toBeCloseTo(0, 10);
  });

  it('retrocausal constraints increase b0 curvature', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[0].boundary, 0, 20);
    const coords = manifoldCoordinates(stack);
    expect(coords.b0_retrocausal).toBeGreaterThan(0);
    // Other layers still flat
    expect(coords.b1_bayesian).toBeCloseTo(0, 10);
    expect(coords.b2_frequentist).toBeCloseTo(0, 10);
  });

  it('frequentist observations increase b2', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[2].boundary, 1, 10);
    const coords = manifoldCoordinates(stack);
    expect(coords.b2_frequentist).toBeGreaterThan(0);
  });

  it('inter-layer distances reflect separation', () => {
    const stack = createBuleyeanStack(4);
    // Make retrocausal and bayesian very different
    updateVoidBoundary(stack.layers[0].boundary, 0, 50);
    updateVoidBoundary(stack.layers[1].boundary, 3, 50);
    const coords = manifoldCoordinates(stack);
    expect(coords.interLayerDistances.retrocausal_bayesian).toBeGreaterThan(0);
  });
});

describe('detectFraud', () => {
  it('returns zero fraud for geodesic trajectory', () => {
    // Generate a geodesic path (should have minimal fraud score)
    const path = geodesicPath([0.7, 0.2, 0.1], [0.2, 0.5, 0.3], 20);
    const analysis = detectFraud(path);
    expect(analysis.windingRatio).toBeCloseTo(1, 1);
    expect(analysis.fraudScore).toBeLessThan(1);
  });

  it('detects high fraud for manipulated trajectory', () => {
    // Create a zigzag trajectory (clearly not geodesic)
    const trajectory: number[][] = [];
    for (let i = 0; i < 20; i++) {
      if (i % 2 === 0) {
        trajectory.push([0.8, 0.1, 0.1]);
      } else {
        trajectory.push([0.1, 0.1, 0.8]);
      }
    }
    const analysis = detectFraud(trajectory);
    expect(analysis.windingRatio).toBeGreaterThan(2);
    expect(analysis.fraudScore).toBeGreaterThan(1);
  });

  it('handles short trajectories gracefully', () => {
    const analysis1 = detectFraud([[0.5, 0.5]]);
    expect(analysis1.fraudScore).toBe(0);

    const analysis2 = detectFraud([
      [0.5, 0.5],
      [0.6, 0.4],
    ]);
    expect(analysis2.fraudScore).toBe(0);
    expect(analysis2.pathLength).toBeGreaterThan(0);
  });

  it('works on Buleyean boundaries', () => {
    const boundaries = [];
    const b = createVoidBoundary(3);
    boundaries.push({ counts: [...b.counts], totalEntries: b.totalEntries });
    for (let i = 0; i < 10; i++) {
      updateVoidBoundary(b, i % 3, 1);
      boundaries.push({ counts: [...b.counts], totalEntries: b.totalEntries });
    }
    const analysis = detectBuleyeanFraud(boundaries);
    expect(analysis.pathLength).toBeGreaterThanOrEqual(0);
  });
});

describe('compareFraud', () => {
  it('geodesic vs zigzag: zigzag is more suspicious', () => {
    const geodesic = geodesicPath([0.7, 0.2, 0.1], [0.2, 0.5, 0.3], 10);

    const zigzag: number[][] = [];
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      if (i % 2 === 0) {
        zigzag.push([0.7 - t * 0.5, 0.2 + t * 0.3, 0.1 + t * 0.2]);
      } else {
        zigzag.push([0.2 + (1 - t) * 0.3, 0.7 - (1 - t) * 0.2, 0.1]);
      }
    }

    const result = compareFraud(geodesic, zigzag);
    expect(result.relativeFraudScore).toBeGreaterThan(1);
  });
});

describe('geodesicInterpolation', () => {
  it('t=0 returns start distribution', () => {
    const p = [0.7, 0.2, 0.1];
    const q = [0.2, 0.5, 0.3];
    const result = geodesicInterpolation(p, q, 0);
    for (let i = 0; i < p.length; i++) {
      expect(result[i]).toBeCloseTo(p[i], 5);
    }
  });

  it('t=1 returns end distribution', () => {
    const p = [0.7, 0.2, 0.1];
    const q = [0.2, 0.5, 0.3];
    const result = geodesicInterpolation(p, q, 1);
    for (let i = 0; i < q.length; i++) {
      expect(result[i]).toBeCloseTo(q[i], 5);
    }
  });

  it('interpolation stays on the simplex (sums to 1)', () => {
    const p = [0.6, 0.3, 0.1];
    const q = [0.1, 0.2, 0.7];
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const result = geodesicInterpolation(p, q, t);
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 8);
      for (const r of result) {
        expect(r).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('midpoint is equidistant from endpoints', () => {
    const p = [0.8, 0.15, 0.05];
    const q = [0.1, 0.6, 0.3];
    const mid = geodesicInterpolation(p, q, 0.5);
    const d_pm = fisherRaoDistance(p, mid);
    const d_mq = fisherRaoDistance(mid, q);
    expect(d_pm).toBeCloseTo(d_mq, 3);
  });
});

describe('geodesicPath', () => {
  it('generates correct number of points', () => {
    const path = geodesicPath([0.5, 0.5], [0.8, 0.2], 10);
    expect(path).toHaveLength(11);
  });

  it('path length equals geodesic distance', () => {
    const p = [0.7, 0.2, 0.1];
    const q = [0.2, 0.5, 0.3];
    const path = geodesicPath(p, q, 50);
    let pathLen = 0;
    for (let i = 1; i < path.length; i++) {
      pathLen += fisherRaoDistance(path[i - 1], path[i]);
    }
    const geodDist = fisherRaoDistance(p, q);
    // Path length should converge to geodesic distance with enough steps
    expect(pathLen).toBeCloseTo(geodDist, 2);
  });
});
