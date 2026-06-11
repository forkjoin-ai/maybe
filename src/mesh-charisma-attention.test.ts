import { describe, expect, it } from '@a0n/gnosis/test';
import {
  absorbed,
  buleyeanMass,
  saturatedMass,
  outranks,
  prospectWeightColumn,
  silencingResistance,
  type Mesh,
} from './mesh-charisma-attention';

function makeMesh(
  baseline: number,
  charisma: number[],
  resistance: number[]
): Mesh {
  return { n: charisma.length, baseline, charisma, resistance };
}

const meshes: Mesh[] = [
  makeMesh(1, [0, 0, 0], [0, 0, 0]),
  makeMesh(2, [3, 1, 5], [0, 0, 0]),
  makeMesh(2, [3, 1, 5], [1, 4, 2]),
  makeMesh(5, [0, 10, 7, 2], [3, 3, 100, 0]),
  makeMesh(10, [1, 2, 3, 4, 5], [0, 2, 10, 1, 5]),
];

describe('MeshCharismaAttention: buleyeanFloorUniform', () => {
  it('baseline ≤ buleyeanMass at every column', () => {
    for (const m of meshes) {
      for (let j = 0; j < m.n; j++) {
        expect(m.baseline).toBeLessThanOrEqual(buleyeanMass(m, j));
      }
    }
  });
});

describe('MeshCharismaAttention: absorbedBelowBuleyean', () => {
  it('absorbed ≤ buleyeanMass at every column', () => {
    for (const m of meshes) {
      for (let j = 0; j < m.n; j++) {
        expect(absorbed(m, j)).toBeLessThanOrEqual(buleyeanMass(m, j));
      }
    }
  });
});

describe('MeshCharismaAttention: columnReachesSilence', () => {
  it('silencing resistance zeros out absorbed mass at every column', () => {
    for (const m of meshes) {
      const sil = silencingResistance(m);
      for (let j = 0; j < m.n; j++) {
        expect(absorbed(sil, j)).toBe(0);
      }
    }
  });
});

describe('MeshCharismaAttention: columnBoundedByCap', () => {
  it('saturatedMass ≤ cap for every column and cap', () => {
    for (const m of meshes) {
      for (const cap of [0, 1, 2, 5, 10, 100]) {
        for (let j = 0; j < m.n; j++) {
          expect(saturatedMass(m, cap, j)).toBeLessThanOrEqual(cap);
        }
      }
    }
  });
});

describe('MeshCharismaAttention: celebrityPlateau', () => {
  it('columns both at/above cap tie in saturatedMass', () => {
    for (const m of meshes) {
      for (const cap of [0, 1, 2, 5]) {
        for (let i = 0; i < m.n; i++) {
          for (let j = 0; j < m.n; j++) {
            if (cap <= absorbed(m, i) && cap <= absorbed(m, j)) {
              expect(saturatedMass(m, cap, i)).toBe(saturatedMass(m, cap, j));
            }
          }
        }
      }
    }
  });
});

describe('MeshCharismaAttention: celebrityDominates', () => {
  it('saturated celebrity (≥ cap) outranks non-celebrity (< cap)', () => {
    for (const m of meshes) {
      for (const cap of [1, 3, 5, 10]) {
        for (let i = 0; i < m.n; i++) {
          for (let j = 0; j < m.n; j++) {
            if (cap <= absorbed(m, i) && absorbed(m, j) < cap) {
              expect(saturatedMass(m, cap, j)).toBeLessThanOrEqual(
                saturatedMass(m, cap, i)
              );
            }
          }
        }
      }
    }
  });
});

describe('MeshCharismaAttention: rankReflexive', () => {
  it('every column outranks itself', () => {
    for (const m of meshes) {
      for (let j = 0; j < m.n; j++) {
        expect(outranks(m, j, j)).toBe(true);
      }
    }
  });
});

describe('MeshCharismaAttention: rankTransitive', () => {
  it('outranks is transitive', () => {
    for (const m of meshes) {
      for (let i = 0; i < m.n; i++) {
        for (let j = 0; j < m.n; j++) {
          for (let k = 0; k < m.n; k++) {
            if (outranks(m, i, j) && outranks(m, j, k)) {
              expect(outranks(m, i, k)).toBe(true);
            }
          }
        }
      }
    }
  });
});

describe('MeshCharismaAttention: rankTotal', () => {
  it('for any pair, at least one direction of outranks holds', () => {
    for (const m of meshes) {
      for (let i = 0; i < m.n; i++) {
        for (let j = 0; j < m.n; j++) {
          expect(outranks(m, i, j) || outranks(m, j, i)).toBe(true);
        }
      }
    }
  });
});

describe('MeshCharismaAttention: prospectWeightAboveOne', () => {
  it('every column has prospect weight ≥ 1', () => {
    for (const m of meshes) {
      for (let j = 0; j < m.n; j++) {
        expect(prospectWeightColumn(m, j)).toBeGreaterThanOrEqual(1);
      }
    }
  });
});
