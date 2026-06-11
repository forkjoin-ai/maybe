import { describe, expect, it } from '@a0n/gnosis/test';
import {
  natDist,
  reception,
  senderRegret,
  receiverRegret,
  isEquilibrium,
  underPushPlateauValue,
  type Game,
} from './charisma-skyrms-game';

const games: Game[] = [
  { baseline: 0, target: 0, bias: 0 },
  { baseline: 1, target: 1, bias: 3 },
  { baseline: 2, target: 2, bias: 5 },
  { baseline: 3, target: 7, bias: 7 },
  { baseline: 0, target: 5, bias: 10 },
];

describe('CharismaSkyrmsGame: natDistSelf', () => {
  it('natDist a a = 0', () => {
    for (const a of [0, 1, 3, 7, 42]) {
      expect(natDist(a, a)).toBe(0);
    }
  });
});

describe('CharismaSkyrmsGame: natDistComm', () => {
  it('natDist a b = natDist b a', () => {
    for (let a = 0; a <= 8; a++) {
      for (let b = 0; b <= 8; b++) {
        expect(natDist(a, b)).toBe(natDist(b, a));
      }
    }
  });
});

describe('CharismaSkyrmsGame: underPushPlateau', () => {
  it('senderRegret g κ 0 = bias - baseline on the plateau', () => {
    for (const g of games) {
      if (g.baseline > g.bias) continue;
      const top = g.bias - g.baseline;
      for (let k = 0; k <= top; k++) {
        expect(senderRegret(g, k, 0)).toBe(underPushPlateauValue(g));
      }
    }
  });
});

describe('CharismaSkyrmsGame: overPushStrict', () => {
  it('κ above plateau strictly exceeds bias - baseline', () => {
    for (const g of games) {
      if (g.baseline > g.bias) continue;
      const plateau = g.bias - g.baseline;
      for (let k = plateau + 1; k <= plateau + 6; k++) {
        expect(senderRegret(g, k, 0)).toBeGreaterThan(plateau);
      }
    }
  });
});

describe('CharismaSkyrmsGame: senderCostFloor', () => {
  it('κ ≤ senderRegret g κ ρ', () => {
    for (const g of games) {
      for (let k = 0; k <= 8; k++) {
        for (let r = 0; r <= 8; r++) {
          expect(k).toBeLessThanOrEqual(senderRegret(g, k, r));
        }
      }
    }
  });
});

describe('CharismaSkyrmsGame: receiverCostFloor', () => {
  it('ρ ≤ receiverRegret g κ ρ', () => {
    for (const g of games) {
      for (let k = 0; k <= 8; k++) {
        for (let r = 0; r <= 8; r++) {
          expect(r).toBeLessThanOrEqual(receiverRegret(g, k, r));
        }
      }
    }
  });
});

describe('CharismaSkyrmsGame: alignedPlateauEquilibrium', () => {
  it('aligned (baseline=target), any k on plateau is equilibrium', () => {
    for (const g of games) {
      if (g.baseline !== g.target || g.baseline > g.bias) continue;
      const plateau = g.bias - g.baseline;
      for (let k = 0; k <= plateau; k++) {
        expect(isEquilibrium(g, k, 0, plateau + 5)).toBe(true);
      }
    }
  });
});

describe('CharismaSkyrmsGame: alignedZeroEquilibrium', () => {
  it('(0, 0) is equilibrium when baseline = target ≤ bias', () => {
    for (const g of games) {
      if (g.baseline !== g.target || g.baseline > g.bias) continue;
      expect(isEquilibrium(g, 0, 0, 10)).toBe(true);
    }
  });
});

describe('CharismaSkyrmsGame: overPushNotEquilibrium', () => {
  it('(k, 0) with k above plateau is not equilibrium (retreat to 0 wins)', () => {
    for (const g of games) {
      if (g.baseline > g.bias) continue;
      const plateau = g.bias - g.baseline;
      for (let k = plateau + 1; k <= plateau + 4; k++) {
        expect(isEquilibrium(g, k, 0, plateau + 5)).toBe(false);
      }
    }
  });
});

describe('CharismaSkyrmsGame: reception formula', () => {
  it('reception = max(0, baseline + κ - ρ)', () => {
    for (const g of games) {
      for (let k = 0; k <= 5; k++) {
        for (let r = 0; r <= 5; r++) {
          expect(reception(g, k, r)).toBe(Math.max(0, g.baseline + k - r));
        }
      }
    }
  });
});
