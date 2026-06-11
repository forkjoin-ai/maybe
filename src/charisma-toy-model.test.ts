import { describe, expect, it } from '@a0n/gnosis/test';
import {
  BASELINE_UPTAKE,
  perReceiverUptake,
  totalReception,
  netPerReceiverUptake,
  saturatedPerReceiverUptake,
  cascadedCharisma,
  cascadeReception,
  type Source,
  type Audience,
  type Resistance,
  type Capacity,
} from './charisma-toy-model';

const sources = [0, 1, 2, 5, 10, 100].map((c) => ({ charisma: c } as Source));
const audiences = [1, 2, 3, 7, 50].map((n) => ({ size: n } as Audience));
const resistances = [0, 1, 2, 5, 10, 100].map(
  (r) => ({ coefficient: r } as Resistance)
);

describe('CharismaToyModel: receptionMonotoneInCharisma', () => {
  it('higher charisma never lowers total reception', () => {
    for (const a of audiences) {
      for (const s1 of sources) {
        for (const s2 of sources) {
          if (s1.charisma <= s2.charisma) {
            expect(totalReception(s1, a)).toBeLessThanOrEqual(
              totalReception(s2, a)
            );
          }
        }
      }
    }
  });
});

describe('CharismaToyModel: receptionMonotoneInAudience', () => {
  it('larger audiences never reduce total reception', () => {
    for (const s of sources) {
      for (const a1 of audiences) {
        for (const a2 of audiences) {
          if (a1.size <= a2.size) {
            expect(totalReception(s, a1)).toBeLessThanOrEqual(
              totalReception(s, a2)
            );
          }
        }
      }
    }
  });
});

describe('CharismaToyModel: zeroCharismaReducesToBaseline', () => {
  it('zero-charisma source delivers exactly baseline per receiver', () => {
    const s: Source = { charisma: 0 };
    for (const a of audiences) {
      expect(totalReception(s, a)).toBe(a.size * BASELINE_UPTAKE);
    }
  });
});

describe('CharismaToyModel: positiveCharismaStrictlyBeatsBaseline', () => {
  it('positive charisma over positive audience > baseline total', () => {
    for (const a of audiences) {
      for (const s of sources) {
        if (s.charisma > 0 && a.size > 0) {
          expect(totalReception(s, a)).toBeGreaterThan(
            a.size * BASELINE_UPTAKE
          );
        }
      }
    }
  });
});

describe('CharismaToyModel: resistanceMonotone', () => {
  it('higher resistance never increases net uptake', () => {
    for (const s of sources) {
      for (const r1 of resistances) {
        for (const r2 of resistances) {
          if (r1.coefficient <= r2.coefficient) {
            expect(netPerReceiverUptake(s, r2)).toBeLessThanOrEqual(
              netPerReceiverUptake(s, r1)
            );
          }
        }
      }
    }
  });
});

describe('CharismaToyModel: charismaResistanceCollision', () => {
  it('ρ = κ collapses net uptake to baseline', () => {
    for (const s of sources) {
      const r: Resistance = { coefficient: s.charisma };
      expect(netPerReceiverUptake(s, r)).toBe(BASELINE_UPTAKE);
    }
  });
});

describe('CharismaToyModel: totalResistanceSilences', () => {
  it('resistance ≥ perReceiverUptake silences', () => {
    for (const s of sources) {
      const r: Resistance = { coefficient: perReceiverUptake(s) };
      expect(netPerReceiverUptake(s, r)).toBe(0);
      const rr: Resistance = { coefficient: perReceiverUptake(s) + 5 };
      expect(netPerReceiverUptake(s, rr)).toBe(0);
    }
  });
});

describe('CharismaToyModel: saturationUpperBound', () => {
  it('saturated uptake ≤ cap', () => {
    for (const s of sources) {
      for (const cap of [1, 2, 5, 10, 50]) {
        const c: Capacity = { cap };
        expect(saturatedPerReceiverUptake(s, c)).toBeLessThanOrEqual(cap);
      }
    }
  });
});

describe('CharismaToyModel: saturationDominatedByRaw', () => {
  it('saturated uptake ≤ raw uptake', () => {
    for (const s of sources) {
      for (const cap of [1, 2, 5, 10, 50]) {
        const c: Capacity = { cap };
        expect(saturatedPerReceiverUptake(s, c)).toBeLessThanOrEqual(
          perReceiverUptake(s)
        );
      }
    }
  });
});

describe('CharismaToyModel: saturationHonorsBaseline', () => {
  it('saturated uptake ≥ baseline when cap ≥ baseline', () => {
    for (const s of sources) {
      for (const cap of [1, 2, 5, 10, 50]) {
        const c: Capacity = { cap };
        expect(saturatedPerReceiverUptake(s, c)).toBeGreaterThanOrEqual(
          BASELINE_UPTAKE
        );
      }
    }
  });
});

describe('CharismaToyModel: cascadeBoundedBySource', () => {
  it('cascade ≤ source charisma', () => {
    for (const s of sources) {
      for (const i of sources) {
        expect(cascadedCharisma(s, i)).toBeLessThanOrEqual(s.charisma);
      }
    }
  });
});

describe('CharismaToyModel: cascadeBoundedByIntermediary', () => {
  it('cascade ≤ intermediary charisma', () => {
    for (const s of sources) {
      for (const i of sources) {
        expect(cascadedCharisma(s, i)).toBeLessThanOrEqual(i.charisma);
      }
    }
  });
});

describe('CharismaToyModel: cascadeSaturatesAtWeakest', () => {
  it('intermediary ≥ source preserves source charisma exactly', () => {
    for (const s of sources) {
      for (const i of sources) {
        if (s.charisma <= i.charisma) {
          expect(cascadedCharisma(s, i)).toBe(s.charisma);
        }
      }
    }
  });
});

describe('CharismaToyModel: cascadeReception (convenience)', () => {
  it('agrees with totalReception using bottleneck charisma', () => {
    for (const s of sources) {
      for (const i of sources) {
        for (const a of audiences) {
          const expected = totalReception(
            { charisma: cascadedCharisma(s, i) },
            a
          );
          expect(cascadeReception(s, i, a)).toBe(expected);
        }
      }
    }
  });
});
