import { describe, expect, it } from '@a0n/gnosis/test';
import { buleyeanWeight } from '@a0n/buleyean-kernel';
import {
  buleyeanPredict,
  celebrityPlateau,
  charismaPopularity,
  charismaRank,
  gain,
  loss,
  prospectWeight,
  sandwichWitness,
  saturatedPopularity,
  type CharismaRanked,
} from './charisma-rank';

// ---------------------------------------------------------------------------
// Shared grid: deterministic triples in ℕ × ℕ × ℕ up to size 8 each.
// ---------------------------------------------------------------------------

interface Triple {
  baseline: number;
  advance: number;
  resist: number;
}

function natGrid(size: number): Triple[] {
  const out: Triple[] = [];
  for (let b = 0; b <= size; b++) {
    for (let a = 0; a <= size; a++) {
      for (let r = 0; r <= size; r++) {
        out.push({ baseline: b, advance: a, resist: r });
      }
    }
  }
  return out;
}

const GRID = natGrid(8);

describe('Buleyean sandwich invariant (Law 3)', () => {
  it('chainHolds = true for every (baseline, advance, resist) in the 0..8 cube', () => {
    for (const { baseline, advance, resist } of GRID) {
      const w = sandwichWitness(baseline, advance, resist);
      expect(w.chainHolds).toBe(true);
      expect(w.pessimistic <= w.actual).toBe(true);
      expect(w.actual <= w.predict).toBe(true);
      expect(w.predict <= w.optimistic).toBe(true);
    }
  });

  it('predictAboveBaseline holds everywhere (Law 1 reflected in witness)', () => {
    for (const { baseline, advance, resist } of GRID) {
      const w = sandwichWitness(baseline, advance, resist);
      expect(w.predictAboveBaseline).toBe(true);
    }
  });
});

describe('Law 1: Impossibility of zero (buleyeanPredict ≥ baseline)', () => {
  it('buleyeanPredict ≥ baseline across the full grid', () => {
    for (const { baseline, advance, resist } of GRID) {
      expect(buleyeanPredict(baseline, advance, resist) >= baseline).toBe(true);
    }
  });

  it('under total resistance (resist >= advance), predict collapses to baseline', () => {
    for (let b = 0; b <= 8; b++) {
      for (let a = 0; a <= 8; a++) {
        for (let r = a; r <= a + 8; r++) {
          expect(buleyeanPredict(b, a, r)).toBe(b);
        }
      }
    }
  });
});

describe('Law 2: Strict ordering / monotonicity of charismaPopularity', () => {
  it('at fixed resist, popularity is monotone non-decreasing in advance', () => {
    for (let baseline = 0; baseline <= 8; baseline++) {
      for (let resist = 0; resist <= 8; resist++) {
        let prev = -Infinity;
        for (let advance = 0; advance <= 8; advance++) {
          const pop = charismaPopularity(baseline, advance, resist);
          expect(pop >= prev).toBe(true);
          prev = pop;
        }
      }
    }
  });

  it('at fixed advance, popularity is monotone non-increasing in resist', () => {
    for (let baseline = 0; baseline <= 8; baseline++) {
      for (let advance = 0; advance <= 8; advance++) {
        let prev = Infinity;
        for (let resist = 0; resist <= 8; resist++) {
          const pop = charismaPopularity(baseline, advance, resist);
          expect(pop <= prev).toBe(true);
          prev = pop;
        }
      }
    }
  });
});

describe('saturatedPopularity: cap behavior', () => {
  it('equals cap when popularity exceeds cap', () => {
    // baseline 10, advance 20, resist 0 -> pop 30, cap 5 -> 5
    expect(saturatedPopularity(10, 20, 0, 5)).toBe(5);
    expect(saturatedPopularity(0, 8, 0, 3)).toBe(3);
    expect(saturatedPopularity(4, 4, 0, 7)).toBe(7);
  });

  it('equals popularity when popularity is at-or-below cap', () => {
    expect(saturatedPopularity(0, 3, 0, 100)).toBe(3);
    // tie at cap
    expect(saturatedPopularity(0, 5, 0, 5)).toBe(5);
    // pop = 0, cap positive
    expect(saturatedPopularity(0, 0, 5, 10)).toBe(0);
  });

  it('saturatedPopularity = min(pop, cap) across the grid at cap=6', () => {
    const cap = 6;
    for (const { baseline, advance, resist } of GRID) {
      const pop = charismaPopularity(baseline, advance, resist);
      expect(saturatedPopularity(baseline, advance, resist, cap)).toBe(
        Math.min(pop, cap)
      );
    }
  });
});

describe('charismaRank: sort, rank indexing, and tie breaking', () => {
  interface Model {
    id: string;
    quality: number;
    cost: number;
  }

  const models: Model[] = [
    { id: 'a', quality: 3, cost: 1 }, // pop 2
    { id: 'b', quality: 5, cost: 1 }, // pop 4
    { id: 'c', quality: 7, cost: 2 }, // pop 5
    { id: 'd', quality: 4, cost: 2 }, // pop 2
    { id: 'e', quality: 6, cost: 6 }, // pop 0
  ];

  it('output is descending by popularity and rank is zero-indexed sequential', () => {
    const ranked = charismaRank<Model>({
      items: models,
      advance: (m) => m.quality,
      resist: (m) => m.cost,
    });
    expect(ranked.length).toBe(models.length);
    for (let i = 0; i < ranked.length; i++) {
      expect(ranked[i].rank).toBe(i);
      if (i > 0) {
        expect(ranked[i - 1].popularity >= ranked[i].popularity).toBe(true);
      }
    }
    // Top should be model c.
    expect(ranked[0].item.id).toBe('c');
  });

  it('default tie breaker is advance (higher advance wins ties)', () => {
    const ranked = charismaRank<Model>({
      items: models,
      advance: (m) => m.quality,
      resist: (m) => m.cost,
    });
    // a and d tie at popularity=2; d has higher advance (4 > 3) so d outranks a.
    const aIdx = ranked.findIndex((r) => r.item.id === 'a');
    const dIdx = ranked.findIndex((r) => r.item.id === 'd');
    expect(ranked[aIdx].popularity).toBe(2);
    expect(ranked[dIdx].popularity).toBe(2);
    expect(dIdx < aIdx).toBe(true);
  });

  it('explicit tieBreaker overrides default advance fallback', () => {
    const ranked = charismaRank<Model>({
      items: models,
      advance: (m) => m.quality,
      resist: (m) => m.cost,
      // Invert: lower id letter wins. Map "a"=1, "d"=4, so "d" > "a" still.
      // Use negative to flip: want "a" to outrank "d".
      tieBreaker: (m) => -m.quality,
    });
    const aIdx = ranked.findIndex((r) => r.item.id === 'a');
    const dIdx = ranked.findIndex((r) => r.item.id === 'd');
    expect(aIdx < dIdx).toBe(true);
  });

  it('empty input yields empty output', () => {
    const ranked = charismaRank<Model>({
      items: [],
      advance: (m) => m.quality,
      resist: (m) => m.cost,
    });
    expect(ranked).toEqual([]);
  });
});

describe('celebrityPlateau', () => {
  interface Item {
    x: number;
  }
  const advance = (i: Item) => i.x;
  const resist = (_: Item) => 0;

  it('true when two or more items hit the cap', () => {
    const ranked = charismaRank<Item>({
      items: [{ x: 10 }, { x: 8 }, { x: 3 }],
      advance,
      resist,
      cap: 5,
    });
    // x=10 and x=8 both exceed cap=5; x=3 does not.
    expect(celebrityPlateau(ranked)).toBe(true);
  });

  it('false when only one item hits the cap', () => {
    const ranked = charismaRank<Item>({
      items: [{ x: 10 }, { x: 4 }, { x: 2 }],
      advance,
      resist,
      cap: 5,
    });
    expect(celebrityPlateau(ranked)).toBe(false);
  });

  it('false when no items hit the cap', () => {
    const ranked = charismaRank<Item>({
      items: [{ x: 1 }, { x: 2 }, { x: 3 }],
      advance,
      resist,
      cap: 100,
    });
    expect(celebrityPlateau(ranked)).toBe(false);
  });

  it('false for an empty ranking', () => {
    const empty: CharismaRanked<Item>[] = [];
    expect(celebrityPlateau(empty)).toBe(false);
  });
});

describe('Prospect Theory: gain / loss', () => {
  it('gain - loss = outcome - reference for every pair in 0..8', () => {
    for (let ref = 0; ref <= 8; ref++) {
      for (let out = 0; out <= 8; out++) {
        expect(gain(ref, out) - loss(ref, out)).toBe(out - ref);
      }
    }
  });

  it('both gain and loss are non-negative', () => {
    for (let ref = 0; ref <= 8; ref++) {
      for (let out = 0; out <= 8; out++) {
        expect(gain(ref, out) >= 0).toBe(true);
        expect(loss(ref, out) >= 0).toBe(true);
      }
    }
  });

  it('exactly one of gain/loss is positive off the diagonal; both zero on it', () => {
    for (let ref = 0; ref <= 8; ref++) {
      for (let out = 0; out <= 8; out++) {
        const g = gain(ref, out);
        const l = loss(ref, out);
        if (out === ref) {
          expect(g).toBe(0);
          expect(l).toBe(0);
        } else if (out > ref) {
          expect(g > 0).toBe(true);
          expect(l).toBe(0);
        } else {
          expect(g).toBe(0);
          expect(l > 0).toBe(true);
        }
      }
    }
  });
});

describe('prospectWeight parity with @a0n/buleyean-kernel.buleyeanWeight', () => {
  it('agrees on every (rounds, rejections) in 0..8', () => {
    for (let rounds = 0; rounds <= 8; rounds++) {
      for (let rejections = 0; rejections <= 8; rejections++) {
        expect(prospectWeight(rounds, rejections)).toBe(
          buleyeanWeight(rounds, rejections)
        );
      }
    }
  });
});

describe('sandwichWitness.exactOnCoveredRegime', () => {
  it('actual === predict whenever resist <= advance', () => {
    for (let baseline = 0; baseline <= 8; baseline++) {
      for (let advance = 0; advance <= 8; advance++) {
        for (let resist = 0; resist <= advance; resist++) {
          const w = sandwichWitness(baseline, advance, resist);
          expect(w.exactOnCoveredRegime).toBe(true);
          expect(w.actual).toBe(w.predict);
        }
      }
    }
  });

  it('flag is true on uncovered regime too (resist > advance collapses both to baseline-bounded forms)', () => {
    for (let baseline = 0; baseline <= 8; baseline++) {
      for (let advance = 0; advance <= 8; advance++) {
        for (let resist = advance + 1; resist <= advance + 5; resist++) {
          const w = sandwichWitness(baseline, advance, resist);
          // Definition sets flag = true when resist > advance (trivially).
          expect(w.exactOnCoveredRegime).toBe(true);
        }
      }
    }
  });
});
