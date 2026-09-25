import { describe, expect, it } from 'bun:test';
import {
  act,
  charges,
  residue,
  skipFuture,
  skipPast,
  separatingCollision,
  type BuleCarrier,
} from './skip-step';

const b = (waste: number, opportunity: number, diversity: number): BuleCarrier => ({
  waste,
  opportunity,
  diversity,
});

describe('skip-step: the predictability boundary', () => {
  it('charges determine the future exactly', () => {
    const c = b(3, 2, 5);
    expect(charges(c)).toEqual([3, 7]);
    expect(act(c)).toEqual({ waste: 3, opportunity: 0, diversity: 7 });
    expect(skipFuture(c).skip).toBe(true);
    expect(skipFuture(c).value).toEqual(act(c));
  });

  it('waste alone is blind to the future', () => {
    const sample = [b(0, 1, 0), b(0, 2, 0)];
    const witness = separatingCollision((x: BuleCarrier) => x.waste, act, sample);
    expect(witness).not.toBeNull();
    expect(skipFuture({ waste: 0 }).class).toBe('free');
  });

  it('the future does not determine the past (the inversion orbit is free)', () => {
    const left = b(0, 1, 0);
    const right = b(0, 0, 1);
    expect(act(left)).toEqual(act(right));
    expect(skipPast(act(left)).class).toBe('free');
    const key = (x: BuleCarrier): string => x.waste + ':' + x.opportunity + ':' + x.diversity;
    const actKey = (x: BuleCarrier): string => x.waste + ':' + residue(x);
    expect(separatingCollision(actKey, key, [left, right])).not.toBeNull();
  });

  it('future plus one gauge coordinate recovers the past', () => {
    const c = b(4, 3, 5);
    const decision = skipPast(act(c), c.opportunity);
    expect(decision.skip).toBe(true);
    expect(decision.value).toEqual(c);
    expect(skipPast(act(c), 9).class).toBe('blind');
  });

  it('residue is the invariant of the inversion', () => {
    expect(residue(b(2, 7, 1))).toBe(residue(b(2, 1, 7)));
  });
});
