import { describe, expect, it } from 'bun:test';
import {
  abduceBang,
  observeClosedShell,
  observeFreeCouple,
  observeProtectedC,
  observeUnknownCodon,
  peptideRejectionRecord,
  peptideSearchBracket,
} from './bang-abduction.ts';

describe('bang abduction', () => {
  it('walls the couple hypothesis when the carboxyl end is absent', () => {
    const posterior = abduceBang(observeProtectedC());
    const names = posterior.eliminated.map((e) => e.name);
    expect(names).toContain('couple');
    expect(posterior.survivors.some((s) => s.name === 'refuseProtected')).toBe(true);
  });

  it('walls couple on an unknown codon (UUU fragment refuse)', () => {
    const posterior = abduceBang(observeUnknownCodon());
    expect(posterior.eliminated.some((e) => e.name === 'couple')).toBe(true);
    expect(posterior.survivors.some((s) => s.name === 'refuseTranslation')).toBe(true);
  });

  it('keeps couple as a survivor when both ends are free', () => {
    const posterior = abduceBang(observeFreeCouple());
    expect(posterior.survivors[0]?.name).toBe('couple');
    expect(posterior.eliminated.some((e) => e.name === 'refuseClosed')).toBe(true);
  });

  it('prunes protected ends in the search bracket', () => {
    const bracket = peptideSearchBracket(3, 1, 1);
    expect(bracket.rawStateUpperBound).toBe(9);
    expect(bracket.bracketedStateUpperBound).toBe(4);
    expect(bracket.compressionRatio).toBeCloseTo(2.25, 5);
  });

  it('emits a buleyean-rl-shaped rejection record', () => {
    const record = peptideRejectionRecord('couple Gly C-protected', 'refuseProtected');
    expect(record.rejected_responses).toEqual(['couple Gly C-protected']);
    expect(record.metadata.gate_reason).toBe('refuseProtected');
    expect(record.metadata.source).toBe('chemical-peptide-gate');
  });

  it('rules couple out on a closed-shell observation', () => {
    const posterior = abduceBang(observeClosedShell());
    expect(posterior.eliminated.some((e) => e.name === 'couple')).toBe(true);
    expect(posterior.survivors.some((s) => s.name === 'refuseClosed')).toBe(true);
  });
});
