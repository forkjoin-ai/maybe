/**
 * Bang / peptide abduction — inference by what the gates rule out.
 *
 * Lean twins: `ChemicalBang`, `ChemicalPeptide`, `ChemicalRejectionLedger`.
 * Hard walls are the gates (protected ends, closed shells, unknown codon).
 * Survivors keep Buleyean sliver mass. Crackerjack-shaped bracket: raw n×n
 * couple attempts vs the admitted (cFree × nFree) rectangle. Enumeration of
 * that rectangle lives in `@a0n/aeon-crackerjack` `ChemistryBruteForce`.
 *
 * Not a ribosome and not a reward model.
 */

import { buildAbduction, type AbductivePosterior, type Observation } from './abduction.js';

export type BangHypothesisId = 'couple' | 'refuseProtected' | 'refuseClosed' | 'refuseTranslation';

const FEATURES = [
  'cFree',
  'nFree',
  'closedShell',
  'unknownCodon',
  'waterBalances',
] as const;

export type BangFeature = (typeof FEATURES)[number];

/**
 * P(feature present | hypothesis), row-major, hypotheses:
 * 0 couple, 1 refuseProtected, 2 refuseClosed, 3 refuseTranslation
 *
 * Hard walls fire when a present feature has near-zero likelihood under a
 * hypothesis (hardWallLo) or an absent feature is near-certain (hardWallHi).
 */
const LIKELIHOOD = new Float64Array([
  // couple: needs free ends, not a closed shell, not an unknown codon
  0.95, 0.95, 0.02, 0.02, 0.80,
  // refuseProtected: carboxyl blocked; amino end may still be free
  0.08, 0.90, 0.05, 0.05, 0.50,
  // refuseClosed: noble-style closed shell
  0.40, 0.40, 0.97, 0.05, 0.40,
  // refuseTranslation: unknown codon (UUU in the Gly/Leu fragment)
  0.50, 0.50, 0.10, 0.97, 0.30,
]);

export function buildBangAbduction() {
  return buildAbduction(
    {
      hypotheses: [
        { id: 0, name: 'couple', prior: 0.4, protocol: { id: 'admit-couple', label: 'admit couple display' } },
        { id: 1, name: 'refuseProtected', prior: 0.25, protocol: { id: 'fizzle-protected', label: 'fizzle: spent or blocked end' } },
        { id: 2, name: 'refuseClosed', prior: 0.2, protocol: { id: 'fizzle-closed', label: 'fizzle: closed shell' } },
        { id: 3, name: 'refuseTranslation', prior: 0.15, protocol: { id: 'fizzle-codon', label: 'fizzle: codon not in fragment' } },
      ],
      features: [...FEATURES],
      pFeatureGivenHypothesis: LIKELIHOOD,
    },
    { hardWallHi: 0.93, hardWallLo: 0.10, protocolThreshold: 0.35, probeK: 3 },
  );
}

/** Protected C-end observation — Lean `couple cProtectedGly freeGly`. */
export function observeProtectedC(): Observation {
  return { present: ['nFree'], absent: ['cFree'] };
}

/** Unknown codon — Lean `aaOf uuu = none`. */
export function observeUnknownCodon(): Observation {
  return { present: ['unknownCodon'], absent: [] };
}

/** Closed-shell bang — Lean `analyze heliumNeon = refuseClosed`. */
export function observeClosedShell(): Observation {
  return { present: ['closedShell'], absent: [] };
}

/** Free Gly·Gly with water balance — Lean `couple freeGly freeGly`. */
export function observeFreeCouple(): Observation {
  return { present: ['cFree', 'nFree', 'waterBalances'], absent: ['closedShell', 'unknownCodon'] };
}

export function abduceBang(obs: Observation): AbductivePosterior {
  return buildBangAbduction().abduce(obs);
}

/**
 * Crackerjack-shaped search bracket: raw n×n couple attempts, then prune
 * protected carboxyl and amino ends. Heuristic bound only — brute-force
 * enumeration is `bruteForcePeptideCouples` in aeon-crackerjack.
 */
export function peptideSearchBracket(n: number, protectedC: number, protectedN: number): {
  rawStateUpperBound: number;
  bracketedStateUpperBound: number;
  compressionRatio: number;
  theoremAnchors: readonly string[];
} {
  const size = Math.max(0, Math.floor(n));
  const pC = Math.min(size, Math.max(0, Math.floor(protectedC)));
  const pN = Math.min(size, Math.max(0, Math.floor(protectedN)));
  const raw = Math.max(1, size * size);
  const admittedC = Math.max(0, size - pC);
  const admittedN = Math.max(0, size - pN);
  const bracketed = Math.max(1, admittedC * admittedN);
  return {
    rawStateUpperBound: raw,
    bracketedStateUpperBound: bracketed,
    compressionRatio: raw / bracketed,
    theoremAnchors: [
      'Gnosis.ChemicalPeptide.couple',
      'Gnosis.ChemicalRejectionLedger.chemical_rejection_ledger_master',
    ],
  };
}

/** JSONL-shaped rejection record buleyean-rl `append_rejection_record` can ingest. */
export function peptideRejectionRecord(action: string, reason: string): {
  prompt: string;
  rejected_responses: string[];
  rejection_counts: number[];
  total_rounds: number;
  metadata: { source: string; gate_reason: string; driver: string };
} {
  return {
    prompt: action,
    rejected_responses: [action],
    rejection_counts: [1],
    total_rounds: 1,
    metadata: {
      source: 'chemical-peptide-gate',
      gate_reason: reason,
      driver: 'chemical-bang',
    },
  };
}
