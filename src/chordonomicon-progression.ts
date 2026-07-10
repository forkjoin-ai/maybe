/**
 * Chordonomicon-style finite chord progression kernel.
 *
 * The runtime shape is intentionally plain Nat arithmetic:
 * six diatonic functions, integer row weights summing to 100, and a finite
 * Markov witness. Chordonomicon-scale data can be projected into this surface
 * by inferring a major tonic per progression, counting adjacent roman-function
 * transitions, then normalizing each row back to a 100-mass kernel.
 */

import {
  compileFiniteKernel,
  compileKernelRow,
  compileMarkovWitness,
  type FiniteKernelWitness,
  type FiniteMarkovWitnessRuntime,
} from '@a0n/aeon-logic';

export const ROMAN_CHORDS = ['I', 'ii', 'iii', 'IV', 'V', 'vi'] as const;
export type RomanChord = (typeof ROMAN_CHORDS)[number];

export type FunctionClass = 'tonic' | 'predominant' | 'dominant';
export type ChordQuality = 'major' | 'minor' | 'dominant' | 'diminished' | 'augmented' | 'unknown';
export type SongSectionName = 'intro' | 'verse' | 'preChorus' | 'chorus' | 'bridge' | 'solo' | 'outro';

export interface ParsedChordSymbol {
  readonly raw: string;
  readonly rootPc: number;
  readonly quality: ChordQuality;
}

export interface KeyInference {
  readonly tonicPc: number;
  readonly score: number;
  readonly projectedCount: number;
}

export interface ChordTransitionEdge {
  readonly from: RomanChord;
  readonly to: RomanChord;
  readonly weight: number;
  readonly probability: number;
  readonly highEntropy: boolean;
  readonly fromClass: FunctionClass;
  readonly toClass: FunctionClass;
}

export interface ChordTransitionKernelWitness {
  readonly rowMass: number;
  readonly kernel: FiniteKernelWitness;
  readonly markov: FiniteMarkovWitnessRuntime;
}

export interface ChordonomiconSection {
  readonly section: SongSectionName;
  readonly label: string;
  readonly tokens: readonly string[];
}

export interface SectionChordSetCandidate {
  readonly section: SongSectionName;
  readonly chordCount: number;
  readonly chords: readonly RomanChord[];
  readonly count: number;
  readonly probability: number;
}

export type ChordTransitionMatrix = Record<RomanChord, readonly number[]>;
export type ChordTransitionCounts = Record<RomanChord, Record<RomanChord, number>>;

export const CHORDONOMICON_BASELINE_ROW_MASS = 100;

export const BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS: ChordTransitionMatrix = {
  I: [10, 15, 10, 30, 25, 10],
  ii: [10, 10, 5, 15, 50, 10],
  iii: [15, 25, 5, 10, 10, 35],
  IV: [30, 10, 10, 10, 30, 10],
  V: [60, 5, 5, 10, 10, 10],
  vi: [15, 20, 15, 30, 10, 10],
};

export const FUNCTION_CLASS: Record<RomanChord, FunctionClass> = {
  I: 'tonic',
  ii: 'predominant',
  iii: 'tonic',
  IV: 'predominant',
  V: 'dominant',
  vi: 'tonic',
};

const NATURAL_ROOT_PC: Record<string, number> = {
  C: 0,
  D: 2,
  E: 4,
  F: 5,
  G: 7,
  A: 9,
  B: 11,
};

const ROMAN_ROOTS: Record<RomanChord, number> = {
  I: 0,
  ii: 2,
  iii: 4,
  IV: 5,
  V: 7,
  vi: 9,
};

const SECTION_TAG_MAP: Record<string, SongSectionName> = {
  intro: 'intro',
  verse: 'verse',
  prechorus: 'preChorus',
  pre_chorus: 'preChorus',
  chorus: 'chorus',
  bridge: 'bridge',
  solo: 'solo',
  instrumental: 'solo',
  interlude: 'solo',
  outro: 'outro',
};

function pitchClass(value: number): number {
  return ((value % 12) + 12) % 12;
}

function rowFor(matrix: ChordTransitionMatrix, chord: RomanChord): readonly number[] {
  const row = matrix[chord];
  if (row.length !== ROMAN_CHORDS.length) {
    throw new Error(`transition row ${chord} must have ${ROMAN_CHORDS.length} entries`);
  }
  return row;
}

function chordAt(index: number): RomanChord {
  const chord = ROMAN_CHORDS[index];
  if (chord === undefined) {
    throw new Error(`unknown roman chord index ${index}`);
  }
  return chord;
}

function emptyTransitionCounts(): ChordTransitionCounts {
  return {
    I: { I: 0, ii: 0, iii: 0, IV: 0, V: 0, vi: 0 },
    ii: { I: 0, ii: 0, iii: 0, IV: 0, V: 0, vi: 0 },
    iii: { I: 0, ii: 0, iii: 0, IV: 0, V: 0, vi: 0 },
    IV: { I: 0, ii: 0, iii: 0, IV: 0, V: 0, vi: 0 },
    V: { I: 0, ii: 0, iii: 0, IV: 0, V: 0, vi: 0 },
    vi: { I: 0, ii: 0, iii: 0, IV: 0, V: 0, vi: 0 },
  };
}

function classifyQuality(suffix: string): ChordQuality {
  const lower = suffix.toLowerCase();
  if (lower.includes('dim') || lower.includes('o')) return 'diminished';
  if (lower.includes('aug') || lower.includes('+')) return 'augmented';
  if (lower.startsWith('min') || (lower.startsWith('m') && !lower.startsWith('maj'))) return 'minor';
  if (lower.includes('7') && !lower.includes('maj')) return 'dominant';
  if (lower.includes('maj') || lower === '' || lower.startsWith('add') || lower.startsWith('sus')) {
    return 'major';
  }
  return 'unknown';
}

/**
 * Parses the Chord Symbol.
 */
export function parseChordSymbol(token: string): ParsedChordSymbol | null {
  const head = token.trim().split('/')[0]?.replace(/^[[(]+|[\])]+$/g, '') ?? '';
  if (head.length === 0 || head === 'N' || head.toLowerCase() === 'nan') return null;
  const match = /^([A-G])([#bs]?)(.*)$/.exec(head);
  if (match === null) return null;
  const natural = match[1];
  const accidental = match[2] ?? '';
  const suffix = match[3] ?? '';
  const base = natural === undefined ? undefined : NATURAL_ROOT_PC[natural];
  if (base === undefined) return null;
  const accidentalShift = accidental === '#' || accidental === 's' ? 1 : accidental === 'b' ? -1 : 0;
  return {
    raw: token,
    rootPc: pitchClass(base + accidentalShift),
    quality: classifyQuality(suffix),
  };
}

/**
 * Parses the Chordonomicon Chord Stream.
 */
export function parseChordonomiconChordStream(source: string): string[] {
  return source
    .replace(/[|,;]+/g, ' ')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0 && parseChordSymbol(token) !== null);
}

/**
 * Handles the maybe normalize Chordonomicon Section Label workflow.
 */
export function normalizeChordonomiconSectionLabel(label: string): SongSectionName | null {
  const normalized = label
    .trim()
    .toLowerCase()
    .replace(/-\d+$/g, '')
    .replace(/_\d+$/g, '')
    .replace(/-/g, '_');
  return SECTION_TAG_MAP[normalized] ?? null;
}

/**
 * Parses the Chordonomicon Sections.
 */
export function parseChordonomiconSections(source: string): ChordonomiconSection[] {
  const sections: ChordonomiconSection[] = [];
  let activeLabel: string | null = null;
  let activeSection: SongSectionName | null = null;
  let activeTokens: string[] = [];

  function flush(): void {
    if (activeLabel === null || activeSection === null || activeTokens.length === 0) return;
    sections.push({
      section: activeSection,
      label: activeLabel,
      tokens: activeTokens.filter((token) => parseChordSymbol(token) !== null),
    });
  }

  for (const token of source.split(/\s+/)) {
    const trimmed = token.trim();
    if (trimmed.length === 0) continue;
    const tag = /^<([^>]+)>$/.exec(trimmed);
    if (tag !== null) {
      flush();
      activeLabel = tag[1] ?? '';
      activeSection = normalizeChordonomiconSectionLabel(activeLabel);
      activeTokens = [];
      continue;
    }
    if (activeSection !== null) activeTokens.push(trimmed);
  }
  flush();
  return sections.filter((section) => section.tokens.length > 0);
}

/**
 * Handles the maybe project Chord To Roman workflow.
 */
export function projectChordToRoman(chord: ParsedChordSymbol, tonicPc: number): RomanChord | null {
  const degree = pitchClass(chord.rootPc - tonicPc);
  for (const roman of ROMAN_CHORDS) {
    if (ROMAN_ROOTS[roman] === degree) return roman;
  }
  return null;
}

/**
 * Returns whether is Diatonic Quality Compatible is true.
 */
export function isDiatonicQualityCompatible(roman: RomanChord, quality: ChordQuality): boolean {
  if (quality === 'unknown') return true;
  if (roman === 'ii' || roman === 'iii' || roman === 'vi') {
    return quality === 'minor' || quality === 'diminished';
  }
  if (roman === 'V') {
    return quality === 'major' || quality === 'dominant' || quality === 'augmented';
  }
  return quality === 'major' || quality === 'dominant' || quality === 'augmented';
}

/**
 * Handles the maybe infer Major Tonic workflow.
 */
export function inferMajorTonic(chords: readonly ParsedChordSymbol[]): KeyInference {
  let best: KeyInference = { tonicPc: 0, score: -1, projectedCount: 0 };
  for (let tonicPc = 0; tonicPc < 12; tonicPc++) {
    let score = 0;
    let projectedCount = 0;
    for (const chord of chords) {
      const roman = projectChordToRoman(chord, tonicPc);
      if (roman === null) continue;
      projectedCount++;
      score += isDiatonicQualityCompatible(roman, chord.quality) ? 3 : 1;
    }
    if (score > best.score) best = { tonicPc, score, projectedCount };
  }
  return best;
}

/**
 * Handles the maybe project Chord Stream To Roman workflow.
 */
export function projectChordStreamToRoman(tokens: readonly string[], tonicPc?: number): RomanChord[] {
  const parsed = tokens.flatMap((token) => {
    const chord = parseChordSymbol(token);
    return chord === null ? [] : [chord];
  });
  const inferredTonic = tonicPc ?? inferMajorTonic(parsed).tonicPc;
  return parsed.flatMap((chord) => {
    const roman = projectChordToRoman(chord, inferredTonic);
    return roman === null ? [] : [roman];
  });
}

/**
 * Handles the maybe project Section To Roman workflow.
 */
export function projectSectionToRoman(section: ChordonomiconSection, tonicPc: number): RomanChord[] {
  return projectChordStreamToRoman(section.tokens, tonicPc);
}

/**
 * Handles the maybe compress Adjacent Roman Chords workflow.
 */
export function compressAdjacentRomanChords(chords: readonly RomanChord[]): RomanChord[] {
  const out: RomanChord[] = [];
  for (const chord of chords) {
    if (out[out.length - 1] !== chord) out.push(chord);
  }
  return out;
}

/**
 * Handles the maybe shortest Exact Roman Cycle workflow.
 */
export function shortestExactRomanCycle(chords: readonly RomanChord[]): RomanChord[] {
  if (chords.length <= 1) return [...chords];
  for (let period = 1; period <= chords.length; period++) {
    let exact = true;
    for (let index = 0; index < chords.length; index++) {
      if (chords[index] !== chords[index % period]) {
        exact = false;
        break;
      }
    }
    if (exact) return chords.slice(0, period);
  }
  return [...chords];
}

/**
 * Handles the maybe section Chord Windows workflow.
 */
export function sectionChordWindows(chords: readonly RomanChord[], chordCount: number): RomanChord[][] {
  const count = Math.max(1, Math.floor(chordCount));
  const windows: RomanChord[][] = [];
  if (chords.length < count) return windows;
  for (let index = 0; index + count <= chords.length; index++) {
    windows.push(chords.slice(index, index + count));
  }
  return windows;
}

function chordSetKey(section: SongSectionName, chords: readonly RomanChord[]): string {
  return `${section}:${chords.join(' ')}`;
}

function inferTonicForSections(sections: readonly ChordonomiconSection[]): number {
  const parsed = sections.flatMap((section) =>
    section.tokens.flatMap((token) => {
      const chord = parseChordSymbol(token);
      return chord === null ? [] : [chord];
    })
  );
  return inferMajorTonic(parsed).tonicPc;
}

/**
 * Handles the maybe estimate Popular Section Chord Sets workflow.
 */
export function estimatePopularSectionChordSets(
  songs: readonly string[],
  chordCount: number = 4,
  topPerSection: number = 8
): SectionChordSetCandidate[] {
  const safeChordCount = Math.max(1, Math.floor(chordCount));
  const counts = new Map<string, { section: SongSectionName; chords: RomanChord[]; count: number }>();
  const sectionTotals = new Map<SongSectionName, number>();

  for (const song of songs) {
    const sections = parseChordonomiconSections(song);
    if (sections.length === 0) continue;
    const tonicPc = inferTonicForSections(sections);
    for (const section of sections) {
      const romans = compressAdjacentRomanChords(projectSectionToRoman(section, tonicPc));
      const seenInSection = new Set<string>();
      for (const window of sectionChordWindows(romans, safeChordCount)) {
        const key = chordSetKey(section.section, window);
        if (seenInSection.has(key)) continue;
        seenInSection.add(key);
        const previous = counts.get(key);
        if (previous === undefined) {
          counts.set(key, { section: section.section, chords: window, count: 1 });
        } else {
          previous.count += 1;
        }
        sectionTotals.set(section.section, (sectionTotals.get(section.section) ?? 0) + 1);
      }
    }
  }

  const bySection = new Map<SongSectionName, SectionChordSetCandidate[]>();
  for (const item of counts.values()) {
    const total = sectionTotals.get(item.section) ?? 1;
    const candidate: SectionChordSetCandidate = {
      section: item.section,
      chordCount: safeChordCount,
      chords: item.chords,
      count: item.count,
      probability: item.count / total,
    };
    const bucket = bySection.get(item.section) ?? [];
    bucket.push(candidate);
    bySection.set(item.section, bucket);
  }

  return [...bySection.values()].flatMap((bucket) =>
    bucket
      .sort((left, right) => right.count - left.count || left.chords.join(' ').localeCompare(right.chords.join(' ')))
      .slice(0, Math.max(1, Math.floor(topPerSection)))
  );
}

/**
 * Handles the maybe estimate Popular Section Chord Cycles workflow.
 */
export function estimatePopularSectionChordCycles(
  songs: readonly string[],
  topPerSection: number = 8
): SectionChordSetCandidate[] {
  const counts = new Map<string, { section: SongSectionName; chords: RomanChord[]; count: number }>();
  const sectionTotals = new Map<SongSectionName, number>();

  for (const song of songs) {
    const sections = parseChordonomiconSections(song);
    if (sections.length === 0) continue;
    const tonicPc = inferTonicForSections(sections);
    for (const section of sections) {
      const cycle = shortestExactRomanCycle(compressAdjacentRomanChords(projectSectionToRoman(section, tonicPc)));
      if (cycle.length === 0) continue;
      const key = chordSetKey(section.section, cycle);
      const previous = counts.get(key);
      if (previous === undefined) {
        counts.set(key, { section: section.section, chords: cycle, count: 1 });
      } else {
        previous.count += 1;
      }
      sectionTotals.set(section.section, (sectionTotals.get(section.section) ?? 0) + 1);
    }
  }

  const bySection = new Map<SongSectionName, SectionChordSetCandidate[]>();
  for (const item of counts.values()) {
    const total = sectionTotals.get(item.section) ?? 1;
    const candidate: SectionChordSetCandidate = {
      section: item.section,
      chordCount: item.chords.length,
      chords: item.chords,
      count: item.count,
      probability: item.count / total,
    };
    const bucket = bySection.get(item.section) ?? [];
    bucket.push(candidate);
    bySection.set(item.section, bucket);
  }

  return [...bySection.values()].flatMap((bucket) =>
    bucket
      .sort((left, right) => right.count - left.count || left.chords.length - right.chords.length || left.chords.join(' ').localeCompare(right.chords.join(' ')))
      .slice(0, Math.max(1, Math.floor(topPerSection)))
  );
}

/**
 * Handles the maybe count Roman Transitions workflow.
 */
export function countRomanTransitions(progressions: readonly (readonly RomanChord[])[]): ChordTransitionCounts {
  const counts = emptyTransitionCounts();
  for (const progression of progressions) {
    for (let index = 0; index + 1 < progression.length; index++) {
      const from = progression[index];
      const to = progression[index + 1];
      if (from === undefined || to === undefined) continue;
      counts[from][to] += 1;
    }
  }
  return counts;
}

function normalizeRow(counts: readonly number[], rowMass: number, floorMass: number): number[] {
  const smoothed = counts.map((count) => count + floorMass);
  const total = smoothed.reduce((sum, value) => sum + value, 0);
  if (total === 0) {
    throw new Error('cannot normalize an empty transition row without floor mass');
  }
  const exact = smoothed.map((value) => (value * rowMass) / total);
  const row = exact.map((value) => Math.floor(value));
  let remainder = rowMass - row.reduce((sum, value) => sum + value, 0);
  const order = exact
    .map((value, index) => ({ index, fraction: value - Math.floor(value) }))
    .sort((left, right) => right.fraction - left.fraction);
  let cursor = 0;
  while (remainder > 0) {
    const target = order[cursor % order.length];
    if (target === undefined) throw new Error('missing rounding target');
    row[target.index] = (row[target.index] ?? 0) + 1;
    remainder--;
    cursor++;
  }
  return row;
}

/**
 * Handles the maybe normalize Transition Counts workflow.
 */
export function normalizeTransitionCounts(
  counts: ChordTransitionCounts,
  rowMass: number = CHORDONOMICON_BASELINE_ROW_MASS,
  floorMass: number = 1
): ChordTransitionMatrix {
  const matrix = {} as Record<RomanChord, number[]>;
  for (const from of ROMAN_CHORDS) {
    const rowCounts = ROMAN_CHORDS.map((to) => counts[from][to]);
    const rowTotal = rowCounts.reduce((sum, value) => sum + value, 0);
    matrix[from] =
      rowTotal === 0
        ? [...rowFor(BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS, from)]
        : normalizeRow(rowCounts, rowMass, floorMass);
  }
  return matrix;
}

/**
 * Handles the maybe estimate Transition Matrix From Chord Streams workflow.
 */
export function estimateTransitionMatrixFromChordStreams(
  chordStreams: readonly (readonly string[])[],
  rowMass: number = CHORDONOMICON_BASELINE_ROW_MASS
): ChordTransitionMatrix {
  const progressions = chordStreams.map((stream) => projectChordStreamToRoman(stream));
  return normalizeTransitionCounts(countRomanTransitions(progressions), rowMass);
}

/**
 * Handles the maybe assert Chord Rows Stochastic workflow.
 */
export function assertChordRowsStochastic(
  matrix: ChordTransitionMatrix = BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS,
  rowMass: number = CHORDONOMICON_BASELINE_ROW_MASS
): true {
  for (const chord of ROMAN_CHORDS) {
    const sum = rowFor(matrix, chord).reduce((acc, value) => acc + value, 0);
    if (sum !== rowMass) {
      throw new Error(`transition row ${chord} sums to ${sum}, expected ${rowMass}`);
    }
  }
  return true;
}

/**
 * Handles the maybe transition Edges workflow.
 */
export function transitionEdges(
  matrix: ChordTransitionMatrix = BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS
): ChordTransitionEdge[] {
  assertChordRowsStochastic(matrix);
  return ROMAN_CHORDS.flatMap((from) =>
    rowFor(matrix, from).map((weight, index) => {
      const to = chordAt(index);
      return {
        from,
        to,
        weight,
        probability: weight / CHORDONOMICON_BASELINE_ROW_MASS,
        highEntropy: weight <= 10,
        fromClass: FUNCTION_CLASS[from],
        toClass: FUNCTION_CLASS[to],
      };
    })
  );
}

/**
 * Handles the maybe compile Chord Transition Kernel workflow.
 */
export function compileChordTransitionKernel(
  matrix: ChordTransitionMatrix = BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS,
  rowMass: number = CHORDONOMICON_BASELINE_ROW_MASS
): ChordTransitionKernelWitness {
  assertChordRowsStochastic(matrix, rowMass);
  const rows = ROMAN_CHORDS.map((chord) => compileKernelRow(rowMass, rowFor(matrix, chord)));
  const kernel = compileFiniteKernel(rows);
  return {
    rowMass,
    kernel,
    markov: compileMarkovWitness([kernel], 0),
  };
}

/**
 * Handles the maybe functional Flow Bonus workflow.
 */
export function functionalFlowBonus(from: RomanChord, to: RomanChord): number {
  const a = FUNCTION_CLASS[from];
  const b = FUNCTION_CLASS[to];
  if (a === 'dominant' && b === 'tonic') return to === 'I' ? 14 : 6;
  if (a === 'predominant' && b === 'dominant') return 10;
  if (a === 'tonic' && b === 'predominant') return 8;
  if (a === 'tonic' && b === 'dominant') return 4;
  return 0;
}

/**
 * Handles the maybe high Entropy Transition workflow.
 */
export function highEntropyTransition(
  from: RomanChord,
  to: RomanChord,
  matrix: ChordTransitionMatrix = BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS
): boolean {
  const index = ROMAN_CHORDS.indexOf(to);
  const weight = rowFor(matrix, from)[index];
  if (weight === undefined) throw new Error(`missing transition ${from}->${to}`);
  return weight <= 10;
}

/**
 * Handles the maybe choose Next Chord workflow.
 */
export function chooseNextChord(
  current: RomanChord,
  unitInterval: number,
  matrix: ChordTransitionMatrix = BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS
): RomanChord {
  assertChordRowsStochastic(matrix);
  const row = rowFor(matrix, current);
  const target = Math.max(0, Math.min(0.999999999, unitInterval)) * CHORDONOMICON_BASELINE_ROW_MASS;
  let cumulative = 0;
  for (let index = 0; index < row.length; index++) {
    cumulative += row[index] ?? 0;
    if (target < cumulative) return chordAt(index);
  }
  const finalChord = ROMAN_CHORDS[ROMAN_CHORDS.length - 1];
  if (finalChord === undefined) {
    throw new Error('Chordonomicon progression table is empty');
  }
  return finalChord;
}

/**
 * Handles the maybe deterministic Transition Unit workflow.
 */
export function deterministicTransitionUnit(seed: number): number {
  let x = seed >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 4294967296;
}

/**
 * Handles the maybe choose Deterministic Next Chord workflow.
 */
export function chooseDeterministicNextChord(
  current: RomanChord,
  seed: number,
  matrix: ChordTransitionMatrix = BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS
): RomanChord {
  return chooseNextChord(current, deterministicTransitionUnit(seed), matrix);
}
