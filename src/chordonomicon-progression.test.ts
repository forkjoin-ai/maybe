import { describe, expect, it } from 'bun:test';

import {
  BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS,
  assertChordRowsStochastic,
  chooseNextChord,
  compileChordTransitionKernel,
  estimatePopularSectionChordSets,
  estimatePopularSectionChordCycles,
  estimateTransitionMatrixFromChordStreams,
  functionalFlowBonus,
  highEntropyTransition,
  parseChordonomiconChordStream,
  parseChordonomiconSections,
  projectChordStreamToRoman,
  transitionEdges,
} from './chordonomicon-progression';

describe('Chordonomicon progression kernel', () => {
  it('keeps each roman-function transition row at exact 100 mass', () => {
    expect(assertChordRowsStochastic()).toBe(true);

    const witness = compileChordTransitionKernel();
    expect(witness.kernel.input.totalMass).toBe(600);
    expect(witness.kernel.output.totalMass).toBe(600);
    expect(witness.kernel.lostMass).toBe(0);
    expect(witness.markov.totalShadow).toBe(0);
  });

  it('exposes human-friendly gravity and high-entropy transitions', () => {
    expect(functionalFlowBonus('ii', 'V')).toBeGreaterThan(functionalFlowBonus('iii', 'IV'));
    expect(functionalFlowBonus('V', 'I')).toBeGreaterThan(functionalFlowBonus('V', 'vi'));
    expect(highEntropyTransition('V', 'iii')).toBe(true);
    expect(highEntropyTransition('V', 'I')).toBe(false);

    const edges = transitionEdges(BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS);
    expect(edges).toHaveLength(36);
    expect(edges.find((edge) => edge.from === 'ii' && edge.to === 'V')?.probability).toBe(0.5);
  });

  it('chooses weighted transitions deterministically from the cumulative row', () => {
    expect(chooseNextChord('I', 0.4)).toBe('IV');
    expect(chooseNextChord('V', 0.2)).toBe('I');
    expect(chooseNextChord('iii', 0.8)).toBe('vi');
  });

  it('projects Chordonomicon-style chord strings into roman transitions', () => {
    const stream = parseChordonomiconChordStream('C F G C Amin Dmin G7 C');
    expect(projectChordStreamToRoman(stream)).toEqual(['I', 'IV', 'V', 'I', 'vi', 'ii', 'V', 'I']);

    const matrix = estimateTransitionMatrixFromChordStreams([stream]);
    expect(assertChordRowsStochastic(matrix)).toBe(true);
    expect(matrix.V[0]).toBeGreaterThan(matrix.V[2]);
    expect(matrix.ii[4]).toBeGreaterThan(matrix.ii[2]);
  });

  it('parses section tags and estimates data-derived section chord sets', () => {
    const songs = [
      '<verse_1> C Amin F G C Amin F G <chorus_1> C G Amin F C G Amin F',
      '<verse_1> Amin G Dmin F Amin G Dmin F <chorus_1> C G Amin F',
      '<bridge_1> Dmin G Emin Amin <chorus_1> C G Amin F',
    ];

    const sections = parseChordonomiconSections(songs[0]);
    expect(sections.map((section) => section.section)).toEqual(['verse', 'chorus']);

    const fourChordSets = estimatePopularSectionChordSets(songs, 4, 4);
    expect(fourChordSets.find((set) => set.section === 'chorus')?.chords).toEqual(['I', 'V', 'vi', 'IV']);
    const widerFourChordSets = estimatePopularSectionChordSets(songs, 4, 12);
    expect(widerFourChordSets.find((set) => set.section === 'verse' && set.chords.join(' ') === 'vi V ii IV')?.count).toBe(1);

    const oneChordSets = estimatePopularSectionChordSets(songs, 1, 2);
    expect(oneChordSets.find((set) => set.section === 'chorus')?.chordCount).toBe(1);
    expect(oneChordSets.find((set) => set.section === 'chorus')?.count).toBeGreaterThan(1);

    const corpusShapedCycles = estimatePopularSectionChordCycles(songs, 4);
    expect(corpusShapedCycles.find((set) => set.section === 'chorus')?.chords).toEqual(['I', 'V', 'vi', 'IV']);
    expect(corpusShapedCycles.find((set) => set.section === 'verse' && set.chords.join(' ') === 'vi V ii IV')?.chordCount).toBe(4);
  });
});
