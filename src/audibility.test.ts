import { describe, expect, it } from 'bun:test';

import { buildHypothesisSpace, type Hypothesis } from './abduction.js';
import {
  auditAudibility,
  assertAudible,
  separability,
  DEFAULT_AUDIBILITY_EPSILON,
} from './audibility.js';

const hyp = (id: number, name: string, prior: number): Hypothesis => ({ id, name, prior });

/** Two causes with DIFFERENT signatures — a probe exists. */
function separableSpace() {
  return buildHypothesisSpace({
    hypotheses: [hyp(0, 'disk-full', 0.5), hyp(1, 'oom-kill', 0.5)],
    features: ['write-error', 'rss-spike'],
    // disk-full: writes fail, memory flat.  oom-kill: writes fine, memory spikes.
    pFeatureGivenHypothesis: new Float64Array([0.95, 0.05, 0.05, 0.95]),
  });
}

/** Two causes with the SAME signature — no probe can ever separate them. */
function inaudibleSpace() {
  return buildHypothesisSpace({
    hypotheses: [
      hyp(0, 'config-key-typo', 0.5),
      hyp(1, 'binding-never-requested', 0.5),
    ],
    // Both present as "the binding is absent". That was a real bug: a dropped Durable Object
    // binding was indistinguishable from one the app never asked for.
    features: ['binding-absent', 'startup-ok'],
    pFeatureGivenHypothesis: new Float64Array([1.0, 1.0, 1.0, 1.0]),
  });
}

describe('audibility: can the evidence separate the explanations at all', () => {
  it('accepts a space where every pair differs somewhere', () => {
    const report = auditAudibility(separableSpace());
    expect(report.audible).toBe(true);
    expect(report.indistinguishable).toHaveLength(0);
    expect(report.minSeparation).toBeGreaterThan(DEFAULT_AUDIBILITY_EPSILON);
  });

  it('names the pair no observation can separate', () => {
    const report = auditAudibility(inaudibleSpace());
    expect(report.audible).toBe(false);
    expect(report.indistinguishable).toHaveLength(1);
    const pair = report.indistinguishable[0]!;
    const names = [pair.a.name, pair.b.name].sort();
    expect(names).toEqual(['binding-never-requested', 'config-key-typo']);
    expect(pair.maxGap).toBeLessThanOrEqual(DEFAULT_AUDIBILITY_EPSILON);
  });

  it('assertAudible names the colliding pair rather than just refusing', () => {
    // A bare "space is inaudible" would itself be a silent failure.
    expect(() => assertAudible(inaudibleSpace())).toThrow(/config-key-typo/);
    expect(() => assertAudible(inaudibleSpace())).toThrow(/binding-never-requested/);
    expect(() => assertAudible(separableSpace())).not.toThrow();
  });

  it('reports a flat feature as inert, distinctly from an indistinguishable pair', () => {
    // `always-logs` fires for everything: it costs a probe and buys nothing. That is a
    // different defect from a colliding pair — you fix it by DELETING the feature, whereas a
    // colliding pair needs a NEW one.
    const space = buildHypothesisSpace({
      hypotheses: [hyp(0, 'a', 0.5), hyp(1, 'b', 0.5)],
      features: ['always-logs', 'only-a'],
      pFeatureGivenHypothesis: new Float64Array([1.0, 0.9, 1.0, 0.1]),
    });
    const report = auditAudibility(space);
    expect(report.audible).toBe(true);
    expect(report.inertFeatures.map((f) => f.feature)).toEqual(['always-logs']);
  });

  it('separability points at the probe worth spending', () => {
    const result = separability(separableSpace(), 0, 1);
    expect(result.separable).toBe(true);
    expect(result.separator).not.toBeNull();
    expect(result.gap).toBeGreaterThan(0.5);
  });

  it('reports no separator when none exists, instead of picking one arbitrarily', () => {
    const result = separability(inaudibleSpace(), 0, 1);
    expect(result.separable).toBe(false);
    expect(result.separator).toBeNull();
  });

  it('a near-collision stays audible but shows a small minSeparation', () => {
    // Technically separable, practically fragile — worth surfacing as a number rather than
    // collapsing to a boolean.
    const space = buildHypothesisSpace({
      hypotheses: [hyp(0, 'a', 0.5), hyp(1, 'b', 0.5)],
      features: ['f'],
      pFeatureGivenHypothesis: new Float64Array([0.5, 0.5001]),
    });
    const report = auditAudibility(space);
    expect(report.audible).toBe(true);
    expect(report.minSeparation).toBeLessThan(0.001);
  });

  it('rejects an unknown hypothesis id rather than guessing', () => {
    expect(() => separability(separableSpace(), 0, 99)).toThrow(/unknown hypothesis/);
  });
});
