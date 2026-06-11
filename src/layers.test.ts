import { describe, expect, it } from 'bun:test';
import {
  createBuleyeanStack,
  getLayer,
  tickBuleyeanStack,
  measureBuleyeanStack,
  constraintCascade,
  contextualizationCascade,
  BULEYEAN_LAYERS,
} from './layers';
import { updateVoidBoundary } from '@a0n/gnosis/src/void';
import { buleyeanDistribution, assertPositivity } from './buleyean';

describe('createBuleyeanStack', () => {
  it('creates four layers with correct names', () => {
    const stack = createBuleyeanStack(4);
    expect(stack.layers).toHaveLength(4);
    expect(stack.layers[0].name).toBe('retrocausal');
    expect(stack.layers[1].name).toBe('bayesian');
    expect(stack.layers[2].name).toBe('frequentist');
    expect(stack.layers[3].name).toBe('solomonoff');
  });

  it('creates layers with correct timescales', () => {
    const stack = createBuleyeanStack(4);
    expect(stack.layers[0].timescale).toBe('lifetime');
    expect(stack.layers[1].timescale).toBe('weeks');
    expect(stack.layers[2].timescale).toBe('minutes');
    expect(stack.layers[3].timescale).toBe('generational');
  });

  it('creates resonance links between retrocausal and solomonoff', () => {
    const stack = createBuleyeanStack(4);
    expect(stack.resonances).toHaveLength(2);
    expect(stack.resonances[0].sourceIdx).toBe(0);
    expect(stack.resonances[0].targetIdx).toBe(3);
    expect(stack.resonances[1].sourceIdx).toBe(3);
    expect(stack.resonances[1].targetIdx).toBe(0);
  });

  it('uses specified dimensions', () => {
    const stack = createBuleyeanStack(8);
    for (const layer of stack.layers) {
      expect(layer.boundary.counts).toHaveLength(8);
    }
  });
});

describe('getLayer', () => {
  it('returns the correct layer by name', () => {
    const stack = createBuleyeanStack(4);
    expect(getLayer(stack, 'retrocausal')).toBe(stack.layers[0]);
    expect(getLayer(stack, 'bayesian')).toBe(stack.layers[1]);
    expect(getLayer(stack, 'frequentist')).toBe(stack.layers[2]);
    expect(getLayer(stack, 'solomonoff')).toBe(stack.layers[3]);
  });
});

describe('tickBuleyeanStack', () => {
  it('applies decay to all layers', () => {
    const stack = createBuleyeanStack(3);
    // Add some void
    updateVoidBoundary(stack.layers[0].boundary, 0, 10);
    updateVoidBoundary(stack.layers[2].boundary, 1, 5);

    const beforeRetro = stack.layers[0].boundary.counts[0];
    tickBuleyeanStack(stack);
    // Lifetime timescale decays very slowly, but upward/downward flows change things
    // Just verify it ran without error and values changed
    const afterRetro = stack.layers[0].boundary.counts[0];
    expect(afterRetro).not.toBe(beforeRetro);
  });

  it('maintains positivity of Buleyean distribution after tick', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[0].boundary, 0, 100);
    updateVoidBoundary(stack.layers[1].boundary, 2, 50);
    updateVoidBoundary(stack.layers[2].boundary, 1, 25);
    updateVoidBoundary(stack.layers[3].boundary, 3, 10);

    tickBuleyeanStack(stack);

    for (const layer of stack.layers) {
      expect(assertPositivity(layer.boundary)).toBe(true);
    }
  });
});

describe('constraintCascade', () => {
  it('returns bias vectors for each layer pair', () => {
    const stack = createBuleyeanStack(4);
    updateVoidBoundary(stack.layers[0].boundary, 0, 10);
    const cascade = constraintCascade(stack);
    expect(cascade.retrocausalToBayesian).toHaveLength(4);
    expect(cascade.bayesianToFrequentist).toHaveLength(4);
    expect(cascade.frequentistToSolomonoff).toHaveLength(4);
  });
});

describe('contextualizationCascade', () => {
  it('returns modulation vectors for each layer pair', () => {
    const stack = createBuleyeanStack(4);
    const cascade = contextualizationCascade(stack);
    expect(cascade.solomonoffToFrequentist).toHaveLength(4);
    expect(cascade.frequentistToBayesian).toHaveLength(4);
    expect(cascade.bayesianToRetrocausal).toHaveLength(4);
  });
});

describe('measureBuleyeanStack', () => {
  it('returns global and per-layer measurements', () => {
    const stack = createBuleyeanStack(4);
    const m = measureBuleyeanStack(stack);
    expect(m.global).toBeDefined();
    expect(m.global.entropy).toBeDefined();
    expect(m.layers).toHaveLength(4);
    expect(m.layers[0].name).toBe('retrocausal');
    expect(m.layers[3].name).toBe('solomonoff');
  });
});
