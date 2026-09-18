import { describe, expect, it } from 'bun:test';
import {
  bhattacharyyaEmbedding,
  decodeWeightQr,
  encodeWeightQr,
  FOUR_PI,
  luneFan,
  normalizeWeights,
  qrMatrix,
  renderQrAscii,
  solidAngles,
  TWO_PI,
} from './probability-die';

describe('probability die -- the spherical realizations', () => {
  it('normalizes positive weights and rejects degenerate input', () => {
    const p = normalizeWeights([1, 1, 2]);
    expect(p).toEqual([0.25, 0.25, 0.5]);
    expect(() => normalizeWeights([])).toThrow();
    expect(() => normalizeWeights([1, -1])).toThrow();
    expect(() => normalizeWeights([0, 0])).toThrow();
  });

  it('lands every distribution on the sphere of radius two', () => {
    for (const weights of [[1, 1, 1, 1], [3, 1, 4, 1, 5], [0.1, 0.9], [7]]) {
      const xi = bhattacharyyaEmbedding(weights);
      const normSq = xi.reduce((s, x) => s + x * x, 0);
      expect(normSq).toBeCloseTo(4, 12);
    }
  });

  it('gives solid angles summing to four pi', () => {
    const angles = solidAngles([2, 3, 5]);
    expect(angles.reduce((a, b) => a + b, 0)).toBeCloseTo(FOUR_PI, 12);
  });

  it('partitions the sphere into lunes whose areas are exactly four pi p_i', () => {
    const weights = [1, 2, 3, 4];
    const p = normalizeWeights(weights);
    const lunes = luneFan(weights);
    expect(lunes.length).toBe(4);
    let widthSum = 0;
    let areaSum = 0;
    for (let i = 0; i < lunes.length; i++) {
      expect(lunes[i].area).toBeCloseTo(FOUR_PI * p[i], 12);
      expect(lunes[i].longitudeWidth).toBeCloseTo(TWO_PI * p[i], 12);
      widthSum += lunes[i].longitudeWidth;
      areaSum += lunes[i].area;
    }
    expect(widthSum).toBeCloseTo(TWO_PI, 12);
    expect(areaSum).toBeCloseTo(FOUR_PI, 12);
    // Contiguous: each lune starts where the previous one ended.
    expect(lunes[0].longitudeStart).toBe(0);
    for (let i = 1; i < lunes.length; i++) {
      expect(lunes[i].longitudeStart).toBeCloseTo(
        lunes[i - 1].longitudeStart + lunes[i - 1].longitudeWidth,
        12,
      );
    }
  });
});

describe('probability die -- the exact QR payload', () => {
  it('round-trips Buleyean integer weights exactly, including the kaiju capacity', () => {
    const weights = [1n, 51_840n, 8_315_553_613_086_720_000n, 0n, 196_560n];
    const payload = encodeWeightQr(weights);
    expect(decodeWeightQr(payload)).toEqual(weights);
  });

  it('is compact for small weights and rejects malformed payloads', () => {
    expect(encodeWeightQr([1n, 1n, 1n])).toBe('pd1:1.1.1');
    expect(() => decodeWeightQr('nope')).toThrow('bad_qr_prefix');
    expect(() => decodeWeightQr('pd1:')).toThrow('empty_qr_payload');
    expect(() => decodeWeightQr('pd1:1.Z')).toThrow('bad_qr_digit');
  });

  it('renders a deterministic glyph with finder corners, different per payload', () => {
    const a = qrMatrix('pd1:1.1.1', 21);
    const b = qrMatrix('pd1:1.1.1', 21);
    const c = qrMatrix('pd1:1.1.2', 21);
    expect(a.length).toBe(21);
    expect(a.every((row) => row.length === 21)).toBe(true);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    // Top-left finder border/corner protocol.
    expect(a[0][0]).toBe(true);
    expect(a[0][6]).toBe(true);
    expect(a[6][0]).toBe(true);
    expect(a[3][3]).toBe(true);
    expect(renderQrAscii(a).split('\n').length).toBe(21);
  });
});
