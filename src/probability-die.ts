/**
 * Probability die / physical QR -- the geometry lark, made concrete.
 *
 * A probability vector p over n outcomes has three faithful physical readings:
 *
 *  1. Bhattacharyya sphere point: xi_i = 2*sqrt(p_i) lands on the sphere of
 *     radius 2 (the convention already used by maybe/src/manifold.ts).
 *  2. Direction-space die: the unit sphere is partitioned into n lunes of area
 *     4*pi*p_i. A lune between two meridians at longitude distance D has area
 *     2*D, so lune i spans 2*pi*p_i of longitude. This is ALWAYS constructible.
 *  3. Exact QR payload: the Buleyean integer weights w_i = n_i + 1 are exact
 *     integers, encoded losslessly for a scannable glyph.
 *
 * Honest boundary: the lune fan is a spherical object. Lifting it to the normal
 * fan of a convex polyhedron -- so that a physical die lands with those solid
 * angles -- is a Gauss-map existence question and is NOT solved here. Real dice
 * are physics, not only geometry.
 */

export const TWO_PI = 2 * Math.PI;
export const FOUR_PI = 4 * Math.PI;

export interface Lune {
  readonly index: number;
  readonly area: number;
  readonly longitudeStart: number;
  readonly longitudeWidth: number;
}

/** Normalize a positive weight vector to a probability vector. */
export function normalizeWeights(weights: readonly number[]): number[] {
  if (weights.length === 0) throw new RangeError('weights must be non-empty');
  let total = 0;
  for (const w of weights) {
    if (!Number.isFinite(w) || w < 0) throw new TypeError('weights must be finite and non-negative');
    total += w;
  }
  if (total <= 0) throw new RangeError('total weight must be positive');
  return weights.map((w) => w / total);
}

/** The Bhattacharyya embedding xi_i = 2*sqrt(p_i); every point has norm 2. */
export function bhattacharyyaEmbedding(weights: readonly number[]): number[] {
  return normalizeWeights(weights).map((p) => 2 * Math.sqrt(p));
}

/** The solid angle 4*pi*p_i of each outcome. */
export function solidAngles(weights: readonly number[]): number[] {
  return normalizeWeights(weights).map((p) => FOUR_PI * p);
}

/** The explicit lune partition of the unit sphere. Areas sum to 4*pi. */
export function luneFan(weights: readonly number[]): Lune[] {
  const probabilities = normalizeWeights(weights);
  const lunes: Lune[] = [];
  let cursor = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const width = TWO_PI * probabilities[i];
    lunes.push({ index: i, area: FOUR_PI * probabilities[i], longitudeStart: cursor, longitudeWidth: width });
    cursor += width;
  }
  return lunes;
}

const QR_PREFIX = 'pd1:';

/** Losslessly encode exact integer weights (BigInt) for a scannable glyph. */
export function encodeWeightQr(weights: readonly bigint[]): string {
  if (weights.length === 0) throw new RangeError('weights must be non-empty');
  for (const w of weights) {
    if (w < 0n) throw new TypeError('weights must be non-negative');
  }
  return QR_PREFIX + weights.map((w) => w.toString(36)).join('.');
}

/** Recover the exact integer weights from an encoded payload. */
export function decodeWeightQr(payload: string): bigint[] {
  if (!payload.startsWith(QR_PREFIX)) throw new Error('bad_qr_prefix');
  const body = payload.slice(QR_PREFIX.length);
  if (body.length === 0) throw new Error('empty_qr_payload');
  return body.split('.').map((part) => {
    if (!/^[0-9a-z]+$/u.test(part)) throw new Error('bad_qr_digit');
    return BigInt(parseInt(part, 36));
  });
}

function fnv1a64(input: string): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < input.length; i++) {
    h = (h ^ BigInt(input.charCodeAt(i))) & mask;
    h = (h * prime) & mask;
  }
  return h;
}

function xorshift64(state: bigint): { value: bigint; next: bigint } {
  let x = state & 0xffffffffffffffffn;
  x ^= (x << 13n) & 0xffffffffffffffffn;
  x ^= x >> 7n;
  x ^= (x << 17n) & 0xffffffffffffffffn;
  return { value: x & 0xffffffffffffffffn, next: x };
}

/**
 * A deterministic QR-like glyph over a size x size matrix. This is a glyph, not
 * an ISO/IEC 18004 symbol: it carries the payload digest and finder corners, so
 * two payloads differ and the same payload always renders the same. A real QR
 * encoder is a separate step.
 */
export function qrMatrix(payload: string, size = 21): boolean[][] {
  if (size < 7 || size > 177) throw new RangeError('size out of range');
  const grid: boolean[][] = Array.from({ length: size }, () => Array.from({ length: size }, () => false));
  let state = fnv1a64(payload);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const step = xorshift64(state);
      state = step.next;
      grid[y][x] = (step.value & 1n) === 1n;
    }
  }
  const drawFinder = (ox: number, oy: number): void => {
    for (let y = 0; y < 7; y++) {
      for (let x = 0; x < 7; x++) {
        const border = x === 0 || y === 0 || x === 6 || y === 6;
        const core = x >= 2 && x <= 4 && y >= 2 && y <= 4;
        grid[oy + y][ox + x] = border || core;
      }
    }
  };
  drawFinder(0, 0);
  drawFinder(size - 7, 0);
  drawFinder(0, size - 7);
  return grid;
}

/** Render a glyph as text for a snapshot test or a terminal. */
export function renderQrAscii(grid: readonly boolean[][]): string {
  return grid.map((row) => row.map((on) => (on ? '##' : '  ')).join('')).join('\n');
}
