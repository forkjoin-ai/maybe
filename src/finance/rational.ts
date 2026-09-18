/**
 * rational.ts -- exact BigInt rational arithmetic for the finance primitives.
 *
 * Every probability and every risk number in src/finance is an exact rational:
 * a pair of BigInt numerator/denominator reduced to lowest terms. Floating
 * point appears only in fields explicitly named for display, and no truth claim
 * is ever made on those floats.
 *
 * Nothing here is financial advice. These are decision and risk arithmetic
 * primitives; the caller owns every judgement about what the numbers mean.
 */

/** An exact rational number reduced to lowest terms, denominator > 0. */
export interface ExactFraction {
  readonly numerator: bigint;
  readonly denominator: bigint;
}

/** Anything the finance primitives accept where a rational is expected. */
export type RationalInput = number | bigint | ExactFraction;

/** Euclid on BigInt magnitudes. gcd(0, 0) is 0. */
export function gcd(left: bigint, right: bigint): bigint {
  let a = left < 0n ? -left : left;
  let b = right < 0n ? -right : right;
  while (b !== 0n) {
    const next = a % b;
    a = b;
    b = next;
  }
  return a;
}

/**
 * Reduce numerator/denominator to lowest terms with a positive denominator.
 * Throws a RangeError when the denominator is zero. Zero reduces to 0/1.
 */
export function exactFraction(numerator: bigint, denominator: bigint): ExactFraction {
  if (denominator === 0n) {
    throw new RangeError('exact fraction denominator must be non-zero');
  }
  let n = numerator;
  let d = denominator;
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const divisor = gcd(n, d);
  const safe = divisor === 0n ? 1n : divisor;
  return { numerator: n / safe, denominator: d / safe };
}

/**
 * Read a finite JS number as an exact decimal rational. Integers are exact;
 * non-integers are parsed from the shortest decimal string (for example
 * 0.25 -> 1/4 and 1e-7 -> 1/10000000). No binary float residue enters the
 * fraction: the decimal spelling is the numerator source.
 */
export function fractionFromNumber(value: number): ExactFraction {
  if (!Number.isFinite(value)) {
    throw new RangeError('value = ' + String(value) + ' must be finite');
  }
  if (Number.isInteger(value)) {
    return exactFraction(BigInt(value), 1n);
  }
  const match = /^([+-]?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/.exec(value.toString());
  if (!match) {
    throw new RangeError('value = ' + String(value) + ' cannot be read as an exact decimal');
  }
  const sign = match[1] === '-' ? -1n : 1n;
  const integerPart = match[2]!;
  const fractionalPart = match[3] ?? '';
  const exponent = match[4] ? Number.parseInt(match[4]!, 10) : 0;
  const digits = BigInt(integerPart + fractionalPart);
  const netExponent = exponent - fractionalPart.length;
  let numerator = sign * digits;
  let denominator = 1n;
  if (netExponent >= 0) {
    numerator *= 10n ** BigInt(netExponent);
  } else {
    denominator = 10n ** BigInt(-netExponent);
  }
  return exactFraction(numerator, denominator);
}

/** Coerce any accepted rational input to exact lowest terms. */
export function toFraction(value: RationalInput, where = 'value'): ExactFraction {
  if (typeof value === 'bigint') return exactFraction(value, 1n);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new RangeError(where + ' = ' + String(value) + ' must be finite');
    }
    return fractionFromNumber(value);
  }
  if (value !== null && typeof value === 'object') {
    if (typeof value.numerator !== 'bigint' || typeof value.denominator !== 'bigint') {
      throw new TypeError(where + ' must carry bigint numerator and denominator');
    }
    return exactFraction(value.numerator, value.denominator);
  }
  throw new TypeError(where + ' must be a number, bigint, or ExactFraction');
}

/** -1, 0, or 1 as left is less than, equal to, or greater than right. */
export function compareFractions(left: RationalInput, right: RationalInput): number {
  const a = toFraction(left, 'left');
  const b = toFraction(right, 'right');
  const l = a.numerator * b.denominator;
  const r = b.numerator * a.denominator;
  return l < r ? -1 : l > r ? 1 : 0;
}

export function addFractions(left: RationalInput, right: RationalInput): ExactFraction {
  const a = toFraction(left, 'left');
  const b = toFraction(right, 'right');
  return exactFraction(
    a.numerator * b.denominator + b.numerator * a.denominator,
    a.denominator * b.denominator,
  );
}

export function subFractions(left: RationalInput, right: RationalInput): ExactFraction {
  const a = toFraction(left, 'left');
  const b = toFraction(right, 'right');
  return exactFraction(
    a.numerator * b.denominator - b.numerator * a.denominator,
    a.denominator * b.denominator,
  );
}

export function mulFractions(left: RationalInput, right: RationalInput): ExactFraction {
  const a = toFraction(left, 'left');
  const b = toFraction(right, 'right');
  return exactFraction(a.numerator * b.numerator, a.denominator * b.denominator);
}

export function divFractions(left: RationalInput, right: RationalInput): ExactFraction {
  const a = toFraction(left, 'left');
  const b = toFraction(right, 'right');
  if (b.numerator === 0n) {
    throw new RangeError('cannot divide by a zero rational');
  }
  return exactFraction(a.numerator * b.denominator, a.denominator * b.numerator);
}

export function isZeroFraction(value: RationalInput): boolean {
  return toFraction(value).numerator === 0n;
}

export function isPositiveFraction(value: RationalInput): boolean {
  return toFraction(value).numerator > 0n;
}

export function isNegativeFraction(value: RationalInput): boolean {
  return toFraction(value).numerator < 0n;
}

/** Approximate bit length of a positive BigInt (0 for 0 or negative). */
function bigintBitLength(value: bigint): number {
  if (value <= 0n) return 0;
  let bits = 0;
  let remaining = value;
  while (remaining >= 1n << 64n) {
    remaining >>= 64n;
    bits += 64;
  }
  let small = Number(remaining);
  while (small >= 1) {
    small = Math.floor(small / 2);
    bits += 1;
  }
  return bits;
}

/**
 * Display-only float for an exact rational. Fractions outside the double range
 * are bit-shifted down on both sides so the ratio stays finite instead of
 * collapsing to NaN. No truth claim is made on the returned number.
 */
export function fractionToNumber(value: RationalInput): number {
  const fraction = toFraction(value);
  if (fraction.numerator === 0n) return 0;
  const negative = fraction.numerator < 0n;
  const numerator = negative ? -fraction.numerator : fraction.numerator;
  const denominator = fraction.denominator;
  if (numerator === denominator) return negative ? -1 : 1;
  const bits = Math.max(bigintBitLength(numerator), bigintBitLength(denominator));
  let ratio: number;
  if (bits <= 1023) {
    ratio = Number(numerator) / Number(denominator);
  } else {
    const drop = BigInt(bits - 1023);
    const scaledDenominator = Number(denominator >> drop);
    ratio = scaledDenominator > 0 ? Number(numerator >> drop) / scaledDenominator : 0;
  }
  return negative ? -ratio : ratio;
}

/** Human-readable exact fraction, e.g. "3/4" or "5". */
export function fractionToString(value: RationalInput): string {
  const fraction = toFraction(value);
  return fraction.denominator === 1n
    ? String(fraction.numerator)
    : String(fraction.numerator) + '/' + String(fraction.denominator);
}

/** Clamp an exact rational into [low, high]. */
export function clampFraction(
  value: RationalInput,
  low: RationalInput,
  high: RationalInput,
): ExactFraction {
  const v = toFraction(value);
  if (compareFractions(v, low) < 0) return toFraction(low);
  if (compareFractions(v, high) > 0) return toFraction(high);
  return v;
}
