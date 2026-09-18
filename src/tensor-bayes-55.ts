/**
 * tensor-bayes-55.ts -- the finite hyper-joint layer over the God Formula.
 *
 * The Buleyean distribution is the two-perspective (H x E) face of one finite
 * object. This module lifts it to the 55-axis hyper-joint: a single joint mass
 * read from every ordered pair of axes, with the consensus law
 *
 *   P_i(J | Face_j) * M(Face_j) = P_j(J | Face_i) * M(Face_i).
 *
 * It mirrors the Lean surface Gnosis.TensorBayes / Gnosis.TensorBayesWitness
 * / Gnosis.TensorBayesSkyrms. Nothing here proves anything; it is the
 * executable check that the same arithmetic holds at runtime.
 *
 * The 55 axes are the edges of the complete graph on eleven walkers, so the
 * Pleroma is simultaneously C(11,2) and the tenth triangular number. The
 * 1,485 pairwise channels split into 495 adjacent and 990 disjoint edge pairs.
 */

/** The Pleroma: fib(10) = triangular(10) = C(11,2) = 55. */
export const PLEROMA = 55;

/** The kenoma: C(5,2) = 10, the ten-boson channel count. */
export const KENOMA = 10;

/**
 * The God Formula with the Lean clamp: w(R, v) = R - min(v, R) + 1.
 * This is the single source of the weight arithmetic; the normalized
 * distribution still delegates to @a0n/aeon-logic.
 */
export function godWeight(rounds: number, rejections: number): number {
  return rounds - Math.min(rejections, rounds) + 1;
}

/** A hypothesis under the floor semantics: budget, rejections, hard refutation. */
export interface FloorHypothesis {
  readonly budget: number;
  readonly rejections: number;
  readonly hardRefuted: boolean;
}

/**
 * The floor keeps a hypothesis alive at weight >= 1 unless it is explicitly
 * hard-refuted, in which case the weight is exactly 0 (the patch).
 */
export function floorWeight(hypothesis: FloorHypothesis): number {
  return hypothesis.hardRefuted ? 0 : godWeight(hypothesis.budget, hypothesis.rejections);
}

/** C(n, 2): the number of unordered pairs. */
export function pairwise(n: number): number {
  return (n * (n - 1)) / 2;
}

/** T(n): the nth triangular number. */
export function triangular(n: number): number {
  return (n * (n + 1)) / 2;
}

/** C(n, 4). */
export function choose4(n: number): number {
  return (n * (n - 1) * (n - 2) * (n - 3)) / 24;
}

/** The nth Fibonacci number, fib(0) = 0. */
export function fib(n: number): number {
  let a = 0;
  let b = 1;
  for (let i = 0; i < n; i++) {
    const next = a + b;
    a = b;
    b = next;
  }
  return a;
}

/** A signal: one boolean per axis, true when the axis is articulated (visible). */
export type Signal = readonly boolean[];

/** Number of axes a signal articulates. */
export function countTrue(signal: Signal): number {
  let count = 0;
  for (const visible of signal) if (visible) count++;
  return count;
}

/** The articulation deficit n - countTrue(signal). */
export function articulationDeficit(signal: Signal): number {
  return signal.length - countTrue(signal);
}

/** Product over the visible axes (the articulated face volume). */
export function faceVolume(weights: readonly number[], signal: Signal): bigint {
  let acc = 1n;
  for (let i = 0; i < weights.length; i++) {
    acc *= BigInt(signal[i] ? weights[i]! : 1);
  }
  return acc;
}

/** Product over the unarticulated axes (the residual void volume). */
export function voidVolume(weights: readonly number[], signal: Signal): bigint {
  let acc = 1n;
  for (let i = 0; i < weights.length; i++) {
    acc *= BigInt(signal[i] ? 1 : weights[i]!);
  }
  return acc;
}

/** Product over every axis (the total informational volume). */
export function totalVolume(weights: readonly number[]): bigint {
  let acc = 1n;
  for (const weight of weights) acc *= BigInt(weight);
  return acc;
}

/** The structural reading of one signal against a weight vector. */
export interface StructuralReport {
  readonly axes: number;
  readonly visible: number;
  readonly deficit: number;
  readonly faceVolume: bigint;
  readonly voidVolume: bigint;
  readonly totalVolume: bigint;
  /** faceVolume * voidVolume === totalVolume (the consensus invariant). */
  readonly conserved: boolean;
  /** Both halves are at least 1 (the never-collapse floor). */
  readonly floorPreserved: boolean;
}

export function structuralReport(weights: readonly number[], signal: Signal): StructuralReport {
  const face = faceVolume(weights, signal);
  const residual = voidVolume(weights, signal);
  const total = totalVolume(weights);
  return {
    axes: weights.length,
    visible: countTrue(signal),
    deficit: articulationDeficit(signal),
    faceVolume: face,
    voidVolume: residual,
    totalVolume: total,
    conserved: face * residual === total,
    floorPreserved: face >= 1n && residual >= 1n,
  };
}

/** The rank-one consensus entry w_i * w_j of the independent hyper-joint. */
export function consensusEntry(weights: readonly number[], i: number, j: number): number {
  return weights[i]! * weights[j]!;
}

/** The full consensus matrix of the independent hyper-joint. */
export function consensusMatrix(weights: readonly number[]): number[][] {
  return weights.map((wi) => weights.map((wj) => wi * wj));
}

/** The consensus matrix is symmetric on every independent hyper-joint. */
export function consensusIsSymmetric(weights: readonly number[]): boolean {
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights.length; j++) {
      if (weights[i]! * weights[j]! !== weights[j]! * weights[i]!) return false;
    }
  }
  return true;
}

/** A kenoma field: a shared observation budget and per-axis rejections. */
export interface KenomaField {
  readonly budget: number;
  readonly rejections: readonly number[];
}

/** The Buleyean weight of every axis of a kenoma field. */
export function fieldWeights(field: KenomaField): number[] {
  return field.rejections.map((rejections) => godWeight(field.budget, rejections));
}

/** The Skyrms peak: an index of least rejection (ties broken by lowest index). */
export function skyRmsPeak(field: KenomaField): number {
  let peak = 0;
  for (let i = 1; i < field.rejections.length; i++) {
    if (field.rejections[i]! < field.rejections[peak]!) peak = i;
  }
  return peak;
}

/** The Skyrms walker peak carries the greatest Buleyean weight. */
export function peakIsMaxWeight(field: KenomaField): boolean {
  const weights = fieldWeights(field);
  const peak = skyRmsPeak(field);
  return weights.every((weight) => weight <= weights[peak]!);
}

/** The adjacency block split of the channels of K_n. */
export interface ChannelBlocks {
  readonly adjacent: number;
  readonly disjoint: number;
  readonly channels: number;
}

/** Pairs of edges of K_n split by whether they share a walker. */
export function channelBlocks(n: number): ChannelBlocks {
  return {
    adjacent: n * pairwise(n - 1),
    disjoint: 3 * choose4(n),
    channels: pairwise(pairwise(n)),
  };
}

/** The counting identities that make 55 the Pleroma. */
export interface PleromaIdentities {
  readonly pleroma: number;
  readonly tenBosonChannels: number;
  readonly elevenWalkerChannels: number;
  readonly pairwiseChannels: number;
  readonly triangularKenoma: number;
  readonly fibTen: number;
  readonly blockSplit: ChannelBlocks;
}

export function pleromaIdentities(): PleromaIdentities {
  return {
    pleroma: PLEROMA,
    tenBosonChannels: pairwise(5),
    elevenWalkerChannels: pairwise(11),
    pairwiseChannels: pairwise(PLEROMA),
    triangularKenoma: triangular(KENOMA),
    fibTen: fib(10),
    blockSplit: channelBlocks(11),
  };
}

/**
 * Human-facing collapse of the hyper-joint onto a chosen target subset.
 *
 * A weight vector is read at two layers at once. The exact layer keeps the
 * integer mass of the target and of the whole vector. The display layer turns
 * those integers into percentages so a person can see the answer without
 * reading BigInt. The interval is the honest one: the target can be no lighter
 * than its width (every target at the floor weight 1) and no heavier than
 * (budget + 1) * width (every target at the cap). The point is its actual mass.
 */
export interface CollapseInput {
  readonly weights: readonly number[];
  /** Indices of the target axes (repeats count each occurrence). */
  readonly target: readonly number[];
  readonly budget: number;
}

/** The exact mass plus the display percentages and the plain-language line. */
export interface CollapseResult {
  readonly mass: bigint;
  readonly width: number;
  readonly total: bigint;
  readonly pointNumerator: bigint;
  readonly pointDenominator: bigint;
  readonly pointPercent: number;
  readonly lowPercent: number;
  readonly highPercent: number;
  readonly inRange: boolean;
  readonly sentence: string;
}

/** Display-only percentage; no truth claim is made on the float. */
function collapsePercent(numerator: bigint, denominator: bigint): number {
  if (denominator === 0n) return 0;
  return (Number(numerator) * 100) / Number(denominator);
}

/**
 * Collapse a weight vector onto the target indices.
 *
 *   mass   = sum of target weights
 *   width  = target length
 *   total  = sum of all weights
 *   point  = mass / total
 *   range  = [width / total, ((budget + 1) * width) / total]
 *
 * inRange is decided exactly as width <= mass <= (budget + 1) * width, so it
 * never depends on a rounded percentage.
 */
export function collapse(input: CollapseInput): CollapseResult {
  const width = input.target.length;
  let mass = 0n;
  for (const index of input.target) mass += BigInt(input.weights[index]!);

  let total = 0n;
  for (const weight of input.weights) total += BigInt(weight);

  const pointPercent = collapsePercent(mass, total);
  const lowPercent = collapsePercent(BigInt(width), total);
  const highPercent = collapsePercent(BigInt(input.budget + 1) * BigInt(width), total);
  const inRange =
    total > 0n && mass >= BigInt(width) && mass <= BigInt(input.budget + 1) * BigInt(width);

  const sentence =
    `Collapsing to ${width} of ${input.weights.length} axes: ` +
    `${mass} of ${total} weight (${pointPercent.toFixed(1)}%), ` +
    `expected between ${lowPercent.toFixed(1)}% and ${highPercent.toFixed(1)}%. ` +
    (inRange
      ? `The collapse point lies inside that range.`
      : `The collapse point falls outside that range.`);

  return {
    mass,
    width,
    total,
    pointNumerator: mass,
    pointDenominator: total,
    pointPercent,
    lowPercent,
    highPercent,
    inRange,
    sentence,
  };
}
