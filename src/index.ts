/**
 * @a0n/maybe -- Buleyean Probability as Executable Gnosis Topologies
 *
 * One formula. Five symbols. Everything.
 *
 *   w_i = R - min(v_i, R) + 1
 *
 * The weight of choice i equals the observation rounds minus the
 * rejection count, plus one. The +1 is the clinamen -- the sliver --
 * the reason no weight ever reaches zero.
 *
 * Seven Universal Laws follow from this formula alone:
 *   1. Impossibility of zero   -- w_i >= 1 always (the sliver)
 *   2. Strict ordering         -- less rejected = more weight
 *   3. Universal sandwich      -- w_i in [1, R+1]
 *   4. Every observation is a cave -- semiotic deficit > 0
 *   5. Conservation            -- remaining + lost = total
 *   6. Sorites sharpness       -- boundaries are discrete
 *   7. Chain termination       -- every abstraction chain converges
 *
 * The reduction chain:
 *   primator (succ != 0) -> clinamen (0 < n+1) -> sliver (0 < w_i)
 *     -> 7 laws -> 35 predictions -> 350+ theorems
 *
 * Three equivalence classes:
 *   α (Sliver):      "X has positive weight"     -- 14/35 predictions
 *   β (Deficit):     "X has positive deficit"     -- 11/35 predictions
 *   γ (Termination): "X terminates / is bounded"  -- 7/35 predictions
 *
 * Two numbers describe any system under irreversible process:
 *   - Buleyean distribution (compass: what to try next)
 *   - Bule number (altimeter: how far from convergence)
 *
 * Mechanized: 3,158+ Lean 4 theorems, 195 modules, zero sorry.
 */

// The God Formula -- core weight computation
export {
  buleyeanDistribution,
  buleyeanWeights,
  sampleBuleyean,
  // Law 1: Impossibility of zero (positivity)
  assertPositivity,
  // Law 2 consequence: normalization
  assertNormalization,
  // Law 2: Strict ordering (monotonicity)
  assertMonotonicity,
  // All three structural axioms at once
  assertAllAxioms,
  klDivergence,
} from './buleyean.js';

// The 55-axis hyper-joint over the God Formula (Gnosis.TensorBayes).
export {
  PLEROMA,
  KENOMA,
  godWeight,
  floorWeight,
  pairwise,
  triangular,
  choose4,
  fib,
  countTrue,
  articulationDeficit,
  faceVolume,
  voidVolume,
  totalVolume,
  structuralReport,
  consensusEntry,
  consensusMatrix,
  consensusIsSymmetric,
  fieldWeights,
  skyRmsPeak,
  peakIsMaxWeight,
  channelBlocks,
  pleromaIdentities,
  collapse,
  type FloorHypothesis,
  type Signal,
  type StructuralReport,
  type KenomaField,
  type ChannelBlocks,
  type PleromaIdentities,
  type CollapseInput,
  type CollapseResult,
} from './tensor-bayes-55.js';

// The classic urn read through the God Formula. The clamped weight
// N - min(v_i, N) + 1 collapses onto counts[i] + 1, so normalizing the
// Buleyean weights reproduces the Laplace posterior exactly.
export {
  urnTotal,
  urnRejections,
  buleyeanWeightFromUrn,
  laplacePosterior,
  buleyeanPosteriorFromUrn,
  empiricalFrequency,
  assertUrnLaplaceIdentity,
  type Urn,
  type UrnLaplaceIdentity,
} from './urn.js';

// The conflict-free replicated posterior. The raw count-vector merge
// (mergeCounts) is componentwise addition -- commutative and associative, but
// not idempotent. The identity-carrying replica join (mergeReplicas, exported
// as `merge`) is commutative, associative, AND idempotent: duplicate delta ids
// are absorbed. The exact posterior is integer numerator/denominator pairs with
// the urn/Laplace identity checked in BigInt.
export {
  MAX_SAFE_COUNT,
  coerceCounts,
  mergeBigCounts,
  mergeCounts,
  evidenceTotal,
  exactPosterior,
  posteriorPairs,
  posterior,
  assertLaplaceIdentity,
  createReplica,
  delta,
  applyDelta,
  mergeReplicas,
  merge,
  mergeAll,
  replicaFromDeltas,
  replicaPosterior,
  probe,
  hasObserved,
  type CountVector,
  type BigCountVector,
  type Counts,
  type PosteriorTerm,
  type ExactPosterior,
  type LaplaceIdentity,
  type DeltaId,
  type Delta,
  type Replica,
} from './void-crdt.js';

// Speculative-decoding acceptance posterior. A drafter proposes candidates and
// the target accepts/rejects them; with acceptance counts a_i over K candidates
// the add-one posterior is (a_i + 1) / (A + K), A = sum a_i. Exact integer pairs
// with display floats, descending head ranking, a bounded target-mass draft
// length policy, the block-position profile, and the never-zero floor. The
// canonical assertLaplaceIdentity name is already exported by void-crdt.ts, so
// this module's identical integer check is re-exported as
// assertAcceptanceLaplaceIdentity.
export {
  acceptancePosterior,
  rankDraftHeads,
  draftLengthPolicy,
  expectedAcceptance,
  blockAcceptanceProfile,
  recommendedBlockLength,
  assertLaplaceIdentity as assertAcceptanceLaplaceIdentity,
  ACCEPTANCE_FLOOR,
  type AcceptanceCounts,
  type AcceptanceTerm,
  type AcceptancePosterior,
  type RankedDraftHead,
  type LengthPolicyOptions,
  type LengthPolicyResult,
  type ExpectedAcceptance,
  type BlockPositionTerm,
  type BlockAcceptanceProfile,
  type AcceptanceLaplaceIdentity,
} from './speculative-acceptance.js';

export {
  selectProbabilityRoutes,
  type ProbabilityRouteDecision,
  type ProbabilityRoutePolicy,
  type ProbabilityRouteSelection,
  type ProbabilityRouteWitness,
  type RouteMassInput,
} from '@a0n/aeon-logic/probability-routing';

export {
  selectJourneyCandidates,
  type JourneyAbductionHint,
  type JourneyPrefixWitness,
  type JourneyResourceCost,
  type JourneySelectionBudget,
  type JourneySelectionCandidate,
  type JourneySelectionInput,
  type JourneySelectionResult,
  type JourneyTraceMass,
  type JourneyUtilityWitness,
} from './journey-selection.js';

export {
  BASELINE_CHORDONOMICON_TRANSITION_WEIGHTS,
  CHORDONOMICON_BASELINE_ROW_MASS,
  FUNCTION_CLASS,
  ROMAN_CHORDS,
  assertChordRowsStochastic,
  chooseDeterministicNextChord,
  chooseNextChord,
  compileChordTransitionKernel,
  countRomanTransitions,
  deterministicTransitionUnit,
  estimateTransitionMatrixFromChordStreams,
  functionalFlowBonus,
  highEntropyTransition,
  inferMajorTonic,
  isDiatonicQualityCompatible,
  normalizeTransitionCounts,
  parseChordSymbol,
  parseChordonomiconChordStream,
  projectChordStreamToRoman,
  projectChordToRoman,
  transitionEdges,
  type ChordQuality,
  type ChordTransitionCounts,
  type ChordTransitionEdge,
  type ChordTransitionKernelWitness,
  type ChordTransitionMatrix,
  type FunctionClass,
  type KeyInference,
  type ParsedChordSymbol,
  type RomanChord,
} from './chordonomicon-progression.js';

// The Seven Laws -- first-class framework
export {
  SEVEN_LAWS,
  GOD_FORMULA_DESCRIPTION,
  REDUCTION_CHAIN,
  EQUIVALENCE_CLASSES,
  type Law,
  type EquivalenceClass,
} from './seven-laws.js';

// Four-layer system (3x3 matrix: conservation x irreversibility x ground state)
export {
  BULEYEAN_LAYERS,
  createBuleyeanStack,
  getLayer,
  tickBuleyeanStack,
  measureBuleyeanStack,
  constraintCascade,
  contextualizationCascade,
} from './layers.js';

// Solomonoff initialization (Law 1: even complex hypotheses retain positive weight)
export {
  estimateComplexity,
  estimateStructuralComplexity,
  solomonoffInit,
  solomonoffInitFromDescriptions,
  solomonoffPrior,
} from './solomonoff.js';

// Retrocausal bounds (Law 7: terminal constraints propagate backward and terminate)
export {
  type TerminalState,
  encodeTerminalStates,
  propagateBackward,
  retrocausalBound,
  verifyRetrocausalBound,
  initRetrocausalLayer,
} from './retrocausal.js';

// Information geometry (Law 4: every observation is a cave -- curvature detects fraud)
export {
  fisherMetric,
  fisherMetricBuleyean,
  fisherInnerProduct,
  fisherNorm,
  bhattacharyyaCoefficient,
  fisherRaoDistance,
  buleyeanDistance,
  scalarCurvature,
  geodesicCurvature,
  trajectoryCurvature,
  totalCurvature,
  type ManifoldCoordinates,
  manifoldCoordinates,
  type FraudAnalysis,
  detectFraud,
  detectBuleyeanFraud,
  compareFraud,
  geodesicInterpolation,
  geodesicPath,
} from './manifold.js';

// Thermodynamics (Law 5: conservation -- V_fork = W_fold + Q_vent)
export {
  BOLTZMANN_K,
  ROOM_TEMPERATURE,
  LANDAUER_LIMIT,
  shannonEntropy,
  maxEntropy,
  landauerHeatFloor,
  landauerHeatCeiling,
  landauerHeatSandwich,
  forkEnergy,
  foldWork,
  ventHeat,
  foldEfficiency,
  verifyFirstLaw,
  thermodynamicAudit,
  transitionCost,
  trajectoryCost,
} from './thermodynamics.js';

// The Bule: altimeter of convergence (Class β: deficit; Class γ: termination)
export {
  PHI,
  PHI_INV,
  buleNumber,
  normalizedBule,
  buleFromPhi,
  topologicalDeficit,
  effectiveParallelism,
  beta1State,
  inverseBule,
  voidRegretBound,
  teleportBuleState,
  convergenceSchedule,
  roundsUntilConvergence,
  sliverWeight,
  sliverAsHeat,
  mergeVoidBoundaries,
  deficitWeightedFold,
  semioticDeficit,
} from './bule.js';

// VoidWalker: the c0-c3 metacognitive loop (Law 3: sandwich bounds the walk)
export {
  type VoidWalkerState,
  type WalkerConfig,
  type MonitoringState,
  type EvaluationResult,
  type StepResult,
  createVoidWalker,
  c0_execute,
  c1_monitor,
  c2_evaluate,
  c3_adapt,
  stepVoidWalker,
  failureDataAdvantage,
  informationPerRound,
} from './void-walker.js';

// Clockwork self-verification (Laplace's Demon)
// Verify void walker convergence state via the sufficient statistic
export {
  verifyBuleyean as clockworkVerify,
  type ClockworkVerdict,
  type VerdictStatus,
} from '@a0n/aeon-clockwork';

// Finite probability compiler -- runtime mirror of
// Gnosis.FiniteProbabilityCore. It turns explicit Nat supports, channels,
// kernels, process chains, Markov witnesses, and finite covers into executable
// certificates with the same mass-balance and residual-shadow contracts.
export {
  compileFiniteChannel,
  compileFiniteCover,
  compileFiniteDistribution,
  compileFiniteKernel,
  compileFiniteProcess,
  compileBoundedWitness,
  boundedWitnessPipelineToProcessChain,
  boundedWitnessAdapterMetadata,
  buildBoundedWitnessWorkflowExample,
  assertBoundedWitnessAdapterContract,
  verifyBoundedWitnessAdapterContract,
  compileBoundedWitnessPipeline,
  compileDomainBoundedWitnessPipeline,
  attentionBoundedWitness,
  finiteApproximationBoundedWitness,
  meshRoutingBoundedWitness,
  queueBoundedWitness,
  thermodynamicBoundedWitness,
  compileKernelRow,
  compileMarkovWitness,
  compileProcessChain,
  composeProcesses,
  eventMass,
  eventProbability,
  kernelToProcess,
  markovWitnessToCover,
  mattressAccounting,
  probabilityRatio,
  processChainToCover,
  checkerTopologyStatsToEvents,
  compileCheckerResultToProbability,
  compileCheckerTopologyStatsToProbability,
  compileTopologyEventsToProbability,
  compileTopologyEventToProcess,
  compileTopologyBoundedWitness,
  witnessCheckerTopologyCompactness,
  witnessObserverAcceptance,
  witnessPositiveVisibleMass,
  witnessTopologyResidualTheorem,
  witnessTopologyShadowEquivalence,
  witnessBoundedEquivalence,
  witnessBoundedPipelineTheorem,
  assertRuntimeTopologyContract,
  runtimeTopologyTheoremNames,
  verifyRuntimeTopologyContract,
  type AttentionResidualSurface,
  type BoundedWitnessCertificate,
  type BoundedWitnessAdapterMetadata,
  type BoundedWitnessAdapterContract,
  type BoundedWitnessAdapterContractEntry,
  type BoundedWitnessAdapterContractReport,
  type BoundedWitnessEquivalence,
  type BoundedWitnessPipeline,
  type BoundedWitnessPipelineTheorem,
  type BoundedWitnessProcessChain,
  type BoundedWitnessWorkflowExample,
  type CheckerTopologyProbabilityCompilation,
  type CheckerTopologyCompactnessWitness,
  type FiniteChannelWitness,
  type FiniteCoverWitness,
  type FiniteDistributionWitness,
  type FiniteHorizonWitness,
  type FiniteKernelRowWitness,
  type FiniteKernelWitness,
  type FiniteMarkovWitnessRuntime,
  type FiniteProbabilityCompilerOptions,
  type FiniteProcessChainWitness,
  type FiniteProcessWitness,
  type DomainBoundedWitnessTheorem,
  type DomainBoundedWitnessPipelineInput,
  type FiniteApproximationResidualSurface,
  type MattressAccountingWitness,
  type MeshRoutingResidualSurface,
  type ObserverAcceptanceWitness,
  type PositiveVisibleMassWitness,
  type ProbabilityRatioWitness,
  type QueueResidualSurface,
  type ThermodynamicResidualSurface,
  type TopologyProbabilityCompilation,
  type TopologyProbabilityCompilerOptions,
  type TopologyProbabilityEventWitness,
  type TopologyBoundedWitnessCertificate,
  type TopologyResidualTheoremWitness,
  type TopologyShadowEquivalenceWitness,
  type RuntimeTopologyContract,
  type RuntimeTopologyContractEntry,
  type RuntimeTopologyContractReport,
} from '@a0n/aeon-logic';

// Charisma / popularity / attention ranking on the Buleyean sliver.
// Generalized ranking from any two scalar fields with the Buleyean sandwich
// soundness contract. Runtime mirror of CharismaRanking / CharismaSandwich /
// CharismaProspectTheory in open-source/gnosis/lean/Lean/ForkRaceFoldTheorems/.
export {
  charismaRank,
  charismaPopularity,
  buleyeanPredict,
  saturatedPopularity,
  gain,
  loss,
  prospectWeight,
  sandwichWitness,
  celebrityPlateau,
  type CharismaRankInput,
  type CharismaRanked,
  type SandwichWitness,
} from './charisma-rank.js';

// Charisma toy model -- unilateral mechanism (Source × Audience × Resistance
// × Capacity × cascade bottleneck). Runtime mirror of
// open-source/gnosis/lean/Lean/ForkRaceFoldTheorems/CharismaToyModel.lean.
export {
  BASELINE_UPTAKE,
  perReceiverUptake,
  totalReception,
  netPerReceiverUptake,
  saturatedPerReceiverUptake,
  cascadedCharisma,
  cascadeReception,
  type Source,
  type Audience,
  type Resistance,
  type Capacity,
} from './charisma-toy-model.js';

// Charisma Skyrms game -- strategic layer over CharismaToyModel:
// indifference plateau, over-push dominated, aligned-plateau equilibrium.
// Mirror of CharismaSkyrmsGame.lean.
export {
  natDist,
  reception,
  senderRegret,
  receiverRegret,
  isEquilibrium,
  underPushPlateauValue,
  type Game,
} from './charisma-skyrms-game.js';

// Mesh Charisma Attention -- Init-only Nat sibling of the Mathlib
// MeshAttention* stack. Mirror of MeshCharismaAttention.lean.
export {
  absorbed,
  buleyeanMass,
  saturatedMass,
  outranks,
  prospectWeightColumn,
  silencingResistance,
  type Mesh,
} from './mesh-charisma-attention.js';

// Abduction — generic inference to the surviving explanation, for sets of any
// things (medical, incident triage, diagnostics, fraud, code/security scanning).
export {
  Abduction,
  buildAbduction,
  buildHypothesisSpace,
  normalizeKey,
  featureChannel,
  priorChannel,
  buildChannels as buildAbductionChannels,
  flattenChannels as flattenAbductionChannels,
  fuse as abductiveFuse,
  voidMetrics as abductiveVoidMetrics,
  nextProbes,
  factorize as abductiveFactorize,
  mapVoid as abductiveMapVoid,
  setAbductionWasm,
  getAbductionWasm,
  isAbductionWasmReady,
  type Protocol,
  type Hypothesis,
  type HypothesisSpace,
  type Polarity as AbductionPolarity,
  type FeatureChannelVerdict,
  type Observation,
  type Survivor as AbductionSurvivor,
  type Eliminated,
  type NextProbe,
  type Factor as AbductionFactor,
  type ProtocolTrigger,
  type AbductivePosterior,
  type AbduceOptions,
  type FuseResult as AbductionFuseResult,
  type VoidCell as AbductionVoidCell,
  type VoidMap as AbductionVoidMap,
  type VoidMapOptions as AbductionVoidMapOptions,
  type AbductionWasm,
} from './abduction.js';

export {
  buildBangAbduction,
  abduceBang,
  observeProtectedC,
  observeUnknownCodon,
  observeClosedShell,
  observeFreeCouple,
  peptideSearchBracket,
  peptideRejectionRecord,
  type BangHypothesisId,
  type BangFeature,
} from './bang-abduction.js';

// Audibility — the prior question to abduction: can this evidence separate these
// explanations at all? Never-collapse protects the inference; this protects the observation.
export {
  auditAudibility,
  assertAudible,
  separability,
  DEFAULT_AUDIBILITY_EPSILON,
  type AudibilityReport,
  type IndistinguishablePair,
  type InertFeature,
} from './audibility.js';

// ===========================================================================
// Finance decision/risk primitives (src/finance/)
// ===========================================================================
//
// Exact rational arithmetic and five decision/risk modules built on the
// add-one (Laplace/Buleyean) posterior. Every probability is a BigInt
// numerator/denominator pair; floats are display-only. These are arithmetic
// primitives, NOT financial advice, and no profitability or alpha is claimed.
//
// Every protection ships an adversarial dual that names its failure mode:
//   risk.ts              MLE prices an unobserved catastrophe at 0
//   kelly.ts             MLE overbets on a small sample
//   newsvendor.ts        ordering the mean is suboptimal when cu != co
//   execution-router.ts  MLE abandons a correctly-rejected venue forever
//   arbitrage.ts         a mass-blind checker misses a real inconsistency

// Exact BigInt rational arithmetic shared by the finance primitives.
export {
  gcd,
  exactFraction,
  fractionFromNumber,
  toFraction,
  compareFractions,
  addFractions,
  subFractions,
  mulFractions,
  divFractions,
  isZeroFraction,
  isPositiveFraction,
  isNegativeFraction,
  fractionToNumber,
  fractionToString,
  clampFraction,
  type ExactFraction,
  type RationalInput,
} from './finance/rational.js';

// Exact expectation and the never-zero tail.
export {
  coerceLosses,
  exactExpectation,
  assertLossSandwich,
  neverZeroTail,
  mlePosterior,
  tailAdversarialDual,
  tailMassBound,
  type LossVector,
  type LossContribution,
  type ExactExpectation,
  type LossSandwichReport,
  type NeverZeroTailReport,
  type MlePosterior,
  type TailAdversarialDual,
  type TailMassBound,
} from './finance/risk.js';

// Bet sizing on the exact posterior, with the conservative range variant.
export {
  kellyFraction,
  kellyDecision,
  buleyeanKelly,
  mleKelly,
  fractionalKellyFromRange,
  robustKellyFromRange,
  kellyRangeFromPosterior,
  kellyAdversarialDual,
  type OddsInput,
  type KellyDecision,
  type BuleyeanKellyDecision,
  type FractionalKellyFromRange,
  type KellyRangeReport,
  type KellyAdversarialDual,
} from './finance/kelly.js';

// Critical fractile and the discrete order quantity.
export {
  coerceDemands,
  criticalFractile,
  newsvendorOrder,
  newsvendorCost,
  posteriorMeanDemand,
  meanOrderDemand,
  newsvendorMeanDual,
  robustNewsvendorOrder,
  type DemandVector,
  type NewsvendorRung,
  type NewsvendorOrder,
  type NewsvendorCost,
  type NewsvendorMeanDual,
  type RobustOrderEndpoint,
  type RobustNewsvendorOrder,
} from './finance/newsvendor.js';

// Execution-venue ranking by rejection counts. The names godWeight,
// skyRmsPeak, peakIsMaxWeight and assertLaplaceIdentity already exist in the
// package (tensor-bayes-55.ts and void-crdt.ts), so the router's identical
// surfaces are re-exported under venue/routing-qualified aliases.
export {
  DEFAULT_REJECTION_WEIGHTS,
  rejectionTotal,
  executionField,
  godWeight as venueGodWeight,
  laplaceCounts,
  venueWeights,
  venuePosterior,
  skyRmsPeak as venueSkyRmsPeak,
  peakIsMaxWeight as venuePeakIsMaxWeight,
  assertFloor,
  assertLaplaceIdentity as assertRoutingLaplaceIdentity,
  collapseRange,
  rankVenues,
  DEFAULT_SELECT_SEED,
  seededRng,
  selectVenue,
  routingAdversarialDual,
  type RejectionCounts,
  type RejectionWeights,
  type ExecutionVenue,
  type ExecutionField,
  type VenuePosterior,
  type ExecutionPosterior,
  type FloorReport,
  type RoutingLaplaceIdentity,
  type CollapseRangeReport,
  type RankedVenue,
  type RoutingAdversarialDual,
} from './finance/execution-router.js';

// The consensus law as a finite no-arbitrage checker.
export {
  asRationalMatrix,
  asMassVector,
  consensusResidual,
  findsArbitrage,
  twoVenueArbitrage,
  directedJointMatrix,
  antisymmetricResidualMatrix,
  directedJointIsSymmetric,
  directionBlindFindsArbitrage,
  arbitrageAdversarialDual,
  type MatrixEntry,
  type RationalMatrix,
  type MassVector,
  type ArbitrageWitness,
  type ArbitrageReport,
  type DirectionBlindDual,
} from './finance/arbitrage.js';

// The deterministic bet-sizing backtest: a seeded market generator, five
// sizing rules, and the honest adversarial dual for FINANCE.md item 3
// (undershoot growth vs. save capital). No profitability is claimed.
export {
  xorshift32,
  staticMarket,
  switchingMarket,
  trueProbabilityAt,
  generateOutcomes,
  trueKellyFraction,
  recommendedTrueKelly,
  expectedLogGrowth,
  kellyGrowthRate,
  mleKellyStrategy,
  addOneKellyStrategy,
  conservativeRangeKellyStrategy,
  fixedFractionStrategy,
  noBetStrategy,
  oracleKellyStrategy,
  defaultStrategies,
  runPath,
  runOutcomeSequence,
  runBacktest,
  seedRange,
  summarizeStrategy,
  strategyById,
  growthUndershootCase,
  capitalSavedCase,
  backtestAdversarialDual,
  type RegimeStep,
  type MarketSpec,
  type BetContext,
  type BetStrategy,
  type PathResult,
  type StrategyReport,
  type BacktestReport,
  type GrowthUndershootReport,
  type CapitalSavedReport,
  type BacktestAdversarialDual,
} from './finance/backtest.js';

// The dynamic precision ladder: score/exp tiers, the value-side slider, and tau miss-not-lie.
// Re-exported as a namespace because the module exposes ~60 symbols and `collapseRange`
// collides with the execution-router export.
export * as qualityLadder from './quality-ladder.js';
