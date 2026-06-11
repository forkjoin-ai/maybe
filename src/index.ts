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

export {
  selectProbabilityRoutes,
  type ProbabilityRouteDecision,
  type ProbabilityRoutePolicy,
  type ProbabilityRouteSelection,
  type ProbabilityRouteWitness,
  type RouteMassInput,
} from '@a0n/aeon-logic/probability-routing';

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
