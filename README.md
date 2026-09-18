# @a0n/maybe

Buleyean probability as executable gnosis topologies.

## Architecture and purpose

This package turns rejection counts into normalized choice weights, then layers
epistemic timescales and thermodynamic audits around that kernel. The formula is
delegated to `@a0n/aeon-logic`; this package owns the executable probability and
topology composition surface.

Two numbers describe any system under irreversible process:
- **Buleyean distribution** (compass: what to try next)
- **Bule number** (altimeter: how far from convergence)

## Core Formula

```
P(i) = (T - v_i + 1) / sum_j(T - v_j + 1)
```

Weight from what things are NOT. Where the void has accumulated least, probability peaks.

Three provable axioms (mechanized in Lean 4, zero sorry):

1. **Positivity**: `P(i) > 0` for all `i` (the sliver: +1 from Landauer heat)
2. **Normalization**: `sum(P) = 1` (by construction)
3. **Monotonicity**: `v_i < v_j => P(i) > P(j)` (less void = more weight)

> **What it is, and is not:** [docs/BULEYEAN_PROBABILITY.md](docs/BULEYEAN_PROBABILITY.md).
> In one line: this is the void-dual reading of Laplace's rule of succession —
> counting what is NOT, and counting what is plus one phantom ball per outcome,
> are the same arithmetic (`w_i = n_i + 1`, `sum w_i = N + K`).

## Modules

### Memory path abduction (`memory-abduction.ts`)

`rankMemoryPaths` (also exported as `app` for embedded evaluation) ranks a bounded
set of local memory paths through the existing abductive engine. It preserves
unknown signals, disables heuristic hard walls, and proposes up to three local
probes. Learned path strengths supply priors; ranking weights do not establish
relevance or empirical confidence. Moonshine owns relevance admission, source
scope and feedback. See [source contracts and checks](src/README.md).

### Core (`buleyean.ts`)
The formula. Delegates to `@a0n/aeon-logic` as single source of truth.

### Four-Layer System (`layers.ts`)
Maps probability theory to four epistemological timescales via gnosis `BoundaryStack`:

| Layer | Name | Timescale | Role |
|-------|------|-----------|------|
| 0 (deepest) | Retrocausal | lifetime | Terminal state constrains trajectory |
| 1 | Bayesian | weeks | Prior-posterior update cycle |
| 2 | Frequentist | minutes | Direct observation counting |
| 3 (shallowest) | Solomonoff | generational | Complexity-weighted initialization |

### Thermodynamics (`thermodynamics.ts`)
The energetic cost of knowing. First Law: `V_fork = W_fold + Q_vent`.

- `shannonEntropy(p)` -- maximum extractable work (Carnot bound)
- `landauerHeatFloor(p)` / `landauerHeatCeiling(N)` -- the heat sandwich
- `forkEnergy(N)` / `foldWork(p, q)` / `ventHeat(V, W)` -- energy accounting
- `thermodynamicAudit(p, q)` -- complete first-law verification
- `transitionCost(p, q)` / `trajectoryCost(path)` -- manifold edge weights

### Finite Probability Compiler
Runtime certificates for the native finite probability stack formalized in
`Gnosis.FiniteProbabilityCore`.

- `compileFiniteDistribution(weights)` -- explicit positive finite support
- `compileFiniteProcess(input, output, residual)` -- mass balance plus residual
  shadow
- `compileProcessChain(processes)` -- folded mass loss and residual
- `compileFiniteKernel(rows)` -- row-wise natural-weight stochastic kernels
- `compileMarkovWitness(kernels, shadowResidual)` -- finite Markov shadow
- `compileFiniteCover(...)` -- bounded horizon/coverage witness
- `mattressAccounting(process)` -- visible output plus shadow accounts for input
- `compileTopologyEventsToProbability(events)` -- topology traces as finite
  process chains and bounded covers
- `compileCheckerResultToProbability(result)` -- checker topology summaries as
  finite probability certificates
- `witnessTopologyResidualTheorem(certificate)` -- chain residual equals the
  sum of event residuals
- `witnessCheckerTopologyCompactness(certificate)` -- checker shadow equals
  `foldCount + ventCount`
- `witnessObserverAcceptance(certificate, budget)` -- finite observer budget
  covers the residual
- `assertRuntimeTopologyContract(contract)` -- verify the Lean-to-TypeScript
  theorem naming contract
- `compileBoundedWitness(surface, residual, budget, theorem)` -- reusable
  observed-surface, residual-shadow, observer-budget, theorem-witness pattern
- `queueBoundedWitness`, `thermodynamicBoundedWitness`,
  `meshRoutingBoundedWitness`, `attentionBoundedWitness`, and
  `finiteApproximationBoundedWitness` -- standard bounded-domain adapters
- `compileBoundedWitnessPipeline(witnesses, budget)` -- sum multiple witnesses
  into one accepted composite certificate
- `compileDomainBoundedWitnessPipeline(workflow, budget)` -- queue,
  thermodynamics, mesh routing, attention, and finite approximation in the
  standard workflow order
- `boundedWitnessPipelineToProcessChain(pipeline)` -- lower an accepted
  pipeline into process-chain accounting
- `buildBoundedWitnessWorkflowExample()` -- concrete queue-to-finite-
  approximation certificate with residual `33`

This is the executable probability compiler: continuous probability and measure
theory stay out of bounds; bounded callers receive exact finite witnesses.
Topology races record unchosen branches as residual, folds and vents add
bounded shadow, and observations keep the visible unit unchanged.

### The Bule (`bule.ts`)
The altimeter of convergence.

- `buleNumber(boundary)` -- integer distance from ground state
- `beta1State(boundary)` -- topological deficit: natural vs actual parallelism
- `inverseBule(boundary)` -- learning rate (nats per round)
- `teleportBuleState(B, k)` -- statistical teleportation: one integer encodes the future
- `sliverWeight(boundary)` -- minimum probability from Landauer heat
- `mergeVoidBoundaries(A, B)` -- CRDT merge: two slivers become one
- `deficitWeightedFold(dists)` -- Glossolalia merge: disagreement = information
- `semioticDeficit(k, V, K)` -- information lost in fold

### VoidWalker (`void-walker.ts`)
The active c0-c3 metacognitive learning agent.

- `createVoidWalker(config)` -- initialize with fresh void boundary
- `stepVoidWalker(state, reward, optimal, rng)` -- one full c0-c1-c2-c3 cycle
- `c0_execute` -- choose action from Buleyean distribution
- `c1_monitor` -- measure Bule, entropy, kurtosis, parallelism
- `c2_evaluate` -- detect regime changes, compute gradient
- `c3_adapt` -- adjust exploration rate and eta
- `failureDataAdvantage(N)` -- rejection provides (N-1)x more data

### Information Geometry (`manifold.ts`)
Fisher metric on the probability simplex.

- `fisherMetric(p)` / `fisherRaoDistance(p, q)` -- Riemannian structure
- `geodesicCurvature(prev, curr, next)` -- path deviation from geodesic
- `detectFraud(trajectory)` -- statistical fraud via curvature anomalies
- `geodesicInterpolation(p, q, t)` -- slerp on the manifold

### 55D Tensor Bayes (`tensor-bayes-55.ts`)
The finite hyper-joint over the God Formula: 55 axes, one consensus law per
ordered pair, a residual void that never collapses. Mirrors `Gnosis.TensorBayes`,
`Gnosis.TensorBayesSkyrms`, and `Gnosis.TensorBayesBuleyBridge`.

- `godWeight(R, v)` / `floorWeight(h)` -- the clamped God Formula and the floor
- `faceVolume` / `voidVolume` / `totalVolume` / `structuralReport` -- BigInt
  volumes with `face * void === total` and both halves `>= 1`
- `consensusMatrix` / `consensusIsSymmetric` -- the rank-one joint `J_ij = w_i w_j`
- `skyRmsPeak` / `peakIsMaxWeight` -- least rejection carries greatest weight
- `pleromaIdentities` / `channelBlocks` -- 55 = C(11,2) = T(10) = fib(10);
  1485 = 495 adjacent + 990 disjoint
- `collapse({ weights, target, budget })` -- the human-facing read of any target
  subset: exact `mass`/`total`/ratio as `bigint`, the display percentage, and
  the honest interval `[width/total, ((budget+1)*width)/total]` with `inRange`
  decided by the exact integer sandwich. Returns a plain-language `sentence`.

### Classic urn (`urn.ts`)

The familiar urn of Laplace smoothing, read through the same clamp. A count
vector has `N = sum counts` draws and per-bucket rejections
`v_i = N - counts[i]`. The God Formula `N - min(v_i, N) + 1` then collapses
exactly onto `counts[i] + 1` -- the Laplace numerator -- so normalizing the
Buleyean weights reproduces Laplace's rule of succession:

```
laplacePosterior(counts)[i] = (counts[i] + 1) / (N + K)
buleyeanPosteriorFromUrn(counts)[i] = w_i / sum_j w_j
```

- `urnTotal(counts)` -- `N = sum counts`
- `urnRejections(counts)` -- `v_i = N - counts[i]`
- `buleyeanWeightFromUrn(counts, i)` -- the clamp, equal to `counts[i] + 1`
- `laplacePosterior` / `buleyeanPosteriorFromUrn` -- the two normalizations
- `empiricalFrequency(counts)` -- `counts[i]/N`, `null` when `N = 0`
- `assertUrnLaplaceIdentity(counts)` -- exact integer check of
  `sum(counts[i]+1) === N + K` and `w_i === counts[i] + 1`

The exact identity is integer arithmetic; the posterior and frequency helpers
return display floats only. Checks: `a0 run @a0n/maybe:test:urn` and
`a0 run @a0n/maybe:typecheck:urn`.

### Conflict-free replicated posterior (`void-crdt.ts`)

The Laplace/Buleyean posterior as a CRDT: independent replicas exchange
observations in any order, with duplication, and agree on the exact posterior.
Two honest layers:

- `mergeCounts(a, b)` / `mergeBigCounts(a, b)` -- componentwise addition of
  count vectors. Commutative and associative, but **not idempotent**
  (`a + a = 2a`), so a re-delivered observation inflates the count. BigInt
  keeps sums exact; the number path throws instead of silently rounding past
  `2^53 - 1`.
- `mergeReplicas(a, b)` (exported as `merge`) -- the identity-carrying replica
  join. Each replica keeps a canonical ledger of distinct observation ids, so
  the join is set union: commutative, associative, **and idempotent**
  (`merge(a, a) = a`). Duplicate ids never change a count.
- `delta(id, category)` / `applyDelta(replica, delta)` -- one observation and
  its idempotent application.
- `exactPosterior(counts)` / `posteriorPairs(counts)` / `probe(replica, i)` --
  the exact posterior as integer numerator `counts[i] + 1` and shared
  denominator `N + K`, with a display-only float alongside. `N = 0` is exactly
  the uniform `1/K` posterior.
- `posterior(counts)` -- the display floats `(counts[i] + 1) / (N + K)`.
- `assertLaplaceIdentity(counts)` -- exact BigInt check that the numerators are
  `counts[i] + 1` and sum to `N + K`.

Decay is deliberately omitted: scaling counts does not commute with addition,
so it would break the join identity. Checks: `a0 run @a0n/maybe:test:void-crdt`
and `a0 run @a0n/maybe:typecheck:void-crdt`.

### Speculative acceptance (`speculative-acceptance.ts`)

The validation posterior for a draft-and-verify speculative decoder. A drafter
proposes draft candidates and a target accepts or rejects them; with acceptance
counts `a_i` over `K` candidates and `A = sum_i a_i`, the add-one
(Laplace/Buleyean) acceptance posterior is

```
P(i) = (a_i + 1) / (A + K)
```

returned as exact `bigint` numerator/denominator pairs with display floats.

- `acceptancePosterior(accepts)` -- exact pairs plus display floats; uniform
  `1/K` when `A = 0`
- `rankDraftHeads(accepts)` -- candidates ranked by descending posterior, ties
  by index; every candidate is retained
- `draftLengthPolicy(accepts, { minLen, maxLen, targetMass })` -- smallest
  bounded draft length whose cumulative posterior reaches `targetMass`
- `expectedAcceptance(accepts, candidate)` -- one candidate's exact posterior
  pair plus its display float
- `blockAcceptanceProfile(countsByPosition)` / `recommendedBlockLength(...)` --
  the same posterior indexed by block position, with the expected accepted
  position and a bounded target-mass block length
- `assertLaplaceIdentity(accepts)` -- exact integer check that the numerators
  are `a_i + 1` and sum to `A + K`; the `+1` floor means no candidate is ever
  pruned

This is a *selection* posterior over which candidate will be accepted next, not
a token sampler. The Gnosis wiring note
[`open-source/gnosis/distributed-inference/SPECULATIVE_ACCEPTANCE_INTEGRATION.md`](../gnosis/distributed-inference/SPECULATIVE_ACCEPTANCE_INTEGRATION.md)
warns explicitly against replacing the sampler's softmax with the affine rule,
citing the measured `O(1/N)` concentration ceiling in
[`experiments/softmax-vs-buleyean-attention/`](experiments/softmax-vs-buleyean-attention/README.md).
Checks: `a0 run @a0n/maybe:test:speculative-acceptance` and
`a0 run @a0n/maybe:typecheck:speculative-acceptance`.

### Dynamic quality ladder (`quality-ladder.ts`)

The same add-one kernel expressed as a *precision* ladder rather than a
probability: `EXACT` (float softmax), `TABLE` (an integer-domain `exp` sealed
once into an array), and `AFFINE` (the Buleyean ramp). `selectTier` picks a tier
from a budget in KB and a load, and a `targetQuality` floor wins last. The table
path reproduces `Math.exp` bit-for-bit for every integer score and makes **zero**
`Math.exp` calls after sealing -- an `O(1)` array read, not an `O(1)` softmax.

`valueAggregate` exposes the value side as a slider (block size, rank, topK) with
a Pareto `paretoSweep`; `missNotLieAdmit` + `collapseRange` refuse to serve a
cheap answer whose *proven* interval leaves the band, converting a would-be lie
into a miss. Honest boundaries: the sealed table removes the transcendental and
`QK^T`, **not** `A.V`; hard retrieval tolerates truncation but a soft readout
does not. The falsification spine is
[`experiments/value-slider-pareto/`](experiments/value-slider-pareto/README.md)
(knee: CHUNKED B=1 removes 93-96% of `A.V`; the fully-linear endpoint is
dominated) and the affine ceiling is in
[`experiments/softmax-vs-buleyean-attention/`](experiments/softmax-vs-buleyean-attention/README.md).
Check: `a0 run @a0n/maybe:test:quality-ladder`. See
[`docs/INDEX.md`](docs/INDEX.md) for the consolidated map.

### Finance decision/risk primitives (`src/finance/`)

Exact-rational decision and risk arithmetic built on the same add-one
(Laplace/Buleyean) posterior. Every probability is a BigInt
numerator/denominator pair; floats are display-only. These are arithmetic
primitives, **not financial advice**; no profitability, alpha, or returns are
claimed. Each module ships an adversarial dual that names its failure mode.

- `rational.ts` -- exact BigInt fractions (reduce, compare, add/sub/mul/div,
  decimal-to-fraction, display float).
- `risk.ts` -- `exactExpectation(counts, losses)` under the add-one posterior;
  the exact sandwich `minLoss <= E[loss] <= maxLoss`; `neverZeroTail`, so an
  unobserved catastrophe is priced at `1/(N+K)` and never 0; and
  `tailMassBound` for the exact mass above a loss threshold. Dual: the MLE
  `counts[i]/N` prices an unobserved state at 0 -- an infinite underpricing.
- `kelly.ts` -- exact Kelly `f* = (b p - q)/b`, `buleyeanKelly` on the
  add-one posterior, `fractionalKellyFromRange` / `robustKellyFromRange`
  sizing at the conservative lower endpoint of a proven range, and
  `kellyRangeFromPosterior` reading the collapse band through Kelly. Dual: on
  1 win in 1 trial the raw MLE stakes 1 where add-one stakes 1/3 at even odds.
- `newsvendor.ts` -- `criticalFractile = cu/(cu+co)`, the discrete order
  quantity as that quantile of the add-one posterior over demand levels, exact
  expected cost, and `robustNewsvendorOrder` over the proven collapse band
  `[r/W, (R+1)r/W]`. Dual: ordering the mean is suboptimal when `cu != co`.
- `execution-router.ts` -- rank execution venues/routes by rejection counts
  (failed fills, adverse selection, latency misses) with
  `w_i = R - min(v_i, R) + 1`, the exact posterior, the Skyrms peak, the
  never-collapse floor, and the proven collapse range
  `[width/W, (R+1)width/W]`. Dual: the MLE prices a correctly-rejected venue at
  0 and abandons it forever.
- `arbitrage.ts` -- the consensus law
  `cond[i][j]*M[j] == cond[j][i]*M[i]` as a finite no-arbitrage checker;
  `findsArbitrage` returns the violating pair as a witness and
  `directedJointMatrix` is the directed joint. Dual: a direction-blind checker
  that drops the masses misses a real inconsistency.
- `backtest.ts` -- a deterministic backtest for the sizing rules, against
  FINANCE.md falsification item 3. A seeded xorshift32 generator makes a biased
  binary bet (true `p*`, net odds `b`, plus a regime-switching variant that
  changes `p*` partway); five rules (MLE Kelly, add-one Kelly, conservative
  range Kelly at the proven lower endpoint, a fixed fraction, no bet) are
  simulated over many seeds. Reports mean/median terminal log-wealth, ruin
  probability, max drawdown, and the overbet-path fraction. Dual
  `backtestAdversarialDual`: on a large sample with a small edge add-one
  underbets and gives up a second-order amount of growth; on a small sample the
  MLE all-in ruins and add-one saves capital. It is not a universal winner.

Checks: `a0 run @a0n/maybe:test:finance`, `a0 run @a0n/maybe:test:backtest`,
`a0 run @a0n/maybe:typecheck:finance`, and
`a0 run @a0n/maybe:typecheck:backtest`.

### Retrocausal (`retrocausal.ts`)
Terminal state backward propagation.

### Solomonoff (`solomonoff.ts`)
Complexity-weighted void initialization.

## Topologies

The `.gg` files in `src/topologies/` ARE the specification:

- `complement.gg` -- Core Buleyean distribution
- `void-walk.gg` -- c0-c3 walking loop
- `thermodynamics.gg` -- Fork/fold/vent energy accounting
- `bule-altimeter.gg` -- Bule number and sliver derivation
- `deficit-weighted-fold.gg` -- Glossolalia semiotic merge
- `four-layers.gg` -- Full BoundaryStack system
- `curvature-detector.gg` -- Geodesic fraud detection
- `ethics-grid.gg` -- Eight compiler checks as theorem nodes
- `tensor-bayes-55.gg` -- the 55-axis hyper-joint over the God Formula

## Mechanization

Lean 4 proofs in the parent repo (zero sorry):

- `BuleyeanProbability.lean` -- positivity, normalization, monotonicity
- `BuleIsValue.lean` -- one Bule = one unit of value = kT ln 2 joules
- `SliverFromVent.lean` -- the +1 is Landauer heat: vent -> heat -> sliver -> positivity
- `SliverOfHope.lean` -- sliver (+1) and proof (-1) are inverses
- `VoidWalking.lean` -- c0-c3 loop, void boundary as sufficient statistic
- `RetrocausalBound.lean` -- backward propagation theorem
- `SolomonoffBuleyean.lean` -- complexity prior consistency
- `FoldHeatHierarchy.lean` -- injective = 0 heat, non-injective > 0
- `LandauerBuley.lean` -- thermodynamic calibration, heat sandwich
- `FisherManifold.lean` -- curvature monotonicity, Solomonoff gap
- `TensorBayes.lean` -- hyper-joint consensus, floor dual, both Pluecker relations
- `TensorBayesSkyrms.lean` -- ten-boson and fifty-five-axis counting
- `TensorBayesWitness.lean` / `TensorBayesBlock.lean` -- 52-dim residual, channel split
- `TensorBayesBuleyBridge.lean` -- the Buley fixed point at 55
