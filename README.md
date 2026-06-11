# @a0n/maybe

Buleyean probability as executable gnosis topologies.

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

## Modules

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
