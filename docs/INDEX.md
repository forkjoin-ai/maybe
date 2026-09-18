# Tensor Bayes / Buleyean — consolidated index

Parent: [Maybe](../README.md) · Theory: [Buleyean probability](./BULEYEAN_PROBABILITY.md) ·
Applications: [APPLICATIONS.md](./APPLICATIONS.md) · Finance: [FINANCE.md](./FINANCE.md)

Everything below was built in one push and independently verified. Status key:
**verified** = I re-ran the build/test myself; **subagent-verified** = the lane ran a
green target I spot-checked. The one-line thesis: the corpus God Formula
`w = R - min(v,R) + 1` is Laplace's add-one rule, and it scales from two axes to a
55-axis hyper-joint, a CRDT, a risk ledger, and a dynamic precision ladder.

## 1. Theory — `open-source/gnosis-math/Gnosis/`

| Module | Proves |
|---|---|
| `TensorBayes` | finite hyper-joint, consensus law, floor dual, both Plücker relations |
| `TensorBayesSkyrms` | 55 = C(11,2) = T(10) = fib(10); 1485 = 495 + 990 |
| `TensorBayesWitness` | anti-vacuity receipts; 3 refuted of 55 => 52-dim residual |
| `TensorBayesVoidBridge` | BuleyeanSpace.weight / StructuralHole = the God Formula |
| `TensorBayesBridge` | channel counts, star partition, Gabriels Horn 55 |
| `TensorBayesDynamics` | data-processing inequality, return sandwich |
| `TensorBayesBlock` | 1,485 channel pairs split 495 adjacent + 990 disjoint |
| `TensorBayesBuleyBridge` | explicit Buley equilibrium at bulk state 55 |
| `TensorBayesEdgeEnum` | constructive Fin 55 edge map; `edgeAt_surjective` **axiom-free** |
| `TensorBayesCoreBridge` | channel composition + FiniteProbabilityCore bridge |
| `TensorBayesUrn` | **Laplace identity** `w_i = n_i + 1`, `Σw = N+K`; collapse point+range |
| `TensorBayesChannelDPI` | channel-level DPI, telescoping loss, overflow dual |
| `TensorBayesStarBlock` | star-block factorization; `sumFin_add'`/`swap'` propext-free |
| `TensorBayesBlockMatrix` | literal 11×11 consensus block matrix (outer product, rank one) |
| `TensorBayesFinance` | never-zero tail, expectation sandwich, no-arbitrage = consensus |
| `CoprimeExpTree` | exp-tree law `map(Σl)=∏map(l)`; integer power tree; single-generator `FiniteCoprimeTree` |
| `BuleyeanProbabilityLaws` | the three Buleyean laws over rejection counts + the floorless adversarial dual |

## 2. Executable — `open-source/maybe/src/`

| Module | What it gives |
|---|---|
| `urn.ts` | urn -> Laplace posterior; exact identity check |
| `tensor-bayes-55.ts` | 55-axis volumes, consensus, `collapse` (point + proven range) |
| `void-crdt.ts` | replicated exact posterior; count layer + idempotent probe ledger |
| `speculative-acceptance.ts` | draft acceptance posterior + block-length policy |
| `quality-ladder.ts` | EXACT/TABLE/AFFINE tiers, value slider, τ miss-not-lie, Hope Jar capacity bridge |
| `probability-die.ts` | spherical lune die, Bhattacharyya gem, exact weight QR |
| `gauss-map-die.ts` | equatorial convex-die solvability (odd n always; even n iff p_even = p_odd) |
| `finance/risk.ts` | exact expectation, never-zero tail, tail bound + MLE dual |
| `finance/kelly.ts` | add-one Kelly, conservative range sizing + overbet dual |
| `finance/newsvendor.ts` | critical-fractile order quantity + mean dual |
| `finance/execution-router.ts` | rejection-ranked venues, floor, collapse band |
| `finance/arbitrage.ts` | consensus no-arbitrage checker with witness |

## 3. Experiments — the falsification spine

| Experiment | Verdict |
|---|---|
| `softmax-vs-buleyean-attention` | **killed** — affine rule has an O(1/N) concentration ceiling |
| `softmax-vs-kernel-attention` | **partial** — richness removes the ceiling but needs m >> N |
| `buleyean-hamming-attention` | **one real win** — table-lookup softmax matches quality; affine still fails |
| `value-slider-pareto` | **knee found** — CHUNKED B=1 removes 93–96% of A.V; fully-linear endpoint is dominated |

## 4. Integrations

| Surface | Artifact | Status |
|---|---|---|
| `distributed-inference` | `src/hope_jar_quality.rs` (Rust tier ladder) | **crate target green** (`a0 run distributed-inference-gnosis:test -- --release -j 2 hope_jar_quality`, 5/5) |
| `distributed-inference` | `src/foil_value_table.rs` + `FOIL_VALUE_TABLE_PRECACHE.md` + derived exp tree | **crate target green** (32/32, `-- --release foil_value_table`) |
| `gnosis-math` | `Gnosis/CoprimeExpTree.lean` in `verify-tensor-bayes` | **verified** (`lake build`, chapel gate clean) |
| `mesh-local-mcp` | `buleyean-router.ts`, `fleet-crdt.ts`, `gnosis_mesh_fleet_posterior` tool | verified 10/10 |
| `forkjoin-hft` | `buleyean-floor-demotion.ts`, `multi-venue-consistency.ts` + Phase-D doc | verified 16/16 |
| `universal-prediction-rt` | `prediction-interval.ts` sound serve gate | verified 5/5 |
| `gnosis/lean` | `SovereignSieveDistribution.lean` soft/noisy sieve | verified (22/22 lean-minimal) |
| `polyglot` | `docs/GNODE_STAGE_PASSTHROUGH.md` annex | verified citations |

## 5. Docs

`BULEYEAN_PROBABILITY.md` (is / is-not) · `APPLICATIONS.md` (+ falsification log) ·
`NEGATIVE_RESULTS_SUBSTRATE.md` (the existing void/eddy substrate + name map) ·
`PROBABILITY_GEOMETRY.md` (physical dice/gems/QR + 55D projections) ·
`FINANCE.md` · `open-source/gnosis-math/docs/TENSOR_BAYES_55.md` (theory) ·
`TENSOR_BAYES_FOLLOWUPS.md` · `HOPE_JAR_QUALITY_LADDER.md` ·
`BULEYEAN_MESH_APPLICATIONS.md` · `SPECULATIVE_ACCEPTANCE_INTEGRATION.md` ·
`FOIL_VACUUM_PARITY_PROTOCOL.md` ·
`open-source/gnosis/distributed-inference/docs/FOIL_VALUE_TABLE_PRECACHE.md` ·
wiki topics (`buleyean-probability`, `laplace-rule-of-succession`, `urn-model`,
`value-table-precache`, ...).

## 6. Verification matrix (all green as of this index)

    a0 run open-source-gnosis-math:verify-tensor-bayes      # 17 targets, 58 jobs (incl. CoprimeExpTree)
    a0 run open-source-gnosis-math:verify-buleyean-probability  # 4 jobs (negative-results laws)
    a0 run open-source-gnosis-math:verify-tensor-bayes-lab  # 50,009 assertions, audited nodeLeaf
    a0 run @a0n/maybe:test                                  # 38/38
    a0 run @affectively/forkjoin-hft:test                   # 16/16
    a0 run @a0n/universal-prediction-rt:test                # 5/5
    a0 run @a0n/gnosis-mesh-local-mcp:test                  # 10/10
    a0 run distributed-inference-gnosis:test -- --release -j 2 hope_jar_quality  # 5/5
    (cd open-source/gnosis/lean-minimal && lake build)      # 22/22
    node scripts/audit-rustic-church.mjs --gate             # chapel gate clean

## 7. The one-sentence results

1. **Buleyean = Laplace's rule of succession** (proved + executable, bit-exact).
2. **The table-lookup softmax works** — an integer-domain `exp` is a sealed O(1)
   array read (up to 27× score win at binary d=2048; parity from dEff ~ 32 bits).
3. **The prime cache is a Hope Jar you derive from** — Pratt certificate trees with
   a free `2^24` leaf bound; and the same multiplicative shape gives the exp tree
   (`b^(a+c) = b^a * b^c`).
4. **The collapse gives a safe human number** — a point plus a proven range, and τ
   converts the cheap tier's lies into misses.

## 8. Open items / flags

* `packages/forkjoin-hft`'s own `tsc -p tsconfig.json` target is **pre-existing red**
  (1005 errors, none in the new files).
* Wiki `econ` / `law` / `scd` course checkers are **pre-existing red** (untouched).
* Root `lake build Gnosis` not run (6,379 files; host strained).
* Nothing committed anywhere — the tree is shared and uncommitted by design.
