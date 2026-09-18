# Src

Parent: [Maybe](../README.md)

Source module for the Src feature area.

## Key Files

- `memory-abduction.ts`: Pure JSON adapter over `buildAbduction` for Moonshine's
  memory paths. Positive learned strengths form priors; observed heuristic
  signals rank candidates with hard walls disabled. Missing signals remain 0.5.
  No observations produces no ranking. Reports indistinguishable evidence and
  at most three discriminating local probes; the caller owns scope, relevance
  admission and probe execution. Weights are not empirical confidence. Embedded
  dependency closure: this file, `abduction.ts`, and `@a0n/buleyean-kernel`.
- `memory-abduction.test.ts`: Actual-engine fixture parity, finite normalized
  weights, unknown evidence, disabled walls and bounded-input checks.

- `journey-selection.ts`: Ordered-prefix probabilities over a bounded observed
  trace census, then greedy marginal visible-wait selection under byte, parse,
  build and candidate budgets. Shared resources cost once and overlapping
  prefixes earn only additional savings. Abduction hints retain their separate
  provenance; no observations returns null probabilities and no empirical
  selection. Scenario/test counts must never be supplied as user observations.
- `journey-selection.test.ts`: Conditioning, empty evidence, overlap, budgets and
  invalid mass/resource regression cases.

- `tensor-bayes-55.ts`: The 55-axis hyper-joint over the God Formula. BigInt
  face/void/total volumes with the conservation invariant, rank-one consensus,
  the Skyrms peak, and the Pleroma counting identities. Also exposes the
  human-facing `collapse` surface: a target subset's exact mass, width and
  total as `bigint`, its point ratio, and the display-only interval
  `[width/total, ((budget+1)*width)/total]` with `inRange` decided by exact
  integer sandwich. Mirrors `Gnosis.TensorBayes`, `Gnosis.TensorBayesSkyrms`,
  `Gnosis.TensorBayesBuleyBridge`.
  Check: `a0 run @a0n/maybe:test:tensor-bayes`.
- `tensor-bayes-55.test.ts`: Floor, face/void conservation, rotation invariance,
  consensus symmetry, Skyrms peak, the 495+990=1485 block split, and the
  collapse point always inside the floor/cap interval.
- `urn.ts`: The classic urn read through the God Formula. Rejections are
  `v_i = N - counts[i]`; the clamp `N - min(v_i, N) + 1` collapses onto
  `counts[i] + 1`, so normalizing the Buleyean weights reproduces the Laplace
  posterior `(counts[i]+1)/(N+K)` exactly. `assertUrnLaplaceIdentity` checks the
  integer backbone (`sum counts[i]+1 === N+K` and `w_i === counts[i]+1`) with no
  floating-point truth claim. Check: `a0 run @a0n/maybe:test:urn`.
- `urn.test.ts`: Random count vectors (deterministic PRNG), uniform `1/K` at
  `N=0`, Buleyean = Laplace normalization, and null empirical frequency at
  `N=0`.
- `void-crdt.ts`: The conflict-free replicated posterior. The raw count-vector
  merge (`mergeCounts`/`mergeBigCounts`) is componentwise addition --
  commutative and associative, but NOT idempotent, so a duplicate delivery
  inflates the count. The identity-carrying replica join (`mergeReplicas`,
  exported as `merge`) is commutative, associative, and idempotent because it
  unions a canonical ledger of distinct observation ids; duplicate ids never
  change a count. The exact posterior is integer numerator (`counts[i]+1`) over
  denominator (`N+K`) in BigInt, with `assertLaplaceIdentity` checking the
  urn/Laplace identity exactly. Decay is deliberately omitted: a scalar decay
  does not commute with addition, so it would break the join. Check:
  `a0 run @a0n/maybe:test:void-crdt`.
- `void-crdt.test.ts`: Count-layer commutativity/associativity, explicit
  non-idempotency disclosure, BigInt exactness beyond 2^53, replica-join
  commutativity/associativity/idempotency, duplicate absorption, order- and
  tree-shape-independent convergence to the same exact posterior, uniform `1/K`
  at `N=0`, and the urn/Laplace identity after arbitrary merges.
- `speculative-acceptance.ts`: The speculative-decoding acceptance posterior.
  A drafter proposes draft candidates and a target accepts/rejects them. With
  acceptance counts `a_i` over `K` candidates and `A = sum a_i`, the add-one
  (Laplace/Buleyean) posterior is `P(i) = (a_i + 1) / (A + K)`, returned as
  exact BigInt numerator/denominator pairs plus display floats.
  `rankDraftHeads` orders heads by descending posterior (ties by index);
  `draftLengthPolicy` and `recommendedBlockLength` choose the smallest bounded
  draft/block length whose cumulative posterior reaches `targetMass`;
  `blockAcceptanceProfile` reads the same posterior indexed by block position;
  `expectedAcceptance` reads one candidate; and `assertLaplaceIdentity` checks
  the integer backbone. The `+1` floor keeps every candidate alive, so no head
  is ever permanently pruned. This is a selection posterior, NOT a token
  sampler: the header cites the measured `O(1/N)` concentration ceiling in
  `experiments/softmax-vs-buleyean-attention/` against replacing the softmax
  with the affine rule. Check: `a0 run @a0n/maybe:test:speculative-acceptance`
  and `a0 run @a0n/maybe:typecheck:speculative-acceptance`.
- `quality-ladder.ts`: The add-one kernel as a precision ladder. `expTable`
  seals an integer-domain `exp` into a fixed array (bit-for-bit vs `Math.exp`,
  zero `Math.exp` calls after seal); `hammingSoftmax` reads it in `O(1)`;
  `selectTier` picks `EXACT`/`TABLE`/`AFFINE` from a KB budget, a load, and a
  `targetQuality` floor. On the value side `valueAggregate` slides over block
  size / rank / topK with a Pareto `paretoSweep`, and `missNotLieAdmit` +
  `collapseRange` convert a cheap answer whose proven range leaves the band into a
  miss (never a lie). Removes the transcendental and `QK^T`, not `A.V`; the
  affine tier has an `O(1/N)` concentration ceiling. Check:
  `a0 run @a0n/maybe:test:quality-ladder`.
- `speculative-acceptance.test.ts`: Deterministic-PRNG identity checks, exact
  descending ranking, uniform `1/K` at `A=0`, the never-zero floor, target-mass
  draft/block length monotonicity, and closed-form candidate/block examples.
- `finance/rational.ts`: Exact BigInt rational arithmetic (lowest-terms
  numerator/denominator, decimal-to-fraction, comparison, add/sub/mul/div,
  display float) shared by the finance primitives.
- `finance/risk.ts`: Exact expectation
  `E[loss] = sum_i losses[i]*(counts[i]+1)/(N+K)` under the add-one posterior,
  the exact loss sandwich `minLoss <= E[loss] <= maxLoss`, `neverZeroTail`
  (every state weight is at least 1, so an unobserved catastrophe is priced at
  `1/(N+K)`, never 0), and `tailMassBound` (exact mass strictly above a loss
  threshold). Adversarial dual `tailAdversarialDual`: the MLE `counts[i]/N`
  prices an unobserved state at exactly 0, an infinite underpricing.
- `finance/risk.test.ts`: Closed-form expectation, 2000-case sandwich and
  monotonicity, never-zero floors, exact tails, and the firing MLE dual.
- `finance/kelly.ts`: Exact Kelly `f* = (b p - q)/b`, `buleyeanKelly` on the
  add-one posterior, `fractionalKellyFromRange` and `robustKellyFromRange`
  sizing at the conservative lower endpoint of a proven range (ambiguity-averse),
  and `kellyRangeFromPosterior`. Adversarial dual `kellyAdversarialDual`: on
  1 win in 1 trial the raw MLE stakes 1 where add-one stakes 1/3 at even odds.
- `finance/kelly.test.ts`: Closed-form fractions, exact posterior agreement,
  conservative/range sizing, and the firing small-sample overbet dual.
- `finance/newsvendor.ts`: `criticalFractile = cu/(cu+co)`, the discrete order
  quantity as that quantile of the add-one posterior over demand levels, exact
  expected cost, and `robustNewsvendorOrder` over the proven collapse band
  `[r/W, (R+1)r/W]`. Adversarial dual `newsvendorMeanDual`: ordering the mean
  is suboptimal when `cu != co`.
- `finance/newsvendor.test.ts`: Critical fractile, closed-form order/cost, the
  sound band, and the firing mean-order dual.
- `finance/execution-router.ts`: Rank execution venues/routes by rejection
  counts (failed fills, adverse selection, latency misses) with the God Formula
  `w_i = R - min(v_i, R) + 1`, the exact posterior, the Skyrms peak, the
  never-collapse floor, and the proven collapse range
  `[width/W, (R+1)width/W]`. Adversarial dual `routingAdversarialDual`: the
  MLE prices a correctly-rejected venue at 0 and abandons it forever. Mirrors
  the shape of `open-source/gnosis/mesh-local-mcp/src/buleyean-router.ts`
  without importing across packages.
- `finance/execution-router.test.ts`: Field construction, exact posterior and
  urn/Laplace identity, ranking ties, collapse-range soundness, deterministic
  selection, and the firing MLE-abandonment dual.
- `finance/arbitrage.ts`: The TensorBayes consensus law
  `cond[i][j]*M[j] == cond[j][i]*M[i]` as a finite no-arbitrage checker.
  `findsArbitrage` returns every violating pair plus a witness;
  `directedJointMatrix` is `A[i][j] = cond[i][j]*M[j]`. Adversarial dual
  `arbitrageAdversarialDual`: a direction-blind checker that drops the masses
  misses a real inconsistency.
- `finance/arbitrage.test.ts`: Consistent/inconsistent 2x2 examples, the
  directed-matrix witness, random consistent constructions, exact BigInt
  magnitudes, and the firing mass-blind dual.
- `finance/backtest.ts`: A deterministic backtest for the bet-sizing primitives,
  aimed at FINANCE.md falsification item 3. A seeded xorshift32 generator makes
  a biased binary bet (true `p*`, net odds `b`) with an optional
  regime-switching variant; five rules are simulated over many seeds -- MLE
  Kelly, add-one Kelly `(wins+1)/(trials+2)`, conservative range Kelly at the
  proven lower endpoint from `kelly.ts`, a fixed fraction, and no bet. Reports
  mean/median terminal log-wealth, ruin probability, max drawdown, and the
  fraction of paths that overbet the true-Kelly stake. Adversarial dual
  `backtestAdversarialDual`: on a large sample with a small edge add-one
  underbets and gives up a second-order amount of growth, while on a small
  sample the MLE all-in ruins and add-one saves capital. Saving capital is not
  profitability: with a very small edge the surviving add-one paths can still
  finish below their start. Neither wins everywhere. The xorshift32 seed is
  passed through a splitmix32 finalizer so nearby seeds 1, 2, 3, ... do not
  produce correlated first draws.
- `finance/backtest.test.ts`: Generator reproducibility and empirical regime
  frequencies, hand-checked fixed-fraction and MLE-ruin paths, the two dual
  cases, ruin-probability ordering, conservative-range stake never exceeding
  true Kelly, growth monotonicity in `p*`, exact sizing rationals, and
  same-seed determinism.

All finance probabilities are exact `bigint` numerator/denominator pairs;
floats are display-only. These are arithmetic primitives, NOT financial advice;
no profitability, alpha, or returns are claimed.

- `abduction.ts`: Source file in this module.
- `bang-abduction.ts`: Peptide/bang hard-wall abduction + couple search bracket + RL rejection records. Brute-force enumeration of the admitted rectangle is `@a0n/aeon-crackerjack` `ChemistryBruteForce`.
- `bule.ts`: Source file in this module.
- `buleyean.test.ts`: Source file in this module.
- `buleyean.ts`: Source file in this module.
- `charisma-rank.test.ts`: Source file in this module.
- `charisma-rank.ts`: Source file in this module.
- `charisma-skyrms-game.test.ts`: Source file in this module.
- `charisma-skyrms-game.ts`: Source file in this module.
- `charisma-toy-model.test.ts`: Source file in this module.
- `charisma-toy-model.ts`: Source file in this module.
- `chordonomicon-progression.test.ts`: Source file in this module.
- `chordonomicon-progression.ts`: Source file in this module.
- ...and 23 more source files.

## Subdirectories

- `topologies/`

## Usage

Use this README as the local map for this directory. Keep implementation details near the owning source files, and update this document when the directory contract changes.

## Verification

Run the closest package or app-level check through the repository-owned `a0` or `monster` target after changing behavior in this directory.

Journey selection checks: `a0 run @a0n/maybe:test:journeys` and
`a0 run @a0n/maybe:typecheck:journeys`.

Memory adapter checks: `a0 run @a0n/maybe:test:memory` and
`a0 run @a0n/maybe:typecheck:memory`.

Urn checks: `a0 run @a0n/maybe:test:urn` and
`a0 run @a0n/maybe:typecheck:urn`.

Void-CRDT checks: `a0 run @a0n/maybe:test:void-crdt` and
`a0 run @a0n/maybe:typecheck:void-crdt`.

Speculative-acceptance checks:
`a0 run @a0n/maybe:test:speculative-acceptance` and
`a0 run @a0n/maybe:typecheck:speculative-acceptance`.

Finance decision/risk checks: `a0 run @a0n/maybe:test:finance` (or
`../gnosis/bin/monster test src/finance`) and
`a0 run @a0n/maybe:typecheck:finance`.

Backtest checks: `a0 run @a0n/maybe:test:backtest` (or
`../gnosis/bin/monster test src/finance/backtest.test.ts`) and
`a0 run @a0n/maybe:typecheck:backtest`.

Moonshine's `moonshine:test:mycelial-memory` target verifies packaged adapter
execution through its linked native evaluator.
