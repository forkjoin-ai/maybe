# Squeezing the orange: ambitious applications

Parent: [Maybe](../README.md) · Companion: [What Buleyean probability is, and is not](./BULEYEAN_PROBABILITY.md)

This is a general vision note, deliberately **not** a monorepo roadmap. It asks:
given the proven primitives below, what is the most ambitious thing each one
makes possible elsewhere? Each frontier states the provable engine, the
ambitious form, and the failure modes that would sink it.

## The primitives (the engine)

1. **The identity.** Buleyean weight `w_i = N - min(v_i, N) + 1` equals the
   Laplace numerator `n_i + 1`; normalizer `N + K`. Rejection-counting and
   add-one Bayesian smoothing are the same object.
2. **The floor is a prior.** `w_i >= 1`: every outcome keeps a phantom
   observation, so finite evidence never zeroes a live hypothesis.
3. **Incremental, commutative update.** A draw is `count[i] += 1`, `N += 1`; a
   merge is componentwise addition. Bayesian updating is a CRDT.
4. **The consensus law.** `cond_i_j * M_j = cond_j_i * M_i` across every ordered
   pair of perspectives; symmetry is load-bearing (proved by a red-team dual).
5. **Residual void.** `face * void = total`; unarticulated dimensions keep
   positive weight; articulation deficit is measurable.
6. **Sound collapse.** A target set collapses to a point plus a *guaranteed*
   floor-ceiling interval `[width/W, (R+1)*width/W]` — a bound, not a CI.
7. **Exactness across languages.** The same finite arithmetic is proved in Lean
   and checked exactly in TypeScript. Inference can be a re-checkable witness.

---

## 1. Proof-carrying inference (the certifiable posterior)

**Engine.** Exact finite rationals + machine-checked identities + a runtime
mirror that re-checks the same integers.

**Ambitious form.** Inference that emits not just a number but a finite
certificate any third party re-verifies in milliseconds — an *audit layer for
machine decisions*. Regulated systems (credit, medicine, law, autonomous
vehicles) could ship, alongside each decision, a witness that a checker
re-proves: the counts, the prior, the normalization, the interval, and the
theorem names. Not "trust the model" but "verify the arithmetic and the
stated assumptions".

**Failure modes.** A certificate certifies arithmetic and a stated model, not
the model's validity. Finite-only: continuous latent models need a
discretization argument.

## 2. The negative-results economy (a falsification ledger at scale)

**Engine.** The input is rejection counts; every null result, failed
replication, or refutation is a phantom-ball update against a hypothesis.

**Ambitious form.** A public, incremental, field-level posterior over
hypotheses in which *negative results are the currency*. Most of what science
produces — nulls, failures, retractions — is currently discarded or
file-drawered. The identity says those are exactly the sufficient statistics
for a Laplace posterior, mergeable across labs. Picture a global "void
ledger" where contributing a failed experiment strengthens the posterior for
everyone, and a field's state of evidence is a live, auditable distribution.

**Failure modes.** Rejections must be the complement of occurrences (assumption
7); publication bias means the rejections themselves are selected; the
add-one prior is a choice, and a wrong prior propagates.

**Where this already starts (verified).** The repo already has the engine and a
public ledger: `packages/void-os-core/src/buleyean.ts` is the God Formula over
rejection counts, and `apps/dashrelay-app` commits a verified
`knowledge.residual.claim` to the eddy knotchain, publicly readable via the scrip
API. The name `void-os` is overloaded — `@a0n/void-os` is the operating shell,
`@a0n/void-os-core` is the rejection engine, and `apps/voids-fyi` is a
lander. The full map, the exact mapping to the Laplace identity, and the gaps are
in [`NEGATIVE_RESULTS_SUBSTRATE.md`](./NEGATIVE_RESULTS_SUBSTRATE.md).

## 3. Conflict-free replicated posteriors (Bayesian CRDTs)

**Engine.** A draw is `+1` to one count; a merge is componentwise addition —
commutative, associative, idempotent.

**Ambitious form.** *Posterior state that converges without coordination.*
Distributed sensors, edge fleets, multi-region systems, and offline-first apps
accumulate evidence independently and merge beliefs by addition; the result is
order-independent and duplicate-safe. Bayesian updating becomes as robust as a
grow-only counter — with the phantom ball guaranteeing no replica ever
collapses a class to zero from partial evidence. (The corpus already has a
`mergeVoidBoundaries` CRDT merge as a seed.)

**Failure modes.** Raw counts grow unboundedly; production needs compaction or
decay that must be proved to preserve the identity. Adversarial contributors
can stuff counts (needs attestation/quorum layered on top).

## 4. Decision-grade ambiguity intervals (sound ranges)

**Engine.** `[width/W, (R+1)*width/W]` is a *guaranteed* interval for any
target set.

**Ambitious form.** A decision primitive in which every uncertain input arrives
with a proven range, and high-stakes choices are made on certified intervals
rather than point estimates — robust decision theory (maximin, Hurwicz,
distributionally robust optimization) with an exact finite substrate. For
decisions where being wrong is catastrophic, "the answer is provably in
[a,b]" is worth more than a confident decimal.

**Failure modes.** Bounds can be loose; tightening needs distributional
knowledge. A range is not a probability: do not treat it as calibrated.

## 5. Arbitrage-free collective belief (markets with a phantom unit)

**Engine.** The consensus law is a no-arbitrage condition across every pairwise
channel; the floor guarantees a strictly positive price for every outcome.

**Ambitious form.** Prediction and decision markets whose prices are jointly
consistent *by construction* across all channels, where the failure of
consistency is a detectable fault (the directed dual). The floor makes a
temperature-free automated market maker that can never price a live outcome at
zero — and the 1,485-channel audit is a concrete cross-check graph for
collusion or stale quotes.

**Failure modes.** The symmetry is a modeling postulate; irrational or
manipulated agents can violate it; the saturation blindspot flattens prices
above budget, so explicit refutation tracking is required.

## 6. Never-collapse exploration (temperature-free AI regulators)

**Engine.** `w_i >= 1` means no option is ever eliminated; weights come from
complement counts, so the update is linear and has no temperature.

**Ambitious form.** An exploration policy for bandits, RL, and online
decisions that *provably never eliminates a live option*, with no temperature
to tune and a finite-sample guarantee. Extended to *safety*: a regulator that
guarantees a system cannot permanently abandon a valid channel — directly
attacking premature convergence and mode collapse. The saturation blindspot
names the one behavior that must be added: explicit hard refutation.

**Failure modes.** Add-one can over-explore when the prior is wrong; the floor
delays legitimate elimination, so the refutation channel must be real.

## 7. Hypergraph-native multi-perspective intelligence

**Engine.** The 55 axes are `C(11,2)` pairwise channels; the primitive is the
interaction, not the node.

**Ambitious form.** Architectures where the unit of representation is the
*pair* — multi-agent debate in which every pair of agents is a channel that
must satisfy the consensus law; sensor fusion in which every pair must agree;
ensembles whose members are audited pairwise. The 495 adjacent + 990 disjoint
split gives a concrete, differentiable block structure instead of a flat
vector of opinions.

**Failure modes.** Combinatorial blow-up; pairwise consistency does **not**
imply global consistency, so higher-order (triple) conditions are genuinely
open.

---

## Cross-cutting: why this is defensible

* **It is old mathematics, honestly labeled.** Bayes → Laplace → Jeffreys →
  Dirichlet-multinomial. The novelty is the coordinate system (void), the
  discipline (never-collapse as a capability), and the delivery (finite,
  exact, proof-carrying, cross-runtime).
* **It is checked, not asserted.** The central identity has a Lean proof and an
  exact runtime checker; the symmetry has a red-team dual; the floor has an
  adversarial blindspot that names its cost.
* **It refuses the continuum on purpose.** That is a limitation and a feature:
  the same code is valid in a proof assistant, a browser, a smart contract, and
  firmware, with identical results.

## Honest boundaries (what would kill an application)

1. **The prior is a choice.** The `+1` is `Dirichlet(1)`. Domains with a
   genuinely different prior should parameterize `alpha` and disclose it.
2. **The complement assumption.** The identity needs `v_i = N - n_i`. If
   "rejections" come from elsewhere, the equivalence is void.
3. **Bounds are not probabilities.** The collapse interval is sound but loose;
   never report it as a confidence interval.
4. **Saturation.** Above budget the floor is flat; discrimination requires
   explicit refutation state.
5. **Finite only.** Continuous dynamics need discretization with a proved error
   bound, which is real work and may dominate.
6. **Pairwise is not global.** The consensus law is pairwise; global coherence
   may need higher-order conditions.

## Falsification log (what we tried and killed)

A vision note earns its keep by being corrected, so the kills are recorded here.

**Softmax substitution — killed for the affine/hinge rule.** The hope was that the
Buleyean/Laplace rule could replace softmax cheaply: `w_ij = R - s_ij + 1` is
affine in the features, so it *is* linear attention (two sufficient statistics,
`O(N)`, zero `exp`), and it is bit-exactly integer-checkable. The experiment
`open-source/maybe/experiments/softmax-vs-buleyean-attention/` falsifies the
substitution: a normalized affine ramp has a hard `O(1/N)` concentration ceiling
`(C + s_max)/(N*C + sum_j s_j)`, independent of temperature, so it can never
retrieve. Parity at small logit scale is real but vacuous (both near-uniform).
Softmax wins monotonically as sharp selection is forced. **Linear cost is not the
bottleneck; expressiveness is.** The exactness and exp-free `O(N)` wins stand
only where a soft kernel suffices.
**Surviving hypothesis — tested, and also killed for sharp retrieval.** A
follow-up (`experiments/softmax-vs-kernel-attention/`) removes the affine ceiling
with richer kernels, but not at `O(N)`: exact polynomial features `(1+s)^p` first
match softmax at `p=14`, `m = C(22,14) = 319770` (`m/N ≈ 4996`, ≈ 9600x cost,
20.5 MB of statistics vs softmax's 16 KB), and Performer positive-random-features
do not match up to `m = 4096`. The required `m/N` *grows* with `N` (2735 at N=16,
33532 at N=128). Exactness is preserved (`exp()` calls = 0, BigInt bit-exact), but
**linear attention does not recover softmax quality for sharp retrieval**; it
remains an option only where a soft kernel is acceptable.
**Hamming scores — the table lookup works; the linear normalization still doesn't.** A
third experiment (`experiments/buleyean-hamming-attention/`) replaced the score
matmul with a popcount over quantized vectors, separating the two normalizations.
*Correction:* the `{0,1}` identity `popcount(q XOR k) = d - q.k` is **false** (1973/2000
counterexamples); for **sign** vectors `z in {-1,+1}^d`, `v = (d - z_q.z_k)/2`, so the
Hamming score is affine in the signed dot product (0/2000). With that correction:
(a) the Buleyean/affine weight **fails** — max target mass `(delta+1)/(delta+N)`, chance
at d=8 — the same ceiling as the affine lane; (b) **softmax over the exact integer
Hamming scores with a precomputed `exp` table works at matched quality** (parity from
`dEff ~ 32` packed bits; binary d=256 -> 14x, d=1024 -> 24x, d=2048 -> 27x vs W=32
floats). The `exp` table is one memory read per pair and `sMax-sMin+1` entries per
batch, and reproduces `Math.exp` bit-for-bit (0/22314 mismatches). **Boundary:**
popcount removes `QK^T` only; `A.V` is still dense, so this is a score-matmul +
transcendental win, not matmul-free attention. At equal bit budget the score win is
`W/32`.

**Value-side knee — found, and the fully-linear end is dominated.** A fourth
experiment (`experiments/value-slider-pareto/`) holds the score side fixed at one
exact float softmax and sweeps the `A.V` replacement continuum (block size B,
rank r, topK). The dream of *replacing* `A.V` cheaply at equal quality is
**killed for the linear endpoint**: rank = d loses hard retrieval (recall
0.000–0.047), qerr 0.926–0.995, and costs **1.89×–4.54× the exact `A.V`**,
because the sufficient statistic is built at `N·d·d_v` and read at `d(d+d_v)`
per query. The knee is inside CHUNKED, not linear attention: **CHUNKED B=1**
removes **92.97%–95.70%** of the full `A.V` op count (**96.09%–98.83%** of the
bare multiply-adds, 25.6×–85.3×) while hard recall stays **0.984–1.000**. Most of
`A.V` is removable **for hard retrieval**; a soft readout is not free — the chunked
mean-tail is a poor tail model (needle qerr 0.368–0.485), and topK k=1 keeps
recall but blows up the distribution (KL 1.16–17.09). τ miss-not-lie stays
zero-lie at every candidate and buys a **16.19×–30.12×** end-to-end speedup by
serving only queries whose target column is provably retained (a mass guarantee,
not a value-vector one). BigInt vs naive Number: 766,976 fields, 0 mismatches.

**Bayesian CRDT — corrected.** Componentwise addition is commutative and
associative but **not idempotent** (`a + a = 2a`). `src/void-crdt.ts` therefore
ships two honest layers: a count layer (addition, explicitly not idempotent) and
a replica layer (a ledger of distinct observation ids) that *is* fully
idempotent, so duplicate deliveries never change the posterior.

## If I had to pick an order

1. **Proof-carrying inference** and **Bayesian CRDTs** — both are direct
   corollaries of properties already proved; fastest to a defensible artifact.
2. **Decision-grade intervals** — the clearest human value; sound ranges for
   high-stakes choices.
3. **The negative-results economy** — highest upside, hardest coordination
   problem; starts as a protocol, not a product.
4. **Markets** and **never-collapse exploration** — strong, but must confront
   adversarial and saturation limits head-on.
5. **Hypergraph-native intelligence** — the deepest research bet; needs the
   open triple-consistency theory first.

**North star.** One finite, exact, proof-carrying epistemic substrate whose
only primitive is "count the complement, add a phantom ball" — running
identically in a proof assistant, a browser, and a contract, from which the
negative-results economy, replicated belief, ambiguity ranges, and collective
markets all fall out as corollaries.
