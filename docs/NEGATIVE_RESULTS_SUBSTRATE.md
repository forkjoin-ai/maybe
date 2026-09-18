# The negative-results economy — what already exists

Parent: [Buleyean probability](./BULEYEAN_PROBABILITY.md) · Applications: [APPLICATIONS.md](./APPLICATIONS.md) · Index: [INDEX.md](./INDEX.md)

Section 2 of [APPLICATIONS.md](./APPLICATIONS.md) proposes a public falsification
ledger: rejection counts as currency, mergeable across labs, a field's evidence a
live Laplace posterior. That proposal does not start from zero. This note records
the substrate that is already in the tree, the exact mapping to the theory, and
the three different things that carry a spelling of 'void os' so the next agent
does not conflate them or rebuild what exists.

Provenance: **Verified** means I read the file or ran the build named.

## 1. The name conflict: three 'void os' spellings, one is not the others

| Name | Path | What it is |
|---|---|---|
| `@a0n/void-os` | `open-source/aeon-shell` | The local-first operating shell (browser/Tauri/R1, `aeon://`). **Not** the negative-results engine. |
| `@a0n/void-os-core` | `packages/void-os-core` | Void OS core: consciousness, personality streaming, psyche digest — and, in `src/buleyean.ts`, the rejection-count God Formula. This is the 'voidos' lead. |
| Voids (Void OS) | `apps/voids-fyi` | A one-Worker marketing lander at `voids.forkjoin.ai`. Aspirational copy, not the shell. |

Adjacent, and easy to conflate: `apps/void-inference-space`, `apps/void-walking-mcp`,
`open-source/gnosis-math/Gnosis/Void/VoidWalking.lean`, and **NOSYS**
(`apps/nosystem`, `packages/edgework-sdk/src/nosys`, `open-source/a0/src/nosys*.ts`),
which is the machine plane that carries the residuals.

## 2. The substrate, by layer (Verified)

**Lean model — the count is a rejection count.**
`open-source/gnosis-math/Gnosis/BuleyeanProbability.lean` defines `BuleyeanSpace`
with `voidBoundary : Fin numChoices -> Nat` (the rejection count per choice) and
`bounded : forall i, voidBoundary i <= rounds`, then
`weight i = rounds - min (voidBoundary i) rounds + 1` (line 32). The module has
that weight strictly positive (`buleyean_positivity`, line 40) and non-decreasing
for a choice that was not rejected (`buleyean_monotone_nonrejected`, line 51).

**Lean laws — reclaimed in Rustic Church.**
`open-source/gnosis-math/Gnosis/BuleyeanProbabilityLaws.lean` (new; target
`verify-buleyean-probability`) adds `BuleyeanSpace.totalWeight`,
`weight_le_rounds_add_one`, `buleyean_normalization`, `buleyean_concentration`,
`buleyean_laws`, `failureInformationRatio`,
`failure_at_least_as_informative`, `failure_strictly_more_informative`, and the
adversarial dual `floorless_can_collapse` (drop the +1 and a fully-rejected choice
absorbs to zero). It is Init-only, uses `decide` only on closed goals, and passes
`audit-rustic-church.mjs --gate`. `void_boundary_sufficient_statistic` lives in
`Gnosis/Void/VoidWalking.lean` (line 144): the boundary encoding `T*logN` is no
larger than storing every discarded path `(N-1)*T*payloadBits`.

**Engine.** `packages/void-os-core/src/buleyean.ts` (`@a0n/void-os-core`)
ships `buleyeanWeight` (`w = R - min(v, R) + 1`, line 34), `complementDistribution`
(line 42), and `buleDistance` (line 62); `src/self-training.ts` ships `SelfTrainer`
over `FailureSignal` (`crash | timeout | oom | permission | rejection | error`),
with a header naming `THM-VOID-GRADIENT` and `rejection_is_knowledge`.

**Embeddable collector.** `shared-utils/src/self-training/collector.ts` has
`SelfTrainingCollector` with `VoidBoundaryState.rejections`, described as the God
Formula extracted into a standalone library for any SaaS product.

## 3. The exact mapping to the theory (Verified by reading the code)

With `n_i := rounds - voidBoundary i` (the rounds in which `i` was NOT rejected),
`BuleyeanSpace.weight i = n_i + 1` when `voidBoundary i <= rounds`. So the
rejection model is Laplace add-one on the **complement counts**, which is the
corpus identity (counting what is not) rather than a second rule. The add-one
prior is the phantom; `buleyean_positivity` is the never-collapse floor;
`void_boundary_sufficient_statistic` is the claim that the boundary is the
sufficient statistic; `failure_strictly_more_informative` is the corpus claim that
a rejection (N-1 bits) outperforms a selection (1 bit).

## 4. The public machine ledger already exists (Verified)

The end-to-end path is built:

1. `apps/edgework-app/src/lib/knowledge-residual.ts` admits a claim at
   `POST /api/v1/knowledge-residuals` matching schema
   `a0.nosys.knowledge-residual.v1` and relays it to
   `dashrelay.com/api/edgework/knowledge-residuals`.
2. `apps/dashrelay-app/src/lib/knowledge-residual.ts` independently resolves the
   claimed NOSYS work identity, requires a GREEN admission, fetches the signed
   occupant, fetches the JDAG manifest, and checks `residual.*` against the
   manifest field by field (tool, platform, encoded/decoded bytes, digests).
   Authorship is the admission `publisherDid`, not a caller-supplied identity.
3. It idempotently commits a `knowledge.residual.claim` fact to **eddy** (the
   knotchain) through `R2D1SolidityLedgerStore` + `commitMutation`, keyed
   `knowledge-residual:<sha256(claim)>`, returning
   `eddy.knowledge-residual-receipt.v1` with `topologyHash` and `height`.
4. Eddy is publicly readable: `knotchain-scrip-read.ts` serves
   `GET /api/knotchain/scrip/chain?chain=eddy` and block/contract routes,
   unauthenticated, CORS `*`, from the shared `ForgeChain` read model
   (`apps/dashrelay-app/src/lib/README.md`, Scrip read API).

So a void failure can already be published as a signed, append-only, publicly
readable fact. The self-authentication is the load-bearing part: you can only
commit a residual for work you can prove was admitted.

## 5. The gap to the ambitious form

1. **Wrong kind of residual.** `a0.nosys.knowledge-residual.v1` records an
   artifact-materialization residual (encoded/decoded bytes of a native or wasm
   bundle), not a scientific hypothesis rejection. There is no `hypothesisId`,
   `field`, or `rejectionCounts`.
2. **No merge and no collapse.** Each claim is a singleton fact keyed by its own
   digest. There is no per-hypothesis aggregation over the `void-crdt.ts` count
   layer, and no `TensorBayesUrn` point+proven-range collapse.
3. **No public residual index.** Eddy serves blocks and contracts; the
   `eddy_knowledge_residual_claims` D1 table is internal, so there is no
   `GET .../knowledge-residuals?workIdentity=...` read model.
4. **Finality is narrower than the machinery suggests.** `EDDY-FINALITY.md`
   measures eddy as a single writer with a signature: `ATTESTATION_THRESHOLD = 1`,
   `ATTESTATION_COUNT = 3` is declared and never read, so f = 0 and the Lean quorum
   theorems hold vacuously. 'Public' therefore means append-only, signed, and
   notarized-mirrored, not Byzantine-final.

## 6. What now exists (public by default)

The public path is implemented:

* Schema `a0.nosys.void-claim.v1` (`packages/edgework-sdk/src/nosys/index.ts`):
  `hypothesisId`, `field`, `outcome`, `rejections`, `rounds`, `evidenceSha256`,
  and `visibility` pinned to `public` — private is not expressible.
* `apps/edgework-app/src/lib/void-claim.ts` relays `POST /api/v1/void-claims`.
* `apps/dashrelay-app/src/lib/void-claim.ts` verifies the claim against the signed
  NOSYS occupant and the admitted `a0.void-evidence.v1` JDAG, then idempotently
  commits a `void.claim` fact to eddy.
* `apps/dashrelay-app/src/lib/void-claims-read.ts` serves the public read model at
  `GET /api/edgework/void-claims` (unauthenticated, CORS `*`): the claims, the
  per-hypothesis pooled add-one complement posterior
  `(rounds - rejections + 1) / (rounds + 2)`, and each hypothesis share of
  surviving mass `w/W` with `w = n + 1`.

Closed since the first draft:

1. `apps/dashrelay-app/scripts/emit-void-evidence.ts` is the `a0.void-evidence.v1`
   producer. Given `{hypothesisId, field, outcome, rejections, rounds}` it
   encodes the JDAG with `encodeJsonDagV2`, sets both `void.evidenceSha256` and
   `workIdentity.topologyFingerprint` to the artifact `sha256`, and emits the
   matching `a0.nosys.void-claim.v1` body. `--check` self-verifies through the
   real `verifyVoidClaim`.
2. The read model exposes the proven `collapseRange` bracket per field and per
   hypothesis (`{low, high, sound, width, total, budget}`); `width` is the
   target-set cardinality and the singleton target is `width = 1`. The imported
   function is the verified one, not a re-implementation.
3. `GET /api/edgework/void-claims?since=<height>` (or `?mode=feed`) is a bounded
   height-cursor feed, oldest first. It is tail-by-polling, not server push:
   eddy writes are D1 rows with no durable notify channel, so a subscriber
   re-polls the returned cursor.

Still missing:

1. Pooling is a plain sum; per-lab weighting and adversarial count-stuffing are
   not modelled.
2. Eddy is f = 0 (section 5.4): public does not mean tamper-proof against the
   operator.

### The refusal bound (A3)

`open-source/gnosis-math/Gnosis/VoidClaimRefusalBound.lean` binds a published
refusal to `ContrarianUniversalKnowledgeLeavesOnlyChaos`: a public refusal is
free universal stock, so publication preserves level (the refusal is not
knowledge gain), any positive vitality is residual chaos, and `crucialWitness`
is an inhabited carrier of the same free-success shape
(`crucialWitness_carries_refusal_shape`). The *numeric* set-mass bracket
`[width/W, (R+1)*width/W]`, `sound` iff `maxWeight <= budget + 1`, is
explicitly **dismissed** there: it is Buleyean arithmetic proved by
`Gnosis.BuleyeanProbabilityLaws` and mirrored by `@a0n/maybe` `collapseRange`,
not a consequence of the Contrarian module (its fixed `crucialWitness` has no
rejection parameter, so the binding would be a nominal wrapper). The file
records that exact reason.

## 7. Verification

    a0 run open-source-gnosis-math:verify-buleyean-probability   # 4 jobs, green
    (cd open-source/gnosis-math && node scripts/audit-rustic-church.mjs --gate)
    a0 run void-os-core:test                # 84 tests, incl. buleyean.test.ts
    a0 run @a0n/dashrelay-app:test         # includes lib/knowledge-residual.test.ts
    node open-source/gnosis/bin/gnode.js run apps/dashrelay-app/scripts/emit-void-evidence.ts -- --check
    (cd open-source/gnosis-math && lake build Gnosis.VoidClaimRefusalBound)

## 8. Honest boundaries

* Add-one is a uniform prior, not evidence. A field posterior built from
  rejections is only as good as the rejection process; publication bias selects
  the rejections themselves.
* Eddy f = 0. Do not present the ledger as tamper-proof against the operator.
* A publicly readable residual can leak platform, tool, and work-identity
  metadata. Public projection is a deliberate act, not a default.
* The Lean laws are about the arithmetic of the count, not about any scientific
  claim's truth.
