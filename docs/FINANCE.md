# Probability to margin: where the Buleyean/Laplace kernel actually helps

Parent: [Maybe](../README.md) · Theory: [Buleyean probability](./BULEYEAN_PROBABILITY.md) ·
Applications: [APPLICATIONS.md](./APPLICATIONS.md)

> **The honest frame.** Probability does not create edge. It converts an edge you
> already have into three things that need no forecast: **arbitrage/replication**
> (no view at all), **sizing and survival** (Kelly/ruin), and **pricing variance**
> (expectation and tail risk). Where a forecast is required, the math is only as
> good as the counts you feed it. Nothing here is financial advice, and no
> profitability is claimed.

## The one kernel

The God Formula `w_i = R - min(v_i, R) + 1` is Laplace's add-one rule on a
rejection count: `w_i = n_i + 1`, normalizer `N + K`, posterior
`(n_i + 1)/(N + K)` **[proved:** `TensorBayesUrn`**]**. Every finance use below is
one of four consequences: the **floor** (`w_i >= 1`, so nothing is priced at
zero), the **exact rational posterior** (no floating error), the **proven collapse
range** `[width/W, (R+1)*width/W]` (a sound band, not a confidence interval), and
the **consensus law** `cond_i_j M_j = cond_j_i M_i` (a finite no-arbitrage
condition).

---

## 1. Microstructure and execution (the Aeon shape)

Rank execution venues and routes by their **rejection counts** — failed fills,
adverse-selection events, latency misses. The weight is the complement, the floor
means a venue is never permanently banned after a bad streak, and the collapse
range gives a sound band on the best-venue share. This is exactly the shape of
`mesh-local-mcp/src/buleyean-router.ts`; the finance analogue is a
`src/finance/execution-router.ts`.

**What it is not:** it is not alpha. Routing on rejection counts improves
execution *given* a signal; it does not tell you what to trade. Latency is a
systems property that this kernel does not touch.

## 2. Arbitrage and replication (the cleanest mathematical fit)

The consensus law *is* a no-arbitrage condition: two venues pricing the same
joint event must agree once weighted by their face masses. An inconsistent pair
is a witnessed arbitrage (`directed_consensus_fails` is the concrete dual). And
the **rank-one Plücker relation** certifies **replicability**: a payoff is a
stock-and-bond replication exactly when the joint is decomposable
(`wedge i j * wedge k l = wedge i k * wedge j l`). So the corpus already contains
a finite, proved version of "same future payments => same present price."

**What it is not:** real frictions (borrow, fees, impact, settlement) decide
whether the residual is tradeable. The checker finds inconsistency, not profit.

## 3. Bet sizing and survival (Kelly)

Kelly `f* = (bp - q)/b` is only as good as `p`. The Buleyean posterior is the
**add-one** estimate of `p`: `(wins + 1)/(trials + 2)` for the binary case, which
is a *regularized* Kelly input. Two concrete consequences:

* the **floor** stops a small sample from producing `p = 0` and killing a live
  bet;
* the **collapse range** gives a sound lower bound to size at — an
  ambiguity-averse / fractional Kelly that trades a little growth for survival.

**Adversarial dual:** on a small sample the maximum-likelihood `p = wins/trials`
overbets relative to add-one (1 win in 1 trial implies `p = 1`). The test ships
that witness. **What it is not:** Kelly maximizes expected log growth under a
*correct* model; it is famously brutal to a wrong `p` or `b`, and it is not a
guarantee of survival.

## 4. Asymmetric obligation (newsvendor and overbooking)

When the penalty for being short differs from the penalty for being long, the
optimal commitment is the **critical fractile** `c_u / (c_u + c_o)` of the demand
distribution — not the mean. The add-one posterior is an exact rational
distribution, so that quantile is computable by ranking categories and
accumulating mass, with no reals; the collapse range gives a robust variant.
Airline overbooking and hospital staffing are the canonical cases.

**Adversarial dual:** ordering the mean is provably suboptimal whenever
`c_u != c_o` (the test exhibits it).

## 5. Expectation and tail-risk pricing

Exact rational expectations, plus the property that matters most for tail risk:
**no state is ever priced at zero.** The `+1` puts mass `1/(N+K)` on a
catastrophe you have never observed. The maximum-likelihood posterior prices it
at `0` — an unbounded underpricing of the tail. The collapse range then bounds
the mass above any loss threshold.

**What it is not:** add-one is a **uniform** prior, so it *overprices* unobserved
events and *underprices* known-but-rare ones (it shrinks toward uniform). For
genuinely fat tails you may want `alpha < 1` (Haldane/Jeffreys) — parameterize
the phantom ball and disclose the choice. A sound bound is not a tight estimate.

## 6. Bayesian action (posterior into utility)

A probability makes money only attached to costs. The decision is `argmax` of
expected utility over actions, and the **collapse range gives a regret bound**:
the worst-case gap between acting now and acting on the best state inside the
band. The CRDT merges evidence from several signals or desks by addition — in
any order, idempotently — *before* the decision.

**What it is not:** utility functions and likelihoods are inputs; the primitive
faithfully propagates whatever error they contain.

---

## Falsification (what would make each primitive worthless)

1. **Execution router:** if rejection counts do not predict out-of-sample
   execution quality, the ranking is noise.
2. **Arbitrage checker:** if every flagged inconsistency disappears after
   realistic frictions, it finds no tradeable arbitrage.
3. **Kelly:** if add-one Kelly underperforms on a backtest because the prior is
   too conservative, the regularization costs more than it saves.
4. **Newsvendor:** if demand is not exchangeable across categories, the
   critical-fractile quantile is mis-specified.
5. **Tail pricing:** if the true tail is heavier than uniform, add-one still
   underprices it — disclose the prior and stress the bound.
6. **Bayesian action:** if the loss matrix is wrong, the optimal action is
   wrong with it.

## Where this lands in the monorepo

- **`packages/forkjoin-hft` already speaks this language.** Its README's
  observables are the God Formula terms (`w/(R+1)`, `v/R`, `R+1-w`, `w-1`) and V42
  is "Phantom Buleyean Variance". The sharpest fix is `src/maker-regime-filter.ts`,
  which demotes below-average voters to **weight 0** — the absorbing zero the floor
  forbids. The floored demotion keeps a voter alive and revivable; hard removal is
  reserved for explicit refutation. Its `docs/buleyean-hft-bridge.md` lists
  **Phase D (multi-venue negative space)** as planned; the consensus no-arbitrage
  checker is that phase, now being implemented.
- **`packages/universal-prediction-rt`'s certified predict gate** is where the
  proven collapse range belongs: a RED/GREEN serve decision made on a sound
  interval instead of a point estimate.
- **`apps/hft-cloud`** is the product/UI surface over the engine; the primitives
  above feed its strategy and intel layers.

## What is being built

| Artifact | What it gives you |
|---|---|
| `maybe/src/finance/risk.ts` | exact expectation, never-zero tail, tail-mass bound + MLE dual |
| `maybe/src/finance/kelly.ts` | add-one Kelly, conservative range Kelly + overbet dual |
| `maybe/src/finance/newsvendor.ts` | critical-fractile order quantity + mean-is-wrong dual |
| `maybe/src/finance/execution-router.ts` | rejection-ranked venues, floor, sound band |
| `maybe/src/finance/arbitrage.ts` | the consensus law as a no-arbitrage witness |
| `Gnosis/TensorBayesFinance.lean` | the same risk facts proved in the finite kernel |

Every one is an exact-rational, testable primitive. None of them forecasts. Use
them to size, to price, and to refuse bets that would end the game.
