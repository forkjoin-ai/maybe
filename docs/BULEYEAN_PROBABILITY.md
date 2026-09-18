# What Buleyean probability is, and what it is not

Parent: [Maybe](../README.md) · Paper: [Buleyean Probability on the Fisher Manifold](../paper/buleyean-manifold.md) · Code: [src](../src/README.md)

> One line: Buleyean probability is the **void-dual (rejection-counting) reading of Laplace's rule of succession**. Counting what is NOT, and counting what is plus one phantom observation per outcome, are the same arithmetic.

## The identity

Take an urn with `K` colours, `N` draws, and `n_i` draws of colour `i`.

    rejections   v_i = N - n_i            (count what is NOT colour i)
    Buleyean     w_i = N - min(v_i, N) + 1
    so           w_i = n_i + 1            (the Laplace numerator)
    normalizer   sum_i w_i = N + K
    posterior    P(i) = w_i / sum_j w_j = (n_i + 1) / (N + K)

`(n_i + 1) / (N + K)` is Laplace's rule of succession: start with one phantom
ball of every colour, then draw. The `+1` is that phantom ball. The map
`n_i ↦ v_i = N - n_i` is an involution, so rejection counts and occurrence
counts carry exactly the same information: neither reading adds or loses data.
That is the whole point — the corpus's *frequentist theory on complement sets*
(counting what is NOT) and the *Bayesian add-one urn* (counting what is, plus a
phantom) are the same object in two coordinate systems.

Proved in Lean: `Gnosis.TensorBayesUrn.urn_weight_is_successor`,
`urn_weight_total`, `urn_laplace_normalized`. Checked exactly at runtime:
`maybe/src/urn.ts` `assertUrnLaplaceIdentity` (see `urn.test.ts`).

## What Buleyean probability IS

1. **A re-coordinatization of a finite count vector.** `v_i = N - n_i`. Same
   data, different axis.
2. **A finite, exact distribution.** Integer weights `w_i >= 1`; exact
   denominator `N + K`; no floating point, no temperature, no exponent.
3. **Laplace's add-one rule.** The posterior predictive of a `Dirichlet(1,...,1)`
   prior for a categorical/multinomial, i.e. the standard add-one smoothing.
4. **A strict-positivity (never-collapse) guarantee.** Every outcome keeps
   weight `>= 1`, including outcomes never observed, so finite evidence cannot
   zero a live hypothesis.
5. **A discipline for rejection data.** It answers *which outcome has been least
   rejected?* — the natural question when what you collected is failures,
   falsifications, or void counts rather than successes.
6. **A finite, executable, adversary-checked object.** Mechanized Init-only in
   `Gnosis.*`; executable in `@a0n/maybe`; shipped with red-team duals (the
   floor's blindspot, weight saturation, directed consensus).
7. **The two-perspective face of a larger joint.** Two axes give the consensus
   law over all ordered pairs (`Gnosis.TensorBayes`); the 55-axis Pleroma
   generalizes it.

## What Buleyean probability is NOT

1. **Not a new probability theory, and not "Laplacian".** The arithmetic is
   Laplace's (1812). "Laplacian" is already taken by the Laplace operator,
   transform, matrix, and distribution. "Buleyean" names the coordinate system
   and the discipline, not a new calculus.
2. **Not an alternative to Bayesian inference.** It is one canonical Bayesian
   posterior — uniform `Dirichlet` prior — written in rejection coordinates.
   It does not compete with Bayes; it is a presentation of one case of it.
3. **Not objective or calibration-free.** Add-one is a prior choice. It biases
   toward uniformity. Haldane (`0`), Jeffreys (`1/2`), and Krichevsky-Trofimov
   give different posteriors; the corpus uses `+1` and does not claim `+1` is
   uniquely correct.
4. **Not a general fix for zero counts.** An unobserved outcome gets weight `1`,
   but that is a prior ball, not evidence. With many unobserved outcomes the
   phantom mass dominates and the posterior is mostly prior.
5. **Not measure-theoretic or continuous.** Finite and discrete only. No
   densities, integrals, or infinite sample spaces; the corpus keeps Reals and
   measure theory out of this kernel deliberately.
6. **Not the softmax.** It is a linear, temperature-free complement weight. The
   Fisher-Rao geometry in the paper is a separate, additional layer, not part of
   the probability kernel.
7. **Not valid when rejections are not the complement of occurrences.** The
   identity assumes `v_i = N - n_i`. If rejection counts come from anything
   else, the equivalence fails, and `assertUrnLaplaceIdentity` reports exactly
   that failure rather than smoothing over it.
8. **Not a claim that one number is enough.** The 55-axis collapse returns a
   point plus a *proven* floor-ceiling range. That range is a bound (every
   weight lies in `[1, R+1]`), not a confidence interval, and the point depends
   on the chosen target set.

## Why the name "Buleyean" is kept

- The name marks the **coordinate system and the discipline**: weight from what
  was NOT observed, the void boundary, the floor as a capability. That framing is
  what makes the residual-void and never-collapse machinery natural across
  dimensions.
- Credit where due, honestly: Bayes (1763) → Laplace (1812) → Jeffreys (1946) →
  Dirichlet-multinomial. "Buleyean" names the discipline; "Laplace" names the
  arithmetic; the identity above proves they coincide, so the framework claims no
  new probability theory.

## Consequences (what the identity buys)

- **Count what is easy to count.** Failures and rejections suffice; the Bayesian
  posterior comes for free.
- **Smoothing and never-collapse are one instruction.** The `+1` that prevents a
  zero weight is the same `+1` that prevents overfitting a zero count.
- **A human explanation.** The urn is the story, `P(i) = (n_i + 1)/(N + K)` is
  the number, and the 55-axis hyper-joint is the multi-perspective
  generalization.
- **Cross-language safety.** The identity is checked exactly in Lean and in
  TypeScript, so drift between the mechanized and executable surfaces is caught.

## Where to look

| Question | Artifact |
|---|---|
| What is the formula? | `Gnosis/GodFormula.lean` (`godWeight`) |
| The corpus's complement kernel | `Gnosis/BuleyeanProbability.lean` (`BuleyeanSpace.weight`) |
| The urn identity (proved) | `Gnosis/TensorBayesUrn.lean` (`urn_weight_is_successor`) |
| The identity (executable) | `open-source/maybe/src/urn.ts` (`assertUrnLaplaceIdentity`) |
| The two-axis / hyper-joint generalization | `Gnosis/TensorBayes.lean` |
| The 55-axis Pleroma and Skyrms bridge | `Gnosis/TensorBayesSkyrms.lean` |
| The human collapse (point + range) | `../src/tensor-bayes-55.ts` (`collapse`) |
| An interactive explanation | `open-source/gnosis-math/tensor-bayes-55.html` |
