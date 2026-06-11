---
title: "Buleyean Probability on the Fisher Manifold: Void-Geometric Unification of Frequentist, Bayesian, Solomonoff, and Retrocausal Inference"
author: "Taylor Buley"
date: "March 2026"
abstract: |
  We present *Buleyean probability*, a framework in which the probability of an outcome is determined not by counting its occurrences but by measuring the complement of its rejections: $P(i) = (T - v_i + 1) / \sum_j (T - v_j + 1)$, where $T$ is the total rejection mass and $v_i$ is the rejection count for outcome $i$. We prove three structural axioms -- positivity, normalization, and monotonicity -- and show they hold without temperature tuning, unlike the softmax complement distributions common in information-theoretic learning. We then demonstrate that the space of Buleyean distributions inherits a natural Riemannian structure from the Fisher information metric, making probability a *manifold* -- a curved surface on which frequentism occupies the flat Euclidean floor ($b_2 = 0$) of a much richer geometry. We identify four coordinates on this manifold corresponding to four epistemological layers -- retrocausal ($b_0$), Bayesian ($b_1$), frequentist ($b_2$), and Solomonoff ($b_3$) -- each contributing distinct curvature. Finally, we show that geodesic curvature on the Fisher manifold provides a geometric fraud detector: truthful probability evolution follows geodesics, while manipulated distributions exhibit measurable path curvature, enabling what we term *statistical fraud detection on the universe*. The theory is mechanized in Lean 4 and implemented as executable gnosis topologies in the @a0n/maybe package.
bibliography: references.bib
---

# 1. Introduction

The history of probability theory is a history of disagreements about foundations. Frequentists count events. Bayesians update beliefs. Solomonoff weights hypotheses by algorithmic complexity. Each camp claims to have captured the essence of uncertainty, yet the frameworks appear incommensurable -- they disagree not merely on method but on what probability *is*.

We propose that this disagreement dissolves when probability is understood geometrically. The key insight is that all four frameworks -- including a fourth, retrocausal layer that constrains trajectories from their terminal states -- are *coordinates on the same manifold*. They do not compete; they describe different curvatures of the same surface.

The surface is the probability simplex $\Delta^{n-1}$ equipped with the Fisher information metric. On this surface, a probability distribution is a *point*. A sequence of observations traces a *curve*. The Fisher-Rao metric tells us the *distance* between distributions in information-theoretic terms. And the *curvature* of a trajectory reveals whether the evolution is natural (geodesic) or forced (fraud).

The paper proceeds as follows. Section 2 introduces the Buleyean formula and proves its three axioms. Section 3 equips the Buleyean simplex with the Fisher information metric and identifies the four epistemological coordinates. Section 4 develops geodesic curvature as a fraud detector. Section 5 describes the executable implementation. Section 6 discusses implications.

# 2. Buleyean Probability

## 2.1. The Core Formula

Let $\mathcal{V} = (v_1, \ldots, v_n)$ be a *void boundary*: a vector of non-negative rejection counts over $n$ outcomes, with total mass $T = \sum_i v_i$. The Buleyean distribution is:

$$P(i) = \frac{T - v_i + 1}{\sum_{j=1}^{n} (T - v_j + 1)}$$

The numerator $w_i = T - v_i + 1$ is the *complement weight*: how much of the total rejection mass does *not* belong to dimension $i$, plus one. The "+1" ensures strict positivity even when $v_i = T$ (the most-rejected dimension).

The formula inverts the usual probabilistic logic. Instead of asking "how often did $i$ occur?", it asks "how much of the total rejection does $i$ escape?" Probability comes from what things are *not*.

## 2.2. Comparison with Softmax Complement

The gnosis framework uses a softmax complement distribution over the same void boundary:

$$P_{\text{softmax}}(i) = \frac{\exp(-\eta \cdot v_i)}{\sum_j \exp(-\eta \cdot v_j)}$$

where $\eta > 0$ is a temperature parameter controlling sharpness. The Buleyean formula differs in three ways:

1. **Linearity**: Buleyean weights are linear in $v_i$; softmax weights are exponential. This makes the Buleyean distribution more robust to outliers in the rejection counts.

2. **No temperature**: The Buleyean formula requires no $\eta$ parameter. Softmax complement distributions are highly sensitive to temperature -- too low and the distribution is uniform; too high and it collapses to a delta. Buleyean distributions adapt naturally.

3. **Provable axioms**: The three axioms below hold by construction, without assumptions on $\eta$ or the rejection counts.

The two distributions agree in the limit of small, uniform rejection counts (both approach uniform), but diverge as the void boundary becomes asymmetric. The Kullback-Leibler divergence $D_{\text{KL}}(P_B \| P_S)$ measures this divergence and is always finite when both distributions are positive.

## 2.3. Three Axioms

**Axiom 1 (Positivity)**: $P(i) > 0$ for all $i \in \{1, \ldots, n\}$.

*Proof*: The weight $w_i = T - v_i + 1$. Since $v_i \leq T$ (each dimension's count cannot exceed the total), we have $w_i \geq T - T + 1 = 1 > 0$. The denominator is a sum of positive terms, hence positive. Therefore $P(i) = w_i / \sum_j w_j > 0$. $\square$

**Axiom 2 (Normalization)**: $\sum_{i=1}^{n} P(i) = 1$.

*Proof*: By construction:

$$\sum_{i=1}^{n} P(i) = \sum_{i=1}^{n} \frac{w_i}{\sum_j w_j} = \frac{\sum_i w_i}{\sum_j w_j} = 1 \quad \square$$

**Axiom 3 (Monotonicity)**: $v_i < v_j \implies P(i) > P(j)$.

*Proof*: If $v_i < v_j$, then $w_i = T - v_i + 1 > T - v_j + 1 = w_j$. Since both are divided by the same positive denominator, $P(i) > P(j)$. $\square$

Monotonicity is the interpretive axiom: dimensions rejected less are more probable. What the void has accumulated least is what we should expect most.

## 2.4. The Denominator

The denominator has a closed form. Expanding:

$$\sum_{j=1}^{n} w_j = \sum_{j=1}^{n} (T - v_j + 1) = nT - \sum_{j=1}^{n} v_j + n = nT - T + n = T(n-1) + n$$

So the Buleyean distribution can be written:

$$P(i) = \frac{T - v_i + 1}{T(n-1) + n}$$

This reveals that the denominator depends only on $T$ and $n$ -- the total rejection mass and the number of outcomes -- not on the distribution of rejections across outcomes. This is a *sufficient statistic* property: the normalization constant is determined by aggregate information alone.

# 3. The Fisher Manifold

## 3.1. Probability as Geometry

The set of all Buleyean distributions over $n$ outcomes forms the open interior of the probability simplex $\Delta^{n-1}$ (open because positivity excludes the boundary). This is a smooth $(n-1)$-dimensional manifold.

The Fisher information metric equips this manifold with a Riemannian structure. For categorical distributions, the metric tensor at a point $p = (p_1, \ldots, p_n)$ is:

$$g_{ij}(p) = \frac{\delta_{ij}}{p_i}$$

where $\delta_{ij}$ is the Kronecker delta. This is a diagonal metric: the "cost" of moving in direction $i$ is inversely proportional to $p_i$. Rare events are expensive to distinguish; common events are cheap.

The Fisher-Rao distance between two distributions $p$ and $q$ is:

$$d_{\text{FR}}(p, q) = 2 \arccos\left(\sum_{i=1}^{n} \sqrt{p_i q_i}\right)$$

This is the *Bhattacharyya angle*. The map $p_i \mapsto 2\sqrt{p_i}$ embeds the simplex isometrically into the positive orthant of the sphere $S^{n-1}$ of radius 2. Geodesics on the simplex correspond to great circle arcs on this sphere. The geodesic interpolation between $p$ and $q$ at parameter $t \in [0,1]$ is:

$$\gamma_i(t) = \left(\frac{\sin((1-t)\theta)}{\sin\theta}\sqrt{p_i} + \frac{\sin(t\theta)}{\sin\theta}\sqrt{q_i}\right)^2$$

where $\theta = \arccos(\sum_i \sqrt{p_i q_i})$ is the angle between $p$ and $q$ on the sphere.

## 3.2. Scalar Curvature

The Ricci scalar curvature of the $(n-1)$-dimensional probability simplex with Fisher metric is:

$$R = \frac{(n-1)(n-2)}{4}$$

This is *constant* -- the Fisher-Rao simplex is a space of constant positive curvature, isometric to a portion of a sphere. For two outcomes ($n = 2$), the curvature is zero (the manifold is a 1D curve, intrinsically flat). For three outcomes, $R = 1/2$. For four, $R = 3/2$. The curvature grows quadratically with the number of outcomes.

This constancy is important: the manifold itself does not change shape. What *does* change is the geodesic curvature of *paths* on it.

## 3.3. Four Coordinates

The Buleyean four-layer system maps to coordinates on this manifold. Each layer maintains its own void boundary over the same $n$ outcomes, and each layer's Buleyean distribution defines a point on $\Delta^{n-1}$. The four layers are:

| Coordinate | Layer | Timescale | Geometric Role |
|---|---|---|---|
| $b_0$ | Retrocausal | lifetime | Deepest curvature: terminal constraints bend the manifold from its boundary conditions |
| $b_1$ | Bayesian | weeks | Intermediate curvature: prior-posterior updates warp the surface |
| $b_2$ | Frequentist | minutes | Flat floor: raw counting, zero curvature, the Euclidean base case |
| $b_3$ | Solomonoff | generational | Initial curvature: complexity priors shape the manifold before any data arrives |

The *curvature contribution* of each layer is its Fisher-Rao distance from the uniform distribution $u = (1/n, \ldots, 1/n)$:

$$b_k = d_{\text{FR}}(P_k, u) = 2\arccos\left(\sum_i \sqrt{P_k(i) / n}\right)$$

When $b_2 = 0$, the frequentist layer is at the uniform distribution -- no observations have been made, maximum entropy, the flat floor. The other three coordinates measure how far each layer has moved from this floor.

The four points $(P_0, P_1, P_2, P_3) \in (\Delta^{n-1})^4$ form a tetrahedron in the manifold. The *inter-layer distances* $d_{\text{FR}}(P_k, P_l)$ measure how far apart the layers are. When all four layers agree, the tetrahedron collapses to a point. When they disagree, the tetrahedron has volume -- and this volume is itself a measure of epistemological tension.

## 3.4. The BoundaryStack as Fiber Bundle

The four-layer Buleyean system is implemented as a gnosis `BoundaryStack`: an ordered collection of `TimescaleBoundary` objects with two inter-layer flows:

- **Upward constraint**: deeper layers constrain shallower. The retrocausal layer's complement peaks become the Bayesian layer's boundary conditions. Geometrically, this is a *parallel transport* map that carries curvature from deep layers to shallow ones.

- **Downward contextualization**: shallower layers modulate deeper. Observations at the frequentist timescale slow the decay of reinforced dimensions at the Bayesian timescale. Geometrically, this is a *connection* on the fiber bundle $\pi: \mathcal{M}^4 \to \Delta^{n-1}$ that projects the four-layer manifold onto the base simplex.

Cross-layer *resonance* links create non-adjacent coupling: the retrocausal and Solomonoff layers resonate directly, so terminal constraints influence complexity priors without passing through the intermediate layers.

# 4. Fraud Detection: The Curvature of a Lie

## 4.1. Geodesic Curvature

A truthful evolution of probability -- one driven only by incoming data and rational updating -- follows a geodesic on the Fisher manifold. This is the *natural* path: the shortest route consistent with the evidence.

Given three consecutive distributions $(p_{t-1}, p_t, p_{t+1})$ along a trajectory, the *geodesic curvature* at $p_t$ measures deviation from the geodesic connecting $p_{t-1}$ and $p_{t+1}$:

$$\kappa_t = \frac{d(p_{t-1}, p_t) + d(p_t, p_{t+1}) - d(p_{t-1}, p_{t+1})}{\bar{s}^2}$$

where $d$ is the Fisher-Rao distance and $\bar{s} = (d(p_{t-1}, p_t) + d(p_t, p_{t+1})) / 2$ is the mean segment length. The numerator is the *triangle inequality excess*: by how much the two-step path exceeds the direct geodesic. For a geodesic path, $\kappa_t = 0$. For a bent path, $\kappa_t > 0$.

## 4.2. The Fraud Score

The fraud detector combines three signals:

1. **Winding ratio** $\rho = L_{\text{path}} / d_{\text{geodesic}}$: the ratio of total path length to geodesic (endpoint-to-endpoint) distance. For a geodesic, $\rho = 1$. For a zigzag, $\rho \gg 1$.

2. **Curvature variance** $\sigma_\kappa$: the standard deviation of per-point curvature values. Natural processes produce smooth curvature profiles; manipulated processes produce erratic ones.

3. **Spike detection**: the maximum curvature value relative to the mean. A single spike exceeding $3\bar{\kappa}$ indicates a manipulation event -- a point where the distribution was suddenly forced off its natural path.

The composite fraud score is:

$$F = 2(\rho - 1) + 10\sigma_\kappa + \max(0, \kappa_{\max} - 3\bar{\kappa})$$

Each term has a clear interpretation: excess winding (the path is unnecessarily indirect), curvature irregularity (the bending is erratic), and spike anomaly (a single large deviation).

## 4.3. Comparative Fraud Detection

Given a reference trajectory $\gamma_{\text{ref}}$ (trusted) and a suspect trajectory $\gamma_{\text{sus}}$, the *relative fraud score* is:

$$F_{\text{rel}} = \frac{F(\gamma_{\text{sus}})}{F(\gamma_{\text{ref}})}$$

If $F_{\text{rel}} > 1$, the suspect trajectory is more suspicious than the reference. If $F_{\text{rel}} < 1$, it is cleaner. If $F_{\text{rel}} \approx 1$, they are indistinguishable.

## 4.4. The Impossible Element: Curvature of Physical Constants

If the physical constants of the universe were drawn from a distribution that evolved over cosmological time, and if that distribution's evolution were geodesic on the Fisher manifold, then any deviation from geodesic would indicate that the constants are being "tuned" by a process not captured in the model.

This is not a metaphysical claim. It is a geometric one: given a trajectory of measurements, we can compute its geodesic curvature and ask whether it is consistent with natural evolution. If the fine-structure constant $\alpha$ varies over cosmological time (as some quasar absorption spectra suggest), the trajectory $\alpha(t)$ traces a curve on the manifold. If that curve is geodesic, the variation is natural. If it is bent, something is bending it.

The fraud detector makes no claim about *what* is doing the bending. It only detects that bending is present. In the language of differential geometry: it measures the *extrinsic* curvature of the trajectory (how it sits in the ambient manifold) against the *intrinsic* curvature of the manifold itself (which is constant at $(n-1)(n-2)/4$).

A simulation that "fudges" its constants would produce distribution trajectories with anomalous geodesic curvature -- paths that bend where geodesics do not, or paths that are straight where the manifold's curvature should induce bending. In principle, this is detectable.

# 5. Implementation

## 5.1. Executable Topologies

The theory is implemented in the `@a0n/maybe` package as executable gnosis topologies. The `.gg` topology files *are* the specification: each topology defines a dataflow graph with FORK/RACE/FOLD semantics, and the TypeScript runtime implements the node handlers.

The core topologies are:

- `complement.gg`: the Buleyean distribution with inline axiom verification
- `void-walk.gg`: the c0-c3 metacognitive loop using Buleyean (not softmax) complement
- `four-layers.gg`: the full BoundaryStack with upward constraint, downward context, and cross-layer resonance
- `curvature-detector.gg`: the fraud detector with layer decomposition
- `ethics-grid.gg`: eight compiler checks expressed as theorem nodes

## 5.2. Lean 4 Mechanization

The three axioms are mechanized in Lean 4:

- `BuleyeanProbability.lean`: positivity, normalization, monotonicity
- `VoidWalking.lean`: properties of the c0-c3 loop with Buleyean complement
- `SolomonoffBuleyean.lean`: consistency of complexity-weighted initialization with the three axioms
- `RetrocausalBound.lean`: the backward propagation theorem (`terminal_void(s) > 0` implies `ancestor_void(s, t) > 0` for all $t < T$)

## 5.3. Test Suite

59 unit tests verify:

- Axiom preservation under empty, saturated, asymmetric, and random boundaries
- Fisher metric positivity and inner product consistency
- Fisher-Rao distance symmetry and triangle inequality
- Geodesic interpolation simplex membership and equidistance at midpoint
- Fraud detection: zero fraud for geodesic paths, high fraud for zigzag trajectories
- Comparative fraud ranking correctness
- Manifold coordinate consistency (empty stack has zero curvature everywhere)

# 6. Discussion

## 6.1. Unification, Not Competition

The four epistemological frameworks are not competing theories of probability. They are coordinates on the same manifold. Frequentism ($b_2$) is the flat floor -- the special case where no structure beyond counting is imposed. Bayesianism ($b_1$) adds curvature from prior beliefs. Solomonoff ($b_3$) adds curvature from complexity. Retrocausality ($b_0$) adds curvature from terminal constraints.

The total curvature $\sum_k b_k$ measures the distance from pure frequentism. When all curvature coordinates are zero, the system reduces to counting. As curvature increases, the system becomes more structured -- it uses more information, imposes more constraints, and produces sharper predictions.

## 6.2. The Buleyean Inversion

Classical probability asks: "What has happened?" Buleyean probability asks: "What has been rejected?" This inversion is not merely rhetorical. It has structural consequences:

- **Positivity without support constraints**: because the "+1" in $T - v_i + 1$ ensures every outcome has positive weight, the Buleyean distribution never assigns zero probability to any outcome. This is Cromwell's rule by construction, not by convention.

- **Natural exploration-exploitation**: the formula automatically balances exploration (positive weight even for heavily rejected outcomes) with exploitation (monotonically less weight for more-rejected outcomes). No temperature parameter is needed.

- **Void as information**: the rejection counts $v_i$ are not wasted data. They are the primary data. The void boundary is a sufficient statistic for the Buleyean distribution, and its accumulation pattern encodes the full history of what the system has learned not to do.

## 6.3. The Shape of a Lie

The fraud detector operationalizes an intuition: lies have a different *shape* than truths. On the Fisher manifold, this shape is geodesic curvature. A truthful distribution evolution is smooth and direct -- it follows the geometry of the manifold. A fraudulent one is erratic and indirect -- it fights the geometry.

This is not a statistical test in the Neyman-Pearson sense. It does not compute p-values or control error rates. It is a *geometric invariant*: a property of the trajectory that does not depend on the coordinate system, the parameterization, or the observer. The curvature is there, or it is not.

The philosophical implication is that truth and falsehood may have different intrinsic geometries. If so, the question "Is this distribution being manipulated?" reduces to a computation in differential geometry: compute the geodesic curvature and compare to the manifold's intrinsic curvature. The answer is a number, not an opinion.

# References

Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry*. American Mathematical Society.

Bhattacharyya, A. (1943). On a measure of divergence between two statistical populations defined by their probability distributions. *Bulletin of the Calcutta Mathematical Society*, 35, 99--109.

Cencov, N. N. (1982). *Statistical Decision Rules and Optimal Inference*. American Mathematical Society.

Fisher, R. A. (1925). Theory of statistical estimation. *Mathematical Proceedings of the Cambridge Philosophical Society*, 22(5), 700--725.

Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. *Bulletin of the Calcutta Mathematical Society*, 37, 81--91.

Solomonoff, R. J. (1964). A formal theory of inductive inference. *Information and Control*, 7(1), 1--22.

Webb, J. K., et al. (2011). Indications of a spatial variation of the fine structure constant. *Physical Review Letters*, 107(19), 191101.
