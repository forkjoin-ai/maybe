# Does a richer kernel recover softmax concentration at O(N) cost?

An executable follow-up to `../softmax-vs-buleyean-attention/` for the
`@a0n/maybe` package. It is dependency-free (no imports at all, not even Node
builtins), touches no `.lean` file, and lives entirely under
`open-source/maybe/experiments/softmax-vs-kernel-attention/`.

## Verdict: PARTIAL -- richness removes the ceiling, but not at m << N

The prior experiment showed the affine Buleyean rule `w_j = R + s_j + 1` is
exp()-free, exactly integer-checkable and O(N), but has a hard `O(1/N)`
concentration ceiling. The surviving hypothesis was that a **richer kernel** --
one whose weight grows faster than a linear ramp in the score -- recovers
concentration at O(N) cost and without exp().

The measured answer is **partial, and the O(N) dream is not recovered**:

- **Concentration is recoverable.** The polynomial kernel `w_j=(1+s_j)^p`
  raises the target mass monotonically with `p` (`0.0303` at p=1 to `0.6468`
  at p=14), so the affine `O(1/N)` ceiling is genuinely gone. The exp()
  random-feature family also converges in principle.
- **But the matching feature budget is enormous.** At the sharp retrieval the
  softmax reference solves (`beta=8`: recall.value `0.8516`, target mass
  `0.6421`), the polynomial kernel first matches at **p=14, m=C(22,14)=319770,
  m/N=4996.4, costRatio=9603.5x**. At that budget its `m x dv` sufficient
  statistic is 20.5 MB against softmax's 16 KB score matrix (~1250x more memory).
- **Positive random features (Performer/FAVOR+) never match up to m=4096
  (m/N=64).** Their per-feature relative standard deviation is
  `sqrt(e^{2 beta}-1)`, so the m needed for 10% error at `beta=8` is
  `(e^{16}-1)/0.01 ~ 8.9e8`. Below that budget the estimator is saturated
  (TV log-log slope `-0.089`, not `-0.5`).
- **A homogeneous ReLU random feature (arc-cosine kernel) never matches** --
  being homogeneous, it has no temperature knob at all.
- **The rectified kernel `max(s_j,0)^p` DOES match** (p=6: recall `0.8203`,
  target mass `0.6536`) but only as an `O(N^2)`, non-factorizing reference:
  the kink has no finite monomial feature map, so it buys no O(N) path.

**m-vs-N crossover**: the matching budget is thousands of times N at every N
tested -- `m/N = 2734.9` (N=16), `1367.4` (N=32), `4996.4` (N=64),
`33532.4` (N=128). It never approaches `m << N`. There is **no** crossover
back to the linear-attention regime for sharp softmax.

The exactness win is preserved: the polynomial path is bit-exactly
integer-checkable (**400 trials, 806 rows, 3255 BigInt field comparisons, 0
mismatches**) and uses **zero** `exp()` calls. Exact factorization and zero
transcendentals were never the bottleneck; the feature budget is.

## Files

| file | role |
| --- | --- |
| `kernel-attention.mjs` | Core: RNG, cost counter, recall task, softmax O(N^2), polynomial monomial feature map + BigInt exact sufficient statistics, positive random features (Performer) + homogeneous ReLU features, metrics. |
| `run.mjs` | Main runnable entry: cost counter, BigInt bit-exactness, feature-budget sweep at matched FLOPs, m-vs-N crossover, PRF variance scaling, verdict. |
| `README.md` | This file, including the exact observed output. |

## Run

Real Node (the repo `node` is a shim), per the package environment:

```sh
cd open-source/maybe/experiments/softmax-vs-kernel-attention
/opt/homebrew/bin/node run.mjs     # full report, ~3 s
```

Everything is seeded (`mulberry32`) and deterministic; no network, no
filesystem, no dependencies.

## Mathematical setting

Score `s_ij = q_i . k_j` with unit keys and unit queries, so `|s_ij| <= 1`
(measured `max|s| = 0.9928`). `beta` is the inverse temperature / logit scale.

1. **Softmax reference** `w_ij = exp(beta s_ij)/Z`, `O(M N (d+dv))` plus
   `M N` `exp()` calls.

2. **Polynomial kernel** `w_ij = (c + s_ij)^p`. With
   `(c+q.k)^p = sum_{|alpha|<=p} [p!/(alpha! (p-|alpha|)!)] c^{p-|alpha|}
   prod_i (q_i k_i)^{alpha_i}`, the explicit feature map is
   `phi_alpha(x) = sqrt(p!/(alpha! (p-|alpha|)!)) c^{(p-|alpha|)/2}
   prod_i x_i^{alpha_i}`, of dimension `m = C(d+p, p)`. Build
   `S = sum_j phi(k_j) V_j^T` (`m x dv`) and `s = sum_j phi(k_j)` in
   `O(N m dv)`, answer in `O(M m dv)`: linear in N, **zero exp()**, exact.
   `c=1` is used so `c+s >= 0` (valid distribution) and is the most
   concentrated shift. The monomial feature map is a **proper PSD kernel**.

3. **ReLU^p variant** `w_ij = max(c+s_ij, 0)^p`. With unit keys and
   `c=1` the clamp **never fires** (`clampFrac=0.0000`), so `ReLU^p` is
   *exactly* the polynomial kernel. It only differs with `c < max|s|` (the
   sweep uses `c=0`, clampFrac `0.4937`). The clamp is a kink, so the function
   is not a polynomial in `s` and has **no finite monomial feature map**; it is
   computed honestly `O(N^2)` as a reference (the integer trials show
   `780/806` field mismatches against the polynomial sufficient statistic).
   Note the homogeneous ReLU *random* features below approximate a **different**
   kernel (arc-cosine), not `max(s,0)^p`.

4. **Positive random features (Performer/FAVOR+)** for `exp(beta q.k)`:
   `phi_r(x) = exp(omega_r.(sqrt(beta) x) - beta||x||^2/2)/sqrt(m)`,
   `omega_r ~ N(0,I)`, so `E[phi(q).phi(k)] = exp(beta q.k)`. This uses
   `(N+M) m` `exp()` calls (not `N` per query), i.e. it trades the `N`-per-query
   transcendental for a budget of `m`. Per-feature
   `E[X]=exp(beta s)`, `E[X^2]=exp(2 beta + 4 beta s)`, hence relative std of
   the m-average `~ sqrt((e^{2 beta}-1)/m)`.

5. **Homogeneous ReLU random features** `phi_r(x) = max(omega_r.x, 0)/sqrt(m)`.
   This is exp()-free and positive, but homogeneous: scaling `x` scales every
   feature equally and normalization cancels, so it has **no temperature knob**.
   Its kernel is the arc-cosine kernel (checked numerically against the
   closed form), a bounded kernel.

6. **BigInt exact mode.** The polynomial coefficients
   `p!/(alpha! (p-|alpha|)!) c^{p-|alpha|}` are integers, so with integer scores
   the sufficient statistics `T_alpha = sum_j prod_i k_{ji}^{alpha_i} V_j` and
   `t_alpha` are integers and the exactness check is literal BigInt equality
   (no square roots). This preserves the prior experiment's exactness win.

## Task

Associative recall / induction, the same construction as the prior experiment:
`N=64` random unit keys and `N=64` Gaussian values; `M=32` queries; query `i`
is unit key `targets[i]=i % N` plus `0.1` Gaussian noise, renormalized.
Metrics: `recall.value` (nearest value to the attention output),
`recall.index` (argmax weight key), `targetMass` (mass on the target key),
`KL(softmax || method)`, and `costRatio` (method elementary ops / softmax
elementary ops; `exp`,`mul`,`add`,`div` counted). A budget "matches" when
`recall.value >= smRecall - 0.02` and `|targetMass - smMass| <= 0.05`. The
main sweep uses `beta=8` (the sharp regime the prior experiment identified);
results are averaged over 4 seeds (`seed = 1 + 101 s`). Polynomial quality is
taken from the `O(N^2)` weights, which the `O(N m dv)` feature path
reproduces to `<= 1.8e-14`; `costRatio` for the polynomial path is the exact
feature-path op count (`sum_{|alpha|<=p}|alpha| = d C(d+p,p-1)`), verified
against the measured `p<=8` counts in section 1.

## Observed output (exact)

`/opt/homebrew/bin/node run.mjs`:

```text
==========================================================================
softmax vs richer-kernel linear attention (polynomial / random features)
node v25.8.0
==========================================================================
1) COST COUNTER  (M=32 N=64 d=8 dv=8 keyNoise=0.1 beta=8)
           softmax exp=     2048 mul=     32768 add=      34816 div=     2048 total=       71680
   poly-linear p=1 exp=        0 mul=      7968 add=       7776 div=      256 total=       16000 ratio=    0.22 m=     9 m/N=  0.141
   poly-linear p=2 exp=        0 mul=     43680 add=      38880 div=      256 total=       82816 ratio=    1.16 m=    45 m/N=  0.703
   poly-linear p=4 exp=        0 mul=    564960 add=     427680 div=      256 total=      992896 ratio=   13.85 m=   495 m/N=  7.734
   poly-linear p=6 exp=        0 mul=   3939936 add=    2594592 div=      256 total=     6534784 ratio=   91.17 m=  3003 m/N= 46.922
   poly-linear p=8 exp=        0 mul=  19081920 add=   11119680 div=      256 total=    30201856 ratio=  421.34 m= 12870 m/N=201.094
    poly-naive p=4 exp=        0 mul=     40960 add=      34816 div=     2048 total=       77824 ratio=    1.09 m=     - m/N=      -
relu-naive p=6 c=0 exp=        0 mul=     47104 add=      36864 div=     2048 total=       86016 ratio=    1.20 m=     - m/N=      -
          prf m=64 exp=     6144 mul=    107264 add=     105216 div=      256 total=      218880 ratio=    3.05 m=    64 m/N=  1.000
         prf m=512 exp=    49152 mul=    852736 add=     836352 div=      256 total=     1738496 ratio=   24.25 m=   512 m/N=  8.000
        prf m=4096 exp=   393216 mul=   6816512 add=    6685440 div=      256 total=    13895424 ratio=  193.85 m=  4096 m/N= 64.000
     relu-rf m=512 exp=        0 mul=    851968 add=     835584 div=      256 total=     1687808 ratio=   23.55 m=   512 m/N=  8.000
poly-linear exp() calls === 0 : true;  prf exp() calls === (N+M)*m : true
softmax score/weight matrix = 16384 bytes; poly-linear p=8 stats (m x dv, m=12870) = 823680 bytes; ratio = 0.020x (poly stores MORE than softmax)
NOTE: poly-linear is O(N*m*dv) with m = C(d+p,p); m exceeds N as soon as p > log N.
--------------------------------------------------------------------------
2) BIGINT BIT-EXACTNESS  (integer polynomial rule, randomized trials)
trials=400 rows=806 field-comparisons=3255 mismatches=0
RELU^p naive vs polynomial statistic: comparisons=806 mismatches=780  (kink has no finite monomial feature map)
example: {"N":6,"d":6,"dv":3,"p":3,"shift":"0","denN":"778547","denStats":"778547","numN0":"-1962765","numStats0":"-1962765"}
--------------------------------------------------------------------------
3) FEATURE-BUDGET SWEEP AT MATCHED FLOPs  (beta=8; M=32 N=64 d=8 dv=8 keyNoise=0.1; 4 seeds)
   match = recall.value within 0.02 of softmax AND |targetMass-smMass| <= 0.05
softmax reference: recall.value=0.8516 recall.index=1.0000 targetMass=0.6421

POLYNOMIAL KERNEL  w_j = (1 + s_j)^p   (exact monomial feature map, m=C(d+p,p), exp-free)
   quality from the O(N^2) weights; the O(N*m*dv) feature path reproduces them to <=1.8e-14.
   costRatio is the exact feature-path op count (analytic; verified against p<=8 in section 1).
   p       m      m/N  costRatio  recallVal  targetMass   KLsm|k    linErr  match
   1       9      0.1        0.2     0.0234      0.0303   2.1249   2.5e-16     no
   2      45      0.7        1.1     0.0234      0.0520   1.6578   4.4e-16     no
   3     165      2.6        4.3     0.0234      0.0817   1.2811   8.3e-16     no
   4     495      7.7       13.3     0.0234      0.1195   0.9754   1.4e-15     no
   5    1287     20.1       35.2     0.0234      0.1651   0.7276   2.2e-15     no
   6    3003     46.9       83.5     0.0469      0.2172   0.5283   6.8e-15     no
   7    6435    100.5      181.6     0.1016      0.2740   0.3704   7.8e-15     no
   8   12870    201.1      367.7     0.2266      0.3334   0.2479   1.8e-14     no
   9   24310    379.8      702.2     0.3750      0.3931   0.1557         -     no
  10   43758    683.7     1276.3     0.4375      0.4514   0.0893         -     no
  11   75582   1181.0     2223.4     0.5234      0.5066   0.0450         -     no
  12  125970   1968.3     3734.1     0.6641      0.5578   0.0192         -     no
  13  203490   3179.5     6073.6     0.7422      0.6046   0.0088         -     no
  14  319770   4996.4     9603.5     0.8438      0.6468   0.0114         -    YES
poly minimal matching budget: p=14 m=319770 (m/N=4996.4) costRatio=9603.5

POSITIVE RANDOM FEATURES (Performer/FAVOR+)  phi(x)_r = exp(omega_r.(sqrt(beta)x) - beta||x||^2/2)/sqrt(m)
      m     m/N  costRatio  expCount  recallVal  targetMass   KLsm|k  match
     16   0.250       0.78      1536     0.0547      0.0753   2.2039     no
     64   1.000       3.05      6144     0.0703      0.1240   1.6918     no
    256   4.000      12.14     24576     0.1328      0.2043   1.0316     no
   1024  16.000      48.48     98304     0.2188      0.2570   0.7707     no
   4096  64.000     193.85    393216     0.2188      0.2951   0.5996     no
prf minimal matching budget: none up to m=4096 (m/N=64.0)

ReLU RANDOM FEATURES (arc-cosine kernel; exp-free but homogeneous -> no temperature knob)
      m     m/N  costRatio  recallVal  targetMass   KLsm|k  match
     16   0.250       0.74     0.0234      0.0407      Inf     no
     64   1.000       2.95     0.0234      0.0422   1.8514     no
    256   4.000      11.78     0.0234      0.0434   1.8191     no
   1024  16.000      47.09     0.0234      0.0434   1.8161     no
   4096  64.000     188.35     0.0234      0.0435   1.8155     no
relu-rf minimal matching budget: none up to m=4096

ReLU^p NAIVE REFERENCE  w_j = max(c + s_j, 0)^p   (O(N^2), no finite exact feature map)
    c   p  costRatio  clampFrac  recallVal  targetMass   KLsm|k  match
    0   2       1.09     0.4937     0.0469      0.2010      Inf     no
    0   4       1.14     0.4937     0.4453      0.4508      Inf     no
    0   6       1.20     0.4937     0.8203      0.6536      Inf     no
    0   8       1.26     0.4937     0.9766      0.7806      Inf     no
    1   2       1.09     0.0000     0.0234      0.0520   1.6578     no
    1   4       1.14     0.0000     0.0234      0.1195   0.9754     no
    1   6       1.20     0.0000     0.0469      0.2172   0.5283     no
    1   8       1.26     0.0000     0.2266      0.3334   0.2479     no
   c=1 with unit keys never clamps (|s|<=1), so ReLU^p is EXACTLY the polynomial kernel
   (matching p only differs when the kink fires, i.e. c < max|s|).
poly linear-vs-naive float agreement across p=1..8: max linErr = 1.8e-14
--------------------------------------------------------------------------
4) m-vs-N CROSSOVER  (softmax recall tuned to ~0.85; M=32 d=8 dv=8; 3 seeds)
   question: as N grows, does the matching feature budget m stay << N?
    N  beta*   smVal   smMass  poly_p    poly_m      pm/N      pCost   prf_m   rm/N  prfCost  relu_m
   16    6.0  0.9583   0.7810      10     43758   2734.88     2591.6   >2048      -        -   >2048
   32    6.0  0.8750   0.6426      10     43758   1367.44     1714.7   >2048      -        -   >2048
   64    8.0  0.8542   0.6459      14    319770   4996.41     9603.5   >2048      -        -   >2048
  128   12.0  0.8646   0.7194      21   4292145  33532.38   110144.8   >2048      -        -   >2048
   poly m = C(8+p,p) is independent of N, but the *required* m at matched softmax
   quality is thousands of times N at every N tested; it never approaches m << N.
   random-feature budgets are capped at m=2048 (16.0x N at N=128).
--------------------------------------------------------------------------
5) RANDOM-FEATURE VARIANCE  (why m must grow exponentially in beta)
   single positive feature X has E[X]=exp(beta*s), E[X^2]=exp(2beta+4beta*s), so
   rel.std of the m-average ~ sqrt((exp(2beta)-1)/m)  =>  m >= (e^{2beta}-1)/eps^2.

fixed m=1024, beta sweep (N=64):
  beta   smMass   prfMass  relMassErr       TV  KLsm|prf m_pred(e=.1)
   0.5   0.0247    0.0248     6.50e-3 1.327e-2    0.0006     1.718e+2
   1.0   0.0378    0.0377     1.39e-2 2.967e-2    0.0030     6.389e+2
   2.0   0.0798    0.0741     7.16e-2 8.710e-2    0.0264     5.360e+3
   4.0   0.2392    0.1537     3.57e-1 2.464e-1    0.1946     2.980e+5
   6.0   0.4560    0.2090     5.41e-1 3.921e-1    0.4850     1.628e+7
   8.0   0.6419    0.2444     6.19e-1 4.998e-1    0.7898     8.886e+8

fixed beta=8, m sweep (N=64): below m ~ e^{2beta} the estimator is saturated, so TV is nearly flat
      m         TV  relMassErr  KLsm|prf
     16   7.336e-1     8.59e-1    2.1037
     64   6.566e-1     7.89e-1    1.5168
    256   5.958e-1     7.24e-1    1.1596
   1024   5.035e-1     6.16e-1    0.7793
   4096   4.506e-1     5.47e-1    0.6547
log-log slope d log(TV) / d log(m) = -0.089;  at beta=8 even m=4096 is
far below the ~ 8.9e+8 needed for 10% error, so TV is saturated (not the -0.5 regime).
--------------------------------------------------------------------------
6) VERDICT
   - Concentration IS recoverable by a richer kernel. The polynomial target mass
     rises monotonically with p (sweep above), so the affine O(1/N) ceiling is gone.
   - But the matching feature budget is not small. Polynomial needs p large enough
     that m=C(d+p,p) far exceeds N; positive random features need
     m >= (e^{2beta}-1)/eps^2, which at beta=8 and eps=0.1 is m ~ 8.9e+8.
   - ReLU^p at c=0 does concentrate, but only as an O(N^2), non-factorizing reference:
     the kink has no finite monomial feature map, so it buys no O(N) path.
   - PARTIAL: richness recovers concentration; it does NOT recover it at m << N.
--------------------------------------------------------------------------
done.
```

The BigInt exactness check is section 2 of the same run (`trials=400`, `mismatches=0`); there is no separate entry file.

## Findings

### 1. Cost: exp()-free is real, but the feature budget dominates

At `M=32, N=64, d=8, dv=8` softmax makes `2048 = M N` `exp()` calls and
`71680` counted elementary ops. The polynomial feature path makes **zero**
`exp()` calls and is *cheaper* than softmax for `p<=2` (`m=9`, ratio `0.22`;
`m=45`, ratio `1.16`), but by the budget where it concentrates (`p=14`,
`m=319770`) it is `9603.5x` more expensive and its `m x dv` sufficient
statistic is 20.5 MB against softmax's 16 KB score matrix. Positive random
features pay `(N+M) m` exps: only `6144` at `m=64`, but `393216` at `m=4096`
(192x softmax's exp count) and still not matching.

### 2. Polynomial: the ceiling is gone, the budget is not small

The target mass rises monotonically with `p` -- `0.0303, 0.0520, 0.0817,
0.1195, 0.1651, 0.2172, 0.2740, 0.3334, 0.3931, 0.4514, 0.5066, 0.5578, 0.6046,
0.6468` for `p=1..14` -- so the affine `O(1/N)` ceiling is removed. But
`KL(softmax||poly)` only falls below `0.02` at `p=12`, and the *matching*
budget is `p=14` with `m=C(22,14)=319770` and `m/N=4996.4`. At the budget
where the linear-attention win would matter (`m < N`, i.e. `p <= 2`) the
target mass is at most `0.052` and recall is at chance. The feature dimension
grows super-exponentially in the required `p`: `m = C(d+p,p)`.

### 3. Positive random features: exponential-in-beta budget

The variance analysis (`sqrt((e^{2 beta}-1)/m)`) is confirmed empirically. At
fixed `m=1024`, the relative target-mass error is `6.5e-3` at `beta=0.5`,
`1.4e-2` at `beta=1`, `7.2e-2` at `beta=2`, then `0.357`, `0.541`,
`0.619` at `beta=4,6,8`, tracking the predicted `m_pred=(e^{2 beta}-1)/0.01`
column. At `beta=8` the prediction is `~8.9e8` features for 10% error; the
`m` sweep confirms saturation below that (`TV` falls only from `7.34e-1` at
`m=16` to `4.51e-1` at `m=4096`, log-log slope `-0.089`). No practical
budget reaches the matching regime, and the actual `exp()` count grows with
`m`, so this is not the exp()-free O(N) path either.

### 4. ReLU families: one matches but is quadratic, one is bounded

`ReLU^p` with `c=0` matches softmax by `p=6` (recall `0.8203` vs `0.8516`,
target mass `0.6536` vs `0.6421`, clamp fraction `0.4937`) at
`costRatio=1.20` -- but it is `O(N^2)` and non-factorizing (the kink), so it
is an exp()-free *quadratic* attention, not a linear one. The homogeneous ReLU
random features never match (`recall.value` stuck at `0.0234`, target mass
`~0.043`) because the induced arc-cosine kernel is bounded and has no
temperature knob. With `c=1` the clamp never fires and `ReLU^p` is exactly the
polynomial kernel, so it inherits the polynomial budget.

### 5. Exactness survives

`trials=400, rows=806, field-comparisons=3255, mismatches=0`: the naive
`O(N^2)` integer polynomial sums and the monomial sufficient-statistic sums
agree as literal BigInt integers. The float feature path agrees with the naive
`O(N^2)` polynomial attention to `<= 1.8e-14`. The ReLU^p control mismatches
in `780/806` cases, confirming the kink does not factorize.

### 6. m-vs-N crossover

The crossover table tunes `beta` per N so softmax recall is near `0.85`, then
finds the minimal matching budget. The required polynomial budget is
`m/N = 2734.9, 1367.4, 4996.4, 33532.4` for `N = 16, 32, 64, 128` (the
non-monotonicity comes from matching softer softmax at N=32, where the tuned
reference is the weakest); the required `m` is thousands of times N at every N
and never approaches `m << N`. The random-feature families match nowhere up to
the `m=2048` cap (`16x` N at N=128).

## Conclusion

A richer kernel does remove the affine `O(1/N)` concentration ceiling, so the
prior result's diagnosis was right: the failure was expressiveness, not cost or
exactness. But the *linear-attention* promise is not recovered. Matching sharp
softmax requires a feature budget that grows with the sharpness -- super-
exponentially for the exact polynomial map, exponentially in `beta` for
positive random features -- so `m` must be many multiples of N, and the method
becomes more expensive than the `O(N^2)` softmax it was meant to replace. A
rectified quadratic kernel does match, exp()-free, but only at `O(N^2)`.
**There is no measured `m << N` at which a linear (feature-space) kernel
matches softmax in the sharp regime.**

## Reproduce

```sh
cd open-source/maybe/experiments/softmax-vs-kernel-attention
/opt/homebrew/bin/node run.mjs
```

No `.lean` file is touched; nothing is committed.
