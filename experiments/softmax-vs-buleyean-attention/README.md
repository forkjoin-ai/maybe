# Softmax vs Buleyean/Laplace affine-or-hinge linear attention

An executable experiment for the `@a0n/maybe` package. It is dependency-free
(no imports at all, not even Node builtins), touches no `.lean` file, and lives
entirely under `open-source/maybe/experiments/softmax-vs-buleyean-attention/`.

## Verdict

**The substitution does not survive.** The brief's hypothesis -- parity at small
logit scale, softmax wins as sharp selection is forced -- is confirmed in
direction, and the reason is stronger than a scaling mismatch:

- **Parity at small sigma is real but vacuous.** For `beta <= 1` the softmax and
  affine weight distributions agree to `KL <= 0.019` and both retrieval scores
  sit at chance (`1/N = 0.0156`). They agree because both are nearly uniform,
  not because the affine rule is a usable substitute.
- **Softmax wins monotonically as sigma grows.** At `beta = 8` softmax top-1
  value recall is `0.852` and its target mass is `0.633`; the void-dual affine
  stays at chance (`0.0195`) and `KL(softmax || affine)` grows to `2.177`
  (`4.284` for the literal reading).
- **The affine rule has a structural O(1/N) concentration ceiling.** A normalized
  linear ramp puts at most `(C + s_max) / (N*C + sum_j s_j)` mass on one key,
  which is `O(1/N)`. Softmax drives that mass to `1`. No temperature rescues it,
  which the concentration-ceiling section demonstrates numerically (`0.0309` at
  `beta = 64` against the analytic ceiling `0.0311`).
- **Being linear is exactly why it fails.** The affine rule *is* bit-exactly
  linearizable (`400` trials, `3933` BigInt field comparisons, `0` mismatches)
  and *is* O(N) with `0` `exp()` calls (`4.72x` fewer counted ops at N=64).
  Linear cost and exact factorization are not the bottleneck; the linear family
  is simply too flat to concentrate.

### Sign note

The brief writes `softmax = exp(s)` with `s = q . k` (monotone increasing) and the
affine rule `w = R - s + 1` (monotone decreasing) on the **same** score. Those two
have opposite order, so the literal affine rule selects the *least* matching key
(index recall `0.0000`). The package's own convention resolves this:
`src/buleyean.ts` compares Buleyean `R - min(v,R) + 1` against softmax
`exp(-eta * v)`, **both antitone in the rejection `v`**. The softmax-aligned
reading therefore sets the rejection `v = -s`, i.e. `w = R + s + 1` in score form.
Both readings are implemented and reported (`voidSign = -1` and `voidSign = +1`);
only `voidSign = -1` is a candidate replacement, and it still fails to concentrate.

## Files

| file | role |
| --- | --- |
| `attention.mjs` | Core: RNG, cost counter, task generator, softmax O(N^2), affine/hinge naive O(N^2), affine sufficient-statistic O(N) path, KL, recall metrics. |
| `exactness.mjs` | Standalone BigInt bit-exactness entry (naive vs prefix-sum affine, plus hinge non-factorization control). |
| `run.mjs` | Main runnable entry: cost counter, O(N) scaling, exactness, sigma sweep, temperature/budget, clamp study, concentration ceiling. |
| `README.md` | This file, including the exact observed output. |

## Run

Real Node (the repo `node` is a shim), per the package environment:

```sh
cd open-source/maybe/experiments/softmax-vs-buleyean-attention
/opt/homebrew/bin/node run.mjs        # full report
/opt/homebrew/bin/node exactness.mjs  # BigInt bit-exactness only
```

## Mathematical setting

```
score             s_ij = q_i . k_j
rejection / void  v_ij = voidSign * s_ij
softmax weight    w_ij ~ exp(beta * s_ij) = exp(-beta * v_ij)
affine weight     w_ij  = R_i - beta * v_ij + 1                 (unclamped Buleyean/Laplace)
hinge weight      w_ij  = 1 + max(R_i - beta * v_ij, 0)         (clamped reference)
budget            R_i   = beta * C,  C = 1 (unit keys bound |score| <= 1)
```

`beta` is the brief's logit scale / inverse temperature `sigma`. With `R = beta*C`
the budget is a **linear** inverse temperature, exactly as the brief describes.

The unclamped affine rule factorizes (`r_j = voidSign * k_j`):

```
sum_j v_ij      = q_i . (sum_j r_j)
sum_j w_ij      = N * (R_i + 1) - beta * q_i . (sum_j r_j)
sum_j w_ij V_jf = (R_i + 1) * (sum_j V_jf) - beta * q_i . (sum_j r_j V_jf)
```

so two sufficient statistics -- `S_k = sum_j r_j` and `S_kv = sum_j r_j V_j^T` --
plus the value sum `S_v` answer every query in `O(d^2)` without materialising N
weights. In the common `M = N` case this is linear in N: `O(N*d^2)` to build the
`d x d` second moment and `O(N*d^2)` to query. (The brief writes `O(N*d)`; the
second-moment statistic is `d x d`, so the honest complexity is `O(N*d^2)`.)

## Task

Associative recall / induction. `N = 64` random unit keys and `N = 64` Gaussian
values. `M = 32` queries; query `i` is unit key `targets[i] = i % N` plus `0.1`
Gaussian noise, renormalized. Metrics:

- `recall.value` (headline top-1): the target value is the nearest value to the
  attention output.
- `recall.index`: the argmax attention weight is the target key. This metric is
  invariant to any strictly monotone reparametrization of the score, so softmax
  and the score-aligned affine rule always share it; it is reported to expose the
  literal (inverted) reading.

## What is implemented

1. Softmax attention `O(M*N*d)` (reference).
2. Affine linear attention `O(N*d^2)` via the sufficient statistics above, plus
   the hinge variant computed naively for reference.
3. A workload counter: `exp`, multiply, add and divide counts, plus analytic
   memory. It confirms the affine path makes **zero** transcendental calls and is
   **O(N)** in sequence length.
4. Integer/BigInt bit-exactness: naive `O(N^2)` sums vs the `O(N)` prefix-sum
   formulas, compared by literal BigInt equality across randomized trials.
5. A logit-scale sweep (`0.05 .. 8`) reporting softmax vs affine recall, KL, and
   whether affine matches at matched cost; plus a temperature/budget study and a
   fixed-budget clamp study, and the concentration-ceiling demonstration.

## Observed output (exact)

`/opt/homebrew/bin/node run.mjs`:

```text
==========================================================================
softmax vs Buleyean/Laplace affine-or-hinge linear attention
node v25.8.0
==========================================================================
1) COST COUNTER  (M=32 N=64 d=8 dv=8 keyNoise=0.1)
         softmax exp=     2048 mul=     32768 add=      34816 div=     2048 total=       71680
   affine-linear exp=        0 mul=      7200 add=       7744 div=      256 total=       15200
    affine-naive exp=        0 mul=     34816 add=      38912 div=     2048 total=       75776
     hinge-naive exp=        0 mul=     34816 add=      38912 div=     2048 total=       75776
ratio softmax/affine-linear total = 4.72x
affine-linear exp() calls === 0 : true
memory softmax score/weight matrix = 16384 bytes; affine stats (d*d + d + dv doubles) = 640 bytes; ratio = 25.6x
NOTE: a streaming softmax needs only O(N) memory but still N exp() calls per query.
--------------------------------------------------------------------------
2) O(N) SCALING  (M=N, d=4, dv=4, beta=1, voidSign=-1)
     N    softmaxOps   affLinOps   affNaiveOps     smOps/N  affLinOps/N  affLinOps/N^2
    16          4864        1584          5376       304.0         99.0          6.188
    32         19456        3168         21504       608.0         99.0          3.094
    64         77824        6336         86016      1216.0         99.0          1.547
   128        311296       12672        344064      2432.0         99.0          0.773
   256       1245184       25344       1376256      4864.0         99.0          0.387
   512       4980736       50688       5505024      9728.0         99.0          0.193
  1024      19922944      101376      22020096     19456.0         99.0          0.097
N grew 64.0x:  softmax ops grew 4096.0x (quadratic);  affine-linear ops grew 64.0x (linear).
affine-linear ops/N is flat (O(N));  softmax ops/N^2 is flat (O(N^2)).
--------------------------------------------------------------------------
3) BIGINT BIT-EXACTNESS  (integer affine rule, randomized trials)
trials=400 rows=1002 field-comparisons=3933 mismatches=0
HINGE vs affine sufficient-statistic formula: comparisons=1002 mismatches=781  (clamp is not linear -> cannot factorise)
example: {"N":6,"d":6,"dv":3,"beta":"0","R":"4","voidSign":"1","denomNaive":"30","denomStats":"30","numNaive0":"35","numStats0":"35"}
--------------------------------------------------------------------------
4) SIGMA SWEEP  (beta = logit scale; M=32 N=64 d=8 dv=8 keyNoise=0.1; averaged over 8 seeds)
   recall.value = top-1 retrieval (nearest value to attention output); recall.index = argmax-weight key

voidSign=-1  (v=-s, softmax-aligned / void-dual affine)
  beta   smVal   smIdx   afVal   afIdx   hiVal   KLsm|af   KLsm|hi  match    linErr
  0.05  0.0156  1.0000  0.0156  1.0000  0.0156    0.0000    0.0000    YES   3.3e-16
  0.10  0.0156  1.0000  0.0156  1.0000  0.0156    0.0000    0.0000    YES   3.9e-16
  0.25  0.0156  1.0000  0.0156  1.0000  0.0156    0.0002    0.0002    YES   4.4e-16
  0.50  0.0156  1.0000  0.0156  1.0000  0.0156    0.0021    0.0021    YES   5.0e-16
  1.00  0.0195  1.0000  0.0156  1.0000  0.0156    0.0188    0.0188    YES   5.6e-16
  2.00  0.0234  1.0000  0.0156  1.0000  0.0156    0.1329    0.1329     no   4.4e-16
  4.00  0.0781  1.0000  0.0195  1.0000  0.0195    0.7018    0.7018     no   3.9e-16
  8.00  0.8516  1.0000  0.0195  1.0000  0.0195    2.1771    2.1771     no   3.9e-16

voidSign=+1  (literal w=R-s+1 exactly as written in the brief)
  beta   smVal   smIdx   afVal   afIdx   hiVal   KLsm|af   KLsm|hi  match    linErr
  0.05  0.0156  1.0000  0.0156  0.0000  0.0156    0.0006    0.0006    YES   4.4e-16
  0.10  0.0156  1.0000  0.0156  0.0000  0.0156    0.0025    0.0025    YES   3.3e-16
  0.25  0.0156  1.0000  0.0156  0.0000  0.0156    0.0139    0.0139    YES   4.4e-16
  0.50  0.0156  1.0000  0.0156  0.0000  0.0156    0.0486    0.0486    YES   4.4e-16
  1.00  0.0195  1.0000  0.0156  0.0000  0.0156    0.1628    0.1628     no   3.9e-16
  2.00  0.0234  1.0000  0.0156  0.0000  0.0156    0.5363    0.5363     no   3.9e-16
  4.00  0.0781  1.0000  0.0156  0.0000  0.0156    1.7237    1.7237     no   4.4e-16
  8.00  0.8516  1.0000  0.0156  0.0000  0.0156    4.2843    4.2843     no   3.3e-16
--------------------------------------------------------------------------
5) TEMPERATURE / LINEAR BUDGET  (R = beta*C; affine slope = inverse temperature beta)
   maxWeightDiff = max_j |softmax_j - affine_j| on one row;  diff/beta^2 ~ const means O(beta^2) error
   beta  R=beta*C  maxWeightDiff  diff/beta^2  KL(sm|af)  affNegFrac
   0.01      0.01       6.674e-6       0.0667   0.000000      0.0000
   0.02      0.02       2.641e-5       0.0660   0.000000      0.0000
   0.05      0.05       1.599e-4       0.0640   0.000001      0.0000
   0.10      0.10       6.081e-4       0.0608   0.000015      0.0000
   0.20      0.20       2.218e-3       0.0554   0.000187      0.0000
   0.40      0.40       7.568e-3       0.0473   0.002003      0.0000
   0.80      0.80       2.358e-2       0.0369   0.016484      0.0000
--------------------------------------------------------------------------
6) FIXED BUDGET R=1 (budgetC=1/beta): clamp activation and validity
   affine is unclamped and may go negative; hinge clamps to 1 and stays a valid distribution
  beta  affNegFrac  hingeClampedFrac   smVal   afVal   hiVal
  0.10      0.0000            0.0000  0.0078  0.0078  0.0078
  0.50      0.0000            0.0000  0.0078  0.0078  0.0078
  1.00      0.0000            0.0000  0.0156  0.0078  0.0078
  2.00      0.0000            0.0840  0.0234  0.0156  0.0156
  4.00      0.0840            0.2560  0.0625  0.0156  0.0156
  8.00      0.2560            0.3660  0.8672  0.0156  0.0156
--------------------------------------------------------------------------
7) CONCENTRATION CEILING (why the affine rule cannot retrieve at ANY beta)
   N=64 C=1 s_target=1.0000 sum_s=0.216 analytic affine ceiling (C+s_t)/(N*C+sum_s) = 0.0311
   note: linear attention is O(N*d^2), not O(N*d): the second-moment statistic S_kv is d x d.
  beta   softmaxTargetFrac   affineTargetFrac   ceiling
  0.10            0.017248           0.017040  0.031145
  0.50            0.025172           0.020810  0.031145
  1.00            0.038853           0.023398  0.031145
  2.00            0.081739           0.025983  0.031145
  4.00            0.235103           0.028049  0.031145
  8.00            0.633495           0.029426  0.031145
 16.00            0.949848           0.030235  0.031145
 64.00            0.999999           0.030907  0.031145
   affine target mass saturates at the O(1/N) ceiling; softmax target mass -> 1. No beta rescues it.
--------------------------------------------------------------------------
done.
```

`/opt/homebrew/bin/node exactness.mjs`:

```text
trials=400 rows=1002 comparisons=3933 mismatches=0
hinge-vs-affine-stats comparisons=1002 mismatches=781
example={"N":6,"d":6,"dv":3,"beta":"0","R":"4","voidSign":"1","denomNaive":"30","denomStats":"30","numNaive0":"35","numStats0":"35"}
```

## Findings

### 1. Cost (`M=32, N=64, d=8, dv=8`)

| path | exp | mul | add | div | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| softmax | 2048 | 32768 | 34816 | 2048 | 71680 |
| affine-linear | 0 | 7200 | 7744 | 256 | 15200 |
| affine-naive | 0 | 34816 | 38912 | 2048 | 75776 |
| hinge-naive | 0 | 34816 | 38912 | 2048 | 75776 |

Softmax makes `2048 = M*N` `exp()` calls; the affine linear path makes `0`.
Total counted ops ratio `71680 / 15200 = 4.72x`. Memory: the materialized softmax
score/weight matrix is `N^2 * 8 = 16384` bytes against `640` bytes of affine
statistics (`25.6x`). A streaming softmax needs only `O(N)` memory but still pays
`N` `exp()` calls per query -- memory can be streamed, the transcendental cannot
be removed from softmax.

### 2. O(N) scaling (`M = N`, `d = 4`, `dv = 4`)

N grows `64x` (`16 -> 1024`): softmax ops grow `4096x` (`= 64^2`, quadratic) while
affine-linear ops grow `64x` (linear). `affine-linear ops / N` is exactly `99.0` at
every N; `softmax ops / N` doubles every time N doubles. The affine path is O(N)
and makes zero `exp()` calls.

### 3. Bit-exactness (BigInt)

`trials=400, rows=1002, field-comparisons=3933, mismatches=0`. The naive `O(N^2)`
affine sum and the `O(N)` prefix-sum agree as literal integers, not merely to
floating tolerance. The same loop compares the **hinge** sum to the affine
sufficient-statistic formula: `781 / 1002` comparisons mismatch, confirming the
clamp is piecewise linear and cannot factorize. In floating point the naive and
linear affine outputs agree to `<= 5.6e-16` (`linErr` column), so the linear path
is numerically the same computation.

### 4. Sigma sweep

`voidSign = -1` (softmax-aligned, `v = -s`):

| beta | sm recall.value | aff recall.value | aff recall.index | KL(sm\|\|af) | match |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.05 | 0.0156 | 0.0156 | 1.0000 | 0.0000 | YES |
| 0.10 | 0.0156 | 0.0156 | 1.0000 | 0.0000 | YES |
| 0.25 | 0.0156 | 0.0156 | 1.0000 | 0.0002 | YES |
| 0.50 | 0.0156 | 0.0156 | 1.0000 | 0.0021 | YES |
| 1.00 | 0.0195 | 0.0156 | 1.0000 | 0.0188 | YES |
| 2.00 | 0.0234 | 0.0156 | 1.0000 | 0.1329 | no |
| 4.00 | 0.0781 | 0.0195 | 1.0000 | 0.7018 | no |
| 8.00 | 0.8516 | 0.0195 | 1.0000 | 2.1771 | no |

`voidSign = +1` (literal `w = R - s + 1` from the brief): recall.value is at chance
throughout, recall.index is `0.0000` for every beta (it selects the least-matching
key), and `KL` reaches `4.2843` at `beta = 8`.

**Matched cost.** Affine matches within tolerance only for `beta <= 1`, where both
methods are at chance -- a vacuous match. Wherever softmax is actually useful
(`beta >= 4`), affine fails despite being `4.72x` cheaper and having zero
transcendentals. The binding constraint is expressiveness, not cost.

`recall.index` is `1.0000` for softmax and for the void-dual affine at every beta,
because both are strictly monotone in the score and argmax is preserved. That is a
trap: index recall cannot distinguish a usable attention distribution from a flat
one. `recall.value` (and KL) can, and they show the collapse.

### 5. Temperature / linear budget

`R = beta*C`, so the budget is linear in inverse temperature. The maximum
per-weight difference `|softmax_j - affine_j|` is `O(beta^2)` at small beta
(`diff/beta^2 ~ 0.066` for `beta <= 0.02`), the signature of a first-order
linearization. The agreement does not persist: by `beta = 0.8` the error is
`2.36e-2` per weight and `KL = 0.0165`, and beyond that softmax's exponential tail
separates completely.

### 6. Fixed budget `R = 1`

When the budget is held at `R = 1` instead of scaling with beta, the unclamped
affine rule acquires negative weights (`25.6%` of weights at `beta = 8`) and is no
longer a distribution; the hinge clamps `36.6%` of weights and stays valid. Neither
concentrates: at `beta = 8` softmax recall.value is `0.867` while both affine and
hinge remain at chance (`0.0156`).

### 7. Concentration ceiling (the structural reason)

For the score-aligned affine rule, `w_j = beta*(C + s_j) + 1`, so the normalized
target mass is

```
m_t(beta) = [beta*(C + s_t) + 1] / [N*(beta*C + 1) + beta*sum_j s_j]
          -> (C + s_t) / (N*C + sum_j s_j)   as beta -> infinity
```

which is `O(1/N)` and independent of beta. With `N = 64`, `C = 1`, `s_t = 1`,
`sum_j s_j = 0.216`, the ceiling is `2 / 64.216 = 0.0311`. Observed: affine target
mass rises from `0.0170` at `beta = 0.1` and saturates at `0.0309` at `beta = 64`,
while softmax target mass rises from `0.0172` to `0.999999`. A linear ramp spreads
mass as `~1/N` no matter how large the inverse temperature; softmax concentrates
exponentially. This is why retrieval stays at chance for every beta.

## Conclusion

The affine sufficient-statistic path is a genuine, bit-exact, `exp()`-free `O(N)`
linear attention. It is *not* a replacement for softmax attention on associative
recall: it matches softmax only in the near-uniform regime where both are useless,
and it has a hard `O(1/N)` concentration ceiling that softmax exceeds at any
practical logit scale. The hinge variant repairs non-negativity but neither
factorizes nor concentrates. If a linear-attention substitute is wanted, the affine
family is the wrong one -- a feature map with unbounded dynamic range (e.g. an
exponential or polynomial feature map) is required, not an affine ramp.

## Reproduce

```sh
cd open-source/maybe/experiments/softmax-vs-buleyean-attention
/opt/homebrew/bin/node run.mjs
/opt/homebrew/bin/node exactness.mjs
```

Everything is seeded (`mulberry32`) and deterministic; no network, no filesystem,
no dependencies.
