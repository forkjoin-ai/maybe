# Does a rejection count (popcount / Hamming) replace the dot-product matmul at usable quality?

An executable experiment for the `@a0n/maybe` package, the third in the attention
series after `../softmax-vs-buleyean-attention/` and `../softmax-vs-kernel-attention/`.
It is dependency-free (imports only its own local module; no Node builtins, no
`node:path`/`node:url`), touches no `.lean` file, lives entirely under
`open-source/maybe/experiments/buleyean-hamming-attention/`, and is never committed.

## Verdict: **PARTIAL**, and routed entirely by the normalization

The prior two experiments falsified the affine Buleyean rule (`w = R + s + 1`:
exp-free and O(N) but with a hard `O(1/N)` concentration ceiling) and then showed
that a richer *feature* kernel only removes the ceiling at a feature budget
`m >> N`. This experiment tests the one remaining matmul-avoidance route: replace
the dot-product **score** with a **rejection count** (popcount of an XOR).

**The score itself is never the problem.** For sign vectors `z in {-1,+1}^d` the
rejection is `v = (d - z_q.z_k)/2`, so the Hamming score `d - v = (d + z_q.z_k)/2`
is an **affine** function of the signed dot product; popcount removes `QK^T` while
preserving the score up to a temperature. (The commonly quoted literal form
`popcount(xor) = d - q.k` for `{0,1}` vectors is **false** -- the agreement count is
`d - |q| - |k| + 2 q.k`; the program counts the counterexamples, `1973/2000`.) So the
outcome is decided by the **normalization**, and the two normalizations diverge:

- **(a) Buleyean / affine normalization** `w_ij = R - min(v_ij,R) + 1`. Being affine
  in the rejection count, it inherits the affine lane's ceiling: the best target mass
  over any budget `R` is `(delta+1)/(delta+N)`, with `delta` the Hamming margin. At
  the sibling task (`d=8, N=64`) the margin is under one bit and the ceiling is
  `0.044`; kernel-only recall is at chance (`0.023` vs softmax `0.852`). **NO** -- it
  fails to retrieve at any usable budget, confirmed for 1/2/4-bit and ternary.
- **(b) Softmax normalization over the exact integer Hamming scores.** `s = dEff - popcount(xor)`
  is an integer with only `dEff+1` values, so `exp(beta*s)` is a **precomputed table**
  of `dEff+1` entries: the transcendental becomes one memory read per pair and the
  score matmul becomes popcount. The normalization is the same exponential family,
  so concentration is retained. **YES at matched quality** -- first match at
  `dEff ~ 32` packed bits; the total primitive-op ratio against float softmax starts at
  `1.5x` (4-bit, `d=8`) and grows with the ambient dimension toward `W`
  (binary: `14.0x` at `d=256`, `24.0x` at `d=1024`, `27.4x` at `d=2048` for W=32;
  `17.7x / 38.3x / 47.9x` for W=64). The full method (integer weighted V) matches
  from `d=8` at 4-bit and from `d=32` at 2-bit; **binary values never match**
  (1-bit value quantisation caps recall near `0.75`).

### Crossover, stated plainly

| normalization | first kernel match | first full-method match | ratio at first match (W=32) | binary at d=256 / d=1024 (W=32 / W=64) |
| --- | --- | --- | ---: | ---: |
| affine `R-min(v,R)+1` | `dEff=512` (d=128, 4-bit; also d=512, binary) | `dEff=1024` (4-bit) | `5.5x` / `20.9x` | kernel-only stays below softmax until a much larger dEff |
| softmax over integer scores | `dEff=32` (d=8, 4-bit) / `dEff=32` (d=32, binary) | `dEff=32` (4-bit) / `dEff=64` (2-bit) | `1.5x` / `3.5x` | `14.0x / 17.7x` and `24.0x / 38.3x` |

Reading: the affine rule needs roughly **16x more packed bits** for quality parity
and still never matches the float reference with 1/2-bit values. The softmax-table
rule reaches parity at `dEff ~ 32` bits, but the *total* primitive-op ratio is only
meaningful once the ambient dimension is large enough that the shared `M*N*dv` value
term stops dominating: `>=5x` (W=32) needs roughly `d >= 64` at 4-bit or `d >= 512`
at binary. At equal **bit budget** (a binary key of `32d` bits vs a 32-bit float key
of `d` coordinates) the score win is only `W/32` (`1x` at W=32, `2x` at W=64); the
larger figures compare a `b`-bit representation against a 32-bit float representation.

### The honest boundary

Popcount removes the `QK^T` **score** matmul only. The value aggregation `A.V` is
still an `M x N` by `dv` dense product (the shared `M*N*dv` term in every total
above) and is **not** removed. Removing it needs a linear-attention factorization,
which the sibling kernel experiment showed costs quality (`m >> N` for sharp
softmax). This is a score-matmul-and-transcendental win at matched quality, **not a
matmul-free attention**.

## Files

| file | role |
| --- | --- |
| `hamming-attention.mjs` | Core: RNG, sibling task generator (verbatim), bit packing / popcount, 1/2/4-bit Gray + ternary quantisers, Buleyean Hamming score and weights, the table-softmax normalization, integer value aggregation, metrics, the exact margin/ceiling analysis, cost model, BigInt exactness, table-exp exactness. |
| `run.mjs` | Main runnable entry: score identity + exactness demo, cost model, BigInt/consensus/floor exactness, R x bit-width quality, concentration ceiling, N sweep, the d/bits crossover for **both** normalizations, JS micro-benchmark, verdict. |
| `exactness.mjs` | Standalone exactness entry (BigInt posteriors + fusion + table exp). |
| `README.md` | This file, with the exact observed output. |

## Run

Real Node (the repo `node` is a shim):

```sh
cd open-source/maybe/experiments/buleyean-hamming-attention
/opt/homebrew/bin/node run.mjs        # full report, ~20-30 s
/opt/homebrew/bin/node exactness.mjs  # BigInt + table exactness only
```

Everything is seeded (`mulberry32`) and deterministic except the wall-clock lines
(the `done.` timing and the section-8 JS micro-benchmark), which vary run to run.

## Mathematical setting

```
scores s_ij = q_i . k_j                     (float reference)
codes  1/2/4-bit Gray over [-kmax,kmax], or ternary {-1,0,+1}
binary/bit rejection  v_ij = popcount(xor(code(q_i), code(k_j)))  in [0, d*b]
          score      s_ij = dEff - v_ij                          in [0, dEff]
ternary   A = popcount(Pq&Pk) + popcount(Mq&Mk)   (agreements)
          B = popcount(Pq&Mk) + popcount(Mq&Pk)   (oppositions)
          score s_ij = A - B  (exactly the integer ternary dot product)
          rejection v_ij = d - s_ij                        in [0, 2d]

(a) Buleyean weight   w_ij = R - min(v_ij, R) + 1        (>= 1: the floor)
    P_ij = w_ij / sum_j w_ij
(b) softmax weight    P_ij = exp(beta*s_ij) / sum_j exp(beta*s_ij)
    exp is a table tab[k] = exp(-beta*k), k = rowMax(s) - s_ij in [0, sMax-sMin]

value aggregation (integer, exact): num_ig = sum_j w_ij * Vint_jg, den_i = sum_j w_ij,
out_ig = vscale * num_ig / den_i, with Vint small integers and vscale RMS-matched.
```

For sign vectors `dEff - popcount(xor) = (d + z_q.z_k)/2`; for higher bit-widths the
integer score is a Gray-code bit metric (not the integer dot product), but it still
has only `sMax-sMin+1` values so the table trick applies.

## Task

Associative recall / induction, the **same construction as the two siblings**:
`N=64` random unit keys, `N=64` Gaussian values, `M=32` queries; query `i` is unit key
`targets[i]=i%N` plus `0.1` Gaussian noise, renormalised. The task generator is copied
verbatim and a parity check imports the sibling generator and compares every element
(4 seeds, 0 mismatches). Metrics: `recall.value` (nearest original value),
`targetMass` (mass on the target key), `KL(softmax||method)`,
`ceil`/`exactMax` (the achievable max target mass). The `d` sweep scales `keyNoise`
by `sqrt(8/d)` to hold the total query-noise norm fixed.

## What is implemented (brief items 1-7)

1. **Quantization** to binary (1-bit sign), ternary `{-1,0,+1}`, and 2/4-bit integer
   levels. b-bit levels use a binary-reflected Gray code; the key/query range is
   data-driven (`[-kmax, kmax]`) so the codec does not collapse as `d` grows.
2. **Buleyean Hamming score**, exact integer arithmetic, with the God Formula on the
   rejection count (`w = R - min(v,R) + 1`) and the honest ternary signed variant
   (`A - B` via four AND-popcounts, which is exactly the ternary dot product).
3. **Integer value aggregation** (`sum w_ij * Vint_jg`, small-integer multiplies) with
   an exact integer numerator/denominator, a float reference, and a kernel-only
   control (`hamFloat`: integer weights x real float V) that isolates the score from
   the value quantisation.
4. **Cost model in primitive instructions** (`bit-ops` vs `FMA`), with W=32 and W=64
   columns and a matched-information baseline; the JS micro-benchmark is labelled
   non-representative.
5. **BigInt exactness**: packed popcount == the naive per-coordinate `O(N^2)` count,
   and the integer posteriors agree bit-for-bit (section 3 of the run).
6. **Sweeps**: bit-width (1/2/4 + ternary), budget `R`, `beta`, and `N`; reporting
   `recall.value`, `targetMass`, `KL`, and the exact concentration ceiling.
7. **Buleyean properties**: the weight floor (`w >= 1`, so no key is excluded) and the
   multi-view consensus (fusion is exact and order-independent only as componentwise
   addition of the integer **weight** vectors -- applying the God Formula to summed
   rejection counts is not equal, because `min` is not linear).

## Observed output (exact)

`/opt/homebrew/bin/node run.mjs`:

```text
==========================================================================
Buleyean Hamming / rejection-count attention vs float softmax attention
node v25.8.0
==========================================================================
task parity vs ../softmax-vs-buleyean-attention/attention.mjs: checked=4 seeds, element mismatches=0

--------------------------------------------------------------------------
1) SCORE EXACTNESS DEMO  (popcount == explicit; ternary masks == signed dot)
--------------------------------------------------------------------------
binary  : popcount(xor) == explicit Hamming distance over 2000 random pairs; mismatches=0
ternary : A - B (four AND-popcounts) == exact ternary dot over 2000 random pairs; mismatches=0
ternary example (d=4): dot=2  A=2  B=0
IDENTITY (sign vectors): d - popcount(xor) == (d + z_q.z_k)/2 over 2000 random pairs; mismatches=0
  => the Hamming score is an AFFINE function of the signed dot product; softmax over it is
     softmax over the quantised dot product (temperature x2) and loses no concentration.
  [literal {0,1} claim "popcount(xor) = d - q.k" is false: mismatches=1973/2000]

--------------------------------------------------------------------------
2) COST MODEL (primitive instructions, not wall time)  M=32 N=64 d=8 dv=8
--------------------------------------------------------------------------
float softmax : score=16384   FMA  value=16384   FMA  exp=2048    div=2048    total=36864   
Hamming       : score = M*N*ceil(dEff/W) bit-ops; value = M*N*dv integer multiply-adds; exp = 0

   method        dEff  wW32  bitopsW32  totalW32  ratioW32   wW64  bitopsW64  totalW64  ratioW64
binary  b=1   8    1     2048       20480     1.800x    1     2048       20480     1.800x    
bit     b=2   16   1     2048       20480     1.800x    1     2048       20480     1.800x    
bit     b=4   32   1     2048       20480     1.800x    1     2048       20480     1.800x    
ternary       16   1     8192       26624     1.385x    1     8192       26624     1.385x    

at d=8 the score term is 8 of ~18 softmax primitives per pair, so the total Hamming win is small;
the exp() count is removed entirely (softmax makes M*N=2048, Hamming makes 0).

MATCHED-INFORMATION baseline (float softmax on the same dEff dequantised coordinates):
binary  b=1   dEff=8      softmaxOnCodes.total=36864     hamming total W32=20480   (score ratio 8.0x)  W64 score ratio 8.0x
bit     b=2   dEff=16     softmaxOnCodes.total=53248     hamming total W32=20480   (score ratio 16.0x)  W64 score ratio 16.0x
bit     b=4   dEff=32     softmaxOnCodes.total=86016     hamming total W32=20480   (score ratio 32.0x)  W64 score ratio 32.0x
ternary       dEff=16     softmaxOnCodes.total=53248     hamming total W32=26624   (score ratio 4.0x)  W64 score ratio 4.0x

CAVEAT: a b-bit code stores b bits/coordinate vs 32 bits for a float. Matching the exact BIT budget
means dEff(float) = d*b, and then the score win is W/32 (1x at W=32, 2x at W=64); the larger
wins above come from the lower-precision representation, not from popcount as such.

SOFTMAX normalization over the integer Hamming scores (precomputed exp table, one memory read per pair):
  float softmax: total=36864 (exp=2048), score=16384 FMA
   method        dEff  bitopsW32  tableReads  expCalls  value  totalW32  ratioW32   totalW64  ratioW64
binary  b=1   8    2048       2048        9         16384  24576     1.500x    24576     1.500x    
bit     b=2   16   2048       2048        17        16384  24576     1.500x    24576     1.500x    
bit     b=4   32   2048       2048        33        16384  24576     1.500x    24576     1.500x    
ternary       16   8192       2048        17        16384  30720     1.200x    30720     1.200x    
  expCalls is a CONSTANT (sMax-sMin+1), independent of M and N; the A.V value matmul (M*N*dv) remains.

--------------------------------------------------------------------------
3) BIGINT INTEGER EXACTNESS + CONSENSUS + FLOOR  (randomized)
--------------------------------------------------------------------------
trials=400 rows=1001 comparisons=45036 mismatches=0
  rejection counts + weights + posterior num/den: naive per-coordinate vs packed popcount, literal BigInt equality
ternary naive-dot vs A-B: comparisons=20449 mismatches=0
multi-view fuse (componentwise weight addition) vs naive per-view sum: comparisons=4138 mismatches=0
fusion ORDER independence (shuffled view order): comparisons=8041 mismatches=0
floor: min integer weight over all trials = 1  (w = R - min(v,R) + 1 >= 1, so every key keeps positive mass)
control: sum rejection counts then God Formula vs sum of weights: comparisons=8041 mismatches=8041  (the min-clamp is NOT linear; only the count(weight) vectors fuse exactly)
example={"N":6,"d":1,"dv":1,"bits":4,"R":"5","denNaive":"21","denPopcount":"21","numNaive0":"8","numPopcount0":"8"}
softmax-table exp: table lookup === direct Math.exp, comparisons=22314 mismatches=0; tableExpEvals=13366 vs directExpEvals=22314 (constant table vs per-pair exp)
  example={"beta":4,"range":34,"s":19,"smax":34,"viaTable":8.75651076269652e-27,"direct":8.75651076269652e-27}

--------------------------------------------------------------------------
4) QUALITY SWEEP OVER R AND BIT-WIDTH  (M=32 N=64 d=8 dv=8 keyNoise=0.1; beta=8; 4 seeds)
--------------------------------------------------------------------------
smVal/smMass = float softmax at beta=8 (sibling sharp regime); sq* = softmax on the SAME dequantised codes
hamVal = full method (integer weights x quantised V); hamFloat = kernel only (integer weights x real float V)
match = hamVal within 0.02 of smVal AND |hamMass - smMass| <= 0.05; chance = 1/N = 0.0156

  method      dEff  smVal  smMass  sqVal  hamVal  hamFloat  hamMass  hsVal  hsFloat  hsMass  hsKL  hsBeta  ceil   exactMax
binary b=1 8    0.852  0.642   0.430  0.023   0.023     0.043   0.352  0.508   0.571   40.557   50.0    0.023  0.044    
bit    b=2 16   0.852  0.642   0.461  0.023   0.023     0.050   0.555  0.641   0.703   34.318   38.8    0.029  0.051    
bit    b=4 32   0.852  0.642   0.805  0.023   0.023     0.093   0.938  0.938   0.949   Inf      50.0    0.067  0.094    
ternary    16   0.852  0.642   0.617  0.023   0.023     0.042   0.531  0.570   0.662   27.014   50.0    0.026  0.043    

hsVal/hsFloat/hsMass = SOFTMAX normalization over the same integer Hamming scores (exp table);
sqVal = float softmax on the dequantised codes (same scores, cross-check).

At d=8 the Hamming margin is under one bit of separation on average and the linear God Formula
saturates at (delta+1)/(delta+N) ~ O(1/N); hamFloat (the kernel alone, with real values) is at
chance too, so the failure is the score/kernel, not the value quantisation.

R curve, bit b=4, seed 1:  R : targetMass : hamVal : hamFloat
    0   : 0.0156     : 0.031  : 0.031 
    1   : 0.0156     : 0.031  : 0.031 
    2   : 0.0161     : 0.031  : 0.031 
    4   : 0.0198     : 0.031  : 0.031 
    8   : 0.0613     : 0.031  : 0.031 
    12  : 0.0944     : 0.031  : 0.031 
    16  : 0.0615     : 0.031  : 0.031 
    24  : 0.0296     : 0.031  : 0.031 
    32  : 0.0236     : 0.031  : 0.031 

--------------------------------------------------------------------------
5) CONCENTRATION: THE HAMMING KERNEL HAS ITS OWN SATURATION CEILING  (d=8 N=64; beta=8)
--------------------------------------------------------------------------
per row: v_t = target rejection, v_min = closest non-target rejection, delta = v_min - v_t
best target mass over ANY budget R is the exact max of w_t(R)/sum_j w_j(R); at R=v_min it is (delta+1)/(delta+N)
ceil uses mean delta; exactMax is the mean of the exact per-row maxima (breakpoint scan, no grid).

  method      dEff   v_t    v_min   delta   ceil(mean delta)   exactMaxOverR   softmaxMass   achievedMax
binary b=1 8    0.67   0.93   0.26   0.0233            0.0436          0.6421       0.0428       
bit    b=2 16   1.10   1.82   0.72   0.0287            0.0512          0.6421       0.0499       
bit    b=4 32   4.85   8.34   3.48   0.0667            0.0939          0.6421       0.0928       
ternary    16   5.01   5.57   0.56   0.0256            0.0426          0.6421       0.0418       

The ceiling is a function of the integer MARGIN: target mass ~= (delta+1)/(delta+N).  Since the
margin grows like dEff/2 - O(sqrt(dEff)), concentration needs dEff >> N.  This is the Hamming
analogue of the affine O(1/N) ceiling: it is reached at finite R and no larger R beats it.

--------------------------------------------------------------------------
6) N SWEEP  (d=8 dv=8 keyNoise=0.1; 4 seeds; softmax tuned to recall ~0.85; hamming over R)
--------------------------------------------------------------------------
   N   method      dEff  smVal  smMass  hamFloat  hamVal  hamMass  hamMax  ceil   bestR
16   binary b=1 8    0.906  0.649   0.117     0.109   0.160    0.160   0.116  3.5    
16   bit    b=4 32   0.906  0.649   0.305     0.297   0.291    0.291   0.261  12.0   
32   binary b=1 8    0.898  0.643   0.039     0.031   0.082    0.082   0.052  3.5    
32   bit    b=4 32   0.898  0.643   0.063     0.070   0.175    0.176   0.146  12.3   
64   binary b=1 8    0.852  0.642   0.023     0.023   0.043    0.043   0.023  3.5    
64   bit    b=4 32   0.852  0.642   0.023     0.023   0.093    0.093   0.067  11.8   
128  binary b=1 8    0.328  0.461   0.016     0.016   0.022    0.022   0.010  3.5    
128  bit    b=4 32   0.328  0.461   0.016     0.016   0.049    0.049   0.030  11.8   
256  binary b=1 8    0.047  0.298   0.016     0.016   0.011    0.011   0.004  3.3    
256  bit    b=4 32   0.047  0.298   0.016     0.016   0.024    0.024   0.012  12.0   

chance is 1/N.  At fixed dEff the margin does not grow with N, so the ceiling falls like 1/N.

--------------------------------------------------------------------------
7) AMBIENT-DIMENSION / BIT-WIDTH CROSSOVER AT MATCHED SOFTMAX QUALITY
--------------------------------------------------------------------------
M=32 N=64 dv=8; unit keys, keyNoise = 0.1*sqrt(8/d) so the total query-noise norm is held fixed;
softmax beta tuned to the sibling sharp regime (target mass 0.64, the d=8/beta=8 point); 8 seeds.
hamFloat = kernel only (integer weights x real V); hamVal = full method (x quantised V).
massMatch (one-sided) = hamFloat within 0.02 of smVal AND hamMass >= smMass - 0.05 (overshooting mass is not a failure);
recallMatch = hamFloat alone; chance = 1/N.

    d  bits   dEff   smVal  smMass  sqVal  hamFloat  hamVal  hamMass  hamMax  kernel  recall  full   ceil   exactMax  ratioW32  ratioW64
8    1    8     0.849  0.637   0.469  0.021     0.021   0.044    0.044   no      no      no    0.024  0.044     1.800x    1.800x    
8    2    16    0.849  0.637   0.490  0.021     0.021   0.050    0.050   no      no      no    0.029  0.052     1.800x    1.800x    
8    4    32    0.849  0.637   0.813  0.021     0.021   0.091    0.093   no      no      no    0.066  0.094     1.800x    1.800x    
16   1    16    0.943  0.645   0.703  0.031     0.031   0.072    0.072   no      no      no    0.047  0.073     2.600x    2.600x    
16   2    32    0.943  0.645   0.792  0.031     0.031   0.081    0.081   no      no      no    0.057  0.083     2.600x    2.600x    
16   4    64    0.943  0.645   0.911  0.063     0.052   0.168    0.168   no      no      no    0.149  0.170     2.364x    2.600x    
32   1    32    0.818  0.575   0.724  0.042     0.042   0.128    0.129   no      no      no    0.107  0.130     4.200x    4.200x    
32   2    64    0.818  0.575   0.760  0.042     0.031   0.140    0.140   no      no      no    0.122  0.143     3.818x    4.200x    
32   4    128   0.818  0.575   0.792  0.115     0.115   0.293    0.293   no      no      no    0.286  0.297     3.231x    3.818x    
64   1    64    0.917  0.617   0.865  0.104     0.115   0.235    0.235   no      no      no    0.225  0.237     6.727x    7.400x    
64   2    128   0.917  0.617   0.891  0.104     0.104   0.240    0.244   no      no      no    0.236  0.248     5.692x    6.727x    
64   4    256   0.917  0.617   0.865  0.411     0.422   0.459    0.459   no      no      no    0.464  0.466     4.353x    5.692x    
128  1    128   0.932  0.637   0.896  0.234     0.182   0.385    0.390   no      no      no    0.389  0.394     10.615x   12.545x   
128  2    256   0.932  0.637   0.891  0.240     0.156   0.396    0.396   no      no      no    0.399  0.403     8.118x    10.615x   
128  4    512   0.932  0.637   0.880  0.932     0.906   0.633    0.635   YES     YES     no    0.646  0.646     5.520x    8.118x    
256  1    256   0.969  0.650   0.969  0.854     0.432   0.566    0.574   no      no      no    0.581  0.581     15.647x   20.462x   
256  2    512   0.969  0.650   0.974  0.859     0.422   0.574    0.574   no      no      no    0.588  0.588     10.640x   15.647x   
256  4    1024  0.969  0.650   0.958  1.000     1.000   0.782    0.782   YES     YES     YES   0.793  0.793     6.488x    10.640x   
512  1    512   0.958  0.655   0.974  1.000     0.526   0.734    0.736   YES     YES     no    0.746  0.746     20.880x   30.706x   
512  2    1024  0.958  0.655   0.979  1.000     0.656   0.741    0.741   YES     YES     no    0.749  0.749     12.732x   20.880x   
512  4    2048  0.958  0.655   0.922  1.000     1.000   0.880    0.880   YES     YES     YES   0.886  0.886     7.151x    12.732x   
1024 1    1024  0.990  0.659   0.995  1.000     0.641   0.855    0.855   YES     YES     no    0.859  0.859     25.220x   41.360x   
1024 2    2048  0.990  0.659   0.995  1.000     0.849   0.853    0.853   YES     YES     no    0.861  0.861     14.164x   25.220x   
1024 4    4096  0.990  0.659   0.974  1.000     1.000   0.936    0.936   YES     YES     YES   0.940  0.940     7.547x    14.164x   
2048 1    2048  0.984  0.660   0.990  1.000     0.729   0.920    0.922   YES     YES     no    0.926  0.926     28.192x   50.195x   
2048 2    4096  0.984  0.660   0.990  1.000     0.802   0.921    0.921   YES     YES     no    0.927  0.927     15.022x   28.192x   
2048 4    8192  0.984  0.660   0.927  1.000     1.000   0.968    0.968   YES     YES     YES   0.969  0.969     7.766x    15.022x   

7b) SOFTMAX normalization over the exact integer Hamming scores (exp = precomputed table lookup)
     expCalls = sMax-sMin+1 per BATCH (table built once, reused across all M rows); hsFloat = kernel x real V, hsVal = kernel x quantised V.
    d  bits   dEff   smVal  smMass  hsFloat  hsVal  hsMass  hsBeta  kernel  full  expCalls  hsRatioW32  hsRatioW64
8    1    8     0.849  0.637   0.531     0.370  0.596   50.0    no      no    9         1.500x      1.500x      
8    2    16    0.849  0.637   0.646     0.568  0.717   42.5    no      no    17        1.500x      1.500x      
8    4    32    0.849  0.637   0.932     0.927  0.940   45.0    YES     YES   33        1.500x      1.500x      
16   1    16    0.943  0.645   0.865     0.641  0.890   42.0    no      no    17        2.167x      2.167x      
16   2    32    0.943  0.645   0.922     0.781  0.940   45.0    no      no    33        2.167x      2.167x      
16   4    64    0.943  0.645   1.000     1.000  1.000   10.0    YES     YES   65        2.000x      2.167x      
32   1    32    0.818  0.575   1.000     0.672  1.000   28.3    YES     no    33        3.500x      3.500x      
32   2    64    0.818  0.575   1.000     0.833  1.000   16.7    YES     YES   65        3.231x      3.500x      
32   4    128   0.818  0.575   1.000     1.000  1.000   3.0     YES     YES   129       2.800x      3.231x      
64   1    64    0.917  0.617   1.000     0.630  1.000   5.0     YES     no    65        5.692x      6.167x      
64   2    128   0.917  0.617   1.000     0.818  1.000   5.0     YES     no    129       4.933x      5.692x      
64   4    256   0.917  0.617   1.000     1.000  1.000   1.0     YES     YES   257       3.895x      4.933x      
128  1    128   0.932  0.637   1.000     0.667  1.000   2.0     YES     no    129       9.200x      10.615x     
128  2    256   0.932  0.637   1.000     0.839  1.000   1.8     YES     no    257       7.263x      9.200x      
128  4    512   0.932  0.637   1.000     1.000  1.000   1.0     YES     YES   513       5.111x      7.263x      
256  1    256   0.969  0.650   1.000     0.734  1.000   1.0     YES     no    257       14.000x     17.733x     
256  2    512   0.969  0.650   1.000     0.802  1.000   1.0     YES     no    513       9.852x      14.000x     
256  4    1024  0.969  0.650   1.000     1.000  1.000   1.0     YES     YES   1025      6.186x      9.852x      
512  1    512   0.958  0.655   1.000     0.677  1.000   1.0     YES     no    513       19.333x     27.474x     
512  2    1024  0.958  0.655   1.000     0.854  1.000   1.0     YES     no    1025      12.140x     19.333x     
512  4    2048  0.958  0.655   1.000     1.000  1.000   1.0     YES     YES   2049      6.960x      12.140x     
1024 1    1024  0.990  0.659   1.000     0.714  1.000   1.0     YES     no    1025      24.047x     38.296x     
1024 2    2048  0.990  0.659   1.000     0.880  1.000   1.0     YES     no    2049      13.787x     24.047x     
1024 4    4096  0.990  0.659   1.000     1.000  1.000   1.0     YES     YES   4097      7.439x      13.787x     
2048 1    2048  0.984  0.660   1.000     0.750  1.000   1.0     YES     no    2049      27.440x     47.860x     
2048 2    4096  0.984  0.660   1.000     0.818  1.000   1.0     YES     no    4097      14.806x     27.440x     
2048 4    8192  0.984  0.660   1.000     1.000  1.000   1.0     YES     YES   8193      7.708x      14.806x     

FIRST KERNEL (mass+recall) MATCH: d=128 bits=4 dEff=512  floatSoftmaxRatio W32=5.520x W64=8.118x  matchedBitsScoreRatio W32=32.00x W64=64.00x
  bits=1: first match d=512 dEff=512 ratioW32=20.880x ratioW64=30.706x
  bits=2: first match d=512 dEff=1024 ratioW32=12.732x ratioW64=20.880x
  bits=4: first match d=128 dEff=512 ratioW32=5.520x ratioW64=8.118x
FIRST RECALL-ONLY MATCH (kernel): d=128 bits=4 dEff=512  floatSoftmaxRatio W32=5.520x W64=8.118x
FIRST FULL-METHOD MATCH (quantised V): d=256 bits=4 dEff=1024 ratioW32=6.488x
  full method bits=1: never matches up to d=2048 (weaker bits lose too much value precision / top-weight sharpness)
  full method bits=2: never matches up to d=2048 (weaker bits lose too much value precision / top-weight sharpness)
  full method bits=4: first match d=256 dEff=1024 ratioW32=6.488x ratioW64=10.640x
FIRST SOFTMAX-NORMALIZED HAMMING MATCH: d=8 bits=4 dEff=32  floatSoftmaxRatio W32=1.500x W64=1.500x  expCalls/batch=33
  bits=1: first softmax-normalized match d=32 dEff=32 ratioW32=3.500x ratioW64=3.500x
  bits=2: first softmax-normalized match d=32 dEff=64 ratioW32=3.231x ratioW64=3.500x
  bits=4: first softmax-normalized match d=8 dEff=32 ratioW32=1.500x ratioW64=1.500x
  softmax-normalized FULL method (quantised V) bits=1: never matches up to d=2048 (binary value quantisation caps recall at ~0.75)
  softmax-normalized FULL method (quantised V) bits=2: first match d=32 dEff=64 ratioW32=3.231x ratioW64=3.500x
  softmax-normalized FULL method (quantised V) bits=4: first match d=8 dEff=32 ratioW32=1.500x ratioW64=1.500x

MATCHED-QUALITY COST BREAKDOWN (first softmax-normalized match per bit-width, W=32):
  bits   d   dEff  | float softmax (scoreFMA/exp/valueFMA/total)  | affine Hamming (bitops/value/total, quality FAILS)  | softmax-table Hamming (bitops/reads/expCalls/value/total, ratio)
  b=1  d=32   dEff=32    | 65536  /2048 /16384 /86016  | 2048  /16384 /20480  (4.20x) | 2048  /2048 /33   /16384 /24576  (3.50x)
  b=2  d=32   dEff=64    | 65536  /2048 /16384 /86016  | 4096  /16384 /22528  (3.82x) | 4096  /2048 /65   /16384 /26624  (3.23x)
  b=4  d=8    dEff=32    | 16384  /2048 /16384 /36864  | 2048  /16384 /20480  (1.80x) | 2048  /2048 /33   /16384 /24576  (1.50x)

Matched-INFORMATION view: the float baseline on the SAME dEff dequantised coordinates costs
M*N*dEff score FMAs; hamming costs M*N*ceil(dEff/W) bit-ops, so the score term is ~W x cheaper
(W=32 -> ~32x, W=64 -> ~64x); the shared M*N*dv value term dilutes the total.

--------------------------------------------------------------------------
8) JS MICRO-BENCHMARK (NON-REPRESENTATIVE OF NATIVE HARDWARE)
--------------------------------------------------------------------------
d=64: float dot ~236.7 ns/pair; JS popcount(xor) ~40.0 ns/pair; ratio 5.92x
acc=9602193.156 (sink)
V8 has no native popcount, the loop carries call/bounds overhead, and the ratio is unstable
across runs (observed 4.20x and 0.20x for the same code) -- it is NOT the native bit-op/FMA
ratio.  The section-2 primitive count is the representative model; this block only shows that
JS cannot express the primitive.

--------------------------------------------------------------------------

--------------------------------------------------------------------------
9) VERDICT
--------------------------------------------------------------------------
  IDENTITY.  For sign vectors z in {-1,+1}^d the Hamming rejection is v = (d - z_q.z_k)/2, so the
  Hamming score d - v = (d + z_q.z_k)/2 is an AFFINE function of the signed dot product (section 1).
  popcount therefore removes the QK^T score matmul while preserving the score up to an affine
  (temperature) transform; concentration is decided by the NORMALIZATION, not by the score.
  [The literal {0,1} form popcount(xor) = d - q.k is false; it is counted in section 1.]

  (a) BULEYEAN / AFFINE normalization  w = R - min(v,R) + 1.  It is affine in the rejection
      count, so it inherits the affine lane ceiling: the best target mass over any budget R is
      (delta+1)/(delta+N) with delta the Hamming margin.  At d=8 the margin is under one bit and
      the kernel-only recall stays at chance (0.023 vs softmax 0.852); the ceiling is 0.044.
      CONFIRMED numerically at every tested bit-width (section 4-5).  This normalization FAILS
      to retrieve at any usable budget.

  (b) SOFTMAX normalization over the exact integer Hamming scores.  s = dEff - popcount(xor) is an
      integer with only dEff+1 values, so exp(beta*s) is a PRECOMPUTED TABLE of dEff+1 entries --
      the transcendental becomes one memory read per pair and the score matmul becomes popcount.
      Because the normalization is the same exponential family, concentration is retained, and
      this variant DOES match float softmax at matched quality:
        - 4-bit: first match at dEff=32 bits (d=8) with a 1.5x W32 total ratio; the full method
          (quantised V) also matches there (hsVal 0.927 vs smVal 0.849).
        - binary: first kernel match at dEff=32 bits (d=32) with 3.5x; the FULL binary method
          never matches up to d=2048 because 1-bit value quantisation caps recall near 0.75.
        - the ratio grows with the ambient dimension (value-term dilution shrinks): binary at
          d=256 is 14.0x W32 / 17.7x W64, at d=1024 is 24.0x / 38.3x, approaching W=32 / 2W=64.

  HONEST BOUNDARY.  Popcount removes the QK^T matmul only.  The value aggregation A.V is still an
  N x N (here M x N) by d_v dense product and is NOT removed; it is the shared M*N*d_v term in
  every cost total above.  Removing it needs a linear-attention factorization, which the sibling
  kernel experiment showed costs quality (m >> N for sharp softmax).  So this is a score-matmul
  and transcendental win at matched quality, NOT a fully matmul-free attention.

  HEADLINE.  PARTIAL, and routed entirely by the normalization.  Buleyean/affine normalization
  NEVER matches (O(1/N) ceiling).  Softmax normalization over the same integer Hamming scores
  matches at every bit-width, first at dEff~32 bits, with a total ratio that is small there
  (1.5x) and grows toward W as d grows (binary: 14x at d=256, 24x at d=1024 at W=32; roughly
  2x that at W=64).  Required bits: 4-bit keys/values support the full method from d=8; binary
  keys need d>~32 and binary VALUES never match.  Popcount does NOT remove the AV matmul.

--------------------------------------------------------------------------
done.  (21.1 s)
```

`/opt/homebrew/bin/node exactness.mjs`:

```text
trials=400 rows=1001 comparisons=45036 mismatches=0
ternary naive-dot vs A-B: comparisons=20449 mismatches=0
multi-view fuse (weight addition) vs naive per-view: comparisons=4138 mismatches=0
fusion order-independence (shuffled views): comparisons=8041 mismatches=0
control: sum rejections then God Formula vs weight sum: comparisons=8041 mismatches=8041
floor: min integer weight=1
example={"N":6,"d":1,"dv":1,"bits":4,"R":"5","denNaive":"21","denPopcount":"21","numNaive0":"8","numPopcount0":"8"}
softmax-table exp: comparisons=22314 mismatches=0 tableExpEvals=13366 directExpEvals=22314
example={"beta":4,"range":34,"s":19,"smax":34,"viaTable":8.75651076269652e-27,"direct":8.75651076269652e-27}
```

## Findings

### 1. The score is exact, and the identity is affine (not literal)

`popcount(xor)` equals the explicit Hamming distance (`0/2000` mismatches), the ternary
`A-B` equals the integer dot product (`0/2000`), and for sign vectors
`d - popcount(xor) == (d + z_q.z_k)/2` exactly (`0/2000`). The literal `{0,1}` claim
`popcount(xor) = d - q.k` is false (`1973/2000`). So the score can be computed without
a matmul and without changing the ranking.

### 2. Integer exactness, floor and consensus

`trials=400 rows=1001 comparisons=45036 mismatches=0` for the rejection counts, the
integer weights and the posterior numerators/denominators. Ternary masks vs the signed
dot: `20449/0`. Multi-view fusion agrees with the naive per-view sum (`4138/0`) and is
order-independent (`8041/0`). The floor holds (`min weight = 1`). The control --
summing rejection counts and then applying the God Formula -- mismatches in
`8041/8041` cases, which is the precise sense in which only the **count (weight)**
vectors fuse exactly. The precomputed table reproduces `Math.exp` bit-for-bit
(`22314/0`), evaluating `13366` exps total instead of `22314`.

### 3. Cost model (M=32 N=64 d=8 dv=8)

Float softmax total `36864` (score `16384` FMA + `2048` exp + value `16384` FMA +
`2048` div). At `d=8` the Buleyean Hamming total is `20480` (`1.80x` W=32 and W=64 --
the same because `dEff<W`), and the table-softmax total is `24576` (`1.50x`). The
score term shrinks, the `exp` disappears, but the shared `M*N*dv` value term keeps the
`d=8` win small. At equal **bits** the score win is `W/32`, not `W`.

### 4. Quality and the concentration ceiling (the affine normalization fails)

At the sibling task (`d=8`, `beta=8`) float softmax is `smVal=0.852`, `smMass=0.642`.
Buleyean affine: `hamVal=0.023`, `hamFloat=0.023`, achieved max mass `0.043`, exact
ceiling `0.044` -- at chance. The ceiling is the exact max over `R` of
`w_t(R)/sum_j w_j(R)`, which at `R=v_min` is `(delta+1)/(delta+N)`; at `d=8` the
margin `delta` is `0.26` (binary), `0.72` (2-bit), `3.48` (4-bit), `0.56` (ternary).
The `N` sweep shows the ceiling falling like `1/N` at fixed `dEff`. Softmax
normalization over the same scores: `hsVal=0.352/0.555/0.938/0.531` and
`hsMass=0.571/0.703/0.949/0.662` for binary/2-bit/4-bit/ternary at `d=8` -- the 4-bit
row already matches the float reference (`0.938` vs `0.852`).

### 5. Crossover at matched quality (section 7 / 7b of the run)

**Affine**, first kernel match: `dEff=512` (4-bit `d=128`, `5.5x`; binary `d=512`,
`20.9x`), 2-bit `dEff=1024` (`12.7x`); first full method (4-bit) `dEff=1024` (`6.5x`);
1/2-bit full methods never match. **Softmax-table**, first kernel match: 4-bit
`dEff=32` (`1.5x`), binary `dEff=32` (`3.5x`), 2-bit `dEff=64` (`3.2x`); first full
method 4-bit `dEff=32`, 2-bit `dEff=64`; binary **never** (1-bit values cap recall).
The softmax-table ratio grows with the ambient dimension: binary at `d=256` is
`14.0x` W=32 / `17.7x` W=64, at `d=1024` is `24.0x` / `38.3x`, at `d=2048` is
`27.4x` / `47.9x`, approaching `W` / `2W`.

### 6. JS micro-benchmark is not a hardware measurement

The JS ratio for the same code was observed at `4.20x`, `0.20x`, `0.66x` and `4.39x`
across runs: V8 has no native popcount and does not vectorise the loop. The primitive
count in section 2 is the representative model.

## Conclusion

Replacing the dot-product score with a rejection count is exact, dependency-free, and
removes the `QK^T` matmul and the transcendental. Whether it is *usable* is decided by
the normalization, not by the score: the affine Buleyean rule inherits the `O(1/N)`
ceiling and fails; softmax normalization over the same integer scores keeps the
concentration and matches float softmax at `dEff ~ 32` bits, with a total cost ratio
that starts near `1.5x` and approaches `W` (`32x` at W=32, `~48x` at W=64) as the
ambient dimension grows. The value aggregation `A.V` is untouched, so this is not a
fully matmul-free attention; and at equal **bit** budget the score win is only `W/32`.

## Reproduce

```sh
cd open-source/maybe/experiments/buleyean-hamming-attention
/opt/homebrew/bin/node run.mjs
/opt/homebrew/bin/node exactness.mjs
```

No `.lean` file is touched; nothing is committed; the sibling experiment directories
are not modified.
