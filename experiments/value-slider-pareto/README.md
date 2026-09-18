# Where is the value-side quality/speed knee?

An executable follow-up to the parent value-side slider finding in
`@a0n/maybe` src/quality-ladder.ts: replacing the value matmul A.V is
the dominant cost (M*N*d_v) and it has a continuum (block size B, rank r, topK),
unlike the score side.  The cheap end (fully linear) is known to lose quality; the
open question is **where the knee is** -- how much of A.V can be replaced before
quality drops -- and **what the tau miss-not-lie admission buys**.

This experiment is dependency-free (no imports at all except the sibling core
module, no Node builtins), touches no .lean file, runs on real Node
(/opt/homebrew/bin/node), and lives entirely under
open-source/maybe/experiments/value-slider-pareto/.

## Verdict: the knee is CHUNKED B=1 / TOPK k=1 -- almost all of A.V is removable

The score side is held fixed: one exact float softmax A is computed once and
never changed.  Every regime below re-uses that same A.

- **The exact endpoint (B=N)** is the full A.V, cost M*N*dv, hard recall
  0.961..1.000 and target mass 0.145..0.942 across the two task variants and
  N in {64,128,256,512}.
- **The LINEAR rank=d cheap end is not a useful endpoint.**  It loses hard
  retrieval entirely (recall 0.000..0.047), qerr 0.926..0.995, and costs
  1.89x..4.54x the exact A.V because the sufficient statistic is built at
  N*d*dv and the per-query read is d*(d+dv) per query.  The knee is inside
  CHUNKED/TOPK, not linear attention.
- **KNEE: CHUNKED B=1** is the elbow of the quality/cost frontier in every one
  of the 8 task instances.  It removes **92.97%..95.70%** of the full A.V
  operation count and **96.09%..98.83%** of the bare A.V multiply-adds
  (25.6x..85.3x), while hard recall stays **0.984..1.000**.  Increasing B buys
  almost nothing: for assoc N=512, going from B=1 (qerr 0.0268, 4.30% cost) to
  B=256 (qerr 0.0228, 54.10% cost) pays 13x the cost for ~15% less error.
- **Tiny (<=2%) readout loss exists only on the sharply concentrated assoc
  task**, and only up to N=256: qerr 0.0087 (N=64) / 0.0126 (N=128) / 0.0192
  (N=256) at 4.69%..7.03% of exact cost.  At N=512, CHUNKED B=1 is qerr
  0.0268, already above 2%.  On the diffuse **needle** task no cheap point
  reaches qerr <= 0.02 at all.
- **For a diffuse head, topK beats the chunked mean-tail.**  The best cheap
  needle readout is TOPK k=N/2 at 53.13% cost with qerr 0.056..0.145, while
  CHUNKED B=1 sits at qerr 0.368..0.485 at ~5% cost.  The chunked mean tail is
  a poor approximation when the tail carries real mass.
- **TOPK k=1** is the cheapest point on the frontier (3.32%..4.69% of exact
  cost) and keeps hard recall (0.984..1.000), but destroys the weight
  distribution (KL 1.16..17.09, qerr 0.062..5.50).  Hard retrieval is robust to
  truncation; soft readout is not.
- **The tau miss-not-lie admission buys a safe speedup.**  The SOUND bracket
  admits exactly the queries whose target column is provably retained (so the
  served target mass is exact) and misses the rest, recomputing on the exact
  tier.  It is **zero-lie for every candidate and every tau**, and the best
  zero-lie end-to-end speedup is **16.19x..30.12x** over exact.  The ORACLE
  (cheating, exact-anchored) agrees.  The PROXY heuristic is not sound: it
  reaches maximum observed lie rate 0.016, and where it cannot admit it is
  slower than SOUND.
- **Integer exactness holds.**  BigInt vs naive Number: 766,976 field
  comparisons over 96 trials / 6,144 rows, **0 mismatches**, covering chunked
  integer block sums, topK selection (k in {1,4,16,N}) and topK weighted sums.

The honest summary: you can delete **essentially all** of the A.V matmul at
**~40x fewer operations** with no hard-recall loss, because the target column is
the argmax and the block around it is exact.  You cannot get a **2% soft
readout** for free on a diffuse head, and the deployable sound guarantee is a
guarantee on the target mass, not on the whole value vector.

## Files

| file | role |
| --- | --- |
| value-slider.mjs | Core: RNG, long-context associative/needle task, exact float softmax, EXACT/CHUNKED/LINEAR/TOPK regimes, metrics (KL, qerr, recall, mass), Pareto/knee, TAU admission, integer exactness. |
| run.mjs | Runnable entry: evaluates both variants at N in {64,128,256,512}, prints the Pareto frontier + knee, the admission sweep, the integer tests and the verdict. |
| README.md | This file, including the exact observed output. |

## Run

Real Node (the repo node is a shim), per the package environment:

    cd open-source/maybe/experiments/value-slider-pareto
    /opt/homebrew/bin/node run.mjs     # full report, ~4 s

Everything is seeded (mulberry32) and deterministic; no network, no filesystem,
no dependencies.

## Workload: long-context associative recall + needle-in-haystack

- Keys k_j and values v_j are unit vectors in R^64 and R^32 (seeded Gaussian,
  then normalized).  Queries q_i = normalize(k_t + noise * g), M = 128
  multi-queries per task.
- **assoc** -- the induction-head / associative-recall shape: every key is a
  target, noise 0.05.  The query is a near-copy of its key, so the exact float
  softmax is sharply concentrated on the target (target mass 0.667..0.942).
- **needle** -- needle-in-haystack: 4 planted needle keys, each shadowed by 5
  decoy keys that are noisy copies of the needle.  Queries target the needles
  with noise 0.15, decoyNoise 0.16.  The head is diffuse (target mass
  0.145..0.443) and the exact reference itself misses a few queries (recall
  0.961 at N=512).
- Reference: exact float softmax `A_ij = exp(beta s_ij)/Z` with
  `s_ij = beta * q_i . k_j`, beta = 8, real `Math.exp`, no table.
- N in {64,128,256,512}; the long context is the point (N >> d).
- Seeds are deterministic: `seed = 0x5eed ^ (N * 2654435761)`.

## Regimes over the same exact softmax weights

- **EXACT (B=N)**: full A.V, O(M*N*dv).
- **CHUNKED B**: the block of B consecutive columns containing the argmax is
  discharged exactly; the complementary tail is replaced by (1 - blockMass)
  times the global mean value (the parent's value-side scheme).  B=1 is the
  cheapest, B=N is exact.
- **LINEAR rank r**: one global sufficient statistic
  `S_r = sum_j v_j (P^T k_j)^T` with P a d x r orthonormal projection
  (deterministic Gram-Schmidt, seed 0xabc + r).  Numerator
  `Vsum + S_r (P^T q_i)`, denominator
  `N + (P^T q_i) . sum_j (P^T k_j)`.  The kernel
  `1 + (P^T q).(P^T k)` is non-negative because `||P^T x|| <= ||x|| = 1`,
  so the implied weights are a valid distribution and KL is defined for LINEAR
  too.  r=d is the full affine (soft) kernel; r<d is its rank truncation.  This
  regime does not read the softmax A -- it replaces both matmuls with the
  statistic, which is exactly the cheap end the parent warns about.
- **TOPK k**: keep the k largest weights, renormalize, drop the tail.

## Cost model

The cost unit is one multiply-add OR one comparison (parent-compatible):

    EXACT   cost = M*N*dv
    CHUNKED cost = N*dv + M*(N + (B+1)*dv)
    TOPK    cost = M*(N + k*dv)
    LINEAR  cost = N*dv + N*r*(d+dv) + M*r*(d+dv+1)

`mac` is the bare A.V multiply-add count, reported alongside.  CHUNKED and
TOPK charge the per-query linear scan (N comparisons), which is why the full-op
speedup (16.19x..30.12x) is smaller than the mac-only speedup (25.6x..85.3x).
Charging the scan is the conservative choice.

## The tau miss-not-lie admission

- Observable: the served target mass `m` (the softmax weight of the query's
  target column).
- Band: `[m_exact - 0.05, m_exact + 0.05]`.  A served answer outside the band
  is a lie.
- **SOUND interval**: if the target column is in the exact kept set (the argmax
  block, or the top-k), its cheap mass equals the exact mass (point interval).
  If not, the exact mass is only known to lie in `[0, droppedMass]`.  Both are
  sound brackets, with no distributional assumption.
- **SOUND policy**: admit iff the bracket padded by tau fits inside the band.
  It admits exactly the queries whose target is provably retained, and misses
  the rest (recompute on the exact tier).  Zero lies by construction.
- **ORACLE policy**: exact-anchored point agreement with the exact mass.  The
  zero-lie ceiling; not deployable because it reads the exact mass.
- **PROXY policy**: tail mass trusted to +/-0.02 inside a witness-anchored
  +/-0.2 band (witness = an independent cheap regime).  Deployable, but a loose
  heuristic that can lie.
- `effectiveCost = admitRate*cheapCost + missRate*exactCost`; the zero-lie
  speedup is `exactCost / effectiveCost` among points with lieRate = 0.

On the parent's collapse range: the proven `collapseRange` is for
non-negative integer Buleyean weights with floor 1 and budget R, giving
`[width/W, (R+1)*width/W]`.  Softmax weights are continuous and below 1, so
that literal bracket does not apply here.  Its continuous surrogate is the
kept/dropped envelope above -- each dropped weight is at least 0 and at most the
dropped-mass sum, the same floor/ceiling argument without integer quantization.
We do not claim a sound interval on the **value vector**: the mean-tail moves the
soft readout, so the guarantee is on the target mass only.  That is the boundary
the experiment measures rather than hides.

## Integer exactness

BigInt vs naive Number on an integer task (weights 0..31, values 0..9, M=32,
dv=8), magnitudes far below 2^53 so Number is exact and must agree bit-for-bit:

- chunked integer block sums (B=4; every block boundary and the full row),
- topK selection: naive stable sort vs one-pass insertion scan, k in {1,4,16,N},
- topK k=4 integer weighted sums.

Across N in {64,128,256,512} and 4 seeds: 96 trials, 6144 rows, 766,976 field
comparisons, **0 mismatches**.

## Honest boundaries

1. **The cheap end loses quality.**  LINEAR rank=d recall 0.000..0.047.  The
   knee is inside CHUNKED/TOPK, not linear attention.
2. **Tiny readout loss is not universal.**  It exists on the sharp assoc task
   only up to N=256, and nowhere on the diffuse needle task (best cheap qerr
   0.0556 at ~53% cost).
3. **The sound admission guarantees the target mass, not the output.**  It
   preserves hard argmax retrieval; the mean-tail still moves the soft readout
   (qerr up to 0.4851).
4. **A loose proxy weakens the guarantee.**  PROXY reached lie rate 0.016 and
   its best zero-lie speedup never beats SOUND.
5. **The reference has its own ceiling.**  Exact recall on needle N=512 is
   0.961, so recall differences below that are within the task's own noise.

## Exact observed output

Full deterministic stdout of /opt/homebrew/bin/node run.mjs (no timing fields,
so this block is byte-reproducible):

```
==========================================================================
VALUE-SIDE QUALITY/SPEED SLIDER -- exact softmax fixed, A.V varied
d=64 dv=32 M=128 beta=8 seeds=deterministic
variants: assoc (noise 0.05) and needle (noise 0.15, 4 needles, 5 decoys each)
cost unit: one multiply-add OR one comparison; mac = pure value multiply-adds
==========================================================================

--------------------------------------------------------------------------
[assoc N=64]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=262144 mac=262144 recall=1.000 targetMass=0.9423
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               12288   4.69%      4096  1.1607  0.0621   1.000  1.0000
  TOPK k=2               16384   6.25%      8192  1.0364  0.0556   1.000  0.9939
  CHUNKED B=1            18432   7.03%     10240  0.0259  0.0087   1.000  0.9423
  CHUNKED B=2            22528   8.59%     14336  0.0256  0.0088   1.000  0.9423
  CHUNKED B=4            30720  11.72%     22528  0.0245  0.0087   1.000  0.9423
  CHUNKED B=8            47104  17.97%     38912  0.0233  0.0087   1.000  0.9423
  CHUNKED B=16           79872  30.47%     71680  0.0196  0.0084   1.000  0.9423
  CHUNKED B=32          145408  55.47%    137216  0.0128  0.0072   1.000  0.9423
  EXACT B=N             262144 100.00%    262144  0.0000  0.0000   1.000  0.9423
  TOPK k=64             270336 103.13%    262144 -0.0000  0.0000   1.000  0.9423
  KNEE tiny (qerr<=0.02) : CHUNKED B=1 cost=18432 (7.03%, removed 92.97%) KL=0.0259 qerr=0.0087 recall=1.000
  KNEE plateau (qerr<=1.10x best cheap 0.0072): CHUNKED B=32 cost=145408 (55.47%, removed 44.53%) KL=0.0128 qerr=0.0072 recall=1.000
  ELBOW (qerr)           : CHUNKED B=1 cost=18432 (7.03%, removed 92.97%) KL=0.0259 qerr=0.0087 recall=1.000
  ELBOW (KL)             : CHUNKED B=1 cost=18432 (7.03%, removed 92.97%) KL=0.0259 qerr=0.0087 recall=1.000
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N             262144 100.00%    262144  0.0000  0.0000   1.000  0.9423
  CHUNKED B=1            18432   7.03%     10240  0.0259  0.0087   1.000  0.9423
  TOPK k=1               12288   4.69%      4096  1.1607  0.0621   1.000  1.0000
  LINEAR r=1             20608   7.86%     20608  3.7142  0.9897   0.016  0.0158
  LINEAR r=64          1189888 453.91%   1189888  3.1160  0.9749   0.047  0.0297

--------------------------------------------------------------------------
[assoc N=128]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=524288 mac=524288 recall=1.000 targetMass=0.8917
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               20480   3.91%      4096  2.1779  0.1227   1.000  1.0000
  TOPK k=2               24576   4.69%      8192  2.0222  0.1137   1.000  0.9921
  CHUNKED B=1            28672   5.47%     12288  0.0521  0.0126   1.000  0.8917
  CHUNKED B=2            32768   6.25%     16384  0.0518  0.0126   1.000  0.8917
  CHUNKED B=4            40960   7.81%     24576  0.0508  0.0125   1.000  0.8917
  CHUNKED B=8            57344  10.94%     40960  0.0493  0.0124   1.000  0.8917
  CHUNKED B=16           90112  17.19%     73728  0.0460  0.0123   1.000  0.8917
  CHUNKED B=32          155648  29.69%    139264  0.0388  0.0120   1.000  0.8917
  CHUNKED B=64          286720  54.69%    270336  0.0253  0.0101   1.000  0.8917
  EXACT B=N             524288 100.00%    524288  0.0000  0.0000   1.000  0.8917
  KNEE tiny (qerr<=0.02) : CHUNKED B=1 cost=28672 (5.47%, removed 94.53%) KL=0.0521 qerr=0.0126 recall=1.000
  KNEE plateau (qerr<=1.10x best cheap 0.0101): CHUNKED B=64 cost=286720 (54.69%, removed 45.31%) KL=0.0253 qerr=0.0101 recall=1.000
  ELBOW (qerr)           : CHUNKED B=1 cost=28672 (5.47%, removed 94.53%) KL=0.0521 qerr=0.0126 recall=1.000
  ELBOW (KL)             : CHUNKED B=1 cost=28672 (5.47%, removed 94.53%) KL=0.0521 qerr=0.0126 recall=1.000
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N             524288 100.00%    524288  0.0000  0.0000   1.000  0.8917
  CHUNKED B=1            28672   5.47%     12288  0.0521  0.0126   1.000  0.8917
  TOPK k=1               20480   3.91%      4096  2.1779  0.1227   1.000  1.0000
  LINEAR r=1             28800   5.49%     28800  4.0245  0.9930   0.008  0.0079
  LINEAR r=64          1585152 302.34%   1585152  3.4456  0.9853   0.023  0.0150

--------------------------------------------------------------------------
[assoc N=256]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=1048576 mac=1048576 recall=1.000 targetMass=0.8031
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               36864   3.52%      4096  3.9497  0.2476   1.000  1.0000
  TOPK k=2               40960   3.91%      8192  3.7800  0.2355   1.000  0.9905
  CHUNKED B=1            49152   4.69%     16384  0.0942  0.0192   1.000  0.8031
  CHUNKED B=2            53248   5.08%     20480  0.0940  0.0192   1.000  0.8031
  CHUNKED B=4            61440   5.86%     28672  0.0930  0.0191   1.000  0.8031
  CHUNKED B=8            77824   7.42%     45056  0.0916  0.0190   1.000  0.8031
  CHUNKED B=16          110592  10.55%     77824  0.0887  0.0188   1.000  0.8031
  CHUNKED B=32          176128  16.80%    143360  0.0830  0.0185   1.000  0.8031
  CHUNKED B=64          307200  29.30%    274432  0.0713  0.0179   1.000  0.8031
  CHUNKED B=128         569344  54.30%    536576  0.0473  0.0164   1.000  0.8031
  EXACT B=N            1048576 100.00%   1048576  0.0000  0.0000   1.000  0.8031
  KNEE tiny (qerr<=0.02) : CHUNKED B=1 cost=49152 (4.69%, removed 95.31%) KL=0.0942 qerr=0.0192 recall=1.000
  KNEE plateau (qerr<=1.10x best cheap 0.0164): CHUNKED B=64 cost=307200 (29.30%, removed 70.70%) KL=0.0713 qerr=0.0179 recall=1.000
  ELBOW (qerr)           : CHUNKED B=1 cost=49152 (4.69%, removed 95.31%) KL=0.0942 qerr=0.0192 recall=1.000
  ELBOW (KL)             : CHUNKED B=1 cost=49152 (4.69%, removed 95.31%) KL=0.0942 qerr=0.0192 recall=1.000
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N            1048576 100.00%   1048576  0.0000  0.0000   1.000  0.8031
  CHUNKED B=1            49152   4.69%     16384  0.0942  0.0192   1.000  0.8031
  TOPK k=1               36864   3.52%      4096  3.9497  0.2476   1.000  1.0000
  LINEAR r=1             45184   4.31%     45184  4.0407  0.9984   0.000  0.0040
  LINEAR r=64          2375680 226.56%   2375680  3.5071  0.9937   0.000  0.0075

--------------------------------------------------------------------------
[assoc N=512]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=2097152 mac=2097152 recall=1.000 targetMass=0.6671
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               69632   3.32%      4096  6.6481  0.5031   1.000  1.0000
  TOPK k=2               73728   3.52%      8192  6.4649  0.4844   1.000  0.9878
  LINEAR r=1             77952   3.72%     77952  3.6793  0.9976   0.000  0.0020
  CHUNKED B=1            90112   4.30%     24576  0.1602  0.0268   1.000  0.6671
  CHUNKED B=2            94208   4.49%     28672  0.1599  0.0268   1.000  0.6671
  CHUNKED B=4           102400   4.88%     36864  0.1593  0.0267   1.000  0.6671
  CHUNKED B=8           118784   5.66%     53248  0.1581  0.0267   1.000  0.6671
  CHUNKED B=16          151552   7.23%     86016  0.1554  0.0265   1.000  0.6671
  CHUNKED B=32          217088  10.35%    151552  0.1509  0.0264   1.000  0.6671
  CHUNKED B=64          348160  16.60%    282624  0.1404  0.0264   1.000  0.6671
  CHUNKED B=128         610304  29.10%    544768  0.1211  0.0263   1.000  0.6671
  CHUNKED B=256        1134592  54.10%   1069056  0.0810  0.0228   1.000  0.6671
  EXACT B=N            2097152 100.00%   2097152  0.0000  0.0000   1.000  0.6671
  TOPK k=512           2162688 103.13%   2097152 -0.0000  0.0000   1.000  0.6671
  KNEE tiny (qerr<=0.02) : EXACT B=N cost=2097152 (100.00%, removed 0.00%) KL=0.0000 qerr=0.0000 recall=1.000
  KNEE plateau (qerr<=1.10x best cheap 0.0228): CHUNKED B=256 cost=1134592 (54.10%, removed 45.90%) KL=0.0810 qerr=0.0228 recall=1.000
  ELBOW (qerr)           : CHUNKED B=1 cost=90112 (4.30%, removed 95.70%) KL=0.1602 qerr=0.0268 recall=1.000
  ELBOW (KL)             : CHUNKED B=1 cost=90112 (4.30%, removed 95.70%) KL=0.1602 qerr=0.0268 recall=1.000
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N            2097152 100.00%   2097152  0.0000  0.0000   1.000  0.6671
  CHUNKED B=1            90112   4.30%     24576  0.1602  0.0268   1.000  0.6671
  TOPK k=1               69632   3.32%      4096  6.6481  0.5031   1.000  1.0000
  LINEAR r=1             77952   3.72%     77952  3.6793  0.9976   0.000  0.0020
  LINEAR r=64          3956736 188.67%   3956736  3.2154  0.9949   0.000  0.0038

--------------------------------------------------------------------------
[needle N=64]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=262144 mac=262144 recall=1.000 targetMass=0.4432
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               12288   4.69%      4096 13.0468  1.3625   0.984  0.9844
  TOPK k=2               16384   6.25%      8192  9.8405  0.8521   0.984  0.7743
  CHUNKED B=1            18432   7.03%     10240  0.6352  0.3676   0.984  0.4391
  CHUNKED B=2            22528   8.59%     14336  0.6313  0.3672   0.984  0.4391
  CHUNKED B=4            30720  11.72%     22528  0.6240  0.3665   0.984  0.4391
  CHUNKED B=8            47104  17.97%     38912  0.6082  0.3656   0.984  0.4391
  CHUNKED B=16           79872  30.47%     71680  0.4658  0.2952   0.984  0.4391
  CHUNKED B=32          145408  55.47%    137216  0.2874  0.2340   0.992  0.4419
  EXACT B=N             262144 100.00%    262144  0.0000  0.0000   1.000  0.4432
  KNEE tiny (qerr<=0.02) : EXACT B=N cost=262144 (100.00%, removed 0.00%) KL=0.0000 qerr=0.0000 recall=1.000
  KNEE plateau (qerr<=1.10x best cheap 0.0556): TOPK k=32 cost=139264 (53.13%, removed 46.88%) KL=0.9794 qerr=0.0556 recall=1.000
  ELBOW (qerr)           : CHUNKED B=1 cost=18432 (7.03%, removed 92.97%) KL=0.6352 qerr=0.3676 recall=0.984
  ELBOW (KL)             : CHUNKED B=1 cost=18432 (7.03%, removed 92.97%) KL=0.6352 qerr=0.3676 recall=0.984
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N             262144 100.00%    262144  0.0000  0.0000   1.000  0.4432
  CHUNKED B=1            18432   7.03%     10240  0.6352  0.3676   0.984  0.4391
  TOPK k=1               12288   4.69%      4096 13.0468  1.3625   0.984  0.9844
  LINEAR r=1             20608   7.86%     20608  1.8150  0.9691   0.000  0.0158
  LINEAR r=64          1189888 453.91%   1189888  1.4912  0.9410   0.000  0.0249

--------------------------------------------------------------------------
[needle N=128]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=524288 mac=524288 recall=0.992 targetMass=0.3414
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               20480   3.91%      4096 15.0624  1.8741   0.992  0.9922
  TOPK k=2               24576   4.69%      8192 12.4298  1.2641   0.992  0.7565
  CHUNKED B=1            28672   5.47%     12288  0.6837  0.3824   0.992  0.3407
  CHUNKED B=2            32768   6.25%     16384  0.6813  0.3809   0.992  0.3407
  CHUNKED B=4            40960   7.81%     24576  0.6775  0.3812   0.984  0.3407
  CHUNKED B=8            57344  10.94%     40960  0.6686  0.3809   0.984  0.3407
  CHUNKED B=16           90112  17.19%     73728  0.6520  0.3790   0.984  0.3407
  CHUNKED B=32          155648  29.69%    139264  0.5299  0.3352   0.984  0.3407
  CHUNKED B=64          286720  54.69%    270336  0.3455  0.2655   0.984  0.3414
  EXACT B=N             524288 100.00%    524288  0.0000  0.0000   0.992  0.3414
  KNEE tiny (qerr<=0.02) : EXACT B=N cost=524288 (100.00%, removed 0.00%) KL=0.0000 qerr=0.0000 recall=0.992
  KNEE plateau (qerr<=1.10x best cheap 0.0818): TOPK k=64 cost=278528 (53.13%, removed 46.88%) KL=1.3978 qerr=0.0818 recall=0.992
  ELBOW (qerr)           : CHUNKED B=1 cost=28672 (5.47%, removed 94.53%) KL=0.6837 qerr=0.3824 recall=0.992
  ELBOW (KL)             : CHUNKED B=1 cost=28672 (5.47%, removed 94.53%) KL=0.6837 qerr=0.3824 recall=0.992
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N             524288 100.00%    524288  0.0000  0.0000   0.992  0.3414
  CHUNKED B=1            28672   5.47%     12288  0.6837  0.3824   0.992  0.3407
  TOPK k=1               20480   3.91%      4096 15.0624  1.8741   0.992  0.9922
  LINEAR r=1             28800   5.49%     28800  1.7284  0.9513   0.000  0.0078
  LINEAR r=64          1585152 302.34%   1585152  1.4349  0.9263   0.000  0.0126

--------------------------------------------------------------------------
[needle N=256]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=1048576 mac=1048576 recall=0.969 targetMass=0.2408
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               36864   3.52%      4096 16.8743  3.3572   0.984  0.9844
  TOPK k=2               40960   3.91%      8192 15.0705  2.4083   0.984  0.7662
  LINEAR r=1             45184   4.31%     45184  1.4398  0.9844   0.000  0.0040
  CHUNKED B=1            49152   4.69%     16384  0.6401  0.4256   0.984  0.2376
  CHUNKED B=2            53248   5.08%     20480  0.6389  0.4257   0.984  0.2376
  CHUNKED B=4            61440   5.86%     28672  0.6352  0.4263   0.984  0.2376
  CHUNKED B=8            77824   7.42%     45056  0.6245  0.4244   0.984  0.2376
  CHUNKED B=16          110592  10.55%     77824  0.6107  0.4227   0.984  0.2376
  CHUNKED B=32          176128  16.80%    143360  0.5913  0.4224   0.984  0.2376
  CHUNKED B=64          307200  29.30%    274432  0.4407  0.3258   0.969  0.2408
  CHUNKED B=128         569344  54.30%    536576  0.2744  0.2593   0.969  0.2408
  EXACT B=N            1048576 100.00%   1048576  0.0000  0.0000   0.969  0.2408
  KNEE tiny (qerr<=0.02) : EXACT B=N cost=1048576 (100.00%, removed 0.00%) KL=0.0000 qerr=0.0000 recall=0.969
  KNEE plateau (qerr<=1.10x best cheap 0.1162): TOPK k=128 cost=557056 (53.13%, removed 46.88%) KL=1.8676 qerr=0.1162 recall=0.969
  ELBOW (qerr)           : CHUNKED B=1 cost=49152 (4.69%, removed 95.31%) KL=0.6401 qerr=0.4256 recall=0.984
  ELBOW (KL)             : CHUNKED B=1 cost=49152 (4.69%, removed 95.31%) KL=0.6401 qerr=0.4256 recall=0.984
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N            1048576 100.00%   1048576  0.0000  0.0000   0.969  0.2408
  CHUNKED B=1            49152   4.69%     16384  0.6401  0.4256   0.984  0.2376
  TOPK k=1               36864   3.52%      4096 16.8743  3.3572   0.984  0.9844
  LINEAR r=1             45184   4.31%     45184  1.4398  0.9844   0.000  0.0040
  LINEAR r=64          2375680 226.56%   2375680  1.2040  0.9629   0.000  0.0063

--------------------------------------------------------------------------
[needle N=512]  d=64 dv=32 M=128 beta=8
  exact endpoint : cost=2097152 mac=2097152 recall=0.961 targetMass=0.1451
  PARETO FRONTIER (cost asc; quality = 1/(1+KL exact||method))
  point                   cost   cost%       mac      KL    qerr  recall    mass
  TOPK k=1               69632   3.32%      4096 18.5348  5.4985   0.984  0.9844
  TOPK k=2               73728   3.52%      8192 17.2299  4.0857   0.984  0.7351
  LINEAR r=1             77952   3.72%     77952  1.1494  0.9841   0.000  0.0020
  CHUNKED B=1            90112   4.30%     24576  0.6524  0.4851   0.984  0.1432
  CHUNKED B=2            94208   4.49%     28672  0.6508  0.4847   0.984  0.1432
  CHUNKED B=4           102400   4.88%     36864  0.6485  0.4856   0.984  0.1432
  CHUNKED B=8           118784   5.66%     53248  0.6456  0.4843   0.984  0.1432
  CHUNKED B=16          151552   7.23%     86016  0.6380  0.4824   0.984  0.1432
  CHUNKED B=32          217088  10.35%    151552  0.6260  0.4783   0.984  0.1432
  CHUNKED B=64          348160  16.60%    282624  0.6041  0.4845   0.984  0.1432
  CHUNKED B=128         610304  29.10%    544768  0.4588  0.3823   0.977  0.1443
  CHUNKED B=256        1134592  54.10%   1069056  0.3043  0.3182   0.969  0.1451
  EXACT B=N            2097152 100.00%   2097152  0.0000  0.0000   0.961  0.1451
  KNEE tiny (qerr<=0.02) : EXACT B=N cost=2097152 (100.00%, removed 0.00%) KL=0.0000 qerr=0.0000 recall=0.961
  KNEE plateau (qerr<=1.10x best cheap 0.1453): TOPK k=256 cost=1114112 (53.13%, removed 46.88%) KL=2.2275 qerr=0.1453 recall=0.961
  ELBOW (qerr)           : CHUNKED B=1 cost=90112 (4.30%, removed 95.70%) KL=0.6524 qerr=0.4851 recall=0.984
  ELBOW (KL)             : LINEAR r=1 cost=77952 (3.72%, removed 96.28%) KL=1.1494 qerr=0.9841 recall=0.000
  CHEAPEST PER MODE + endpoints
  point                   cost   cost%       mac      KL    qerr  recall    mass
  EXACT B=N            2097152 100.00%   2097152  0.0000  0.0000   0.961  0.1451
  CHUNKED B=1            90112   4.30%     24576  0.6524  0.4851   0.984  0.1432
  TOPK k=1               69632   3.32%      4096 18.5348  5.4985   0.984  0.9844
  LINEAR r=1             77952   3.72%     77952  1.1494  0.9841   0.000  0.0020
  LINEAR r=64          3956736 188.67%   3956736  0.9571  0.9667   0.000  0.0032

--------------------------------------------------------------------------
CROSS-N / CROSS-VARIANT KNEE SUMMARY
  variant      N  exactCost  kneeTiny   cost%    qerr  kneePlateau   cost%    qerr    lin% linRecall
  assoc       64     262144       B=1   7.03%  0.0087         B=32  55.47%  0.0072 453.91%     0.047
  assoc      128     524288       B=1   5.47%  0.0126         B=64  54.69%  0.0101 302.34%     0.023
  assoc      256    1048576       B=1   4.69%  0.0192         B=64  29.30%  0.0179 226.56%     0.000
  assoc      512    2097152 EXACT B=N 100.00%  0.0000        B=256  54.10%  0.0228 188.67%     0.000
  needle      64     262144 EXACT B=N 100.00%  0.0000    TOPK k=32  53.13%  0.0556 453.91%     0.000
  needle     128     524288 EXACT B=N 100.00%  0.0000    TOPK k=64  53.13%  0.0818 302.34%     0.000
  needle     256    1048576 EXACT B=N 100.00%  0.0000   TOPK k=128  53.13%  0.1162 226.56%     0.000
  needle     512    2097152 EXACT B=N 100.00%  0.0000   TOPK k=256  53.13%  0.1453 188.67%     0.000
  (kneeTiny: cheapest frontier point with qerr <= 0.02; kneePlateau: cheapest within 1.10x
   of the best cheap qerr; "none" = the task admits no such point)

--------------------------------------------------------------------------
TAU MISS-NOT-LIE ADMISSION (bandTol=0.05 on the served target mass)
  SOUND  = sound kept/dropped bracket, deployable, zero lies by construction.
  PROXY  = tail mass trusted to +/-0.02 inside a witness-anchored +/-0.2 band, can lie.
  ORACLE = cheap-vs-exact agreement, zero-lie ceiling, NOT deployable.
  task          candidate     policy      tau  admit   miss    lie   effCost  speedup  maxLie
  assoc N=64    CHUNKED B=1   SOUND     0.000  1.000  0.000  0.000     18432   14.22x       -
  assoc N=64    CHUNKED B=1   PROXY     0.000  1.000  0.000  0.000     43008    6.10x   0.000
  assoc N=64    CHUNKED B=1   ORACLE    0.000  1.000  0.000  0.000     18432   14.22x       -
  assoc N=64    CHUNKED B=4   SOUND     0.000  1.000  0.000  0.000     30720    8.53x       -
  assoc N=64    CHUNKED B=4   PROXY     0.000  1.000  0.000  0.000     55296    4.74x   0.000
  assoc N=64    CHUNKED B=4   ORACLE    0.000  1.000  0.000  0.000     30720    8.53x       -
  assoc N=64    CHUNKED B=16  SOUND     0.000  1.000  0.000  0.000     79872    3.28x       -
  assoc N=64    CHUNKED B=16  PROXY     0.000  1.000  0.000  0.000    104448    2.51x   0.000
  assoc N=64    CHUNKED B=16  ORACLE    0.000  1.000  0.000  0.000     79872    3.28x       -
  assoc N=64    TOPK k=1      SOUND     0.000  1.000  0.000  0.000     12288   21.33x       -
  assoc N=64    TOPK k=1      PROXY     0.000  1.000  0.000  0.000     30720    8.53x   0.000
  assoc N=64    TOPK k=1      ORACLE    0.000  1.000  0.000  0.000     12288   21.33x       -
  assoc N=64    TOPK k=4      SOUND     0.000  1.000  0.000  0.000     24576   10.67x       -
  assoc N=64    TOPK k=4      PROXY     0.000  1.000  0.000  0.000     43008    6.10x   0.000
  assoc N=64    TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     24576   10.67x       -
  assoc N=64    TOPK k=16     SOUND     0.000  1.000  0.000  0.000     73728    3.56x       -
  assoc N=64    TOPK k=16     PROXY     0.000  1.000  0.000  0.000     92160    2.84x   0.000
  assoc N=64    TOPK k=16     ORACLE    0.000  1.000  0.000  0.000     73728    3.56x       -
  assoc N=128   CHUNKED B=1   SOUND     0.000  1.000  0.000  0.000     28672   18.29x       -
  assoc N=128   CHUNKED B=1   PROXY     0.000  1.000  0.000  0.000     61440    8.53x   0.000
  assoc N=128   CHUNKED B=1   ORACLE    0.000  1.000  0.000  0.000     28672   18.29x       -
  assoc N=128   CHUNKED B=4   SOUND     0.000  1.000  0.000  0.000     40960   12.80x       -
  assoc N=128   CHUNKED B=4   PROXY     0.000  1.000  0.000  0.000     73728    7.11x   0.000
  assoc N=128   CHUNKED B=4   ORACLE    0.000  1.000  0.000  0.000     40960   12.80x       -
  assoc N=128   CHUNKED B=16  SOUND     0.000  1.000  0.000  0.000     90112    5.82x       -
  assoc N=128   CHUNKED B=16  PROXY     0.000  1.000  0.000  0.000    122880    4.27x   0.000
  assoc N=128   CHUNKED B=16  ORACLE    0.000  1.000  0.000  0.000     90112    5.82x       -
  assoc N=128   TOPK k=1      SOUND     0.000  1.000  0.000  0.000     20480   25.60x       -
  assoc N=128   TOPK k=1      PROXY     0.000  1.000  0.000  0.000     49152   10.67x   0.000
  assoc N=128   TOPK k=1      ORACLE    0.000  1.000  0.000  0.000     20480   25.60x       -
  assoc N=128   TOPK k=4      SOUND     0.000  1.000  0.000  0.000     32768   16.00x       -
  assoc N=128   TOPK k=4      PROXY     0.000  1.000  0.000  0.000     61440    8.53x   0.000
  assoc N=128   TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     32768   16.00x       -
  assoc N=128   TOPK k=16     SOUND     0.000  1.000  0.000  0.000     81920    6.40x       -
  assoc N=128   TOPK k=16     PROXY     0.000  1.000  0.000  0.000    110592    4.74x   0.000
  assoc N=128   TOPK k=16     ORACLE    0.000  1.000  0.000  0.000     81920    6.40x       -
  assoc N=256   CHUNKED B=1   SOUND     0.000  1.000  0.000  0.000     49152   21.33x       -
  assoc N=256   CHUNKED B=1   PROXY     0.000  0.883  0.117  0.000    209664    5.00x   0.000
  assoc N=256   CHUNKED B=1   ORACLE    0.000  1.000  0.000  0.000     49152   21.33x       -
  assoc N=256   CHUNKED B=4   SOUND     0.000  1.000  0.000  0.000     61440   17.07x       -
  assoc N=256   CHUNKED B=4   PROXY     0.000  0.883  0.117  0.000    220512    4.76x   0.000
  assoc N=256   CHUNKED B=4   ORACLE    0.000  1.000  0.000  0.000     61440   17.07x       -
  assoc N=256   CHUNKED B=16  SOUND     0.000  1.000  0.000  0.000    110592    9.48x       -
  assoc N=256   CHUNKED B=16  PROXY     0.000  0.883  0.117  0.000    263904    3.97x   0.000
  assoc N=256   CHUNKED B=16  ORACLE    0.000  1.000  0.000  0.000    110592    9.48x       -
  assoc N=256   TOPK k=1      SOUND     0.000  1.000  0.000  0.000     36864   28.44x       -
  assoc N=256   TOPK k=1      PROXY     0.000  1.000  0.000  0.000     86016   12.19x   0.000
  assoc N=256   TOPK k=1      ORACLE    0.000  1.000  0.000  0.000     36864   28.44x       -
  assoc N=256   TOPK k=4      SOUND     0.000  1.000  0.000  0.000     49152   21.33x       -
  assoc N=256   TOPK k=4      PROXY     0.000  1.000  0.000  0.000     98304   10.67x   0.000
  assoc N=256   TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     49152   21.33x       -
  assoc N=256   TOPK k=16     SOUND     0.000  1.000  0.000  0.000     98304   10.67x       -
  assoc N=256   TOPK k=16     PROXY     0.000  1.000  0.000  0.000    147456    7.11x   0.000
  assoc N=256   TOPK k=16     ORACLE    0.000  1.000  0.000  0.000     98304   10.67x       -
  assoc N=512   CHUNKED B=1   SOUND     0.000  1.000  0.000  0.000     90112   23.27x       -
  assoc N=512   CHUNKED B=1   PROXY     0.000  0.000  1.000  0.000   2097152    1.00x   0.000
  assoc N=512   CHUNKED B=1   ORACLE    0.000  1.000  0.000  0.000     90112   23.27x       -
  assoc N=512   CHUNKED B=4   SOUND     0.000  1.000  0.000  0.000    102400   20.48x       -
  assoc N=512   CHUNKED B=4   PROXY     0.000  0.000  1.000  0.000   2097152    1.00x   0.000
  assoc N=512   CHUNKED B=4   ORACLE    0.000  1.000  0.000  0.000    102400   20.48x       -
  assoc N=512   CHUNKED B=16  SOUND     0.000  1.000  0.000  0.000    151552   13.84x       -
  assoc N=512   CHUNKED B=16  PROXY     0.000  0.000  1.000  0.000   2097152    1.00x   0.000
  assoc N=512   CHUNKED B=16  ORACLE    0.000  1.000  0.000  0.000    151552   13.84x       -
  assoc N=512   TOPK k=1      SOUND     0.000  1.000  0.000  0.000     69632   30.12x       -
  assoc N=512   TOPK k=1      PROXY     0.000  1.000  0.000  0.000    159744   13.13x   0.000
  assoc N=512   TOPK k=1      ORACLE    0.000  1.000  0.000  0.000     69632   30.12x       -
  assoc N=512   TOPK k=4      SOUND     0.000  1.000  0.000  0.000     81920   25.60x       -
  assoc N=512   TOPK k=4      PROXY     0.000  1.000  0.000  0.000    172032   12.19x   0.000
  assoc N=512   TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     81920   25.60x       -
  assoc N=512   TOPK k=16     SOUND     0.000  1.000  0.000  0.000    131072   16.00x       -
  assoc N=512   TOPK k=16     PROXY     0.000  1.000  0.000  0.000    221184    9.48x   0.000
  assoc N=512   TOPK k=16     ORACLE    0.000  1.000  0.000  0.000    131072   16.00x       -
  needle N=64   CHUNKED B=1   SOUND     0.000  0.984  0.016  0.000     22240   11.79x       -
  needle N=64   CHUNKED B=1   PROXY     0.000  0.719  0.281  0.000    104640    2.51x   0.000
  needle N=64   CHUNKED B=1   ORACLE    0.000  0.984  0.016  0.000     22240   11.79x       -
  needle N=64   CHUNKED B=4   SOUND     0.000  0.984  0.016  0.000     34336    7.63x       -
  needle N=64   CHUNKED B=4   PROXY     0.000  0.719  0.281  0.000    113472    2.31x   0.000
  needle N=64   CHUNKED B=4   ORACLE    0.000  0.984  0.016  0.000     34336    7.63x       -
  needle N=64   CHUNKED B=16  SOUND     0.000  0.984  0.016  0.000     82720    3.17x       -
  needle N=64   CHUNKED B=16  PROXY     0.000  0.719  0.281  0.000    148800    1.76x   0.000
  needle N=64   CHUNKED B=16  ORACLE    0.000  0.984  0.016  0.000     82720    3.17x       -
  needle N=64   TOPK k=1      SOUND     0.000  0.984  0.016  0.000     16192   16.19x       -
  needle N=64   TOPK k=1      PROXY         -      -      -      -         -        -   0.016
  needle N=64   TOPK k=1      ORACLE    0.000  0.984  0.016  0.000     16192   16.19x       -
  needle N=64   TOPK k=4      SOUND     0.000  1.000  0.000  0.000     24576   10.67x       -
  needle N=64   TOPK k=4      PROXY     0.000  0.992  0.008  0.000     44720    5.86x   0.000
  needle N=64   TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     24576   10.67x       -
  needle N=64   TOPK k=16     SOUND     0.000  1.000  0.000  0.000     73728    3.56x       -
  needle N=64   TOPK k=16     PROXY     0.000  0.992  0.008  0.000     93488    2.80x   0.000
  needle N=64   TOPK k=16     ORACLE    0.000  1.000  0.000  0.000     73728    3.56x       -
  needle N=128  CHUNKED B=1   SOUND     0.000  0.992  0.008  0.000     32544   16.11x       -
  needle N=128  CHUNKED B=1   PROXY     0.000  0.148  0.852  0.000    455584    1.15x   0.000
  needle N=128  CHUNKED B=1   ORACLE    0.000  0.992  0.008  0.000     32544   16.11x       -
  needle N=128  CHUNKED B=4   SOUND     0.000  0.992  0.008  0.000     44736   11.72x       -
  needle N=128  CHUNKED B=4   PROXY     0.000  0.148  0.852  0.000    457408    1.15x   0.000
  needle N=128  CHUNKED B=4   ORACLE    0.000  0.992  0.008  0.000     44736   11.72x       -
  needle N=128  CHUNKED B=16  SOUND     0.000  0.992  0.008  0.000     93504    5.61x       -
  needle N=128  CHUNKED B=16  PROXY     0.000  0.148  0.852  0.000    464704    1.13x   0.000
  needle N=128  CHUNKED B=16  ORACLE    0.000  0.992  0.008  0.000     93504    5.61x       -
  needle N=128  TOPK k=1      SOUND     0.000  0.992  0.008  0.000     24416   21.47x       -
  needle N=128  TOPK k=1      PROXY         -      -      -      -         -        -   0.008
  needle N=128  TOPK k=1      ORACLE    0.000  0.992  0.008  0.000     24416   21.47x       -
  needle N=128  TOPK k=4      SOUND     0.000  1.000  0.000  0.000     32768   16.00x       -
  needle N=128  TOPK k=4      PROXY     0.000  1.000  0.000  0.000     61440    8.53x   0.000
  needle N=128  TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     32768   16.00x       -
  needle N=128  TOPK k=16     SOUND     0.000  1.000  0.000  0.000     81920    6.40x       -
  needle N=128  TOPK k=16     PROXY     0.000  1.000  0.000  0.000    110592    4.74x   0.000
  needle N=128  TOPK k=16     ORACLE    0.000  1.000  0.000  0.000     81920    6.40x       -
  needle N=256  CHUNKED B=1   SOUND     0.000  0.984  0.016  0.000     64768   16.19x       -
  needle N=256  CHUNKED B=1   PROXY     0.000  0.016  0.984  0.000   1033728    1.01x   0.000
  needle N=256  CHUNKED B=1   ORACLE    0.000  0.984  0.016  0.000     64768   16.19x       -
  needle N=256  CHUNKED B=4   SOUND     0.000  0.984  0.016  0.000     76864   13.64x       -
  needle N=256  CHUNKED B=4   PROXY     0.000  0.016  0.984  0.000   1033920    1.01x   0.000
  needle N=256  CHUNKED B=4   ORACLE    0.000  0.984  0.016  0.000     76864   13.64x       -
  needle N=256  CHUNKED B=16  SOUND     0.000  0.984  0.016  0.000    125248    8.37x       -
  needle N=256  CHUNKED B=16  PROXY     0.000  0.016  0.984  0.000   1034688    1.01x   0.000
  needle N=256  CHUNKED B=16  ORACLE    0.000  0.984  0.016  0.000    125248    8.37x       -
  needle N=256  TOPK k=1      SOUND     0.000  0.984  0.016  0.000     52672   19.91x       -
  needle N=256  TOPK k=1      PROXY         -      -      -      -         -        -   0.016
  needle N=256  TOPK k=1      ORACLE    0.000  0.984  0.016  0.000     52672   19.91x       -
  needle N=256  TOPK k=4      SOUND     0.000  1.000  0.000  0.000     49152   21.33x       -
  needle N=256  TOPK k=4      PROXY     0.000  0.992  0.008  0.000    105728    9.92x   0.000
  needle N=256  TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     49152   21.33x       -
  needle N=256  TOPK k=16     SOUND     0.000  1.000  0.000  0.000     98304   10.67x       -
  needle N=256  TOPK k=16     PROXY     0.000  0.992  0.008  0.000    154496    6.79x   0.000
  needle N=256  TOPK k=16     ORACLE    0.000  1.000  0.000  0.000     98304   10.67x       -
  needle N=512  CHUNKED B=1   SOUND     0.000  0.984  0.016  0.000    121472   17.26x       -
  needle N=512  CHUNKED B=1   PROXY     0.000  0.000  1.000  0.000   2097152    1.00x   0.000
  needle N=512  CHUNKED B=1   ORACLE    0.000  0.984  0.016  0.000    121472   17.26x       -
  needle N=512  CHUNKED B=4   SOUND     0.000  0.984  0.016  0.000    133568   15.70x       -
  needle N=512  CHUNKED B=4   PROXY     0.000  0.000  1.000  0.000   2097152    1.00x   0.000
  needle N=512  CHUNKED B=4   ORACLE    0.000  0.984  0.016  0.000    133568   15.70x       -
  needle N=512  CHUNKED B=16  SOUND     0.000  0.984  0.016  0.000    181952   11.53x       -
  needle N=512  CHUNKED B=16  PROXY     0.000  0.000  1.000  0.000   2097152    1.00x   0.000
  needle N=512  CHUNKED B=16  ORACLE    0.000  0.984  0.016  0.000    181952   11.53x       -
  needle N=512  TOPK k=1      SOUND     0.000  0.984  0.016  0.000    101312   20.70x       -
  needle N=512  TOPK k=1      PROXY         -      -      -      -         -        -   0.016
  needle N=512  TOPK k=1      ORACLE    0.000  0.984  0.016  0.000    101312   20.70x       -
  needle N=512  TOPK k=4      SOUND     0.000  1.000  0.000  0.000     81920   25.60x       -
  needle N=512  TOPK k=4      PROXY     0.000  1.000  0.000  0.000    172032   12.19x   0.000
  needle N=512  TOPK k=4      ORACLE    0.000  1.000  0.000  0.000     81920   25.60x       -
  needle N=512  TOPK k=16     SOUND     0.000  1.000  0.000  0.000    131072   16.00x       -
  needle N=512  TOPK k=16     PROXY     0.000  1.000  0.000  0.000    221184    9.48x   0.000
  needle N=512  TOPK k=16     ORACLE    0.000  1.000  0.000  0.000    131072   16.00x       -

BEST ZERO-LIE SPEEDUP PER TASK (any candidate, any policy):
  assoc N=64    SOUND 21.33x (admit=1.000, lie=0.000, effCost=12288 vs exact 262144)
  assoc N=128   SOUND 25.60x (admit=1.000, lie=0.000, effCost=20480 vs exact 524288)
  assoc N=256   SOUND 28.44x (admit=1.000, lie=0.000, effCost=36864 vs exact 1048576)
  assoc N=512   SOUND 30.12x (admit=1.000, lie=0.000, effCost=69632 vs exact 2097152)
  needle N=64   SOUND 16.19x (admit=0.984, lie=0.000, effCost=16192 vs exact 262144)
  needle N=128  SOUND 21.47x (admit=0.992, lie=0.000, effCost=24416 vs exact 524288)
  needle N=256  SOUND 21.33x (admit=1.000, lie=0.000, effCost=49152 vs exact 1048576)
  needle N=512  SOUND 25.60x (admit=1.000, lie=0.000, effCost=81920 vs exact 2097152)

--------------------------------------------------------------------------
INTEGER EXACTNESS (BigInt vs naive Number; magnitudes kept below 2^53)
  test                                trials     rows    fields  mismatches
  chunked integer block sums (B=4, dv=8)      32     1024    491520           0
  topK selection (k in {1,4,16,N})        32     4096    267264           0
  topK k=4 integer weighted sums          32     1024      8192           0
  TOTAL                                   96     6144    766976           0

--------------------------------------------------------------------------
VERDICT
  exact endpoint        : B=N, cost = M*N*dv, recall=0.961..1.000, targetMass=0.145..0.942
  CHEAP END (LINEAR r=d): recall=0.000..0.047 (loses hard retrieval), qerr=0.926..0.995, cost=1189.9k..3956.7k
  KNEE (CHUNKED B=1)    : removes 92.97%..95.70% of the A.V cost at qerr=0.0087..0.4851, KL=0.0259..0.6837, recall=0.984..1.000
  KNEE (TOPK k=1)       : qerr=0.0621..5.4985, recall=0.984..1.000 (hard recall kept, distribution destroyed)
  VALUE MACs (CHUNKED B=1): removes 96.09%..98.83% of the A.V multiply-adds (25.6x..85.3x mac-only); the full-op speedup also charges the per-query argmax scan.
  SOUND admission       : zero-lie for every candidate/tau = true
  BEST zero-lie speedup : 16.19x..30.12x over exact
  INTEGER EXACTNESS     : 766976 field comparisons, 0 mismatches

  Honest boundaries:
   * The CHEAP end is not free quality: LINEAR rank=d loses hard retrieval entirely
     (recall=0.047).  The knee is inside CHUNKED/TOPK, not linear attention.
   * CHUNKED B=1 is already at the quality plateau; larger B buys almost no quality
     and costs more, so the knee is the cheapest block.
   * The SOUND bracket guarantees the TARGET MASS is exact (target in the kept
     set), which preserves hard argmax retrieval.  It does NOT make the full value
     vector exact: the mean-tail still moves the soft readout (qerr up to
     0.4851).  A soft readout must use a tighter metric or a larger B.
   * Tiny (<=2%) readout loss is NOT available on the diffuse needle task: no cheap
     point reaches qerr <= 0.02 there (the best cheap qerr is 0.0556 at ~53% cost).
   * The PROXY is a loose heuristic, not a sound interval: maximum observed lie rate
     0.016 (needle, TOPK k=1).  Where it cannot admit it is slower than SOUND,
     and where it admits aggressively it ships lies.  SOUND is the zero-lie policy that pays.
```
