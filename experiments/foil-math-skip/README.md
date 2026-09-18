# FOIL math skip — precompute the math, skip the computer

Parent: [Makehay experiments](../README.md) · Thesis: [Skip is the computer](../../../docs/BUSINESS%20BINDER/SKIP_COMPUTER_FOUNDATION.md) · Benchmark ledger: [BENCHMARK_WINS.md](../../../docs/BUSINESS%20BINDER/BENCHMARK_WINS.md)

The FOIL thesis is that a completely identified program can be frozen as a value:
fingerprint the fork/race/fold shape plus inputs, address the answer, and never run
it twice. This experiment tests the cheapest possible instance of that thesis --
common math expressions -- and measures how much recomputation a sealed table
actually removes, under the tau miss-not-lie admission.

Run it: /opt/homebrew/bin/node run.mjs (~2 s, deterministic).

## Method

For each expression, on a bounded integer key domain:

1. build an EXACT table (T[k] = f(k), a Float64Array);
2. build an approximate table quantized to step, with a per-entry bound
   bound(k) = (step/2) * max(1, |f(k)|);
3. evaluate a concentrated workload (70% of keys in the hot decile, 30% spread);
4. measure real ns/call for recomputation and for the array read, and compute the
   speedup.

EXACT serves every in-domain key and is bit-for-bit identical to recomputation.
SOUND serves key k only when 2*bound(k) <= margin (margin 0.5 here), else it
reports a miss and the caller recomputes.

## Results (N = 2,000,000 keys/expression, 3 seeds)

| expr | recompute | lookup | EXACT speedup | SOUND speedup | SOUND miss | lies |
|---|---:|---:|---:|---:|---:|---:|
| sin | 32.3 ns | 1.11 ns | 29.37x | 29.37x | 0.0% | 0 |
| sigmoid | 29.2 ns | 1.01 ns | 29.03x | 29.03x | 0.0% | 0 |
| tanh | 15.6 ns | 1.40 ns | 11.41x | 11.41x | 0.0% | 0 |
| log1p | 18.1 ns | 1.36 ns | 13.90x | 13.90x | 0.0% | 0 |
| sqrt | 9.3 ns | 1.52 ns | 6.31x | 6.31x | 0.0% | 0 |
| exp | 7.7 ns | 1.67 ns | 5.57x | 0.85x | 97.5% | 0 |

EXACT speedup range 5.57x..29.37x; zero-lie SOUND range 0.85x..29.37x; served
values outside the declared bound: 0.

## The honest finding

The first cut reported 100,399 approximate servings outside the declared bound,
all from exp. The cause was a single global absolute tau over a domain where exp
overflows Float64: a global tau is simply false there. The per-entry bound fixes it
and the cost is visible -- exp's SOUND speedup collapses to 0.85x because 97.5% of
keys miss. That is the same lesson the Rust lane landed as Precision::PerEntry (D3):

* for bounded-value expressions (sin, sigmoid, tanh, log1p, sqrt) a sealed table is
  a clean 6x..29x skip with zero lies;
* for a wide-domain transcendental the table must be bounded (log-domain, narrow
  range) or use a per-entry bound that honestly converts the steep region into
  misses.

## What this does and does not claim

* It measures a local skip: table read versus recomputation on one host.
* It does not claim the transport or occupancy win (that is FOIL/protocol69, a
  different measurement).
* The cache-hit win is the concentration of real workloads; a uniform workload over
  a huge domain would not hit.
* No profitability, energy, or end-to-end serving claim is made here.
