/**
 * exactness.mjs -- standalone exactness entry for the Buleyean Hamming experiment.
 *
 *   cd open-source/maybe/experiments/buleyean-hamming-attention
 *   /opt/homebrew/bin/node exactness.mjs
 *
 * Prints the integer/BigInt exactness report (naive per-coordinate vs packed
 * popcount; ternary masks vs signed dot; multi-view consensus; the floor) and
 * the precomputed-table exp exactness.  No imports beyond the local core.
 */
import { bigintHammingExactness, hammingSoftmaxTableExactness } from './hamming-attention.mjs';

function main() {
  const r = bigintHammingExactness({ trials: 400, seed: 20250701, maxN: 40, maxD: 8, maxDV: 5 });
  console.log('trials=' + r.trials + ' rows=' + r.rows + ' comparisons=' + r.comparisons + ' mismatches=' + r.mismatches);
  console.log('ternary naive-dot vs A-B: comparisons=' + r.ternaryComparisons + ' mismatches=' + r.ternaryMismatches);
  console.log('multi-view fuse (weight addition) vs naive per-view: comparisons=' + r.fusedComparisons + ' mismatches=' + r.fusedMismatches);
  console.log('fusion order-independence (shuffled views): comparisons=' + r.orderComparisons + ' mismatches=' + r.orderMismatches);
  console.log('control: sum rejections then God Formula vs weight sum: comparisons=' + r.fuseThenRuleComparisons + ' mismatches=' + r.fuseThenRuleMismatches);
  console.log('floor: min integer weight=' + r.minWeight);
  console.log('example=' + JSON.stringify(r.example));
  const te = hammingSoftmaxTableExactness({ trials: 400, seed: 20250702, maxN: 40, maxRange: 64 });
  console.log('softmax-table exp: comparisons=' + te.comparisons + ' mismatches=' + te.mismatches +
    ' tableExpEvals=' + te.tableEvals + ' directExpEvals=' + te.directEvals);
  console.log('example=' + JSON.stringify(te.example));
}

main();
