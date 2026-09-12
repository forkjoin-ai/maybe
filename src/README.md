# Src

Parent: [Maybe](../README.md)

Source module for the Src feature area.

## Key Files

- `memory-abduction.ts`: Pure JSON adapter over `buildAbduction` for Moonshine's
  memory paths. Positive learned strengths form priors; observed heuristic
  signals rank candidates with hard walls disabled. Missing signals remain 0.5.
  No observations produces no ranking. Reports indistinguishable evidence and
  at most three discriminating local probes; the caller owns scope, relevance
  admission and probe execution. Weights are not empirical confidence. Embedded
  dependency closure: this file, `abduction.ts`, and `@a0n/buleyean-kernel`.
- `memory-abduction.test.ts`: Actual-engine fixture parity, finite normalized
  weights, unknown evidence, disabled walls and bounded-input checks.

- `journey-selection.ts`: Ordered-prefix probabilities over a bounded observed
  trace census, then greedy marginal visible-wait selection under byte, parse,
  build and candidate budgets. Shared resources cost once and overlapping
  prefixes earn only additional savings. Abduction hints retain their separate
  provenance; no observations returns null probabilities and no empirical
  selection. Scenario/test counts must never be supplied as user observations.
- `journey-selection.test.ts`: Conditioning, empty evidence, overlap, budgets and
  invalid mass/resource regression cases.

- `abduction.ts`: Source file in this module.
- `bang-abduction.ts`: Peptide/bang hard-wall abduction + couple search bracket + RL rejection records. Brute-force enumeration of the admitted rectangle is `@a0n/aeon-crackerjack` `ChemistryBruteForce`.
- `bule.ts`: Source file in this module.
- `buleyean.test.ts`: Source file in this module.
- `buleyean.ts`: Source file in this module.
- `charisma-rank.test.ts`: Source file in this module.
- `charisma-rank.ts`: Source file in this module.
- `charisma-skyrms-game.test.ts`: Source file in this module.
- `charisma-skyrms-game.ts`: Source file in this module.
- `charisma-toy-model.test.ts`: Source file in this module.
- `charisma-toy-model.ts`: Source file in this module.
- `chordonomicon-progression.test.ts`: Source file in this module.
- `chordonomicon-progression.ts`: Source file in this module.
- ...and 23 more source files.

## Subdirectories

- `topologies/`

## Usage

Use this README as the local map for this directory. Keep implementation details near the owning source files, and update this document when the directory contract changes.

## Verification

Run the closest package or app-level check through the repository-owned `a0` or `monster` target after changing behavior in this directory.

Journey selection checks: `a0 run @a0n/maybe:test:journeys` and
`a0 run @a0n/maybe:typecheck:journeys`.

Memory adapter checks: `a0 run @a0n/maybe:test:memory` and
`a0 run @a0n/maybe:typecheck:memory`.

Moonshine's `moonshine:test:mycelial-memory` target verifies packaged adapter
execution through its linked native evaluator.
