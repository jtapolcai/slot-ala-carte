# RANDAO Forking Attack Simulator

A Python/C++ simulation framework for analysing **RANDAO biasability attacks** on Ethereum's proof-of-stake consensus.  An adversary with fraction `alpha` of total stake can selectively publish or withhold blocks to bias the RANDAO beacon, gaining a disproportionate share of proposer slots in future epochs.

## Overview

Ethereum uses a RANDAO beacon to assign block-proposer roles.  An adversary controlling multiple validators can **look ahead** at their slot assignments and choose to fork the chain or sacrifice blocks in order to obtain a better RANDAO outcome for the next epoch.

This simulator:
- Models the adversary's optimal attack policy over 32-slot epochs.
- Computes per-epoch utility using an exact myopic model and a multi-epoch heuristic.
- Runs Monte Carlo experiments to estimate the expected long-run slot gain relative to proportional share.

## Repository Structure

```
monte_carlo_attack.py        Entry point: Monte Carlo simulation
epoch_utility_function.py    Heuristic multi-epoch utility (pygtrie trie)
attack_string.py             Myopic per-epoch utility for a single attack string
tail_slots.py                Groups attack strings by epoch tail, indexed by head
decision_tree.py             Per-epoch attack decision policy
forking_string.py            Attack string helpers (generation, parsing, encoding)
forking_attack.py            ForkingAttack: models a forking strategy
honest_attack.py             HonestAttack: base class, binomial slot distributions
selfish_mixing_attack.py     SelfishMixingAttack: selfish-mixing variant
realization.py               Epoch realization sampling and outcome evaluation
decompose_attack_string.py   Decomposes an epoch string into attack string edges
find_longest_attack_string.py  Finds the longest profitable attack string
utility_distr.py             PMF/CDF utilities, expected-value helpers
known_outcome.py             Fixed (known-outcome) realization helper
save_policy.py               Serialise/deserialise utility function to JSON/XML
logger.py                    Lightweight verbosity-level logger
```

## Requirements

- Python 3.11+
- [pygtrie](https://github.com/google/pygtrie)

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install pygtrie
```

## Usage

### Monte Carlo simulation

```bash
# Sweep all alpha values (default: alpha=25, 1000 epochs)
python monte_carlo_attack.py

# Single alpha value with a summary info about each epoch
python monte_carlo_attack.py -alpha 30 -epoch 50 -log 3

# Target-slot mode (utility weight w=100)
python monte_carlo_attack.py -alpha 25 -target
```

### Detailed debug example (with explanation)

Command:

```bash
python monte_carlo_attack.py -alpha 35 -epoch 1 -log 4
```

Typical output with inline comments (excerpt):

```text
Utility is generated on the fly for alpha=0.35 avg utility: 3.00  # Trie utility table is built dynamically during simulation.
As a refrence, in Table 2 the average number of slots was 44.36%   # Baseline reference from the paper/comparison table.
	head utility: [0]                                                   # Initial continuation utility vector.
	Avg. head utility: 0                                               # Mean continuation utility at start.
	simulate attack AHHAAAH. with utility 2.723845265247273 (->43.5)  # Candidate attack string for current decision point.
	--- Decision point (0) AHHAAAH. (AHHAAAH.) the public chain is    # Branching step: evaluate publish/fork options.
	Realization CNNCCCN ( PFFOPOF) has sacrifice 0                    # One sampled realization with zero sacrifice.
	For CNNCCCN ( PFFOPOF) the output is ... AS utility: 4.33 ... expected utility:7.39  # AS utility + reward/sacrifice => expected utility.
	...
	Decide to fork at AHHAAAH. (7.39 > treashold-11.2+2.9952000000000005+0 ...)  # Best expected utility beats fork threshold.
	...
	Publish private chain CNNCCCN to extend public chain to 7 slots   # Private chain is published when optimal.
	...
	After the attack the epoch string is HHHHAHAAHHAHHAHAAAHHHAAAHHHHAAAH  # Final epoch outcome for the step.
	0.44 attack:AHHAAAH. (CNNCCCN 0.44) sacrifice 0 ... RANDAO outcomes seen:6  # Per-attack summary line.
Results in 0.0000% slots alpha=35.0% and after 1 measured slots (out of 1) ...  # Aggregate run summary.
number of attack strings:14 expected 46.46% (from 35%)                           # Current discovered strategy space size.
Result saved to results.xml                                                       # XML summary written.
Attacks saved realisation_string.json                                             # Utility/attack database written.
```

### Inspect the utility function for a specific epoch string

```bash
python epoch_utility_function.py -attack AAHAHH -alpha 35 -log 3

# Tail-focused attack input example
python epoch_utility_function.py -attack AH.HA -alpha 35

# Confirmed stable utility example (recommended format)
python epoch_utility_function.py -attack AHHAA. -alpha 35
```

### Forking string parameters and realizations

`forking_string.py -attack` prints the structural parameters and all possible realizations for a given attack string.

Command:

```bash
python forking_string.py -attack AAHA
```

Output:

```text
AAHA -> a:3, h:1, a1:2, realization string length:4, A before epoch boundary:3 self-forking A slots:[None,None] number of slots that can be missed:1
POFO
PMFO
```

**Parameter meanings:**

| Field | Value | Meaning |
|-------|-------|---------|
| `a` | 3 | Total adversary (`A`) slots in the attack string |
| `h` | 1 | Total honest (`H`) slots in the attack string |
| `a1` | 2 | Consecutive leading `A` slots (before the first `H`) |
| `realization string length` | 4 | Length of the realization string (tail slots only) |
| `A before epoch boundary` | 3 | Adversary slots in the tail (before `.`) |
| `self-forking A slots` | `[None,None]` | Range of adversary slots that can be self-forked; `None` means no self-forking range applies |
| `number of slots that can be missed` | 1 | How many adversary slots can be sacrificed while the attack remains profitable |

**Realization string encoding** — each character maps to one slot in the tail:

| Symbol | Meaning |
|--------|---------|
| `P` | **P**ublished — adversary's first (anchor) slot, always published |
| `O` | **O**pted — adversary slot included in the private chain (published) |
| `M` | **M**issed — adversary slot sacrificed (withheld) |
| `F` | **F**orked — honest slot that is forked over by the adversary chain |
| `S` | **S**elf-forked — adversary slot used as a self-fork pivot |

The two realizations above are:
- `POFO` — all adversary slots published (no sacrifice)
- `PMFO` — the second adversary slot is sacrificed (one missed)

### Decompose attack string example

Command:

```bash
python decompose_attack_string.py -alpha 35 -attack AAHAHH
```

Typical output (excerpt):

```text
weak forking string                                               # Weak-forking feasibility check for a candidate prefix.
weak forking string                                               # Repeated for each relevant slot while scanning backward.
weak forking string
save:29->AHH.A                                                    # New decomposed state saved for slot 29.
DEBUG HHHHHHHHHHHHHHHHHHHHHHHHHHAAHAHH Edge: 29 --F--> 33 (through epoch) : AHH.A -> .
save:27->AHAHH.A                                                  # New decomposed state saved for slot 27.
DEBUG HHHHHHHHHHHHHHHHHHHHHHHHHHAAHAHH Edge: 27 --WF--> 29 (through epoch) : AHAHH.A -> AHH.A
save:26->AAHAHH.A                                                 # New decomposed state saved for slot 26.
DEBUG HHHHHHHHHHHHHHHHHHHHHHHHHHAAHAHH Edge: 26 --SM--> 27 (through epoch) : AAHAHH.A -> AHAHH.A
DEBUG HHHHHHHHHHHHHHHHHHHHHHHHHHAAHAHH Edge: 26 --WF--> 29 (through epoch) : AAHAHH.A -> AHH.A
```

### Generate a LaTeX table of the most probable attack strings

`find_longest_attack_string.py` can run a Monte Carlo sweep over `alpha` values and write a LaTeX table to `popular_attack_string_table.tex`.

**Reproduce Table 5 from the paper** (top-25 attack strings per alpha, with utility and realization counts, 10 000 Monte Carlo epochs):

```bash
python find_longest_attack_string.py -table 25 -latex -utility -epoch 10000
```

Output is written to `popular_attack_string_table.tex`.

Other examples:

```bash
# Top-10 attack strings for each alpha (no utility column)
python find_longest_attack_string.py -table 10 -latex

# Same, with utility and realization-count columns
python find_longest_attack_string.py -table 10 -latex -utility

# Restrict to a single alpha value
python find_longest_attack_string.py -table 10 -latex -utility -alpha 35

# Choose alpha sweep range and step
python find_longest_attack_string.py -table 10 -latex -alpha_min 25 -alpha_step 5
```

Flags specific to table generation:

| Flag | Description | Default |
|------|-------------|---------|
| `-table N` | Include the N most probable attack strings per alpha column | `0` (off) |
| `-latex` | Write the result to `popular_attack_string_table.tex` | off |
| `-utility` | Add utility value and realization-count columns (requires extra computation) | off |
| `-epoch N` | Monte Carlo epochs per alpha value used to estimate probabilities | `10000` |
| `-alpha_min` | Smallest alpha (integer %) in the sweep | `20` |
| `-alpha_step` | Step size between alpha values in the sweep | `5` |

The output file uses the macros `\AS`, `\HS`, `\epoch` for `A`, `H`, `.`
— include the appropriate macro definitions in your LaTeX preamble.
Entries that differ between the with- and without-weak-forking variants are
marked with a superscript `^*`.

### Key flags


| Flag | Description | Default |
|------|-------------|---------|
| `-alpha` | Adversary stake percentage (integer) | `35` |
| `-epoch` | Number of epochs to simulate | `1000` |
| `-log` | Verbosity: 1=critical … 5=trace | `3` |
| `-target` | Enable target-slot utility weighting | off |
| `-attack` | Run with a fixed epoch string | — |
| `-filter` | Restrict attack string set (`selfish_mixing`, `no_weak_forking`, ) | `""` |
| `-um` | Utility multiplier: weight of an `A` slot in the utility function | `1` |
| `-pw` | Head-weight parameter (recommended: `< 1.0 - alpha`) | `0` |
| `-fm` | Fork multiplier: scales the forking threshold | `1` |

The performance tuning flags are: `-um`, `-pw`, `-fm`.

## Attack String Format

An attack string has the form `tail.head`, where:

- **tail** — sequence of `A`/`H` slots from the current epoch that the adversary commits to
- **head** — adversary slot positions at the start of the next epoch that determine future RANDAO influence

Example: `AAH.A` — two adversary slots, one honest slot in the tail; first slot of the next epoch is adversary.

## Output

Results are written to:
- `realisation_string.json` — serialised utility trie
- `results.xml` — simulation statistics (slot gain per alpha)


