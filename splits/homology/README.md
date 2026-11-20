# splits/homology

Produced by `make_homology_split.py` on 2026-10-15 (MMseqs2 `8cc5ce36`), with
sequences read from the structures in `datasets/pdb_structures/` rather than
from the FASTA files. `manifest.json` carries the full parameter set and
`leakage_report.txt` the run's own narrative.

Two linking rules, and the asymmetry between them matters:

| Purpose | Rule |
|---|---|
| Build homology components | E ≤ 0.001 |
| Verify partitions | E ≤ 0.001 **or** identity ≥ 0.20 over ≥ 50% of the shorter sequence, any E |

Verification is the looser rule, so two proteins can fall into different
components and still violate the check once they sit in different partitions.
Repairing that is where the partition drops come from.

## Two sets of numbers, and they mean different things

```
input                     407 effectors   5505 non-effectors
  - decontamination                     -   89          (81 benchmark duplicates + 8 label conflicts)
cleaned development set   407 effectors   5416 non-effectors   = 5823
  - partition drops        -  4           -   40          (44 identity-only, family members)
partitioned set           403 effectors   5376 non-effectors   = 5779
```

**The released FASTAs hold the cleaned set, 407 + 5416.** The 89 are genuinely
not data: the same protein was present twice, already in the benchmark or under
a conflicting label. **`split.tsv` holds the partitioned set, 5779**, and is
what records which proteins the model actually trained on.

The 44 are not contamination and are not errors. They are correctly labelled
proteins dropped from validation and test because an identity-only homologue
sat across a partition boundary and they belonged to a family that could not be
relocated intact. Four of them are effectors. They stay in the dataset because
a reader re-splitting the data differently should have them; `split.tsv` is
what excludes them from this split.

Report the training set as **4109 train / 845 validation / 825 test**
(effectors 289 / 56 / 58), drawn from a cleaned development set of 5823.

## `removed.tsv` — 133 rows

| Reason | Rows | Identity |
|---|---|---|
| duplicates an independent-benchmark protein | 81 | 1.000 |
| label conflict (non-effector duplicating an effector) | 8 | 5 at 1.000, 2 at 1.000 over the shorter sequence, 1 at 0.997 |
| validation/test drop, identity-only family member | 44 | 0.212 – 0.390 |

Decontamination used `--dup-id 0.95`, not exact string equality, which is why
three of the eight label conflicts are not byte-identical: `negative_5903`
differs from `seq_365` at one residue, `negative_5342` is a 173 aa truncation
of the 214 aa `seq_486`, and `negative_5650` contains `seq_501`.

**Do not reconstruct this file by sequence matching.** Exact matching finds 81
of the 89 decontamination rows, misses those three, and cannot derive the 44
partition drops at all. Anything consuming it must also separate the 89 from
the 44 by the `reason` column — removing all 133 from the dataset would publish
5372 non-effectors and describe the dataset as smaller than it is.
`scripts/prepare_release_datasets.py` does this separation.

## A caveat on `manifest.json`

`input_fasta_sha256` records the hashes of the **pre-audit** FASTA files that
the split consumed. The FASTAs in this repo have since been regenerated from
the structures and carry the 89 removals, so those hashes will not match the
committed files. The pre-audit versions are recoverable from git history. The
`notes` at the top of `leakage_report.txt` list the sequences that differed
between the old FASTAs and the structures, which is why the regeneration was
done from the structures in the first place.

## `benchmark_homology.tsv`

Per-benchmark-protein homology against the cleaned development set. This is the
source for the central claim: of the 53 benchmark effectors, 22 have no
homologue anywhere in the development data and 22 have a homologue in the train
partition. Benchmark proteins were never altered — duplicates were removed from
the development side.
