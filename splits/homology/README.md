# splits/homology

## `removed_89_reconstructed.tsv`

The 89 development non-effector records excluded from v7 training, one row
each, with the reason and the record each duplicates.

| Basis | Rows |
|---|---|
| `duplicate_of_benchmark_non_effector` | 80 |
| `duplicate_of_benchmark_effector` (`negative_7301` == `Fung-24`) | 1 |
| `duplicate_of_training_effector` | 8 |

5505 development non-effector structures − 89 = **5416**. With the 407
effectors, the released training set is **407 + 5416**. Positives and the
213-protein benchmark are untouched.

Sequences are taken from the structures in `datasets/pdb_structures/`, which is
what the features were computed from, not from the pre-audit FASTA files.

### Provenance, and why the name says "reconstructed"

81 of the 89 are recoverable mechanically: their sequence is an exact copy of a
benchmark protein. The remaining 8 duplicate a *training* effector, and three
of those are not byte-identical, so no exact-matching procedure finds them:

| Removed | Training effector | Relationship |
|---|---|---|
| `negative_5903` (401 aa) | `seq_365` (401 aa) | one residue differs, 99.8% |
| `negative_5342` (173 aa) | `seq_486` (214 aa) | truncation, 100% over its length |
| `negative_5650` (342 aa) | `seq_501` (333 aa) | contains the positive, 100% |

Those eight ids come from the training run's own `removed.tsv`, produced by
`scripts/make_homology_split.py` with homology clustering (E ≤ 1e-3, or ≥20%
identity over ≥50% of the shorter sequence). This file reproduces that list and
adds the partner and relationship for each row; it has not yet been diffed
against the original. Do that when `removed.tsv` is to hand, and keep the
original as the authority if they disagree.

## `split.tsv` — not yet committed

The train/validation/test partition of the development set. It cannot be
reconstructed from the data: single-linkage homology components depend on the
all-against-all alignment, and the reported metrics are computed on these
partitions. Copy it from the training machine. `scripts/verify_release.py`
fails until it is here.
