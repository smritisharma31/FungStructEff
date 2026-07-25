"""
fungstructeff/features/entropy.py
Sequence conservation features from an MMseqs2 search.

IDENTITY SCALE — READ BEFORE "FIXING"
-------------------------------------
`mmseqs convertalis` emits `fident` as a FRACTION (0-1). The thresholds below
(`> 70` conserved, `< 30` variable) were written for PERCENT identity (0-100),
so with raw fident they are never/always satisfied:

    pct_conserved == 0.0  for every protein
    pct_variable  == 1.0  for every protein

The training features (final_features_cd_hit_NEW_ENTROPY_v4.csv) clearly DID
use percent identity — pct_conserved there takes 685 distinct values and equals
1.0 for 272 proteins, which is impossible on a 0-1 scale. Converting to percent
therefore looks like the obvious correction.

It was tried, and it measurably HURTS. On the 213-protein independent set:

    raw fident (this file)      ROC-AUC 0.901 +/- 0.005   (4 replicates)
    fident * 100                ROC-AUC 0.8754
    fident * 100, MIN_HITS=1    ROC-AUC 0.8754

The penalty is ~5x the replicate noise and is unaffected by MIN_HITS, so it is
attributable to the scale change alone. The likely explanation is that the
training feature pipeline differed from this one in some other respect (search
parameters, identity definition), so "correct" percent identities computed here
do not land in the distribution the model was fitted on. Two constant features
are at least consistently uninformative; miscalibrated real ones are actively
misleading to the classifier.

Raw fident is kept deliberately. Resolving this properly would mean
regenerating the training features with this pipeline and retraining — not
changing this file in isolation.

HIT COUNT
---------
Entropy statistics scale with the NUMBER of hits (for k roughly-equal hits the
mean of -p*log2(p) is about log2(k)/k). A MIN_HITS rule that zero-fills queries
with few homologs was tested; it changed ROC-AUC by 0.0001 and is disabled by
default (MIN_HITS = 1). It remains available via the environment variable.
"""

import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import skew


ENTROPY_COLUMNS = [
    "mean_entropy", "variance_entropy", "std_entropy",
    "skew_entropy", "pct_conserved", "pct_variable",
]

# ---- search parameters (override via environment) --------------------
SENSITIVITY = os.environ.get("FUNGSTRUCTEFF_MMSEQS_S", "7.5")
MAX_SEQS    = os.environ.get("FUNGSTRUCTEFF_MMSEQS_MAX_SEQS", "300")
EVALUE      = os.environ.get("FUNGSTRUCTEFF_MMSEQS_EVALUE", "0.001")

# Minimum hits required to compute conservation statistics; below this the
# query is zero-filled. Default 1 = disabled (see module docstring).
MIN_HITS = int(os.environ.get("FUNGSTRUCTEFF_MIN_HITS", "1"))

# convertalis default column order
ALN_COLUMNS = [
    "query", "target", "fident", "alnlen",
    "mismatch", "gapopen", "qstart", "qend",
    "tstart", "tend", "evalue", "bits",
]


def run_mmseqs2(fasta_file, db_path, output_dir, threads=8,
                sensitivity=None, max_seqs=None, evalue=None):
    """Run an MMseqs2 search and return the path to alignment.tsv."""
    os.makedirs(output_dir, exist_ok=True)
    query_db  = os.path.join(output_dir, "queryDB")
    result_db = os.path.join(output_dir, "resultDB")
    tmp_dir   = os.path.join(output_dir, "tmp")
    aln_file  = os.path.join(output_dir, "alignment.tsv")

    sensitivity = str(sensitivity or SENSITIVITY)
    max_seqs    = str(max_seqs or MAX_SEQS)
    evalue      = str(evalue if evalue is not None else EVALUE)

    print("\n🔹 Running MMseqs2 search...")
    print(f"   -s {sensitivity}  --max-seqs {max_seqs}  -e {evalue}")

    # MMseqs refuses to overwrite an existing resultDB, so clear leftovers
    # from any previous run on the same output directory.
    for prefix in (query_db, result_db):
        for f in glob.glob(prefix + "*"):
            os.remove(f)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    if os.path.exists(aln_file):
        os.remove(aln_file)

    subprocess.run(["mmseqs", "createdb", fasta_file, query_db], check=True)
    subprocess.run([
        "mmseqs", "search",
        query_db, db_path, result_db, tmp_dir,
        "--threads", str(threads),
        "-s", sensitivity,
        "--max-seqs", max_seqs,
        "-e", evalue,
    ], check=True)
    subprocess.run([
        "mmseqs", "convertalis",
        query_db, db_path, result_db, aln_file,
    ], check=True)

    print(f"  Alignment saved: {aln_file}")
    return aln_file


def compute_entropy_features(group):
    """
    Conservation statistics for one query's hits.

    `fident` is used AS REPORTED by MMseqs (fraction, 0-1). See the module
    docstring: rescaling to percent identity degrades performance and is
    deliberately not done. A consequence is that pct_conserved is always 0.0
    and pct_variable always 1.0.

    skew() returns NaN when the entropy values have zero variance (e.g. a
    single hit). That is intentional and handled downstream by predict.py's
    np.nan_to_num, matching the validated baseline behaviour.
    """
    identities = group["fident"].astype(float).values

    probs        = identities / (identities.sum() + 1e-9)
    entropy_vals = -probs * np.log2(probs + 1e-9)

    return {
        "mean_entropy"    : np.mean(entropy_vals),
        "variance_entropy": np.var(entropy_vals),
        "std_entropy"     : np.std(entropy_vals),
        "skew_entropy"    : skew(entropy_vals),
        "pct_conserved"   : np.sum(identities > 70) / len(identities),
        "pct_variable"    : np.sum(identities < 30) / len(identities),
    }


def _zero_feats(protein_id):
    feats = {c: 0.0 for c in ENTROPY_COLUMNS}
    feats["protein_id"] = protein_id
    return feats


def extract_entropy_features(aln_file, protein_ids=None):
    """
    Compute per-protein conservation features from the MMseqs2 alignment.

    Robust to the no-homolog case: if the alignment file is missing or empty,
    every query is zero-filled rather than crashing. Queries absent from the
    alignment are zero-filled too, so the output always covers every input
    protein and the downstream merge cannot silently drop rows.
    """
    print("\n🔹 Computing entropy features...")

    if (not os.path.exists(aln_file)) or os.path.getsize(aln_file) == 0:
        print("  No alignment hits — filling entropy features with zeros")
        ids = list(protein_ids) if protein_ids is not None else []
        df_out = pd.DataFrame([_zero_feats(pid) for pid in ids])
        print(f"  Proteins with entropy features: {len(df_out)}")
        return df_out

    df = pd.read_csv(aln_file, sep="\t", header=None).iloc[:, :12]
    df.columns = ALN_COLUMNS

    results, hit_ids, hit_counts = [], set(), []
    n_sparse = 0

    for protein, group in df.groupby("query"):
        k = len(group)
        hit_counts.append(k)
        hit_ids.add(protein)

        if k < MIN_HITS:
            results.append(_zero_feats(protein))
            n_sparse += 1
            continue

        feats = compute_entropy_features(group)
        feats["protein_id"] = protein
        results.append(feats)

    if protein_ids is not None:
        for pid in protein_ids:
            if pid not in hit_ids:
                results.append(_zero_feats(pid))

    df_out = pd.DataFrame(results)

    if hit_counts:
        print(f"  Hits per query: median={int(np.median(hit_counts))}, "
              f"min={min(hit_counts)}, max={max(hit_counts)}")
        if n_sparse:
            print(f"  {n_sparse} query(ies) had < {MIN_HITS} hits — "
                  f"entropy zero-filled")
    print(f"  Proteins with entropy features: {len(df_out)}")
    return df_out


def get_default_entropy_features(protein_ids):
    """Zero-filled entropy features, used when MMseqs2 is skipped."""
    print("\n⚠️  MMseqs2 not used — filling entropy features with zeros")
    return pd.DataFrame([_zero_feats(pid) for pid in protein_ids])
