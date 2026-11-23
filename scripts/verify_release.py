#!/usr/bin/env python3
"""
Pre-release gate for FungStructEff. Run this before pushing or deploying.

Every check here corresponds to something that has already gone wrong once, or
that would silently invalidate the published numbers if it went wrong:

  1. The model bundle predict.py expects is actually present.
  2. Its threshold is the released operating point (0.18) and carries the rule
     it was selected by.
  3. The scaler is stored as arrays, so inference does not depend on the
     scikit-learn version.
  4. The bundle is marked as trained with conservation features, which is what
     makes predict.py refuse to run without MMseqs2.
  5. No development sequence is a copy of a benchmark sequence, and no sequence
     is labelled both positive and negative. This is the leakage the audit was
     about; it must not come back.
  6. The split that defines the result is committed, not left on one machine.
  7. predict.py has no hardcoded threshold shadowing the bundle.

Exit status is 0 only if every check passes. Warnings do not fail the run.

Usage
-----
    python scripts/verify_release.py
    python scripts/verify_release.py --expect-threshold 0.18
    python scripts/verify_release.py --skip-bundle     # datasets/code only
"""

import argparse
import collections
import os
import pickle
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = "XGB_MODEL_v8_hardneg_val_homology_thr-overall.pkl"
FASTA_DIR = os.path.join("datasets", "fasta_sequences")
BENCH = "fungi_additional_dataset.fasta"
NEG = "fungi_negative_dataset.fasta"
POS = "fungi_positive_dataset.fasta"

# Frozen release composition — see check_datasets() for the derivation.
EXPECT_NEG = 5416
EXPECT_POS = 407
EXPECT_BENCH = 213
# removed.tsv holds 89 decontamination rows plus 44 validation/test proteins
# dropped for identity-only cross-partition homology. Only the 89 leave the
# dataset; the 44 are real data that split.tsv excludes from training.
EXPECT_REMOVED = 133
EXPECT_DECONTAMINATION = 89
EXPECT_SPLIT_ROWS = 5779

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def record(status, what, detail=""):
    results.append((status, what, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
    print(f"[{mark}] {what}" + (f"\n          {detail}" if detail else ""))


def read_fasta(path):
    """Return a list of (id, sequence). Duplicate IDs are preserved."""
    out, sid, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if sid is not None:
                    out.append((sid, "".join(buf)))
                sid, buf = line[1:].split()[0], []
            elif line:
                buf.append(line.upper())
    if sid is not None:
        out.append((sid, "".join(buf)))
    return out


# --------------------------------------------------------------- bundle ----

def check_bundle(expect_threshold):
    path = os.path.join(REPO, BUNDLE)
    if not os.path.exists(path):
        record(FAIL, f"{BUNDLE} present",
               "predict.py points at this file. Without it the repo does not run. "
               "Copy the release bundle in (Git LFS or a release attachment if >100 MB).")
        return
    size_mb = os.path.getsize(path) / 1e6
    record(PASS, f"{BUNDLE} present", f"{size_mb:.1f} MB")
    if size_mb > 100:
        record(WARN, "bundle over GitHub's 100 MB limit",
               "use Git LFS or attach it to the release instead of committing it")

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from fungstructeff.siamese import Siamese
        import __main__
        __main__.Siamese = Siamese
        with open(path, "rb") as fh:
            b = pickle.load(fh)
    except Exception as e:
        record(FAIL, "bundle loads", str(e)[:160])
        return

    for key in ("model_cl", "classifier"):
        record(PASS if key in b else FAIL, f"bundle has '{key}'")

    thr = b.get("threshold")
    if thr is None:
        record(FAIL, "bundle stores a threshold")
    elif abs(float(thr) - expect_threshold) < 1e-9:
        record(PASS, f"threshold == {expect_threshold}", str(thr))
    else:
        record(FAIL, f"threshold == {expect_threshold}",
               f"bundle stores {thr}. Run scripts/freeze_threshold.py")

    if b.get("threshold_rule"):
        record(PASS, "threshold_rule recorded", str(b["threshold_rule"])[:110])
    else:
        record(FAIL, "threshold_rule recorded",
               "a released threshold must carry how it was selected")

    if b.get("scaler_mean") is not None and b.get("scaler_scale") is not None:
        n = len(b["scaler_mean"])
        record(PASS, "scaler stored as arrays",
               f"{n} features, fingerprint {b.get('scaler_fingerprint', 'n/a')} "
               "— inference is scikit-learn independent")
    else:
        record(FAIL, "scaler stored as arrays",
               "bundle still relies on a pickled StandardScaler, so the serving "
               "environment must be pinned to scikit-learn 1.6.1. "
               "Run scripts/scaler_to_arrays.py")

    if b.get("uses_conservation") is True:
        record(PASS, "uses_conservation set",
               "predict.py will refuse to run without --mmseqs-db")
    else:
        record(FAIL, "uses_conservation set",
               "without this flag the MMseqs2 guard is inert and a zero-filled "
               "run can happen by accident. Run scripts/freeze_threshold.py")

    if b.get("version"):
        record(PASS, "bundle version string", str(b["version"]))


# -------------------------------------------------------------- datasets ----

def check_datasets():
    d = os.path.join(REPO, FASTA_DIR)
    paths = {name: os.path.join(d, name) for name in (BENCH, NEG, POS)}
    missing = [n for n, p in paths.items() if not os.path.exists(p)]
    if missing:
        record(FAIL, "dataset FASTA present", f"missing: {', '.join(missing)}")
        return

    bench = read_fasta(paths[BENCH])
    neg = read_fasta(paths[NEG])
    pos = read_fasta(paths[POS])

    # Checked before anything else: an empty file passes every overlap test
    # below for the wrong reason.
    empty = [n for n, e in ((BENCH, bench), (NEG, neg), (POS, pos)) if not e]
    if empty:
        record(FAIL, "dataset FASTA non-empty",
               f"no sequences in: {', '.join(empty)} — the overlap checks below "
               "would pass vacuously")
        return
    record(PASS, "dataset FASTA present",
           f"benchmark={len(bench)}  dev_neg={len(neg)}  dev_pos={len(pos)}")

    # Frozen release composition. 5505 negative structures minus 89 removals:
    # 80 copies of benchmark non-effectors, 1 copy of a benchmark effector
    # (negative_7301 == Fung-24), and 8 copies of training effectors. Positives
    # and the benchmark are untouched. These are exact counts, not a sanity
    # range: the whole point of the audit was that the published data must be
    # the data the model was trained on.
    for label, got, want in (("negative", len(neg), EXPECT_NEG),
                             ("positive", len(pos), EXPECT_POS),
                             ("benchmark", len(bench), EXPECT_BENCH)):
        if got == want:
            record(PASS, f"{label} set size == {want}")
        else:
            record(FAIL, f"{label} set size == {want}",
                   f"found {got} ({got - want:+d}). The released {label} set is "
                   f"fixed at {want}; a different count means the regeneration "
                   "did not use the same removal list as training.")

    # Duplicate IDs within a file.
    for label, entries in (("negative", neg), ("positive", pos), ("benchmark", bench)):
        ids = [i for i, _ in entries]
        dups = [k for k, v in collections.Counter(ids).items() if v > 1]
        if dups:
            record(FAIL, f"unique IDs in {label} set",
                   f"{len(dups)} repeated: {', '.join(sorted(dups)[:6])}"
                   + (" ..." if len(dups) > 6 else ""))
        else:
            record(PASS, f"unique IDs in {label} set", f"{len(ids)} records")

    # Benchmark sequences must not appear anywhere in development data.
    dev = {}
    for name, entries in (("dev_neg", neg), ("dev_pos", pos)):
        for k, v in entries:
            dev.setdefault(v, []).append(f"{name}:{k}")
    leaked = [(bid, dev[s]) for bid, s in bench if s in dev]
    if leaked:
        shown = "; ".join(f"{b} == {','.join(w)}" for b, w in sorted(leaked)[:4])
        record(FAIL, "no benchmark sequence in development data",
               f"{len(leaked)} benchmark proteins are exact copies of development "
               f"records. This is the leakage the audit removed from TRAINING; the "
               f"published FASTA still contains it. {shown}"
               + (" ..." if len(leaked) > 4 else "")
               + "\n          Fix with scripts/prepare_release_datasets.py")
    else:
        record(PASS, "no benchmark sequence in development data",
               f"checked {len(bench)} benchmark sequences against "
               f"{len(neg) + len(pos)} development records")

    # A sequence cannot be both positive and negative.
    negseq = {}
    for k, v in neg:
        negseq.setdefault(v, []).append(k)
    posseq = {}
    for k, v in pos:
        posseq.setdefault(v, []).append(k)
    both = sorted(set(negseq) & set(posseq))
    if both:
        shown = "; ".join(f"{','.join(negseq[s])}=={','.join(posseq[s])}"
                          for s in both[:4])
        record(FAIL, "no sequence labelled both positive and negative",
               f"{len(both)} conflicting sequences: {shown}"
               + (" ..." if len(both) > 4 else ""))
    else:
        record(PASS, "no sequence labelled both positive and negative")


# ----------------------------------------------------------------- split ----

def check_split():
    base = os.path.join(REPO, "splits", "homology")

    # The removal list may be the training run's own removed.tsv or the
    # reconstructed equivalent committed to the repo; either satisfies this.
    p = os.path.join(base, "removed.tsv")
    if os.path.exists(p):
        with open(p) as fh:
            n = sum(1 for _ in fh) - 1
        if n == EXPECT_REMOVED:
            record(PASS, "splits/homology/removed.tsv",
                   f"{n} rows = {EXPECT_DECONTAMINATION} decontamination + "
                   f"{EXPECT_REMOVED - EXPECT_DECONTAMINATION} partition drops")
        else:
            record(FAIL, f"splits/homology/removed.tsv has {EXPECT_REMOVED} rows",
                   f"found {n}. Expected {EXPECT_DECONTAMINATION} decontamination "
                   f"rows plus {EXPECT_REMOVED - EXPECT_DECONTAMINATION} "
                   "identity-only partition drops.")
    else:
        record(FAIL, "splits/homology/removed.tsv",
               "missing — this is make_homology_split.py's own record of what it "
               "excluded and why. It exists on both ravi and aicob; copy it in. "
               "Do not substitute a list reconstructed by sequence matching: 3 of "
               "the 8 label conflicts are not byte-identical and 44 partition "
               "drops cannot be derived at all.")

    p = os.path.join(base, "split.tsv")
    if os.path.exists(p):
        with open(p) as fh:
            n = sum(1 for _ in fh) - 1
        if n == EXPECT_SPLIT_ROWS:
            record(PASS, "splits/homology/split.tsv", f"{n} rows")
        else:
            record(FAIL, f"splits/homology/split.tsv has {EXPECT_SPLIT_ROWS} rows",
                   f"found {n}. The partition the reported metrics are computed on "
                   f"holds {EXPECT_SPLIT_ROWS} proteins "
                   "(train 4109 / val 845 / test 825).")
    else:
        record(FAIL, "splits/homology/split.tsv",
               "missing — this is the partition the reported metrics are computed "
               "on, and it cannot be reconstructed from the data. Copy it from the "
               "training machine.")


# ------------------------------------------------------------------ code ----

def check_code():
    p = os.path.join(REPO, "predict.py")
    if not os.path.exists(p):
        record(FAIL, "predict.py present")
        return
    src = open(p).read()

    if re.search(r"^THRESHOLD\s*=", src, re.M):
        record(FAIL, "no hardcoded THRESHOLD in predict.py",
               "a module-level THRESHOLD shadows the bundle's operating point — "
               "this is exactly the bug that made runs report 0.64")
    else:
        record(PASS, "no hardcoded THRESHOLD in predict.py")

    if 'bundle["threshold"]' in src or "bundle['threshold']" in src:
        record(PASS, "predict.py reads the threshold from the bundle")
    else:
        record(FAIL, "predict.py reads the threshold from the bundle")

    if BUNDLE in src:
        record(PASS, "predict.py points at the release bundle")
    else:
        record(FAIL, "predict.py points at the release bundle",
               "MODEL_PATH does not reference " + BUNDLE)

    if "--no-mmseqs" in src and "uses_conservation" in src:
        record(PASS, "MMseqs2 guard present in predict.py")
    else:
        record(FAIL, "MMseqs2 guard present in predict.py",
               "without it, omitting --mmseqs-db silently zero-fills conservation "
               "features and degrades predictions")

    if "run_manifest" in src:
        record(PASS, "predict.py writes a run manifest")
    else:
        record(WARN, "predict.py writes a run manifest",
               "results become untraceable to the threshold that produced them")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect-threshold", type=float, default=0.18)
    ap.add_argument("--skip-bundle", action="store_true",
                    help="Skip bundle checks (e.g. on a machine without the model)")
    ap.add_argument("--expect-neg", type=int, default=EXPECT_NEG,
                    help=f"Expected development negative count (default {EXPECT_NEG})")
    ap.add_argument("--expect-pos", type=int, default=EXPECT_POS,
                    help=f"Expected development positive count (default {EXPECT_POS})")
    ap.add_argument("--expect-bench", type=int, default=EXPECT_BENCH,
                    help=f"Expected benchmark count (default {EXPECT_BENCH})")
    a = ap.parse_args()

    globals()["EXPECT_NEG"] = a.expect_neg
    globals()["EXPECT_POS"] = a.expect_pos
    globals()["EXPECT_BENCH"] = a.expect_bench

    print(f"FungStructEff release verification\nrepo: {REPO}\n")
    print("-- model bundle " + "-" * 44)
    if a.skip_bundle:
        record(WARN, "bundle checks skipped", "--skip-bundle was passed")
    else:
        check_bundle(a.expect_threshold)
    print("\n-- datasets " + "-" * 48)
    check_datasets()
    print("\n-- split " + "-" * 51)
    check_split()
    print("\n-- code " + "-" * 52)
    check_code()

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    print("\n" + "=" * 60)
    print(f"{n_pass} passed, {n_warn} warnings, {n_fail} failures")
    if n_fail:
        print("\nNOT READY TO RELEASE. Failures:")
        for s, what, detail in results:
            if s == FAIL:
                print(f"  - {what}")
        print("\nDo not push or deploy until these are resolved.")
        return 1
    print("\nReady to release." + (" Review the warnings first." if n_warn else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
