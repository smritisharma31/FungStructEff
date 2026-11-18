#!/usr/bin/env python3
"""
Regenerate the released development FASTA so it matches the trained model.

The FASTA currently in the repo predates the leakage audit. It still contains
81 exact copies of benchmark sequences and 5 sequences labelled both positive
and negative, none of which were in the data v7 was trained on. Publishing it
alongside the paper would hand a reader a dataset that reproduces the leakage
the audit removed.

Two things this script does deliberately:

  Sequences come from the STRUCTURES, not from the old FASTA. The model saw
  structure-derived sequences, and the repo's own fasta_from_structures() is
  what produces them, so calling it here guarantees the published sequences are
  the ones the features were computed from. It also fixes the duplicate
  `negative_4310` record for free: there are 5505 negative structures and 5506
  old FASTA records.

  Removals come from splits/homology/removed.tsv, not from matching done here.
  removed.tsv is what make_homology_split.py actually excluded from training,
  including near-identical homologues that exact sequence matching cannot see.
  Reconstructing the list by matching would produce a dataset that is close to,
  but not the same as, what was trained on — a subtler version of the original
  bug. --derive exists for when removed.tsv is genuinely unavailable, and says
  so loudly.

Usage
-----
    python scripts/prepare_release_datasets.py --removed splits/homology/removed.tsv

    # dry run: report what would change, write nothing
    python scripts/prepare_release_datasets.py --removed ... --dry-run

    # fallback when removed.tsv cannot be recovered (NOT equivalent)
    python scripts/prepare_release_datasets.py --derive
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

STRUCT_DIR = os.path.join("datasets", "pdb_structures")
FASTA_DIR = os.path.join("datasets", "fasta_sequences")
SETS = [
    ("negative", "negative", "fungi_negative_dataset.fasta"),
    ("positive", "positive", "fungi_positive_dataset.fasta"),
    ("benchmark", "additional", "fungi_additional_dataset.fasta"),
]
ID_COLUMNS = ("id", "protein_id", "seq_id", "sequence_id", "query", "removed_id", "name")


def read_fasta(path):
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


def write_fasta(path, entries, width=60):
    with open(path, "w") as fh:
        for sid, seq in entries:
            fh.write(f">{sid}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


def extract_from_structures(subdir):
    """
    Run the repo's own extractor over a structure directory.

    fasta_from_structures() catches per-file exceptions into a notes file, so a
    missing dependency yields an empty FASTA and no traceback. That must be a
    hard failure here: silently publishing an empty or partial dataset is worse
    than publishing the stale one.
    """
    from fungstructeff.structures import fasta_from_structures
    src = os.path.join(REPO, STRUCT_DIR, subdir)
    if not os.path.isdir(src):
        sys.exit(f"❌ Structure directory not found: {src}")
    n_pdb = len([f for f in os.listdir(src) if f.endswith(".pdb")])

    tmpdir = tempfile.mkdtemp(prefix="fse_release_")
    out = os.path.join(tmpdir, f"{subdir}.fasta")
    try:
        fasta_from_structures(src, out)
        entries = read_fasta(out) if os.path.exists(out) else []
        notes_path = os.path.join(tmpdir, "sequence_extraction_notes.txt")
        notes = open(notes_path).read().splitlines() if os.path.exists(notes_path) else []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not entries:
        head = "\n      ".join(notes[:3]) if notes else "(no notes written)"
        sys.exit(f"❌ Extracted 0 sequences from {subdir}/ ({n_pdb} PDB files present).\n"
                 f"   First extraction notes:\n      {head}\n\n"
                 "   fasta_from_structures() swallows per-file errors, so this is "
                 "usually a\n   missing dependency rather than bad structures. "
                 "Check: python -c 'import Bio'\n   and install with: "
                 "pip install biopython")

    failed = [n for n in notes if "extraction failed" in n or "no polypeptide chain" in n]
    if failed:
        print(f"    ⚠️  {len(failed)} structure(s) produced no sequence; "
              f"first: {failed[0][:90]}")
    if len(entries) != n_pdb:
        print(f"    ⚠️  {n_pdb} PDB files -> {len(entries)} sequences "
              f"({n_pdb - len(entries)} missing)")
    return entries


def load_removed(path):
    """Read removed.tsv tolerantly: a bare ID list, or a TSV with an ID column."""
    if not os.path.exists(path):
        sys.exit(f"❌ {path} not found.\n"
                 "   This file is what make_homology_split.py excluded from training.\n"
                 "   Copy it from the training machine, or use --derive (not equivalent).")
    with open(path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else ","
        rows = list(csv.reader(fh, delimiter=delim))
    if not rows:
        sys.exit(f"❌ {path} is empty.")

    header = [h.strip().lower() for h in rows[0]]
    col, body = 0, rows
    if any(h in ID_COLUMNS for h in header):
        col = next(i for i, h in enumerate(header) if h in ID_COLUMNS)
        body = rows[1:]
        print(f"  removed.tsv: using column '{rows[0][col]}' (of {len(header)})")
    else:
        print(f"  removed.tsv: no recognised header; using column 1")

    ids = {r[col].strip() for r in body if r and r[col].strip()}
    print(f"  removed.tsv: {len(ids)} ids to exclude")
    return ids


def derive_removed(sets):
    """
    Fallback: exact-match removals. Measured to be wrong, in two ways.

    The authoritative list removes 89 negatives: 80 copies of benchmark
    non-effectors, 1 copy of a benchmark effector (negative_7301 == Fung-24),
    and 8 copies of training effectors. Positives are untouched, giving
    407 + 5416.

    Exact matching gets neither number right. It finds only 5 of the 8
    effector duplicates, because 3 are the same protein without being
    byte-identical: negative_5903 differs from seq_365 at one residue (99.8%),
    negative_5342 is a 173 aa truncation of the 214 aa seq_486, and
    negative_5650 contains seq_501. Homology clustering catches those; string
    equality cannot. It then also drops the 5 positives it did match, which
    training kept. Result: 5419 + 402 instead of 5416 + 407.
    """
    print("\n⚠️  --derive: computing removals by exact sequence identity.")
    print("    This is known to be wrong. It finds 5 of the 8 training-effector")
    print("    duplicates (3 are homologous, not identical) and wrongly drops the")
    print("    5 positives it does match. Expect 5419 + 402 instead of the correct")
    print("    5416 + 407. Use removed.tsv unless it is genuinely unrecoverable.")
    bench = {s for _, s in sets["benchmark"]}
    negseq, posseq = {}, {}
    for k, v in sets["negative"]:
        negseq.setdefault(v, []).append(k)
    for k, v in sets["positive"]:
        posseq.setdefault(v, []).append(k)
    ids = set()
    for s in bench:
        ids.update(negseq.get(s, []))
        ids.update(posseq.get(s, []))
    for s in set(negseq) & set(posseq):
        ids.update(negseq[s])
        ids.update(posseq[s])
    print(f"    derived {len(ids)} ids to exclude")
    return ids


def verify(kept):
    """Same invariants verify_release.py enforces."""
    ok = True

    # An empty set satisfies every invariant below for the wrong reason.
    for name in ("negative", "positive", "benchmark"):
        if not kept[name]:
            print(f"  ❌ {name} set is empty — nothing to verify, refusing to pass")
            ok = False
    if not ok:
        return False

    bench = kept["benchmark"]
    dev = {}
    for name in ("negative", "positive"):
        for k, v in kept[name]:
            dev.setdefault(v, []).append(f"{name}:{k}")
    leaked = [(b, dev[s]) for b, s in bench if s in dev]
    if leaked:
        ok = False
        print(f"  ❌ {len(leaked)} benchmark sequences still present in development data")
        for b, w in sorted(leaked)[:5]:
            print(f"       {b} == {','.join(w)}")
    else:
        print(f"  ✅ no benchmark sequence in development data")

    negseq, posseq = {}, {}
    for k, v in kept["negative"]:
        negseq.setdefault(v, []).append(k)
    for k, v in kept["positive"]:
        posseq.setdefault(v, []).append(k)
    both = sorted(set(negseq) & set(posseq))
    if both:
        ok = False
        print(f"  ❌ {len(both)} sequences labelled both positive and negative")
    else:
        print("  ✅ no sequence labelled both positive and negative")

    for name in ("negative", "positive", "benchmark"):
        ids = [k for k, _ in kept[name]]
        if len(ids) != len(set(ids)):
            ok = False
            print(f"  ❌ duplicate ids in {name} set")
    if ok:
        print("  ✅ all ids unique within each set")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--removed", default=None,
                    help="List of ids excluded from training. Defaults to "
                         "splits/homology/removed.tsv if present, otherwise "
                         "splits/homology/removed_89_reconstructed.tsv.")
    ap.add_argument("--derive", action="store_true",
                    help="Compute removals here instead of reading removed.tsv")
    ap.add_argument("--outdir", default=FASTA_DIR, help="Where to write the FASTA")
    ap.add_argument("--rewrite-benchmark", action="store_true",
                    help="Also rewrite the benchmark FASTA from its structures. Off by "
                         "default: the benchmark is an external dataset (Yuan et al.) "
                         "and altering its published sequences would make our copy "
                         "diverge from everyone else's. Discrepancies are reported "
                         "either way.")
    ap.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="Write even if verification fails (not for a release)")
    a = ap.parse_args()

    print("Regenerating development FASTA from structures\n")
    sets = {}
    for label, subdir, _ in SETS:
        sets[label] = extract_from_structures(subdir)
        print(f"  {label:10s} {len(sets[label])} sequences from {subdir}/")

    print("\nRemovals")
    if a.derive:
        removed = derive_removed(sets)
    else:
        if a.removed:
            path = a.removed if os.path.isabs(a.removed) else os.path.join(REPO, a.removed)
        else:
            base = os.path.join(REPO, "splits", "homology")
            path = next((os.path.join(base, n) for n in
                         ("removed.tsv", "removed_89_reconstructed.tsv")
                         if os.path.exists(os.path.join(base, n))),
                        os.path.join(base, "removed.tsv"))
            print(f"  using {os.path.relpath(path, REPO)}")
        removed = load_removed(path)
        if len(removed) != 89:
            print(f"  ⚠️  expected 89 removals (80 + 1 + 8), found {len(removed)}")

    kept, dropped = {}, {}
    for label, _, _ in SETS:
        if label == "benchmark":
            kept[label] = sets[label]          # the benchmark is never filtered
            dropped[label] = []
            continue
        kept[label] = [(k, v) for k, v in sets[label] if k not in removed]
        dropped[label] = [k for k, _ in sets[label] if k in removed]

    print("\nResult")
    for label, _, fname in SETS:
        old_path = os.path.join(REPO, FASTA_DIR, fname)
        old_n = len(read_fasta(old_path)) if os.path.exists(old_path) else 0
        print(f"  {label:10s} {old_n:5d} old -> {len(kept[label]):5d} new "
              f"({len(dropped[label])} removed, "
              f"{len(sets[label]) - len(kept[label]) - len(dropped[label]):+d} unaccounted)")

    # Regenerating from structures is not only a removal: for some proteins the
    # structure-derived sequence differs from the old FASTA (SEQRES vs modelled
    # coordinates, or a different chain chosen). Those differences are what the
    # model actually saw, but they change published sequences, so they are
    # reported rather than applied silently.
    changed = {}
    for label, _, fname in SETS:
        old_path = os.path.join(REPO, FASTA_DIR, fname)
        if not os.path.exists(old_path):
            continue
        old = {}
        for k, v in read_fasta(old_path):
            old.setdefault(k, v)
        diff = [(k, len(old[k]), len(v)) for k, v in sets[label]
                if k in old and old[k] != v]
        if diff:
            changed[label] = diff

    if changed:
        print("\nSequences that differ from the old FASTA (structure-derived wins)")
        for label, diff in changed.items():
            print(f"  {label:10s} {len(diff)} changed")
            for k, a_len, b_len in sorted(diff)[:4]:
                print(f"       {k:16s} {a_len}aa -> {b_len}aa")
            if len(diff) > 4:
                print(f"       ... and {len(diff) - 4} more")
        if "benchmark" in changed and not a.rewrite_benchmark:
            print(f"\n  ⚠️  {len(changed['benchmark'])} BENCHMARK sequences differ from "
                  "the published FASTA.\n"
                  "      The benchmark FASTA will NOT be rewritten: it is an external\n"
                  "      dataset and our copy should stay comparable to everyone else's.\n"
                  "      These ids are listed in sequence_discrepancies.txt. If the\n"
                  "      paper reports metrics computed from the structures, say so\n"
                  "      rather than republishing an altered benchmark.")

    unmatched = removed - {k for label in ("negative", "positive")
                           for k, _ in sets[label]}
    if unmatched:
        print(f"\n  ⚠️  {len(unmatched)} ids in removed.tsv matched no structure: "
              f"{', '.join(sorted(unmatched)[:8])}"
              + (" ..." if len(unmatched) > 8 else ""))
        print("      Check that removed.tsv and the structures come from the same run.")

    print("\nVerification")
    ok = verify(kept)

    if a.dry_run:
        print("\n(dry run — nothing written)")
        return 0 if ok else 1
    if not ok and not a.force:
        print("\n❌ Verification failed; nothing written. Fix the inputs, or --force.")
        return 1

    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(REPO, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    print()
    for label, _, fname in SETS:
        if label == "benchmark" and not a.rewrite_benchmark:
            print(f"  kept {fname} unchanged (external dataset)")
            continue
        dest = os.path.join(outdir, fname)
        if os.path.exists(dest):
            bak = dest + ".preaudit"
            if not os.path.exists(bak):
                shutil.copy2(dest, bak)
                print(f"  backed up {fname} -> {os.path.basename(bak)}")
        write_fasta(dest, kept[label])
        print(f"  wrote {fname}  ({len(kept[label])} sequences)")

    if changed:
        disc = os.path.join(outdir, "sequence_discrepancies.txt")
        with open(disc, "w") as fh:
            fh.write("# structure-derived sequence differs from the pre-audit FASTA\n")
            fh.write("# set\tid\told_len\tstructure_len\n")
            for label, diff in changed.items():
                for k, a_len, b_len in sorted(diff):
                    fh.write(f"{label}\t{k}\t{a_len}\t{b_len}\n")
        print(f"  wrote {os.path.basename(disc)}")

    man = os.path.join(outdir, "removed_from_development.txt")
    with open(man, "w") as fh:
        fh.write("# ids present as structures but excluded from the released "
                 "development FASTA\n")
        fh.write(f"# source: {'derived (exact match)' if a.derive else a.removed}\n")
        for label in ("negative", "positive"):
            for k in sorted(dropped[label]):
                fh.write(f"{label}\t{k}\n")
    print(f"  wrote {os.path.basename(man)}")
    print("\nNext: python scripts/verify_release.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
