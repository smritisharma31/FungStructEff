#!/usr/bin/env python3
"""
Independent audit of splits/homology/split.tsv.

Re-running the split is NOT the cheap option: the v7 model was trained on this
exact partition, so a new split means retraining and every reported number
changes. This verifies the existing file instead, and only a real failure here
justifies re-running.

The central check is component integrity. split.tsv carries a `component`
column, so the claim "no homology component straddles two partitions" can be
verified from the file alone, with no alignment and no MMseqs2. If that holds
and the counts reconcile, the split is sound whatever else is going on.

Usage
-----
    python audit_split.py --split splits/homology/split.tsv \
                          --removed splits/homology/removed.tsv \
                          --structures datasets/pdb_structures \
                          --split-used SPLIT_USED_v7_hardneg_val_homology.tsv
"""

import argparse
import collections
import csv
import os
import sys

# From leakage_report.txt / manifest.json, 2026-10-15 run.
EXPECT = {
    "split_rows": 5779,
    "partitions": {"train": (289, 3820), "val": (56, 789), "test": (58, 767)},
    "removed_rows": 133,
    "decontamination": 89,
    "partition_drops": 44,
    "components": 2294,
    "largest_component": 1591,
    "structures": {"negative": 5505, "positive": 407, "additional": 213},
    "cleaned": (407, 5416),
}
DUP_ID = 0.95
fails, warns = [], []


def ok(what, detail=""):
    print(f"[  ok  ] {what}" + (f"\n          {detail}" if detail else ""))


def bad(what, detail=""):
    fails.append(what)
    print(f"[ FAIL ] {what}" + (f"\n          {detail}" if detail else ""))


def warn(what, detail=""):
    warns.append(what)
    print(f"[ warn ] {what}" + (f"\n          {detail}" if detail else ""))


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/homology/split.tsv")
    ap.add_argument("--removed", default="splits/homology/removed.tsv")
    ap.add_argument("--structures", default="datasets/pdb_structures")
    ap.add_argument("--split-used", default=None,
                    help="SPLIT_USED_*.tsv the training run recorded, if present")
    a = ap.parse_args()

    for p in (a.split, a.removed):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    rows = read_tsv(a.split)
    cols = list(rows[0]) if rows else []
    print(f"split.tsv columns: {cols}\n")

    # ---------------------------------------------------------- 1. shape ----
    print("-- shape " + "-" * 50)
    ids = [r["protein_id"] for r in rows]
    if len(rows) == EXPECT["split_rows"]:
        ok(f"row count == {EXPECT['split_rows']}")
    else:
        bad(f"row count == {EXPECT['split_rows']}", f"found {len(rows)}")
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dups:
        bad("protein_id unique", f"{len(dups)} repeated: {sorted(dups)[:5]}")
    else:
        ok("protein_id unique", f"{len(ids)} ids")

    # ------------------------------------------------- 2. partition sizes ----
    print("\n-- partition composition " + "-" * 34)
    tab = collections.Counter((r["partition"], r["label"]) for r in rows)
    for part, (eff, non) in EXPECT["partitions"].items():
        g_eff = tab.get((part, "1"), 0)
        g_non = tab.get((part, "0"), 0)
        if (g_eff, g_non) == (eff, non):
            ok(f"{part}: {eff} effectors + {non} non-effectors = {eff + non}")
        else:
            bad(f"{part}: expected {eff} eff + {non} non",
                f"found {g_eff} eff + {g_non} non")
    extra = {p for p, _ in tab} - set(EXPECT["partitions"])
    if extra:
        bad("only train/val/test partitions", f"also found {sorted(extra)}")

    # label vs id-prefix consistency
    wrong = [r["protein_id"] for r in rows
             if (r["protein_id"].startswith("seq_") and r["label"] != "1")
             or (r["protein_id"].startswith("negative_") and r["label"] != "0")]
    if wrong:
        bad("label matches id prefix", f"{len(wrong)} mismatched: {wrong[:5]}")
    else:
        ok("label matches id prefix")

    # --------------------------------------- 3. THE CENTRAL CHECK: components
    print("\n-- component integrity " + "-" * 36)
    if "component" not in cols:
        warn("no component column", "cannot verify the partition guarantee here")
    else:
        comp_parts = collections.defaultdict(set)
        comp_rows = collections.Counter()
        for r in rows:
            comp_parts[r["component"]].add(r["partition"])
            comp_rows[r["component"]] += 1
        straddling = {c: p for c, p in comp_parts.items() if len(p) > 1}
        if straddling:
            bad(f"no component spans partitions",
                f"{len(straddling)} do. This is the guarantee the paper rests "
                f"on. Examples: " +
                "; ".join(f"{c} in {sorted(p)}" for c, p in
                          list(straddling.items())[:4]))
        else:
            ok("no component spans partitions",
               f"{len(comp_parts)} components, each wholly inside one partition "
               "— this is the leakage guarantee, verified from the file alone")

        biggest, n = comp_rows.most_common(1)[0]
        print(f"          largest component present: {biggest} with {n} proteins "
              f"(report says {EXPECT['largest_component']})")
        if n > EXPECT["largest_component"]:
            bad("largest component <= reported",
                f"found {n} > {EXPECT['largest_component']}")

        if "component_size" in cols:
            # stated size is pre-drop, so present count must not EXCEED it
            over = [(r["component"], comp_rows[r["component"]], r["component_size"])
                    for r in rows
                    if comp_rows[r["component"]] > int(r["component_size"])]
            if over:
                bad("component_size >= rows present",
                    f"{len(over)} components have more rows than their stated "
                    f"size: {over[:3]}")
            else:
                ok("component_size consistent with rows present")

    # ----------------------------------------------- 4. removed.tsv split ----
    print("\n-- removed.tsv " + "-" * 44)
    rem = read_tsv(a.removed)
    if len(rem) == EXPECT["removed_rows"]:
        ok(f"removed.tsv row count == {EXPECT['removed_rows']}")
    else:
        bad(f"removed.tsv row count == {EXPECT['removed_rows']}", f"found {len(rem)}")

    def ident(r):
        try:
            return float(r.get("identity", "") or -1)
        except ValueError:
            return -1.0
    decon = {r["protein_id"] for r in rem if ident(r) >= DUP_ID}
    drops = {r["protein_id"] for r in rem if 0 <= ident(r) < DUP_ID}
    unknown = {r["protein_id"] for r in rem if ident(r) < 0}
    print(f"          by identity: {len(decon)} >= {DUP_ID}, {len(drops)} below, "
          f"{len(unknown)} unparseable")
    if len(decon) == EXPECT["decontamination"] and len(drops) == EXPECT["partition_drops"]:
        ok(f"{EXPECT['decontamination']} decontamination + "
           f"{EXPECT['partition_drops']} partition drops")
    else:
        bad(f"{EXPECT['decontamination']} decontamination + "
            f"{EXPECT['partition_drops']} partition drops",
            f"found {len(decon)} + {len(drops)}")
    if "reason" in (rem[0] if rem else {}):
        print("          reasons: " + "; ".join(
            f"{v}x {k[:46]}" for k, v in
            collections.Counter(r["reason"] for r in rem).most_common()))

    idset = set(ids)
    leaked_decon = sorted(decon & idset)
    leaked_drops = sorted(drops & idset)
    if leaked_decon:
        bad("decontaminated ids absent from split",
            f"{len(leaked_decon)} present: {leaked_decon[:5]}")
    else:
        ok("decontaminated ids absent from split")
    if leaked_drops:
        bad("dropped ids absent from split",
            f"{len(leaked_drops)} present: {leaked_drops[:5]}")
    else:
        ok("dropped ids absent from split")

    # effectors among the drops — the report says 4
    eff_drops = sorted(i for i in drops if i.startswith("seq_"))
    print(f"          effectors among the 44 drops: {len(eff_drops)} "
          f"(report says 4) {eff_drops[:6]}")
    if len(eff_drops) != 4:
        warn("4 effectors among the drops", f"found {len(eff_drops)}")

    # ------------------------------------------------ 5. against structures --
    print("\n-- structures " + "-" * 45)
    have = {}
    for sub in ("negative", "positive", "additional"):
        d = os.path.join(a.structures, sub)
        have[sub] = {f[:-4] for f in os.listdir(d) if f.endswith(".pdb")} \
            if os.path.isdir(d) else set()
        n, want = len(have[sub]), EXPECT["structures"][sub]
        (ok if n == want else bad)(f"{sub} structures == {want}", f"found {n}")

    dev = have["negative"] | have["positive"]
    orphan = sorted(idset - dev)
    if orphan:
        bad("every split id has a structure",
            f"{len(orphan)} without one: {orphan[:5]}")
    else:
        ok("every split id has a structure")

    # closure
    print("\n-- accounting " + "-" * 45)
    total = len(dev)
    expect_split = total - len(decon) - len(drops)
    if expect_split == len(rows):
        ok(f"{total} structures - {len(decon)} decontamination - {len(drops)} "
           f"drops = {len(rows)} split rows")
    else:
        bad("accounting closes",
            f"{total} - {len(decon)} - {len(drops)} = {expect_split}, but split "
            f"has {len(rows)}. Gap of {len(rows) - expect_split}.")

    cleaned_eff = len(have["positive"]) - len({i for i in decon if i.startswith("seq_")})
    cleaned_non = len(have["negative"]) - len({i for i in decon if i.startswith("negative_")})
    we, wn = EXPECT["cleaned"]
    if (cleaned_eff, cleaned_non) == (we, wn):
        ok(f"cleaned development set == {we} + {wn}")
    else:
        bad(f"cleaned development set == {we} + {wn}",
            f"computes to {cleaned_eff} + {cleaned_non}")

    # -------------------------------------------- 6. what training used -----
    if a.split_used and os.path.exists(a.split_used):
        print("\n-- SPLIT_USED vs split.tsv " + "-" * 32)
        used = read_tsv(a.split_used)
        u = {r["protein_id"]: r["partition"] for r in used}
        if len(u) == len(idset) and set(u) == idset:
            ok("same protein set", f"{len(u)} ids")
        else:
            bad("same protein set",
                f"split has {len(idset)}, SPLIT_USED has {len(u)}; "
                f"only-in-split {len(idset - set(u))}, "
                f"only-in-used {len(set(u) - idset)}")
        mism = [i for r in rows if (i := r["protein_id"]) in u
                and u[i] != r["partition"]]
        if mism:
            bad("same partition assignment",
                f"{len(mism)} differ: {mism[:5]} — the model was not trained on "
                "the split committed to the repo")
        else:
            ok("same partition assignment",
               "the model was trained on this exact partition")

    print("\n" + "=" * 60)
    print(f"{len(fails)} failures, {len(warns)} warnings")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  - {f}")
        print("\nA failure in component integrity or partition composition would "
              "justify\nre-running the split. Count mismatches alone usually mean "
              "the expectations\nin this script are stale, not that the split is "
              "wrong — check against\nleakage_report.txt before retraining "
              "anything.")
        return 1
    print("\nSplit verified. No reason to re-run it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
