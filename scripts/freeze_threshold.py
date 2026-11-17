#!/usr/bin/env python3
"""
Write the final operating point into a FungStructEff model bundle.

The v7 bundle was saved with the threshold the training script happened to
select (0.05). The released model uses 0.18, chosen on the held-out validation
partition, where maximum F1 and maximum MCC independently selected the same
grid point. This script writes that value in, records how it was chosen, and
marks the bundle as having been trained with conservation features so
predict.py can refuse to run without them.

Nothing about the model is changed: the contrastive network, the scaler and
the XGBoost classifier are rewritten byte-for-byte as loaded. Only metadata
moves.

Usage
-----
    python scripts/freeze_threshold.py XGB_MODEL_v7_hardneg_val_homology.pkl

    # write to a different file instead of in place
    python scripts/freeze_threshold.py in.pkl --out XGB_MODEL_v7_release.pkl

    # a different threshold, if you ever reselect on validation
    python scripts/freeze_threshold.py in.pkl --threshold 0.22 --rule "max F1 on validation"

An in-place run keeps the original as <name>.pkl.bak and refuses to clobber an
existing backup, so the pre-release bundle is always recoverable.
"""

import argparse
import os
import pickle
import shutil
import sys

THRESHOLD = 0.18
RULE = ("max F1 and max MCC (concordant) on the held-out validation partition "
        "of the homology-controlled split; grid np.arange(0.01, 1.00, 0.01)")


def load_bundle(path):
    """Load a bundle, making the Siamese class importable the way training saved it."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from fungstructeff.siamese import Siamese
    except ImportError:
        sys.exit("❌ Cannot import fungstructeff.siamese — run this from the repo root.")
    import __main__
    __main__.Siamese = Siamese          # training pickled it from __main__
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="Model bundle to update")
    ap.add_argument("--out", default=None,
                    help="Write here instead of in place")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"Operating point to store (default {THRESHOLD})")
    ap.add_argument("--rule", default=RULE,
                    help="How the threshold was selected — stored as provenance")
    a = ap.parse_args()

    if not 0.0 < a.threshold < 1.0:
        sys.exit(f"❌ Threshold must be strictly between 0 and 1, got {a.threshold}")

    bundle = load_bundle(a.bundle)
    if not isinstance(bundle, dict):
        sys.exit(f"❌ Expected a dict bundle, got {type(bundle).__name__}")
    for key in ("model_cl", "scaler", "classifier"):
        if key not in bundle:
            sys.exit(f"❌ Bundle is missing '{key}' — is this a FungStructEff model?")

    old = bundle.get("threshold")
    print(f"Bundle        : {a.bundle}")
    print(f"Version       : {bundle.get('version', 'unknown')}")
    print(f"Threshold     : {old}  ->  {a.threshold}")
    if bundle.get("threshold_rule"):
        print(f"Previous rule : {bundle['threshold_rule']}")

    bundle["threshold"] = float(a.threshold)
    bundle["threshold_rule"] = a.rule
    bundle["uses_conservation"] = True   # trained with MMseqs2 entropy features

    out = a.out or a.bundle
    if a.out is None:
        backup = a.bundle + ".bak"
        if os.path.exists(backup):
            sys.exit(f"❌ {backup} already exists — move it aside first, or use --out.")
        shutil.copy2(a.bundle, backup)
        print(f"Backup        : {backup}")

    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh)
    os.replace(tmp, out)

    # Reload and confirm, so a corrupt write cannot pass silently.
    check = load_bundle(out)
    assert abs(float(check["threshold"]) - a.threshold) < 1e-12, "threshold did not persist"
    assert check.get("uses_conservation") is True, "conservation flag did not persist"
    for key in ("model_cl", "scaler", "classifier"):
        assert key in check, f"{key} missing after write"

    print(f"\n✅ Wrote {out} and verified it reloads")
    print(f"   threshold        : {check['threshold']}")
    print(f"   threshold_rule   : {check['threshold_rule']}")
    print(f"   uses_conservation: {check['uses_conservation']}")
    print("\nNext: re-run the benchmark and confirm TP 48 / FP 39 / TN 121 / FN 5.")


if __name__ == "__main__":
    main()
