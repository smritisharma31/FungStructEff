#!/usr/bin/env python3
"""
Lift a FungStructEff bundle's StandardScaler into plain numpy arrays.

WHY
---
The bundle pickles a scikit-learn StandardScaler fitted under 1.6.1. Loading it
under a different version raises InconsistentVersionWarning, which is why the
serving environment has been pinned to 1.6.1. That pin is a standing liability:
every future deployment has to find an old scikit-learn, and the pin silently
becomes load-bearing for reproducing the published numbers.

But a StandardScaler is only an affine map over two fitted arrays:

    X_scaled = (X - mean_) / scale_

and predict.py's ONLY use of scikit-learn anywhere in the inference path is the
single `scaler.transform(X_struct)` call. So storing those two arrays removes
scikit-learn from inference entirely. The version question then does not need
answering, because nothing unpickles a scikit-learn object any more.

RUN THIS IN THE TRAINING ENVIRONMENT (scikit-learn 1.6.1)
---------------------------------------------------------
That is the environment the scaler was fitted in, so its unpickled arrays are
authoritative by construction. Extracting there sidesteps the question of
whether a newer scikit-learn unpickles the object faithfully.

    python scripts/scaler_to_arrays.py XGB_MODEL_v7_hardneg_val_homology.pkl

The script refuses to run under a version other than the one that fitted the
scaler unless you pass --allow-version-mismatch, and it verifies the array
transform reproduces scaler.transform() to machine precision before writing
anything. If that check fails, nothing is written.

Usage
-----
    python scripts/scaler_to_arrays.py BUNDLE [--out OUT] [--allow-version-mismatch]
"""

import argparse
import hashlib
import os
import pickle
import shutil
import sys
import warnings

import numpy as np


def load_bundle(path):
    """Load a bundle, making Siamese importable the way training pickled it.

    Returns (bundle, fitted_under). `fitted_under` is the scikit-learn version
    the pickled estimators were fitted under, or None when it matches this
    environment.

    The version is read from InconsistentVersionWarning rather than from the
    estimator: scikit-learn's __setstate__ consumes the `_sklearn_version`
    entry while unpickling, so it is NOT readable as an attribute afterwards.
    Reading it off the object always yields None and would make this check
    silently vacuous.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from fungstructeff.siamese import Siamese
    except ImportError:
        sys.exit("❌ Cannot import fungstructeff.siamese — run this from the repo root.")
    import __main__
    __main__.Siamese = Siamese

    fitted_under = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        for w in caught:
            orig = getattr(w.message, "original_sklearn_version", None)
            if orig:
                fitted_under = orig
    return bundle, fitted_under


def versions():
    out = {}
    for mod, key in (("sklearn", "sklearn"), ("xgboost", "xgboost"),
                     ("torch", "torch"), ("numpy", "numpy")):
        try:
            out[key] = __import__(mod).__version__
        except Exception:
            out[key] = None
    return out


def digest(*arrays):
    """Stable hash of the fitted arrays, so two environments can be compared."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--out", default=None, help="Write here instead of in place")
    ap.add_argument("--allow-version-mismatch", action="store_true",
                    help="Extract even if scikit-learn differs from the fitting version")
    a = ap.parse_args()

    vers = versions()
    print("Environment")
    for k, v in vers.items():
        print(f"  {k:<10}: {v}")

    bundle, fitted_under = load_bundle(a.bundle)
    if not isinstance(bundle, dict) or "scaler" not in bundle:
        sys.exit("❌ Not a FungStructEff bundle (no 'scaler' key).")

    scaler = bundle["scaler"]
    print(f"\nScaler class : {type(scaler).__module__}.{type(scaler).__name__}")
    if fitted_under:
        print(f"Fitted under : scikit-learn {fitted_under}  "
              f"(this environment: {vers['sklearn']})  — MISMATCH")
    else:
        print(f"Fitted under : scikit-learn {vers['sklearn']} — matches this environment")

    if fitted_under and not a.allow_version_mismatch:
        sys.exit(
            f"\n❌ This environment has scikit-learn {vers['sklearn']}, but the estimators\n"
            f"   were fitted under {fitted_under}. Run this in the training environment so\n"
            f"   the extracted arrays are authoritative by construction.\n\n"
            f"   If you need to extract here anyway, run the script in BOTH environments\n"
            f"   and compare the fingerprint it prints. Identical fingerprints mean the\n"
            f"   arrays are unaffected by the version difference; then re-run with\n"
            f"   --allow-version-mismatch.\n")

    # --- read the fitted state -------------------------------------------------
    n_feat = int(getattr(scaler, "n_features_in_", 0)) or None
    with_mean = bool(getattr(scaler, "with_mean", True))
    with_std = bool(getattr(scaler, "with_std", True))

    # Normalise to an unconditional (X - mean) / scale, so inference has no branches.
    mean = np.asarray(scaler.mean_, dtype=np.float64) if with_mean and getattr(scaler, "mean_", None) is not None \
        else np.zeros(n_feat, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64) if with_std and getattr(scaler, "scale_", None) is not None \
        else np.ones(n_feat, dtype=np.float64)

    if n_feat is None:
        n_feat = int(mean.shape[0])

    print(f"\nFitted state")
    print(f"  n_features_in_ : {n_feat}")
    print(f"  with_mean      : {with_mean}")
    print(f"  with_std       : {with_std}")
    print(f"  mean range     : [{mean.min():.6g}, {mean.max():.6g}]")
    print(f"  scale range    : [{scale.min():.6g}, {scale.max():.6g}]")

    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        sys.exit("❌ Non-finite values in the fitted arrays — refusing to write.")
    if np.any(scale == 0):
        sys.exit("❌ Zero in scale_ — would divide by zero. Refusing to write.")

    fp = digest(mean, scale)
    print(f"  fingerprint    : {fp}   (compare this across environments)")

    # --- prove the arrays reproduce the object --------------------------------
    # Random data plus edge cases: zeros, the mean itself, and large magnitudes.
    rng = np.random.default_rng(0)
    tests = [
        ("random normal",    rng.normal(size=(512, n_feat)) * 10.0),
        ("zeros",            np.zeros((4, n_feat))),
        ("the mean vector",  np.tile(mean, (4, 1))),
        ("large magnitude",  rng.normal(size=(64, n_feat)) * 1e6),
    ]
    print("\nVerifying (X - mean) / scale  ==  scaler.transform(X)")
    worst = 0.0
    for name, X in tests:
        ref = scaler.transform(X.copy())
        got = (X - mean) / scale
        d = float(np.max(np.abs(ref - got)))
        worst = max(worst, d)
        print(f"  {name:<16}: max abs diff {d:.3e}")
    if worst > 1e-9:
        sys.exit(f"\n❌ Array transform does not reproduce the scaler (worst {worst:.3e}).\n"
                 f"   Nothing written. Do not deploy the array path.")
    print(f"  → exact to {worst:.3e}; the arrays are a faithful replacement")

    # --- write ----------------------------------------------------------------
    bundle["scaler_mean"] = mean
    bundle["scaler_scale"] = scale
    bundle["scaler_n_features"] = n_feat
    bundle["scaler_fingerprint"] = fp
    bundle["scaler_fitted_sklearn"] = fitted_under
    bundle["scaler_extracted_under"] = vers["sklearn"]
    bundle["bundle_versions"] = vers
    # The original object is KEPT, so an older predict.py still works.

    out = a.out or a.bundle
    if a.out is None:
        backup = a.bundle + ".prescaler.bak"
        if os.path.exists(backup):
            sys.exit(f"❌ {backup} exists — move it aside, or use --out.")
        shutil.copy2(a.bundle, backup)
        print(f"\nBackup: {backup}")

    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh)
    os.replace(tmp, out)

    check, _ = load_bundle(out)
    assert check["scaler_fingerprint"] == fp, "fingerprint did not persist"
    assert np.array_equal(check["scaler_mean"], mean), "mean did not persist"
    assert np.array_equal(check["scaler_scale"], scale), "scale did not persist"
    assert abs(float(check["threshold"]) - float(bundle["threshold"])) < 1e-12, \
        "threshold changed unexpectedly"

    print(f"\n✅ Wrote {out} and verified it reloads")
    print(f"   scaler_mean/scaler_scale  : {n_feat} features, fingerprint {fp}")
    print(f"   threshold                 : {check['threshold']} (unchanged)")
    print("\nNext: apply the predict.py patch so inference uses the arrays, then")
    print("re-run the benchmark and confirm TP 48 / FP 39 / TN 121 / FN 5.")


if __name__ == "__main__":
    main()
