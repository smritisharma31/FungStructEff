#!/usr/bin/env python3
"""
check_setup.py — verify FungStructEff dependencies before running a prediction.

Run from the repo root:
    python check_setup.py

Reports PASS / WARN / FAIL for Python version, required packages, torch+CUDA
(including Blackwell sm_120), DSSP, MMseqs2, and the model bundle.
Exits non-zero if a hard requirement is missing, so it can be used in CI.
WARN items are optional or contextual (MMseqs2 is only needed without
--no-mmseqs; a GPU is recommended but CPU works).
"""
import importlib
import os
import shutil
import sys

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(level, label, detail=""):
    results.append((level, label, detail))


# ---- Python version --------------------------------------------------
v = sys.version_info
if v.major == 3 and v.minor == 10:
    record(PASS, "Python 3.10", f"{v.major}.{v.minor}.{v.micro}")
else:
    record(FAIL, "Python 3.10 required",
           f"found {v.major}.{v.minor}.{v.micro} — cu128 torch wheels are cp310")


# ---- Required Python packages ---------------------------------------
required = {
    "torch": "torch",
    "esm": "fair-esm",
    "Bio": "biopython",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "scipy": "scipy",
    "pandas": "pandas",
    "numpy": "numpy",
    "tqdm": "tqdm",
}
for mod, pipname in required.items():
    try:
        m = importlib.import_module(mod)
        record(PASS, f"import {mod}", str(getattr(m, "__version__", "?")))
    except Exception as e:
        record(FAIL, f"import {mod}", f"missing — pip install {pipname} ({e})")


# ---- scikit-learn: only needed if the bundle lacks scaler arrays ------
# A bundle processed by scripts/scaler_to_arrays.py carries the scaler as
# plain numpy arrays, so inference applies (X - mean) / scale directly and
# never unpickles a scikit-learn estimator. The version is then irrelevant.
# Only a bundle still relying on the pickled object needs the 1.6.1 pin.
try:
    import pickle

    import sklearn

    # Resolved locally: `model_path` is not defined until later in this file.
    _bundle = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "XGB_MODEL_v7_hardneg_val_homology.pkl")
    uses_arrays = False
    if os.path.exists(_bundle):
        try:
            from fungstructeff.siamese import Siamese
            import __main__
            __main__.Siamese = Siamese
            with open(_bundle, "rb") as _fh:
                _b = pickle.load(_fh)
            uses_arrays = _b.get("scaler_mean") is not None
            del _b
        except Exception as _e:
            record(WARN, "could not inspect bundle for scaler arrays", str(_e)[:80])

    if uses_arrays:
        record(PASS, "scaler stored as arrays",
               f"scikit-learn {sklearn.__version__} not used at inference")
    elif sklearn.__version__ == "1.6.1":
        record(PASS, "scikit-learn == 1.6.1", sklearn.__version__)
    else:
        record(WARN, "scikit-learn version",
               f"{sklearn.__version__}; bundle still uses a pickled scaler fitted "
               "under 1.6.1. Run scripts/scaler_to_arrays.py to drop the pin")
except Exception:
    pass


# ---- FungStructEff package imports ----------------------------------
for mod in ("fungstructeff.structures",
            "fungstructeff.fasta_utils",
            "fungstructeff.siamese",
            "fungstructeff.merge"):
    try:
        importlib.import_module(mod)
        record(PASS, f"import {mod}")
    except Exception as e:
        record(FAIL, f"import {mod}", f"{e}")


# ---- Torch / CUDA / Blackwell sm_120 --------------------------------
try:
    import torch
    if torch.cuda.is_available():
        try:
            arches = torch.cuda.get_arch_list()
            name = torch.cuda.get_device_name(0)
            x = torch.randn(64, 64, device="cuda")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            record(PASS, "CUDA GPU usable", f"{name} | arches={arches}")
            cap = torch.cuda.get_device_capability(0)
            cap_tag = f"sm_{cap[0]}{cap[1]}"
            if cap_tag == "sm_120" and "sm_120" not in arches:
                record(FAIL, "Blackwell sm_120 kernels",
                       "GPU is sm_120 but torch lacks sm_120 — install the cu128 build")
        except Exception as e:
            record(FAIL, "CUDA kernel launch", f"GPU present but compute failed: {e}")
    else:
        record(WARN, "CUDA GPU",
               "not available — CPU works but ESM2 embeddings will be slow")
except Exception:
    pass


# ---- External binaries ----------------------------------------------
dssp = shutil.which("mkdssp") or shutil.which("dssp")
if dssp:
    record(PASS, "DSSP (mkdssp/dssp)", dssp)
else:
    record(FAIL, "DSSP (mkdssp/dssp)",
           "not found — required for residue features "
           "(conda install -c salilab dssp)")

if shutil.which("mmseqs"):
    record(PASS, "mmseqs on PATH", shutil.which("mmseqs"))
else:
    record(WARN, "mmseqs on PATH",
           "not found — needed for conservation features "
           "(omit only with --no-mmseqs)")


# ---- Model bundle ----------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(here, "XGB_MODEL_v7_hardneg_val_homology.pkl")
if os.path.exists(model_path):
    record(PASS, "model bundle present",
           f"{os.path.basename(model_path)} ({os.path.getsize(model_path)/1e6:.1f} MB)")
else:
    record(FAIL, "model bundle present",
           "XGB_MODEL_v7_hardneg_val_homology.pkl not found next to this script")


# ---- UniRef50 database (optional) ------------------------------------
db_prefix = os.path.join(here, "mmseqs_db", "uniref50")
if os.path.exists(db_prefix) or os.path.exists(db_prefix + ".dbtype"):
    record(PASS, "UniRef50 MMseqs DB", db_prefix)
else:
    record(WARN, "UniRef50 MMseqs DB",
           "not found at mmseqs_db/uniref50 — build it or use --no-mmseqs")


# ---- Report ----------------------------------------------------------
icon = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌"}
print("\nFungStructEff dependency check")
print("=" * 62)
for level, label, detail in results:
    line = f"{icon[level]} {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
print("=" * 62)

n_fail = sum(1 for r in results if r[0] == FAIL)
n_warn = sum(1 for r in results if r[0] == WARN)

if n_fail:
    print(f"\n{n_fail} hard requirement(s) missing. Fix the ❌ items above.")
    sys.exit(1)
if n_warn:
    print(f"\nAll hard requirements met. {n_warn} optional warning(s).")
    sys.exit(0)
print("\nAll checks passed. Ready to run predict.py.")
sys.exit(0)
