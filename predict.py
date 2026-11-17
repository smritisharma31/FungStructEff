"""
FungStructEff — Fungal Effector Prediction Tool
============================================
Input is protein STRUCTURES (PDB / mmCIF) — AlphaFold models recommended.
Sequences are extracted from the structures automatically.

Usage:
    python predict.py --pdb-dir structures/ --outdir results/ --mmseqs-db /path/to/uniref50

An MMseqs2 database is required: the model was trained with conservation
features and running without them degrades predictions. --no-mmseqs zero-fills
them for smoke tests only.

The decision threshold comes from the model bundle. It was selected on the
validation partition and should not normally be overridden.
"""

import os
import sys
import gc
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm

from fungstructeff.fasta_utils      import prepare_fasta
from fungstructeff.structures        import prepare_structures, fasta_from_structures
from fungstructeff.features.residue  import extract_residue_features
from fungstructeff.features.graph    import aggregate_to_protein
from fungstructeff.features.struct   import extract_struct_features
from fungstructeff.features.entropy  import (
    run_mmseqs2,
    extract_entropy_features,
    get_default_entropy_features
)
from fungstructeff.features.bio      import extract_bio_features
from fungstructeff.merge             import merge_all_features, TRAINING_COLUMNS
from fungstructeff.siamese           import Siamese

MODEL_PATH = os.path.join(os.path.dirname(__file__), "XGB_MODEL_v7_hardneg_val_homology.pkl")
ESM_LAYER  = 33


def parse_args():
    parser = argparse.ArgumentParser(
        description="FungStructEff: Fungal Effector Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --pdb-dir alphafold_pdbs/ --outdir results/ --mmseqs-db /path/to/uniref50
  python predict.py --pdb-dir alphafold_pdbs/ --outdir results/ --no-mmseqs   # degraded, smoke test only
        """
    )
    parser.add_argument("--pdb-dir",   required=True,               help="Directory (or file) of PDB/CIF structures. AlphaFold models recommended.")
    parser.add_argument("--outdir",    default="results",           help="Output directory")
    parser.add_argument("--mmseqs-db", default=None,                help="Path to MMseqs2 UniRef50 database. Required unless --no-mmseqs.")
    parser.add_argument("--threads",   default=8,      type=int,    help="Threads for MMseqs2")
    parser.add_argument("--threshold", default=None,  type=float,
                        help="Override the decision threshold. Default: the operating point "
                             "stored in the model bundle, selected on the validation "
                             "partition. Only change this if you have your own validation "
                             "data to select on.")
    parser.add_argument("--no-mmseqs", action="store_true",
                        help="Run without MMseqs2 conservation features, zero-filling them. "
                             "The model was trained WITH these features, so predictions will "
                             "be degraded. Use only for a quick smoke test, never to produce "
                             "results you intend to report.")
    return parser.parse_args()


def generate_esm_embeddings(fasta_file, device):
    print("\n🔹 Loading ESM2-650M...")
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(device)
    model_esm.eval()
    batch_converter = alphabet.get_batch_converter()
    print("✅ ESM2-650M loaded")

    records  = list(SeqIO.parse(fasta_file, "fasta"))
    esm_rows = []

    print(f"\n🔹 Generating ESM2 embeddings for {len(records)} sequences...")
    for record in tqdm(records):
        pid = record.id.strip()
        # Input FASTA was already sanitized by prepare_fasta at ingestion, so
        # sequences are tokenizer-safe here. The 1022 cap is ESM2's hard
        # architectural limit (1024 positions minus BOS/EOS) and must stay.
        seq = str(record.seq)[:1022]
        batch = [("protein", seq)]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            output = model_esm(tokens, repr_layers=[ESM_LAYER])
        emb = output["representations"][ESM_LAYER][0, 1:-1].mean(0).cpu().numpy()
        esm_rows.append((pid, emb))

    del model_esm
    gc.collect()

    X_esm    = np.array([x[1] for x in esm_rows])
    ids      = [x[0] for x in esm_rows]
    esm_cols = [f"esm_{i}" for i in range(X_esm.shape[1])]
    df_esm   = pd.DataFrame(X_esm, columns=esm_cols)
    df_esm["protein_id"] = ids

    print(f"  ESM shape: {X_esm.shape}")
    return df_esm, esm_cols


def load_model(model_path):
    print(f"\n🔹 Loading StructEff model...")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Make sure XGB_MODEL_v7_hardneg_val_homology.pkl is in the same "
              "directory as predict.py")
        sys.exit(1)

    # Register Siamese class for pickle
    try:
        from fungstructeff.siamese import Siamese
        import __main__
        __main__.Siamese = Siamese
    except ImportError:
        pass

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    print("✅ Model loaded")
    print(f"   Version  : {bundle.get('version', 'unknown')}")
    print(f"   Threshold: {bundle['threshold']}")
    return bundle


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print("  FungStructEff — Fungal Effector Prediction Tool v1.0")
    print(f"{'='*55}")
    print(f"Device   : {device}")
    print(f"Input    : {args.pdb_dir}  (structures)")
    print(f"Output   : {args.outdir}")
    print(f"Threshold: {'model bundle default' if args.threshold is None else args.threshold}")
    print(f"MMseqs2  : {'disabled' if args.no_mmseqs else args.mmseqs_db or 'not provided'}")

    os.makedirs(args.outdir, exist_ok=True)
    pdb_dir   = os.path.join(args.outdir, "pdb_structures")
    mmseq_dir = os.path.join(args.outdir, "mmseqs_results")

    # Load model
    bundle     = load_model(MODEL_PATH)
    model_cl   = bundle["model_cl"]
    scaler     = bundle.get("scaler")
    classifier = bundle["classifier"]
    model_cl.eval()

    # Prefer the bundle's extracted scaler arrays over the pickled scikit-learn
    # object; see the scaling step below for why.
    scaler_mean  = bundle.get("scaler_mean")
    scaler_scale = bundle.get("scaler_scale")
    if scaler_mean is not None and scaler_scale is not None:
        scaler_mean  = np.asarray(scaler_mean,  dtype=np.float64)
        scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        print(f"   Scaler   : arrays, {scaler_mean.shape[0]} features "
              f"(fingerprint {bundle.get('scaler_fingerprint', 'n/a')}) — no sklearn needed")
    elif scaler is not None:
        scaler_mean = scaler_scale = None
        print("   Scaler   : pickled sklearn object — version-sensitive; run "
              "scripts/scaler_to_arrays.py")
    else:
        sys.exit("❌ Bundle has neither scaler arrays nor a scaler object.")

    # Decision threshold: the bundle's stored operating point unless the user
    # overrides it. The stored value was selected on the validation partition,
    # which is the only set it may be selected on — picking a threshold on the
    # test or benchmark data would make the reported metrics optimistic.
    if args.threshold is None:
        threshold = float(bundle["threshold"])
        rule      = bundle.get("threshold_rule", "selected on validation partition")
        print(f"   Active threshold: {threshold:.4g}  ({rule})")
    else:
        threshold = float(args.threshold)
        print(f"   Active threshold: {threshold:.4g}  "
              f"(user override; bundle stores {float(bundle['threshold']):.4g})")

    # Conservation features: the model was fitted with them, so running without
    # them silently zero-fills 6 of the 1571 features and degrades the scores
    # (measured on the 213-protein benchmark: ROC-AUC 0.886 vs 0.907). Refuse
    # rather than produce quietly wrong numbers.
    if bundle.get("uses_conservation", True) and not args.no_mmseqs and args.mmseqs_db is None:
        sys.exit(
            "\n❌ No MMseqs2 database given (--mmseqs-db).\n"
            "   This model was trained with conservation (entropy) features. Running\n"
            "   without them zero-fills those features and degrades predictions —\n"
            "   on the benchmark set, ROC-AUC drops from 0.907 to 0.886.\n\n"
            "   Either:\n"
            "     --mmseqs-db /path/to/uniref50   compute the features properly\n"
            "     --no-mmseqs                     accept degraded predictions (smoke tests only)\n"
        )

    # Step 0: obtain structures + a matching cleaned FASTA.
    #   --pdb-dir : use pre-computed structures (AlphaFold recommended) and
    #               derive sequences from them, so features and embeddings
    #               describe exactly the residues that are in the structures.
    #   --fasta   : fold the sequences locally with ESMFold.
    print(f"\n{'='*55}")
    print("STEP 0: Ingesting structures and extracting sequences")
    print(f"{'='*55}")
    clean_fasta = os.path.join(args.outdir, "cleaned_input.fasta")

    prepare_structures(args.pdb_dir, pdb_dir)
    raw_fasta = os.path.join(args.outdir, "from_structures.fasta")
    fasta_from_structures(pdb_dir, raw_fasta)

    report = prepare_fasta(raw_fasta, clean_fasta)
    print(f"  Read     : {report['n_in']} sequences")
    print(f"  Kept     : {report['n_out']}")
    print(f"  Modified : {len(report['modified'])} (non-standard chars cleaned)")
    print(f"  Dropped  : {len(report['dropped'])}")
    if report["dropped"]:
        drop_path = os.path.join(args.outdir, "dropped_sequences.txt")
        with open(drop_path, "w") as fh:
            for pid, reason in report["dropped"]:
                fh.write(f"{pid}\t{reason}\n")
        print(f"  Dropped IDs written to: {drop_path}")
    if report["n_out"] == 0:
        print("\n[FATAL] No usable sequences. Check the input.")
        sys.exit(1)

    # Get protein IDs (from the cleaned FASTA, so they match every later step)
    records     = list(SeqIO.parse(clean_fasta, "fasta"))
    protein_ids = [r.id.strip() for r in records]
    print(f"\n📋 Proteins to predict: {len(protein_ids)}")

    # Step 1: structures
    print(f"\n{'='*55}")
    print("STEP 1: Structures ready (no folding — user-supplied models)")
    print(f"{'='*55}")

    # Step 2: Residue features
    print(f"\n{'='*55}")
    print("STEP 2: Extracting residue-level features")
    print(f"{'='*55}")
    df_residue = extract_residue_features(pdb_dir)
    if df_residue is None or df_residue.empty or "rsa" not in df_residue.columns:
        print("\n[FATAL] No parseable residue features were produced.")
        print("        DSSP could not read the supplied structures.")
        print("        Check results/pdb_structures/ and failed_structures.txt.")
        sys.exit(1)

    # Step 3: Aggregate
    print(f"\n{'='*55}")
    print("STEP 3: Aggregating to protein level")
    print(f"{'='*55}")
    df_graph = aggregate_to_protein(df_residue)

    # Step 4: Structural features
    print(f"\n{'='*55}")
    print("STEP 4: Extracting structural features")
    print(f"{'='*55}")
    df_struct = extract_struct_features(pdb_dir)

    # Step 5: Entropy features
    print(f"\n{'='*55}")
    print("STEP 5: Conservation features")
    print(f"{'='*55}")
    if args.no_mmseqs or args.mmseqs_db is None:
        print("⚠️  Conservation features ZERO-FILLED (--no-mmseqs). The model was "
              "trained with\n    these features; predictions below are degraded and "
              "must not be reported.")
        conservation = False
        df_entropy = get_default_entropy_features(protein_ids)
    else:
        conservation = True
        aln_file   = run_mmseqs2(clean_fasta, args.mmseqs_db, mmseq_dir, args.threads)
        df_entropy = extract_entropy_features(aln_file, protein_ids)

    # Step 6: Bio features
    print(f"\n{'='*55}")
    print("STEP 6: Extracting biological features")
    print(f"{'='*55}")
    df_bio = extract_bio_features(pdb_dir)

    # Step 7: Merge
    print(f"\n{'='*55}")
    print("STEP 7: Merging all features")
    print(f"{'='*55}")
    df_features = merge_all_features(df_graph, df_struct, df_entropy, df_bio, protein_ids)

    # Scale structural features.
    #
    # A StandardScaler is just (X - mean) / scale over two fitted arrays. When
    # the bundle carries those arrays we apply them directly, which keeps
    # scikit-learn out of the inference path altogether: nothing here unpickles
    # a scikit-learn object, so the serving environment's version cannot change
    # the numbers and need not be pinned. scripts/scaler_to_arrays.py writes
    # the arrays, having first proved they reproduce scaler.transform() exactly.
    #
    # Older bundles have only the pickled scaler, so that path still works.
    X_struct = np.nan_to_num(df_features[TRAINING_COLUMNS].values)
    if scaler_mean is not None:
        if X_struct.shape[1] != scaler_mean.shape[0]:
            print(f"\n[FATAL] Structural feature count {X_struct.shape[1]} does not match "
                  f"the scaler's {scaler_mean.shape[0]}.")
            sys.exit(1)
        X_struct = (X_struct - scaler_mean) / scaler_scale
    else:
        X_struct = scaler.transform(X_struct)
    struct_cols = [f"struct_{i}" for i in range(X_struct.shape[1])]

    # Step 8: ESM2 embeddings
    print(f"\n{'='*55}")
    print("STEP 8: Generating ESM2 embeddings")
    print(f"{'='*55}")
    df_esm, esm_cols = generate_esm_embeddings(clean_fasta, device)

    # Merge struct + ESM
    df_struct_df = pd.DataFrame(X_struct, columns=struct_cols)
    df_struct_df["protein_id"] = df_features["protein_id"].values

    df_merged    = df_struct_df.merge(df_esm, on="protein_id", how="left").dropna()
    X            = np.concatenate([
        df_merged[struct_cols].values,
        df_merged[esm_cols].values
    ], axis=1)

    print(f"\n  Final feature dimension: {X.shape}")

    # Safety check: feature count must match what the model was trained on.
    # Catches feature-pipeline drift before it reaches the classifier and
    # produces silently-wrong probabilities.
    expected_input = bundle.get("input_dim", 1315)
    if X.shape[1] != expected_input:
        print(f"\n[FATAL] Feature dimension mismatch: built {X.shape[1]}, "
              f"model expects {expected_input}.")
        print("        A feature step changed shape — predictions would be invalid.")
        sys.exit(1)

    # Contrastive transform
    print("\n🔹 Applying contrastive transform...")
    model_cl = model_cl.to(device)
    model_cl.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        X_emb    = model_cl(X_tensor).cpu().numpy()

    X_final = np.concatenate([X, X_emb], axis=1)
    print(f"  Transformed shape: {X_final.shape}")

    expected_final = bundle.get("final_dim", 1571)
    if X_final.shape[1] != expected_final:
        print(f"\n[FATAL] Transformed dimension mismatch: got {X_final.shape[1]}, "
              f"model expects {expected_final}.")
        sys.exit(1)

    # Predict
    print(f"\n🔹 Predicting with threshold={threshold}...")
    y_prob = classifier.predict_proba(X_final)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Save results
    df_results = pd.DataFrame({
        "protein_id" : df_merged["protein_id"].values,
        "probability": np.round(y_prob, 4),
        "prediction" : y_pred,
        "label"      : ["Effector" if p == 1 else "Non-effector" for p in y_pred]
    })

    output_path = os.path.join(args.outdir, "fungstructeff_predictions.csv")
    df_results.to_csv(output_path, index=False)

    # Run manifest: a predictions file should never be separable from the
    # threshold and the feature set that produced it.
    manifest = {
        "model_bundle"         : os.path.basename(MODEL_PATH),
        "model_version"        : bundle.get("version", "unknown"),
        "threshold"            : threshold,
        "threshold_source"     : "model bundle" if args.threshold is None else "user override",
        "threshold_rule"       : bundle.get("threshold_rule", ""),
        "conservation_features": "computed" if conservation else "ZERO-FILLED",
        "mmseqs_db"            : args.mmseqs_db or "",
        "struct_seed"          : os.environ.get("FUNGSTRUCTEFF_SEED", "42"),
        "n_proteins"           : int(len(df_results)),
        "n_predicted_effector" : int(y_pred.sum()),
    }
    manifest_path = os.path.join(args.outdir, "run_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print("  FUNGSTRUCTEFF PREDICTION COMPLETE")
    print(f"{'='*55}")
    print(f"Total proteins          : {len(df_results)}")
    print(f"Predicted effectors     : {y_pred.sum()}")
    print(f"Predicted non-effectors : {(y_pred==0).sum()}")
    print(f"Threshold               : {threshold:.4g}")
    if not conservation:
        print("⚠️  Conservation features were ZERO-FILLED — do not report these numbers.")
    print(f"\n✅ Results saved: {output_path}")
    print(f"✅ Run manifest  : {manifest_path}")

    print("\nTop effector predictions:")
    top = df_results[df_results["prediction"]==1].sort_values(
        "probability", ascending=False
    ).head(10)
    print(top[["protein_id","probability","label"]].to_string(index=False))

    print("\n🔥 PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
