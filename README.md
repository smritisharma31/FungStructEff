# FungStructEff

**Structure-based prediction of fungal effector proteins.**

FungStructEff predicts fungal effectors from **3D protein structures**. It
combines residue-level, structural, and biological features extracted from the
structure with MMseqs2 conservation entropy and ESM2-650M sequence embeddings,
transforms them through a contrastive (Siamese) network, and classifies with
XGBoost.

> **Input is a structure, not a sequence.** Supply pre-computed PDB/mmCIF
> models — **AlphaFold structures are strongly recommended**. The model's
> structural features were derived from AlphaFold-quality models, and
> prediction accuracy degrades noticeably on structures from other predictors.
> Sequences are extracted from the structures automatically.

## Benchmark

Evaluated on an independent dataset of **213 proteins (53 effectors, 160
non-effectors)**:

| Method | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC |
|--------|----|----|----|----|----------|-----------|--------|----|-----|
| **FungStructEff** | **43** | 143 | 17 | **10** | 87.3% | 0.717 | **0.811** | **0.761** | 0.678 |
| PPEPFinder | 36 | **152** | **8** | 17 | 88.3% | **0.818** | 0.679 | 0.742 | 0.672 |
| EffectorP 3.0 | 36 | 148 | 12 | 17 | 86.4% | 0.750 | 0.679 | 0.713 | 0.620 |
| EffectorP-fungi 3.0 | 33 | **159** | **1** | 20 | **90.1%** | **0.971** | 0.623 | 0.759 | **0.726** |
| DeepRedEff | 32 | 91 | 69 | 21 | 57.7% | 0.317 | 0.604 | 0.416 | 0.149 |

- FungStructEff detects the **most true effectors (TP = 43)** — 7 more than
  PPEPFinder and EffectorP 3.0.
- It achieves the **highest recall (0.811)** and the **best F1 (0.761)**, with
  **ROC-AUC = 0.8979**.
- EffectorP-fungi 3.0 is the most conservative (FP = 1) but misses 20 effectors.
- DeepRedEff performs poorly on this dataset (69 false positives).

FungStructEff is tuned for **recall** — the default threshold of 0.64 favours
finding candidate effectors over minimising false positives, which suits
discovery workflows. Raise `--threshold` if you need higher precision.

## Pipeline

```
PDB / mmCIF structures
 ├─ STEP 0  ingest structures + extract sequences
 ├─ STEP 1  structures ready (no folding)
 ├─ STEP 2  residue-level features        (DSSP)
 ├─ STEP 3  aggregate to protein level
 ├─ STEP 4  structural features
 ├─ STEP 5  conservation entropy          (MMseqs2 vs UniRef50)
 ├─ STEP 6  biological features
 ├─ STEP 7  merge + scale
 ├─ STEP 8  ESM2-650M embeddings
 └─ contrastive transform → XGBoost → prediction
```

## Repository layout

```
FungStructEff/
├── predict.py
├── check_setup.py              # dependency checker
├── fungstructeff/              # pipeline package
│   ├── structures.py           # PDB/CIF ingestion + sequence extraction
│   ├── fasta_utils.py          # sequence sanitization
│   ├── siamese.py              # contrastive network (needed to load the model)
│   ├── merge.py
│   └── features/               # residue, graph, struct, entropy, bio
├── XGB_MODEL_v5_hardneg.pkl    # trained model bundle
├── examples/Fung-14.pdb        # example structure (known effector)
├── requirements.txt
├── tools/                      # MMseqs2 binary      (gitignored)
├── mmseqs_db/                  # UniRef50 DB ~150 GB (gitignored)
└── results/                    # outputs             (gitignored)
```

---

## System requirements

| Component  | Requirement |
|------------|-------------|
| OS         | Linux (x86-64). Tested on Ubuntu. |
| Python     | 3.10 (required — cu128 torch wheels are cp310) |
| GPU        | NVIDIA CUDA GPU recommended (ESM2-650M ≈ 2.5 GB VRAM). CPU works but is slower. |
| GPU driver | CUDA 12.8-capable. **Blackwell cards (RTX 50-series, sm_120) require a cu128 torch build.** |
| CPU        | AVX2 recommended (standard MMseqs2 binary; an SSE4.1 build exists otherwise) |
| RAM        | 16 GB+ (MMseqs2 search against UniRef50 splits automatically if memory is limited) |
| Disk       | ~155 GB+ free: UniRef50 MMseqs DB (~150 GB) + ESM2-650M weights (~2.5 GB) + outputs |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/smritisharma31/FungStructEff.git
cd FungStructEff
```

All subsequent commands assume you are inside this directory.

### 2. Environment (Python 3.10)

```bash
conda create -n FungStructEff python=3.10 -y
conda activate FungStructEff
```

### 3. PyTorch — install FIRST

Blackwell GPUs (RTX 50-series, sm_120) need a CUDA 12.8 build; cu121/cu124
wheels lack sm_120 kernels and fail at runtime even though
`cuda.is_available()` returns True.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
# if the stable cu128 wheel lacks sm_120:
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

python -c "import torch; print(torch.cuda.get_arch_list()); \
x=torch.randn(2000,2000,device='cuda'); print('ok', tuple((x@x).shape))"
```

For other GPUs, pick the build matching your driver: https://pytorch.org/get-started/locally/

### 4. Python dependencies

```bash
pip install -r requirements.txt
```

### 5. External tools

**DSSP** (required — residue-level features):

```bash
conda install -c salilab dssp -y      # provides `mkdssp`
```

**MMseqs2** (required for conservation features):

```bash
mkdir -p tools
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs-linux-avx2.tar.gz -C tools/
rm mmseqs-linux-avx2.tar.gz
ls tools/mmseqs/bin/mmseqs

echo "export PATH=$(pwd)/tools/mmseqs/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
mmseqs version
```

> "Illegal instruction" means your CPU lacks AVX2 — use the
> `mmseqs-linux-sse41` build instead.

### 6. UniRef50 database (~150 GB, several hours)

```bash
mkdir -p mmseqs_db
mmseqs databases UniRef50 mmseqs_db/uniref50 mmseqs_db/tmp --threads 8
mmseqs createindex mmseqs_db/uniref50 mmseqs_db/tmp_index -s 7.5 --threads 8
rm -rf mmseqs_db/tmp mmseqs_db/tmp_index

#Confirm the index with:
ls -lh mmseqs_db/uniref50.idx*
```

---

## Verify the installation

**1. Check dependencies:**

```bash
python check_setup.py
```

Reports PASS / WARN / FAIL for Python, packages, torch + CUDA (including
Blackwell sm_120), DSSP, MMseqs2, the model bundle, and the UniRef50 database.
Exits non-zero if a hard requirement is missing.

**2. Run the bundled example structure:**

```bash
python predict.py --pdb-dir examples/ --outdir results_example/ --no-mmseqs
```

`examples/Fung-14.pdb` is the structure of a characterised fungal effector.
Similarly, `examples/Non-10.pdb` is an example of non-effector.
A successful run prints STEP 0–8 and writes
`results_example/fungstructeff_predictions.csv` with two rows, 
predicting `Fung-14` as effector and `Non-10` as non-effector.

This is a positive control as well as a smoke test: the example protein is a
known effector, so it should be predicted as **Effector** at the default
threshold. If it is not, check that DSSP is working and that the structure
parsed correctly before running on your own data.

> Note: `--no-mmseqs` zero-fills the conservation features, so probabilities
> from this quick check differ slightly from a full run with `--mmseqs-db`.

---

## Usage

```bash
# with conservation features (recommended):
python predict.py --pdb-dir alphafold_pdbs/ --outdir results/ \
    --mmseqs-db mmseqs_db/uniref50

```

| Option        | Default   | Description                                        |
|---------------|-----------|----------------------------------------------------|
| `--pdb-dir`   | (required)| Directory (or single file) of PDB/mmCIF structures |
| `--outdir`    | `results` | Output directory                                   |
| `--mmseqs-db` | none      | Path to the MMseqs2 UniRef50 DB prefix             |
| `--threads`   | `8`       | Threads for MMseqs2                                |
| `--threshold` | `0.64`     | Prediction threshold (high-recall default)        |
| `--no-mmseqs` | off       | Skip MMseqs2; zero-fill entropy features           |

### Input structures

- `.pdb`, `.ent`, `.cif`, `.mmcif` are accepted; mmCIF is converted automatically.
- Protein IDs come from filenames. AlphaFold DB names are unwrapped:
  `AF-Q9XYZ1-F1-model_0_v4.pdb` → `Q9XYZ1`.
- IDs are sanitized and de-duplicated, so unusual filenames cannot collide.
  `pdb_structures/structure_manifest.tsv` maps every ID to its source file.
- **Experimental PDBs:** sequences are taken from `SEQRES` when present, so
  unresolved regions are not silently deleted. Chain breaks and multi-chain
  files are recorded in `sequence_extraction_notes.txt`.

### Outputs

| File | Contents |
|------|----------|
| `fungstructeff_predictions.csv` | `protein_id`, `probability`, `prediction`, `label` |
| `pdb_structures/` | normalized structures + `structure_manifest.tsv` |
| `from_structures.fasta` | sequences extracted from the structures |
| `cleaned_input.fasta` | sanitized sequences used downstream |
| `sequence_extraction_notes.txt` | chain breaks, multi-chain, extraction warnings |

The default threshold is **0.64**, matching the high-recall objective the model
was tuned for. Override with `--threshold`.

---

## Notes

- ESM2-650M weights (~2.5 GB) download on first run and are cached afterwards.
- `scikit-learn` is pinned to **1.6.1** to match the `StandardScaler` stored in
  the model bundle; other versions may warn or alter feature scaling.
- FungStructEff does **not** fold sequences. Generate structures beforehand
  with AlphaFold or ColabFold (https://alphafoldserver.com/).
