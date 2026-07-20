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
rm -rf mmseqs_db/tmp
```

---

## Verify the installation

```bash
python check_setup.py
```

Reports PASS / WARN / FAIL for Python, packages, torch + CUDA (including
Blackwell sm_120), DSSP, MMseqs2, the model bundle, and the UniRef50 database.
Exits non-zero if a hard requirement is missing.

---

## Usage

```bash
# with conservation features (recommended):
python predict.py --pdb-dir alphafold_pdbs/ --outdir results/ \
    --mmseqs-db mmseqs_db/uniref50

# fast run without MMseqs2 (entropy features zero-filled):
python predict.py --pdb-dir alphafold_pdbs/ --outdir results/ --no-mmseqs
```

| Option        | Default   | Description                                        |
|---------------|-----------|----------------------------------------------------|
| `--pdb-dir`   | (required)| Directory (or single file) of PDB/mmCIF structures |
| `--outdir`    | `results` | Output directory                                   |
| `--mmseqs-db` | none      | Path to the MMseqs2 UniRef50 DB prefix             |
| `--threads`   | `8`       | Threads for MMseqs2                                |
| `--threshold` | `0.2`     | Prediction threshold (high-recall default)         |
| `--no-mmseqs` | off       | Skip MMseqs2; zero-fill entropy features           |

### Input structures

- `.pdb`, `.ent`, `.cif`, `.mmcif` are accepted; mmCIF is converted automatically.
- Protein IDs come from filenames. AlphaFold DB names are unwrapped:
  `AF-Q9XYZ1-F1-model_v4.pdb` → `Q9XYZ1`.
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

The default threshold is **0.2**, matching the high-recall objective the model
was tuned for. Override with `--threshold`.

---

## Notes

- ESM2-650M weights (~2.5 GB) download on first run and are cached afterwards.
- `scikit-learn` is pinned to **1.6.1** to match the `StandardScaler` stored in
  the model bundle; other versions may warn or alter feature scaling.
- FungStructEff does **not** fold sequences. Generate structures beforehand
  with AlphaFold (e.g. AlphaFold DB or ColabFold).
