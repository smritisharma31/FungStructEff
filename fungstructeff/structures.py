"""
fungstructeff/structures.py
Ingest protein structures (PDB / mmCIF) — the ONLY input to FungStructEff.

The model's structural features were derived from AlphaFold-quality
structures, so predictions are made from user-supplied models rather than
from sequences folded on the fly.

Sequences are extracted FROM the structures, so sequence-based features
(ESM2 embeddings, MMseqs2 conservation) always describe the same molecule
as the structural features.

Contract:
    prepare_structures(src, pdb_dir) -> {protein_id: pdb_path}
    fasta_from_structures(...)       -> path to a FASTA with clean, unique IDs
"""
import os
import re
import glob

# DSSP requires a CRYST1 record before the coordinate section. AlphaFold PDBs
# begin with MODEL, and CIF->PDB conversions often omit CRYST1 entirely, so
# every incoming structure is normalised.
_CRYST1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1"

_PDB_EXT = (".pdb", ".ent")
_CIF_EXT = (".cif", ".mmcif")

# Minimum residues for a chain to be considered a real polypeptide.
MIN_CHAIN_LEN = 5

# 3-letter -> 1-letter, including common modified residues seen in
# experimental structures (MSE = selenomethionine is very common).
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P", "SEP": "S",
    "TPO": "T", "PTR": "Y", "CSO": "C", "CME": "C", "MLY": "K",
    "UNK": "X",
}


# ----------------------------------------------------------------------
# ID handling — clean, filesystem-safe, collision-free
# ----------------------------------------------------------------------

def sanitize_id(raw):
    """
    Make a FASTA/filesystem-safe protein ID:
      - convert whitespace and special characters to underscores
      - collapse repeats, trim, and guarantee a non-empty result
    Whitespace is converted rather than truncated at, so a filename like
    "my protein (copy)" becomes "my_protein_copy" and not "my".
    """
    s = re.sub(r"[^A-Za-z0-9._-]", "_", str(raw).strip())
    s = re.sub(r"_{2,}", "_", s).strip("_.-")
    return s or "structure"


def protein_id_from_filename(path):
    """
    Derive a protein ID from a structure filename.

    AlphaFold DB naming is unwrapped to the accession:
        AF-Q9XYZ1-F1-model_v4.pdb -> Q9XYZ1
    Anything else uses the sanitized filename stem.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("AF-"):
        parts = stem.split("-")
        if len(parts) >= 2 and parts[1]:
            return sanitize_id(parts[1])
    return sanitize_id(stem)


def _unique_id(candidate, taken):
    """Append _2, _3, ... until the ID is unused. Prevents name conflicts."""
    if candidate not in taken:
        taken.add(candidate)
        return candidate
    i = 2
    while f"{candidate}_{i}" in taken:
        i += 1
    new = f"{candidate}_{i}"
    taken.add(new)
    return new


# ----------------------------------------------------------------------
# Structure normalisation
# ----------------------------------------------------------------------

def _sanitize_pdb_text(pdb_text):
    """Drop blank/unsupported records; guarantee CRYST1 before MODEL/ATOM."""
    lines = [ln for ln in pdb_text.splitlines() if ln.strip() != ""]
    lines = [ln for ln in lines if not ln.startswith("PARENT")]
    if not any(ln.startswith("CRYST1") for ln in lines):
        for i, ln in enumerate(lines):
            if ln.startswith(("MODEL", "ATOM", "HETATM")):
                lines.insert(i, _CRYST1)
                break
        else:
            lines.insert(0, _CRYST1)
    return "\n".join(lines) + "\n"


def _cif_to_pdb(cif_path, out_path):
    """Convert mmCIF to PDB via BioPython."""
    from Bio.PDB import MMCIFParser, PDBIO
    parser = MMCIFParser(QUIET=True)
    sid = os.path.splitext(os.path.basename(cif_path))[0]
    structure = parser.get_structure(sid, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))
    return out_path


def prepare_structures(src, pdb_dir):
    """
    Collect structures from `src` (a directory or a single file) into
    `pdb_dir` as sanitized {protein_id}.pdb with unique IDs.

    Returns {protein_id: pdb_path}. Bad files are reported and listed in
    `pdb_dir/failed_structures.txt`; a `structure_manifest.tsv` records the
    protein_id -> original filename mapping.
    """
    os.makedirs(pdb_dir, exist_ok=True)

    if os.path.isdir(src):
        files = []
        for ext in _PDB_EXT + _CIF_EXT:
            files.extend(glob.glob(os.path.join(src, "*" + ext)))
            files.extend(glob.glob(os.path.join(src, "*" + ext.upper())))
        files = sorted(set(files))
    elif os.path.isfile(src):
        files = [src]
    else:
        raise FileNotFoundError(f"Structure path not found: {src}")

    if not files:
        raise FileNotFoundError(f"No .pdb/.cif structures found in: {src}")

    print(f"\n🔹 Ingesting {len(files)} structure file(s)...")

    results, failed, manifest = {}, [], []
    taken = set()

    for path in files:
        base = os.path.basename(path)
        try:
            pid = _unique_id(protein_id_from_filename(path), taken)
            dest = os.path.join(pdb_dir, f"{pid}.pdb")

            if path.lower().endswith(_CIF_EXT):
                tmp = dest + ".tmp"
                _cif_to_pdb(path, tmp)
                with open(tmp) as fh:
                    text = fh.read()
                os.remove(tmp)
            else:
                with open(path) as fh:
                    text = fh.read()

            if not any(ln.startswith(("ATOM", "HETATM"))
                       for ln in text.splitlines()):
                raise ValueError("no ATOM records found")

            with open(dest, "w") as out:
                out.write(_sanitize_pdb_text(text))

            results[pid] = dest
            manifest.append(f"{pid}\t{base}")
        except Exception as e:
            print(f"[FAILED] {base}: {e}")
            failed.append(f"{base}\t{e}")

    print(f"✅ Structures ready: {len(results)} | Failed: {len(failed)}")

    with open(os.path.join(pdb_dir, "structure_manifest.tsv"), "w") as fh:
        fh.write("protein_id\tsource_file\n")
        fh.write("\n".join(manifest) + ("\n" if manifest else ""))

    if failed:
        fp = os.path.join(pdb_dir, "failed_structures.txt")
        with open(fp, "w") as fh:
            fh.write("\n".join(failed) + "\n")
        print(f"   Failures written to: {fp}")

    return results


# ----------------------------------------------------------------------
# Sequence extraction (experimental-PDB aware)
# ----------------------------------------------------------------------

def _seqres_by_chain(pdb_path):
    """
    Read SEQRES records -> {chain_id: one-letter sequence}.

    SEQRES is the full construct sequence, so for EXPERIMENTAL structures it
    includes residues that were never resolved. That is the true sequence of
    the protein and is what sequence-based features should use.
    Returns {} when no SEQRES records are present (e.g. AlphaFold models).
    """
    chains = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("SEQRES"):
                continue
            chain = line[11:12].strip() or "A"
            for res in line[19:].split():
                chains.setdefault(chain, []).append(
                    _THREE_TO_ONE.get(res.upper(), "X")
                )
    return {c: "".join(v) for c, v in chains.items()}


def _coord_chains(pdb_path):
    """
    Build per-chain sequences from modelled coordinates.

    Returns {chain_id: (sequence, n_fragments)}. n_fragments > 1 means the
    chain has unmodelled gaps (breaks), which is normal for experimental
    structures and never happens in AlphaFold models.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    sid = os.path.splitext(os.path.basename(pdb_path))[0]
    structure = parser.get_structure(sid, pdb_path)

    out = {}
    for model in structure:
        for chain in model:
            frags = [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
            frags = [f for f in frags if f]
            if not frags:
                continue
            out[chain.id] = ("".join(frags), len(frags))
        break  # first model only
    return out


def fasta_from_structures(pdb_dir, out_fasta):
    """
    Write one sequence per structure with clean, unique FASTA headers.

    Sequence source, in order of preference:
      1. SEQRES records  — the complete construct sequence. Used for
         experimental PDBs so unresolved regions are not silently deleted.
      2. Modelled coordinates — used for AlphaFold/predicted models, which
         have no SEQRES and no gaps.

    Chain selection: the longest polypeptide chain. Gaps and multi-chain
    files are reported so nothing is dropped silently.

    Returns the FASTA path written.
    """
    print("\n🔹 Extracting sequences from structures...")
    n_written = 0
    notes = []
    taken = set()

    with open(out_fasta, "w") as out:
        for path in sorted(glob.glob(os.path.join(pdb_dir, "*.pdb"))):
            pid = sanitize_id(os.path.splitext(os.path.basename(path))[0])
            try:
                coord = _coord_chains(path)
                coord = {c: v for c, v in coord.items()
                         if len(v[0]) >= MIN_CHAIN_LEN}
                if not coord:
                    notes.append(f"{pid}\tno polypeptide chain >= {MIN_CHAIN_LEN} aa")
                    continue

                # longest modelled chain decides which chain we describe
                chain_id = max(coord, key=lambda c: len(coord[c][0]))
                coord_seq, n_frag = coord[chain_id]

                seqres = _seqres_by_chain(path)
                if seqres.get(chain_id) and len(seqres[chain_id]) >= len(coord_seq):
                    seq = seqres[chain_id]
                    source = "SEQRES"
                else:
                    seq = coord_seq
                    source = "coordinates"

                if len(coord) > 1:
                    notes.append(
                        f"{pid}\tmulti-chain ({len(coord)} chains); used chain {chain_id}")
                if n_frag > 1:
                    notes.append(
                        f"{pid}\t{n_frag} modelled fragments (chain breaks); "
                        f"sequence from {source}")

                header = _unique_id(pid, taken)
                out.write(f">{header}\n{seq}\n")
                n_written += 1
            except Exception as e:
                notes.append(f"{pid}\textraction failed: {e}")

    print(f"  Sequences written: {n_written} -> {out_fasta}")
    if notes:
        np_ = os.path.join(os.path.dirname(out_fasta) or ".",
                           "sequence_extraction_notes.txt")
        with open(np_, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        print(f"  {len(notes)} note(s) written to: {np_}")
    return out_fasta
