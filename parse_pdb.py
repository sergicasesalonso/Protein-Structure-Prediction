"""
parse_pdb.py
============
Parses .pdb.gz files from RCSB to extract:
  - Amino acid sequence (1-letter codes)
  - Cα (alpha-carbon) 3D coordinates
  - Pairwise Cα distance matrix

Works with the RCSB divided archive structure:
  pdb_files/
    ab/pdb1abc.ent.gz
    cd/pdb2xyz.ent.gz
    ...
Or a flat directory:
  pdb_files/
    1abc.pdb.gz
    2xyz.pdb.gz
"""

import gzip
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── Amino acid mappings ──────────────────────────────────────────────────────

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# ─── Core parsing functions ───────────────────────────────────────────────────

def parse_pdb_gz(filepath, min_len=50, max_len=500):
    """
    Parse a single .pdb.gz file.

    Returns a list of dicts, one per valid chain:
        {
          'pdb_id':   str,
          'chain':    str,
          'sequence': str,            # 1-letter amino acid codes
          'coords':   np.ndarray,     # (N, 3)  Cα coordinates in Å
          'dist_mat': np.ndarray,     # (N, N)  pairwise distances
        }
    """
    results = []
    pdb_id  = Path(filepath).stem.replace('pdb', '').replace('.ent', '')

    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        with opener(filepath, 'rt') as f:
            lines = f.readlines()
    except Exception:
        return results

    # Collect CA atoms per chain
    chains = {}
    for line in lines:
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        if line[0:4] == 'HETATM':
            continue                      # skip ligands, waters, etc.

        atom_name = line[12:16].strip()
        if atom_name != 'CA':
            continue

        res_name = line[17:20].strip()
        if res_name not in AA_3TO1:
            continue                      # skip non-standard residues

        chain    = line[21]
        res_seq  = line[22:27].strip()   # residue number + insertion code
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        if chain not in chains:
            chains[chain] = {}

        # Use (res_seq) as key so duplicate alt-locs are overwritten
        if res_seq not in chains[chain]:
            chains[chain][res_seq] = (AA_3TO1[res_name], np.array([x, y, z], dtype=np.float32))

    # Build results for each chain
    for chain_id, residues in chains.items():
        if not residues:
            continue

        # Sort residues by residue number (handles insertion codes lexicographically)
        sorted_res = sorted(residues.items(), key=lambda x: x[0])
        seq    = ''.join(aa    for _, (aa,  _)    in sorted_res)
        coords = np.array([coord for _, (_, coord) in sorted_res], dtype=np.float32)

        if not (min_len <= len(seq) <= max_len):
            continue
        if len(seq) != len(coords):
            continue

        dist_mat = compute_distance_matrix(coords)

        results.append({
            'pdb_id':   pdb_id.upper(),
            'chain':    chain_id,
            'sequence': seq,
            'coords':   coords,
            'dist_mat': dist_mat,
        })

    return results


def compute_distance_matrix(coords):
    """
    Compute pairwise Euclidean distances between Cα atoms.

    Args:
        coords: (N, 3) numpy array of Cα coordinates

    Returns:
        dist: (N, N) numpy array of distances in Å
    """
    # Vectorised: ||x_i - x_j||
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))                  # (N, N)
    return dist.astype(np.float32)


# ─── Directory-level processing ───────────────────────────────────────────────

def process_pdb_directory(pdb_dir, min_len=50, max_len=500, max_samples=None, cache_file=None):
    """
    Walk a directory (flat or RCSB divided) and parse all .pdb.gz / .ent.gz files.

    Args:
        pdb_dir:     path to directory containing PDB files
        min_len:     minimum sequence length to keep
        max_len:     maximum sequence length to keep
        max_samples: stop after this many valid chains (None = process all)
        cache_file:  if given, save/load from this pickle file

    Returns:
        dataset: list of dicts (see parse_pdb_gz)
    """
    # Try loading cache
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached dataset from '{cache_file}'...")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} proteins.")
        return dataset

    pdb_dir = Path(pdb_dir)
    gz_files = sorted(list(pdb_dir.rglob('*.gz')) + list(pdb_dir.rglob('*.ent')))
    print(f"Found {len(gz_files)} PDB files in '{pdb_dir}'")

    dataset = []
    failed  = 0

    for gz_file in tqdm(gz_files, desc='Parsing PDB files'):
        if max_samples and len(dataset) >= max_samples:
            break
        try:
            results = parse_pdb_gz(gz_file, min_len=min_len, max_len=max_len)
            dataset.extend(results)
        except Exception:
            failed += 1

    print(f"\nParsed {len(dataset)} valid protein chains  |  {failed} files failed")

    # Save cache
    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved dataset to '{cache_file}'")

    return dataset


# ─── Quick sanity check ───────────────────────────────────────────────────────

def print_dataset_stats(dataset):
    lengths = [len(d['sequence']) for d in dataset]
    print(f"\n{'='*40}")
    print(f"Dataset Statistics")
    print(f"{'='*40}")
    print(f"  Total chains : {len(dataset)}")
    print(f"  Min length   : {min(lengths)}")
    print(f"  Max length   : {max(lengths)}")
    print(f"  Mean length  : {np.mean(lengths):.1f}")
    print(f"  Median length: {np.median(lengths):.1f}")
    print(f"{'='*40}\n")


if __name__ == '__main__':
    import sys
    pdb_dir = sys.argv[1] if len(sys.argv) > 1 else './pdb_files'
    dataset = process_pdb_directory(pdb_dir, cache_file='dataset_cache.pkl')
    print_dataset_stats(dataset)
    if dataset:
        example = dataset[0]
        print(f"Example protein: {example['pdb_id']} chain {example['chain']}")
        print(f"  Sequence ({len(example['sequence'])} aa): {example['sequence'][:30]}...")
        print(f"  Coords shape:   {example['coords'].shape}")
        print(f"  Dist mat shape: {example['dist_mat'].shape}")
        print(f"  First Cα:       {example['coords'][0]}")
