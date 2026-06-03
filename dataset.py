"""
dataset.py
==========
PyTorch Dataset and DataLoader utilities for protein structure prediction.

Each sample contains:
  - seq_encoded : (N,)    integer-encoded amino acid sequence
  - dist_matrix : (N, N)  pairwise Cα distance matrix (Å)
  - coords      : (N, 3)  true Cα coordinates (for RMSD evaluation)

Because proteins have different lengths, collate_fn pads sequences and
distance matrices to the longest in each batch and returns a boolean mask.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# ─── Amino acid vocabulary ────────────────────────────────────────────────────

AA_VOCAB   = 'ACDEFGHIKLMNPQRSTVWY'          # 20 standard amino acids
AA_TO_IDX  = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}  # 1-based; 0 = PAD
IDX_TO_AA  = {v: k for k, v in AA_TO_IDX.items()}
VOCAB_SIZE = len(AA_VOCAB) + 1               # 21  (including padding token)


def encode_sequence(seq: str) -> list[int]:
    """
    Convert 1-letter amino acid string to list of integer indices.
    Unknown amino acids map to 0 (same as padding — treated as unknown).
    """
    return [AA_TO_IDX.get(aa, 0) for aa in seq]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ProteinDataset(Dataset):
    """
    Wraps the list of protein dicts produced by parse_pdb.process_pdb_directory.
    """

    def __init__(self, data: list[dict]):
        """
        Args:
            data: list of dicts with keys 'sequence', 'dist_mat', 'coords'
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq_enc   = torch.tensor(encode_sequence(item['sequence']), dtype=torch.long)
        dist_mat  = torch.tensor(item['dist_mat'],  dtype=torch.float32)
        coords    = torch.tensor(item['coords'],    dtype=torch.float32)
        return seq_enc, dist_mat, coords


# ─── Collate function (handles variable-length sequences) ─────────────────────

def collate_fn(batch):
    """
    Pads a batch of variable-length proteins.

    Returns:
        seq_padded  : (B, L)    integer-encoded sequences, padded with 0
        dist_padded : (B, L, L) distance matrices, padded with 0
        pad_mask    : (B, L)    bool — True where position is padding
        coords_list : list of (N_i, 3) tensors (unpadded, for evaluation)
        lengths     : list of int, actual sequence lengths
    """
    sequences, dist_matrices, coords_list = zip(*batch)

    B        = len(sequences)
    lengths  = [s.shape[0] for s in sequences]
    L        = max(lengths)                            # max length in this batch

    seq_padded  = torch.zeros(B, L,    dtype=torch.long)
    dist_padded = torch.zeros(B, L, L, dtype=torch.float32)
    pad_mask    = torch.ones(B, L,     dtype=torch.bool)   # True = masked (padding)

    for i, (seq, dist, _) in enumerate(zip(sequences, dist_matrices, coords_list)):
        n = lengths[i]
        seq_padded[i,  :n]     = seq
        dist_padded[i, :n, :n] = dist
        pad_mask[i,    :n]     = False                 # False = real token

    return seq_padded, dist_padded, pad_mask, list(coords_list), lengths


# ─── Train / val / test split ────────────────────────────────────────────────

def make_data_loaders(dataset, batch_size=8, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Randomly split dataset into train / val / test and return DataLoaders.

    Args:
        dataset    : ProteinDataset
        batch_size : samples per batch
        train_frac : fraction for training   (default 80%)
        val_frac   : fraction for validation (default 10%)
        seed       : random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val   = int(val_frac   * n_total)
    n_test  = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    print(f"Split: {n_train} train | {n_val} val | {n_test} test")

    shared = dict(collate_fn=collate_fn, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **shared)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **shared)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **shared)

    return train_loader, val_loader, test_loader
