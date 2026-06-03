"""
utils.py
========
Mathematical utilities for the protein structure prediction pipeline:

  - distance_matrix_loss   : MSE loss on pairwise Cα distances
  - bond_constraint_loss   : physics prior — consecutive Cα must be ~3.8 Å apart
  - combined_loss          : MSE + weighted bond constraint (used in training)
  - classical_mds          : convert predicted distance matrix → 3D coordinates
  - kabsch_align           : optimal superposition of two point clouds
  - compute_rmsd           : RMSD after Kabsch alignment (in Å)
"""

import numpy as np
import torch


# ─── MSE loss on distance matrix ─────────────────────────────────────────────

def distance_matrix_loss(
    pred_dist: torch.Tensor,
    true_dist: torch.Tensor,
    lengths:   list[int],
) -> torch.Tensor:
    """
    Mean squared error (MSE) on the *upper triangle* of the distance matrix,
    computed only over real (non-padded) residue pairs.

    Using only the upper triangle avoids counting each pair twice and
    avoids the trivially-zero diagonal (d_ii = 0).

    Args:
        pred_dist : (B, L, L)  predicted distances
        true_dist : (B, L, L)  ground-truth distances
        lengths   : list of int, actual sequence length per sample

    Returns:
        loss : scalar tensor
    """
    total = torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
    count = 0

    for i, n in enumerate(lengths):
        pred_i = pred_dist[i, :n, :n]
        true_i = true_dist[i, :n, :n]

        triu_idx  = torch.triu_indices(n, n, offset=1, device=pred_dist.device)
        pred_vals = pred_i[triu_idx[0], triu_idx[1]]
        true_vals = true_i[triu_idx[0], triu_idx[1]]

        total = total + torch.mean((pred_vals - true_vals) ** 2)
        count += 1

    return total / max(count, 1)


# ─── Physics-informed bond constraint loss ────────────────────────────────────

# Average Cα–Cα distance along the peptide backbone (Å).
# Every pair of consecutive residues (i, i+1) must be this distance apart —
# this is a hard geometric constraint of the peptide bond.
CA_BOND_LENGTH = 3.8   # Å


def bond_constraint_loss(
    pred_dist: torch.Tensor,
    lengths:   list[int],
    bond_len:  float = CA_BOND_LENGTH,
) -> torch.Tensor:
    """
    Physics-informed constraint loss: penalises deviations of consecutive
    Cα–Cα distances from the known peptide bond length of 3.8 Å.

    Inspired by the use of holonomic constraints in Lagrangian mechanics:
    just as an inextensible pendulum enforces ||r|| = L via a Lagrange
    multiplier, we add a soft constraint that pushes d(i, i+1) → 3.8 Å
    for all consecutive residue pairs.

    L_bond = (1 / (N-1)) * Σ_{i=1}^{N-1} (d_pred(i, i+1) - 3.8)²

    Args:
        pred_dist : (B, L, L)  predicted distance matrix
        lengths   : list of actual sequence lengths (unpadded)
        bond_len  : target Cα–Cα bond distance (default 3.8 Å)

    Returns:
        loss : scalar tensor (mean over all proteins in the batch)
    """
    total = torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
    count = 0

    for i, n in enumerate(lengths):
        if n < 2:
            continue   # need at least two residues for a bond

        # Extract the super-diagonal: d(0,1), d(1,2), ..., d(N-2, N-1)
        # These are the consecutive Cα–Cα distances
        idx       = torch.arange(n - 1, device=pred_dist.device)
        bond_pred = pred_dist[i, idx, idx + 1]              # (N-1,)
        target    = torch.full_like(bond_pred, bond_len)    # all 3.8 Å

        total = total + torch.mean((bond_pred - target) ** 2)
        count += 1

    return total / max(count, 1)


# ─── Combined loss ────────────────────────────────────────────────────────────

def combined_loss(
    pred_dist:   torch.Tensor,
    true_dist:   torch.Tensor,
    lengths:     list[int],
    bond_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total training loss combining MSE on all pairwise distances with a
    physics-informed constraint on consecutive Cα bond lengths.

    L_total = L_MSE + bond_weight * L_bond

    The bond_weight controls how strongly we enforce the 3.8 Å constraint.
    bond_weight=0.0 recovers the plain MSE baseline.
    bond_weight=1.0 (default) weights both terms equally.

    Args:
        pred_dist   : (B, L, L)  predicted distance matrix
        true_dist   : (B, L, L)  ground-truth distance matrix
        lengths     : list of actual sequence lengths
        bond_weight : λ — weight on the bond constraint term

    Returns:
        total_loss  : scalar — L_MSE + λ * L_bond   (used for .backward())
        mse_loss    : scalar — L_MSE component       (for logging)
        bond_loss   : scalar — L_bond component      (for logging)
    """
    mse_loss  = distance_matrix_loss(pred_dist, true_dist, lengths)
    bond_loss = bond_constraint_loss(pred_dist, lengths)
    total     = mse_loss + bond_weight * bond_loss
    return total, mse_loss, bond_loss


# ─── Classical MDS ───────────────────────────────────────────────────────────

def classical_mds(distance_matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Classical Multidimensional Scaling (cMDS).

    Converts a pairwise distance matrix D into Cartesian coordinates X
    such that ||X_i - X_j|| ≈ D_ij.

    Algorithm:
      1. Double-centre the squared distance matrix to get the Gram matrix B.
      2. Eigendecompose B.
      3. Take the top-k eigenvectors scaled by sqrt(eigenvalues).

    This is the deterministic, parameter-free reconstruction step
    described in the project report (Step 3).

    Args:
        distance_matrix : (N, N) numpy array of pairwise distances
        n_components    : target dimensionality (3 for 3-D protein structure)

    Returns:
        coords : (N, n_components) numpy array of reconstructed coordinates
    """
    D  = np.array(distance_matrix, dtype=np.float64)
    n  = D.shape[0]

    # 1. Double centring: B = -1/2 * J D² J   where J = I - (1/n) 11ᵀ
    D_sq      = D ** 2
    row_mean  = D_sq.mean(axis=1, keepdims=True)
    col_mean  = D_sq.mean(axis=0, keepdims=True)
    grand_mean= D_sq.mean()
    B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

    # 2. Eigendecomposition of the symmetric Gram matrix
    eigenvalues, eigenvectors = np.linalg.eigh(B)   # ascending order

    # 3. Sort descending
    idx         = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors= eigenvectors[:, idx]

    # 4. Clip negative eigenvalues (numerical noise) and take top-k
    eigenvalues_pos = np.maximum(eigenvalues[:n_components], 0.0)
    coords = eigenvectors[:, :n_components] * np.sqrt(eigenvalues_pos)

    return coords.astype(np.float32)   # (N, 3)


# ─── Kabsch alignment ────────────────────────────────────────────────────────

def kabsch_align(P: np.ndarray, Q: np.ndarray):
    """
    Find the optimal rotation R that minimises ||P R - Q||²  (Kabsch algorithm).

    Both structures are centred first. The rotation handles reflections
    via the sign-corrected SVD.

    Args:
        P : (N, 3) predicted coordinates (to be rotated)
        Q : (N, 3) true Cα coordinates   (reference)

    Returns:
        P_aligned : (N, 3) rotated and centred P
        Q_centred : (N, 3) centred Q
    """
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # Cross-covariance matrix
    H = P_c.T @ Q_c            # (3, 3)

    U, S, Vt = np.linalg.svd(H)

    # Correct for reflections: det(V Uᵀ) must be +1
    d   = np.sign(np.linalg.det(Vt.T @ U.T))
    D   = np.diag([1.0, 1.0, d])

    # Optimal rotation
    R   = Vt.T @ D @ U.T       # (3, 3)

    P_aligned = P_c @ R.T      # (N, 3)
    return P_aligned, Q_c


# ─── RMSD ────────────────────────────────────────────────────────────────────

def compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Root Mean Square Deviation between predicted and true Cα coordinates
    after optimal Kabsch superposition.

    RMSD = sqrt( (1/N) Σ_i ||P_i - Q_i||² )

    Lower is better. AlphaFold achieves < 1 Å on many targets;
    a simple sequence-only model typically achieves 5–20 Å.

    Args:
        P : (N, 3) predicted coordinates (e.g. from MDS)
        Q : (N, 3) true experimental coordinates

    Returns:
        rmsd : float, RMSD in Angstroms
    """
    P_aligned, Q_c = kabsch_align(P, Q)
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q_c) ** 2, axis=1))))
    return rmsd


# ─── End-to-end: distances → RMSD ────────────────────────────────────────────

def distances_to_rmsd(pred_dist_np: np.ndarray, true_coords_np: np.ndarray) -> float:
    """
    Convenience wrapper:
      1. Run classical MDS on the predicted distance matrix.
      2. Align with Kabsch.
      3. Return RMSD.

    Args:
        pred_dist_np   : (N, N) predicted distance matrix
        true_coords_np : (N, 3) true Cα coordinates

    Returns:
        rmsd : float
    """
    pred_coords = classical_mds(pred_dist_np, n_components=3)
    return compute_rmsd(pred_coords, true_coords_np)


if __name__ == '__main__':
    # Sanity check: perfect distance matrix should give near-zero RMSD
    np.random.seed(0)
    N = 50
    true_coords = np.random.randn(N, 3).astype(np.float32) * 10.0

    # Compute exact distances
    diff     = true_coords[:, np.newaxis] - true_coords[np.newaxis, :]
    true_D   = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Reconstruct via MDS
    pred_coords = classical_mds(true_D)
    rmsd        = compute_rmsd(pred_coords, true_coords)
    print(f"RMSD from perfect distances: {rmsd:.4f} Å  (should be ~0)")

    # Check with noisy distances
    noisy_D     = true_D + np.random.randn(N, N) * 1.0
    noisy_D     = (noisy_D + noisy_D.T) / 2
    np.fill_diagonal(noisy_D, 0)
    pred_noisy  = classical_mds(noisy_D)
    rmsd_noisy  = compute_rmsd(pred_noisy, true_coords)
    print(f"RMSD from noisy distances:   {rmsd_noisy:.4f} Å")