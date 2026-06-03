"""
evaluate.py
===========
Post-training analysis and visualisation:

  - plot_training_curves    : loss and RMSD vs epoch
  - plot_distance_matrix    : predicted vs true distance matrix (heatmap)
  - plot_structure          : predicted vs true 3D Cα trace
  - evaluate_single_protein : full pipeline for one protein
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # headless backend — works on any machine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from utils import classical_mds, compute_rmsd, kabsch_align


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str = 'training_curves.png'):
    """
    Plot train/val loss and val RMSD curves and save to file.
    """
    epochs = history['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training History', fontsize=13)

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax1.plot(epochs, history['train_loss'], label='Train loss', color='steelblue')
    ax1.plot(epochs, history['val_loss'],   label='Val loss',   color='orangered')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Distance MSE loss (Ų)')
    ax1.set_title('Loss curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── RMSD ─────────────────────────────────────────────────────────────────
    rmsd_epochs = [e for e, r in zip(epochs, history['val_rmsd']) if r is not None]
    rmsd_vals   = [r for r in history['val_rmsd'] if r is not None]
    if rmsd_vals:
        ax2.plot(rmsd_epochs, rmsd_vals, 'o-', color='forestgreen', label='Val RMSD')
        ax2.axhline(y=history.get('test_rmsd', np.nan), color='purple',
                    linestyle='--', alpha=0.7, label=f"Test RMSD: {history.get('test_rmsd', 0):.2f} Å")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSD (Å)')
        ax2.set_title('Cα RMSD (after MDS + Kabsch)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves → {save_path}")


# ─── Distance matrix heatmaps ─────────────────────────────────────────────────

def plot_distance_matrix(
    pred_dist: np.ndarray,
    true_dist: np.ndarray,
    pdb_id:    str = '',
    save_path: str = 'distance_matrix.png',
):
    """
    Side-by-side heatmaps: predicted vs true Cα distance matrix.
    """
    vmax = float(true_dist.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Cα Distance Matrix  —  {pdb_id}', fontsize=12)

    # True
    im0 = axes[0].imshow(true_dist,  cmap='viridis', vmin=0, vmax=vmax)
    axes[0].set_title('True distance matrix')
    axes[0].set_xlabel('Residue j')
    axes[0].set_ylabel('Residue i')
    plt.colorbar(im0, ax=axes[0], label='Distance (Å)')

    # Predicted
    im1 = axes[1].imshow(pred_dist,  cmap='viridis', vmin=0, vmax=vmax)
    axes[1].set_title('Predicted distance matrix')
    axes[1].set_xlabel('Residue j')
    plt.colorbar(im1, ax=axes[1], label='Distance (Å)')

    # Absolute error
    err = np.abs(pred_dist - true_dist)
    im2 = axes[2].imshow(err, cmap='hot', vmin=0)
    axes[2].set_title(f'|Error|  (mean={err.mean():.2f} Å)')
    axes[2].set_xlabel('Residue j')
    plt.colorbar(im2, ax=axes[2], label='|Δd| (Å)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance matrix plot → {save_path}")


# ─── 3D structure visualisation ───────────────────────────────────────────────

def plot_structure(
    pred_coords:  np.ndarray,
    true_coords:  np.ndarray,
    pdb_id:       str = '',
    save_path:    str = 'structure.png',
):
    """
    3D scatter + Cα trace: predicted (red) vs true (blue) after alignment.
    """
    pred_aligned, true_c = kabsch_align(pred_coords, true_coords)
    rmsd = compute_rmsd(pred_coords, true_coords)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    # True backbone
    ax.plot(true_c[:, 0],    true_c[:, 1],    true_c[:, 2],
            'b-o', markersize=2, linewidth=1, label='True Cα trace', alpha=0.8)

    # Predicted backbone
    ax.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
            'r-o', markersize=2, linewidth=1, label='Predicted Cα trace', alpha=0.8)

    ax.set_title(f'{pdb_id}  —  RMSD = {rmsd:.2f} Å')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved structure plot → {save_path}")


# ─── Single-protein evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate_single_protein(
    model,
    sequence:    str,
    true_coords: np.ndarray,
    device:      str = 'cpu',
    pdb_id:      str = 'unknown',
    plot:        bool = True,
):
    """
    Run the full prediction pipeline for one protein and produce plots.

    Args:
        model       : trained ProteinStructurePredictor
        sequence    : amino acid string (1-letter codes)
        true_coords : (N, 3) true Cα coordinates
        device      : 'cuda' or 'cpu'
        pdb_id      : label for plot titles
        plot        : if True, save diagnostic plots

    Returns:
        pred_coords : (N, 3) MDS-reconstructed coordinates
        rmsd        : float RMSD in Å
    """
    from dataset import encode_sequence

    model.eval()
    model.to(device)

    # Encode sequence
    seq_enc  = torch.tensor(encode_sequence(sequence), dtype=torch.long).unsqueeze(0).to(device)  # (1, N)
    pad_mask = torch.zeros(1, len(sequence), dtype=torch.bool).to(device)

    # Predict distance matrix
    dist_pred = model(seq_enc, pad_mask)                     # (1, N, N)
    pred_dist_np = dist_pred[0].cpu().numpy()                # (N, N)

    # Compute true distance matrix
    diff     = true_coords[:, np.newaxis] - true_coords[np.newaxis, :]
    true_dist_np = np.sqrt(np.sum(diff ** 2, axis=-1))

    # MDS → 3D coordinates
    pred_coords = classical_mds(pred_dist_np)                # (N, 3)
    rmsd        = compute_rmsd(pred_coords, true_coords)

    print(f"\n{pdb_id}  (length {len(sequence)})")
    print(f"  Distance MAE : {np.abs(pred_dist_np - true_dist_np).mean():.2f} Å")
    print(f"  RMSD         : {rmsd:.2f} Å")

    if plot:
        plot_distance_matrix(pred_dist_np, true_dist_np, pdb_id, f'{pdb_id}_dist_matrix.png')
        plot_structure(pred_coords, true_coords, pdb_id, f'{pdb_id}_structure.png')

    return pred_coords, rmsd
