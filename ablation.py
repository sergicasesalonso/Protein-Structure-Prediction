"""
ablation.py
===========
Ablation study for the Transformer + GNN protein structure predictor.

Tests five configurations to measure the contribution of each component:

  A) Full model          — Transformer (4L) + GNN (3L)          [paper model]
  B) No GNN              — Transformer (4L) only                 [remove GNN]
  C) No Transformer      — Positional MLP + GNN (3L)             [remove attention]
  D) MLP baseline        — Positional MLP only                   [remove both]
  E) Mean baseline       — always predicts the training mean      [no learning]

Each neural model is trained from scratch with identical hyperparameters
and evaluated on the same held-out test set. Results are saved to:
  ablation_results.csv        — numeric summary table
  ablation_barplot.png        — RMSD / loss comparison bar chart
  ablation_curves.png         — loss curves for all models on one plot

Usage:
  python ablation.py                          # full run, all defaults
  python ablation.py --max_samples 500        # quick smoke test
  python ablation.py --epochs 30 --pdb_dir ./pdb_files
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from parse_pdb  import process_pdb_directory, print_dataset_stats
from dataset    import ProteinDataset, make_data_loaders
from model      import ProteinStructurePredictor, count_parameters
from utils      import combined_loss, distances_to_rmsd


# ─── Ablation configurations ─────────────────────────────────────────────────

ABLATION_CONFIGS = [
    # ── Architecture ablations (bond_weight=1.0 for all) ─────────────────────
    {
        'name':        'Full (TF + GNN)',
        'label':       'Full',
        'color':       '#2196F3',
        'model_kwargs': dict(n_transformer_layers=4, n_gnn_layers=3),
        'bond_weight':  1.0,
    },
    {
        'name':        'No GNN',
        'label':       'No GNN',
        'color':       '#FF9800',
        'model_kwargs': dict(n_transformer_layers=4, n_gnn_layers=0),
        'bond_weight':  1.0,
    },
    {
        'name':        'No Transformer',
        'label':       'No TF',
        'color':       '#9C27B0',
        'model_kwargs': dict(n_transformer_layers=0, n_gnn_layers=3),
        'bond_weight':  1.0,
    },
    {
        'name':        'MLP Only',
        'label':       'MLP only',
        'color':       '#F44336',
        'model_kwargs': dict(n_transformer_layers=0, n_gnn_layers=0),
        'bond_weight':  1.0,
    },
    # ── Loss ablations (full architecture, varying bond weight λ) ────────────
    {
        'name':        'Full, λ=0 (MSE only)',
        'label':       'λ=0',
        'color':       '#00BCD4',
        'model_kwargs': dict(n_transformer_layers=4, n_gnn_layers=3),
        'bond_weight':  0.0,   # plain MSE — no physics constraint
    },
    {
        'name':        'Full, λ=0.5',
        'label':       'λ=0.5',
        'color':       '#8BC34A',
        'model_kwargs': dict(n_transformer_layers=4, n_gnn_layers=3),
        'bond_weight':  0.5,
    },
    {
        'name':        'Full, λ=2.0',
        'label':       'λ=2.0',
        'color':       '#FF5722',
        'model_kwargs': dict(n_transformer_layers=4, n_gnn_layers=3),
        'bond_weight':  2.0,   # strongly enforce the 3.8 Å bond
    },
    # Mean baseline added separately (not a neural model)
]


# ─── Mean distance baseline ───────────────────────────────────────────────────

def compute_mean_baseline(train_loader, test_loader, device):
    """
    Predicts every pairwise distance as the global mean Cα–Cα distance
    computed from the training set. No parameters, no learning.

    Returns:
        test_loss : float   MSE when predicting mean for every pair
        test_rmsd : float   mean RMSD after MDS + Kabsch
    """
    print("\n[Baseline] Computing training-set mean distance...")
    all_dists = []

    for _, dist_true, pad_mask, coords_list, lengths in tqdm(train_loader, desc='  Baseline mean', leave=False):
        dist_true_np = dist_true.numpy()
        for i, n in enumerate(lengths):
            triu = np.triu_indices(n, k=1)
            all_dists.extend(dist_true_np[i, :n, :n][triu].tolist())

    mean_dist = float(np.mean(all_dists))
    print(f"  Training mean Cα distance: {mean_dist:.2f} Å")

    # Evaluate on test set
    total_loss = 0.0
    total_rmsd = 0.0
    n_batches  = 0
    n_proteins = 0

    for _, dist_true, pad_mask, coords_list, lengths in tqdm(test_loader, desc='  Baseline test', leave=False):
        dist_np = dist_true.numpy()
        for i, (n, true_coords) in enumerate(zip(lengths, coords_list)):
            true_d = dist_np[i, :n, :n]
            pred_d = np.full_like(true_d, mean_dist)
            np.fill_diagonal(pred_d, 0.0)

            triu   = np.triu_indices(n, k=1)
            mse    = float(np.mean((pred_d[triu] - true_d[triu]) ** 2))
            total_loss += mse
            n_batches  += 1

            try:
                rmsd = distances_to_rmsd(pred_d, true_coords.numpy())
                total_rmsd += rmsd
                n_proteins += 1
            except Exception:
                pass

    avg_loss = total_loss / max(n_batches, 1)
    avg_rmsd = total_rmsd / max(n_proteins, 1)
    return avg_loss, avg_rmsd, mean_dist


# ─── Train one model for the ablation ────────────────────────────────────────

def train_one_ablation(
    config,
    train_loader,
    val_loader,
    test_loader,
    n_epochs,
    lr,
    weight_decay,
    bond_weight,
    device,
    d_model,
    n_heads,
    d_feedforward,
    d_proj,
):
    """
    Instantiate, train, and evaluate one ablation model.
    Returns a dict with all metrics + loss history.
    """
    name = config['name']
    print(f"\n{'='*60}")
    print(f"  Ablation: {name}  (λ_bond={bond_weight})")
    print(f"{'='*60}")

    # Build model
    model = ProteinStructurePredictor(
        d_model       = d_model,
        n_heads       = n_heads,
        d_feedforward = d_feedforward,
        d_proj        = d_proj,
        **config['model_kwargs'],
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    train_losses = []
    val_losses   = []
    best_val     = float('inf')
    best_state   = None

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for seq, dist_true, pad_mask, coords_list, lengths in train_loader:
            seq       = seq.to(device)
            dist_true = dist_true.to(device)
            pad_mask  = pad_mask.to(device)

            optimizer.zero_grad()
            dist_pred = model(seq, pad_mask)
            loss, _, _ = combined_loss(dist_pred, dist_true, lengths, bond_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            n_batch += 1

        avg_train = ep_loss / max(n_batch, 1)
        train_losses.append(avg_train)

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        vl_loss = 0.0
        vl_n    = 0
        with torch.no_grad():
            for seq, dist_true, pad_mask, coords_list, lengths in val_loader:
                seq       = seq.to(device)
                dist_true = dist_true.to(device)
                pad_mask  = pad_mask.to(device)
                dist_pred = model(seq, pad_mask)
                loss, _, _ = combined_loss(dist_pred, dist_true, lengths, bond_weight)
                vl_loss  += loss.item()
                vl_n     += 1

        avg_val = vl_loss / max(vl_n, 1)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val   = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == n_epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"[{elapsed:.0f}s]")

    # ── Test evaluation with best checkpoint ──────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    test_loss_total = 0.0
    test_rmsd_total = 0.0
    test_n_batches  = 0
    test_n_proteins = 0

    with torch.no_grad():
        for seq, dist_true, pad_mask, coords_list, lengths in tqdm(test_loader, desc='  Test eval', leave=False):
            seq       = seq.to(device)
            dist_true = dist_true.to(device)
            pad_mask  = pad_mask.to(device)
            dist_pred = model(seq, pad_mask)
            loss, _, _ = combined_loss(dist_pred, dist_true, lengths, bond_weight)
            test_loss_total += loss.item()
            test_n_batches  += 1

            pred_np = dist_pred.cpu().numpy()
            for i, (n, true_coords) in enumerate(zip(lengths, coords_list)):
                try:
                    rmsd = distances_to_rmsd(pred_np[i, :n, :n], true_coords.numpy()[:n])
                    test_rmsd_total += rmsd
                    test_n_proteins += 1
                except Exception:
                    pass

    test_loss = test_loss_total / max(test_n_batches, 1)
    test_rmsd = test_rmsd_total / max(test_n_proteins, 1)

    print(f"\n  ✓ {name}")
    print(f"    Test loss : {test_loss:.4f} Ų")
    print(f"    Test RMSD : {test_rmsd:.2f} Å")
    print(f"    Params    : {n_params:,}")

    return {
        'name':         name,
        'label':        config['label'],
        'color':        config['color'],
        'n_params':     n_params,
        'best_val_loss':best_val,
        'test_loss':    test_loss,
        'test_rmsd':    test_rmsd,
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'model':        model,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_bar_comparison(results, save_path='ablation_barplot.png'):
    """
    Side-by-side bar charts: test RMSD and test loss for all ablations.
    """
    names  = [r['label']     for r in results]
    rmsds  = [r['test_rmsd'] for r in results]
    losses = [r['test_loss'] for r in results]
    colors = [r['color']     for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ablation Study — Component Contribution', fontsize=13, fontweight='bold')

    x = np.arange(len(names))
    w = 0.55

    # RMSD
    bars1 = ax1.bar(x, rmsds, width=w, color=colors, edgecolor='white', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel('Test RMSD (Å)', fontsize=11)
    ax1.set_title('Cα RMSD after MDS + Kabsch alignment\n(lower is better)', fontsize=10)
    ax1.bar_label(bars1, fmt='%.2f Å', padding=3, fontsize=9)
    ax1.set_ylim(0, max(rmsds) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    # Loss
    bars2 = ax2.bar(x, losses, width=w, color=colors, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel('Test Distance MSE (Ų)', fontsize=11)
    ax2.set_title('Pairwise distance MSE\n(lower is better)', fontsize=10)
    ax2.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9)
    ax2.set_ylim(0, max(losses) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart → {save_path}")


def plot_loss_curves(results, save_path='ablation_curves.png'):
    """
    Training and validation loss curves for all neural ablations on one plot.
    (Skips mean baseline which has no curves.)
    """
    neural = [r for r in results if 'train_losses' in r and r['train_losses']]
    if not neural:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Study — Loss Curves', fontsize=13, fontweight='bold')

    for r in neural:
        ep = range(1, len(r['train_losses']) + 1)
        ax1.plot(ep, r['train_losses'], color=r['color'], label=r['label'], linewidth=1.8)
        ax2.plot(ep, r['val_losses'],   color=r['color'], label=r['label'], linewidth=1.8)

    for ax, title in [(ax1, 'Training loss'), (ax2, 'Validation loss')]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Distance MSE (Ų)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves → {save_path}")


def save_csv(results, save_path='ablation_results.csv'):
    """Write a clean summary CSV."""
    fieldnames = ['Model', 'Parameters', 'Best Val Loss', 'Test Loss (Ų)', 'Test RMSD (Å)']
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'Model':             r['name'],
                'Parameters':        r.get('n_params', 'N/A'),
                'Best Val Loss':     f"{r.get('best_val_loss', float('nan')):.4f}",
                'Test Loss (Ų)':    f"{r['test_loss']:.4f}",
                'Test RMSD (Å)':    f"{r['test_rmsd']:.2f}",
            })
    print(f"Saved CSV → {save_path}")


def print_summary_table(results):
    """Print a formatted summary to stdout."""
    print(f"\n{'='*72}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'='*72}")
    print(f"  {'Model':<25} {'Params':>10}  {'Val Loss':>10}  {'Test Loss':>10}  {'RMSD (Å)':>10}")
    print(f"  {'-'*65}")
    for r in results:
        params_str = f"{r.get('n_params', 0):,}" if r.get('n_params') else 'N/A'
        val_str    = f"{r.get('best_val_loss', float('nan')):.4f}" if 'best_val_loss' in r else 'N/A'
        print(
            f"  {r['name']:<25} {params_str:>10}  {val_str:>10}  "
            f"{r['test_loss']:>10.4f}  {r['test_rmsd']:>10.2f}"
        )
    print(f"{'='*72}\n")


# ─── Argument parser ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Ablation study for protein structure predictor')
    p.add_argument('--pdb_dir',      type=str,   default='./pdb_files')
    p.add_argument('--cache_file',   type=str,   default='dataset_cache.pkl')
    p.add_argument('--max_samples',  type=int,   default=None,
                   help='Limit dataset size (useful for fast tests, e.g. 500)')
    p.add_argument('--min_len',      type=int,   default=50)
    p.add_argument('--max_len',      type=int,   default=500)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--d_model',      type=int,   default=64)
    p.add_argument('--n_heads',      type=int,   default=4)
    p.add_argument('--d_feedforward',type=int,   default=128)
    p.add_argument('--d_proj',       type=int,   default=32)
    p.add_argument('--device',       type=str,   default='auto')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--out_dir',      type=str,   default='.',
                   help='Directory to save all output files')
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device  : {device}")
    print(f"Epochs  : {args.epochs}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n--- Loading dataset ---")
    raw_data = process_pdb_directory(
        pdb_dir     = args.pdb_dir,
        min_len     = args.min_len,
        max_len     = args.max_len,
        max_samples = args.max_samples,
        cache_file  = args.cache_file,
    )

    if not raw_data:
        print("ERROR: No proteins found. Check --pdb_dir.")
        return

    print_dataset_stats(raw_data)

    dataset = ProteinDataset(raw_data)
    train_loader, val_loader, test_loader = make_data_loaders(
        dataset, batch_size=args.batch_size, seed=args.seed
    )

    all_results = []

    # ── Mean baseline ─────────────────────────────────────────────────────────
    base_loss, base_rmsd, mean_d = compute_mean_baseline(train_loader, test_loader, device)
    all_results.append({
        'name':      f'Mean baseline ({mean_d:.1f} Å)',
        'label':     'Baseline',
        'color':     '#9E9E9E',
        'n_params':  0,
        'best_val_loss': float('nan'),
        'test_loss': base_loss,
        'test_rmsd': base_rmsd,
        'train_losses': [],
        'val_losses':   [],
    })

    # ── Neural ablations ──────────────────────────────────────────────────────
    for cfg in ABLATION_CONFIGS:
        result = train_one_ablation(
            config        = cfg,
            train_loader  = train_loader,
            val_loader    = val_loader,
            test_loader   = test_loader,
            n_epochs      = args.epochs,
            lr            = args.lr,
            weight_decay  = args.weight_decay,
            bond_weight   = cfg.get('bond_weight', 1.0),
            device        = device,
            d_model       = args.d_model,
            n_heads       = args.n_heads,
            d_feedforward = args.d_feedforward,
            d_proj        = args.d_proj,
        )
        all_results.append(result)

    # ── Outputs ───────────────────────────────────────────────────────────────
    print_summary_table(all_results)

    save_csv(all_results,         os.path.join(args.out_dir, 'ablation_results.csv'))
    plot_bar_comparison(all_results, os.path.join(args.out_dir, 'ablation_barplot.png'))
    plot_loss_curves(all_results,    os.path.join(args.out_dir, 'ablation_curves.png'))

    print("\nAll ablation outputs saved:")
    print(f"  {args.out_dir}/ablation_results.csv")
    print(f"  {args.out_dir}/ablation_barplot.png")
    print(f"  {args.out_dir}/ablation_curves.png")


if __name__ == '__main__':
    main()