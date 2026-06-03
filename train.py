"""
train.py
========
Training loop for the Transformer + GNN protein structure predictor.

Key design choices:
  - Loss       : MSE on upper-triangle of Cα distance matrix
                 + physics constraint: consecutive Cα must be ~3.8 Å (bond term)
  - Optimiser  : Adam  lr=1e-3, weight_decay=1e-5
  - Scheduler  : ReduceLROnPlateau (halves LR if val loss stalls for 5 epochs)
  - Grad clip  : max norm 1.0 (prevents exploding gradients)
  - Checkpoint : saves best model (lowest val loss) to disk
  - Evaluation : MSE loss + bond loss + mean RMSD over validation set
"""

import torch
import numpy as np
from tqdm import tqdm

from utils import combined_loss, distance_matrix_loss, distances_to_rmsd


# ─── One training epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, bond_weight=1.0, chunk_size=64):
    """
    Run one full pass over the training set.

    Returns:
        avg_total : average combined loss (MSE + bond constraint)
        avg_mse   : average MSE component
        avg_bond  : average bond constraint component
    """
    model.train()
    total_total = 0.0
    total_mse   = 0.0
    total_bond  = 0.0
    n_batches   = 0

    for seq, dist_true, pad_mask, coords_list, lengths in tqdm(loader, desc='  Train', leave=False):
        seq       = seq.to(device)
        dist_true = dist_true.to(device)
        pad_mask  = pad_mask.to(device)

        optimizer.zero_grad()

        dist_pred              = model(seq, pad_mask, chunk_size=chunk_size)
        loss, mse_l, bond_l   = combined_loss(dist_pred, dist_true, lengths, bond_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_total += loss.item()
        total_mse   += mse_l.item()
        total_bond  += bond_l.item()
        n_batches   += 1

    nb = max(n_batches, 1)
    return total_total / nb, total_mse / nb, total_bond / nb


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, bond_weight=1.0, compute_rmsd_flag=True, chunk_size=64):
    """
    Evaluate the model.

    Returns:
        avg_total : combined loss
        avg_mse   : MSE component
        avg_bond  : bond constraint component
        avg_rmsd  : mean RMSD in Å (nan if compute_rmsd_flag=False)
    """
    model.eval()
    total_total = 0.0
    total_mse   = 0.0
    total_bond  = 0.0
    total_rmsd  = 0.0
    n_batches   = 0
    n_proteins  = 0

    for seq, dist_true, pad_mask, coords_list, lengths in tqdm(loader, desc='  Eval ', leave=False):
        seq       = seq.to(device)
        dist_true = dist_true.to(device)
        pad_mask  = pad_mask.to(device)

        dist_pred            = model(seq, pad_mask, chunk_size=chunk_size)
        loss, mse_l, bond_l  = combined_loss(dist_pred, dist_true, lengths, bond_weight)

        total_total += loss.item()
        total_mse   += mse_l.item()
        total_bond  += bond_l.item()
        n_batches   += 1

        if compute_rmsd_flag:
            pred_np = dist_pred.cpu().numpy()
            for i, (n, true_coords) in enumerate(zip(lengths, coords_list)):
                try:
                    rmsd = distances_to_rmsd(pred_np[i, :n, :n], true_coords.numpy()[:n])
                    total_rmsd += rmsd
                    n_proteins += 1
                except Exception:
                    pass

    nb       = max(n_batches, 1)
    avg_rmsd = total_rmsd / max(n_proteins, 1) if compute_rmsd_flag else float('nan')
    return total_total / nb, total_mse / nb, total_bond / nb, avg_rmsd


# ─── Main training function ───────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    n_epochs:    int   = 100,
    lr:          float = 1e-3,
    weight_decay:float = 1e-5,
    bond_weight: float = 1.0,
    device:      str   = 'cpu',
    save_path:   str   = 'best_model.pt',
    rmsd_every:  int   = 5,
    chunk_size:  int   = 64,         # passed to model.forward() to limit VRAM
):
    """
    Full training run.

    Args:
        model         : ProteinStructurePredictor instance  (d_model=256 recommended)
        train_loader  : training DataLoader
        val_loader    : validation DataLoader
        test_loader   : test DataLoader
        n_epochs      : total training epochs
        lr            : learning rate  (1e-3 from ablation)
        weight_decay  : L2 regularisation
        bond_weight   : λ — weight on the 3.8 Å bond constraint term
                        0.0 = plain MSE, 1.0 = equal weighting (default)
        device        : 'cuda' or 'cpu'
        save_path     : path to save best model weights
        rmsd_every    : compute RMSD every N epochs (expensive step)

    Returns:
        history : dict with loss components and RMSD curves
    """
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {n_params:,}")
    print(f"Device           : {device}")
    print(f"Epochs           : {n_epochs}")
    print(f"Learning rate    : {lr}")
    print(f"Bond weight (λ)  : {bond_weight}  (3.8 Å constraint)")
    print(f"Chunk size       : {chunk_size}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    history = {
        'epoch':      [],
        'train_loss': [], 'train_mse': [], 'train_bond': [],
        'val_loss':   [], 'val_mse':   [], 'val_bond':   [],
        'val_rmsd':   [],
    }
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        tr_loss, tr_mse, tr_bond = train_one_epoch(
            model, train_loader, optimizer, device, bond_weight, chunk_size
        )

        # ── Validate ──────────────────────────────────────────────────────────
        do_rmsd = (epoch % rmsd_every == 0) or (epoch == n_epochs)
        vl_loss, vl_mse, vl_bond, vl_rmsd = evaluate(
            model, val_loader, device, bond_weight, compute_rmsd_flag=do_rmsd,
            chunk_size=chunk_size
        )

        scheduler.step(vl_loss)

        # ── Log ───────────────────────────────────────────────────────────────
        history['epoch'].append(epoch)
        history['train_loss'].append(tr_loss)
        history['train_mse'].append(tr_mse)
        history['train_bond'].append(tr_bond)
        history['val_loss'].append(vl_loss)
        history['val_mse'].append(vl_mse)
        history['val_bond'].append(vl_bond)
        history['val_rmsd'].append(vl_rmsd if do_rmsd else None)

        lr_now   = optimizer.param_groups[0]['lr']
        rmsd_str = f"{vl_rmsd:.2f} Å" if do_rmsd else "  ---  "
        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Loss: {tr_loss:.4f} (MSE {tr_mse:.4f} + bond {tr_bond:.4f}) | "
            f"Val: {vl_loss:.4f} | "
            f"RMSD: {rmsd_str} | "
            f"LR: {lr_now:.2e}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    vl_loss,
                'val_rmsd':    vl_rmsd if do_rmsd else None,
                'bond_weight': bond_weight,
            }, save_path)
            print(f"  → Best model saved  (val loss {best_val_loss:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Loading best model for final test evaluation...")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    test_loss, test_mse, test_bond, test_rmsd = evaluate(
        model, test_loader, device, bond_weight, compute_rmsd_flag=True,
        chunk_size=chunk_size
    )
    print(f"Test total loss  : {test_loss:.4f}")
    print(f"  └─ MSE term    : {test_mse:.4f}")
    print(f"  └─ Bond term   : {test_bond:.4f}  (λ={bond_weight})")
    print(f"Test RMSD        : {test_rmsd:.2f} Å")
    print("="*65 + "\n")

    history['test_loss'] = test_loss
    history['test_mse']  = test_mse
    history['test_bond'] = test_bond
    history['test_rmsd'] = test_rmsd

    return history