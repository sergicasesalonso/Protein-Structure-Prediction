"""
main.py
=======
Entry point for the Transformer + GNN protein structure predictor.

Best hyperparameters from ablation study:
  lr=1e-3, d_model=256, loss=MSE + bond constraint (λ=1.0)

Usage:
    python main.py
    python main.py --pdb_dir ./pdb_files --epochs 100
    python main.py --max_samples 500        # quick test
    python main.py --device cuda
"""

import argparse
import torch

from parse_pdb  import process_pdb_directory, print_dataset_stats
from dataset    import ProteinDataset, make_data_loaders
from model      import ProteinStructurePredictor, count_parameters
from train      import train
from evaluate   import plot_training_curves, evaluate_single_protein


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_dir',      type=str,   default='./pdb_files')
    p.add_argument('--cache_file',   type=str,   default='dataset_cache.pkl')
    p.add_argument('--max_samples',  type=int,   default=None)
    p.add_argument('--min_len',      type=int,   default=50)
    p.add_argument('--max_len',      type=int,   default=500)
    p.add_argument('--epochs',       type=int,   default=100)      # ← 100 epochs
    p.add_argument('--lr',           type=float, default=1e-3)     # best from ablation
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--bond_weight',  type=float, default=1.0)      # λ physics constraint
    p.add_argument('--batch_size',   type=int,   default=4)        # ← reduced: fixes OOM
    p.add_argument('--chunk_size',   type=int,   default=64,
                   help='Rows of dist matrix computed per VRAM chunk. '
                        'Lower (32/16) if still OOM. Higher (128) if VRAM allows.')
    p.add_argument('--rmsd_every',   type=int,   default=5)
    p.add_argument('--save_path',    type=str,   default='best_model.pt')
    p.add_argument('--d_model',      type=int,   default=256)      # best from ablation
    p.add_argument('--n_heads',      type=int,   default=4)
    p.add_argument('--n_tf_layers',  type=int,   default=4)
    p.add_argument('--n_gnn_layers', type=int,   default=3)
    p.add_argument('--device',       type=str,   default='auto')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


def main():
    args = get_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else args.device
    print(f"Device: {device}")

    # Free any leftover VRAM before we start
    if device == 'cuda':
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n--- Loading dataset ---")
    raw_data = process_pdb_directory(
        args.pdb_dir, args.min_len, args.max_len,
        args.max_samples, args.cache_file,
    )
    if not raw_data:
        print("ERROR: No proteins found. Check --pdb_dir.")
        return
    print_dataset_stats(raw_data)

    dataset = ProteinDataset(raw_data)
    train_loader, val_loader, test_loader = make_data_loaders(
        dataset, batch_size=args.batch_size, seed=args.seed
    )

    # ── 2. Build model ────────────────────────────────────────────────────────
    print("\n--- Building model ---")
    model = ProteinStructurePredictor(
        d_model              = args.d_model,           # 256
        n_heads              = args.n_heads,
        n_transformer_layers = args.n_tf_layers,
        n_gnn_layers         = args.n_gnn_layers,
        d_feedforward        = args.d_model * 4,       # 1024
        d_proj               = args.d_model // 8,      # 32  ← smaller proj saves VRAM
    )
    print(f"Trainable parameters : {count_parameters(model):,}")
    if device == 'cuda':
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU VRAM             : {total_mem:.1f} GB")
    print(f"Chunk size           : {args.chunk_size}  "
          f"(reduce to 32/16 if OOM, increase for speed)")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n--- Training ---")
    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        n_epochs     = args.epochs,        # 100
        lr           = args.lr,            # 1e-3
        weight_decay = args.weight_decay,
        bond_weight  = args.bond_weight,   # 1.0
        device       = device,
        save_path    = args.save_path,
        rmsd_every   = args.rmsd_every,
        chunk_size   = args.chunk_size,
    )

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n--- Saving plots ---")
    plot_training_curves(history, save_path='training_curves.png')

    # ── 5. Example protein ────────────────────────────────────────────────────
    print("\n--- Example protein evaluation ---")
    example = raw_data[0]
    evaluate_single_protein(
        model       = model,
        sequence    = example['sequence'],
        true_coords = example['coords'],
        device      = device,
        pdb_id      = f"{example['pdb_id']}_{example['chain']}",
        plot        = True,
    )

    print("\nDone. Output files:")
    print("  best_model.pt | training_curves.png | "
          "<pdb>_dist_matrix.png | <pdb>_structure.png")


if __name__ == '__main__':
    main()