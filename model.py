"""
model.py
========
Protein structure prediction model as described in the project report:

  Step 1 — Transformer encoder
      Embeds the amino acid sequence and captures long-range pairwise
      co-evolutionary signals using multi-head self-attention.
      (Skipped cleanly when n_transformer_layers=0 for ablation.)

  Step 2 — Graph Neural Network (GNN)
      Operates on the protein chain graph (linear chain of residues).
      Each node aggregates information from its sequential neighbours.
      (Skipped cleanly when n_gnn_layers=0 for ablation.)

  Step 3 — MLP distance head
      For every residue pair (i, j) predicts the positive Cα–Cα distance.
"""

import math
import torch
import torch.nn as nn


# ─── Positional Encoding ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─── GNN Layer ───────────────────────────────────────────────────────────────

class ChainGNNLayer(nn.Module):
    """
    One round of message passing on the protein chain graph.
    Each residue i aggregates messages from neighbours i-1 and i+1.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        left  = torch.cat([x[:, :1, :],  x[:, :-1, :]], dim=1)
        right = torch.cat([x[:, 1:,  :], x[:, -1:, :]], dim=1)
        msg_l = self.message_mlp(torch.cat([x, left],  dim=-1))
        msg_r = self.message_mlp(torch.cat([x, right], dim=-1))
        agg   = (msg_l + msg_r) / 2.0
        x_new = self.dropout(self.update_mlp(torch.cat([x, agg], dim=-1)))
        x_out = self.norm(x + x_new)
        if pad_mask is not None:
            x_out = x_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x_out


# ─── Full Model ───────────────────────────────────────────────────────────────

class ProteinStructurePredictor(nn.Module):
    """
    Ablation-safe Transformer + GNN model.

    Setting n_transformer_layers=0 removes the Transformer completely
    (uses a plain linear layer instead so dimensions still match).
    Setting n_gnn_layers=0 removes all GNN message-passing rounds.
    Both can be set to 0 simultaneously for a pure MLP baseline.
    """

    def __init__(
        self,
        vocab_size:            int   = 21,
        d_model:               int   = 64,
        n_heads:               int   = 4,
        n_transformer_layers:  int   = 4,
        n_gnn_layers:          int   = 3,
        d_feedforward:         int   = 128,
        d_proj:                int   = 32,
        dropout:               float = 0.1,
        max_seq_len:           int   = 600,
    ):
        super().__init__()
        self.d_model             = d_model
        self.d_proj              = d_proj
        self.n_transformer_layers= n_transformer_layers
        self.n_gnn_layers        = n_gnn_layers

        # ── 1. Amino acid embedding ──────────────────────────────────────────
        self.aa_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # ── 2. Positional encoding ───────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # ── 3. Transformer encoder (or fallback linear projection) ───────────
        # ABLATION FIX: PyTorch crashes on TransformerEncoder(num_layers=0).
        # When n_transformer_layers=0 we use a two-layer MLP that keeps the
        # same embedding dimension but does NO attention — purely positional.
        if n_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model        = d_model,
                nhead          = n_heads,
                dim_feedforward= d_feedforward,
                dropout        = dropout,
                batch_first    = True,
                norm_first     = True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers           = n_transformer_layers,
                enable_nested_tensor = False,
            )
            self.use_transformer = True
        else:
            # No-attention fallback: position-aware MLP, no cross-residue info
            self.transformer = nn.Sequential(
                nn.Linear(d_model, d_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_feedforward, d_model),
                nn.LayerNorm(d_model),
            )
            self.use_transformer = False

        # ── 4. GNN layers ────────────────────────────────────────────────────
        # n_gnn_layers=0 → empty list → loop is skipped → no message passing
        self.gnn_layers = nn.ModuleList([
            ChainGNNLayer(d_model, dropout=dropout)
            for _ in range(n_gnn_layers)
        ])

        # ── 5. Projection before pair MLP ────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.GELU(),
        )

        # ── 6. Pairwise distance MLP head ────────────────────────────────────
        self.dist_head = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 2, d_proj),     nn.GELU(),
            nn.Linear(d_proj, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, seq: torch.Tensor, pad_mask=None,
                chunk_size: int = 64) -> torch.Tensor:
        """
        Args:
            seq        : (B, N)  integer-encoded sequence
            pad_mask   : (B, N)  bool, True = padding
            chunk_size : rows of the N×N matrix computed at once.
                         Reduce if you still run out of VRAM (try 32 or 16).
                         Increase for speed if VRAM allows (try 128).
        """
        B, N = seq.shape

        # ── 1. Embed + positional encoding ──────────────────────────────────
        x = self.aa_embedding(seq)
        x = self.pos_encoding(x)

        # ── 2. Transformer (or fallback MLP) ────────────────────────────────
        if self.use_transformer:
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)

        # ── 3. GNN message passing ───────────────────────────────────────────
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, pad_mask)

        # ── 4. Project to lower dimension ────────────────────────────────────
        h = self.proj(x)   # (B, N, d_proj)

        # ── 5. Chunked pairwise distance head ────────────────────────────────
        # Instead of materialising the full (B, N, N, 2*d_proj) tensor at once
        # (which is ~1 GB for N=498, d_proj=64, B=8), we process chunk_size
        # rows of the distance matrix at a time.
        # Memory per chunk: B * chunk_size * N * 2*d_proj * 4 bytes
        # e.g. chunk=64: 8 * 64 * 498 * 128 * 4 ≈ 130 MB  ✓
        rows = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            c   = end - start

            # h_i : embeddings for rows [start:end]  → (B, c, 1, d_proj)
            # h_j : all column embeddings            → (B, 1, N, d_proj)
            h_i = h[:, start:end, :].unsqueeze(2)          # (B, c, 1, d_proj)
            h_j = h.unsqueeze(1)                           # (B, 1, N, d_proj)

            # Broadcast gives (B, c, N, d_proj) without storing N×N copies
            h_i = h_i.expand(B, c, N, self.d_proj)
            h_j = h_j.expand(B, c, N, self.d_proj)

            chunk_pairs = torch.cat([h_i, h_j], dim=-1)   # (B, c, N, 2*d_proj)
            chunk_dist  = self.dist_head(chunk_pairs).squeeze(-1)  # (B, c, N)
            rows.append(chunk_dist)

        dist_matrix = torch.cat(rows, dim=1)               # (B, N, N)

        # Enforce symmetry: d_ij = d_ji
        dist_matrix = (dist_matrix + dist_matrix.transpose(1, 2)) / 2.0
        return dist_matrix


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Smoke-test all four ablation configurations
    configs = [
        ('Full (TF+GNN)',    dict(n_transformer_layers=4, n_gnn_layers=3)),
        ('No GNN',           dict(n_transformer_layers=4, n_gnn_layers=0)),
        ('No Transformer',   dict(n_transformer_layers=0, n_gnn_layers=3)),
        ('MLP only',         dict(n_transformer_layers=0, n_gnn_layers=0)),
    ]
    B, N = 2, 80
    seq      = torch.randint(1, 21, (B, N))
    pad_mask = torch.zeros(B, N, dtype=torch.bool)
    pad_mask[1, 60:] = True

    for name, cfg in configs:
        m = ProteinStructurePredictor(**cfg)
        out = m(seq, pad_mask)
        print(f"{name:25s}  params={count_parameters(m):>8,}  out={out.shape}  sym={torch.allclose(out, out.transpose(1,2), atol=1e-5)}")