import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig


def sinusoidal(idx, d):
    half = d // 2
    freq = torch.exp(torch.arange(half, device=idx.device) * (-np.log(10000.0) / max(half - 1, 1)))
    ang = idx.unsqueeze(-1).float() * freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def block_causal_mask(turn):
    ti = turn.unsqueeze(-1)
    tj = turn.unsqueeze(-2)
    can = (tj <= ti) & (tj >= 0)
    return ~can


class DecoderOnlyDraft(nn.Module):
    def __init__(self, vocab_size, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.emb_card = nn.Embedding(vocab_size, d, padding_idx=0)
        self.emb_role = nn.Embedding(2, d)
        self.emb_pack = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, dim_feedforward=cfg.ff_dim, dropout=cfg.dropout, batch_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.n_heads = cfg.n_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)

    def embed(self, batch):
        return (self.emb_card(batch["token_card"]) + self.emb_role(batch["token_role"])
                + self.emb_pack(batch["token_pack"]) + sinusoidal(batch["token_pick"], self.d))

    def b_forward(self, x, attn_mask, attention_mask):
        kp = attention_mask == 0
        return self.backbone(x, mask=attn_mask, src_key_padding_mask=kp)

    def score(self, h, pick_batch, pick_pos, cand_idx, cand_mask):
        q = self.Wq(h[pick_batch, pick_pos])
        k = self.Wk(h[pick_batch.unsqueeze(1), cand_idx])
        score = (q.unsqueeze(1) * k).sum(-1) / np.sqrt(self.d)
        return score.masked_fill(cand_mask == 0, float("-inf"))

    def forward(self, batch, attn_mask):
        x = self.embed(batch)
        h = self.b_forward(x, attn_mask, batch["attention_mask"])
        return self.score(h, batch["pick_batch"], batch["pick_pos"], batch["cand_idx"], batch["cand_mask"])
