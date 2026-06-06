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
    eye = torch.eye(turn.shape[-1], dtype=torch.bool, device=turn.device)
    can = can | eye
    return ~can


class CardEncoder(nn.Module):
    def __init__(self, face_dim, n_layouts, d):
        super().__init__()
        self.face_enc = nn.Sequential(
            nn.Linear(face_dim, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(d + n_layouts, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, face_vecs, face_mask, layout):
        h = self.face_enc(face_vecs)
        att = (h * self.query).sum(-1) / np.sqrt(h.shape[-1])
        att = att.masked_fill(face_mask == 0, -1e9)
        w = torch.softmax(att, dim=-1) * face_mask
        w = w.unsqueeze(-1)
        pooled = (w * h).sum(1)
        return self.proj(torch.cat([pooled, layout], dim=-1))


class DecoderOnlyDraft(nn.Module):
    def __init__(self, n_cards, face_dim, n_layouts, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.n_cards = n_cards
        self.card_encoder = CardEncoder(face_dim, n_layouts, d)
        self.pad_vec = nn.Parameter(torch.zeros(d))
        self.pick_vec = nn.Parameter(torch.randn(d) * 0.02)
        self.emb_role = nn.Embedding(2, d)
        self.emb_pack = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, dim_feedforward=cfg.ff_dim,
                                           dropout=cfg.dropout, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.n_heads = cfg.n_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.aux_head = nn.Linear(d, 2)

    def card_table(self, face_vecs, face_mask, layout):
        enc = self.card_encoder(face_vecs, face_mask, layout)
        return torch.cat([self.pad_vec.unsqueeze(0), enc, self.pick_vec.unsqueeze(0)], dim=0)

    def embed(self, batch, table):
        return (table[batch["token_card"]] + self.emb_role(batch["token_role"])
                + self.emb_pack(batch["token_pack"]) + sinusoidal(batch["token_pick"], self.d))

    def b_forward(self, x, attn_mask):
        B, L, _ = x.shape
        m = attn_mask.unsqueeze(1).expand(B, self.n_heads, L, L).reshape(B * self.n_heads, L, L)
        return self.backbone(x, mask=m)

    def score(self, h, pick_batch, pick_pos, cand_idx, cand_mask):
        q = self.Wq(h[pick_batch, pick_pos])
        k = self.Wk(h[pick_batch.unsqueeze(1), cand_idx])
        s = (q.unsqueeze(1) * k).sum(-1) / np.sqrt(self.d)
        return s.masked_fill(cand_mask == 0, float("-inf"))

    def forward(self, batch, cards):
        table = self.card_table(cards["face_vecs"], cards["face_mask"], cards["layout"])
        x = self.embed(batch, table)
        attn_mask = block_causal_mask(batch["token_turn"])
        h = self.b_forward(x, attn_mask)
        logits = self.score(h, batch["pick_batch"], batch["pick_pos"],
                            batch["cand_idx"], batch["cand_mask"])
        aux = self.aux_head(table[1:1 + self.n_cards])
        return logits, aux
