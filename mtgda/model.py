import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig


class EncDecDraft(nn.Module):
    def __init__(self, compact_dim, t_max, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.t_max = t_max
        self.proj = nn.Sequential(nn.Linear(compact_dim, d), nn.GELU(), nn.Linear(d, d))
        self.qpool = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_e = nn.Embedding(t_max, d)
        self.pos_d = nn.Embedding(t_max, d)
        enc = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff_dim, dropout=cfg.dropout,
                                         batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg.n_layers)
        dec = nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.ff_dim, dropout=cfg.dropout,
                                         batch_first=True, activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, cfg.n_layers)
        self.score_mlp = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, batch, compact):
        pack_gid = batch["pack_gid"]
        B, T, P = pack_gid.shape
        packmask = pack_gid != 0
        pv = self.proj(compact[pack_gid])
        att = (pv * self.qpool).sum(-1) / np.sqrt(self.d)
        att = att.masked_fill(~packmask, -1e9)
        w = torch.softmax(att, -1).unsqueeze(-1)
        mem = (w * pv).sum(2) + self.pos_e(torch.arange(T, device=pack_gid.device))
        mem_mask = ~packmask.any(-1)
        mem = self.encoder(mem, src_key_padding_mask=mem_mask)

        pk = self.proj(compact[batch["pickvec_gid"]])
        pk = torch.cat([torch.zeros(B, 1, self.d, device=pk.device), pk[:, :-1]], 1)
        pk = pk + self.pos_d(torch.arange(T, device=pk.device))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=pk.device)
        state = self.decoder(pk, mem, tgt_mask=causal, memory_key_padding_mask=mem_mask)

        st = state.unsqueeze(2).expand(-1, -1, P, -1)
        s = self.score_mlp(torch.cat([st, pv], -1)).squeeze(-1)
        return s.masked_fill(~packmask, -1e4)
