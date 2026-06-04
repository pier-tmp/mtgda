import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from . import config

def unroll_draft(pack_cards_d, picked_card_d, pack_pick_d, pack_size_d, picks_per_pack, pick_id):
    tok_card, tok_role, tok_pack, tok_pick, tok_turn = [], [], [], [], []
    pick_pos, cand_idx, target = [], [], []
    for t in range(len(pack_size_d)):
        n = int(pack_size_d[t])
        if n == 0:
            continue
        pack_no = t // picks_per_pack
        pk = int(pack_pick_d[t])
        cands = []
        for c in pack_cards_d[t, :n]:
            cands.append(len(tok_card))
            tok_card.append(int(c)); tok_role.append(0)
            tok_pack.append(pack_no); tok_pick.append(pk); tok_turn.append(t)
        if n > 1:
            pick_pos.append(len(tok_card))
            cand_idx.append(cands)
            target.append(int(picked_card_d[t]))
        tok_card.append(pick_id); tok_role.append(1)
        tok_pack.append(pack_no); tok_pick.append(pk); tok_turn.append(t)
    return {
        "token_card": np.array(tok_card, dtype=np.int64),
        "token_role": np.array(tok_role, dtype=np.int64),
        "token_pack": np.array(tok_pack, dtype=np.int64),
        "token_pick": np.array(tok_pick, dtype=np.int64),
        "token_turn": np.array(tok_turn, dtype=np.int64),
        "pick_pos":   np.array(pick_pos, dtype=np.int64),
        "cand_idx":   cand_idx,
        "target":     np.array(target, dtype=np.int64),
    }


class DraftSeqDataset(Dataset):

    def __init__(self, z, meta, indices, shuffle_within_pack=False):
        self.pack_cards  = z["pack_cards"]
        self.picked_card = z["picked_card"]
        self.pack_pick   = z["pack_pick"]
        self.pack_size   = z["pack_size"]
        self.picks_per_pack = meta["picks_per_pack"]
        self.pad = meta["pad_id"]
        self.pick_id = meta["pick_id"]
        self.indices = np.asarray(indices)
        self.shuffle_within_pack = shuffle_within_pack

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        d = int(self.indices[i])
        pc = self.pack_cards[d].copy()
        if self.shuffle_within_pack:
            for t in range(pc.shape[0]):
                n = int(self.pack_size[d, t])
                if n > 1:
                    np.random.shuffle(pc[t, :n])
        return unroll_draft(pc, self.picked_card[d], self.pack_pick[d], self.pack_size[d], self.picks_per_pack, self.pick_id)


def collate(batch):

    B = len(batch)
    L_max = max(len(b["token_card"]) for b in batch)
    P_max = max(max((len(c) for c in b["cand_idx"]), default=0) for b in batch)

    token_card = np.zeros((B, L_max), dtype=np.int64)
    token_role = np.zeros((B, L_max), dtype=np.int64)
    token_pack = np.zeros((B, L_max), dtype=np.int64)
    token_pick = np.zeros((B, L_max), dtype=np.int64)
    token_turn = np.full((B, L_max), -1, dtype=np.int64)
    attn_mask  = np.zeros((B, L_max), dtype=np.int64)

    pick_b, pick_pos, target = [], [], []
    cand_rows, cand_mask = [], []

    for bi, b in enumerate(batch):
        L = len(b["token_card"])
        token_card[bi, :L] = b["token_card"]
        token_role[bi, :L] = b["token_role"]
        token_pack[bi, :L] = b["token_pack"]
        token_pick[bi, :L] = b["token_pick"]
        token_turn[bi, :L] = b["token_turn"]
        attn_mask[bi, :L]  = 1
        for j, pos in enumerate(b["pick_pos"]):
            pick_b.append(bi); pick_pos.append(int(pos)); target.append(int(b["target"][j]))
            cands = b["cand_idx"][j]
            row  = np.zeros(P_max, dtype=np.int64)
            mrow = np.zeros(P_max, dtype=np.int64)
            row[:len(cands)]  = cands
            mrow[:len(cands)] = 1
            cand_rows.append(row); cand_mask.append(mrow)

    t = torch.from_numpy
    return {
        "token_card": t(token_card), "token_role": t(token_role),
        "token_pack": t(token_pack), "token_pick": t(token_pick),
        "token_turn": t(token_turn), "attention_mask": t(attn_mask),
        "pick_batch": torch.tensor(pick_b), "pick_pos": torch.tensor(pick_pos),
        "cand_idx":   t(np.stack(cand_rows)),
        "cand_mask":  t(np.stack(cand_mask)),
        "target":     torch.tensor(target),
    }


def split_by_draft(n_drafts, val_frac=config.VAL_FRAC, test_frac=config.TEST_FRAC, seed=config.SEED):
    rng = np.random.default_rng(seed)
    ids = np.arange(n_drafts)
    rng.shuffle(ids)
    n_val = int(n_drafts * val_frac)
    n_test = int(n_drafts * test_frac)
    val = ids[:n_val]
    test = ids[n_val:n_val + n_test]
    train = ids[n_val + n_test:]
    return np.sort(train), np.sort(val), np.sort(test)


def make_dataloaders(data_dir=config.DATA_DIR, batch_size=64, val_frac=config.VAL_FRAC, test_frac=config.TEST_FRAC, seed=config.SEED, num_workers=0, shuffle_within_pack=True):

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text())
    z = np.load(data_dir / "drafts.npz")
    n_drafts = z["pack_cards"].shape[0]

    train_idx, val_idx, test_idx = split_by_draft(n_drafts, val_frac, test_frac, seed)
    train_ds = DraftSeqDataset(z, meta, train_idx, shuffle_within_pack=shuffle_within_pack)
    val_ds = DraftSeqDataset(z, meta, val_idx,   shuffle_within_pack=False)
    test_ds = DraftSeqDataset(z, meta, test_idx,  shuffle_within_pack=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    from .model import block_causal_mask
    turn = torch.from_numpy(train_ds[0]["token_turn"])
    meta["attn_mask"] = block_causal_mask(turn)
    return train_loader, val_loader, test_loader, meta
