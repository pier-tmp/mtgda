import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from . import config
from .config import FACE_DIM, N_LAYOUTS, LAYOUT_FLAGS


def load_cards(data_dir):
    data_dir = Path(data_dir)
    cv = np.load(data_dir / "card_vectors.npz")
    meta = json.loads((data_dir / "card_meta.json").read_text(encoding="utf-8"))
    oracle_ids = list(cv.files)
    oracle2gid = {oid: i + 1 for i, oid in enumerate(oracle_ids)}
    n = len(oracle_ids)
    max_faces = max(cv[o].shape[0] for o in oracle_ids)
    face_vecs = np.zeros((n + 2, max_faces, FACE_DIM), dtype=np.float32)
    face_mask = np.zeros((n + 2, max_faces), dtype=np.float32)
    layout = np.zeros((n + 2, N_LAYOUTS), dtype=np.float32)
    li = {f: i for i, f in enumerate(LAYOUT_FLAGS)}
    for oid, gid in oracle2gid.items():
        v = cv[oid]
        face_vecs[gid, :v.shape[0]] = v
        face_mask[gid, :v.shape[0]] = 1.0
        layout[gid, li.get(meta[oid]["layout"], 0)] = 1.0
    return {
        "oracle2gid": oracle2gid, "n_cards": n,
        "pad_gid": 0, "pick_gid": n + 1, "vocab_size": n + 2,
        "face_vecs": face_vecs, "face_mask": face_mask, "layout": layout,
        "meta": meta,
    }


def _load_aux_col(path, col, oracle2gid, n_cards):
    z = np.load(path, allow_pickle=True)
    ci = list(z["cols"]).index(col)
    target = np.zeros(n_cards + 2, dtype=np.float32)
    mask = np.zeros(n_cards + 2, dtype=np.float32)
    for oid, row in zip(z["oracle_ids"], z["values"]):
        gid = oracle2gid.get(str(oid))
        if gid is not None:
            target[gid] = row[ci]
            mask[gid] = 1.0
    return target[1:1 + n_cards], mask[1:1 + n_cards]


def _zscore(t, m):
    v = t[m > 0]
    mu, sd = v.mean(), v.std()
    out = np.zeros_like(t)
    out[m > 0] = (t[m > 0] - mu) / (sd if sd > 0 else 1.0)
    return out


def load_aux(data_dir, oracle2gid, n_cards):
    data_dir = Path(data_dir)
    ata_t, ata_m = _load_aux_col(data_dir / "ata.npz", "ata_strength", oracle2gid, n_cards)
    wr_t, wr_m = _load_aux_col(data_dir / "aux.npz", "gih_wr", oracle2gid, n_cards)
    target = np.stack([_zscore(ata_t, ata_m), _zscore(wr_t, wr_m)], axis=1)
    mask = np.stack([ata_m, wr_m], axis=1)
    return target, mask


def unroll_draft(pack_cards_d, picked_card_d, pack_pick_d, pack_size_d, picks_per_pack, pick_gid):
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
        tok_card.append(pick_gid); tok_role.append(1)
        tok_pack.append(pack_no); tok_pick.append(pk); tok_turn.append(t)
    return {
        "token_card": np.array(tok_card, dtype=np.int64),
        "token_role": np.array(tok_role, dtype=np.int64),
        "token_pack": np.array(tok_pack, dtype=np.int64),
        "token_pick": np.array(tok_pick, dtype=np.int64),
        "token_turn": np.array(tok_turn, dtype=np.int64),
        "pick_pos": np.array(pick_pos, dtype=np.int64),
        "cand_idx": cand_idx,
        "target": np.array(target, dtype=np.int64),
    }


class MultiSetDraft(Dataset):
    def __init__(self, records, pick_gid, shuffle_within_pack=False):
        self.records = records
        self.pick_gid = pick_gid
        self.shuffle_within_pack = shuffle_within_pack
        self.index = []
        for si, s in enumerate(records):
            for di in range(s["pack_cards"].shape[0]):
                self.index.append((si, di))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, di = self.index[i]
        s = self.records[si]
        pc = s["pack_cards"][di].copy()
        ps = s["pack_size"][di]
        if self.shuffle_within_pack:
            for t in range(pc.shape[0]):
                n = int(ps[t])
                if n > 1:
                    np.random.shuffle(pc[t, :n])
        return unroll_draft(pc, s["picked_card"][di], s["pack_pick"][di], ps,
                            s["picks_per_pack"], self.pick_gid)


def remap_set(data_dir, set_code, oracle2gid):
    sdir = Path(data_dir) / set_code
    z = np.load(sdir / "drafts.npz")
    meta = json.loads((sdir / "metadata.json").read_text(encoding="utf-8"))
    id2oracle = json.loads((sdir / "id2oracle.json").read_text(encoding="utf-8"))
    n_local = max(int(k) for k in id2oracle) + 1
    local2gid = np.zeros(n_local, dtype=np.int64)
    for k, oid in id2oracle.items():
        local2gid[int(k)] = oracle2gid.get(oid, 0) if oid else 0
    return {
        "pack_cards": local2gid[z["pack_cards"]],
        "pack_size": z["pack_size"],
        "picked_card": local2gid[z["picked_card"]],
        "pack_pick": z["pack_pick"],
        "picks_per_pack": meta["picks_per_pack"],
    }


def collate(batch):
    B = len(batch)
    L_max = max(len(b["token_card"]) for b in batch)
    P_max = max(max((len(c) for c in b["cand_idx"]), default=0) for b in batch)

    token_card = np.zeros((B, L_max), dtype=np.int64)
    token_role = np.zeros((B, L_max), dtype=np.int64)
    token_pack = np.zeros((B, L_max), dtype=np.int64)
    token_pick = np.zeros((B, L_max), dtype=np.int64)
    token_turn = np.full((B, L_max), -1, dtype=np.int64)
    attn = np.zeros((B, L_max), dtype=np.int64)
    pick_b, pick_pos, target = [], [], []
    cand_rows, cand_mask = [], []

    for bi, b in enumerate(batch):
        L = len(b["token_card"])
        token_card[bi, :L] = b["token_card"]
        token_role[bi, :L] = b["token_role"]
        token_pack[bi, :L] = b["token_pack"]
        token_pick[bi, :L] = b["token_pick"]
        token_turn[bi, :L] = b["token_turn"]
        attn[bi, :L] = 1
        for j, pos in enumerate(b["pick_pos"]):
            pick_b.append(bi); pick_pos.append(int(pos)); target.append(int(b["target"][j]))
            cands = b["cand_idx"][j]
            row = np.zeros(P_max, dtype=np.int64)
            mrow = np.zeros(P_max, dtype=np.int64)
            row[:len(cands)] = cands
            mrow[:len(cands)] = 1
            cand_rows.append(row); cand_mask.append(mrow)

    t = torch.from_numpy
    return {
        "token_card": t(token_card), "token_role": t(token_role),
        "token_pack": t(token_pack), "token_pick": t(token_pick),
        "token_turn": t(token_turn), "attention_mask": t(attn),
        "pick_batch": torch.tensor(pick_b), "pick_pos": torch.tensor(pick_pos),
        "cand_idx": t(np.stack(cand_rows)), "cand_mask": t(np.stack(cand_mask)),
        "target": torch.tensor(target),
    }


def stratified_split(records, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for si, s in enumerate(records):
        n = s["pack_cards"].shape[0]
        ids = np.arange(n)
        rng.shuffle(ids)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val_idx += [(si, int(d)) for d in ids[:n_val]]
        test_idx += [(si, int(d)) for d in ids[n_val:n_val + n_test]]
        train_idx += [(si, int(d)) for d in ids[n_val + n_test:]]
    return train_idx, val_idx, test_idx


def make_dataloaders(data_dir=config.DATA_DIR, batch_size=64, val_frac=config.VAL_FRAC,
                     test_frac=config.TEST_FRAC, seed=config.SEED, num_workers=0,
                     shuffle_within_pack=True, holdout=config.HOLDOUT):
    data_dir = Path(data_dir)
    cards = load_cards(data_dir)
    aux_target, aux_mask = load_aux(data_dir, cards["oracle2gid"], cards["n_cards"])
    pick_gid = cards["pick_gid"]

    set_dirs = [p.name for p in sorted(data_dir.iterdir())
                if p.is_dir() and (p / "drafts.npz").exists()]
    holdout = set(holdout)
    train_sets = [s for s in set_dirs if s not in holdout]
    test_sets = [s for s in set_dirs if s in holdout]

    train_records = [remap_set(data_dir, s, cards["oracle2gid"]) for s in train_sets]
    holdout_records = [remap_set(data_dir, s, cards["oracle2gid"]) for s in test_sets]

    tr_idx, va_idx, tk_idx = stratified_split(train_records, val_frac, test_frac, seed)
    train_full = MultiSetDraft(train_records, pick_gid, shuffle_within_pack=shuffle_within_pack)
    eval_full = MultiSetDraft(train_records, pick_gid, shuffle_within_pack=False)
    pos = {p: i for i, p in enumerate(train_full.index)}
    train_ds = torch.utils.data.Subset(train_full, [pos[p] for p in tr_idx])
    val_ds = torch.utils.data.Subset(eval_full, [pos[p] for p in va_idx])
    test_known_ds = torch.utils.data.Subset(eval_full, [pos[p] for p in tk_idx])
    test_holdout_ds = (MultiSetDraft(holdout_records, pick_gid, shuffle_within_pack=False)
                       if holdout_records else None)

    def dl(ds, shuffle=False, drop_last=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=collate, drop_last=drop_last,
                          pin_memory=num_workers > 0,
                          persistent_workers=num_workers > 0)

    meta = {
        "n_cards": cards["n_cards"], "vocab_size": cards["vocab_size"],
        "face_dim": FACE_DIM, "n_layouts": N_LAYOUTS,
        "cards": {
            "face_vecs": torch.from_numpy(cards["face_vecs"]),
            "face_mask": torch.from_numpy(cards["face_mask"]),
            "layout": torch.from_numpy(cards["layout"]),
        },
        "aux_target": torch.from_numpy(aux_target),
        "aux_mask": torch.from_numpy(aux_mask),
        "train_sets": train_sets, "holdout_sets": test_sets,
    }
    return {
        "train": dl(train_ds, shuffle=True, drop_last=True),
        "val": dl(val_ds),
        "test_known": dl(test_known_ds),
        "test_holdout": dl(test_holdout_ds) if test_holdout_ds else None,
    }, meta
