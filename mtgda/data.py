import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from . import config


def load_cards(data_dir):
    data_dir = Path(data_dir)
    cv = np.load(data_dir / "card_vectors.npz")
    oracle_ids = list(cv.files)
    oracle2gid = {oid: i + 1 for i, oid in enumerate(oracle_ids)}
    n = len(oracle_ids)
    return {"oracle2gid": oracle2gid, "n_cards": n, "vocab_size": n + 2}


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


def load_compact(data_dir, oracle2gid, n_cards):
    z = np.load(Path(data_dir) / "card_compact.npz")
    dim = z[z.files[0]].shape[0]
    table = np.zeros((n_cards + 2, dim), dtype=np.float32)
    for oid, gid in oracle2gid.items():
        if oid in z.files:
            table[gid] = z[oid]
    return table, dim


class EncDecDataset(Dataset):
    def __init__(self, records, t_max, p_max, shuffle_within_pack=False):
        self.records = records
        self.t_max, self.p_max = t_max, p_max
        self.shuffle_within_pack = shuffle_within_pack
        self.index = [(si, di) for si, s in enumerate(records)
                      for di in range(s["pack_cards"].shape[0])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, di = self.index[i]
        s = self.records[si]
        T, P = self.t_max, self.p_max
        pc = s["pack_cards"][di]
        ps = s["pack_size"][di]
        pk = s["picked_card"][di]
        pack_gid = np.zeros((T, P), dtype=np.int64)
        pickvec_gid = np.zeros(T, dtype=np.int64)
        step_valid = np.zeros(T, dtype=bool)
        pick_idx = np.zeros(T, dtype=np.int64)
        for t in range(min(T, pc.shape[0])):
            n = int(ps[t])
            cards = [int(c) for c in pc[t, :n] if c != 0][:P]
            if not cards:
                continue
            if self.shuffle_within_pack and len(cards) > 1:
                np.random.shuffle(cards)
            for j, c in enumerate(cards):
                pack_gid[t, j] = c
            pickvec_gid[t] = int(pk[t])
            if len(cards) > 1 and int(pk[t]) in cards:
                pick_idx[t] = cards.index(int(pk[t]))
                step_valid[t] = True
        return {"pack_gid": pack_gid, "pickvec_gid": pickvec_gid,
                "step_valid": step_valid, "pick_idx": pick_idx}


def collate_encdec(batch):
    t = torch.from_numpy
    stack = lambda k: t(np.stack([b[k] for b in batch]))
    return {
        "pack_gid": stack("pack_gid"),
        "pickvec_gid": stack("pickvec_gid"),
        "step_valid": stack("step_valid"),
        "pick_idx": stack("pick_idx"),
    }


def make_encdec_loaders(data_dir=config.DATA_DIR, batch_size=64, val_frac=config.VAL_FRAC,
                        test_frac=config.TEST_FRAC, seed=config.SEED, num_workers=0,
                        shuffle_within_pack=True, holdout=config.HOLDOUT):
    data_dir = Path(data_dir)
    cards = load_cards(data_dir)
    o2g, n_cards = cards["oracle2gid"], cards["n_cards"]
    compact, cdim = load_compact(data_dir, o2g, n_cards)

    set_dirs = [p.name for p in sorted(data_dir.iterdir())
                if p.is_dir() and (p / "drafts.npz").exists()]
    holdout = set(holdout)
    train_sets = [s for s in set_dirs if s not in holdout]
    test_sets = [s for s in set_dirs if s in holdout]
    train_records = [remap_set(data_dir, s, o2g) for s in train_sets]
    holdout_records = [remap_set(data_dir, s, o2g) for s in test_sets]

    t_max = max(r["pack_cards"].shape[1] for r in train_records + holdout_records)
    p_max = max(r["pack_cards"].shape[2] for r in train_records + holdout_records)

    tr_idx, va_idx, tk_idx = stratified_split(train_records, val_frac, test_frac, seed)
    train_full = EncDecDataset(train_records, t_max, p_max, shuffle_within_pack=shuffle_within_pack)
    eval_full = EncDecDataset(train_records, t_max, p_max, shuffle_within_pack=False)
    pos = {p: i for i, p in enumerate(train_full.index)}
    train_ds = torch.utils.data.Subset(train_full, [pos[p] for p in tr_idx])
    val_ds = torch.utils.data.Subset(eval_full, [pos[p] for p in va_idx])
    test_known_ds = torch.utils.data.Subset(eval_full, [pos[p] for p in tk_idx])
    test_holdout_ds = (EncDecDataset(holdout_records, t_max, p_max, shuffle_within_pack=False)
                       if holdout_records else None)

    def dl(ds, shuffle=False, drop_last=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=collate_encdec, drop_last=drop_last,
                          pin_memory=num_workers > 0, persistent_workers=num_workers > 0)

    meta = {
        "n_cards": n_cards, "vocab_size": cards["vocab_size"],
        "compact_dim": cdim, "t_max": t_max, "p_max": p_max,
        "compact": torch.from_numpy(compact),
        "train_sets": train_sets, "holdout_sets": test_sets,
    }
    return {
        "train": dl(train_ds, shuffle=True, drop_last=True),
        "val": dl(val_ds),
        "test_known": dl(test_known_ds),
        "test_holdout": dl(test_holdout_ds) if test_holdout_ds else None,
    }, meta
