import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import EncDecDraft, ModelConfig
from featurizer import CardFeaturizer

ASSETS = Path(__file__).parent / "featurizer" / "assets"


class CardDB:
    def __init__(self, cards_path=ASSETS / "cards.json"):
        raw = json.loads(Path(cards_path).read_text(encoding="utf-8"))
        self.by_name = {}
        for c in raw:
            self._index(c["name"], c)
            for f in c.get("card_faces", []) or []:
                self._index(f.get("name", ""), c)

    def _index(self, name, card):
        if name:
            self.by_name.setdefault(name.lower(), card)

    def get(self, name):
        c = self.by_name.get(name.lower().strip())
        if c is None:
            raise KeyError(f"card not found: {name!r} (pass a dict for custom/unseen cards)")
        return c


def _card_key(card):
    return card["name"] if isinstance(card, dict) else card


class MtgDraftAssistant:
    def __init__(self, model, featurizer, db, cfg):
        self.model = model.eval()
        self.fz = featurizer
        self.db = db
        self.cfg = cfg
        self._cache = {}

    @classmethod
    def from_pretrained(cls, hf_dir=None, device="cpu"):
        hf_dir = Path(hf_dir) if hf_dir else Path(__file__).parent
        ckpt_path = next((hf_dir / "model").glob("*.ckpt"))
        cfg = ModelConfig()
        model = EncDecDraft(255, 45, cfg)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = {k.replace("model.", "", 1): v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(sd)
        fz = CardFeaturizer(device=device)
        db = CardDB()
        return cls(model, fz, db, cfg)

    def _vec(self, card):
        key = _card_key(card).lower().strip()
        if key not in self._cache:
            data = card if isinstance(card, dict) else self.db.get(card)
            self._cache[key] = self.fz.featurize(data)
        return self._cache[key]

    @torch.no_grad()
    def rank(self, pack_history, pool):
        T = len(pack_history)
        P = max(len(p) for p in pack_history)
        cards = {}
        for pk in pack_history:
            for c in pk:
                cards.setdefault(_card_key(c), c)
        for c in pool:
            cards.setdefault(_card_key(c), c)

        gid = {name: i + 1 for i, name in enumerate(sorted(cards))}
        table = np.zeros((len(gid) + 1, 255), dtype=np.float32)
        for name, card in cards.items():
            table[gid[name]] = self._vec(card)

        pack_gid = np.zeros((1, T, P), dtype=np.int64)
        for t, pk in enumerate(pack_history):
            for j, c in enumerate(pk):
                pack_gid[0, t, j] = gid[_card_key(c)]
        pickvec = np.zeros((1, T), dtype=np.int64)
        for t, c in enumerate(pool):
            pickvec[0, t] = gid[_card_key(c)]

        batch = {"pack_gid": torch.from_numpy(pack_gid), "pickvec_gid": torch.from_numpy(pickvec)}
        compact = torch.from_numpy(table)
        score = self.model(batch, compact)[0, T - 1]
        cur = [_card_key(c) for c in pack_history[-1]]
        prob = torch.softmax(score[:len(cur)], -1)
        out = [(cur[j], float(prob[j])) for j in range(len(cur))]
        return sorted(out, key=lambda x: -x[1])

    def new_draft(self):
        return Draft(self)
    
    

class Draft:
    def __init__(self, assistant):
        self.a = assistant
        self.pack_history = []
        self.pool = []

    def see(self, pack):
        self.pack_history.append(list(pack))
        return self.a.rank(self.pack_history, self.pool)

    def pick(self, card):
        self.pool.append(card)
        return card
