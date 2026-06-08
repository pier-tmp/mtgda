import re
import json
from pathlib import Path

import numpy as np
import joblib

ASSETS = Path(__file__).parent / "assets"
N_PCA = 32
PIPS = ["W", "U", "B", "R", "G", "C"]

_COLOR = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green", "C": "colorless"}
_NUM = {str(i): w for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_SPECIAL = {"T": "tap", "Q": "untap", "E": "energy", "S": "snow mana", "P": "phyrexian mana",
            "X": "variable x mana", "Y": "variable y mana", "Z": "variable z mana"}
_TOK = re.compile(r"\{([^}]+)\}")


def _one(inner):
    s = inner.strip().upper()
    if s in _SPECIAL: return _SPECIAL[s]
    if s in _COLOR:   return f"{_COLOR[s]} mana"
    if s.isdigit():   return f"{_NUM.get(s, s)} generic mana"
    if "/" in s:
        parts = s.split("/"); phy = "P" in parts
        parts = [p for p in parts if p != "P"]
        w = [_COLOR[p] if p in _COLOR else
             (f"{_NUM.get(p, p)} generic" if p.isdigit() else p.lower()) for p in parts]
        j = " or ".join(w)
        return f"{j} phyrexian mana" if phy else f"{j} mana"
    return f"{s.lower()} symbol"


def _expand_symbols(text):
    out = _TOK.sub(lambda m: f" {_one(m.group(1))} ", text or "")
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.:;])", r"\1", out)
    return out.strip()


def _self_name_forms(card_name):
    forms = {card_name}
    short = card_name.split(",")[0].strip()
    forms.add(short)
    words = short.split()
    if len(words) > 1:
        forms.add(words[0])
        forms.add(" ".join(words[:2]))
    return forms


def _normalize_oracle(card_name, text):
    out = text or ""
    for cand in sorted(_self_name_forms(card_name), key=len, reverse=True):
        out = re.sub(r"\b" + re.escape(cand) + r"\b", "this card", out)
    return _expand_symbols(out)


def _typeline_text(face):
    tl = (face.get("type_line", "") or "").strip()
    return f"This card is a {tl}. " if tl else ""


def _parse_cost(mana_cost):
    pips = re.findall(r"\{([^}]+)\}", mana_cost or "")
    counts = {c: 0.0 for c in PIPS}
    generic = 0.0
    mv = 0.0
    for p in pips:
        if p.isdigit():
            generic += float(p); mv += float(p)
        elif p in ("X", "Y", "Z"):
            pass
        else:
            mv += 1.0
            for c in PIPS:
                if c in p: counts[c] += 1.0
    return mv, generic, counts


def _ptval(x):
    try:
        return float(x), 0.0
    except (TypeError, ValueError):
        return 0.0, (1.0 if x not in (None, "") else 0.0)


def _numeric_vec(face):
    mc = face.get("mana_cost")
    castable = 1.0 if (mc not in (None, "")) else 0.0
    mv, generic, counts = _parse_cost(mc)
    has_p = "power" in face and face.get("power") is not None
    has_t = "toughness" in face and face.get("toughness") is not None
    has_l = "loyalty" in face and face.get("loyalty") is not None
    pwr, var_p = _ptval(face.get("power")) if has_p else (0.0, 0.0)
    tou, var_t = _ptval(face.get("toughness")) if has_t else (0.0, 0.0)
    loy, _ = _ptval(face.get("loyalty")) if has_l else (0.0, 0.0)
    vec = [
        mv / 10.0, castable,
        generic / 10.0,
        pwr / 10.0, 1.0 if has_p else 0.0,
        tou / 10.0, 1.0 if has_t else 0.0,
        loy / 10.0, 1.0 if has_l else 0.0,
        1.0 if (var_p or var_t) else 0.0,
    ]
    vec += [counts[c] / 5.0 for c in PIPS]
    return np.array(vec, dtype=np.float32)


def _faces_of(card):
    if "card_faces" in card and card["card_faces"]:
        out = []
        for f in card["card_faces"]:
            g = dict(f)
            g.setdefault("name", card["name"])
            out.append(g)
        return out
    return [card]


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class CardFeaturizer:
    def __init__(self, sbert_model="all-MiniLM-L6-v2", assets=ASSETS, device=None):
        from sentence_transformers import SentenceTransformer
        self.stm = SentenceTransformer(sbert_model, device=device)
        self.pca = joblib.load(Path(assets) / "sbert_pca.joblib")
        self.patterns = self._load_patterns(Path(assets) / "patterns")

    @staticmethod
    def _load_patterns(pdir):
        out = []
        for jf in sorted(Path(pdir).glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            for p in d["patterns"]:
                out.append(re.compile(p["regex"]))
        return out

    def _feature_vec(self, text):
        t = (text or "").lower()
        return np.array([1.0 if rx.search(t) else 0.0 for rx in self.patterns], dtype=np.float32)

    def _face_607(self, card, faces, sb):
        out = []
        for i, f in enumerate(faces):
            nm = _l2(_numeric_vec(f))
            ft = self._feature_vec(f.get("oracle_text", ""))
            out.append(np.concatenate([_l2(sb[i]), nm, ft]).astype(np.float32))
        return np.stack(out)

    def featurize_batch(self, cards):
        all_texts, spans = [], []
        per_card_faces = []
        for card in cards:
            faces = _faces_of(card)
            per_card_faces.append(faces)
            start = len(all_texts)
            for f in faces:
                all_texts.append(_typeline_text(f) + _normalize_oracle(f.get("name", card["name"]),
                                                                       f.get("oracle_text", "")))
            spans.append((start, len(all_texts)))
        sb = self.stm.encode(all_texts, convert_to_numpy=True,
                             normalize_embeddings=True).astype(np.float32)
        out = []
        for (a, b), faces in zip(spans, per_card_faces):
            face607 = self._face_607(None, faces, sb[a:b])
            fused = np.maximum.reduce(face607, 0) if face607.shape[0] > 1 else face607[0]
            sb32 = self.pca.transform(fused[None, 0:384])[0]
            out.append(np.concatenate([sb32, fused[384:400], fused[400:607]]).astype(np.float32))
        return np.stack(out)

    def featurize(self, card):
        return self.featurize_batch([card])[0]
