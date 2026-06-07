from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# data.py
DATA_DIR = ROOT / "data" / "processed"
VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEED = 42

FACE_DIM = 1196
N_LAYOUTS = 6
LAYOUT_FLAGS = ["single", "mdfc", "transform", "split", "adventure", "flip"]
HOLDOUT = ("SOS",)


@dataclass
class ModelConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    warmup_epochs: int = 0
    grad_clip: float = 0.0
    shuffle_within_pack: bool = True


@dataclass
class TuneConfig:
    d_model: list = None
    n_heads: list = None
    n_layers: tuple = None
    ff_dim: list = None
    dropout: tuple = None
    lr: tuple = None
    weight_decay: tuple = None
    batch_size: list = None

    def __post_init__(self):
        if self.d_model is None:         self.d_model = [64, 96, 128]
        if self.n_heads is None:         self.n_heads = [4, 8]
        if self.n_layers is None:        self.n_layers = (2, 4)
        if self.ff_dim is None:          self.ff_dim = [128, 192, 256]
        if self.batch_size is None:      self.batch_size = [128, 256]
        if self.dropout is None:         self.dropout = (0.0, 0.3)
        if self.lr is None:              self.lr = (3e-4, 3e-3, "log")
        if self.weight_decay is None:    self.weight_decay = (1e-5, 1e-3, "log")
