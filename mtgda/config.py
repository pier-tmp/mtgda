from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# data.py
DATA_DIR = ROOT / "data" / "processed"
VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEED = 42

FACE_DIM = 989
N_LAYOUTS = 6
LAYOUT_FLAGS = ["single", "mdfc", "transform", "split", "adventure", "flip"]
HOLDOUT = ("SOS",)


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 3
    ff_dim: int = 128
    dropout: float = 0.32


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 5.2e-4
    weight_decay: float = 2.6e-4
    label_smoothing: float = 0.079
    aux_lambda: float = 0.1
    max_epochs: int = 40
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
    label_smoothing: tuple = None
    batch_size: list = None

    def __post_init__(self):
        if self.d_model is None:         self.d_model = [128, 256]
        if self.n_heads is None:         self.n_heads = [4, 8]
        if self.n_layers is None:        self.n_layers = (3, 6)
        if self.ff_dim is None:          self.ff_dim = [256, 512, 768]
        if self.batch_size is None:      self.batch_size = [128, 256]
        if self.dropout is None:         self.dropout = (0.1, 0.4)
        if self.lr is None:              self.lr = (1e-4, 3e-3, "log")
        if self.weight_decay is None:    self.weight_decay = (1e-4, 1e-2, "log")
        if self.label_smoothing is None: self.label_smoothing = (0.05, 0.13)
