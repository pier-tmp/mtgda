import lightning as L
import torch

from .config import ModelConfig, TrainConfig
from .model import DecoderOnlyDraft
from . import engine


class DraftLit(L.LightningModule):
    def __init__(self, meta, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.train_cfg = train_cfg
        self.model = DecoderOnlyDraft(meta["n_cards"], meta["face_dim"],
                                      meta["n_layouts"], model_cfg)
        self.register_buffer("face_vecs", meta["cards"]["face_vecs"])
        self.register_buffer("face_mask", meta["cards"]["face_mask"])
        self.register_buffer("layout", meta["cards"]["layout"])
        self.register_buffer("aux_target", meta["aux_target"])
        self.register_buffer("aux_mask", meta["aux_mask"])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.train_cfg.lr,
                                weight_decay=self.train_cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.train_cfg.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def _cards(self):
        return {"face_vecs": self.face_vecs, "face_mask": self.face_mask, "layout": self.layout}

    def forward(self, batch):
        return self.model(batch, self._cards())

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _step(self, batch, stage):
        score, aux = self.model(batch, self._cards())
        target_idx = engine.target_in_pack(batch["token_card"], batch["pick_batch"],
                                            batch["cand_idx"], batch["target"])
        pick_loss = engine.masked_ce(score, batch["cand_mask"], target_idx,
                                     self.train_cfg.label_smoothing)
        aloss = engine.aux_loss(aux, self.aux_target, self.aux_mask)
        loss = pick_loss + self.train_cfg.aux_lambda * aloss
        metrics = engine.batch_metrics(score, target_idx)
        n = target_idx.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=n)
        self.log(f"{stage}_pick_loss", pick_loss, batch_size=n)
        self.log(f"{stage}_aux_loss", aloss, batch_size=n)
        self.log(f"{stage}_top1", metrics["top1"], prog_bar=True, batch_size=n)
        self.log(f"{stage}_top3", metrics["top3"], prog_bar=True, batch_size=n)
        return loss
