import lightning as L
import torch

from .config import ModelConfig, TrainConfig
from .model import EncDecDraft
from . import engine


class EncDecLit(L.LightningModule):
    def __init__(self, meta, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.train_cfg = train_cfg
        self.model = EncDecDraft(meta["compact_dim"], meta["t_max"], model_cfg)
        self.register_buffer("compact", meta["compact"])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.train_cfg.lr,
                                weight_decay=self.train_cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.train_cfg.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_train_epoch_start(self):
        warmup = self.train_cfg.warmup_epochs
        if warmup and self.current_epoch < warmup:
            scale = (self.current_epoch + 1) / (warmup + 1)
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.train_cfg.lr * scale
        elif warmup and self.current_epoch == warmup:
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.train_cfg.lr
        self.log("lr", self.optimizers().param_groups[0]["lr"])

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, ("val", "monitor_known", "monitor_holdout")[dataloader_idx])

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test_" + getattr(self, "test_stage", "known"))

    def _step(self, batch, stage):
        score = self.model(batch, self.compact)
        loss = engine.encdec_loss(score, batch["step_valid"], batch["pick_idx"])
        m = engine.encdec_metrics(score, batch["step_valid"], batch["pick_idx"])
        n = int(batch["step_valid"].sum())
        kw = dict(batch_size=max(n, 1), add_dataloader_idx=False)
        self.log(f"{stage}_loss", loss, prog_bar=True, **kw)
        self.log(f"{stage}_top1", m["top1"], prog_bar=True, **kw)
        self.log(f"{stage}_top3", m["top3"], prog_bar=True, **kw)
        return loss
