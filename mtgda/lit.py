import lightning as L
import torch

from .config import ModelConfig, TrainConfig
from .model import DecoderOnlyDraft
from . import engine


class DraftLit(L.LightningModule):
    def __init__(self, vocab_size, attn_mask, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["attn_mask", "model_cfg", "train_cfg"])
        self.train_cfg = train_cfg
        self.model = DecoderOnlyDraft(vocab_size, model_cfg)
        self.register_buffer("attn_mask", attn_mask)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.train_cfg.lr, weight_decay=self.train_cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.train_cfg.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def forward(self, batch):
        return self.model(batch, self.attn_mask)

    # Training
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def _step(self, batch, stage):
        score = self.model(batch, self.attn_mask)
        target_idx = engine.target_in_pack(batch["token_card"], batch["pick_batch"], batch["cand_idx"], batch["target"])
        loss = engine.masked_ce(score, batch["cand_mask"], target_idx, self.train_cfg.label_smoothing)
        metrics = engine.batch_metrics(score, target_idx)
        n = target_idx.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=n)
        self.log(f"{stage}_top1", metrics["top1"], prog_bar=True, batch_size=n)
        self.log(f"{stage}_top3", metrics["top3"], batch_size=n)
        return loss

    # Validation
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # Test
    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
