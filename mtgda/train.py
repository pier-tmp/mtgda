import tempfile

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from . import config
from .config import ModelConfig, TrainConfig
from .data import make_encdec_loaders
from .lit import EncDecLit


def run_training_encdec(model_cfg, train_cfg, data_dir=None, mlflow_uri=None, artifact_location=None,
                        experiment_name="mtgda-v2", run_name=None, num_workers=0, monitor_per_epoch=True):
    data_dir = data_dir or config.DATA_DIR
    if mlflow_uri is None:
        (config.ROOT / "volume").mkdir(parents=True, exist_ok=True)
        mlflow_uri = f"sqlite:///{config.ROOT / 'volume' / 'mlflow.db'}"
    artifact_location = artifact_location or f"file:{config.ROOT / 'volume' / 'mlartifacts'}"

    loaders, meta = make_encdec_loaders(data_dir, batch_size=train_cfg.batch_size,
                                        shuffle_within_pack=train_cfg.shuffle_within_pack,
                                        num_workers=num_workers)
    lit = EncDecLit(meta, model_cfg, train_cfg)
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_uri,
                          artifact_location=artifact_location, log_model="all", run_name=run_name)
    logger.log_hyperparams({**vars(model_cfg), **vars(train_cfg)})

    with tempfile.TemporaryDirectory() as ckpt_dir:
        ckpt = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1,
                               filename="best-{epoch}-{val_loss:.3f}")
        callbacks = [ckpt, EarlyStopping(monitor="val_loss", mode="min", patience=8)]
        trainer = L.Trainer(max_epochs=train_cfg.max_epochs, logger=logger, callbacks=callbacks,
                            accelerator="auto", devices="auto", precision="bf16-mixed")
        if monitor_per_epoch and loaders["test_holdout"] is not None:
            val_loaders = [loaders["val"], loaders["test_known"], loaders["test_holdout"]]
        else:
            val_loaders = loaders["val"]
        trainer.fit(lit, loaders["train"], val_loaders)
        lit.test_stage = "known"
        test_known = trainer.test(lit, loaders["test_known"], ckpt_path=ckpt.best_model_path)
        test_holdout = None
        if loaders["test_holdout"] is not None:
            lit.test_stage = "holdout"
            test_holdout = trainer.test(lit, loaders["test_holdout"], ckpt_path=ckpt.best_model_path)
    return {"best_val_loss": float(ckpt.best_model_score),
            "test_known": test_known[0] if test_known else None,
            "test_holdout": test_holdout[0] if test_holdout else None}


def configs_from_params(params, max_epochs=None):
    model_cfg = ModelConfig(
        d_model=params["d_model"], n_heads=params["n_heads"],
        n_layers=params["n_layers"], ff_dim=params["ff_dim"], dropout=params["dropout"])
    train_cfg = TrainConfig(
        batch_size=params["batch_size"], lr=params["lr"], weight_decay=params["weight_decay"])
    if max_epochs is not None:
        train_cfg.max_epochs = max_epochs
    return model_cfg, train_cfg


def main():
    run_training_encdec(ModelConfig(), TrainConfig())


if __name__ == "__main__":
    main()
