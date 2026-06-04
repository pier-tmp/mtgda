import argparse
import json
from pathlib import Path

import lightning as L
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

from . import config
from .config import ModelConfig, TrainConfig, TuneConfig
from .data import make_dataloaders
from .lit import DraftLit


def _suggest_float(trial, name, spec):
    log = len(spec) == 3 and spec[2] == "log"
    return trial.suggest_float(name, spec[0], spec[1], log=log)


def suggest_cfg(trial, space):
    d_model = trial.suggest_categorical("d_model", space.d_model)
    heads = [h for h in space.n_heads if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", heads)
    model_cfg = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=trial.suggest_int("n_layers", space.n_layers[0], space.n_layers[1]),
        ff_dim=trial.suggest_categorical("ff_dim", space.ff_dim),
        dropout=_suggest_float(trial, "dropout", space.dropout),
    )
    train_cfg = TrainConfig(
        batch_size=trial.suggest_categorical("batch_size", space.batch_size),
        lr=_suggest_float(trial, "lr", space.lr),
        weight_decay=_suggest_float(trial, "weight_decay", space.weight_decay),
        label_smoothing=_suggest_float(trial, "label_smoothing", space.label_smoothing),
    )
    return model_cfg, train_cfg


def objective(trial, space, epochs, data_dir):
    model_cfg, train_cfg = suggest_cfg(trial, space)
    loaders, meta = make_dataloaders(
        data_dir, batch_size=train_cfg.batch_size,
        shuffle_within_pack=train_cfg.shuffle_within_pack)
    lit = DraftLit(meta, model_cfg, train_cfg)
    pruning = PyTorchLightningPruningCallback(trial, monitor="val_top3")
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        accelerator="auto",
        devices="auto",
        callbacks=[pruning],
    )
    try:
        trainer.fit(lit, loaders["train"], loaders["val"])
    except torch.cuda.OutOfMemoryError:
        del lit, trainer
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()
    pruning.check_pruned()
    return float(trainer.callback_metrics["val_top3"])


def run_study(data_dir, study_db, n_trials, epochs, study_name="mtgda", fresh=False, space=None):
    data_dir = data_dir or config.DATA_DIR
    space = space or TuneConfig()
    Path(study_db).parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_db}"
    if fresh:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="maximize", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(lambda t: objective(t, space, epochs, data_dir), n_trials=n_trials)
    best = {"best_value": study.best_value, "best_params": study.best_params,
            "best_trial": study.best_trial.number}
    Path(study_db).parent.mkdir(parents=True, exist_ok=True)
    (Path(study_db).parent / "best_params.json").write_text(json.dumps(best, indent=2))
    return study


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--study-db", default=str(config.ROOT / "volume" / "study.db"))
    p.add_argument("--study-name", default="mtgda")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_study(args.data_dir, args.study_db, args.n_trials, args.epochs,
              study_name=args.study_name, fresh=args.fresh)


if __name__ == "__main__":
    main()
