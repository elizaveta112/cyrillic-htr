import subprocess
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.data.factory import build_datamodule
from cyrillic_htr.training.lightning_modules.crnn_ctc_module import CRNNCTCLightningModule


def get_git_commit_id() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return "unknown"

    return result.stdout.strip()


def build_logger(config: DictConfig) -> MLFlowLogger | bool:
    if not config.logger.enabled:
        return False

    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        tracking_uri=config.logger.tracking_uri,
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    logger.experiment.set_tag(logger.run_id, "git_commit_id", get_git_commit_id())

    return logger


def train_model(config: DictConfig) -> None:
    if config.model.name != "crnn_ctc":
        raise NotImplementedError("Only CRNN + CTC training is implemented at this stage.")

    save_dir = Path(config.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config)
    lightning_module = CRNNCTCLightningModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-{epoch:02d}-{val_cer:.4f}",
        monitor=config.train.best_metric,
        mode="min" if config.train.minimize_metric else "max",
        save_top_k=1,
        save_last=True,
    )

    logger = build_logger(config)

    callbacks = [checkpoint_callback]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = L.Trainer(
        max_epochs=config.train.epochs,
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        precision=config.train.precision,
        gradient_clip_val=config.train.gradient_clip_val,
        limit_train_batches=config.train.limit_train_batches,
        limit_val_batches=config.train.limit_val_batches,
        limit_test_batches=config.train.limit_test_batches,
        log_every_n_steps=config.train.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=lightning_module, datamodule=datamodule)
    trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path="best")
