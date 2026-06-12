import subprocess
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.data.dvc_utils import dvc_pull_targets
from cyrillic_htr.data.factory import build_datamodule
from cyrillic_htr.training.lightning_modules.crnn_ctc_module import CRNNCTCLightningModule
from cyrillic_htr.training.lightning_modules.transformer_htr_module import (
    TransformerHTRLightningModule,
)
from cyrillic_htr.training.plots import plot_training_metrics


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


def get_resume_checkpoint_path(config: DictConfig) -> str | None:
    checkpoint_path = config.train.get("resume_from_checkpoint")

    if checkpoint_path in {None, "", "null"}:
        return None

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    return str(checkpoint_path)


def pull_training_data(config: DictConfig) -> None:
    if not config.dvc.enabled or not config.dvc.pull_on_train:
        return

    dvc_pull_targets(
        targets=config.dvc.data_targets,
        remote=config.dvc.data_remote,
    )


def build_loggers(
    config: DictConfig,
) -> tuple[list[CSVLogger | MLFlowLogger], CSVLogger, MLFlowLogger | None]:
    csv_logger = CSVLogger(
        save_dir=config.train.log_dir,
        name="csv",
    )
    loggers: list[CSVLogger | MLFlowLogger] = [csv_logger]

    mlflow_logger = None
    if config.logger.enabled:
        mlflow_logger = MLFlowLogger(
            experiment_name=config.logger.experiment_name,
            tracking_uri=config.logger.tracking_uri,
        )
        mlflow_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        mlflow_logger.experiment.set_tag(
            mlflow_logger.run_id,
            "git_commit_id",
            get_git_commit_id(),
        )
        loggers.append(mlflow_logger)

    return loggers, csv_logger, mlflow_logger


def build_lightning_module(config: DictConfig) -> L.LightningModule:
    if config.model.name == "crnn_ctc":
        return CRNNCTCLightningModule(config)

    if config.model.name == "transformer_htr":
        return TransformerHTRLightningModule(config)

    raise ValueError(f"Unknown model name: {config.model.name}")


def train_model(config: DictConfig) -> None:
    pull_training_data(config)

    save_dir = Path(config.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(config.train.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config)
    lightning_module = build_lightning_module(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-{epoch:02d}-{val_cer:.4f}",
        monitor=config.train.best_metric,
        mode="min" if config.train.minimize_metric else "max",
        save_top_k=1,
        save_last=True,
    )

    loggers, csv_logger, mlflow_logger = build_loggers(config)
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]

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
        logger=loggers,
        callbacks=callbacks,
    )

    resume_checkpoint_path = get_resume_checkpoint_path(config)

    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=resume_checkpoint_path,
    )

    trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path="best")

    metrics_csv_path = Path(csv_logger.log_dir) / "metrics.csv"
    plot_training_metrics(
        metrics_csv_path=metrics_csv_path,
        output_dir=plots_dir,
    )

    if mlflow_logger is not None:
        mlflow_logger.experiment.log_artifacts(
            run_id=mlflow_logger.run_id,
            local_dir=str(plots_dir),
            artifact_path="plots",
        )
