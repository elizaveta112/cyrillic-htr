import json
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.data.dvc_utils import dvc_pull_targets
from cyrillic_htr.data.factory import build_datamodule
from cyrillic_htr.training.lightning_modules.crnn_ctc_module import CRNNCTCLightningModule


def resolve_device(config: DictConfig) -> str:
    configured_device = str(config.infer.device)

    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return configured_device


def prepare_evaluation_config(config: DictConfig) -> DictConfig:
    """Use inference artifacts for evaluation.

    The trained checkpoint must be evaluated with the same vocabulary that was
    used during training. For exported/final models this vocabulary is stored
    next to the checkpoint in models/vocab.json and configured via
    infer.vocab_path.
    """
    prepared_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    prepared_config.data.vocab_path = prepared_config.infer.vocab_path
    return prepared_config


def load_crnn_module_from_checkpoint(
    checkpoint_path: str | Path,
    config: DictConfig,
) -> CRNNCTCLightningModule:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    lightning_module = CRNNCTCLightningModule(config=config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lightning_module.load_state_dict(checkpoint["state_dict"])

    return lightning_module


def pull_evaluation_artifacts(config: DictConfig) -> None:
    if not config.dvc.enabled or not config.dvc.pull_on_infer:
        return

    dvc_pull_targets(
        targets=config.dvc.data_targets,
        remote=config.dvc.data_remote,
    )
    dvc_pull_targets(
        targets=config.dvc.model_targets,
        remote=config.dvc.models_remote,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    config = prepare_evaluation_config(config)

    if config.model.name != "crnn_ctc":
        raise NotImplementedError("Only CRNN + CTC evaluation is implemented.")

    pull_evaluation_artifacts(config)

    device = resolve_device(config)
    accelerator = "gpu" if device == "cuda" else "cpu"

    datamodule = build_datamodule(config)
    lightning_module = load_crnn_module_from_checkpoint(
        checkpoint_path=config.infer.checkpoint_path,
        config=config,
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision=config.train.precision,
        limit_test_batches=config.train.limit_test_batches,
    )

    test_results = trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        verbose=True,
    )

    metrics = test_results[0] if test_results else {}
    metrics = {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}

    output_path = Path(config.infer.metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_path": str(config.infer.checkpoint_path),
        "vocab_path": str(config.infer.vocab_path),
        "model": config.model.name,
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Evaluation metrics saved to: {output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
