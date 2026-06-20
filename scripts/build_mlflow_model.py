import shutil
from pathlib import Path
from typing import Any

import hydra
import mlflow
import mlflow.pyfunc
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.data.dvc_utils import dvc_pull_targets
from cyrillic_htr.serving.mlflow_model import TransformerHTRPyfuncModel


def prepare_serving_config(config: DictConfig) -> DictConfig:
    prepared_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    prepared_config.data.vocab_path = prepared_config.infer.vocab_path
    return prepared_config


def validate_transformer_config(config: DictConfig) -> None:
    if str(config.model.name) != "transformer_htr":
        raise ValueError(
            "MLflow serving is implemented only for Transformer HTR. "
            "Run with: model=transformer_htr +serving=mlflow"
        )


def pull_serving_artifacts(config: DictConfig) -> None:
    if not bool(config.dvc.enabled) or not bool(config.dvc.pull_on_infer):
        return

    dvc_pull_targets(
        targets=config.dvc.model_targets,
        remote=config.dvc.models_remote,
    )


def validate_artifacts(config: DictConfig) -> None:
    checkpoint_path = Path(str(config.infer.checkpoint_path))
    vocab_path = Path(str(config.infer.vocab_path))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")


def remove_existing_model(model_path: Path, overwrite: bool) -> None:
    if not model_path.exists():
        return

    if not overwrite:
        raise FileExistsError(
            f"MLflow model path already exists: {model_path}. "
            "Set +serving.overwrite=true to overwrite it."
        )

    shutil.rmtree(model_path)


def build_mlflow_model(config: DictConfig) -> None:
    config = prepare_serving_config(config)
    validate_transformer_config(config)

    pull_serving_artifacts(config)
    validate_artifacts(config)

    model_path = Path(str(config.serving.model_path))
    remove_existing_model(model_path=model_path, overwrite=bool(config.serving.overwrite))

    python_model = TransformerHTRPyfuncModel(
        config=OmegaConf.to_container(config, resolve=True),
        device=str(config.serving.device),
    )

    artifacts = {
        "checkpoint": str(config.infer.checkpoint_path),
        "vocab": str(config.infer.vocab_path),
    }

    pip_requirements = list(config.serving.pip_requirements)

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=python_model,
        artifacts=artifacts,
        code_paths=["src"],
        pip_requirements=pip_requirements,
    )

    print(f"Transformer HTR MLflow pyfunc model saved to: {model_path}")

    if bool(config.serving.log_to_tracking_server):
        log_model_to_tracking_server(
            config=config,
            python_model=python_model,
            artifacts=artifacts,
            pip_requirements=pip_requirements,
        )


def log_model_to_tracking_server(
    config: DictConfig,
    python_model: TransformerHTRPyfuncModel,
    artifacts: dict[str, str],
    pip_requirements: list[str],
) -> None:
    mlflow.set_tracking_uri(str(config.serving.tracking_uri))

    log_model_kwargs: dict[str, Any] = {
        "artifact_path": str(config.serving.artifact_path),
        "python_model": python_model,
        "artifacts": artifacts,
        "code_paths": ["src"],
        "pip_requirements": pip_requirements,
    }

    registered_model_name = config.serving.registered_model_name
    if registered_model_name:
        log_model_kwargs["registered_model_name"] = str(registered_model_name)

    with mlflow.start_run(run_name="build_transformer_htr_mlflow_serving_model"):
        mlflow.log_params(
            {
                "model_name": str(config.model.name),
                "checkpoint_path": str(config.infer.checkpoint_path),
                "vocab_path": str(config.infer.vocab_path),
                "serving_device": str(config.serving.device),
            }
        )
        mlflow.pyfunc.log_model(**log_model_kwargs)

    print(f"MLflow model logged to tracking URI: {config.serving.tracking_uri}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    build_mlflow_model(config)


if __name__ == "__main__":
    main()
