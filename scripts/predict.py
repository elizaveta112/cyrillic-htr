from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from cyrillic_htr.data.dvc_utils import dvc_pull_targets
from cyrillic_htr.data.factory import build_datamodule
from cyrillic_htr.inference.ctc_prediction import (
    build_index_to_token,
    character_error_rate,
    greedy_ctc_decode,
    word_error_rate,
)
from cyrillic_htr.training.lightning_modules.crnn_ctc_module import CRNNCTCLightningModule


def pull_prediction_artifacts(config: DictConfig) -> None:
    if not config.dvc.enabled or not config.dvc.pull_on_infer:
        return

    dvc_pull_targets(
        targets=config.dvc.data_targets,
        remote=config.dvc.data_remote,
    )


def resolve_device(config: DictConfig) -> torch.device:
    configured_device = str(config.infer.device)

    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(configured_device)


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


def get_batch_value(batch: dict[str, Any], plural_key: str, singular_key: str) -> Any:
    if plural_key in batch:
        return batch[plural_key]

    if singular_key in batch:
        return batch[singular_key]

    raise KeyError(f"Batch does not contain '{plural_key}' or '{singular_key}'")


def run_model(
    lightning_module: CRNNCTCLightningModule,
    images: torch.Tensor,
    image_widths: torch.Tensor | None,
) -> torch.Tensor:
    try:
        if image_widths is not None:
            return lightning_module.model(images, image_widths)
    except TypeError:
        pass

    return lightning_module.model(images)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if config.model.name != "crnn_ctc":
        raise NotImplementedError("Only CRNN + CTC prediction is implemented.")

    pull_prediction_artifacts(config)

    device = resolve_device(config)
    index_to_token = build_index_to_token(config.infer.vocab_path)

    datamodule = build_datamodule(config)
    datamodule.setup(stage="test")

    lightning_module = load_crnn_module_from_checkpoint(
        checkpoint_path=config.infer.checkpoint_path,
        config=config,
    )
    lightning_module.to(device)
    lightning_module.eval()

    rows = []
    max_predictions = config.infer.max_predictions

    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            images = get_batch_value(batch, "images", "image").to(device)

            image_widths = None
            if "image_widths" in batch:
                image_widths = batch["image_widths"].to(device)
            elif "image_width" in batch:
                image_widths = batch["image_width"].to(device)

            target_texts = get_batch_value(batch, "texts", "text")
            image_paths = get_batch_value(batch, "image_paths", "image_path")

            logits = run_model(
                lightning_module=lightning_module,
                images=images,
                image_widths=image_widths,
            )

            predicted_texts = greedy_ctc_decode(
                logits=logits,
                index_to_token=index_to_token,
                blank_index=0,
                batch_first=False,
            )

            for image_path, target_text, predicted_text in zip(
                image_paths,
                target_texts,
                predicted_texts,
                strict=True,
            ):
                target_text = str(target_text)
                predicted_text = str(predicted_text)

                rows.append(
                    {
                        "image_path": str(image_path),
                        "target_text": target_text,
                        "predicted_text": predicted_text,
                        "cer": character_error_rate(
                            target_text=target_text,
                            predicted_text=predicted_text,
                        ),
                        "wer": word_error_rate(
                            target_text=target_text,
                            predicted_text=predicted_text,
                        ),
                        "is_exact_match": target_text == predicted_text,
                    }
                )

                if max_predictions is not None and len(rows) >= int(max_predictions):
                    break

            if max_predictions is not None and len(rows) >= int(max_predictions):
                break

    output_path = Path(config.infer.predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, sep="\t", index=False)

    print(f"Predictions saved to: {output_path}")
    print(f"Prediction rows: {len(dataframe)}")

    if not dataframe.empty:
        print(f"Mean CER: {dataframe['cer'].mean():.6f}")
        print(f"Mean WER: {dataframe['wer'].mean():.6f}")
        print(f"Line accuracy: {dataframe['is_exact_match'].mean():.6f}")


if __name__ == "__main__":
    main()
