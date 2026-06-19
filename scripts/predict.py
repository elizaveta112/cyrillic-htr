import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from cyrillic_htr.data.dvc_utils import dvc_pull_targets
from cyrillic_htr.training.lightning_modules.crnn_ctc_module import CRNNCTCLightningModule
from cyrillic_htr.training.lightning_modules.transformer_htr_module import (
    TransformerHTRLightningModule,
)

SPECIAL_TOKENS = {"<blank>", "<pad>", "<sos>", "<eos>", "<unk>"}


def resolve_device(config: DictConfig) -> str:
    configured_device = str(config.infer.device)

    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return configured_device


def prepare_inference_config(config: DictConfig) -> DictConfig:
    prepared_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    prepared_config.data.vocab_path = prepared_config.infer.vocab_path
    return prepared_config


def pull_inference_artifacts(config: DictConfig) -> None:
    if not config.dvc.enabled or not config.dvc.pull_on_infer:
        return

    dvc_pull_targets(
        targets=config.dvc.model_targets,
        remote=config.dvc.models_remote,
    )


def load_vocab(vocab_path: str | Path) -> dict[str, int]:
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    payload = json.loads(vocab_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and all(isinstance(value, int) for value in payload.values()):
        return {str(token): int(index) for token, index in payload.items()}

    if isinstance(payload, dict) and "token_to_idx" in payload:
        token_to_idx = payload["token_to_idx"]
        return {str(token): int(index) for token, index in token_to_idx.items()}

    if isinstance(payload, list):
        return {str(token): index for index, token in enumerate(payload)}

    raise ValueError(f"Unsupported vocabulary format: {vocab_path}")


def build_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(index): str(token) for token, index in vocab.items()}


def load_lightning_module_from_checkpoint(
    checkpoint_path: str | Path,
    config: DictConfig,
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config.model.name == "crnn_ctc":
        lightning_module = CRNNCTCLightningModule(config=config)
    elif config.model.name == "transformer_htr":
        lightning_module = TransformerHTRLightningModule(config=config)
    else:
        raise ValueError(f"Unsupported model name: {config.model.name}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing_keys, unexpected_keys = lightning_module.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: missing checkpoint keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: unexpected checkpoint keys: {unexpected_keys[:10]}")

    return lightning_module


def preprocess_image(
    image_path: str | Path,
    image_height: int,
    max_width: int,
    image_mean: float,
    image_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("L")
    original_width, original_height = image.size

    if original_height <= 0 or original_width <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    resized_width = int(round(original_width * image_height / original_height))
    resized_width = max(1, min(resized_width, max_width))

    image = image.resize((resized_width, image_height), Image.BILINEAR)

    canvas = Image.new("L", (max_width, image_height), color=255)
    canvas.paste(image, (0, 0))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - image_mean) / image_std

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    image_widths = torch.tensor([resized_width], dtype=torch.long)

    return tensor, image_widths


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    batch_size: int,
    id_to_token: dict[int, str],
    blank_idx: int,
) -> list[str]:
    if log_probs.ndim != 3:
        raise ValueError(f"Expected CTC output with 3 dimensions, got {log_probs.shape}")

    if log_probs.shape[0] == batch_size:
        batch_first_logits = log_probs
    elif log_probs.shape[1] == batch_size:
        batch_first_logits = log_probs.transpose(0, 1)
    else:
        raise ValueError(
            "Cannot infer batch dimension for CTC output "
            f"with shape {tuple(log_probs.shape)} and batch_size={batch_size}",
        )

    predicted_ids = batch_first_logits.argmax(dim=-1).detach().cpu()
    predictions = []

    for sequence in predicted_ids.tolist():
        characters = []
        previous_idx = None

        for token_idx in sequence:
            token_idx = int(token_idx)

            if token_idx != blank_idx and token_idx != previous_idx:
                token = id_to_token.get(token_idx, "")
                if token not in SPECIAL_TOKENS:
                    characters.append(token)

            previous_idx = token_idx

        predictions.append("".join(characters))

    return predictions


@torch.no_grad()
def predict_one(
    lightning_module: torch.nn.Module,
    image_path: str | Path,
    config: DictConfig,
    vocab: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    images, image_widths = preprocess_image(
        image_path=image_path,
        image_height=int(config.data.image_height),
        max_width=int(config.data.max_width),
        image_mean=float(config.data.image_mean),
        image_std=float(config.data.image_std),
    )

    images = images.to(device)
    image_widths = image_widths.to(device)

    if config.model.name == "crnn_ctc":
        id_to_token = build_id_to_token(vocab)
        blank_idx = int(vocab.get("<blank>", 0))
        model_outputs = lightning_module.model(
            images=images,
            image_widths=image_widths,
        )
        log_probs = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        prediction = ctc_greedy_decode(
            log_probs=log_probs,
            batch_size=1,
            id_to_token=id_to_token,
            blank_idx=blank_idx,
        )[0]
    elif config.model.name == "transformer_htr":
        token_sequences = lightning_module.model.predict(
            images=images,
            image_widths=image_widths,
            max_length=int(config.model.max_decoding_length),
        )
        prediction = lightning_module.decode_tokens(token_sequences[0])
    else:
        raise ValueError(f"Unsupported model name: {config.model.name}")

    return {
        "image_path": str(image_path),
        "prediction": prediction,
    }


def collect_input_images(config: DictConfig) -> list[Path]:
    image_path = OmegaConf.select(config, "infer.image_path", default=None)
    input_path = OmegaConf.select(config, "infer.input_path", default=None)
    images_dir = OmegaConf.select(config, "infer.images_dir", default=None)

    if image_path:
        return [Path(str(image_path))]

    if input_path:
        path = Path(str(input_path))
        if path.is_dir():
            return sorted(
                candidate
                for candidate in path.iterdir()
                if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            )
        return [path]

    if images_dir:
        path = Path(str(images_dir))
        return sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )

    raise ValueError(
        "Provide an input image with +infer.image_path=... "
        "or a directory with +infer.images_dir=...",
    )


def resolve_output_path(config: DictConfig) -> Path | None:
    output_path = OmegaConf.select(config, "infer.output_path", default=None)

    if output_path:
        return Path(str(output_path))

    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    config = prepare_inference_config(config)
    pull_inference_artifacts(config)

    device = torch.device(resolve_device(config))
    vocab = load_vocab(config.infer.vocab_path)

    lightning_module = load_lightning_module_from_checkpoint(
        checkpoint_path=config.infer.checkpoint_path,
        config=config,
    )
    lightning_module.to(device)
    lightning_module.eval()

    image_paths = collect_input_images(config)
    results = [
        predict_one(
            lightning_module=lightning_module,
            image_path=image_path,
            config=config,
            vocab=vocab,
            device=device,
        )
        for image_path in image_paths
    ]

    payload = {
        "model": str(config.model.name),
        "checkpoint_path": str(config.infer.checkpoint_path),
        "vocab_path": str(config.infer.vocab_path),
        "predictions": results,
    }

    output_path = resolve_output_path(config)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Predictions saved to: {output_path}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
