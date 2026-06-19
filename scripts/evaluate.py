import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from cyrillic_htr.data.dvc_utils import dvc_pull_targets
from cyrillic_htr.data.factory import build_datamodule
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


def prepare_evaluation_config(config: DictConfig) -> DictConfig:
    """Use inference artifacts for evaluation."""
    prepared_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    prepared_config.data.vocab_path = prepared_config.infer.vocab_path
    return prepared_config


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


def build_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(index): str(token) for token, index in vocab.items()}


def allowed_characters_from_vocab(vocab: dict[str, int]) -> set[str]:
    return {token for token in vocab if token not in SPECIAL_TOKENS}


def levenshtein(sequence_a: Any, sequence_b: Any) -> int:
    if len(sequence_a) < len(sequence_b):
        sequence_a, sequence_b = sequence_b, sequence_a

    previous = list(range(len(sequence_b) + 1))

    for i, item_a in enumerate(sequence_a, start=1):
        current = [i]

        for j, item_b in enumerate(sequence_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + int(item_a != item_b)
            current.append(min(insert_cost, delete_cost, replace_cost))

        previous = current

    return int(previous[-1])


def normalize_text(text: str) -> str:
    """Normalize text only for diagnostic normalized CER/WER."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().replace("ё", "е")
    characters = []

    for character in text:
        if unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)

    return re.sub(r"\s+", " ", "".join(characters)).strip()


def character_error_rate_for_one(prediction: str, reference: str) -> float:
    denominator = max(len(reference), 1)
    return float(levenshtein(prediction, reference) / denominator)


def word_error_rate_for_one(prediction: str, reference: str) -> float:
    prediction_words = str(prediction).split()
    reference_words = str(reference).split()
    denominator = max(len(reference_words), 1)
    return float(levenshtein(prediction_words, reference_words) / denominator)


def valid_character_rate(predictions: list[str], allowed_characters: set[str]) -> float:
    generated_text = "".join(predictions)

    if not generated_text:
        return 1.0

    valid_count = sum(character in allowed_characters for character in generated_text)
    return float(valid_count / len(generated_text))


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    batch_size: int,
    id_to_token: dict[int, str],
    blank_idx: int,
) -> list[str]:
    """Decode CTC outputs with greedy decoding."""
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


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved_batch = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    return moved_batch


@torch.no_grad()
def predict_batch(
    lightning_module: torch.nn.Module,
    batch: dict[str, Any],
    config: DictConfig,
    id_to_token: dict[int, str],
    blank_idx: int,
) -> list[str]:
    images = batch["images"]
    image_widths = batch.get("image_widths")
    batch_size = int(images.shape[0])

    if config.model.name == "crnn_ctc":
        model_outputs = lightning_module.model(
            images=images,
            image_widths=image_widths,
        )
        log_probs = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs

        return ctc_greedy_decode(
            log_probs=log_probs,
            batch_size=batch_size,
            id_to_token=id_to_token,
            blank_idx=blank_idx,
        )

    if config.model.name == "transformer_htr":
        token_sequences = lightning_module.model.predict(
            images=images,
            image_widths=image_widths,
            max_length=int(config.model.max_decoding_length),
        )

        return [lightning_module.decode_tokens(sequence) for sequence in token_sequences]

    raise ValueError(f"Unsupported model name: {config.model.name}")


def build_prediction_rows(
    lightning_module: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: DictConfig,
    vocab: dict[str, int],
    device: torch.device,
) -> list[dict[str, Any]]:
    id_to_token = build_id_to_token(vocab)
    blank_idx = int(vocab.get("<blank>", 0))

    lightning_module.to(device)
    lightning_module.eval()

    rows = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=True):
        batch = move_batch_to_device(batch=batch, device=device)
        references = [str(text) for text in batch["texts"]]

        image_paths = batch.get("image_paths")
        if image_paths is None:
            image_paths = [""] * len(references)
        image_paths = [str(path) for path in image_paths]

        predictions = predict_batch(
            lightning_module=lightning_module,
            batch=batch,
            config=config,
            id_to_token=id_to_token,
            blank_idx=blank_idx,
        )

        for image_path, reference, prediction in zip(
            image_paths, references, predictions, strict=True
        ):
            normalized_reference = normalize_text(reference)
            normalized_prediction = normalize_text(prediction)

            rows.append(
                {
                    "image_path": image_path,
                    "target_text": reference,
                    "predicted_text": prediction,
                    "strict_cer": character_error_rate_for_one(prediction, reference),
                    "strict_wer": word_error_rate_for_one(prediction, reference),
                    "normalized_target_text": normalized_reference,
                    "normalized_predicted_text": normalized_prediction,
                    "normalized_cer": character_error_rate_for_one(
                        normalized_prediction,
                        normalized_reference,
                    ),
                    "normalized_wer": word_error_rate_for_one(
                        normalized_prediction,
                        normalized_reference,
                    ),
                },
            )

    return rows


def build_metrics(
    rows: list[dict[str, Any]],
    vocab: dict[str, int],
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    model_name: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No prediction rows were produced during evaluation.")

    dataframe = pd.DataFrame(rows)
    predictions = dataframe["predicted_text"].fillna("").astype(str).tolist()
    references = dataframe["target_text"].fillna("").astype(str).tolist()
    strict_cer = float(dataframe["strict_cer"].mean())
    strict_wer = float(dataframe["strict_wer"].mean())
    normalized_cer = float(dataframe["normalized_cer"].mean())
    normalized_wer = float(dataframe["normalized_wer"].mean())

    line_accuracy = sum(
        int(prediction == reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(rows)

    edit_similarity = sum(
        max(0.0, 1.0 - float(cer_value)) for cer_value in dataframe["strict_cer"].tolist()
    ) / len(rows)

    allowed_characters = allowed_characters_from_vocab(vocab)

    return {
        "model": str(model_name),
        "num_samples": int(len(rows)),
        "test_cer": strict_cer,
        "test_wer": strict_wer,
        "test_line_accuracy": float(line_accuracy),
        "test_edit_similarity": float(edit_similarity),
        "test_valid_character_rate": valid_character_rate(predictions, allowed_characters),
        "strict_cer": strict_cer,
        "strict_wer": strict_wer,
        "normalized_cer": normalized_cer,
        "normalized_wer": normalized_wer,
        "checkpoint_path": str(checkpoint_path),
        "vocab_path": str(vocab_path),
    }


def resolve_predictions_path(config: DictConfig, metrics_path: Path) -> Path:
    predictions_path = OmegaConf.select(config, "infer.predictions_path", default=None)

    if predictions_path:
        return Path(str(predictions_path))

    return metrics_path.with_name("test_predictions.tsv")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    config = prepare_evaluation_config(config)
    pull_evaluation_artifacts(config)

    device = torch.device(resolve_device(config))

    vocab = load_vocab(config.infer.vocab_path)
    datamodule = build_datamodule(config)
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    lightning_module = load_lightning_module_from_checkpoint(
        checkpoint_path=config.infer.checkpoint_path,
        config=config,
    )

    rows = build_prediction_rows(
        lightning_module=lightning_module,
        dataloader=test_dataloader,
        config=config,
        vocab=vocab,
        device=device,
    )

    metrics_path = Path(config.infer.metrics_path)
    predictions_path = resolve_predictions_path(config=config, metrics_path=metrics_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_dataframe = pd.DataFrame(rows)
    predictions_dataframe.to_csv(predictions_path, sep="\t", index=False)

    metrics = build_metrics(
        rows=rows,
        vocab=vocab,
        checkpoint_path=config.infer.checkpoint_path,
        vocab_path=config.infer.vocab_path,
        model_name=config.model.name,
    )

    payload = {
        **metrics,
        "predictions_path": str(predictions_path),
        "config": OmegaConf.to_container(config, resolve=True),
    }

    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Predictions saved to: {predictions_path}")
    print(f"Evaluation metrics saved to: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
