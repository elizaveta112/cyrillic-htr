import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cyrillic_htr.data.vocab import load_vocab
from cyrillic_htr.inference.ctc_decoder import build_idx_to_char, ctc_greedy_decode
from cyrillic_htr.metrics.text import (
    character_error_rate,
    line_accuracy,
    normalized_edit_similarity,
    valid_character_rate,
    word_error_rate,
)
from cyrillic_htr.models.factory import build_model


class CRNNCTCLightningModule(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.config = config
        self.vocab = load_vocab(config.data.vocab_path)
        self.idx_to_char = build_idx_to_char(self.vocab)
        self.blank_idx = int(config.model.blank_idx)
        self.allowed_characters = set(self.vocab) - {"<blank>"}
        self.model = build_model(config=config, vocab_size=len(self.vocab))

        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_idx,
            reduction=config.model.ctc.reduction,
            zero_infinity=config.model.ctc.zero_infinity,
        )

        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

    def forward(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(images, image_widths)

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        log_probs, input_lengths = self(
            images=batch["images"],
            image_widths=batch["image_widths"],
        )

        loss = self.ctc_loss(
            log_probs,
            batch["targets"],
            input_lengths,
            batch["target_lengths"],
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["texts"]),
        )

        return loss

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        log_probs, input_lengths = self(
            images=batch["images"],
            image_widths=batch["image_widths"],
        )

        loss = self.ctc_loss(
            log_probs,
            batch["targets"],
            input_lengths,
            batch["target_lengths"],
        )

        predictions = ctc_greedy_decode(
            log_probs=log_probs,
            idx_to_char=self.idx_to_char,
            blank_idx=self.blank_idx,
        )
        references = list(batch["texts"])

        self._log_text_metrics(
            prefix="val",
            loss=loss,
            predictions=predictions,
            references=references,
        )

        return loss

    def test_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        log_probs, input_lengths = self(
            images=batch["images"],
            image_widths=batch["image_widths"],
        )

        loss = self.ctc_loss(
            log_probs,
            batch["targets"],
            input_lengths,
            batch["target_lengths"],
        )

        predictions = ctc_greedy_decode(
            log_probs=log_probs,
            idx_to_char=self.idx_to_char,
            blank_idx=self.blank_idx,
        )
        references = list(batch["texts"])

        self._log_text_metrics(
            prefix="test",
            loss=loss,
            predictions=predictions,
            references=references,
        )

        return loss

    def _log_text_metrics(
        self,
        prefix: str,
        loss: torch.Tensor,
        predictions: list[str],
        references: list[str],
    ) -> None:
        batch_size = len(references)

        cer = character_error_rate(predictions, references)
        wer = word_error_rate(predictions, references)
        exact_match = line_accuracy(predictions, references)
        edit_similarity = normalized_edit_similarity(predictions, references)
        valid_rate = valid_character_rate(predictions, self.allowed_characters)

        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{prefix}_cer", cer, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{prefix}_wer", wer, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            f"{prefix}_line_accuracy",
            exact_match,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}_edit_similarity",
            edit_similarity,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}_valid_character_rate",
            valid_rate,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
