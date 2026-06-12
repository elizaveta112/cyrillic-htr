import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cyrillic_htr.data.vocab import load_vocab
from cyrillic_htr.metrics.text import (
    character_error_rate,
    line_accuracy,
    normalized_edit_similarity,
    valid_character_rate,
    word_error_rate,
)
from cyrillic_htr.models.factory import build_model


class TransformerHTRLightningModule(L.LightningModule):
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        raw_vocab = load_vocab(config.data.vocab_path)
        self.characters = [
            character
            for character, _ in sorted(raw_vocab.items(), key=lambda item: item[1])
            if character != ""
        ]

        self.char_to_token = {
            character: index + 3 for index, character in enumerate(self.characters)
        }
        self.token_to_char = {
            index + 3: character for index, character in enumerate(self.characters)
        }
        self.allowed_characters = set(self.characters)

        self.model = build_model(
            config=config,
            vocab_size=len(self.characters) + 3,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

    def encode_texts(
        self,
        texts: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        encoded_sequences: list[list[int]] = []

        for text in texts:
            unknown_characters = sorted(
                {character for character in text if character not in self.char_to_token},
            )
            if unknown_characters:
                raise ValueError(f"Text contains unknown characters: {unknown_characters}")

            encoded = [self.sos_idx]
            encoded.extend(self.char_to_token[character] for character in text)
            encoded.append(self.eos_idx)
            encoded_sequences.append(encoded)

        max_length = max(len(sequence) for sequence in encoded_sequences)
        tokens = torch.full(
            size=(max_length, len(encoded_sequences)),
            fill_value=self.pad_idx,
            dtype=torch.long,
            device=device,
        )

        for batch_index, sequence in enumerate(encoded_sequences):
            tokens[: len(sequence), batch_index] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )

        return tokens

    def decode_tokens(self, token_ids: list[int]) -> str:
        characters: list[str] = []

        for token_id in token_ids:
            if token_id == self.eos_idx:
                break
            if token_id in {self.pad_idx, self.sos_idx}:
                continue

            character = self.token_to_char.get(token_id)
            if character is not None:
                characters.append(character)

        return "".join(characters)

    def forward(
        self,
        images: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(images=images, target_tokens=target_tokens)

    def training_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
    ) -> torch.Tensor:
        images = batch["images"]
        references = list(batch["texts"])

        target_tokens = self.encode_texts(references, device=images.device)
        decoder_input = target_tokens[:-1]
        expected_output = target_tokens[1:]

        logits = self(images=images, target_tokens=decoder_input)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            expected_output.reshape(-1),
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(references),
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_eval_step(batch=batch, prefix="val")

    def test_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_eval_step(batch=batch, prefix="test")

    def _shared_eval_step(
        self,
        batch: dict[str, object],
        prefix: str,
    ) -> torch.Tensor:
        images = batch["images"]
        references = list(batch["texts"])

        target_tokens = self.encode_texts(references, device=images.device)
        decoder_input = target_tokens[:-1]
        expected_output = target_tokens[1:]

        logits = self(images=images, target_tokens=decoder_input)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            expected_output.reshape(-1),
        )

        predicted_token_ids = self.model.predict(
            images,
            max_length=self.config.model.max_decoding_length,
        )
        predictions = [self.decode_tokens(sequence) for sequence in predicted_token_ids]

        self._log_text_metrics(
            prefix=prefix,
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )

        scheduler_config = self.config.model.get("scheduler", {})
        if not scheduler_config.get("enabled", False):
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_config.get("monitor", "val_loss"),
            },
        }
