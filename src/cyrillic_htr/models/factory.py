from omegaconf import DictConfig

from cyrillic_htr.models.crnn_ctc import CRNNCTC
from cyrillic_htr.models.transformer_htr import TransformerHTR


def build_model(config: DictConfig, vocab_size: int):
    if config.model.name == "crnn_ctc":
        return CRNNCTC(
            vocab_size=vocab_size,
            image_channels=config.model.image_channels,
            rnn_hidden_size=config.model.rnn_hidden_size,
            rnn_num_layers=config.model.rnn_num_layers,
            dropout=config.model.dropout,
        )

    if config.model.name == "transformer_htr":
        return TransformerHTR(
            vocab_size=vocab_size,
            image_channels=config.model.image_channels,
            hidden_size=config.model.hidden_size,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            pad_idx=config.model.pad_idx,
            sos_idx=config.model.sos_idx,
            eos_idx=config.model.eos_idx,
            max_decoding_length=config.model.max_decoding_length,
        )

    raise ValueError(f"Unknown model name: {config.model.name}")
