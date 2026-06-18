import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model),
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(
            position * div_term[: positional_encoding[:, 1::2].shape[1]],
        )
        positional_encoding = positional_encoding.unsqueeze(1)
        self.register_buffer('pe', positional_encoding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs + self.scale * self.pe[: inputs.size(0)]
        return self.dropout(outputs)


class TransformerHTR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_channels: int = 1,
        hidden_size: int = 512,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_decoding_length: int = 100,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_decoding_length = max_decoding_length

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )
        self.feature_projection = nn.Linear(512, hidden_size)
        self.source_positional_encoding = PositionalEncoding(d_model=hidden_size, dropout=dropout)
        self.target_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx,
        )
        self.target_positional_encoding = PositionalEncoding(d_model=hidden_size, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = features.permute(0, 3, 1, 2).flatten(2)
        features = features.permute(1, 0, 2)
        return self.feature_projection(features)

    @staticmethod
    def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)

    @staticmethod
    def make_source_padding_mask(
        image_widths: torch.Tensor | None,
        memory_length: int,
    ) -> torch.Tensor | None:
        if image_widths is None:
            return None
        feature_lengths = torch.div(image_widths, 4, rounding_mode='floor') + 1
        feature_lengths = feature_lengths.clamp(min=1, max=memory_length)
        positions = torch.arange(memory_length, device=image_widths.device).unsqueeze(0)
        return positions >= feature_lengths.unsqueeze(1)

    def encode(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.extract_features(images)
        features = self.source_positional_encoding(features)
        source_padding_mask = self.make_source_padding_mask(
            image_widths=image_widths,
            memory_length=features.size(0),
        )
        memory = self.transformer.encoder(features, src_key_padding_mask=source_padding_mask)
        return memory, source_padding_mask

    def decode(
        self,
        memory: torch.Tensor,
        target_tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_mask = self.generate_square_subsequent_mask(
            size=target_tokens.size(0),
            device=target_tokens.device,
        )
        target_padding_mask = target_tokens.transpose(0, 1).eq(self.pad_idx)
        target_embeddings = self.target_embedding(target_tokens)
        target_embeddings = target_embeddings * math.sqrt(self.hidden_size)
        target_embeddings = self.target_positional_encoding(target_embeddings)
        decoder_output = self.transformer.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_layer(decoder_output)

    def forward(
        self,
        images: torch.Tensor,
        target_tokens: torch.Tensor,
        image_widths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_key_padding_mask = self.encode(images=images, image_widths=image_widths)
        return self.decode(
            memory=memory,
            target_tokens=target_tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
        max_length: int | None = None,
    ) -> list[list[int]]:
        max_length = max_length or self.max_decoding_length
        memory, memory_key_padding_mask = self.encode(images=images, image_widths=image_widths)
        batch_size = images.size(0)
        generated = torch.full(
            size=(1, batch_size),
            fill_value=self.sos_idx,
            dtype=torch.long,
            device=images.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

        for _ in range(max_length):
            logits = self.decode(
                memory=memory,
                target_tokens=generated,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            next_tokens = logits[-1].argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.pad_idx),
                next_tokens,
            )
            generated = torch.cat([generated, next_tokens.unsqueeze(0)], dim=0)
            finished |= next_tokens.eq(self.eos_idx)
            if finished.all():
                break

        return generated.transpose(0, 1).detach().cpu().tolist()
