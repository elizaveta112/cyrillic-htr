import torch
import torch.nn.functional as F
from torch import nn


class CRNNCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_channels: int = 1,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.classifier = nn.Linear(rnn_hidden_size * 2, vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(images)
        # [B, C, H', W']

        features = features.mean(dim=2)
        # [B, C, W']

        features = features.permute(2, 0, 1)
        # [T, B, C]

        rnn_output, _ = self.rnn(features)
        logits = self.classifier(rnn_output)
        log_probs = F.log_softmax(logits, dim=-1)

        time_steps = log_probs.size(0)

        if image_widths is None:
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=time_steps,
                dtype=torch.long,
                device=images.device,
            )
        else:
            input_lengths = torch.clamp(image_widths // 4, min=1, max=time_steps)
            input_lengths = input_lengths.to(device=images.device, dtype=torch.long)

        return log_probs, input_lengths
