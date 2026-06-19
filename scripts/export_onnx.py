from pathlib import Path
from typing import Any

import hydra
import onnx
import torch
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.training.lightning_modules.transformer_htr_module import (
    TransformerHTRLightningModule,
)


class TransformerHTRONNXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            images=images,
            image_widths=image_widths,
            target_tokens=target_tokens,
        )


def prepare_export_config(config: DictConfig) -> DictConfig:
    prepared_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    prepared_config.data.vocab_path = prepared_config.export.vocab_path
    return prepared_config


def load_transformer_module(
    checkpoint_path: str | Path,
    config: DictConfig,
) -> TransformerHTRLightningModule:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    lightning_module = TransformerHTRLightningModule(config=config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    missing_keys, unexpected_keys = lightning_module.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys:
        print(f"Warning: missing checkpoint keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: unexpected checkpoint keys: {unexpected_keys[:10]}")

    lightning_module.eval()
    return lightning_module


def build_example_inputs(config: DictConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(config.export.batch_size)
    image_height = int(config.data.image_height)
    max_width = int(config.data.max_width)
    target_sequence_length = int(config.export.target_sequence_length)

    images = torch.randn(
        batch_size,
        int(config.model.image_channels),
        image_height,
        max_width,
        dtype=torch.float32,
    )
    image_widths = torch.full(
        size=(batch_size,),
        fill_value=max_width,
        dtype=torch.long,
    )
    target_tokens = torch.full(
        size=(target_sequence_length, batch_size),
        fill_value=int(config.model.sos_idx),
        dtype=torch.long,
    )

    return images, image_widths, target_tokens


def build_export_kwargs(config: DictConfig) -> dict[str, Any]:
    export_kwargs: dict[str, Any] = {
        "input_names": ["images", "image_widths", "target_tokens"],
        "output_names": ["logits"],
    }

    if bool(config.export.dynamic_axes):
        export_kwargs["dynamic_axes"] = {
            "images": {0: "batch", 3: "image_width"},
            "image_widths": {0: "batch"},
            "target_tokens": {0: "target_sequence_length", 1: "batch"},
            "logits": {0: "target_sequence_length", 1: "batch"},
        }

    return export_kwargs


def export_transformer_to_onnx(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_path: str | Path,
    config: DictConfig,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    torch.onnx.export(
        model,
        example_inputs,
        output_path,
        export_params=True,
        opset_version=int(config.export.opset_version),
        do_constant_folding=True,
        dynamo=False,
        **build_export_kwargs(config),
    )

    print(f"ONNX model exported to: {output_path}")


def validate_onnx_model(output_path: str | Path) -> None:
    output_path = Path(output_path)
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX validation passed: {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    config = prepare_export_config(config)

    lightning_module = load_transformer_module(
        checkpoint_path=config.export.checkpoint_path,
        config=config,
    )
    wrapper = TransformerHTRONNXWrapper(model=lightning_module.model)
    example_inputs = build_example_inputs(config)

    export_transformer_to_onnx(
        model=wrapper,
        example_inputs=example_inputs,
        output_path=config.export.output_path,
        config=config,
    )

    if bool(config.export.validate):
        validate_onnx_model(config.export.output_path)


if __name__ == "__main__":
    main()
