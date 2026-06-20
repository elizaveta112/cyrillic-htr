import base64
import io
import math
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from cyrillic_htr.training.lightning_modules.transformer_htr_module import (
    TransformerHTRLightningModule,
)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def strip_base64_prefix(payload: str) -> str:
    if "," in payload and payload.strip().startswith("data:"):
        return payload.split(",", maxsplit=1)[1]
    return payload


class TransformerHTRPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for Transformer HTR inference.

    The model accepts a pandas DataFrame with one of the following fields:
    - image_base64: base64-encoded image content;
    - image_path: local path to an image file available on the serving machine.

    The model returns recognized Cyrillic text.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.config_dict = config
        self.device_name = device
        self.config: DictConfig | None = None
        self.device: torch.device | None = None
        self.lightning_module: TransformerHTRLightningModule | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        checkpoint_path = Path(context.artifacts["checkpoint"])
        vocab_path = Path(context.artifacts["vocab"])

        self.config = OmegaConf.create(self.config_dict)
        OmegaConf.update(self.config, "model.name", "transformer_htr", force_add=True)
        OmegaConf.update(self.config, "data.vocab_path", str(vocab_path), force_add=True)
        OmegaConf.update(self.config, "infer.vocab_path", str(vocab_path), force_add=True)
        OmegaConf.update(
            self.config,
            "infer.checkpoint_path",
            str(checkpoint_path),
            force_add=True,
        )

        self.device = torch.device(self.resolve_device())
        self.lightning_module = self.load_lightning_module(checkpoint_path)
        self.lightning_module.to(self.device)
        self.lightning_module.eval()

    def resolve_device(self) -> str:
        if self.device_name == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_name

    def load_lightning_module(
        self,
        checkpoint_path: Path,
    ) -> TransformerHTRLightningModule:
        if self.config is None:
            raise RuntimeError("Config is not initialized.")

        if str(self.config.model.name) != "transformer_htr":
            raise ValueError(
                "MLflow serving is implemented only for Transformer HTR. "
                "Use: model=transformer_htr +serving=mlflow"
            )

        lightning_module = TransformerHTRLightningModule(config=self.config)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        lightning_module.load_state_dict(state_dict, strict=True)

        return lightning_module

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params

        rows = self.normalize_model_input(model_input)
        predictions = []

        for row in rows:
            image = self.load_image_from_row(row)
            prediction = self.predict_image(image)

            predictions.append(
                {
                    "image_path": str(row.get("image_path", "")),
                    "prediction": prediction,
                }
            )

        return pd.DataFrame(predictions)

    @staticmethod
    def normalize_model_input(
        model_input: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    ) -> list[dict[str, Any]]:
        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        if isinstance(model_input, dict):
            return [model_input]
        return list(model_input)

    @staticmethod
    def load_image_from_row(row: dict[str, Any]) -> Image.Image:
        image_base64 = row.get("image_base64")
        image_path = row.get("image_path")

        if has_value(image_base64):
            payload = strip_base64_prefix(str(image_base64))
            image_bytes = base64.b64decode(payload)
            return Image.open(io.BytesIO(image_bytes)).convert("L")

        if has_value(image_path):
            path = Path(str(image_path))
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist on server: {path}")
            return Image.open(path).convert("L")

        raise ValueError("Request must contain either 'image_base64' or 'image_path'.")

    def preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config is None:
            raise RuntimeError("Config is not initialized.")

        image_height = int(self.config.data.image_height)
        max_width = int(self.config.data.max_width)
        image_mean = float(self.config.data.image_mean)
        image_std = float(self.config.data.image_std)

        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError(f"Invalid image size: {image.size}")

        resized_width = int(round(original_width * image_height / original_height))
        resized_width = max(1, min(resized_width, max_width))

        resized_image = image.resize(
            (resized_width, image_height),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("L", (max_width, image_height), color=255)
        canvas.paste(resized_image, (0, 0))

        image_array = np.asarray(canvas, dtype=np.float32) / 255.0
        image_array = (image_array - image_mean) / image_std

        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        image_widths = torch.tensor([resized_width], dtype=torch.long)

        return image_tensor, image_widths

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> str:
        if self.config is None:
            raise RuntimeError("Config is not initialized.")
        if self.device is None:
            raise RuntimeError("Device is not initialized.")
        if self.lightning_module is None:
            raise RuntimeError("Model is not initialized.")

        images, image_widths = self.preprocess_image(image)
        images = images.to(self.device)
        image_widths = image_widths.to(self.device)

        token_sequences = self.lightning_module.model.predict(
            images=images,
            image_widths=image_widths,
            max_length=int(self.config.model.max_decoding_length),
        )

        return self.lightning_module.decode_tokens(token_sequences[0])
