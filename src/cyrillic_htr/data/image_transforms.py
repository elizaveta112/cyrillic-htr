from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_and_preprocess_image(
    image_path: str | Path,
    image_height: int,
    max_width: int,
    image_mean: float,
    image_std: float,
) -> tuple[torch.Tensor, int]:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("L")
    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid image size for {image_path}: {image.size}")

    scaled_width = round(original_width * image_height / original_height)
    resized_width = max(1, min(scaled_width, max_width))

    image = image.resize((resized_width, image_height), Image.Resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    image_tensor = (image_tensor - image_mean) / image_std

    padded_image = torch.ones(
        size=(1, image_height, max_width),
        dtype=torch.float32,
    )
    padded_image[:, :, :resized_width] = image_tensor

    return padded_image, resized_width
