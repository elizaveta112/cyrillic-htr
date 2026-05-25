from pathlib import Path

import torch
from PIL import Image

from cyrillic_htr.data.image_transforms import load_and_preprocess_image


def test_load_and_preprocess_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("L", (30, 10), color=255)
    image.save(image_path)

    image_tensor, image_width = load_and_preprocess_image(
        image_path=image_path,
        image_height=16,
        max_width=64,
        image_mean=0.5,
        image_std=0.5,
    )

    assert image_tensor.shape == (1, 16, 64)
    assert image_width == 48
    assert image_tensor.dtype == torch.float32
