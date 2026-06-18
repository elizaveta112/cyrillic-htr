from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


def _apply_train_augmentation(image: Image.Image) -> Image.Image:
    """Lightweight Kaggle-style augmentation for HTR line images."""
    image = image.convert("L")

    if random.random() < 0.70:
        angle = random.uniform(-7.0, 7.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)

    if random.random() < 0.55:
        shear = random.uniform(-0.14, 0.14)
        width, height = image.size
        x_shift = abs(shear) * height
        new_width = width + int(round(x_shift))
        x_offset = -x_shift if shear > 0 else 0
        image = image.transform(
            (new_width, height),
            Image.Transform.AFFINE,
            (1, shear, x_offset, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=255,
        )

    if random.random() < 0.55:
        factor = random.uniform(0.55, 1.35)
        image = ImageEnhance.Contrast(image).enhance(factor)

    if random.random() < 0.45:
        gamma = random.uniform(0.65, 1.45)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.clip(array**gamma, 0.0, 1.0)
        image = Image.fromarray((array * 255.0).astype(np.uint8), mode="L")

    if random.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))

    if random.random() < 0.35:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=random.uniform(2.0, 9.0), size=array.shape)
        array = np.clip(array + noise, 0.0, 255.0)
        image = Image.fromarray(array.astype(np.uint8), mode="L")

    return image


def load_and_preprocess_image(
    image_path: str | Path,
    image_height: int,
    max_width: int,
    image_mean: float,
    image_std: float,
    augment: bool = False,
) -> tuple[torch.Tensor, int]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("L")
    if augment:
        image = _apply_train_augmentation(image)

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
