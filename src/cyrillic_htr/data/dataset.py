from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from cyrillic_htr.data.image_transforms import load_and_preprocess_image
from cyrillic_htr.data.vocab import load_vocab

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def build_image_index(dataset_dir: str | Path) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths = [
        image_path
        for image_path in dataset_dir.rglob("*")
        if image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]

    return {image_path.name: image_path for image_path in image_paths}


def encode_text(text: str, vocab: dict[str, int]) -> torch.Tensor:
    unknown_characters = sorted({character for character in text if character not in vocab})

    if unknown_characters:
        raise ValueError(f"Text contains unknown characters: {unknown_characters}")

    encoded_text = [vocab[character] for character in text]
    return torch.tensor(encoded_text, dtype=torch.long)


class HTRDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        split_tsv: str | Path,
        dataset_dir: str | Path,
        vocab_path: str | Path,
        image_column: str,
        text_column: str,
        image_height: int,
        max_width: int,
        image_mean: float,
        image_std: float,
    ) -> None:
        self.split_tsv = Path(split_tsv)
        self.dataset_dir = Path(dataset_dir)
        self.vocab = load_vocab(vocab_path)
        self.image_column = image_column
        self.text_column = text_column
        self.image_height = image_height
        self.max_width = max_width
        self.image_mean = image_mean
        self.image_std = image_std

        if not self.split_tsv.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_tsv}")

        dataframe = pd.read_csv(self.split_tsv, sep="\t")

        required_columns = {self.image_column, self.text_column}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {self.split_tsv}: {missing_columns}. "
                f"Available columns: {list(dataframe.columns)}"
            )

        dataframe = dataframe.dropna(subset=[self.image_column, self.text_column])
        dataframe[self.image_column] = dataframe[self.image_column].astype(str)
        dataframe[self.text_column] = dataframe[self.text_column].astype(str)
        dataframe = dataframe[dataframe[self.text_column].str.len() > 0]

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_index = build_image_index(self.dataset_dir)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, image_name: str) -> Path:
        direct_path = self.dataset_dir / image_name

        if direct_path.exists():
            return direct_path

        filename = Path(image_name).name
        indexed_path = self.image_index.get(filename)

        if indexed_path is None:
            raise FileNotFoundError(
                f"Image '{image_name}' was not found inside dataset directory: {self.dataset_dir}"
            )

        return indexed_path

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.dataframe.iloc[index]

        image_name = str(row[self.image_column])
        text = str(row[self.text_column])
        image_path = self._resolve_image_path(image_name)

        image_tensor, image_width = load_and_preprocess_image(
            image_path=image_path,
            image_height=self.image_height,
            max_width=self.max_width,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        target = encode_text(text=text, vocab=self.vocab)

        return {
            "image": image_tensor,
            "image_width": image_width,
            "target": target,
            "target_length": len(target),
            "text": text,
            "image_path": str(image_path),
        }
