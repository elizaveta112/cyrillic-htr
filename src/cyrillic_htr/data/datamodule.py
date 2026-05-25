from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from cyrillic_htr.data.collate import htr_collate_fn
from cyrillic_htr.data.dataset import HTRDataset


class HTRDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_split_tsv: str | Path,
        val_split_tsv: str | Path,
        test_split_tsv: str | Path,
        dataset_dir: str | Path,
        vocab_path: str | Path,
        image_column: str,
        text_column: str,
        image_height: int,
        max_width: int,
        image_mean: float,
        image_std: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.train_split_tsv = train_split_tsv
        self.val_split_tsv = val_split_tsv
        self.test_split_tsv = test_split_tsv
        self.dataset_dir = dataset_dir
        self.vocab_path = vocab_path
        self.image_column = image_column
        self.text_column = text_column
        self.image_height = image_height
        self.max_width = max_width
        self.image_mean = image_mean
        self.image_std = image_std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: HTRDataset | None = None
        self.val_dataset: HTRDataset | None = None
        self.test_dataset: HTRDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {"fit", None}:
            self.train_dataset = self._make_dataset(self.train_split_tsv)
            self.val_dataset = self._make_dataset(self.val_split_tsv)

        if stage in {"test", None}:
            self.test_dataset = self._make_dataset(self.test_split_tsv)

        if stage == "validate":
            self.val_dataset = self._make_dataset(self.val_split_tsv)

        if stage == "predict":
            self.test_dataset = self._make_dataset(self.test_split_tsv)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized. Call setup('fit') first.")

        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup('fit') first.")

        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test') first.")

        return self._make_dataloader(self.test_dataset, shuffle=False)

    def _make_dataset(self, split_tsv: str | Path) -> HTRDataset:
        return HTRDataset(
            split_tsv=split_tsv,
            dataset_dir=self.dataset_dir,
            vocab_path=self.vocab_path,
            image_column=self.image_column,
            text_column=self.text_column,
            image_height=self.image_height,
            max_width=self.max_width,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

    def _make_dataloader(self, dataset: HTRDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=htr_collate_fn,
            persistent_workers=self.num_workers > 0,
        )
