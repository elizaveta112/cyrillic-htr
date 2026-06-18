from omegaconf import DictConfig

from cyrillic_htr.data.datamodule import HTRDataModule


def build_datamodule(config: DictConfig) -> HTRDataModule:
    augmentation_config = config.data.get("augmentation", {})
    augment_train = bool(augmentation_config.get("enabled", False))

    return HTRDataModule(
        train_split_tsv=config.data.train_split_tsv,
        val_split_tsv=config.data.val_split_tsv,
        test_split_tsv=config.data.test_split_tsv,
        dataset_dir=config.data.dataset_dir,
        vocab_path=config.data.vocab_path,
        image_column=config.data.image_column,
        text_column=config.data.text_column,
        image_height=config.data.image_height,
        max_width=config.data.max_width,
        image_mean=config.data.image_mean,
        image_std=config.data.image_std,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.device == "cuda",
        augment_train=augment_train,
    )
