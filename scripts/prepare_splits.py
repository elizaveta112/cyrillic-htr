import hydra
from omegaconf import DictConfig

from cyrillic_htr.data.splits import make_train_val_test_splits


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    make_train_val_test_splits(
        train_tsv=config.data.train_tsv,
        test_tsv=config.data.test_tsv,
        train_split_tsv=config.data.train_split_tsv,
        val_split_tsv=config.data.val_split_tsv,
        test_split_tsv=config.data.test_split_tsv,
        image_column=config.data.image_column,
        text_column=config.data.text_column,
        tsv_has_header=config.data.tsv_has_header,
        val_size=config.data.val_size,
        random_seed=config.data.random_seed,
    )


if __name__ == "__main__":
    main()
