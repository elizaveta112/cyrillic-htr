import hydra
from omegaconf import DictConfig

from cyrillic_htr.data.download import download_data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    download_data(
        dataset_name=config.data.kaggle_dataset,
        output_dir=config.data.dataset_dir,
        force_download=config.data.force_download,
    )


if __name__ == "__main__":
    main()
