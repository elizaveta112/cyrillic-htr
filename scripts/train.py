import hydra
from omegaconf import DictConfig

from cyrillic_htr.training.runner import train_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    train_model(config)


if __name__ == "__main__":
    main()
