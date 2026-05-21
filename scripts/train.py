import hydra
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.training.device import get_device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    device = get_device(config.train.device)

    print("Starting training")
    print(f"Project: {config.project.name}")
    print(f"Model: {config.model.name}")
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
