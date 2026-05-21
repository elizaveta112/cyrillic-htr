import hydra
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.training.device import get_device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    device = get_device(config.infer.device)

    print("Running prediction")
    print(f"Model: {config.model.name}")
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(config.infer))


if __name__ == "__main__":
    main()
