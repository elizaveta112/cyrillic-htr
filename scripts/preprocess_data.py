import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    print("Running preprocessing")
    print(OmegaConf.to_yaml(config.data))


if __name__ == "__main__":
    main()
