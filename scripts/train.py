import hydra
from omegaconf import DictConfig, OmegaConf

from cyrillic_htr.data.factory import build_datamodule
from cyrillic_htr.training.device import get_device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    device = get_device(config.train.device)

    print("Starting training")
    print(f"Project: {config.project.name}")
    print(f"Model: {config.model.name}")
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(config))

    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    print("DataModule smoke test passed")
    print(f"Images shape: {batch['images'].shape}")
    print(f"Image widths shape: {batch['image_widths'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    print(f"Target lengths shape: {batch['target_lengths'].shape}")


if __name__ == "__main__":
    main()
