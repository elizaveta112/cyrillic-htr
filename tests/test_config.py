from pathlib import Path

from cyrillic_htr.config import load_yaml


def test_load_data_config() -> None:
    config_path = Path("configs/data/default.yaml")
    config = load_yaml(config_path)

    assert config["random_seed"] == 42
    assert config["image_height"] == 64
