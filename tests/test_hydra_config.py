from hydra import compose, initialize


def test_hydra_default_config() -> None:
    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="config")

    assert config.project.name == "cyrillic-htr"
    assert config.model.name == "crnn_ctc"
    assert config.train.device == "cpu"


def test_hydra_transformer_colab_config() -> None:
    with initialize(version_base=None, config_path="../configs"):
        config = compose(
            config_name="config",
            overrides=["model=transformer_htr", "train=colab"],
        )

    assert config.model.name == "transformer_htr"
    assert config.train.device == "cuda"
