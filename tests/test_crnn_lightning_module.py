import json
from pathlib import Path

from hydra import compose, initialize

from cyrillic_htr.training.lightning_modules.crnn_ctc_module import CRNNCTCLightningModule


def test_crnn_ctc_lightning_module_initialization(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab = {"<blank>": 0, "а": 1, "б": 2}
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    with initialize(version_base=None, config_path="../configs"):
        config = compose(
            config_name="config",
            overrides=[
                "model=crnn_ctc",
                f"data.vocab_path={vocab_path}",
            ],
        )

    module = CRNNCTCLightningModule(config)

    assert module.blank_idx == 0
    assert len(module.vocab) == 3
