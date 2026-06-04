import hydra
from omegaconf import DictConfig

from cyrillic_htr.data.vocab import build_and_save_vocab_from_tsv


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    vocab = build_and_save_vocab_from_tsv(
        tsv_path=config.data.train_tsv,
        text_column=config.data.text_column,
        vocab_path=config.data.vocab_path,
        blank_token="<blank>",
    )

    print(f"Vocabulary saved to: {config.data.vocab_path}")
    print(f"Vocabulary size: {len(vocab)}")


if __name__ == "__main__":
    main()
