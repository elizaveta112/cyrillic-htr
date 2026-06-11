import hydra
from omegaconf import DictConfig

from cyrillic_htr.data.vocab import build_and_save_vocab_from_tsv


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    text_column = config.data.get("text_column", "text")

    vocab = build_and_save_vocab_from_tsv(
        train_split_tsv=config.data.train_tsv,
        text_column=text_column,
        vocab_path=config.data.vocab_path,
    )

    print(f"Vocabulary saved to: {config.data.vocab_path}")
    print(f"Vocabulary size: {len(vocab)}")


if __name__ == "__main__":
    main()
