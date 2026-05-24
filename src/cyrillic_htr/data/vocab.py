import json
from pathlib import Path

import pandas as pd


def build_char_vocab_from_texts(
    texts: list[str],
    blank_token: str = "<blank>",
) -> dict[str, int]:
    clean_texts = [text for text in texts if isinstance(text, str) and text]

    if not clean_texts:
        raise ValueError("Cannot build vocabulary from empty text list.")

    characters = sorted(set("".join(clean_texts)))

    vocab = {blank_token: 0}
    for index, character in enumerate(characters, start=1):
        vocab[character] = index

    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)


def load_vocab(path: str | Path) -> dict[str, int]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        vocab: dict[str, int] = json.load(file)

    return vocab


def build_and_save_vocab_from_tsv(
    train_split_tsv: str | Path,
    text_column: str,
    vocab_path: str | Path,
    blank_token: str = "<blank>",
) -> dict[str, int]:
    train_split_tsv = Path(train_split_tsv)

    if not train_split_tsv.exists():
        raise FileNotFoundError(
            f"Train split file not found: {train_split_tsv}. Run scripts/prepare_splits.py first."
        )

    dataframe = pd.read_csv(train_split_tsv, sep="\t")

    if text_column not in dataframe.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in {train_split_tsv}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    before_drop = len(dataframe)
    dataframe = dataframe.dropna(subset=[text_column])
    dataframe[text_column] = dataframe[text_column].astype(str)
    dataframe = dataframe[dataframe[text_column].str.len() > 0]
    after_drop = len(dataframe)

    dropped_rows = before_drop - after_drop
    if dropped_rows > 0:
        print(f"Dropped rows with empty target text: {dropped_rows}")

    texts = dataframe[text_column].tolist()

    vocab = build_char_vocab_from_texts(texts=texts, blank_token=blank_token)
    save_vocab(vocab=vocab, path=vocab_path)

    return vocab
