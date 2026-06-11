import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def build_char_vocab_from_texts(
    texts: Iterable[str],
    blank_token: str = "<blank>",
) -> dict[str, int]:
    characters = sorted(set("".join(str(text) for text in texts)))

    vocab = {blank_token: 0}

    for index, character in enumerate(characters, start=1):
        if character != blank_token:
            vocab[character] = index

    return vocab


def build_vocab_from_texts(
    texts: Iterable[str],
    blank_token: str = "<blank>",
) -> dict[str, int]:
    return build_char_vocab_from_texts(
        texts=texts,
        blank_token=blank_token,
    )


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)


def load_vocab(path: str | Path) -> dict[str, int]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_texts_from_tsv(
    tsv_path: str | Path,
    text_column: str,
) -> list[str]:
    tsv_path = Path(tsv_path)

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    dataframe = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if text_column in dataframe.columns:
        return dataframe[text_column].astype(str).tolist()

    headerless_dataframe = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
    )

    if headerless_dataframe.shape[1] < 2:
        raise ValueError(
            f"Text column '{text_column}' not found in {tsv_path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return headerless_dataframe.iloc[:, 1].astype(str).tolist()


def build_and_save_vocab_from_tsv(
    train_split_tsv: str | Path,
    text_column: str,
    vocab_path: str | Path,
    blank_token: str = "<blank>",
) -> dict[str, int]:
    texts = read_texts_from_tsv(
        tsv_path=train_split_tsv,
        text_column=text_column,
    )
    vocab = build_char_vocab_from_texts(
        texts=texts,
        blank_token=blank_token,
    )
    save_vocab(vocab=vocab, path=vocab_path)

    return vocab
