from pathlib import Path

from cyrillic_htr.data.vocab import (
    build_char_vocab_from_texts,
    load_vocab,
    save_vocab,
)


def test_build_char_vocab_from_texts() -> None:
    texts = ["мама", "папа"]
    vocab = build_char_vocab_from_texts(texts)

    assert vocab["<blank>"] == 0
    assert "м" in vocab
    assert "а" in vocab
    assert "п" in vocab


def test_save_and_load_vocab(tmp_path: Path) -> None:
    vocab = {"<blank>": 0, "а": 1, "б": 2}
    vocab_path = tmp_path / "vocab.json"

    save_vocab(vocab, vocab_path)
    loaded_vocab = load_vocab(vocab_path)

    assert loaded_vocab == vocab
