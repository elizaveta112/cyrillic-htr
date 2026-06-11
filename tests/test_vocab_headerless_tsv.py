import json

from cyrillic_htr.data.vocab import build_and_save_vocab_from_tsv


def test_build_vocab_from_headerless_kaggle_tsv(tmp_path) -> None:
    tsv_path = tmp_path / "train.tsv"
    vocab_path = tmp_path / "vocab.json"

    tsv_path.write_text(
        "aa1.png\tМолдова\naa2.png\txy\n",
        encoding="utf-8",
    )

    vocab = build_and_save_vocab_from_tsv(
        train_split_tsv=tsv_path,
        text_column="text",
        vocab_path=vocab_path,
    )

    assert vocab["<blank>"] == 0
    assert "М" in vocab
    assert "x" in vocab
    assert "y" in vocab

    with vocab_path.open(encoding="utf-8") as file:
        saved_vocab = json.load(file)

    assert saved_vocab == vocab
