import json
from pathlib import Path

from PIL import Image

from cyrillic_htr.data.dataset import HTRDataset


def test_htr_dataset_get_item(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    image_path = dataset_dir / "sample.png"
    image = Image.new("L", (30, 10), color=255)
    image.save(image_path)

    split_tsv = tmp_path / "split.tsv"
    split_tsv.write_text("image\ttext\nsample.png\tмама\n", encoding="utf-8")

    vocab_path = tmp_path / "vocab.json"
    vocab = {"<blank>": 0, "а": 1, "м": 2}
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    dataset = HTRDataset(
        split_tsv=split_tsv,
        dataset_dir=dataset_dir,
        vocab_path=vocab_path,
        image_column="image",
        text_column="text",
        image_height=16,
        max_width=64,
        image_mean=0.5,
        image_std=0.5,
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (1, 16, 64)
    assert sample["image_width"] == 48
    assert sample["text"] == "мама"
    assert sample["target"].tolist() == [2, 1, 2, 1]
    assert sample["target_length"] == 4
