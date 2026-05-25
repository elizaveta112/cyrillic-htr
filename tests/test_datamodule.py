import json
from pathlib import Path

from PIL import Image

from cyrillic_htr.data.datamodule import HTRDataModule


def create_tsv(path: Path, image_name: str, text: str) -> None:
    path.write_text(f"image\ttext\n{image_name}\t{text}\n", encoding="utf-8")


def test_htr_datamodule_train_dataloader(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    image_path = dataset_dir / "sample.png"
    image = Image.new("L", (30, 10), color=255)
    image.save(image_path)

    train_tsv = tmp_path / "train.tsv"
    val_tsv = tmp_path / "val.tsv"
    test_tsv = tmp_path / "test.tsv"

    create_tsv(train_tsv, "sample.png", "мама")
    create_tsv(val_tsv, "sample.png", "мама")
    create_tsv(test_tsv, "sample.png", "мама")

    vocab_path = tmp_path / "vocab.json"
    vocab = {"<blank>": 0, "а": 1, "м": 2}
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    datamodule = HTRDataModule(
        train_split_tsv=train_tsv,
        val_split_tsv=val_tsv,
        test_split_tsv=test_tsv,
        dataset_dir=dataset_dir,
        vocab_path=vocab_path,
        image_column="image",
        text_column="text",
        image_height=16,
        max_width=64,
        image_mean=0.5,
        image_std=0.5,
        batch_size=1,
        num_workers=0,
    )

    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    assert batch["images"].shape == (1, 1, 16, 64)
    assert batch["targets"].tolist() == [2, 1, 2, 1]
    assert batch["target_lengths"].tolist() == [4]
