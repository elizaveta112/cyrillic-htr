import torch

from cyrillic_htr.data.collate import htr_collate_fn


def test_htr_collate_fn() -> None:
    samples = [
        {
            "image": torch.zeros(1, 16, 64),
            "image_width": 48,
            "target": torch.tensor([1, 2], dtype=torch.long),
            "target_length": 2,
            "text": "аб",
            "image_path": "a.png",
        },
        {
            "image": torch.ones(1, 16, 64),
            "image_width": 32,
            "target": torch.tensor([2], dtype=torch.long),
            "target_length": 1,
            "text": "б",
            "image_path": "b.png",
        },
    ]

    batch = htr_collate_fn(samples)

    assert batch["images"].shape == (2, 1, 16, 64)
    assert batch["image_widths"].tolist() == [48, 32]
    assert batch["targets"].tolist() == [1, 2, 2]
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["texts"] == ["аб", "б"]
