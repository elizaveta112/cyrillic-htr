import torch


def htr_collate_fn(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    images = torch.stack([sample["image"] for sample in samples])
    image_widths = torch.tensor(
        [sample["image_width"] for sample in samples],
        dtype=torch.long,
    )

    targets = [sample["target"] for sample in samples]
    target_lengths = torch.tensor(
        [sample["target_length"] for sample in samples],
        dtype=torch.long,
    )

    concatenated_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)

    texts = [sample["text"] for sample in samples]
    image_paths = [sample["image_path"] for sample in samples]

    return {
        "images": images,
        "image_widths": image_widths,
        "targets": concatenated_targets,
        "target_lengths": target_lengths,
        "texts": texts,
        "image_paths": image_paths,
    }
