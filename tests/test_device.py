import torch

from cyrillic_htr.training.device import get_device


def test_get_cpu_device() -> None:
    device = get_device("cpu")

    assert device == torch.device("cpu")


def test_get_auto_device() -> None:
    device = get_device("auto")

    assert device.type in {"cpu", "cuda"}
