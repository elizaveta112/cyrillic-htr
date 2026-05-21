import torch

from cyrillic_htr.models.crnn_ctc import CRNNCTC


def test_crnn_ctc_forward_shape() -> None:
    vocab_size = 40
    model = CRNNCTC(vocab_size=vocab_size)

    images = torch.randn(2, 1, 64, 256)
    image_widths = torch.tensor([256, 200])

    log_probs, input_lengths = model(images, image_widths)

    assert log_probs.ndim == 3
    assert log_probs.shape[1] == 2
    assert log_probs.shape[2] == vocab_size
    assert input_lengths.shape == (2,)
    assert torch.all(input_lengths > 0)
