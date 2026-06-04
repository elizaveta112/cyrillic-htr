import torch

from cyrillic_htr.inference.ctc_prediction import (
    character_error_rate,
    greedy_ctc_decode,
    word_error_rate,
)


def test_greedy_ctc_decode_collapses_repeats_and_removes_blank() -> None:
    index_to_token = {
        0: "<blank>",
        1: "а",
        2: "б",
    }

    # Shape: [time, batch, vocab]
    logits = torch.tensor(
        [
            [[0.0, 5.0, 0.0]],  # а
            [[0.0, 5.0, 0.0]],  # repeated а
            [[5.0, 0.0, 0.0]],  # blank
            [[0.0, 0.0, 5.0]],  # б
        ]
    )

    assert greedy_ctc_decode(
        logits=logits,
        index_to_token=index_to_token,
        blank_index=0,
        batch_first=False,
    ) == ["аб"]


def test_greedy_ctc_decode_supports_batch_first_logits() -> None:
    index_to_token = {
        0: "<blank>",
        1: "а",
        2: "б",
    }

    # Shape: [batch, time, vocab]
    logits = torch.tensor(
        [
            [
                [0.0, 5.0, 0.0],  # а
                [0.0, 5.0, 0.0],  # repeated а
                [5.0, 0.0, 0.0],  # blank
                [0.0, 0.0, 5.0],  # б
            ]
        ]
    )

    assert greedy_ctc_decode(
        logits=logits,
        index_to_token=index_to_token,
        blank_index=0,
        batch_first=True,
    ) == ["аб"]


def test_character_error_rate() -> None:
    assert character_error_rate("кот", "кот") == 0.0
    assert character_error_rate("кот", "кит") == 1 / 3


def test_word_error_rate() -> None:
    assert word_error_rate("мой кот", "мой кот") == 0.0
    assert word_error_rate("мой кот", "твой кот") == 1 / 2
