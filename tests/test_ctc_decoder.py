import torch

from cyrillic_htr.inference.ctc_decoder import ctc_greedy_decode


def test_ctc_greedy_decode() -> None:
    idx_to_char = {
        0: "<blank>",
        1: "а",
        2: "б",
    }

    token_indices = torch.tensor(
        [
            [0],
            [1],
            [1],
            [0],
            [2],
            [2],
        ]
    )

    log_probs = torch.full((6, 1, 3), fill_value=-10.0)

    for timestep, token_index in enumerate(token_indices.squeeze(1)):
        log_probs[timestep, 0, token_index] = 0.0

    decoded = ctc_greedy_decode(
        log_probs=log_probs,
        idx_to_char=idx_to_char,
        blank_idx=0,
    )

    assert decoded == ["аб"]
