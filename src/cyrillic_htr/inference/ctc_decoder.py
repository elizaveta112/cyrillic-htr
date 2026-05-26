import torch


def build_idx_to_char(vocab: dict[str, int]) -> dict[int, str]:
    return {index: character for character, index in vocab.items()}


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    idx_to_char: dict[int, str],
    blank_idx: int = 0,
) -> list[str]:
    predicted_indices = log_probs.argmax(dim=-1)
    predicted_indices = predicted_indices.detach().cpu().transpose(0, 1)

    decoded_texts = []

    for sequence in predicted_indices:
        previous_index: int | None = None
        characters = []

        for token_index_tensor in sequence:
            token_index = int(token_index_tensor.item())

            if token_index != blank_idx and token_index != previous_index:
                characters.append(idx_to_char[token_index])

            previous_index = token_index

        decoded_texts.append("".join(characters))

    return decoded_texts
