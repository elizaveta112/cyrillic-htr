from pathlib import Path

import torch

from cyrillic_htr.data.vocab import load_vocab


def build_index_to_token(vocab_path: str | Path) -> dict[int, str]:
    vocab = load_vocab(vocab_path)
    return {index: token for token, index in vocab.items()}


def greedy_ctc_decode(
    logits: torch.Tensor,
    index_to_token: dict[int, str],
    blank_index: int = 0,
    batch_first: bool = False,
) -> list[str]:
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits tensor, got shape: {tuple(logits.shape)}")

    predicted_indices = logits.argmax(dim=-1)

    if not batch_first:
        predicted_indices = predicted_indices.transpose(0, 1)

    predictions = []

    for sequence in predicted_indices.cpu().tolist():
        tokens = []
        previous_index = None

        for index in sequence:
            if index != blank_index and index != previous_index:
                tokens.append(index_to_token.get(index, ""))

            previous_index = index

        predictions.append("".join(tokens))

    return predictions


def levenshtein_distance(first: list[str] | str, second: list[str] | str) -> int:
    first_length = len(first)
    second_length = len(second)

    previous_row = list(range(second_length + 1))

    for first_index in range(1, first_length + 1):
        current_row = [first_index]

        for second_index in range(1, second_length + 1):
            deletion = previous_row[second_index] + 1
            insertion = current_row[second_index - 1] + 1
            substitution = previous_row[second_index - 1] + (
                first[first_index - 1] != second[second_index - 1]
            )

            current_row.append(min(deletion, insertion, substitution))

        previous_row = current_row

    return previous_row[-1]


def character_error_rate(target_text: str, predicted_text: str) -> float:
    if not target_text:
        return 0.0 if not predicted_text else 1.0

    return levenshtein_distance(target_text, predicted_text) / len(target_text)


def word_error_rate(target_text: str, predicted_text: str) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if not target_words:
        return 0.0 if not predicted_words else 1.0

    return levenshtein_distance(target_words, predicted_words) / len(target_words)
