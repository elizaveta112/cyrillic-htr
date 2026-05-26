from cyrillic_htr.metrics.text import (
    character_error_rate,
    line_accuracy,
    normalized_edit_similarity,
    valid_character_rate,
    word_error_rate,
)


def test_character_error_rate() -> None:
    predictions = ["мама"]
    references = ["мама"]

    assert character_error_rate(predictions, references) == 0.0


def test_word_error_rate() -> None:
    predictions = ["мама мыла"]
    references = ["мама мыла"]

    assert word_error_rate(predictions, references) == 0.0


def test_line_accuracy() -> None:
    predictions = ["мама", "папа"]
    references = ["мама", "мама"]

    assert line_accuracy(predictions, references) == 0.5


def test_normalized_edit_similarity() -> None:
    predictions = ["мама"]
    references = ["мама"]

    assert normalized_edit_similarity(predictions, references) == 1.0


def test_valid_character_rate() -> None:
    predictions = ["абвx"]
    allowed_characters = {"а", "б", "в"}

    assert valid_character_rate(predictions, allowed_characters) == 0.75
