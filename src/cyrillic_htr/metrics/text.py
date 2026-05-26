import editdistance


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    total_distance = 0
    total_characters = 0

    for prediction, reference in zip(predictions, references, strict=True):
        total_distance += editdistance.eval(prediction, reference)
        total_characters += len(reference)

    if total_characters == 0:
        return 0.0

    return total_distance / total_characters


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    total_distance = 0
    total_words = 0

    for prediction, reference in zip(predictions, references, strict=True):
        prediction_words = prediction.split()
        reference_words = reference.split()

        total_distance += editdistance.eval(prediction_words, reference_words)
        total_words += len(reference_words)

    if total_words == 0:
        return 0.0

    return total_distance / total_words


def line_accuracy(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0

    correct_lines = sum(
        prediction == reference
        for prediction, reference in zip(predictions, references, strict=True)
    )

    return correct_lines / len(references)


def normalized_edit_similarity(predictions: list[str], references: list[str]) -> float:
    total_similarity = 0.0

    if not references:
        return 0.0

    for prediction, reference in zip(predictions, references, strict=True):
        max_length = max(len(prediction), len(reference))

        if max_length == 0:
            total_similarity += 1.0
            continue

        distance = editdistance.eval(prediction, reference)
        similarity = 1.0 - distance / max_length
        total_similarity += max(0.0, similarity)

    return total_similarity / len(references)


def valid_character_rate(predictions: list[str], allowed_characters: set[str]) -> float:
    total_characters = 0
    valid_characters = 0

    for prediction in predictions:
        for character in prediction:
            total_characters += 1
            valid_characters += int(character in allowed_characters)

    if total_characters == 0:
        return 0.0

    return valid_characters / total_characters
