"""Basic request validation before retrieval and generation."""

MAX_QUESTION_LENGTH = 2000


def validate_question(question):
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question exceeds {MAX_QUESTION_LENGTH} characters."
        )

    return question
