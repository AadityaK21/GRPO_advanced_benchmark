"""GSM8K answer parsing and the exact-match reward."""


def extract_answer(text: str) -> str | None:
    """Return the final numeric answer in `text` as a normalised string, or None if it has no number.

    GSM8K's reference solutions end with '#### <answer>'. Model completions may not.
    - If '####' appears, take the first number after it.
    - Otherwise take the last number in the text.

    Normalise the number: drop '$' and thousands commas, and drop trailing zeros after a decimal
    point. So '$1,200', '1200' and '1200.00' all become '1200', and '0.50' becomes '0.5'.
    A minus sign stays ('-3'). A full stop after a number ends it ('the answer is 7.' gives '7').
    """
    raise NotImplementedError


def exact_match_reward(completions: list[str], answers: list[str]) -> list[float]:
    """1.0 for each completion whose answer equals its reference answer's, else 0.0.

    completions[i] is scored against answers[i], a GSM8K reference solution.
    A completion with no number scores 0.0.
    """
    raise NotImplementedError
