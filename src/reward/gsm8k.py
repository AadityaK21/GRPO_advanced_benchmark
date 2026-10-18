"""GSM8K answer parsing and the exact-match reward."""
import re

# A number: optional minus, optional $, digits with thousands commas, and a decimal part
# only when a digit follows the dot. So "the answer is 7." gives "7", not "7.".
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _normalise(s: str) -> str:
    """'$1,200' -> '1200', '18.00' -> '18', '0.50' -> '0.5', '-3' -> '-3'."""
    s = s.replace("$", "").replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_answer(text: str) -> str | None:
    """Return the final numeric answer in `text` as a normalised string, or None if it has no number.

    GSM8K's reference solutions end with '#### <answer>'. Model completions may not.
    - If '####' appears, take the first number after it.
    - Otherwise take the last number in the text.

    Normalise the number: drop '$' and thousands commas, and drop trailing zeros after a decimal
    point. So '$1,200', '1200' and '1200.00' all become '1200', and '0.50' becomes '0.5'.
    A minus sign stays ('-3'). A full stop after a number ends it ('the answer is 7.' gives '7').
    """
    if "####" in text:
        # Reference solutions: the answer is the first number after the four hashes.
        numbers = _NUMBER.findall(text.split("####", 1)[1])
        pick = 0
    else:
        # Model completions: the answer is usually the last number written.
        numbers = _NUMBER.findall(text)
        pick = -1
    if not numbers:
        return None
    return _normalise(numbers[pick])


def exact_match_reward(completions: list[str], answers: list[str]) -> list[float]:
    """1.0 for each completion whose answer equals its reference answer's, else 0.0.

    completions[i] is scored against answers[i], a GSM8K reference solution.
    A completion with no number scores 0.0.
    """
    rewards = []
    for completion, answer in zip(completions, answers, strict=True):
        got = extract_answer(completion)
        # No number in the completion is a wrong answer, never a match.
        rewards.append(1.0 if got is not None and got == extract_answer(answer) else 0.0)
    return rewards
