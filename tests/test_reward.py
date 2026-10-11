"""Checks for src/reward/gsm8k.py."""
import pytest

from src.reward.gsm8k import exact_match_reward, extract_answer

GSM8K_SOLUTION = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def test_reads_the_answer_after_the_four_hashes():
    assert extract_answer(GSM8K_SOLUTION) == "72"


def test_text_after_the_hashes_does_not_change_the_answer():
    assert extract_answer("#### 18\nIt took 3 steps.") == "18", "use the number right after ####"


@pytest.mark.parametrize(
    "text, answer",
    [
        ("So she has 5 apples, and the answer is 7.", "7"),
        ("The total is $1,200.", "1200"),
        ("It costs 18.00 dollars", "18"),
        ("That leaves 0.50 of the cake", "0.5"),
        ("The change is -3", "-3"),
        ("\\boxed{42}", "42"),
    ],
)
def test_without_hashes_takes_the_last_number_and_normalises_it(text, answer):
    assert extract_answer(text) == answer


def test_no_number_gives_none():
    assert extract_answer("I don't know.") is None


def test_reward_is_one_only_for_a_matching_answer():
    completions = ["... so the answer is 72.", "The answer is 70.", "No idea.", "#### 72.00"]
    answers = [GSM8K_SOLUTION] * 4
    assert exact_match_reward(completions, answers) == [1.0, 0.0, 0.0, 1.0]


def test_reward_is_a_float_per_completion():
    rewards = exact_match_reward(["1", "2"], ["#### 1", "#### 3"])
    assert rewards == [1.0, 0.0] and all(isinstance(r, float) for r in rewards)
