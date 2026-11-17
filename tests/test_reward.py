import time

import pytest

from iterreason.reward import (
    answers_match,
    completion_to_text,
    conciseness_reward_fn,
    correctness_reward_fn,
    extract_answer,
    extract_boxed,
    format_reward_fn,
    normalise,
)


@pytest.mark.parametrize("text,expected", [
    (r"so \boxed{42}", "42"),
    (r"\boxed{1} then \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\sqrt{x^{2}}}", r"\sqrt{x^{2}}"),
    (r"\boxed {7}", "7"),
    (r"\boxed{12", None),          # truncated → no answer
    ("no box here", None),
])
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize("text,expected", [
    (r"... \boxed{1,234}", "1234"),
    ("work\n#### 72", "72"),
    ("The answer is 3.5.", "3.5"),
    ("The answer is 3.5 apples.", "3.5"),
    ("The final answer is: 12\nThanks!", "12"),
    ("the answer is 1. Then the answer is 2.", "2"),
    ("I computed 3 then 17", "17"),
    ("", None),
    (r"\boxed{}", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1,000", "1000"),
    ("$5$", "5"),
    (r"\$5", "5"),
    ("5.0", "5"),
    (r"\frac{1}{2}", "0.5"),
    (r"\dfrac{3}{4}", "0.75"),
    (r"\frac12", "0.5"),
    (r"-\frac{1}{4}", "-0.25"),
    ("3/4", "0.75"),
    (r"50\%", "50"),
    (r"90^\circ", "90"),
    (r"\text{Yes}", "yes"),
    ("x = 5", "5"),
    ("12.", "12"),
    (None, ""),
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize("raw", ["inf", "infinity", "-inf", "nan", "1e400"])
def test_normalise_non_finite_does_not_crash(raw):
    # float('infinity') used to raise OverflowError in int(v)
    assert isinstance(normalise(raw), str)
    assert isinstance(answers_match(raw, "5"), bool)


@pytest.mark.parametrize("pred,gt,ok", [
    ("72", "72", True),
    ("72.0", "72", True),
    ("0.5", r"\frac{1}{2}", True),
    ("1234567", "1234568", False),
    (r"2\sqrt{2}", r"\sqrt{8}", True),
    (r"\frac{\pi}{2}", r"\frac{1}{2}\pi", True),
    ("x+1", "1+x", True),
    ("3", "", False),
    ("", "", False),
])
def test_answers_match(pred, gt, ok):
    assert answers_match(pred, gt) is ok


def test_sympy_fallback_rejects_dangerous_input_quickly():
    t = time.time()
    assert answers_match("9^9^9^9", "1") is False
    assert answers_match("__import__('os')", "1") is False
    assert answers_match("2^99999", "1") is False
    assert time.time() - t < 2


def test_completion_to_text():
    assert completion_to_text("abc") == "abc"
    assert completion_to_text([{"role": "assistant", "content": "abc"}]) == "abc"


@pytest.mark.parametrize("wrap", [
    lambda s: s,
    lambda s: [{"role": "assistant", "content": s}],   # conversational GRPO
])
def test_reward_fns_accept_both_completion_formats(wrap):
    comps = [wrap(r"so \boxed{4}"), wrap("so 5"), wrap("word " * 1500)]
    assert correctness_reward_fn(comps, answer=["4", "4", "4"]) == [1.0, 0.0, 0.0]
    assert format_reward_fn(comps) == [0.1, 0.0, 0.0]
    conc = conciseness_reward_fn(comps)
    assert conc[:2] == [0.0, 0.0] and conc[2] == pytest.approx(-0.1)
