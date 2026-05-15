"""Tests for the Countdown reward function (eval/rl/reward.py)."""

from __future__ import annotations

import pytest

from saffron.eval.rl.reward import (
    _equation_reward,  # pyright: ignore[reportPrivateUsage]
    _format_reward,  # pyright: ignore[reportPrivateUsage]
    _safe_arith_eval,  # pyright: ignore[reportPrivateUsage]
    compute_reward,
)

# ---------------------------------------------------------------------------
# _format_reward
# ---------------------------------------------------------------------------


def test_format_reward_perfect() -> None:
    """Complete <think>...</think>\n<answer>...</answer> with allowed answer chars → 1.0"""
    completion = "reasoning here</think>\n<answer>(1 + 2) * 3</answer>"
    assert _format_reward(completion) == 1.0


def test_format_reward_answer_has_disallowed_chars() -> None:
    """Structure right, but answer contains letters → 0.5"""
    completion = "reasoning</think>\n<answer>one plus two</answer>"
    assert _format_reward(completion) == 0.5


def test_format_reward_missing_close_think() -> None:
    """No </think> tag → 0.0"""
    completion = "reasoning <answer>1+2</answer>"
    assert _format_reward(completion) == 0.0


def test_format_reward_missing_answer_block() -> None:
    """No <answer> block → 0.0"""
    completion = "reasoning</think>"
    assert _format_reward(completion) == 0.0


def test_format_reward_extra_content_after_answer() -> None:
    """Anything after </answer> breaks the strict full-string match → 0.0"""
    completion = "reasoning</think>\n<answer>1+2</answer> extra"
    assert _format_reward(completion) == 0.0


def test_format_reward_missing_newline_between_blocks() -> None:
    """Format requires \\n between </think> and <answer> → 0.0 without it"""
    completion = "reasoning</think><answer>1+2</answer>"
    assert _format_reward(completion) == 0.0


def test_format_reward_answer_contains_think_tags() -> None:
    """Answer block containing <think>...</think> is a hard format failure → 0.0.
    The regex restricts answer content to non-`<` chars."""
    completion = "reasoning</think>\n<answer><think>nested</think>1+2</answer>"
    assert _format_reward(completion) == 0.0


def test_format_reward_extra_think_in_completion() -> None:
    """The prep step already prefilled <think>; if the model emits another <think>
    in its completion, that's a malformed response → 0.0."""
    completion = "<think> repeated think </think>\n<answer>1+2</answer>"
    assert _format_reward(completion) == 0.0


# ---------------------------------------------------------------------------
# _equation_reward
# ---------------------------------------------------------------------------


def test_equation_reward_correct() -> None:
    """Uses exactly the given nums and equals target → 1.0"""
    completion = "<think>...</think>\n<answer>(44 - 19) * 35 / 5 + 23</answer>"
    assert _equation_reward(completion, nums=[44, 19, 35, 5, 23], target=198) == 1.0


def test_equation_reward_wrong_result() -> None:
    """Uses correct nums but math doesn't equal target → 0.0"""
    completion = "<think>...</think>\n<answer>44 + 19 + 35</answer>"
    assert _equation_reward(completion, nums=[44, 19, 35], target=999) == 0.0


def test_equation_reward_missing_nums() -> None:
    """Doesn't use exactly the given nums → 0.0"""
    completion = "<think>...</think>\n<answer>44 + 19</answer>"  # missing 35
    assert _equation_reward(completion, nums=[44, 19, 35], target=63) == 0.0


def test_equation_reward_extra_nums() -> None:
    """Uses extra numbers → 0.0"""
    completion = "<think>...</think>\n<answer>1 + 2 + 3</answer>"
    assert _equation_reward(completion, nums=[1, 2], target=3) == 0.0


def test_equation_reward_duplicate_nums_allowed_in_input() -> None:
    """If nums contains duplicates, each must be used (multiset match)."""
    completion = "<think>...</think>\n<answer>2 + 2 + 3</answer>"
    assert _equation_reward(completion, nums=[2, 2, 3], target=7) == 1.0
    # Same nums but equation uses only one 2 → multiset mismatch
    completion2 = "<think>...</think>\n<answer>2 + 3</answer>"
    assert _equation_reward(completion2, nums=[2, 2, 3], target=5) == 0.0


def test_equation_reward_division_by_zero() -> None:
    """1/0 in equation → 0.0 (gracefully handled)"""
    completion = "<think>...</think>\n<answer>5 / 0</answer>"
    assert _equation_reward(completion, nums=[5, 0], target=0) == 0.0


def test_equation_reward_no_answer_block() -> None:
    completion = "no answer here"
    assert _equation_reward(completion, nums=[1], target=1) == 0.0


def test_equation_reward_rejects_exponentiation() -> None:
    """Exponentiation (**) is rejected at AST parse → 0.0, even when the math would
    otherwise be 'correct'. 3**4 = 81, but we return 0 because we don't allow Pow."""
    completion = "<think>...</think>\n<answer>3 ** 4</answer>"
    assert _equation_reward(completion, nums=[3, 4], target=81) == 0.0


def test_equation_reward_float_target_tolerance() -> None:
    """Result within 1e-5 of target counts as correct."""
    completion = "<think>...</think>\n<answer>1 / 3</answer>"
    assert _equation_reward(completion, nums=[1, 3], target=0) == 0.0  # 0.333... vs 0


# ---------------------------------------------------------------------------
# compute_reward (integration)
# ---------------------------------------------------------------------------


def test_compute_reward_sums_components() -> None:
    """Total reward is the sum of format + equation."""
    completion = "reasoning</think>\n<answer>1 + 2</answer>"
    reward, metrics = compute_reward(completion, {"nums": [1, 2], "target": 3})
    assert reward == metrics["format_reward"] + metrics["equation_reward"]


def test_compute_reward_metrics_keys() -> None:
    """Metrics dict has the two expected component keys."""
    completion = "reasoning</think>\n<answer>1 + 2</answer>"
    _, metrics = compute_reward(completion, {"nums": [1, 2], "target": 3})
    assert set(metrics.keys()) == {"format_reward", "equation_reward"}


def test_compute_reward_perfect() -> None:
    """A perfect rollout scores 2.0 (1.0 format + 1.0 equation)."""
    completion = "let me think...</think>\n<answer>(44 - 19) * 35 / 5 + 23</answer>"
    reward, _ = compute_reward(completion, {"nums": [44, 19, 35, 5, 23], "target": 198})
    assert reward == pytest.approx(2.0)  # type: ignore[reportUnknownMemberType]


def test_compute_reward_format_only() -> None:
    """Format right, equation wrong → 1.0 (just format reward)."""
    completion = "thinking...</think>\n<answer>1 + 2</answer>"
    reward, metrics = compute_reward(completion, {"nums": [1, 2], "target": 999})
    assert reward == pytest.approx(1.0)  # type: ignore[reportUnknownMemberType]
    assert metrics["format_reward"] == 1.0
    assert metrics["equation_reward"] == 0.0


def test_compute_reward_garbage() -> None:
    """Totally malformed completion → 0.0"""
    reward, metrics = compute_reward("complete garbage with no tags", {"nums": [1, 2], "target": 3})
    assert reward == 0.0
    assert metrics["format_reward"] == 0.0
    assert metrics["equation_reward"] == 0.0


# ---------------------------------------------------------------------------
# _safe_arith_eval (defense-in-depth check, mostly covered above)
# ---------------------------------------------------------------------------


def test_safe_arith_eval_basic() -> None:
    assert _safe_arith_eval("5 + 3") == 8.0
    assert _safe_arith_eval("(44 - 19) * 35 / 5") == pytest.approx(175.0)  # type: ignore[reportUnknownMemberType]
    assert _safe_arith_eval("-5 + 10") == 5.0


@pytest.mark.parametrize(
    "expr",
    [
        "9 ** 9",  # exponentiation
        '__import__("os")',  # injection attempt
        "abs(-5)",  # function call
        "a + b",  # name lookup
    ],
)
def test_safe_arith_eval_rejects_unsafe(expr: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        _safe_arith_eval(expr)


def test_safe_arith_eval_division_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        _safe_arith_eval("1 / 0")
