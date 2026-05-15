import ast
import re
from typing import Any

# The prep step prefilled the assistant turn with "...\n<think>", so model
# completions start *after* that opening tag. We prepend "<think>" before
# checking format so the regex sees a complete <think>...</think> block.
# Answer block is restricted to non-`<` chars so nested tags fail the format outright
# (otherwise we'd misclassify <think> inside <answer> as a soft "charset wrong" issue).
_FORMAT_RE = re.compile(
    r"^<think>.*?</think>\n<answer>(?P<answer>[^<]*)</answer>$",
    re.DOTALL,
)

# For _format_reward's partial-credit step: the answer block should only contain
# digits, the four arithmetic operators, parens, and whitespace.
_ANSWER_CHARS_RE = re.compile(r"^[\d+\-*/().\s]+$")

# Used by _equation_reward to find the answer block independently of full format.
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Extracts integer literals from an equation (for multiset comparison against nums).
_INT_RE = re.compile(r"\d+")


def compute_reward(
    completion: str,
    sample: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Combined reward (format + equation) for a Countdown rollout."""
    format_reward = _format_reward(completion)
    equation_reward = _equation_reward(
        completion=completion,
        nums=sample["nums"],
        target=sample["target"],
    )

    reward = format_reward + equation_reward
    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }
    return reward, metrics


def _format_reward(completion: str) -> float:
    """Score the structural format of the completion.

    Returns:
        0.0 — missing or malformed <think>...</think>\\n<answer>...</answer> structure
        0.5 — structure present but answer contains disallowed characters
        1.0 — structure and answer charset both correct
    """
    full = "<think>" + completion
    # Each tag must appear exactly once. Extra <think> or <answer> tags (even inside
    # think content, which the non-greedy regex would otherwise tolerate) → reject.
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if full.count(tag) != 1:
            return 0.0
    match = _FORMAT_RE.fullmatch(full)
    if match is None:
        return 0.0
    answer = match.group("answer").strip()
    if not _ANSWER_CHARS_RE.fullmatch(answer):
        return 0.5
    return 1.0


def _equation_reward(completion: str, nums: list[int], target: int) -> float:
    """Score whether the equation in <answer> uses the given nums and evaluates to target.

    Returns:
        1.0 — equation uses exactly the given nums (multiset match) and equals target (±1e-5)
        0.0 — otherwise
    """
    match = _ANSWER_BLOCK_RE.search(completion)
    if match is None:
        return 0.0
    equation = match.group(1).strip()

    # Each number from `nums` must be used exactly once.
    used = [int(n) for n in _INT_RE.findall(equation)]
    if sorted(used) != sorted(nums):
        return 0.0

    try:
        result = _safe_arith_eval(equation)
    except (ValueError, ZeroDivisionError, SyntaxError):
        return 0.0

    return 1.0 if abs(result - target) < 1e-5 else 0.0


def _safe_arith_eval(expr: str) -> float:
    """Evaluate +, -, *, / arithmetic on numeric literals only. No names, no calls,
    no exponentiation. Raises ValueError on any unsupported expression."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand)
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"unsupported expression node: {type(node).__name__}")
