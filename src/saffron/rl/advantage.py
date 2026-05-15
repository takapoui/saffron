import numpy as np


def compute_grpo_advantages(
    rewards: list[float],  # shape: (B * G,)
    response_lens: list[int],  # shape: (B * G,) — # tokens in each completion
    group_size: int,  # G
) -> list[list[float]]:
    """Returns per-token advantages: one list per completion, len = that completion's token count.
    Step 1: group-normalize rewards → one scalar advantage per completion
    Step 2: broadcast each scalar to all tokens of its completion
    """

    assert len(rewards) == len(response_lens)
    assert len(rewards) % group_size == 0

    rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1, group_size)  # (B, G)
    means = rewards_np.mean(axis=1, keepdims=True)
    stds = rewards_np.std(axis=1, keepdims=True)
    advantages_per_completion = ((rewards_np - means) / (stds + 1e-4)).flatten().tolist()

    # Broadcast scalar advantage across all tokens of each completion
    per_token = [[adv] * n for adv, n in zip(advantages_per_completion, response_lens, strict=True)]
    return per_token
