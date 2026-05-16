from dataclasses import dataclass
from typing import cast

import torch

from ..model import BaseModel


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor  # (B*G, T_full) — prompt + completion + padding
    attention_mask: torch.Tensor  # (B*G, T_full) — 1 on real tokens, 0 on pad
    response_mask: torch.Tensor  # (B*G, T_full) — 1 on completion tokens only
    completion_texts: list[str]  # length B*G — for reward computation
    response_lens: list[int]  # length B*G — for compute_grpo_advantages


def rollout(
    model: BaseModel,
    prompt_input_ids: torch.Tensor,  # (B, T_prompt)
    prompt_attention_mask: torch.Tensor,  # (B, T_prompt)
    group_size: int,
    max_new_tokens: int,
    temperature: float,
) -> RolloutBatch:

    idx = prompt_input_ids.repeat_interleave(group_size, dim=0)
    attn = prompt_attention_mask.repeat_interleave(group_size, dim=0)

    tokenizer = model.get_tokenizer()
    stop_token_ids = tokenizer.stop_token_ids

    generated = model.generate(
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
        attention_mask=attn,
    )  # (B*G, T_full)

    pad_id = stop_token_ids[0]
    T_prompt = idx.shape[1]
    completion = generated[:, T_prompt:]
    is_pad = (completion == pad_id).int()
    pads_before = is_pad.cumsum(dim=-1) - is_pad
    completion_valid = pads_before == 0

    attention_mask = torch.ones_like(generated)
    # Preserve the (repeated) prompt mask so left-padded prompts aren't promoted to 1s.
    attention_mask[:, :T_prompt] = attn
    attention_mask[:, T_prompt:] = completion_valid.long()

    response_mask = attention_mask.clone()
    response_mask[:, :T_prompt] = 0

    response_lens = cast(list[int], response_mask.sum(dim=-1).tolist())  # pyright: ignore[reportUnknownMemberType]

    stop_set = set(stop_token_ids)
    completion_texts = [
        tokenizer.decode(
            [
                t
                for t in cast(
                    list[int],
                    generated[i, T_prompt : T_prompt + response_lens[i]].tolist(),  # pyright: ignore[reportUnknownMemberType]
                )
                if t not in stop_set
            ]
        )
        for i in range(generated.shape[0])
    ]

    return RolloutBatch(
        input_ids=generated,
        attention_mask=attention_mask,
        response_mask=response_mask,
        completion_texts=completion_texts,
        response_lens=response_lens,
    )
