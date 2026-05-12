from __future__ import annotations

from typing import cast

import torch

from ..model import BaseModel
from ..tokenizer import HFTokenizer


@torch.no_grad()
def evaluate_generate(
    model: BaseModel,
    device: str,
    prompt: str,
    n_samples: int = 5,
    max_new_tokens: int = 50,
    use_chat_template: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[str]:
    model.eval()
    tokenizer = model.get_tokenizer()
    if use_chat_template:
        assert isinstance(tokenizer, HFTokenizer), "use_chat_template requires HFTokenizer"
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    else:
        prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long)
    prompt_len = idx.shape[0]
    idx = idx.unsqueeze(0).repeat(n_samples, 1).to(device)
    stop_token_ids = tokenizer.stop_token_ids
    tokens = model.generate(
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
        attention_mask=torch.ones_like(idx),
    )
    stop_set = set(stop_token_ids)
    completions: list[str] = []
    for i in range(tokens.shape[0]):
        seq = cast(list[int], tokens[i, :].tolist())  # pyright: ignore[reportUnknownMemberType]
        stop = len(seq)
        for j in range(prompt_len, len(seq)):
            if seq[j] in stop_set:
                stop = j
                break
        completions.append(tokenizer.decode(seq[:stop]))
    model.train()
    return completions
