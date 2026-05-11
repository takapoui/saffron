import contextlib
from typing import cast

import torch

from ..models import BaseModel
from ..tokenizer import HFTokenizer, Tokenizer


@torch.no_grad()
def evaluate_generate(
    model: BaseModel,
    tokenizer: Tokenizer,
    device: str,
    prompt: str,
    n_samples: int = 5,
    max_new_tokens: int = 50,
    use_chat_template: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[str]:
    model.eval()
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
    stop_token_ids = [tokenizer.eot_token]
    if isinstance(tokenizer, HFTokenizer) and (im_end := tokenizer.im_end_token) is not None:
        stop_token_ids.append(im_end)
    tokens = model.generate(
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
    )
    completions: list[str] = []
    for i in range(tokens.shape[0]):
        seq = cast(list[int], tokens[i, :].tolist())  # pyright: ignore[reportUnknownMemberType]
        for stop_id in stop_token_ids:
            with contextlib.suppress(ValueError):
                seq = seq[: seq.index(stop_id, prompt_len)]
        completions.append(tokenizer.decode(seq))
    model.train()
    return completions
