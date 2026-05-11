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
    idx = idx.unsqueeze(0).repeat(n_samples, 1).to(device)
    tokens = model.generate(
        idx=idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    completions = [
        tokenizer.decode(tokens[i, :].tolist())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        for i in range(tokens.shape[0])
    ]
    model.train()
    return completions
