import torch
import torch.nn.functional as F

from ..model import BaseModel


def compute_token_log_probs(
    model: BaseModel,
    input_ids: torch.Tensor,  # (B, T)
    attention_mask: torch.Tensor,  # (B, T)
    temperature: float,
) -> torch.Tensor:  # (B, T-1) — log p(token_t | token_<t)
    """Forward pass on input_ids, return per-token log-probabilities of the actual next tokens.

    Predicts token t+1 from tokens [0..t], so the output has T-1 positions
    (no prediction for the very last token's successor since there is none).
    """
    logits, _ = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
    logits /= temperature  # (B, T-1, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T-1, vocab_size)
    return torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
