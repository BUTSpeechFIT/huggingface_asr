import torch
from transformers import LogitsProcessor, PreTrainedModel


class LMRescorerLogitsProcessor(LogitsProcessor):
    """Logits Processor to rescore the next token scores with a language model."""

    def __init__(self, lm_weight: float, lm_model: PreTrainedModel, device: torch.device):
        super().__init__()
        self.lm_model = lm_model.to(device)
        self.lm_weight = lm_weight
        # self.past_key_values = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # TODO: KarelB: Can you implement the past_key_values logic?
        outputs = self.lm_model(
            input_ids,
            # input_ids[:, -1]
            # past_key_values=self.past_key_values,
            # use_cache=True
        )
        # self.past_key_values = outputs.past_key_values
        lm_scores = torch.nn.functional.log_softmax(outputs.logits[:, -1, :], dim=-1)
        next_token_scores = scores + self.lm_weight * lm_scores
        return next_token_scores
