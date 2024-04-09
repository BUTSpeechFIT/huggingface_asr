import torch
from transformers import LogitsProcessor, PreTrainedModel


class LMRescorerLogitsProcessor(LogitsProcessor):
    """Logits Processor to rescore the next token scores with a language model."""

    def __init__(self, lm_weight: float, lm_model: PreTrainedModel, device: torch.device):
        super().__init__()
        self.lm_model = lm_model.to(device)
        self.lm_weight = lm_weight
        # self.past_key_values = None

    @staticmethod
    def analyze_predictions(scores, lm_scores, next_token_scores, input_ids, k=10, tokenizer="Lakoc/ted_uni500"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        best_att_ids = scores.topk(k=k, dim=1)
        best_ctc_ids = lm_scores.topk(k=k, dim=1)
        best_ids = next_token_scores.topk(k=k, dim=1)

        def print_prediction(best_ids, name):
            new_tensor = torch.zeros((best_ids.indices.shape[0], best_ids.indices.shape[1] * 2), dtype=torch.long)
            new_tensor[:, 0::2] = best_ids.indices
            new_tensor[:, 1::2] = 1
            print(f"{name}:")
            for index, (next_ids, scores) in enumerate(zip(tokenizer.batch_decode(new_tensor), best_ids.values)):
                print(f"HYP {index}:\n{next_ids} {scores}")

        print(f"PREFIX:")
        for index, prefix in enumerate(tokenizer.batch_decode(input_ids)):
            print(f"HYP {index}:\n{prefix}")
        print_prediction(best_att_ids, "ACCUSTIC_SCORES")
        print()
        print_prediction(best_ctc_ids, "LM_SCORES")
        print()
        print_prediction(best_ids, "NEXT_TOKEN_SCORES")
        print()

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
        # self.analyze_predictions(scores, lm_scores, next_token_scores, input_ids)
        return next_token_scores
