from typing import Dict, List

import torch
from jiwer import cer, compute_measures
from torchaudio.models.decoder import ctc_decoder
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput

import wandb


def write_wandb_pred(pred_str: List[str], label_str: List[str], rows_to_log: int = 10):
    current_step = wandb.run.step
    columns = ["id", "label_str", "hyp_str"]
    wandb.log(
        {
            f"eval_predictions/step_{int(current_step)}": wandb.Table(
                columns=columns,
                data=[
                    [i, ref, hyp] for i, hyp, ref in zip(range(min(len(pred_str), rows_to_log)), pred_str, label_str)
                ],
            )
        },
        current_step,
    )


def get_metrics(labels: List[str], preds: List[str]):
    metrics = compute_measures(labels, preds)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]
    return {"cer": cer(labels, preds), **metrics}


def ctc_greedy_decode(logits: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def ctc_beam_decode(logits: torch.Tensor, _: torch.Tensor, tokenizer, beam_size) -> torch.Tensor:
    beam_search_decoder = ctc_decoder(
        lexicon=None,
        tokens=list(tokenizer.get_vocab().keys()),
        beam_size=beam_size,
        beam_size_token=beam_size,
        blank_token=tokenizer.pad_token,
        sil_token=tokenizer.replace_word_delimiter_char,
    )

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    output = beam_search_decoder(
        log_probs.float().cpu(), torch.tensor(logits.shape[1], dtype=torch.int32).repeat(logits.shape[0])
    )
    predictions = [torch.tensor(pred[0].tokens) for pred in output]
    output = torch.nn.utils.rnn.pad_sequence(predictions, batch_first=True, padding_value=tokenizer.pad_token_id)
    return output


def compute_metrics_ctc(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred.predictions[pred.predictions == -100] = tokenizer.pad_token_id
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
    is_degenerated_vocab = hasattr(tokenizer, "vocab_type") and tokenizer.vocab_type == "degenerated"
    label_str = [
        label if label else "-"
        for label in tokenizer.batch_decode(pred.label_ids, skip_special_tokens=not is_degenerated_vocab)
    ]
    pred_str = tokenizer.batch_decode(
        pred.predictions, skip_special_tokens=not is_degenerated_vocab, group_ctc_tokens=is_degenerated_vocab
    )

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)


def compute_metrics(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)
