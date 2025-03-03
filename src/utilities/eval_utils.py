import itertools as it
import re
from typing import Dict, List

import torch
import wandb
from torchaudio.models.decoder import ctc_decoder
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput

import jiwer
from jiwer import cer, compute_measures
from utilities.general_utils import save_predictions

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


def extract_err_rate_from_sclite(file_content):
    """
    Extracts the Err (Error Rate) value from the given file content.

    Parameters:
        file_content (str): The content of the file as a string.

    Returns:
        float: The extracted Err value.
    """
    # Regular expression to find the Err value from the line containing it
    match = re.search(r'.*\s(\d+.\d+)\s+\d+.\d+', file_content)
    if match:
        return float(match.group(1)) / 100
    else:
        raise ValueError("Error rate not found in the provided content.")


def transform_text(text: str) -> list[str]:
    """
    done:
    - remove (LNG), (UNK), (SPN)
    - remove punctuation [,.!?;:]
    - remove truncated words with '-wrd', 'wrd-'
    - remove capitalization

    not done:
    - map shortened lexical forms (I've, I'll), or colloquial words
    - more variants -> expand to all of them
    - digit verbalization
    """

    tokens = text.split()

    # remove non-lexical symbols, punctuation
    remove_tokens = {"(LNG)", "(UNK)", "(SPN)", ",", ".", "?", "!", ";", ":"}
    tokens = [t for t in tokens if t not in remove_tokens]

    # remove partial words
    tokens = [t for t in tokens if (t[0] != "-" and t[-1] != "-")]

    text = " ".join(tokens)

    # remove capitalization
    text = text.lower()

    return text.split()


def get_metrics(labels: List[str], preds: List[str]):
    metrics = compute_measures(labels, preds)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]

    hyps_mapped = [transform_text(hyp) for hyp in preds]
    refs_mapped = [transform_text(ref) for ref in labels]

    wer_mapped = jiwer.wer(
        reference=refs_mapped,
        hypothesis=hyps_mapped,
        reference_transform=jiwer.NoTransform(),
        hypothesis_transform=jiwer.NoTransform(),
    )

    return {"cer": cer(labels, preds), **metrics, "wer_mapped": wer_mapped}


def ctc_greedy_decode(logits: torch.Tensor, blank, pad_token_id) -> torch.Tensor:
    idxs = torch.argmax(logits, dim=-1)
    for i, prediction in enumerate(idxs):
        deduplicated = [k for k, g in it.groupby(prediction) if k != blank]
        idxs[i, : len(deduplicated)] = torch.tensor(deduplicated)
        idxs[i, len(deduplicated):] = pad_token_id
    return idxs


def ctc_beam_decode(logits: torch.Tensor, _: torch.Tensor, tokenizer, beam_size) -> torch.Tensor:
    beam_search_decoder = ctc_decoder(
        lexicon=None,
        tokens=list(tokenizer.get_vocab().keys()),
        beam_size=beam_size,
        beam_size_token=beam_size,
        blank_token=tokenizer.pad_token,
        sil_token=tokenizer.pad_token,
    )

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    output = beam_search_decoder(
        log_probs.float().cpu(), torch.tensor(logits.shape[1], dtype=torch.int32).repeat(logits.shape[0])
    )
    predictions = [pred[0].tokens.to(logits.device) for pred in output]
    output = torch.nn.utils.rnn.pad_sequence(predictions, batch_first=True, padding_value=tokenizer.pad_token_id)
    return output


def compute_metrics_ctc(
        tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10, path_to_save_predictions: str = None
) -> Dict[str, float]:
    pred.predictions[pred.predictions == -100] = tokenizer.pad_token_id
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    if path_to_save_predictions is not None:
        save_predictions(tokenizer=tokenizer, predictions=pred, text_transforms=None,path=path_to_save_predictions)
    is_degenerated_vocab = hasattr(tokenizer, "vocab_type") and tokenizer.vocab_type == "degenerated"
    label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=not is_degenerated_vocab)
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

    pred_ids[pred_ids == -100] = tokenizer.pad_token_id


    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)
