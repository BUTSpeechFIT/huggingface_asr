import itertools as it
import math
import re
import string
import subprocess  # nosec
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer, Seq2SeqTrainer, Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

import wandb
from utilities import data_utils
from utilities.collators import SpeechCollatorWithPadding
from utilities.eval_utils import get_metrics, write_wandb_pred
from utilities.general_utils import function_aggregator, text_transform_partial
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


@dataclass
class CustomModelArguments(ModelArguments):
    llm_model: Optional[str] = field(default="google/gemma-2b-it", metadata={"help": "The model to use for the LLM."})


@dataclass
class CustomCollator(SpeechCollatorWithPadding):
    token_mapping: dict = None

    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["labels"].apply_(lambda x: self.token_mapping[int(x)] if x != -100 else x)
        return batch


def compute_metrics_ctc(
    tokenizer: PreTrainedTokenizer, token_mapping, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred.predictions[pred.predictions == -100] = tokenizer.pad_token_id
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
    map_tokens = np.vectorize(lambda x: token_mapping[int(x)])
    label_str = [
        label if label else "-"
        for label in tokenizer.batch_decode(map_tokens(pred.label_ids), skip_special_tokens=True)
    ]
    pred_str = tokenizer.batch_decode(map_tokens(pred.predictions), skip_special_tokens=True)

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)


def ctc_greedy_decode(logits: torch.Tensor, _: torch.Tensor, blank, pad_token_id) -> torch.Tensor:
    idxs = torch.argmax(logits, dim=-1)
    for i, prediction in enumerate(idxs):
        deduplicated = [k for k, g in it.groupby(prediction) if k != blank]
        idxs[i, : len(deduplicated)] = torch.tensor(deduplicated)
        idxs[i, len(deduplicated) :] = pad_token_id
    return idxs


def get_token_subset(tokenizer):
    removed_token_ids = []
    unwanted_tokens = []
    charset = string.digits + string.ascii_lowercase + string.punctuation + " "
    new_token_ids_mapping = {}

    # keep only english lowercase characters
    for i in range(len(tokenizer.vocab)):
        token = tokenizer.decode(i)
        if all(char in charset for char in token) or token in tokenizer.all_special_tokens:
            new_token_ids_mapping[i] = len(new_token_ids_mapping)
        else:
            unwanted_tokens.append(token)
            removed_token_ids.append(i)

    new_token_ids_mapping_inverted = {v: k for k, v in new_token_ids_mapping.items()}

    return new_token_ids_mapping, new_token_ids_mapping_inverted, removed_token_ids


def save_predictions(
    tokenizer: PreTrainedTokenizer,
    predictions: PredictionOutput,
    path: str,
    text_transforms: Optional[Callable] = None,
    token_mapping: Optional[dict] = None,
):
    """Save predictions to a csv file and sclite files to evaluate wer."""
    pred_ids = predictions.predictions

    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    map_tokens = np.vectorize(lambda x: token_mapping[int(x)])

    pred_str = [
        re.sub(
            r"\[(\S*)\]",
            r"\1",
            re.sub(r"\(\[[^\]]+\]\)(?=\w)|(?<=\w)\(\[[^\]]+\]\)", "-", text_transforms(pred)),
        )
        if text_transforms
        else pred
        for pred in tokenizer.batch_decode(map_tokens(pred_ids), skip_special_tokens=True)
    ]
    label_str = [
        re.sub(r"\[(\S*)\]", r"\1", re.sub(r"\(\[[^\]]+\]\)(?=\w)|(?<=\w)\(\[[^\]]+\]\)", "-", label)) if label else ""
        for label in tokenizer.batch_decode(map_tokens(label_ids), skip_special_tokens=True)
    ]
    df = pd.DataFrame({"label": label_str, "prediction": pred_str})
    df.to_csv(path, index=False)

    sclite_files = [path.replace(".csv", f"_{type}.trn") for type in ["hyp", "ref"]]
    for strings, file_to_save in zip([pred_str, label_str], sclite_files):
        with open(file_to_save, "w") as file_handler:
            for index, text in enumerate(strings):
                file_handler.write(f"{text} (utterance_{index})\n")

    sclite_cmd = f"sclite -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum dtl"
    process = subprocess.Popen(sclite_cmd.split())  # nosec
    try:
        process.wait(30)
    except subprocess.TimeoutExpired:
        process.kill()
        logger.warning("Sclite evaluation timed out.")


def do_evaluate(
    trainer: Union[Trainer, Seq2SeqTrainer],
    dataset: DatasetDict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    gen_args: Optional[GenerationArguments],
    training_args: GeneralTrainingArguments,
    data_args: DataTrainingArguments,
    token_mapping: Optional[dict] = None,
):
    if data_args.test_splits is None:
        return
    if hasattr(gen_args, "override_for_evaluation") and gen_args.override_for_evaluation is not None:
        num_beams_orig = model.generation_config.num_beams
        model.generation_config.update_from_string(gen_args.override_for_evaluation)
        trainer.args.generation_num_beams = model.generation_config.num_beams
        if model.generation_config.num_beams != num_beams_orig:
            trainer.args.per_device_eval_batch_size = math.ceil(
                trainer.args.per_device_eval_batch_size / (model.generation_config.num_beams / num_beams_orig)
            )
    for split in data_args.test_splits:
        start_time = time.time()
        if isinstance(trainer, Seq2SeqTrainer):
            predictions = trainer.predict(
                dataset[split],
                output_hidden_states=True,
            )
        else:
            predictions = trainer.predict(
                dataset[split],
            )
        end_time = time.time()
        tokens_produced = (predictions.predictions != tokenizer.pad_token_id).sum()
        logger.info(f"Metrics for {split} split: {predictions.metrics}")
        logger.info(f"Time taken for evaluation: {end_time - start_time} seconds")
        logger.info(f"Tokens produced: {tokens_produced}")
        logger.info(f"Tokens per second: {tokens_produced / (end_time - start_time)}")

        if gen_args.post_process_predictions and data_args.text_transformations is not None:
            callable_transform = function_aggregator(
                [
                    text_transform_partial(
                        getattr(data_utils, transform_name, lambda x, label_column: {label_column: x})
                    )
                    for transform_name in data_args.text_transformations
                ]
            )
        else:
            callable_transform = None

        save_predictions(
            tokenizer,
            predictions,
            f"{training_args.output_dir}/" f'predictions_{split}_wer{100 * predictions.metrics["test_wer"]:.2f}.csv',
            callable_transform,
            token_mapping=token_mapping,
        )
