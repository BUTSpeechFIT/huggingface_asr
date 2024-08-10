"""Main training script for training of CTC ASR models."""
import string
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
)
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging
from whisper_ctc import WhisperEncoderForCTC

import wandb
from utilities.callbacks import init_callbacks
from utilities.collators import SpeechCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import ctc_greedy_decode, get_metrics, write_wandb_pred
from utilities.general_utils import do_evaluate
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)


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


def get_model(model_args):
    llm = AutoModelForCausalLM.from_pretrained(model_args.llm_model)
    model = WhisperEncoderForCTC.from_pretrained(
        model_args.from_pretrained, llm_dim=llm.config.hidden_size, sub_sample=True
    )
    llm_head = llm.lm_head
    unwanted_tokens_mask = np.ones((llm_head.weight.shape[0],), dtype=bool)
    unwanted_tokens_mask[removed_token_ids] = False
    llm_head.out_features = len(tokenizer) - len(removed_token_ids)
    llm_head.weight = torch.nn.Parameter(llm_head.weight[unwanted_tokens_mask])
    model.lm_head = llm_head
    model.lm_head.weight.requires_grad = False
    model.config.ctc_zero_infinity = True
    model.config.blank_token_id = tokenizer.pad_token_id
    for module in [
        *model.encoder.layers[: int(len(model.encoder.layers) * 2 / 3)],
        model.encoder.conv1,
        model.encoder.conv2,
    ]:
        for param in module.parameters():
            param.requires_grad = False
    del llm
    return model


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser(
        (CustomModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments)
    )

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset, training_eval_dataset = get_dataset(
        data_args=data_args,
        len_column=training_args.length_column_name,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model)
    tokenizer.padding_side = "right"

    new_token_ids_mapping, new_token_ids_mapping_inverted, removed_token_ids = get_token_subset(tokenizer)

    # 3. Instantiate model
    model = get_model(model_args)

    # 4. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 5. Initialize data collator
    data_collator = CustomCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
        mask_unks=training_args.mask_unks,
        pad_to_multiple_of=data_args.pad_to_multiples_of,
        token_mapping=new_token_ids_mapping,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        preprocess_logits_for_metrics=ctc_greedy_decode,
        compute_metrics=lambda pred: compute_metrics_ctc(
            tokenizer, new_token_ids_mapping_inverted, pred, gen_args.wandb_predictions_to_save
        ),
    )

    # 6. Train
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 7. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            data_args=data_args,
            training_args=training_args,
        )
