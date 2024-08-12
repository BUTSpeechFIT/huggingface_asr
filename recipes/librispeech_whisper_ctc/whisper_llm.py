"""Main training script for training of CTC ASR models."""
import sys

import numpy as np
import torch
from local_utils import (
    CustomCollator,
    CustomModelArguments,
    compute_metrics_ctc,
    ctc_greedy_decode,
    do_evaluate,
    get_token_subset,
)
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)
from transformers.utils import logging
from whisper_ctc import WhisperEncoderForCTC

from utilities.callbacks import init_callbacks
from utilities.data_utils import get_dataset
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
)


def get_model(m_args: CustomModelArguments):
    llm = AutoModelForCausalLM.from_pretrained(m_args.llm_model)
    new_model = WhisperEncoderForCTC.from_pretrained(
        m_args.from_pretrained, llm_dim=llm.config.hidden_size, sub_sample=True
    )
    llm_head = llm.lm_head
    unwanted_tokens_mask = np.ones((llm_head.weight.shape[0],), dtype=bool)
    unwanted_tokens_mask[removed_token_ids] = False
    llm_head.out_features = len(tokenizer) - len(removed_token_ids)
    llm_head.weight = torch.nn.Parameter(llm_head.weight[unwanted_tokens_mask])
    new_model.lm_head = llm_head
    new_model.lm_head.weight.requires_grad = False
    new_model.config.ctc_zero_infinity = True
    new_model.config.blank_token_id = tokenizer.pad_token_id
    new_model.config.bos_token_id = tokenizer.bos_token_id
    new_model.config.eos_token_id = tokenizer.eos_token_id
    new_model.config.pad_token_id = tokenizer.pad_token_id
    for module in [
        *new_model.encoder.layers[: int(len(new_model.encoder.layers) * 5 / 6)],
        new_model.encoder.conv1,
        new_model.encoder.conv2,
    ]:
        for param in module.parameters():
            param.requires_grad = False
    del llm
    return new_model


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
        preprocess_logits_for_metrics=lambda x, y: ctc_greedy_decode(
            x, y, model.config.blank_token_id, tokenizer.pad_token_id
        ),
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
            token_mapping=new_token_ids_mapping_inverted,
        )
