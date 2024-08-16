"""Main training script for training of CTC ASR models."""
import sys

from local_utils import (
    CustomCollator,
    CustomModelArgumentsPrompting,
    compute_metrics_ctc,
    do_evaluate,
    get_token_subset,
)
from safetensors.torch import load_file
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    SpeechEncoderDecoderConfig,
)
from transformers.utils import logging

from models import LLMASRModel, get_model
from utilities.callbacks import init_callbacks
from utilities.data_utils import get_dataset
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
)

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser(
        (CustomModelArgumentsPrompting, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments)
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
    asr, llm = get_model(model_args)

    merged_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(asr.config, llm.config)
    model = LLMASRModel(
        merged_config, asr, llm, model_args.number_of_prompt_tokens, model_args.freeze_asr, model_args.freeze_llm
    )
    model.encoder.load_state_dict(load_file(model_args.asr_model_checkpoint))

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
        model_input_name=asr.main_input_name,
        mask_unks=training_args.mask_unks,
        pad_to_multiple_of=data_args.pad_to_multiples_of,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
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
