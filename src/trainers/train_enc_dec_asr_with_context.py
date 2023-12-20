"""Main training script for training of attention based encoder decoder ASR models."""
import os.path
import sys

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    WhisperForConditionalGeneration,
)
from transformers.utils import logging

from utilities.callbacks import init_callbacks
from utilities.context import (
    ContextAwareTrainer,
    SpeechCollatorWithPaddingWithRecordingId,
    instantiate_aed_model,
)
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics
from utilities.general_utils import do_evaluate, do_generate
from utilities.generation_utils import activate_joint_decoding
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArgumentsContext,
)

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser(
        (ModelArgumentsContext, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments)
    )

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset = get_dataset(
        datasets_creation_config_path=data_args.datasets_creation_config,
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        writer_batch_size=data_args.writer_batch_size,
        sampling_rate=data_args.sampling_rate,
        max_input_len=data_args.max_duration_in_seconds,
        min_input_len=data_args.min_duration_in_seconds,
        len_column=training_args.length_column_name,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        train_split=data_args.train_split,
        validation_split=data_args.validation_split,
        unk_token=data_args.unk_token,
        fix_apostrophes=data_args.fix_apostrophes,
        remove_train_unks=data_args.remove_train_unks,
        do_lower_case=data_args.do_lower_case,
        remove_punctuation=data_args.remove_punctuation,
    )

    """"""
    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            lambda x: {model_args.conv_ids_column_name: split + "_" + os.path.splitext(os.path.basename(x))[0]},
            input_columns=["file"],
        )

    def custom_sort(string):
        # Split the string into parts
        parts = string.strip("'").split("-")

        # Extract name, start time, and convert start time to float for proper sorting
        name = parts[0]
        start_time = float(parts[1])

        return name, start_time

    for split in dataset.keys():
        sorted_ids = sorted(dataset[split]["id"], key=custom_sort)
        id_to_turn_mapping = {key: index for index, key in enumerate(sorted_ids)}
        dataset[split] = dataset[split].map(lambda x: {model_args.turn_index_column_name: id_to_turn_mapping[x["id"]]})

    if data_args.validation_slice:
        training_eval_dataset = dataset[data_args.validation_split].shuffle().select(range(data_args.validation_slice))
        # Ensure that transformations are also attached to the sliced validation dataset
        dataset[data_args.validation_split + str(data_args.validation_slice)] = training_eval_dataset
    else:
        training_eval_dataset = dataset[data_args.validation_split]

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(training_args.tokenizer_name)

    # 3. Instantiate model
    model = instantiate_aed_model(model_args, tokenizer, feature_extractor)

    # 4. Update generation config
    gen_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        length_penalty=gen_args.length_penalty,
        early_stopping=gen_args.early_stopping,
        eos_token_id=tokenizer.eos_token_id,
        max_length=gen_args.max_length,
        num_beams=gen_args.num_beams,
    )
    logger.info(f"Model updating generation config:\n {str(gen_config)}")
    training_args.generation_max_length = gen_args.max_length
    training_args.generation_num_beams = gen_args.num_beams
    model.generation_config = gen_config
    training_args.generation_config = gen_config

    if isinstance(model, WhisperForConditionalGeneration) and model_args.whisper_task and model_args.whisper_language:
        model.config.suppress_tokens = []
        model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
            language=model_args.whisper_language, task=model_args.whisper_task
        )

    # 5. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)
    callbacks.append(model.context_manager)

    # 6. Initialize data collator
    data_collator = SpeechCollatorWithPaddingWithRecordingId(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
    )

    # 7. Initialize trainer
    trainer = ContextAwareTrainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred, gen_args.wandb_predictions_to_save),
        conv_ids_column_name=model_args.conv_ids_column_name,
        turn_index_column_name=model_args.turn_index_column_name,
    )

    # 8. Train model
    if training_args.do_train:
        if gen_args.decoding_ctc_weight > 0 and training_args.joint_decoding_during_training:
            activate_joint_decoding(
                model=model,
                ctc_weight=gen_args.decoding_ctc_weight,
                ctc_margin=gen_args.ctc_margin,
                num_tokens=len(tokenizer),
                eos_token=tokenizer.eos_token_id,
                external_lm=None,
                external_lm_weight=0,
            )
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 9. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            training_args=training_args,
            data_args=data_args,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 10. N-best generation
    if training_args.do_generate:
        do_generate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            data_args=data_args,
            eos_token_id=tokenizer.eos_token_id,
            gen_config=gen_config,
        )
