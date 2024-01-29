"""Main training script for training of attention based encoder decoder ASR models."""
import sys

from transformers import AutoFeatureExtractor, HfArgumentParser
from transformers.utils import logging

from utilities.callbacks import GumbelTemperatureCallback, init_callbacks
from utilities.collators import DataCollatorForWav2Vec2Pretraining
from utilities.data_utils import get_dataset
from utilities.model_utils import instantiate_speech_encoder_model
from utilities.training_arguments import (
    DataTrainingArguments,
    GenerationArguments,
    ModelArguments,
    PretrainingArguments,
)
from utilities.training_utils import SSLTrainer

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PretrainingArguments, GenerationArguments))

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
        text_transformations=data_args.text_transformations,
        split_long_segments_to_chunks=data_args.split_long_segments_to_chunks,
        filter_empty_labels=data_args.filter_empty_labels,
    )

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

    # 2. Create feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)

    # 3. Instantiate model
    model = instantiate_speech_encoder_model(model_args, feature_extractor)

    # 4. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    temperature_callback = GumbelTemperatureCallback(
        training_args.gumbel_temperature_decay,
        training_args.min_gumbel_temperature,
        training_args.max_gumbel_temperature,
    )
    callbacks.append(temperature_callback)

    # 6. Initialize data collator
    data_collator = DataCollatorForWav2Vec2Pretraining(
        feature_extractor=feature_extractor,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        model=model,
        audio_path=data_args.audio_column_name,
        model_input_name=model.main_input_name,
    )

    # 7. Initialize trainer
    trainer = SSLTrainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
    )

    # 8. Train model
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)
