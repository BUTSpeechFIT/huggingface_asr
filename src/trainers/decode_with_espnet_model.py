"""Main training script for training of attention based encoder decoder ASR models."""
import os
import sys
from dataclasses import dataclass, field
from functools import partial

import multiprocess as mp
import torch
from espnet2.bin.s2t_inference import Speech2Text
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.utils import logging

from utilities import data_utils
from utilities.data_utils import get_dataset
from utilities.general_utils import function_aggregator, text_transform_partial
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)


# Function to process samples and save predictions with corresponding labels
def process_sample(sample, pipeline, callable_transform, text_column):
    hypothesis = pipeline(sample["audio"]["array"])
    prediction = hypothesis[0][0]
    if callable_transform:
        prediction = callable_transform(prediction)
    return prediction, sample[text_column]


@dataclass
class GenArgs(GenerationArguments):
    """Arguments pertaining to generation of predictions from a trained model."""

    num_workers: int = field(
        default=1,
        metadata={
            "help": "The number of workers to use for generation. "
            "Will be used to create a pool of workers to process the samples."
        },
    )


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenArgs))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset, training_eval_dataset = get_dataset(
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
        validation_slice_str=data_args.validation_slice,
        cut_validation_from_train=data_args.cut_validation_from_train,
        seed=data_args.validation_slice_seed,
        reshuffle_at_start=data_args.reshuffle_at_start,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Instantiate model
    mp.set_start_method("spawn", force=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Speech2Text.from_pretrained(model_args.from_pretrained, lang_sym="<en>", beam_size=1, device=device)

    # 3. Init callable transformation
    if gen_args.post_process_predicitons and data_args.text_transformations is not None:
        callable_transform = function_aggregator(
            [
                text_transform_partial(getattr(data_utils, transform_name, lambda x, label_column: {label_column: x}))
                for transform_name in data_args.text_transformations
            ]
        )
    else:
        callable_transform = None

    # 4. Generate predictions for each split
    manager = mp.Manager()
    for split_name in data_args.test_splits:
        pred_str = []
        label_str = []
        logger.info(f"Generating predictions for split: {split_name}")
        split = dataset[split_name]

        # 4a. Init array to store predictions and labels and manager
        hyp_list = manager.list()
        label_list = manager.list()
        # pylint: disable=not-callable
        pool = mp.Pool(gen_args.num_workers)

        # 4b. Process samples in parallel
        for result in tqdm(
            pool.imap(
                partial(
                    process_sample,
                    pipeline=pipeline,
                    callable_transform=callable_transform,
                    text_column=data_args.text_column_name,
                ),
                split,
            ),
            total=len(split),
            desc=f"Processing samples in {split_name} dataset",
        ):
            hyp, label = result
            hyp_list.append(hyp)
            label_list.append(label)
        pool.close()
        pool.join()

        # 4c. Parse results
        out_path = (
            f"{training_args.output_dir}/"
            f"predictions_{split_name}_{model_args.from_pretrained.replace('/', '_')}.csv"
        )
        sclite_files = [out_path.replace(".csv", f"_{type}.trn") for type in ["hyp", "ref"]]
        for strings, file_to_save in zip([pred_str, label_str], sclite_files):
            with open(file_to_save, "w") as file_handler:
                for index, string in enumerate(strings):
                    file_handler.write(f"{string} (utterance_{index})\n")

        # evaluate wer also with sclite
        os.system(f"sclite -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum")  # nosec
