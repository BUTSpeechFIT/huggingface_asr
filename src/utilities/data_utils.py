"""Utilities for data loading and preprocessing."""
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch.distributed
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers.utils import logging

from utilities.english_normalizer import EnglishNormalizer
from utilities.training_arguments import DataTrainingArguments

logger = logging.get_logger("transformers")

whisper_normalizer = EnglishNormalizer()
special_tokens = [
    "([noise])",
    "([laughter])",
    "([vocalized noise])",
    "([hesitation])",
    "([breath])",
    "([cough])",
    "([silence])",
    "([noise])",
    "([pause])",
    "([skip])",
    "([sneeze])",
]

spec_tokens_mapping_gigaspeech = {"<COMMA>": ",", "<PERIOD>": ".", "<QUESTIONMARK>": "?", "<EXCLAMATIONMARK>": "!"}

tokens_escaped_regex = re.compile("\(\S+\)")

MIN_INPUT_LEN = 0.1
MAX_INPUT_LEN = 100.0


def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return torch.distributed.get_rank()


class DistributedContext:
    """Context manager for distributed training."""

    def __init__(self):
        """Initializes distributed context."""
        self.local_rank = None
        self.world_size = None

    def __enter__(self):
        """Initializes distributed context."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.local_rank = get_local_rank()
            self.global_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Performs barrier synchronization."""
        if self.world_size > 1:
            torch.distributed.barrier()

    def wait_before(self):
        if self.world_size > 1:
            if self.local_rank > 0:
                logger.info(f"Rank {self.global_rank}: Waiting for main process to perform operation.")
                torch.distributed.barrier()

    def wait_after(self):
        if self.world_size > 1:
            if self.local_rank == 0:
                logger.info(f"Rank {self.global_rank}: Waiting for other processes to finish operation.")
                torch.distributed.barrier()


def distributed_process(dataset, process_by, **kwargs):
    """Performs distributed processing of dataset."""
    with DistributedContext() as context:
        context.wait_before()
        mapped_dataset = getattr(dataset, process_by)(**kwargs)
        context.wait_after()
    return mapped_dataset


"""

Text manipulation functions.

"""


def do_lower_case(example: str, label_column: str) -> Dict[str, str]:
    """Lower cases batch."""
    return {label_column: example.lower()}


def remove_punctuation(example: str, label_column: str) -> Dict[str, str]:
    """Removes punctuation."""
    return {label_column: re.sub(r"[!\"#$%&\'()*+,./\\:;<=>?@^_`{|}~]", "", example)}


def remove_multiple_whitespaces_and_strip(example: str, label_column: str) -> Dict[str, str]:
    """Removes multiple whitespaces from batch."""
    return {label_column: re.sub(r"\s+", " ", example).strip()}


def clean_special_tokens_english(example: str, label_column: str) -> Dict[str, str]:
    """Cleans special tokens from labels."""
    return {label_column: tokens_escaped_regex.sub("", example)}


def transforms_unfinished_words_to_unks(example: str, label_column: str) -> Dict[str, str]:
    """Transforms unfinished words to UNKs."""
    return {label_column: re.sub(r"\(?\w+-\)?", "([unk])", example)}


tedlium_contractions = [" 's", " 't", " 're", " 've", " 'm", " 'll", " 'd", " 'clock", " 'all"]


def fix_tedlium_apostrophes(example: str, label_column: str) -> Dict[str, str]:
    for contraction in tedlium_contractions:
        example = example.replace(contraction, contraction[1:])
    return {label_column: example.replace(r"\s+ '", r" '")}


def filter_empty_transcriptions(example: str) -> bool:
    """Filters out empty transcriptions."""
    return example != ""


def filter_tedlium_empty_labels(example: str) -> bool:
    """Filters out empty transcriptions."""
    return example != "ignore_time_segment_in_scoring"


def whisper_normalize_english(example: str, label_column: str) -> Dict[str, str]:
    """Normalizes text using adapted whisper normalizer."""
    return {label_column: whisper_normalizer(example)}


def map_gigaspeech_spec_tokens(example: str, label_column: str) -> Dict[str, str]:
    """Maps special tokens from GigaSpeech to common ones."""
    for token, replacement in spec_tokens_mapping_gigaspeech.items():
        example = example.replace(token, replacement)
    return {label_column: example}


"""

Audio manipulation functions.

"""


def audio_object_stripper(audio: Union[Dict, np.ndarray, List[float]], key="array"):
    """Strips audio object to numpy array."""
    audio_array = audio[key] if isinstance(audio, dict) and key in audio else audio
    trimmed = np.trim_zeros(audio_array)
    return trimmed


def split_long_segments_to_chunks_fun(
    audios: List[Dict],
    lens: List[float],
    audio_column: str,
    length_column_name: str,
    max_input_len: float,
    sampling_rate: int,
) -> Dict[str, List[List[float]]]:
    audio_encoder = Audio(sampling_rate=sampling_rate, mono=True)
    chunks = []
    lens_new = []
    for index, example_len in enumerate(lens):
        for i in range(0, len(audios[index]["array"]), int(max_input_len * sampling_rate)):
            new_chunk = audio_object_stripper(audios[index])[i : i + int(max_input_len * sampling_rate)]
            chunks.append(audio_encoder.encode_example({"array": new_chunk, "sampling_rate": sampling_rate}))
            lens_new.append(len(new_chunk) / sampling_rate)
    return {audio_column: chunks, length_column_name: lens_new}


def filter_sequences_in_range_batched(batch: List[float], max_input_len: float, min_input_len: float) -> List[bool]:
    """Filters out sequences form dataset which are in bounds."""
    arr = np.array(batch)
    return (arr <= max_input_len) & (arr >= min_input_len)


def filter_zero_length_audio_batched(lens: List[List[float]]) -> List[bool]:
    """Filters out sequences form dataset which are in bounds."""
    arr = np.array(lens)
    return arr != 0.0


def extract_lens_batched(audios: List[List[float]], len_column: str, sampling_rate: int) -> Dict[str, List[float]]:
    """Extracts audio lens from dataset."""
    lens = [len(audio_object_stripper(example)) / sampling_rate for example in audios]
    batch = {len_column: lens}
    return batch


def prepare_dataset(
    dataset: DatasetDict,
    dataset_name: str,
    length_column_name: str,
    text_column_name: str,
    audio_column_name: str,
    preprocessing_num_workers: int,
    writer_batch_size: int,
    train_split: str,
    text_transformations: Optional[List[str]],
    split_long_segments_to_chunks: bool,
    sampling_rate: int,
    do_resample: bool,
    max_input_len: float,
    min_input_len: float,
    reshuffle_at_start: bool,
) -> DatasetDict:
    """Preprocesses dataset."""
    if reshuffle_at_start:
        with DistributedContext() as context:
            context.wait_before()
            dataset = dataset.shuffle(seed=42)
            context.wait_after()

    # 0. Resample audio to target sampling rate
    if audio_column_name is not None and do_resample:
        dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=sampling_rate))

    if audio_column_name is not None and split_long_segments_to_chunks:
        if length_column_name is not None and length_column_name not in set().union(*dataset.column_names.values()):
            dataset = distributed_process(
                dataset,
                process_by="map",
                function=extract_lens_batched,
                num_proc=preprocessing_num_workers,
                input_columns=[audio_column_name],
                batched=True,
                batch_size=writer_batch_size // 4,
                writer_batch_size=writer_batch_size,
                fn_kwargs={"sampling_rate": sampling_rate, "len_column": length_column_name},
                desc="Extracting audio lens",
            )
        dataset = distributed_process(
            dataset,
            process_by="map",
            function=split_long_segments_to_chunks_fun,
            num_proc=preprocessing_num_workers,
            input_columns=[audio_column_name, length_column_name],
            batched=True,
            batch_size=writer_batch_size // 4,
            remove_columns=dataset.column_names[train_split],
            writer_batch_size=writer_batch_size,
            fn_kwargs={
                "audio_column": audio_column_name,
                "length_column_name": length_column_name,
                "max_input_len": max_input_len,
                "sampling_rate": sampling_rate,
            },
            desc=f"Splitting segments to chunks of size {max_input_len}s",
        )

    # 1. Preprocess audio columns
    if (
        length_column_name is not None
        and length_column_name not in set().union(*dataset.column_names.values())
        or "kaldi_dataset" in dataset_name
    ):
        dataset = distributed_process(
            dataset,
            process_by="map",
            function=extract_lens_batched,
            num_proc=preprocessing_num_workers,
            input_columns=[audio_column_name],
            batched=True,
            batch_size=writer_batch_size // 4,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"sampling_rate": sampling_rate, "len_column": length_column_name},
            desc="Extracting audio lens",
        )

    if length_column_name is not None and train_split is not None:
        dataset[train_split] = distributed_process(
            dataset[train_split],
            process_by="filter",
            function=filter_sequences_in_range_batched,
            batched=True,
            input_columns=[length_column_name],
            num_proc=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"max_input_len": max_input_len, "min_input_len": min_input_len},
            desc="Filtering out too long and too short sequences",
        )

    # Filter samples shorter than 0.1s - {MIN_INPUT_LEN},
    # due to the conv subsampling and mel fbank extraction in model encoder
    # and longer than 100s - {MAX_INPUT_LEN} to avoid memory issues
    for split in list(dataset.keys()):
        if split != train_split:
            dataset[split] = distributed_process(
                dataset[split],
                process_by="filter",
                function=filter_sequences_in_range_batched,
                batched=True,
                input_columns=[length_column_name],
                num_proc=preprocessing_num_workers,
                writer_batch_size=writer_batch_size,
                fn_kwargs={"max_input_len": MAX_INPUT_LEN, "min_input_len": MIN_INPUT_LEN},
                desc="Filter samples that the model is not able to process due to the conv subsampling.",
            )

    # 2. Preprocess label columns
    if text_column_name is not None and text_transformations is not None:
        for transformation_name in text_transformations:
            if transformation_name.startswith("filter_"):
                process_by = "filter"
                fn_kwargs = {}
            else:
                process_by = "map"
                fn_kwargs = {"label_column": text_column_name}
            if transformation_name.endswith("_train"):
                if train_split is not None:
                    transformation = globals()[re.sub("_train", "", transformation_name)]
                    dataset[train_split] = distributed_process(
                        dataset[train_split],
                        process_by=process_by,
                        function=transformation,
                        input_columns=[text_column_name],
                        num_proc=preprocessing_num_workers,
                        writer_batch_size=writer_batch_size,
                        fn_kwargs=fn_kwargs,
                        desc=f"Applying {transformation_name} transformation",
                    )
            else:
                transformation = globals()[transformation_name]
                dataset = distributed_process(
                    dataset,
                    process_by=process_by,
                    function=transformation,
                    input_columns=[text_column_name],
                    num_proc=preprocessing_num_workers,
                    writer_batch_size=writer_batch_size,
                    fn_kwargs=fn_kwargs,
                    desc=f"Applying {transformation_name} transformation",
                )

    logger.info("Casting audio column to Audio, and length column to float32")
    feature_types = dataset[list(dataset.keys())[0]].features
    if audio_column_name is not None:
        feature_types[audio_column_name] = Audio(sampling_rate=sampling_rate)
    if length_column_name is not None:
        feature_types[length_column_name] = Value(dtype="float32")
    for split in dataset:
        dataset[split] = distributed_process(
            dataset[split],
            process_by="cast",
            writer_batch_size=writer_batch_size,
            num_proc=preprocessing_num_workers,
            features=feature_types,
        )

    logger.info(str(dataset))
    return dataset


def merge_splits(dataset: DatasetDict, splits_to_merge: List[str], new_name: str) -> DatasetDict:
    """Merge splits of the provided dataset."""
    if len(splits_to_merge) > 1:
        dataset[new_name] = concatenate_datasets([dataset[split] for split in splits_to_merge])
        for split in splits_to_merge:
            if split != new_name:
                del dataset[split]
    if len(splits_to_merge) == 1 and splits_to_merge[0] != new_name:
        dataset[new_name] = dataset[splits_to_merge[0]]
        del dataset[splits_to_merge[0]]
    return dataset


def join_datasets(
    dataset1: DatasetDict,
    dataset2: DatasetDict,
    test_splits: List[str],
    local_dataset_prefix: str,
    train_split: str,
    validation_splits: Union[str, List[str]],
) -> DatasetDict:
    """Add local datasets to the global dataset."""
    if train_split is not None:
        if train_split in dataset1:
            dataset1[train_split] = concatenate_datasets([dataset1[train_split], dataset2[train_split]])
        else:
            dataset1[train_split] = dataset2[train_split]
    if validation_splits is not None:
        if isinstance(validation_splits, str):
            validation_split = validation_splits
            if validation_split in dataset1:
                dataset1[validation_split] = concatenate_datasets(
                    [dataset1[validation_split], dataset2[validation_split]]
                )
            else:
                dataset1[validation_split] = dataset2[validation_split]
        else:
            for split in validation_splits:
                dataset1["validation_" + local_dataset_prefix.split("/")[-1] + "_" + split] = dataset2[split]
    for split in test_splits:
        dataset1[local_dataset_prefix.split("/")[-1] + "_" + split] = dataset2[split]
    return dataset1


def load_multiple_datasets(
    config_path: str,
    num_proc: int,
    writer_batch_size: int,
    sampling_rate: int,
    max_input_len: float,
    min_input_len: float,
    global_len_column: str,
    global_text_column: str,
    global_audio_column: str,
    global_train_split: str,
    global_validation_split: str,
    split_long_segments_to_chunks: bool,
    load_pure_dataset_only: bool = False,
    merge_validation_splits: bool = True,
) -> DatasetDict:
    """Loads multiple datasets, preprocess them and join to single dataset instance."""
    with open(config_path) as config_handle:
        config_dict = json.load(config_handle)
    dataset_merged = DatasetDict()
    for dataset_config in config_dict:
        logger.info(f"Loading dataset {dataset_config['dataset_name']} {dataset_config['dataset_id']}")
        with DistributedContext() as context:
            context.wait_before()
            if dataset_config["load_from_disk"]:
                dataset = load_from_disk(
                    dataset_config["dataset_name"],
                    keep_in_memory=False,
                    **dataset_config["additional_args"],
                )

            else:
                dataset = load_dataset(
                    dataset_config["dataset_name"],
                    keep_in_memory=False,
                    writer_batch_size=writer_batch_size,
                    num_proc=num_proc,
                    **dataset_config["additional_args"],
                )
            context.wait_after()
        new_train_split_name = global_train_split if len(dataset_config["train_splits"]) > 0 else None
        new_dev_split_name = global_validation_split if len(dataset_config["validation_splits"]) > 0 else None
        dataset = merge_splits(dataset, dataset_config["train_splits"], new_train_split_name)
        if merge_validation_splits:
            dataset = merge_splits(dataset, dataset_config["validation_splits"], new_dev_split_name)

        # Remove unused splits
        for split in list(dataset.keys()):
            if not split.startswith("validation") and split not in dataset_config["test_splits"] + [
                new_train_split_name,
                new_dev_split_name,
            ]:
                del dataset[split]

        logger.info(f"Preprocessing dataset {dataset_config['dataset_name']}")
        if not load_pure_dataset_only:
            dataset_processed = prepare_dataset(
                dataset=dataset,
                dataset_name=dataset_config["dataset_name"],
                length_column_name=dataset_config.get("length_column_name"),
                text_column_name=dataset_config.get("text_column_name"),
                audio_column_name=dataset_config.get("audio_column_name"),
                preprocessing_num_workers=num_proc,
                writer_batch_size=writer_batch_size,
                train_split=new_train_split_name,
                text_transformations=dataset_config.get("text_transformations"),
                sampling_rate=sampling_rate,
                do_resample=dataset_config.get("do_resample", False),
                max_input_len=max_input_len,
                min_input_len=min_input_len,
                split_long_segments_to_chunks=split_long_segments_to_chunks,
                reshuffle_at_start=dataset_config.get("reshuffle_at_start", False),
            )
        else:
            logger.info(str(dataset))
            dataset_processed = dataset

        for column, global_column in [
            ("length_column_name", global_len_column),
            ("text_column_name", global_text_column),
            ("audio_column_name", global_audio_column),
        ]:
            if dataset_config.get(column) is not None and dataset_config.get(column) != global_column:
                dataset_processed = dataset_processed.rename_column(dataset_config.get(column), global_column)

        dataset_local = dataset_processed.remove_columns(
            list(
                set()
                .union(*dataset_processed.column_names.values())
                .difference([global_len_column, global_text_column, global_audio_column])
            )
        )
        dataset_merged = join_datasets(
            dataset_merged,
            dataset_local,
            dataset_config["test_splits"],
            dataset_config["dataset_id"],
            new_train_split_name,
            new_dev_split_name if merge_validation_splits else dataset_config["validation_splits"],
        )
    return dataset_merged


def get_eval_dataset(
    dataset: DatasetDict,
    train_split_name: str,
    validation_split_name: str,
    data_slice_str: str,
    cut_validation_from_train: bool,
    seed: Optional[int],
) -> Union[Dataset, DatasetDict]:
    num_validation_splits = np.sum([key.startswith("validation") for key in dataset.keys()])
    if num_validation_splits > 1:
        if cut_validation_from_train:
            raise ValueError("Cannot use cut_validation_from_train with multiple validation splits.")
        if data_slice_str is not None:
            raise ValueError("Cannot use data_slice with multiple validation splits.")
        return DatasetDict(
            {key.split("validation_")[1]: dataset[key] for key in dataset.keys() if key.startswith("validation")}
        )

    if cut_validation_from_train:
        if validation_split_name in dataset:
            raise ValueError(
                "Cannot use cut_validation_from_train and validation_split that exist in the dataset at the same time."
            )
        if data_slice_str is not None:
            train_split = dataset[train_split_name]
            data_slice = extract_num_samples(train_split, data_slice_str)
            new_splits = train_split.train_test_split(test_size=data_slice, shuffle=True, seed=seed)
            dataset[train_split_name] = new_splits["train"]
            dataset[validation_split_name + data_slice_str] = new_splits["test"]
            return new_splits["test"]
        else:
            raise ValueError("Cannot use cut_validation_from_train without specifying data_slice.")
    elif train_split_name == validation_split_name:
        raise ValueError("Cannot use the same split for training and validation.")
    else:
        if validation_split_name not in dataset:
            return Dataset.from_dict({})
        validation_split = dataset[validation_split_name]
        if data_slice_str is not None:
            data_slice = extract_num_samples(validation_split, data_slice_str)
            training_eval_dataset = validation_split.shuffle(seed=seed).select(range(data_slice))
            dataset[validation_split_name + data_slice_str] = training_eval_dataset
            return training_eval_dataset
        else:
            return validation_split


def get_dataset(
    data_args: DataTrainingArguments,
    len_column: str,
) -> Tuple[DatasetDict, Dataset]:
    """Loads single or multiple datasets, preprocess, and merge them."""
    if data_args.datasets_creation_config is not None:
        dataset = load_multiple_datasets(
            config_path=data_args.datasets_creation_config,
            num_proc=data_args.preprocessing_num_workers,
            writer_batch_size=data_args.writer_batch_size,
            sampling_rate=data_args.sampling_rate,
            max_input_len=data_args.max_duration_in_seconds,
            min_input_len=data_args.min_duration_in_seconds,
            global_len_column=len_column,
            global_text_column=data_args.text_column_name,
            global_audio_column=data_args.audio_column_name,
            global_train_split=data_args.train_split,
            global_validation_split=data_args.validation_split,
            split_long_segments_to_chunks=data_args.split_long_segments_to_chunks,
            load_pure_dataset_only=data_args.load_pure_dataset_only,
            merge_validation_splits=data_args.merge_validation_splits,
        )
    else:
        with DistributedContext() as context:
            context.wait_before()
            if data_args.dataset_config is not None:
                dataset = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config,
                    keep_in_memory=False,
                    num_proc=data_args.preprocessing_num_workers,
                    writer_batch_size=data_args.writer_batch_size,
                )
            else:
                dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
            context.wait_after()

        # 3. Preprocess dataset
        if not data_args.load_pure_dataset_only:
            dataset = prepare_dataset(
                dataset=dataset,
                dataset_name=data_args.dataset_name,
                length_column_name=len_column,
                text_column_name=data_args.text_column_name,
                audio_column_name=data_args.audio_column_name,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                writer_batch_size=data_args.writer_batch_size,
                train_split=data_args.train_split,
                sampling_rate=data_args.sampling_rate,
                do_resample=data_args.do_resample,
                max_input_len=data_args.max_duration_in_seconds,
                min_input_len=data_args.min_duration_in_seconds,
                text_transformations=data_args.text_transformations,
                split_long_segments_to_chunks=data_args.split_long_segments_to_chunks,
                reshuffle_at_start=data_args.reshuffle_at_start,
            )

    if data_args.dump_prepared_dataset_to is not None:
        logger.info("Dumping prepared datasets to %s", data_args.dump_prepared_dataset_to)

        if data_args.concatenate_splits_before_dumping:
            dataset = DatasetDict(
                {data_args.train_split: concatenate_datasets([dataset[split] for split in dataset.keys()])}
            )

        dataset.save_to_disk(
            dataset_dict_path=data_args.dump_prepared_dataset_to,
            num_proc=data_args.preprocessing_num_workers,
            max_shard_size=data_args.dataset_shard_size,
        )

    train_eval_dataset = get_eval_dataset(
        dataset,
        data_args.train_split,
        data_args.validation_split,
        data_args.validation_slice,
        data_args.cut_validation_from_train,
        data_args.validation_slice_seed,
    )

    return dataset, train_eval_dataset


def is_number(s):
    try:
        complex(s)
    except ValueError:
        return False

    return True


def extract_num_samples(dataset: Dataset, data_slice: str) -> int:
    if data_slice.isnumeric():
        data_slice = int(data_slice)
    elif "%" in data_slice:
        data_slice = data_slice.replace("%", "")
        if is_number(data_slice):
            data_slice = int(float(data_slice) * len(dataset) / 100)
        else:
            raise ValueError(f"Invalid slice value: {data_slice}, must be number or percentage")
    else:
        raise ValueError(f"Invalid slice value: {data_slice}, must be number or percentage")
    return data_slice
