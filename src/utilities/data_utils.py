"""Utilities for data loading and preprocessing."""
import json
import re
from typing import Dict, List, Optional, Tuple, Union

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

tokens_escaped_regex = re.compile("|".join([r"\s" + re.escape(token) for token in special_tokens]))

MIN_INPUT_LEN = 0.1


class DistributedContext:
    """Context manager for distributed training."""

    def __init__(self):
        """Initializes distributed context."""
        self.local_rank = None
        self.world_size = None

    def __enter__(self):
        """Initializes distributed context."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Performs barrier synchronization."""
        if self.world_size > 1:
            torch.distributed.barrier()

    def wait_before(self):
        if self.world_size > 1:
            if self.local_rank > 0:
                logger.info(f"Rank {self.local_rank}: Waiting for main process to perform the mapping")
                torch.distributed.barrier()

    def wait_after(self):
        if self.world_size > 1:
            if self.local_rank == 0:
                logger.info(f"Rank {self.local_rank}: Waiting for other processes to finish")
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


def remove_multiple_whitespaces_and_strip(example: str, label_column: str) -> Dict[str, str]:
    """Removes multiple whitespaces from batch."""
    return {label_column: re.sub(r"\s+", " ", example).strip()}


def clean_special_tokens_english(example: str, label_column: str) -> Dict[str, str]:
    """Cleans special tokens from labels."""
    return {label_column: tokens_escaped_regex.sub("", example)}


def transforms_unfinished_words_to_unks(example: str, label_column: str) -> Dict[str, str]:
    """Transforms unfinished words to UNKs."""
    return {label_column: re.sub(r"\(?\w+-\)?", "([unk])", example)}


def filter_empty_transcriptions(example: str) -> bool:
    """Filters out empty transcriptions."""
    return example != ""


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
    text_transformations: List[str],
    split_long_segments_to_chunks: bool,
    filter_empty_labels: bool,
    sampling_rate: int,
    max_input_len: float,
    min_input_len: float,
) -> DatasetDict:
    """Preprocesses dataset."""
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

    # 2. Preprocess label columns
    if text_column_name is not None:
        for transformation_name in text_transformations:
            if transformation_name.endswith("_train"):
                if train_split is not None:
                    transformation = globals()[re.sub("_train", "", transformation_name)]
                    dataset[train_split] = distributed_process(
                        dataset[train_split],
                        process_by="map",
                        function=transformation,
                        input_columns=[text_column_name],
                        num_proc=preprocessing_num_workers,
                        writer_batch_size=writer_batch_size,
                        fn_kwargs={"label_column": text_column_name},
                        desc=f"Applying {transformation_name} transformation",
                    )
            else:
                transformation = globals()[transformation_name]
                dataset = distributed_process(
                    dataset,
                    process_by="map",
                    function=transformation,
                    input_columns=[text_column_name],
                    num_proc=preprocessing_num_workers,
                    writer_batch_size=writer_batch_size,
                    fn_kwargs={"label_column": text_column_name},
                    desc=f"Applying {transformation_name} transformation",
                )

        # 3. Remove segments with empty annotations
        if filter_empty_labels:
            dataset = distributed_process(
                dataset,
                process_by="filter",
                function=filter_empty_transcriptions,
                input_columns=[text_column_name],
                writer_batch_size=writer_batch_size,
                num_proc=preprocessing_num_workers,
                desc="Filtering out empty transcriptions",
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
    validation_split: str,
) -> DatasetDict:
    """Add local datasets to the global dataset."""
    if train_split is not None:
        if train_split in dataset1:
            dataset1[train_split] = concatenate_datasets([dataset1[train_split], dataset2[train_split]])
        else:
            dataset1[train_split] = dataset2[train_split]
    if validation_split is not None:
        if validation_split in dataset1:
            dataset1[validation_split] = concatenate_datasets([dataset1[validation_split], dataset2[validation_split]])
        else:
            dataset1[validation_split] = dataset2[validation_split]
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
    filter_empty_labels: bool,
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
        dataset = merge_splits(dataset, dataset_config["validation_splits"], new_dev_split_name)

        # Remove unused splits
        for split in list(dataset.keys()):
            if split not in dataset_config["test_splits"] + [new_train_split_name, new_dev_split_name]:
                del dataset[split]

        logger.info(f"Preprocessing dataset {dataset_config['dataset_name']}")
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
            max_input_len=max_input_len,
            min_input_len=min_input_len,
            split_long_segments_to_chunks=split_long_segments_to_chunks,
            filter_empty_labels=filter_empty_labels,
        )

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
            new_dev_split_name,
        )
    return dataset_merged


def get_eval_split(
    dataset: DatasetDict,
    train_split_name: str,
    validation_split_name: str,
    data_slice_str: str,
    cut_validation_from_train: bool,
    seed: Optional[int],
) -> Dataset:
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
        validation_split = dataset[validation_split_name]
        if data_slice_str is not None:
            data_slice = extract_num_samples(validation_split, data_slice_str)
            training_eval_dataset = validation_split.shuffle(seed=seed).select(range(data_slice))
            dataset[validation_split_name + data_slice_str] = training_eval_dataset
        return validation_split


def get_dataset(
    datasets_creation_config_path: str,
    dataset_name: str,
    dataset_config: str,
    preprocessing_num_workers: int,
    writer_batch_size: int,
    sampling_rate: int,
    max_input_len: float,
    min_input_len: float,
    len_column: str,
    text_column: str,
    audio_column: str,
    train_split: str,
    validation_split: str,
    text_transformations: List[str],
    split_long_segments_to_chunks: bool,
    filter_empty_labels: bool,
    validation_slice_str: str,
    cut_validation_from_train: bool,
    seed: Optional[int],
) -> Tuple[DatasetDict, Dataset]:
    """Loads single or multiple datasets, preprocess, and merge them."""
    if datasets_creation_config_path is not None:
        dataset = load_multiple_datasets(
            config_path=datasets_creation_config_path,
            num_proc=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            sampling_rate=sampling_rate,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
            global_len_column=len_column,
            global_text_column=text_column,
            global_audio_column=audio_column,
            global_train_split=train_split,
            global_validation_split=validation_split,
            split_long_segments_to_chunks=split_long_segments_to_chunks,
            filter_empty_labels=filter_empty_labels,
        )
    else:
        with DistributedContext() as context:
            context.wait_before()
            if dataset_config is not None:
                dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    keep_in_memory=False,
                    num_proc=preprocessing_num_workers,
                    writer_batch_size=writer_batch_size,
                )
            else:
                dataset = load_from_disk(dataset_name, keep_in_memory=False)
            context.wait_after()

        # 3. Preprocess dataset
        dataset = prepare_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            length_column_name=len_column,
            text_column_name=text_column,
            audio_column_name=audio_column,
            preprocessing_num_workers=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            train_split=train_split,
            sampling_rate=sampling_rate,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
            text_transformations=text_transformations,
            split_long_segments_to_chunks=split_long_segments_to_chunks,
            filter_empty_labels=filter_empty_labels,
        )

    # Filter samples shorter than 0.1s - {MIN_INPUT_LEN},
    # due to the conv subsampling and mel fbank extraction in model encoder
    dataset_splits = list(dataset.keys())
    dataset_splits.remove(train_split)
    for split in dataset_splits:
        dataset[split] = distributed_process(
            dataset[split],
            process_by="filter",
            function=filter_sequences_in_range_batched,
            batched=True,
            input_columns=[len_column],
            num_proc=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"max_input_len": np.finfo(np.float32).max, "min_input_len": MIN_INPUT_LEN},
            desc="Filter samples that the model is not able to process due to the conv subsampling.",
        )
    train_eval_split = get_eval_split(
        dataset, train_split, validation_split, validation_slice_str, cut_validation_from_train, seed
    )
    return dataset, train_eval_split


def extract_num_samples(dataset: Dataset, data_slice: str) -> int:
    if data_slice.isnumeric():
        data_slice = int(data_slice)
    else:
        data_slice = data_slice.replace("%", "")
        if data_slice.isnumeric():
            data_slice = int(float(data_slice) * len(dataset) / 100)
        else:
            raise ValueError(f"Invalid slice value: {data_slice}, must be number or percentage")
    return data_slice
