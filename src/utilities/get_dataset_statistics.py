from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from transformers import HfArgumentParser


@dataclass
class DataTrainingArguments:
    _argument_group_name = "Data related arguments"
    """Dataset source related arguments."""
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    length_column_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the column containing the length of the audio files."}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((DataTrainingArguments))
    (data_args,) = parser.parse_args_into_dataclasses()

    df = datasets.load_from_disk(data_args.dataset_path)
    print(df)
    if isinstance(df, datasets.DatasetDict):
        for split in df:
            print(f"Split: {split}")
            lengths = np.array(df[split][data_args.length_column_name])
            # print statistics as mean, std, min, max
            print(f"Overall length: {lengths.sum()}")
            print(f"Mean: {lengths.mean()}")
            print(f"Std: {lengths.std()}")
            print(f"Min: {lengths.min()}")
            print(f"Max: {lengths.max()}")
    else:
        lengths = np.array(df[data_args.length_column_name])
        # print statistics as mean, std, min, max
        print(f"Overall length: {lengths.sum()}")
        print(f"Mean: {lengths.mean()}")
        print(f"Std: {lengths.std()}")
        print(f"Min: {lengths.min()}")
        print(f"Max: {lengths.max()}")
